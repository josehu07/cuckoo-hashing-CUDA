#ifndef _CUCKOO_CUDA_MULTI_HPP_
#define _CUCKOO_CUDA_MULTI_HPP_


#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "common.h"


/**
 *
 * Cuckoo hash table generic class.
 * 
 */
template <typename T>
class CuckooHashTableCuda_Multi {

private:

    /** Input parameters. */
    const int _size;
    const int _evict_bound;
    const int _num_funcs;
    const int _pos_width;
    const int _num_buckets;

    /** Actual data table. */
    T *_data;

    /** Cuckoo hash function set. */
    FuncConfig *_hash_func_configs;

    /** Private operations. */
    void gen_hash_funcs() {

        // Calculate bit width of value range and table size.
        int val_width = 8 * sizeof(T) - ceil(log2((double) _num_funcs));
        int bucket_width = ceil(log2((double) _num_buckets));
        int size_width = ceil(log2((double) BUCKET_SIZE));

        // Generate randomized configurations.
        for (int i = 0; i < _num_funcs; ++i) {  // At index 0 is a dummy function.
            if (val_width - bucket_width <= size_width)
                _hash_func_configs[i] = {rand(), 0};
            else
                _hash_func_configs[i] = {rand(), rand() % (val_width - bucket_width - size_width + 1)
                                                 + bucket_width};
        }
    };

    /** Inline helper functions. */
    inline T fetch_val(const T data) {
        return data >> _pos_width;
    }
    inline int fetch_func(const T data) {
        return data & ((0x1 << _pos_width) - 1);
    }

public:

    /** Constructor & Destructor. */
    CuckooHashTableCuda_Multi(const int size, const int evict_bound, const int num_funcs)
        : _size(size), _evict_bound(evict_bound), _num_funcs(num_funcs),
          _pos_width(ceil(log2((double) _num_funcs))),
          _num_buckets(ceil((double) _size / BUCKET_SIZE)) {

        // Allocate space for data table, map, and hash function configs.
        _data = new T[num_funcs * size]();   // Use "all-zero" mode, indicating initially all empty.
        _hash_func_configs = new FuncConfig[num_funcs];

        // Generate initial hash function configs.
        gen_hash_funcs();
    };
    ~CuckooHashTableCuda_Multi() {
        delete[] _data;
        delete[] _hash_func_configs;
    };

    /** Supported operations. */
    int insert_vals(const T * const vals, const int n);
    void delete_vals(const T * const vals, const int n);
    void lookup_vals(const T * const vals, bool * const results, const int n);
    void show_content();
};


/**
 * 
 * Cuckoo: insert operation (bucket kernel + insert kernel + host function).
 *
 * Attention: Can only be invoked when table is empty!
 * Attention: Assumes that input is uniform, and all buckets will not overflow!
 *
 * Returns:
 *   Number of rehashings beneath.
 *   
 */
template <typename T>
__global__ void
cuckooBucketKernel_Multi(T * const data_buf, const int size,
                         const T * const vals, const int n,
                         int * const counters, const int num_buckets) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {

        // Do 1st-level hashing to get bucket id, then do atomic add to get index inside the bucket.
        T val = vals[idx];
        int bucket_num = do_1st_hash(val, num_buckets);
        int bucket_ofs = atomicAdd(&counters[bucket_num], 1);

        // Directly write the key into the table buffer.
        if (bucket_ofs >= BUCKET_SIZE)
            printf("ERROR: bucket overflow!\n");
        else
            data_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = val;
    }
}

template <typename T>
__global__ void
cuckooInsertKernel_Multi(T * const data, const T * const data_buf, const int size,
                         const FuncConfig * const hash_func_configs, const int num_funcs,
                         const int * const counters, const int num_buckets,
                         const int evict_bound, const int pos_width,
                         int * const rehash_requests) {
    
    // Create local cuckoo table in shared memory. Size passed in as the third kernel parameter.
    extern __shared__ T local_data[];
    for (int i = 0; i < num_funcs; ++i)
        local_data[i * BUCKET_SIZE + threadIdx.x] = EMPTY_CELL;

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within local bucket range are active.
    if (threadIdx.x < counters[blockIdx.x]) {

        // Set initial conditions.
        T cur_val = data_buf[idx];
        int cur_func = 0;
        int evict_count = 0;

        // Start the test-kick-and-reinsert loops.
        do {
            int pos = do_2nd_hash(cur_val, hash_func_configs, cur_func, BUCKET_SIZE);
            T old_data = atomicExch(&local_data[cur_func * BUCKET_SIZE + pos],
                                    make_data(cur_val, cur_func, pos_width));
            if (old_data != EMPTY_CELL) {
                cur_val = fetch_val(old_data, pos_width);
                cur_func = (fetch_func(old_data, pos_width) + 1) % num_funcs;
                evict_count++;
            } else
                break;
        } while (evict_count < num_funcs * evict_bound);

        // If exceeds eviction bound, then needs rehashing.
        if (evict_count >= num_funcs * evict_bound)
            atomicAdd(rehash_requests, 1);
    }

    // Every thread write its responsible local slot into the global data table.
    __syncthreads();
    for (int i = 0; i < num_funcs; ++i)
        data[i * size + idx] = local_data[i * BUCKET_SIZE + threadIdx.x];
}

template <typename T>
int
CuckooHashTableCuda_Multi<T>::insert_vals(const T * const vals, const int n) {
    
    //
    // Phase 1: Distribute keys into buckets.
    //

    // Allocate GPU memory.
    T *d_vals;
    T *d_data_buf;
    int *d_counters;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data_buf, _size * sizeof(T));
    cudaMalloc((void **) &d_counters, _num_buckets * sizeof(int));

    // Copy values into GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_data_buf, EMPTY_CELL, _size * sizeof(T));
    cudaMemset(d_counters, 0, _num_buckets * sizeof(int));

    // Invoke bucket kernel.
    cuckooBucketKernel_Multi<<<ceil((double) n / BUCKET_SIZE), BUCKET_SIZE>>>(d_data_buf, _size,
                                                                              d_vals, n,
                                                                              d_counters, _num_buckets);

    //
    // Phase 2: Local cuckoo hashing.
    //

    // Allocate GPU memory.
    T *d_data;
    FuncConfig *d_hash_func_configs;
    int *d_rehash_requests;
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));
    cudaMalloc((void **) &d_rehash_requests, sizeof(int));

    // Copy values onto GPU memory.
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);

    // Invoke insert kernel. Passes shared memory table size by the third argument.
    // Loops until no rehashing needed.
    int rehash_count = 0;
    do {
        int rehash_requests = 0;
        cudaMemset(d_rehash_requests, 0, sizeof(int));
        cuckooInsertKernel_Multi<<<ceil((double) _size / BUCKET_SIZE), BUCKET_SIZE, \
                                   _num_funcs * BUCKET_SIZE * sizeof(T)>>>(d_data, d_data_buf, _size,
                                                                           d_hash_func_configs, _num_funcs,
                                                                           d_counters, _num_buckets,
                                                                           _evict_bound, _pos_width,
                                                                           d_rehash_requests);
        cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int), cudaMemcpyDeviceToHost);
        if (rehash_requests == 0)
            break;
        else {
            rehash_count++;
            gen_hash_funcs();
            cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
                       cudaMemcpyHostToDevice);
        }
    } while (rehash_count < MAX_DEPTH);

    // Retrive results into main memory data table.
    cudaMemcpy(_data, d_data, (_num_funcs * _size) * sizeof(T), cudaMemcpyDeviceToHost);

    // Free GPU resources.
    cudaFree(d_vals);
    cudaFree(d_data);
    cudaFree(d_data_buf);
    cudaFree(d_counters);
    cudaFree(d_hash_func_configs);
    cudaFree(d_rehash_requests);

    return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
}


/**
 * 
 * Cuckoo: delete operation (kernel + host function).
 *   
 */
template <typename T>
__global__ void
cuckooDeleteKernel_Multi(const T * const vals, const int n,
                         T * const data, const int size,
                         const FuncConfig * const hash_func_configs, const int num_funcs,
                         const int num_buckets, const int pos_width) {
    
    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        int bucket_num = val % num_buckets;

        for (int i = 0; i < num_funcs; ++i) {
            int pos = bucket_num * BUCKET_SIZE + do_2nd_hash(val, hash_func_configs, i, BUCKET_SIZE);
            if (fetch_val(data[i * size + pos], pos_width) == val) {
                data[i * size + pos] = EMPTY_CELL;
                return;
            }
        }
    }
}

template <typename T>
void
CuckooHashTableCuda_Multi<T>::delete_vals(const T * const vals, const int n) {
    
    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    FuncConfig *d_hash_func_configs;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);

    // Launch the delete kernel.
    cuckooDeleteKernel_Multi<<<ceil((double) n / BUCKET_SIZE), BUCKET_SIZE>>>(d_vals, n,
                                                                              d_data, _size,
                                                                              d_hash_func_configs, _num_funcs,
                                                                              _num_buckets, _pos_width);

    // Retrieve results.
    cudaMemcpy(_data, d_data, (_num_funcs * _size) * sizeof(T), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_data);
    cudaFree(d_hash_func_configs);
}


/**
 * 
 * Cuckoo: lookup operation (kernel + host function).
 *   
 */
template <typename T>
__global__ void
cuckooLookupKernel_Multi(const T * const vals, bool * const results, const int n,
                         const T * const data, const int size,
                         const FuncConfig * const hash_func_configs, const int num_funcs,
                         const int num_buckets, const int pos_width) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        int bucket_num = val % num_buckets;

        for (int i = 0; i < num_funcs; ++i) {
            int pos = bucket_num * BUCKET_SIZE + do_2nd_hash(val, hash_func_configs, i, BUCKET_SIZE);
            if (fetch_val(data[i * size + pos], pos_width) == val) {
                results[idx] = true;
                return;
            }
        }

        results[idx] = false;
    }
}

template <typename T>
void
CuckooHashTableCuda_Multi<T>::lookup_vals(const T * const vals, bool * const results, const int n) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    bool *d_results;
    FuncConfig *d_hash_func_configs;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_results, n * sizeof(bool));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);

    // Launch the lookup kernel.
    cuckooLookupKernel_Multi<<<ceil((double) n / BUCKET_SIZE), BUCKET_SIZE>>>(d_vals, d_results, n,
                                                                              d_data, _size,
                                                                              d_hash_func_configs, _num_funcs,
                                                                              _num_buckets, _pos_width);

    // Retrieve results.
    cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_results);
    cudaFree(d_data);
    cudaFree(d_hash_func_configs);
}


/** Cuckoo: print content out. */
template <typename T>
void
CuckooHashTableCuda_Multi<T>::show_content() {
    std::cout << "Buckets: " << _num_buckets << std::endl;
    std::cout << "Funcs: ";
    for (int i = 0; i < _num_funcs; ++i) {
        FuncConfig fc = _hash_func_configs[i];
        std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
    }
    std::cout << std::endl;
    for (int i = 0; i < _num_funcs; ++i) {
        std::cout << "Table " << i << ": ";
        for (int j = 0; j < _size; ++j)
            std::cout << std::setw(10) << fetch_val(_data[i * _size + j]) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


#endif
