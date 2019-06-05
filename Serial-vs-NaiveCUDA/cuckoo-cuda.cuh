#ifndef _CUCKOO_CUDA_HPP_
#define _CUCKOO_CUDA_HPP_


#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>


/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)


/** Max rehashing depth, and error depth. */
#define MAX_DEPTH (100)
#define ERR_DEPTH (-1)


/** CUDA thread block size. */
#define BLOCK_SIZE (256)


/** Struct of a hash function config. */
typedef struct {
    int rv;     // Randomized XOR value.
    int ss;     // Randomized shift filter start position.
} FuncConfig;


/** Hard code XOR hash functions and all inline helper functions for CUDA kernels' use. */
template <typename T>
static inline __device__ int
do_hash(const T val, const FuncConfig * const hash_func_configs, const int func_idx,
        const int size) {
    FuncConfig fc = hash_func_configs[func_idx];
    return ((val ^ fc.rv) >> fc.ss) % size;
}

template <typename T>
static inline __device__ T
fetch_val(const T data, const int pos_width) {
    return data >> pos_width;
}

template <typename T>
static inline __device__ int
fetch_func(const T data, const int pos_width) {
    return data & ((0x1 << pos_width) - 1);
}

template <typename T>
static inline __device__ T
make_data(const T val, const int func, const int pos_width) {
    return (val << pos_width) ^ func;   // CANNOT handle signed values currently!
}

/**
 *
 * Cuckoo hash table generic class.
 *
 * Hash function choice:
 *   Use a random integer number, do bit-wise XOR, then modulo table size.
 *   Since XOR gives a uniform randomization, it should be a good choice.
 *
 * To simplify CUDA codes, hash functions are hard coded into the kernel.
 * 
 */
template <typename T>
class CuckooHashTableCuda_Naive {

private:

    /** Input parameters. */
    const int _size;
    const int _evict_bound;
    const int _num_funcs;
    const int _pos_width;

    /** Actual data. */
    T *_data;
    FuncConfig *_hash_func_configs;

    /** Private operations. */
    void gen_hash_funcs() {

        // Calculate bit width of value range and table size.
        int val_width = 8 * sizeof(T) - ceil(log2((double) _num_funcs));
        int size_width = ceil(log2((double) _size));

        // Generate randomized configurations.
        for (int i = 0; i < _num_funcs; ++i) {  // At index 0 is a dummy function.
            if (val_width <= size_width)
                _hash_func_configs[i] = {rand(), 0};
            else
                _hash_func_configs[i] = {rand(), rand() % (val_width - size_width + 1)};
        }
    };
    int rehash(const T * const vals, const int n, const int depth);

public:

    /** Constructor & Destructor. */
    CuckooHashTableCuda_Naive(const int size, const int evict_bound, const int num_funcs)
        : _size(size), _evict_bound(evict_bound), _num_funcs(num_funcs),
          _pos_width(ceil(log2((double) _num_funcs))) {

        // Allocate space for data table, map, and hash function configs.
        _data = new T[num_funcs * size]();   // Use "all-zero" mode, indicating initially all empty.
        _hash_func_configs = new FuncConfig[num_funcs];
        
        // Generate initial hash function configs.
        gen_hash_funcs();
    };
    ~CuckooHashTableCuda_Naive() {
        delete[] _data;
        delete[] _hash_func_configs;
    };

    /** Supported operations. */
    int insert_vals(const T * const vals, const int n, const int depth);
    void delete_vals(const T * const vals, const int n);
    void lookup_vals(const T * const vals, bool * const results, const int n);
    void show_content();
};


/**
 * 
 * Cuckoo: insert operation (kernel + host function).
 *
 * Returns:
 *   Number of rehashings beneath.
 *   
 */
template <typename T>
__global__ void
cuckooInsertKernel(const T * const vals, const int n,
                   T * const data, const int size,
                   const FuncConfig * const hash_func_configs, const int num_funcs,
                   const int evict_bound, const int pos_width,
                   int * const rehash_requests) {
    
    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {

        // Set initial conditions.
        T cur_val = vals[idx];
        int cur_func = 0;
        int evict_count = 0;

        // Start the test-kick-and-reinsert loops.
        do {
            int pos = do_hash(cur_val, hash_func_configs, cur_func, size);
            T old_data = atomicExch(&data[cur_func * size + pos], make_data(cur_val, cur_func, pos_width));
            if (old_data != EMPTY_CELL) {
                cur_val = fetch_val(old_data, pos_width);
                cur_func = (fetch_func(old_data, pos_width) + 1) % num_funcs;
                evict_count++;
            } else
                return;
        } while (evict_count < num_funcs * evict_bound);

        // Exceeds eviction bound, needs rehashing.
        atomicAdd(rehash_requests, 1);
    }
}

template <typename T>
int
CuckooHashTableCuda_Naive<T>::insert_vals(const T * const vals, const int n, const int depth) {
    
    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    FuncConfig *d_hash_func_configs;
    int rehash_requests = 0;
    int *d_rehash_requests;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));
    cudaMalloc((void **) &d_rehash_requests, sizeof(int));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_rehash_requests, &rehash_requests, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the insert kernel.
    cuckooInsertKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, n,
                                                                      d_data, _size,
                                                                      d_hash_func_configs, _num_funcs,
                                                                      _evict_bound, _pos_width,
                                                                      d_rehash_requests);

    // If need rehashing, do rehash with original data + VALS. Else, retrieve results into data.
    cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int), cudaMemcpyDeviceToHost);
    if (rehash_requests > 0) {
        cudaFree(d_vals);
        cudaFree(d_data);
        cudaFree(d_hash_func_configs);
        cudaFree(d_rehash_requests);
        int levels_beneath = rehash(vals, n, depth + 1);
        if (levels_beneath == ERR_DEPTH)
            return ERR_DEPTH;
        else
            return levels_beneath + 1;
    } else {
        cudaMemcpy(_data, d_data, (_num_funcs * _size) * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_vals);
        cudaFree(d_data);
        cudaFree(d_hash_func_configs);
        cudaFree(d_rehash_requests);
        return 0;
    }
}


/**
 * 
 * Cuckoo: delete operation (kernel + host function).
 *   
 */
template <typename T>
__global__ void
cuckooDeleteKernel(const T * const vals, const int n,
                   T * const data, const int size,
                   const FuncConfig * const hash_func_configs, const int num_funcs,
                   const int pos_width) {
    
    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        for (int i = 0; i < num_funcs; ++i) {
            int pos = do_hash(val, hash_func_configs, i, size);
            if (fetch_val(data[i * size + pos], pos_width) == val) {
                data[i * size + pos] = EMPTY_CELL;
                return;
            }
        }
    }
}

template <typename T>
void
CuckooHashTableCuda_Naive<T>::delete_vals(const T * const vals, const int n) {
    
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
    cuckooDeleteKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, n,
                                                                      d_data, _size,
                                                                      d_hash_func_configs, _num_funcs,
                                                                      _pos_width);

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
cuckooLookupKernel(const T * const vals, bool * const results, const int n,
                   const T * const data, const int size,
                   const FuncConfig * const hash_func_configs, const int num_funcs,
                   const int pos_width) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        for (int i = 0; i < num_funcs; ++i) {
            int pos = do_hash(val, hash_func_configs, i, size);
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
CuckooHashTableCuda_Naive<T>::lookup_vals(const T * const vals, bool * const results, const int n) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    bool *d_results;
    FuncConfig *d_hash_func_configs;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_results, n * sizeof(bool));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);

    // Launch the lookup kernel.
    cuckooLookupKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, d_results, n,
                                                                      d_data, _size,
                                                                      d_hash_func_configs, _num_funcs,
                                                                      _pos_width);

    // Retrieve results.
    cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_results);
    cudaFree(d_data);
    cudaFree(d_hash_func_configs);
}


/**
 *
 * Cuckoo: generate new set of hash functions and rehash.
 *
 * Returns:
 *   Number of rehashings beneath.
 * 
 */
template <typename T>
int
CuckooHashTableCuda_Naive<T>::rehash(const T * const vals, const int n, const int depth) {

    // If exceeds max rehashing depth, abort.
    if (depth > MAX_DEPTH)
        return ERR_DEPTH;

    // Generate new set of hash functions.
    gen_hash_funcs();

    // Clear data and map, put values into a buffer.
    std::vector<T> val_buffer;
    for (int i = 0; i < _num_funcs; ++i) {
        for (int j = 0; j < _size; ++j) {
            if (fetch_val(_data[i * _size + j]) != EMPTY_CELL)
                val_buffer.push_back(fetch_val(_data[i * _size + j]));
            _data[i * _size + j] = EMPTY_CELL;
        }
    }
    for (int i = 0; i < n; ++i)
        val_buffer.push_back(vals[i]);

    // Re-insert all values.
    int levels_beneath = insert_vals(val_buffer.data(), val_buffer.size(), depth);
    if (levels_beneath == ERR_DEPTH)
        return ERR_DEPTH;
    else
        return levels_beneath;
}


/** Cuckoo: print content out. */
template <typename T>
void
CuckooHashTableCuda_Naive<T>::show_content() {
    std::cout << "Funcs: ";
    for (int i = 0; i < _num_funcs; ++i) {
        FuncConfig fc = _hash_func_configs[i];
        std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
    }
    std::cout << std::endl;
    for (int i = 0; i < _num_funcs; ++i) {
        std::cout << "Table " << i << ": ";
        for (int j = 0; j < _size; ++j)
            std::cout << std::setw(10) << fetch_val(_data[i * _size + j], _pos_width) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


#endif
