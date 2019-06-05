#ifndef _CUCKOO_SERIAL_HPP_
#define _CUCKOO_SERIAL_HPP_


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


/**
 *
 * Cuckoo hash table generic class.
 * 
 */
template <typename T>
class CuckooHashTable {

private:

    /** Struct of a hash function config. */
    typedef struct {
        int rv;     // Randomized XOR value.
        int ss;     // Randomized shift filter start position.
    } FuncConfig;

    /** Input parameters. */
    const int _size;
    const int _evict_bound;
    const int _num_funcs;

    /** Actual data. */
    T *_data;
    int *_pos_to_func_map;
    FuncConfig *_hash_func_configs;

    /** Private operations. */
    void gen_hash_funcs() {

        // Calculate bit width of value range and table size.
        int val_width = 8 * sizeof(T) - ceil(log2((double) _num_funcs));
        int size_width = ceil(log2((double) _size));

        // Generate randomized configurations.
        for (int i = 1; i <= _num_funcs; ++i) {     // At index 0 is a dummy function.
            if (val_width <= size_width)
                _hash_func_configs[i] = {rand(), 0};
            else
                _hash_func_configs[i] = {rand(), rand() % (val_width - size_width + 1)};
        }
    };
    int rehash(const T val_in_hand, const int depth);

    /** Inline helper functions. */
    template <typename SWAP_T>
    inline void do_swap(SWAP_T * const p1, SWAP_T * const p2) {
        SWAP_T tmp = *p1;
        *p1 = *p2;
        *p2 = tmp;
    }
    inline int do_hash(const T val, const int func_idx) {   // FUNC_IDX >= 1.
        FuncConfig fc = _hash_func_configs[func_idx];
        return ((val ^ fc.rv) >> fc.ss) % _size;
    }

public:

    /** Constructor & Destructor. */
    CuckooHashTable(const int size, const int evict_bound, const int num_funcs)
        : _size(size), _evict_bound(evict_bound), _num_funcs(num_funcs) {

        // Allocate space for data table, map, and hash function configs.
        _data = new T[size]();                  // Use "all-zero" mode,
        _pos_to_func_map = new int[size]();     // indicating initially they are all empty.
        _hash_func_configs = new FuncConfig[num_funcs + 1];
        
        // Generate initial hash function configs.
        gen_hash_funcs();
    };
    ~CuckooHashTable() {
        delete[] _data;
        delete[] _pos_to_func_map;
        delete[] _hash_func_configs;
    };

    /** Supported operations. */
    int insert_val(const T val, const int depth);
    bool delete_val(const T val);
    bool lookup_val(const T val);
    void show_content();
};


/**
 * 
 * Cuckoo: insert operation.
 *
 * Returns:
 *   Number of rehashings beneath.
 *   
 */
template <typename T>
int
CuckooHashTable<T>::insert_val(const T val, const int depth) {

    // Initial conditions.
    T cur_val = val;
    int cur_func = 1;
    int evict_count = 0;

    // Start the test-kick-and-reinsert loops.
    do {

        // Loop through all hash positions.
        for (int i = 0; i < _num_funcs; ++i) {
            int func_idx = (cur_func + i - 1) % _num_funcs + 1;
            int pos = do_hash(cur_val, func_idx);
            if (_data[pos] == EMPTY_CELL) {     // Empty cell found.
                _data[pos] = cur_val;
                _pos_to_func_map[pos] = func_idx;
                return 0;
            }
        }

        // No empty slots in all positions, then kick the key in first slot.
        int pos = do_hash(cur_val, cur_func);
        do_swap<T>(&cur_val, &_data[pos]);
        do_swap<int>(&cur_func, &_pos_to_func_map[pos]);
        cur_func = cur_func % _num_funcs + 1;
        evict_count++;
        
    } while (evict_count < _evict_bound);

    // Eviction bound met, need rehashing.
    int levels_beneath = rehash(cur_val, depth + 1);
    if (levels_beneath == ERR_DEPTH)
            return ERR_DEPTH;
        else
            return levels_beneath + 1;
}


/**
 * 
 * Cuckoo: delete operation.
 *
 * Returns:
 *   True,  if delete succeeds.
 *   False, if VAL not in the table.
 *   
 */
template <typename T>
bool
CuckooHashTable<T>::delete_val(const T val) {
    for (int i = 1; i <= _num_funcs; ++i) {
        int pos = do_hash(val, i);
        if (_data[pos] == val) {
            _data[pos] = EMPTY_CELL;
            _pos_to_func_map[pos] = EMPTY_CELL;
            return true;
        }
    }
    return false;
}


/**
 * 
 * Cuckoo: lookup operation.
 *
 * Returns:
 *   True,  if VAL is in the table.
 *   False, otherwise.
 *   
 */
template <typename T>
bool
CuckooHashTable<T>::lookup_val(const T val) {
    for (int i = 1; i <= _num_funcs; ++i) {
        int pos = do_hash(val, i);
        if (_data[pos] == val)
            return true;
    }
    return false;
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
CuckooHashTable<T>::rehash(const T val_in_hand, const int depth) {

    // If exceeds max rehashing depth, abort.
    if (depth > MAX_DEPTH)
        return ERR_DEPTH;

    // Generate new set of hash functions.
    gen_hash_funcs();

    // Clear data and map, put values into a buffer.
    std::vector<T> val_buffer;
    for (int i = 0; i < _size; ++i) {
        if (_data[i] != EMPTY_CELL)
            val_buffer.push_back(_data[i]);
        _data[i] = EMPTY_CELL;
        _pos_to_func_map[i] = EMPTY_CELL;
    }
    val_buffer.push_back(val_in_hand);

    // Re-insert all values.
    int max_levels_beneath = 0;
    for (auto val : val_buffer) {
        int levels_beneath = insert_val(val, depth);
        if (levels_beneath == ERR_DEPTH)
            return ERR_DEPTH;
        else if (levels_beneath > max_levels_beneath)
            max_levels_beneath = levels_beneath;
    }
    return max_levels_beneath;
}


/** Cuckoo: print content out. */
template <typename T>
void
CuckooHashTable<T>::show_content() {
    std::cout << "Funcs: ";
    for (int i = 1; i <= _num_funcs; ++i) {
        FuncConfig fc = _hash_func_configs[i];
        std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
    }
    std::cout << std::endl << "Table: ";
    for (int i = 0; i < _size; ++i)
        std::cout << std::setw(10) << _data[i] << " ";
    std::cout << std::endl << "Fcmap: ";
    for (int i = 0; i < _size; ++i)
        std::cout << std::setw(10) << _pos_to_func_map[i] << " ";
    std::cout << std::endl << std::endl;
}


#endif
