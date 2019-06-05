#include "cuckoo-serial.hpp"
#include "cuckoo-cuda.cuh"
#include <map>
#include <cstdlib>
#include <cmath>
#include <cstdint>


/** Random input generator. */
static void
gen_rnd_input(uint32_t * const vals, const int n, const uint32_t limit) {
    std::map<uint32_t, bool> val_map;
    int count = 0;
    while (count < n) {
        uint32_t val = (rand() % (limit - 1)) + 1;
        if (val_map.find(val) != val_map.end())
            continue;
        val_map[val] = true;
        vals[count] = val;
        count++;
    }
}


/**
 *
 * Main entrance for the demo.
 * 
 * Prerequirests: we assume
 *   1. Value range do not cover EMPTY_CELL (i.e. 0).
 *   2. Value range do not exceed value-field width.
 *   3. No repeated keys inserted (so we skipped duplication check).
 *
 * Currently supported types:
 *   uint[8, 16, 32]_t
 *   
 */
int
main(void) {

    // DEMO of serial implementation.
    std::cout << "Serial implementation DEMO -->" << std::endl << std::endl;
    {
        CuckooHashTable<uint32_t> table_serial(8, 4 * ceil(log2((double) 8)), 3);
        table_serial.show_content();

        std::cout << "Insert 8 values -" << std::endl;
        uint32_t vals_to_insert[8];
        gen_rnd_input(vals_to_insert, 8, 0x1 << 30);
        for (int i = 0 ; i < 8; ++i)
            table_serial.insert_val(vals_to_insert[i], 0);
        table_serial.show_content();

        std::cout << "Delete values [0..4] -" << std::endl;
        for (int i = 0; i < 4; ++i)
            table_serial.delete_val(vals_to_insert[i]);
        table_serial.show_content();

        std::cout << "Lookup values [2..6] -" << std::endl;
        bool results[4];
        for (int i = 0; i < 4; ++i)
            results[i] = table_serial.lookup_val(vals_to_insert[i + 2]);
        std::cout << "Results - ";
        for (int i = 0; i < 4; ++i)
            std::cout << results[i] << " ";
        std::cout << std::endl;
        table_serial.show_content();         
    }

    // DEMO of CUDA implementation.
    std::cout << "CUDA implementation DEMO -->" << std::endl << std::endl;
    {
        CuckooHashTableCuda_Naive<uint32_t> table_cuda(8, 4 * ceil(log2((double) 8)), 3);
        table_cuda.show_content();

        std::cout << "Insert 8 values -" << std::endl;
        uint32_t vals_to_insert[8];
        gen_rnd_input(vals_to_insert, 8, 0x1 << 30);
        table_cuda.insert_vals(vals_to_insert, 8, 0);
        table_cuda.show_content();

        std::cout << "Delete values [0..4] -" << std::endl;
        uint32_t vals_to_delete[4];
        for (int i = 0; i < 4; ++i)
            vals_to_delete[i] = vals_to_insert[i];
        table_cuda.delete_vals(vals_to_delete, 4);
        table_cuda.show_content();

        std::cout << "Lookup values [2..6] -" << std::endl;
        uint32_t vals_to_lookup[4];
        for (int i = 0; i < 4; ++i)
            vals_to_lookup[i] = vals_to_insert[i + 2];
        bool results[4];
        table_cuda.lookup_vals(vals_to_lookup, results, 4);
        std::cout << "Results - ";
        for (int i = 0; i < 4; ++i)
            std::cout << results[i] << " ";
        std::cout << std::endl;
        table_cuda.show_content();  
    }

    return 0;
}
