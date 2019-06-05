#include "cuckoo-cuda-naive.cuh"
#include "cuckoo-cuda-multi.cuh"
#include <map>
#include <cstdlib>
#include <cmath>
#include <cstdint>


/**
 *
 * Main entrance for the demo.
 * 
 * Prerequisits: we assume
 *   1. Value range do not cover EMPTY_CELL (i.e. 0).
 *   2. Value range do not exceed value-field width.
 *   3. No repeated keys inserted (so we skipped duplication check).
 *   4. Table size must be a multiple of BUCKET_SIZE.
 *   5. Only inserting into an empty table. No updating. (o.w. the rehashing part should be rewritten.)
 *
 * Currently supported types:
 *   uint[8, 16, 32]_t
 *   
 */
int
main(void) {

    // DEMO of CUDA naive implementation.
    std::cout << "CUDA naive implementation DEMO -->" << std::endl << std::endl;
    {
        CuckooHashTableCuda_Naive<uint32_t> table_cuda(8, 4 * ceil(log2((double) 8)), 3);
        table_cuda.show_content();

        std::cout << "Insert 6 values -" << std::endl;
        uint32_t vals_to_insert[6] = {5, 6, 7, 8, 9, 10};
        table_cuda.insert_vals(vals_to_insert, 6, 0);
        table_cuda.show_content();

        std::cout << "Delete values [0..3] -" << std::endl;
        uint32_t vals_to_delete[3];
        for (int i = 0; i < 3; ++i)
            vals_to_delete[i] = vals_to_insert[i];
        table_cuda.delete_vals(vals_to_delete, 3);
        table_cuda.show_content();

        std::cout << "Lookup values [2..4] -" << std::endl;
        uint32_t vals_to_lookup[2];
        for (int i = 0; i < 2; ++i)
            vals_to_lookup[i] = vals_to_insert[i + 2];
        bool results[2];
        table_cuda.lookup_vals(vals_to_lookup, results, 2);
        std::cout << "Results - ";
        for (int i = 0; i < 2; ++i)
            std::cout << results[i] << " ";
        std::cout << std::endl;
        table_cuda.show_content();
    }

    // DEMO of CUDA multi-level implementation.
    std::cout << "CUDA multi-level implementation DEMO -->" << std::endl << std::endl;
    {
        if (BUCKET_SIZE != 4) {
            std::cerr << "ERROR: This demo is only workable when setting BUCKET_SIZE = 4." << std::endl;
            exit(-1);
        }

        CuckooHashTableCuda_Multi<uint32_t> table_cuda(8, 4 * ceil(log2((double) 8)), 3);
        table_cuda.show_content();

        std::cout << "Insert 6 values -" << std::endl;
        uint32_t vals_to_insert[6] = {5, 6, 7, 8, 9, 10};
        table_cuda.insert_vals(vals_to_insert, 6);
        table_cuda.show_content();

        std::cout << "Delete values [0..3] -" << std::endl;
        uint32_t vals_to_delete[3];
        for (int i = 0; i < 3; ++i)
            vals_to_delete[i] = vals_to_insert[i];
        table_cuda.delete_vals(vals_to_delete, 3);
        table_cuda.show_content();

        std::cout << "Lookup values [2..4] -" << std::endl;
        uint32_t vals_to_lookup[2];
        for (int i = 0; i < 2; ++i)
            vals_to_lookup[i] = vals_to_insert[i + 2];
        bool results[2];
        table_cuda.lookup_vals(vals_to_lookup, results, 2);
        std::cout << "Results - ";
        for (int i = 0; i < 2; ++i)
            std::cout << results[i] << " ";
        std::cout << std::endl;
        table_cuda.show_content(); 
    }

    return 0;
}
