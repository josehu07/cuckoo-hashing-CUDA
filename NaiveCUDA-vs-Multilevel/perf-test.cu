#include "cuckoo-cuda-naive.cuh"
#include "cuckoo-cuda-multi.cuh"
#include <map>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <ctime>


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
 * Main entrance for the performance test.
 * 
 * Prerequirests: we assume
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

    // Random seed.
    srand(time(NULL));

    // Experiment 1 - Insertion time.
    std::cout << "Experiment 1 -->" << std::endl;
    {
        int num_funcs = 3;
        for (int scale = 18; scale < 28; ++scale) {
            int n = 0x1 << scale, size = 2 * n;
            uint32_t *vals_to_insert = new uint32_t[n];
            gen_rnd_input(vals_to_insert, n, 0x1 << 30);
            for (int rep = 0; rep < 5; ++rep) {
                std::cout << " scale = " << scale << ","
                          << " rep " << rep << ":  " << std::flush;

                // CUDA naive.
                {
                    CuckooHashTableCuda_Naive<uint32_t> hash_table(size, 4 * ceil(log2((double) n)),
                                                                   num_funcs);
                    auto ts = std::chrono::high_resolution_clock::now();
                    int levels = hash_table.insert_vals(vals_to_insert, n, 0);
                    auto te = std::chrono::high_resolution_clock::now();
                    std::cout << "[Naive] " << std::setw(5)
                              << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                              << " ms - ";
                    if (levels == ERR_DEPTH)
                        std::cout << "exceeds " << MAX_DEPTH << " levels | " << std::flush;
                    else
                        std::cout << std::setw(2) << levels << " rehash(es) | " << std::flush;
                }

                // CUDA multi-level.
                {
                    CuckooHashTableCuda_Multi<uint32_t> hash_table(size, 4 * ceil(log2((double) n)),
                                                                   num_funcs);
                    auto ts = std::chrono::high_resolution_clock::now();
                    int levels = hash_table.insert_vals(vals_to_insert, n);
                    auto te = std::chrono::high_resolution_clock::now();
                    std::cout << "[Multi] " << std::setw(5)
                              << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                              << " ms - ";
                    if (levels == ERR_DEPTH)
                        std::cout << "exceeds " << MAX_DEPTH << " levels" << std::endl;
                    else
                        std::cout << std::setw(2) << levels << " rehash(es)" << std::endl;
                }
            }
            delete[] vals_to_insert;
        }
    }

    // Experiment 2 - Lookup time.
    std::cout << "Experiment 2 -->" << std::endl;
    {
        int num_funcs = 3, percent = 5;
        for (int scale = 18; scale < 28; ++scale) {
            int n = 0x1 << scale, size = 2 * n;
            uint32_t *vals_to_insert = new uint32_t[n];
            gen_rnd_input(vals_to_insert, n, 0x1 << 30);
            uint32_t *vals_to_lookup = new uint32_t[n];
            bool *results = new bool[n];
            int bound = ceil((1 - 0.1 * percent) * n);
            for (int rep = 0; rep < 5; ++rep) {
                for (int i = 0; i < bound; ++i)
                    vals_to_lookup[i] = vals_to_insert[rand() % n];
                for (int i = bound; i < n; ++i)
                    vals_to_lookup[i] = (rand() % ((0x1 << 30) - 1)) + 1;
                std::cout << " rep " << rep << ":  " << std::flush;

                // CUDA naive.
                {
                    CuckooHashTableCuda_Naive<uint32_t> hash_table(size, 4 * ceil(log2((double) n)),
                                                                   num_funcs);
                    hash_table.insert_vals(vals_to_insert, n, 0);
                    auto ts = std::chrono::high_resolution_clock::now();
                    hash_table.lookup_vals(vals_to_lookup, results, n);
                    auto te = std::chrono::high_resolution_clock::now();
                    std::cout << "[Naive] " << std::setw(5)
                              << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                              << " ms | " << std::flush;
                }

                // CUDA multi-level.
                {
                    CuckooHashTableCuda_Multi<uint32_t> hash_table(size, 4 * ceil(log2((double) n)),
                                                                   num_funcs);
                    hash_table.insert_vals(vals_to_insert, n);
                    auto ts = std::chrono::high_resolution_clock::now();
                    hash_table.lookup_vals(vals_to_lookup, results, n);
                    auto te = std::chrono::high_resolution_clock::now();
                    std::cout << "[Multi] " << std::setw(5)
                              << std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count()
                              << " ms" << std::endl;
                }
            }
            delete[] vals_to_insert;
            delete[] vals_to_lookup;
            delete[] results;
        }
    }
}
