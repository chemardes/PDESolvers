//
// Created by Chelsea De Marseilla on 17/04/2025.
//

#ifndef COMMON_CUH
#define COMMON_CUH

/**
 * Define defaults
 */
#define DEFAULT_FPX double
#define THREADS_PER_BLOCK 512

/**
 * Determines the number of blocks needed for a given number of tasks, n,
 * and number of threads per block
 *
 * @param n problem size
 * @param threads_per_block threads per block (defaults to THREADS_PER_BLOCK)
 * @return number of blocks
 */
constexpr size_t numBlocks(size_t n, size_t threads_per_block = THREADS_PER_BLOCK) {
    return (n / threads_per_block + (n % threads_per_block != 0));
}

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), __FILE__, __LINE__); } while(false)

template<typename T = DEFAULT_FPX>
inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else if constexpr(std::is_same_v<T, cusparseStatus_t>) {
        if (code != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "cublas error. Code: " << cusparseGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}

/**
 * Option type (call or put)
 */
enum class OptionType {
    Call,
    Put
};

#endif //COMMON_CUH
