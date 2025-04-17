# TESTING
# ------------------------------------------
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# for Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(gpusolver_test)

target_sources(gpusolver_test
        PRIVATE
        test_bse_solvers_parallel.cu
)
target_link_libraries(gpusolver_test
        PRIVATE
        CUDA::cusparse
        CUDA::cudart
        GTest::gtest_main)
target_include_directories(gpusolver_test
        PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gpu"
)

include(GoogleTest)
gtest_discover_tests(gpusolver_test)
