# GPUSolver 🚀

GPU-accelerated CUDA C++ library for solving partial differential equations (PDEs), designed for high-performance
numerical simulations.
> **📝 Note:** This library is developed independently from the Python package `pdesolvers` and is not current currently integrated with it

## 🧩 GPU-Accelerated Features
- **Explicit Method for Black-Scholes PDE**
- **Crank-Nicolson Method for Black-Scholes PDE**
- **Geometric Brownian Motion (GBM) Simulations**

## 🔧 Dependencies
- **cuSPARSE**: For sparse matrix operations.
- **cuBLAS**: For dense matrix operations.
- **cuRAND**: For random number generation.

## Example Usage for GPU-Accelerated PDE Solvers

#### 1. Include the necessary headers for the GPU solvers:
```c++
#include "gpu/bse_solvers_parallel.cuh"
#include "gpu/gbm_parallel.cuh"
```
#### 2. Define parameters for the Black-Scholes PDE and GBM simulation:

```c++
/* Parameters for Black-Scholes PDE */
constexpr OptionType type = OptionType::Call;
double s_max = 300.0;
double expiry = 1;
double sigma = 0.2;
double rate = 0.05;
double strike_price = 100;
int s_nodes = 1000;
int t_nodes = 1000000;

/* Parameters for Geometric Brownian Motion */
double initial_stock_price = 290.0;
double time = 1;
int time_steps = 365;
int num_of_simulations = 800000;
```
#### 3. Perform GPU computations
```c++
/* GPU Computation and timing */
Solution<double> solution1 = solve_bse_explicit<type>(s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes);
Solution<double> solution2 = solve_bse_cn<type>(s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes);
SimulationResults<double> solution3 = simulate<double>(initial_stock_price, rate, sigma, time, num_of_simulations, time_steps);
```
#### 4. Output Execution Time
```c++
/* Output timing information */
std::cout << "[GPU] Explicit method finished in " << solution1.m_duration << "s" << std::endl;
std::cout << "[GPU] Crank Nicolson method finished in " << solution2.m_duration << "s" << std::endl;
std::cout << "[GPU] GBM method finished in " << solution3.m_duration << "s" << std::endl;
```
#### 5. Download results from GPU to host memory
```c++
/* 3. Download results from GPU to host */
double *host_grid1 = new double[solution1.grid_size()];
solution1.download(host_grid1);
double *host_grid2 = new double[solution2.grid_size()];
solution2.download(host_grid2);
double *host_grid3 = new double[solution3.grid_size()];
solution3.download(host_grid3);
```
#### 6. Export results to CSV
```c++
/* Export the results to CSV */
std::cout << solution1;
std::cout << solution2;
std::cout << solution3;
```

#### 7. Memory Cleanup
```c++
delete[] host_grid1;
delete[] host_grid2;
delete[] host_grid3;
```
