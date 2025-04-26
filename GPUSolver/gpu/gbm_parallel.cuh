//
// Created by Chelsea De Marseilla on 17/04/2025.
//

#ifndef GBM_PARALLEL_CUH
#define GBM_PARALLEL_CUH

#include <cmath>
#include <iostream>
#include <curand_kernel.h>
#include <fstream>
#include <chrono>
#include <filesystem>
#include "common.cuh"

/**
 * Define defaults
 */
#define SEED 12345

template<typename T = DEFAULT_FPX>
__global__ static void simulate_gbm(T* grid, T initial_stock_price, T mu, T sigma, T time, int time_steps, int num_of_simulations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_of_simulations)
    {

        // initialising for random normal distribution
        curandState state;
        curand_init(SEED, idx, 0, &state);

        T dt = time/static_cast<T>(time_steps);

        T br = 0;
        T br_prev = 0;
        grid[idx * time_steps] = initial_stock_price;

        for (int i = 1; i < time_steps; i++)
        {
            T Z = curand_normal(&state);
            br = br_prev + std::sqrt(dt) * Z;
            T delta_br = br - br_prev;
            grid[idx * time_steps + i] = grid[idx * time_steps + i - 1] * expf((mu - 0.5f * powf(sigma, 2)) * dt + sigma * delta_br);
            br = br_prev;
        }
    }
}

/**
 * GBM Simulation Results
 * @tparam T
 */
template<typename T = DEFAULT_FPX>
class SimulationResults
{
private:
    static std::string get_output_file_path(const std::string& filename)
    {
        std::filesystem::path current_path = std::filesystem::current_path();
        return (current_path/ filename).string();
    }

    std::ostream &print(std::ostream &out) const
    {
        T* host_data = new T[grid_size()];
        download(host_data);

        // gets output for file path
        std::string file_path = get_output_file_path("gbm.csv");

        // exports to csv file
        std::ofstream csv_file(file_path);

        csv_file << "Simulation,Time Step,Stock Price\n";
        for (int i = 0; i < m_number_simulations; i++)
        {
            for (int j = 0; j < m_time_steps; j++)
            {
                csv_file << i << "," << j << "," << host_data[i * m_time_steps + j] << "\n";
            }
        }
        csv_file.close();

        out << "Data exported to " << file_path <<" successfully" << std::endl;

        delete[] host_data;
        return out;
    }

public:
    T* m_d_data  = nullptr;
    size_t m_number_simulations;
    size_t m_time_steps;
    double m_duration = 0;

    SimulationResults(T *data, size_t number_simulations, size_t time_steps)
    {
        m_d_data = data;
        m_number_simulations = number_simulations;
        m_time_steps = time_steps;
    }

    ~SimulationResults()
    {
        if (m_d_data)
        {
            gpuErrChk(cudaFree(m_d_data));
            m_d_data = 0;
        }
    }

    size_t grid_size() const
    {
        return m_number_simulations * m_time_steps;
    }

    void download(T *host_data) const {
        if (m_d_data) {
            gpuErrChk(cudaMemcpy(host_data, m_d_data, grid_size() * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    friend std::ostream &operator << (std::ostream &out, const SimulationResults<T> &data) {
        return data.print(out);
    }

};

template<typename T = DEFAULT_FPX>
SimulationResults<T> simulate(T initial_stock_price, T mu, T sigma, T time, int num_of_simulations, int time_steps)
{
    auto start = std::chrono::high_resolution_clock::now();

    size_t grid_size = num_of_simulations * time_steps;

    T *dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(T)));

    simulate_gbm<<<numBlocks(num_of_simulations), THREADS_PER_BLOCK>>>(dev_grid, initial_stock_price, mu, sigma, time, time_steps, num_of_simulations);
    gpuErrChk(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    SimulationResults<T> sol(dev_grid, num_of_simulations, time_steps);
    sol.m_duration = (double) duration.count() / 1e6 ;

    return sol;
}

#endif //GBM_PARALLEL_CUH
