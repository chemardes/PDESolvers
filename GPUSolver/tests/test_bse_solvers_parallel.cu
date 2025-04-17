//
// Created by Chelsea De Marseilla on 18/03/2025.
//

#include "../gpu/bse_solvers_parallel.cuh"
#include <gtest/gtest.h>

class BSESolversTest : public ::testing::Test
{
protected:
    BSESolversTest() {
    }

    ~BSESolversTest() override {
    }

    void SetUp() override {
      grid_size = (t_nodes + 1) * (s_nodes + 1);
        gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(double)));
        gpuErrChk(cudaMemset(dev_grid, 0, grid_size * sizeof(double)));

        dS = s_max / s_nodes;
        dt = expiry / t_nodes;
    }

    void TearDown() override {
      if (dev_grid != nullptr) {
        cudaFree(dev_grid);
        dev_grid = nullptr;
      }
    }


    double *dev_grid = nullptr;
    int s_nodes = 100;
    int t_nodes = 100;
    double s_max = 200.0;
    double strike_price = 100.0;
    double rate = 0.05;
    double sigma = 0.2;
    double expiry = 1.0;
    double dS{};
    double dt{};
    size_t grid_size{};
};

TEST_F(BSESolversTest, TestSetBoundaryConditionsCall)
{
  set_boundary_conditions<OptionType::Call><<<numBlocks(t_nodes + 1), THREADS_PER_BLOCK>>>(
    dev_grid, t_nodes, s_nodes, s_max, strike_price, rate, expiry, dt);
  cudaDeviceSynchronize();

  double *host_grid = new double[grid_size];
  gpuErrChk(cudaMemcpy(host_grid, dev_grid, grid_size * sizeof(double), cudaMemcpyDeviceToHost));

  for(int i=0; i<=t_nodes; i++){
    double expected_value = s_max - strike_price * std::exp(-rate * (expiry - i * dt));
    // checks lower boundary condition
    EXPECT_DOUBLE_EQ(host_grid[i * (s_nodes + 1)], 0.0);

    // checks upper boundary condition
    EXPECT_NEAR(host_grid[i * (s_nodes + 1) + s_nodes], expected_value, 1e-10);
  }

  delete[] host_grid;
}

TEST_F(BSESolversTest, TestSetBoundaryConditionsPut)
{
  set_boundary_conditions<OptionType::Put><<<numBlocks(t_nodes + 1), THREADS_PER_BLOCK>>>(
    dev_grid, t_nodes, s_nodes, s_max, strike_price, rate, expiry, dt);
  cudaDeviceSynchronize();

  double *host_grid = new double[grid_size];
  gpuErrChk(cudaMemcpy(host_grid, dev_grid, grid_size * sizeof(double), cudaMemcpyDeviceToHost));

  for(int i=0; i<=t_nodes; i++){
    double expected_value = strike_price * std::exp(-rate * (expiry - i * dt));

    // checks lower boundary condition
    EXPECT_NEAR(host_grid[i * (s_nodes + 1)], expected_value, 1e-10);

    // checks upper boundary condition
    EXPECT_DOUBLE_EQ(host_grid[i * (s_nodes + 1) + s_nodes], 0.0);
  }

  delete[] host_grid;
}

TEST_F(BSESolversTest, TestSetTerminalConditionCall)
{
  set_terminal_condition<OptionType::Call><<<numBlocks(s_nodes + 1), THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
  cudaDeviceSynchronize();

  double *host_grid = new double[grid_size];
  gpuErrChk(cudaMemcpy(host_grid, dev_grid, grid_size * sizeof(double), cudaMemcpyDeviceToHost));

  int idx = (t_nodes) * (s_nodes + 1);
  for (int i=0; i<=s_nodes; i++)
  {
    double expected_value = max(i * dS - strike_price, 0.0);
    ASSERT_EQ(host_grid[idx + i], expected_value);
  }

  delete[] host_grid;
}

TEST_F(BSESolversTest, TestSetTerminalConditionPut)
{
  set_terminal_condition<OptionType::Put><<<numBlocks(s_nodes + 1), THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
  cudaDeviceSynchronize();

  double *host_grid = new double[grid_size];
  gpuErrChk(cudaMemcpy(host_grid, dev_grid, grid_size * sizeof(double), cudaMemcpyDeviceToHost));

  int idx = (t_nodes) * (s_nodes + 1);
  for (int i=0; i<=s_nodes; i++)
  {
    double expected_value = max(strike_price - i * dS, 0.0);
    ASSERT_EQ(host_grid[idx + i], expected_value);
  }

  delete[] host_grid;
}

TEST_F(BSESolversTest, TestComputeCoefficients)
{
  const int size = (s_nodes + 1);

  double *test_alpha, *test_beta, *test_gamma;
  gpuErrChk(cudaMalloc(&test_alpha, size * sizeof(double)));
  gpuErrChk(cudaMalloc(&test_beta, size * sizeof(double)));
  gpuErrChk(cudaMalloc(&test_gamma, size * sizeof(double)));

  compute_coefficients<<<numBlocks(size), THREADS_PER_BLOCK>>>(test_alpha, test_beta, test_gamma, sigma, rate, size, dS, dt);
  cudaDeviceSynchronize();

  double *host_alpha = new double[size];
  double *host_beta = new double[size];
  double *host_gamma = new double[size];

  gpuErrChk(cudaMemcpy(host_alpha, test_alpha, size * sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(host_beta, test_beta, size * sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(host_gamma, test_gamma, size * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i=0; i<size;i++)
  {
    double current_s = i * dS;
    double sigma_sq = sigma * sigma;
    double dS_sq_inv = 1.0 / pow(dS,2);

    double expected_alpha = 0.25 * dt * (sigma_sq * pow(current_s, 2) * dS_sq_inv - rate * current_s / dS);
    double expected_beta = -dt * 0.5 * (sigma_sq * pow(current_s, 2) * dS_sq_inv + rate);
    double expected_gamma = 0.25 * dt * (sigma_sq * pow(current_s, 2) * dS_sq_inv + rate * current_s / dS);

    EXPECT_NEAR(host_alpha[i], expected_alpha, 1e-10);
    EXPECT_NEAR(host_beta[i], expected_beta, 1e-10);
    EXPECT_NEAR(host_gamma[i], expected_gamma, 1e-10);

  }

  delete[] host_alpha;
  delete[] host_beta;
  delete[] host_gamma;
  cudaFree(test_alpha);
  cudaFree(test_beta);
  cudaFree(test_gamma);
}