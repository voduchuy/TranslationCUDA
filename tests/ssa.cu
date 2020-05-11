//
// Created by huy on 12/14/19.
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "CuTransSimulator.h"

int main(int argc, char **argv) {
  const int num_rib_max = 128;
  const int num_samples = 1000;
  const int n_excl = 4;
  const int gene_len = 5000;

  const int num_times = 4;

  thrust::host_vector<double> t_array(4);
  thrust::host_vector<double> rates(gene_len + 1, 1.0);
  thrust::host_vector<int> probe_design(gene_len + 1);
  thrust::host_vector<int> x0(num_rib_max);

  t_array[0] = 0.1;
  t_array[1] = 1.0;
  t_array[2] = 10.0;
  t_array[3] = 100.0;
  t_array[4] = 1000.0;

  thrust::fill(probe_design.begin() + 1, probe_design.end(), 1);
  thrust::fill(x0.begin(), x0.end(), 0);

  auto my_simulator = ssit::CuTransSimulator();
  my_simulator.SetInitialState(x0);
  my_simulator.SetTimeNodes(t_array);
  my_simulator.SetModel(rates, probe_design, n_excl);
  my_simulator.SetSampleSize(num_samples);
  my_simulator.SetUp();
  my_simulator.Simulate();

  thrust::host_vector<int> intensity_host;
  my_simulator.GetIntensityTrajectories(&intensity_host);

  for (int i{0}; i < num_samples; ++i) {
    for (int j{0}; j < num_times; ++j) {
      std::cout << intensity_host[i * num_times + j] << " ";
    }
    std::cout << "\n";
  }

  cudaDeviceReset();
  return 0;
}