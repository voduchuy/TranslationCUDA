//
// Created by huy on 12/14/19.
//
// This test program will test the ability of the library to sample 1000 gene intensity trajectories for a simple gene model of 5000 codons with all elongation and initiation rates set to 1. The results pass the test if all intensity values are between 0 and the length of the gene (5000 in this case), and that the final ribosome locations are more than 4 codons apart from each other.
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cutrans.h"

const int num_rib_max = 128;
const int num_samples = 1000;
const int n_excl = 4;
const int gene_len = 5000;
const int num_times = 4;

int main(int argc, char **argv) {
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

  thrust::host_vector<int> final_locations, intensity_host;
  my_simulator.GetIntensityTrajectories(&intensity_host);
  my_simulator.GetFinalStates(&final_locations);

  bool pass = true;
  for (int i{0}; i < num_samples; ++i) {
    for (int j{1}; j < num_rib_max; ++j) {
      if ((final_locations[i * num_rib_max + j] > 0) &&
          (final_locations[i * num_rib_max + j - 1] - final_locations[i * num_rib_max + j] < n_excl)) {
        std::cout << "Ribosome locations violate exclusion constraint!\n";
        pass = false;
        break;
      }
    }
    for (int j{0}; j < num_times; ++j) {
      if (intensity_host[i * num_times + j] > gene_len || intensity_host[i * num_times + j] < 0) {
        pass = false;
        break;
      }
    }
  }
  if (!pass) {
    std::cout << "Library does not pass a simple model test. DO NOT USE!" << std::endl;
  } else {
    std::cout << "Library passes a simple test case." << std::endl;
  }
  return 0;
}