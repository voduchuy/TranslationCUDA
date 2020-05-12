//
// Created by huy on 12/14/19.
//
// This program demonstrates the library simulation functionalities using translation model input by the user.
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cutrans.h"
#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sys/stat.h>
#include <chrono>

template <typename T>
void load_vec(const std::string& filename, thrust::host_vector<T>* out_vec){
  std::ifstream f;

  f.open(filename, std::ifstream::in);
  if (f.is_open()){
    std::istream_iterator<T> fstart(f);
    std::vector<T> stl_vec(fstart, std::istream_iterator<T>());
    out_vec->resize(stl_vec.size());
    thrust::copy(stl_vec.begin(), stl_vec.end(), out_vec->begin());
  }
  else{
    throw std::runtime_error("ERROR: Cannot open file "+filename+".\n");
  }
  f.close();
}


int main(int argc, char **argv) {
  thrust::host_vector<double> times, rates;
  thrust::host_vector<int> rib_locations_0, probe_design;
  int num_samples, n_excl;
  std::string directory;

  try{
    load_vec("times.txt", &times);
    load_vec("rates.txt", &rates);
    load_vec("x0.txt", &rib_locations_0);
    load_vec("c.txt", &probe_design);
  }
  catch(std::runtime_error& e){
    std::cout << e.what() << std::endl;
    return -1;
  }

  num_samples = 1000;
  n_excl = 7;

  ssit::CuTransSimulator simulator;
  simulator.SetSampleSize(num_samples);
  simulator.SetModel(rates, probe_design, n_excl);
  simulator.SetTimeNodes(times);
  simulator.SetInitialState(rib_locations_0);


  auto tic = std::chrono::high_resolution_clock::now();
  simulator.Simulate();
  cudaDeviceSynchronize();
  auto toc = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic);
  std::cout << "Simulation time is " << elapsed.count() << " miliseconds. \n";

  thrust::host_vector<int> intensity;
  simulator.GetIntensityTrajectories(&intensity);

  thrust::host_vector<int> x;
  simulator.GetFinalStates(&x);

  std::ofstream f;
  f.open("output.txt", std::ios_base::out);
  for (int i{0}; i < num_samples; ++i){
    thrust::copy(intensity.begin()+i*times.size(), intensity.begin() + (i+1)*times.size(),
                 std::ostream_iterator<int>(f, " "));
    f << "\n";
  }

  f.close();

  std::cout << "Samples are written to output.txt in current working directory.\n";
  return 0;
}