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

template <typename T>
void interactive_vec_input(const std::string& dir,
                           const std::string& default_filename,
                           const std::string& array_name,
                           thrust::host_vector<T> *out_vec) {
  std::string filename;
  while (true){
    try{
      std::cout << "Please enter filename for "+array_name+" (default: " << default_filename << "): ";
      std::getline(std::cin, filename);
      if (filename.empty()){
        filename = default_filename;
      }
      load_vec<T>(dir + filename, out_vec);
      break;
    }
    catch(std::runtime_error& e){
      std::cout << "Wrong path. Please enter again.\n";
    }
  }
}

int main(int argc, char **argv) {
  thrust::host_vector<double> times, rates;
  thrust::host_vector<int> rib_locations_0, probe_design;
  int num_samples, n_excl;
  std::string directory;

  while(true){
    try{
      std::cout << "Please enter path to the directory that contain model information (default: cwd):";
      std::getline(std::cin, directory);

      if (!directory.empty()){
        struct stat info;
        if (stat(directory.c_str(), &info) != 0){
          throw std::runtime_error("Directory does not exist.");
        }
        else if(! (info.st_mode & S_IFDIR)){
          throw std::runtime_error("Not a directory.");
        }
        directory += "/";
      }
      break;
    }
    catch(std::runtime_error& e){
      std::cout << e.what() << " Please try again.\n";
    }
  }

  interactive_vec_input(directory, "times.txt", "times", &times);
  interactive_vec_input(directory, "rates.txt", "rates", &rates);
  interactive_vec_input(directory, "x0.txt", "initial ribosome locations", &rib_locations_0);
  interactive_vec_input(directory, "c.txt", "probe design", &probe_design);

  std::cout << "Enter number of samples to draw: ";
  std::cin >> num_samples;

  std::cout << "Enter the exclusion parameter:";
  std::cin >> n_excl;

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