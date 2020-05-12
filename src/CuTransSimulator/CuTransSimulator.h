//
// Created by huy on 5/10/20.
//

#ifndef TRANSLATIONCUDA_SRC_CUTRANSSIMULATOR_H_
#define TRANSLATIONCUDA_SRC_CUTRANSSIMULATOR_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <curand_kernel.h>
#include "CuTransKernels.h"

#define CUDACHKERR() { \
cudaError_t ierr = cudaGetLastError();\
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\
}\

namespace ssit{
class CuTransSimulator {
 protected:
  int _num_samples = 1000;
  int _num_ribosomes = 64;
  int _gene_length;
  int _num_exclusion = 1;
  bool _set_up = false;
  int _rand_seed = 0;

  thrust::host_vector<int> _initial_state;
  thrust::host_vector<double> _rates;
  thrust::host_vector<int> _probe_design;
  thrust::host_vector<double> _time_nodes;
  thrust::device_vector<int> _dev_states;

  thrust::device_vector<curandState_t> _rand_states;
  thrust::device_vector<double> _dev_rates;
  thrust::device_vector<int> _dev_probe_design;
  thrust::device_vector<int> _dev_intensity;
  thrust::device_vector<double> _dev_time_nodes;
 public:
  int SetInitialState(int num_rib, int *rib_locations);
  int SetInitialState(const thrust::host_vector<int>& rib_locations);

  int SetSampleSize(int n_samp);
  int SetNumRibosomes(int n_rib);
  int SetTimeNodes(const thrust::host_vector<double> &time_nodes);
  int SetSeed(int seed);

  int SetModel(int gene_length, const double *rates, const int *probe_design, int n_exclusion);
  int SetModel(const thrust::host_vector<double> &rates, const thrust::host_vector<int> &probe_design, int n_exclusion);

  int SetUp();
  int Reset();

  int Simulate();
  int GetFinalStates(thrust::host_vector<int> *x_out);
  int GetIntensityTrajectories(thrust::host_vector<int> *intensity_out);
};
}

#endif //TRANSLATIONCUDA_SRC_CUTRANSSIMULATOR_H_
