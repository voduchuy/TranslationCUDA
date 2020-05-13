//
// Created by huy on 5/10/20.
//

#include "CuTransSimulator.h"
using namespace ssit;

int ssit::CuTransSimulator::SetInitialState(int num_rib, int *rib_locations) {
  _num_ribosomes = num_rib;
  _initial_state.resize(num_rib);
  thrust::copy(thrust::host, rib_locations, rib_locations + num_rib, _initial_state.begin());
  return 0;
}
int ssit::CuTransSimulator::SetInitialState(const thrust::host_vector<int> &rib_locations) {
  _initial_state.resize(rib_locations.size());
  thrust::copy(rib_locations.begin(), rib_locations.end(), _initial_state.begin());
  _num_ribosomes = _initial_state.size();
  return 0;
}
int ssit::CuTransSimulator::SetSampleSize(int n_samp) {
  _num_samples = n_samp;
  return 0;
}
int ssit::CuTransSimulator::SetNumRibosomes(int n_rib) {
  _num_ribosomes = n_rib;
  return 0;
}

int ssit::CuTransSimulator::SetModel(int gene_length, const double *rates, const int *probe_design, int n_exclusion=1) {
  _probe_design.resize(gene_length + 1);
  _rates.resize(gene_length + 1);
  _num_exclusion = n_exclusion;

  thrust::copy(thrust::host, rates, rates + gene_length + 1, _rates.begin());
  thrust::copy(thrust::host, probe_design, probe_design + gene_length + 1, _probe_design.begin());
  return 0;
}

int CuTransSimulator::SetModel(const thrust::host_vector<double> &rates,
                               const thrust::host_vector<int> &probe_design,
                               int n_exclusion) {
  _num_exclusion = n_exclusion;
  _rates = rates;
  _probe_design = probe_design;
  _gene_length = _rates.size()-1;
  return 0;
}

int ssit::CuTransSimulator::SetUp() {

  _dev_time_nodes = _time_nodes;
  _dev_probe_design = _probe_design;
  _dev_rates = _rates;

  _dev_states.resize(_num_ribosomes * _num_samples);
  _dev_intensity.resize(_num_samples*_time_nodes.size());

  _rand_states.resize(_num_samples);
  init_rand_states<<<_num_samples, 1>>>(thrust::raw_pointer_cast(&_rand_states[0]), _rand_seed);
  CUDACHKERR();

  _set_up = true;
  return 0;
}
int CuTransSimulator::Reset() {
  _dev_states.clear();
  _dev_rates.clear();
  _dev_probe_design.clear();
  _dev_intensity.clear();
  _rand_states.clear();
  _set_up = false;
  return 0;
}

int CuTransSimulator::Simulate() {
  if (!_set_up) SetUp();
  thrust::copy(_initial_state.begin(), _initial_state.end(), _dev_states.begin());
  if (_num_samples > 1) {
    _replicate_array<<<_num_samples-1, _threads_per_trajectory, 0>>>(_num_ribosomes,
                                                thrust::raw_pointer_cast(&_dev_states[0]),
                                                thrust::raw_pointer_cast(&_dev_states[_num_ribosomes]));
  }
  CUDACHKERR();

  size_t shared_mem_size =
      _num_ribosomes * sizeof(double) // for propensities
      + _num_ribosomes * sizeof(int) // for ribosome locations
      + _num_ribosomes * sizeof(int) // temporary space to copy ribosome locations (when shifting)
  ;

  update_state<<<_num_samples, _threads_per_trajectory, shared_mem_size>>>(_time_nodes.size(),
                                                     thrust::raw_pointer_cast(&_dev_time_nodes[0]),
                                                     _num_exclusion,
                                                     _gene_length,
                                                     _num_ribosomes,
                                                     thrust::raw_pointer_cast(&_dev_states[0]),
                                                     thrust::raw_pointer_cast(&_rand_states[0]),
                                                     thrust::raw_pointer_cast(&_dev_rates[0]),
                                                     thrust::raw_pointer_cast(&_dev_probe_design[0]),
                                                     thrust::raw_pointer_cast(&_dev_intensity[0]));
  cudaDeviceSynchronize();
  CUDACHKERR();
  return 0;
}

int CuTransSimulator::SetTimeNodes(const thrust::host_vector<double> &time_nodes) {
  _time_nodes = time_nodes;
  return 0;
}
int CuTransSimulator::SetSeed(int seed) {
  _rand_seed = seed;
  return 0;
}

int CuTransSimulator::GetIntensityTrajectories(thrust::host_vector<int> *intensity_out) {
  intensity_out->resize(_num_samples*_time_nodes.size());
  (*intensity_out) = _dev_intensity;
  return 0;
}
int CuTransSimulator::GetFinalStates(thrust::host_vector<int> *x_out) {
  x_out->resize(_num_samples*_num_ribosomes);
  (*x_out) = _dev_states;
  return 0;
}

