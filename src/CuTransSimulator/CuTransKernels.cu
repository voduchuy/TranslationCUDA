//
// Created by huy on 5/10/20.
//

#include "CuTransKernels.h"

namespace ssit {

__global__
void init_rand_states(curandState_t *rstates, int seed = 0) {
  curand_init(seed, blockIdx.x*blockDim.x + threadIdx.x, 0, &rstates[blockIdx.x*blockDim.x + threadIdx.x]);
}

__global__
void update_state(const int num_times,
                  const double *t_array,
                  int num_excl,
                  int gene_len,
                  int num_rib_max,
                  int *X,
                  curandState_t *rstates,
                  const double *rates,
                  const int *probe_design,
                  int *intensity) {
  const uint &thread_id = threadIdx.x;
  const uint &sample_id = blockIdx.x;
  curandState_t rstate_loc = rstates[sample_id*blockDim.x + thread_id];

  extern __shared__ double wsp[];
  // Partition the shared memory into appropriate arrays
  double *rn = wsp;
  double *t_now_ptr = wsp + blockDim.x;
  double *stepsize_ptr = t_now_ptr + 1;
  double *doub_wsp = stepsize_ptr + 1;
  double *rate_caches = doub_wsp + num_rib_max;
  int *x_shared = ( int * ) (rate_caches + num_rib_max);
  int *x_wsp = x_shared + num_rib_max;
  int *c_caches = x_wsp + num_rib_max;
  int *to_shift_ptr = c_caches + num_rib_max;
  int *idx_to_output = to_shift_ptr + 1;
  int *n_active_ptr = idx_to_output + 1;
  int *i_random_ptr = n_active_ptr + 1;

  int *current_intensity;
  double &t_now = *t_now_ptr;
  double &stepsize = *stepsize_ptr;
  int &to_shift = *to_shift_ptr;
  const double &t_final = t_array[num_times - 1];
  int &n_active = *n_active_ptr;
  int &i_random = *i_random_ptr;

  // INITIALIZATION
  // Init time
  if (thread_id == 0) {
    t_now = 0.0;
    to_shift = 0;
    *idx_to_output = 0;
  }

  // Copy initial ribosome locations and rates to shared memory
  uint idx;
  uint k{0};
  while ((idx = k * blockDim.x + thread_id) < num_rib_max) {
    x_shared[idx] = X[sample_id * num_rib_max + idx];
    rate_caches[idx] = rates[x_shared[idx]];
    c_caches[idx] = probe_design[x_shared[idx]];
    k++;
  }
  __syncthreads();
  if (thread_id == 0) {
    n_active = 0;
    for (int i{0}; i < num_rib_max; ++i) {
      if (x_shared[i] > 0) n_active++;
    }
    n_active++;
  }
  __syncthreads();
  // Draw the first cache of random numbers
  rn[thread_id] = curand_uniform_double(&rstate_loc);
  if (thread_id == 0) i_random = 0;
  __syncthreads();
  // STEPPING
  while (t_now < t_final) {
    // Compute current intensity
    k = 0;
    while ((idx = k * blockDim.x + thread_id) < n_active) {
      x_wsp[idx] = c_caches[idx];
      k++;
    }
    __syncthreads();
    if (thread_id == 0) thrust::inclusive_scan(thrust::seq, x_wsp, x_wsp + n_active, x_wsp);
    __syncthreads();
    if (thread_id == 0) current_intensity = x_wsp + n_active - 1;
    // copy current intensity to appropriate locations in global memory
    if (thread_id == 0) {
      while (t_array[*idx_to_output] <= t_now) {
        intensity[sample_id * num_times + (*idx_to_output)] = (*current_intensity);
        (*idx_to_output)++;
        if (*idx_to_output >= num_times) {
          break;
        }
      }
    }
    __syncthreads();

    // compute propensities
    k = 0;
    while ((idx = k * blockDim.x + thread_id) < n_active) {
      doub_wsp[idx] = rate_caches[idx] * (
          (idx == 0) + (idx != 0) * (x_shared[idx - 1] - x_shared[idx] > num_excl)
      );
      k++;
    }
    __syncthreads();

    // transform the propensities array to its cumsum array
    if (thread_id == 0) {
      thrust::inclusive_scan(thrust::seq, doub_wsp, doub_wsp + n_active, doub_wsp);
    }
    __syncthreads();

    // determine stepsize
    if (thread_id == 0) {
      stepsize = log(1.0/rn[i_random]) / doub_wsp[n_active - 1];
    }

    // choose the ribosome to move
    k = 0;
    while ((idx = k * blockDim.x + thread_id) < n_active) {
      x_wsp[idx] = (doub_wsp[idx] >= rn[i_random + 1] * doub_wsp[n_active - 1]);
      k++;
    }
    __syncthreads();

    if (thread_id == 0) idx = thrust::find(thrust::seq, x_wsp, x_wsp + n_active, 1) - x_wsp;
    __syncthreads();

    if (thread_id == 0) {
      if (t_now + stepsize <= t_final) {
        t_now += stepsize;

        x_shared[idx] = (x_shared[idx] + 1) % (gene_len + 1);
        if ((idx == n_active - 1) & (n_active < num_rib_max)) {
          n_active++;
          rate_caches[n_active - 1] = rates[0];
          c_caches[n_active - 1] = probe_design[0];
        } else if (x_shared[idx] == 0) {
          to_shift = ( int ) 1;
        }

        // update rates and probe design coefficients
        rate_caches[idx] = rates[x_shared[idx]];
        c_caches[idx] = probe_design[x_shared[idx]];
      } else {
        t_now = t_final;
      }
    }
    __syncthreads();

    // check if we need to shift ribosomes locations so that the first nonzero is at the beginning
    if (to_shift > 0) {
      _blockwise_shift_arrays(1, n_active, x_shared, x_wsp);
      _blockwise_shift_arrays(1, n_active, c_caches, x_wsp);
      _blockwise_shift_arrays(1, n_active, rate_caches, doub_wsp);
      n_active = max(1, n_active-1);
      if (thread_id == 0) {
        to_shift = 0;
      }
    }
    __syncthreads();
    // Pop out the used random numbers and check that we have enough random numbers in store
    if (thread_id == 0) i_random += 2;
    __syncthreads();
    if (i_random >= blockDim.x){
      rn[thread_id] = curand_uniform_double(&rstate_loc);
      __syncthreads();
      if (thread_id == 0) i_random = 0;
    }
    __syncthreads();
  }

  // Compute current intensity
  k = 0;
  while ((idx = k * blockDim.x + thread_id) < n_active) {
    x_wsp[idx] = c_caches[idx];
    k++;
  }
  __syncthreads();
  if (thread_id == 0) thrust::inclusive_scan(thrust::seq, x_wsp, x_wsp + n_active, x_wsp);
  __syncthreads();
  if (thread_id == 0) current_intensity = x_wsp + n_active - 1;
  // copy current intensity to appropriate locations in global memory
  if (thread_id == 0 && (*idx_to_output < num_times)) {
    while (t_array[*idx_to_output] <= t_now) {
      intensity[sample_id * num_times + (*idx_to_output)] = (*current_intensity);
      (*idx_to_output)++;
      if (*idx_to_output >= num_times) {
        break;
      }
    }
  }
  __syncthreads();

// COPY FINAL RIBOSOMES LOCATIONS TO GLOBAL MEMORY
  k = 0;
  while ((idx = k * blockDim.x + thread_id) < num_rib_max) {
    X[sample_id * num_rib_max + idx] = x_shared[idx];
    k++;
  }
// Update random states
  rstates[sample_id*blockDim.x + thread_id] = rstate_loc;
  __syncthreads();
}
}