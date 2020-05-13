//
// Created by huy on 5/10/20.
//

#include "CuTransKernels.h"

namespace ssit {

__global__
void init_rand_states(curandState_t *rstates, int seed = 0) {
  curand_init(seed, blockIdx.x , 0, &rstates[blockIdx.x]);
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

  __shared__ curandState_t rstate_loc;
  __shared__ int n_active;
  __shared__ double t_now, t_final, rn, stepsize;
  __shared__ int current_intensity;
  __shared__ int to_shift, to_move;
  __shared__ int idx_to_output;
  extern __shared__ double dyn_shared_mem[];
  // Partition the shared memory into appropriate arrays
  double *doub_wsp = dyn_shared_mem;
  int *x_shared = ( int * ) (doub_wsp + num_rib_max);
  int *x_wsp = x_shared + num_rib_max;

  // INITIALIZATION
  if (thread_id == 0) {
    t_now = 0.0;
    t_final = t_array[num_times - 1];
    to_shift = 0;
    idx_to_output = 0;
    rstate_loc = rstates[sample_id];
  }

  // Copy initial ribosome locations and rates to shared memory
  uint idx;
  uint k{0};
  while ((idx = k * blockDim.x + thread_id) < num_rib_max) {
    x_shared[idx] = X[sample_id * num_rib_max + idx];
    // Polling all threads for number of active ribosomes (i.e., those not in the 0 state)
    atomicAdd(&n_active, int(x_shared[idx] != 0));
    k++;
  }
  __syncthreads();
  if (thread_id == 0) {
    n_active++;
  }
  __syncthreads();
  // STEPPING
  while (t_now < t_final) {
    // Compute current intensity
    if (thread_id == 0) current_intensity = 0;
    __syncthreads();
    k = 0;
    while ((idx = k * blockDim.x + thread_id) < n_active) {
      atomicAdd(&current_intensity, probe_design[x_shared[idx]]);
      k++;
    }
    __syncthreads();
    // copy current intensity to appropriate locations in global memory
    if (thread_id == 0) {
      while (t_array[idx_to_output] <= t_now) {
        intensity[sample_id * num_times + (idx_to_output)] = current_intensity;
        idx_to_output++;
        if (idx_to_output >= num_times) {
          break;
        }
      }
    }
    __syncthreads();
    // compute propensities
    k = 0;
    while ((idx = k * blockDim.x + thread_id) < n_active) {
      doub_wsp[idx] = rates[x_shared[idx]] * (
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
      rn = curand_uniform(&rstate_loc);
      stepsize = log(1.0 / rn) / doub_wsp[n_active - 1];
    }
    __syncthreads();

    if (t_now + stepsize <= t_final) {
      if (thread_id == 0) {
        rn = curand_uniform(&rstate_loc);
        to_move = 0;
      }
      __syncthreads();
      // choose the ribosome to move
      k = 0;
      while ((idx = k * blockDim.x + thread_id) < n_active) {
        atomicAdd(&to_move, int(doub_wsp[idx] >= rn * doub_wsp[n_active - 1]));
        k++;
      }
      // to_move is now the number of ribsome i such that cumsum(propensities)[0:i] >= rn*sum(propensities)
      __syncthreads();

      if (thread_id == 0) to_move = n_active - to_move;
      __syncthreads();

      if (thread_id == 0) {
          t_now += stepsize;
          x_shared[to_move]++;
          if (x_shared[to_move] > gene_len) x_shared[to_move] = 0;
          if ((to_move == n_active - 1) & (n_active < num_rib_max)) {
            n_active++;
          } else if (x_shared[to_move] == 0) {
            to_shift = ( int ) 1;
          }
      }

      // check if we need to shift ribosomes locations so that the first nonzero is at the beginning
      if (to_shift > 0) {
        _blockwise_shift_arrays(1, n_active, x_shared, x_wsp);
        n_active = max(1, n_active - 1);
        if (thread_id == 0) {
          to_shift = 0;
        }
      }
      __syncthreads();
    }
    else{
      break;
    }
  }

  // Compute current intensity
  if (thread_id == 0) current_intensity = 0;
  __syncthreads();
  k = 0;
  while ((idx = k * blockDim.x + thread_id) < n_active) {
    atomicAdd(&current_intensity, probe_design[x_shared[idx]]);
    k++;
  }
  __syncthreads();
  // copy current intensity to appropriate locations in global memory
  if (thread_id == 0) {
    while (t_array[idx_to_output] <= t_final) {
      intensity[sample_id * num_times + (idx_to_output)] = current_intensity;
      idx_to_output++;
      if (idx_to_output >= num_times) {
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
  if (thread_id == 0) {
    rstates[sample_id] = rstate_loc;
  }
  __syncthreads();
}

}