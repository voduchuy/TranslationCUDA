//
// Created by huy on 5/10/20.
//

#include "CuTransKernels.h"

namespace ssit {

__global__
void init_rand_states(curandState_t *rstates, int seed=0) {
  curand_init(seed, blockIdx.x, 0, &rstates[blockIdx.x]);
}

__global__
void initialize_ribosome_locations(const int num_rib, int *X) {
  const uint &thread_id = threadIdx.x;
  const uint &sample_id = blockIdx.x;

  uint ncodon_loc = num_rib / blockDim.x;
  uint idx_start = sample_id * num_rib + thread_id * ncodon_loc;
  for (int i{0}; i < ncodon_loc; ++i) {
    X[idx_start + i] = 0;
  }

  if (thread_id < num_rib % blockDim.x) {
    X[sample_id * num_rib + blockDim.x * ncodon_loc + thread_id] = 0;
  }
}

__device__
void draw_uniforms(curandState_t *rstate, double *rn) {
  rn[0] = curand_uniform_double(rstate);
  rn[1] = curand_uniform_double(rstate);
}

__device__
void shift_arrays(const int to_shift, const uint n, int *x_shared, int *x_shared_copy) {
  // copy x[to_shift:] to x_copy
  uint n_passes = (n - to_shift) / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    x_shared_copy[k * blockDim.x + threadIdx.x] = x_shared[to_shift + k * blockDim.x + threadIdx.x];
  }
  if (threadIdx.x < (n - to_shift) % blockDim.x) {
    x_shared_copy[n_passes * blockDim.x + threadIdx.x] = x_shared[to_shift + n_passes * blockDim.x + threadIdx.x];
  }
  __syncthreads();
  // make everything in x zero
  n_passes = n / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    x_shared[k * blockDim.x + threadIdx.x] = 0;
  }
  if (threadIdx.x < n % blockDim.x) {
    x_shared[n_passes * blockDim.x + threadIdx.x] = 0;
  }
  __syncthreads();
  // copy back from x_copy to x
  n_passes = (n - to_shift) / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    x_shared[k * blockDim.x + threadIdx.x] = x_shared_copy[k * blockDim.x + threadIdx.x];
  }
  if (threadIdx.x < (n - to_shift) % blockDim.x) {
    x_shared[n_passes * blockDim.x + threadIdx.x] = x_shared_copy[n_passes * blockDim.x + threadIdx.x];
  }
  __syncthreads();
}

__global__
void update_state(const int num_times,
                  const double *t_array,
                  int num_excl,
                  int gene_len,
                  int num_rib,
                  int *X,
                  curandState_t *rstates,
                  const double *rates,
                  const int *probe_design,
                  int *intensity) {
  const uint &thread_id = threadIdx.x;
  const uint &sample_id = blockIdx.x;

  extern __shared__ double wsp[];
  // Partition the shared memory into appropriate arrays
  double *rn = wsp;
  double *t_now_ptr = wsp + 2;
  double *stepsize_ptr = t_now_ptr + 1;
  double *propensities = stepsize_ptr + 1;
  int *x_shared = ( int * ) (propensities + num_rib);
  int *x_wsp = x_shared + num_rib;
  int *to_shift_ptr = x_wsp + num_rib;
  int *idx_to_output = to_shift_ptr + 1;

  int *current_intensity;
  double &t_now = *t_now_ptr;
  double &stepsize = *stepsize_ptr;
  int &to_shift = *to_shift_ptr;
  const double &t_final = t_array[num_times - 1];

  // INITIALIZATION
  // Init time
  if (thread_id == 0) {
    t_now = 0.0;
    to_shift = 0;
    *idx_to_output = 0;
  }

  // Copy initial ribosome locations to shared memory
  uint idx;
  uint n_passes = num_rib / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    x_shared[k * blockDim.x + thread_id] = X[sample_id * num_rib + k * blockDim.x + thread_id];
  }
  if (thread_id < num_rib % blockDim.x) {
    x_shared[n_passes * blockDim.x + thread_id] = X[sample_id * num_rib + n_passes * blockDim.x + thread_id];
  }
  __syncthreads();

  // STEPPING
  while (t_now < t_final) {
    // Compute current intensity
    n_passes = num_rib / blockDim.x;
    for (uint k{0}; k < n_passes; ++k) {
      idx = k * blockDim.x + thread_id;
      x_wsp[idx] = probe_design[x_shared[idx]];
    }
    if (thread_id < num_rib % blockDim.x) {
      idx = num_rib - num_rib % blockDim.x + thread_id;
      x_wsp[idx] = probe_design[x_shared[idx]];
    }
    __syncthreads();
    if (thread_id == 0) thrust::inclusive_scan(thrust::seq, x_wsp, x_wsp + num_rib, x_wsp);
    __syncthreads();
    if (thread_id == 0) current_intensity = x_wsp + num_rib - 1;
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
    n_passes = num_rib / blockDim.x;
    for (uint k{0}; k < n_passes; ++k) {
      idx = k * blockDim.x + thread_id;
      propensities[idx] = rates[x_shared[idx]] * (
          (idx == 0) + (idx != 0) * (x_shared[idx - 1] - x_shared[idx] > num_excl)
      );
    }
    if (thread_id < num_rib % blockDim.x) {
      idx = n_passes * blockDim.x + thread_id;
      propensities[idx] = rates[x_shared[idx]] * (
          (idx == 0) + (idx != 0) * (x_shared[idx - 1] - x_shared[idx] > num_excl)
      );
    }
    __syncthreads();

    // transform the propensities array to its cumsum array
    if (thread_id == 0) {
      thrust::inclusive_scan(thrust::device, propensities, propensities + num_rib, propensities);
    }
    __syncthreads();

    // determine stepsize
    if (thread_id == 0) {
      draw_uniforms(rstates + sample_id, rn);
      stepsize = -1.0 * log(rn[0]) / propensities[num_rib - 1];
    }
    __syncthreads();

    // choose the ribosome to move
    n_passes = num_rib / blockDim.x;
    for (uint k{0}; k < n_passes; ++k) {
      idx = k * blockDim.x + thread_id;
      x_wsp[idx] = (propensities[idx] >= rn[1] * propensities[num_rib - 1]);
    }
    if (thread_id < num_rib % blockDim.x) {
      idx = n_passes * blockDim.x + thread_id;
      x_wsp[idx] = (propensities[idx] >= rn[1] * propensities[num_rib - 1]);
    }
    __syncthreads();
    if (thread_id == 0) idx = thrust::find(thrust::device, x_wsp, x_wsp + num_rib, 1) - x_wsp;
    __syncthreads();
    if (thread_id == 0) {
      if (t_now + stepsize <= t_final) {
        t_now += stepsize;
        x_shared[idx] = (x_shared[idx] + 1) % (gene_len + 1);
        if (x_shared[idx] == 0) {
          to_shift = ( int ) idx + 1;
        }
      } else {
        t_now = t_final;
      }
    }
    __syncthreads();
    // check if we need to shift ribosomes locations so that the first nonzero is at the beginning
    if (to_shift > 0) {
      shift_arrays(to_shift, num_rib, x_shared, x_wsp);

      __syncthreads();
      if (thread_id == 0) {
        to_shift = 0;
      }
    }
    __syncthreads();
  }
  // Compute current intensity
  n_passes = num_rib / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    idx = k * blockDim.x + thread_id;
    x_wsp[idx] = probe_design[x_shared[idx]];
  }
  if (thread_id < num_rib % blockDim.x) {
    idx = num_rib - num_rib % blockDim.x + thread_id;
    x_wsp[idx] = probe_design[x_shared[idx]];
  }
  __syncthreads();
  if (thread_id == 0) thrust::inclusive_scan(thrust::device, x_wsp, x_wsp + num_rib, x_wsp);
  __syncthreads();
  if (thread_id == 0) current_intensity = x_wsp + num_rib - 1;
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
  n_passes = num_rib / blockDim.x;
  for (int k{0}; k < n_passes; ++k) {
    idx = thread_id + k * blockDim.x;
    X[sample_id * num_rib + idx] = x_shared[idx];
  }
  if (thread_id < num_rib % blockDim.x) {
    X[
        sample_id * num_rib
            + blockDim.
                x * n_passes
            + thread_id] = x_shared[blockDim.
        x * n_passes
        + thread_id];
  }
}
}