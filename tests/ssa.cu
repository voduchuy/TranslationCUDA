//
// Created by huy on 12/14/19.
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <curand_kernel.h>

#define CUDACHKERR() { \
cudaError_t ierr = cudaGetLastError();\
if (ierr != cudaSuccess){ \
    printf("%s in %s at line %d\n", cudaGetErrorString(ierr), __FILE__, __LINE__);\
    exit(EXIT_FAILURE); \
}\
}\


__global__
void init_rand_states(curandState_t *rstates) {
  curand_init(0, blockIdx.x, 0, &rstates[blockIdx.x]);
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
int draw_uniforms(curandState_t *rstate, double *rn) {
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
  // make everything in x zero
  n_passes = n / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    x_shared[k * blockDim.x + threadIdx.x] = 0;
  }
  if (threadIdx.x < n % blockDim.x) {
    x_shared[n_passes * blockDim.x + threadIdx.x] = 0;
  }
  // copy back from x_copy to x
  n_passes = (n - to_shift) / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    x_shared[k * blockDim.x + threadIdx.x] = x_shared_copy[k * blockDim.x + threadIdx.x];
  }
  if (threadIdx.x < (n - to_shift) % blockDim.x) {
    x_shared[n_passes * blockDim.x + threadIdx.x] = x_shared_copy[n_passes * blockDim.x + threadIdx.x];
  }
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

  // INITIALIZARION
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
    thrust::inclusive_scan(thrust::device, x_wsp, x_wsp + num_rib, x_wsp);
    current_intensity = x_wsp + num_rib - 1;
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
    // compute propensities
    n_passes = num_rib / blockDim.x;
    for (uint k{0}; k < n_passes; ++k) {
      idx = k * blockDim.x + thread_id;
      propensities[idx] = rates[x_shared[idx]] * (
          (idx == 0)
              +
                  (idx != 0)
                      *
                          (x_shared[idx - 1] - x_shared[idx] > num_excl)
      );
    }
    if (thread_id < num_rib % blockDim.x) {
      idx = num_rib - num_rib % blockDim.x + thread_id;
      propensities[idx] = rates[x_shared[idx]] *
          (
              (idx == 0)
                  +
                      (idx != 0)
                          *
                              (x_shared[idx - 1] - x_shared[idx] > num_excl)
          );

    }

    // transform the propensities array to its cumsum array
    thrust::inclusive_scan(thrust::device, propensities, propensities + num_rib, propensities);

    // determine stepsize and the next ribosome to move
    if (thread_id == 0) {
      draw_uniforms(rstates + sample_id, rn);
      stepsize = -1.0 * log(rn[0]) / propensities[num_rib - 1];
    }
    t_now += stepsize;
    // update state
    if (thread_id == 0) {
      if (t_now + stepsize <= t_final) {
        for (int i{0}; i < num_rib; ++i) {
          if (propensities[i] >= rn[1] * propensities[num_rib - 1]) {
            x_shared[i] = (x_shared[i] + 1) % (gene_len + 1);
            if (x_shared[i] == 0) {
              to_shift = i + 1;
            }
            break;
          }
        }
      }
    }
    // check if we need to shift ribosomes locations so that the first nonzero is at the beginning
    if (to_shift > 0) {
      shift_arrays(to_shift, num_rib, x_shared, x_wsp);
      if (thread_id == 0) {
        to_shift = 0;
      }
    }
  }
  // Compute intensity at final time
  n_passes = num_rib / blockDim.x;
  for (uint k{0}; k < n_passes; ++k) {
    idx = k * blockDim.x + thread_id;
    x_wsp[idx] = probe_design[x_shared[idx]];
  }
  if (thread_id < num_rib % blockDim.x) {
    idx = num_rib - num_rib % blockDim.x + thread_id;
    x_wsp[idx] = probe_design[x_shared[idx]];
  }
  thrust::inclusive_scan(thrust::device, x_wsp, x_wsp + num_rib, x_wsp);
  current_intensity = x_wsp + num_rib - 1;
  // copy current intensity to appropriate locations in global memory
  if (thread_id == 0) {
    intensity[sample_id * num_times + num_times - 1] = (*current_intensity);
  }

// COPY FINAL RIBOSOMES LOCATIONS TO GLOBAL MEMORY
  n_passes = num_rib / blockDim.x;
  for (
      int k{0};
      k < n_passes;
      ++k) {
    idx = thread_id + k * blockDim.x;
    X[
        sample_id * num_rib
            + idx] = x_shared[idx];
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

int main(int argc, char **argv) {
  const int num_rib_max = 64;
  const int num_samples = 10;
  const int n_excl = 3;
  const int gene_len = 10;

  const int num_times = 5;
  const double t_array[] = {0.0, 10.0, 20.0, 50.0, 1000.0};

  thrust::device_vector<double> rates(gene_len + 1, 1.0);
  thrust::device_vector<int> probe_design(gene_len + 1);
  thrust::device_vector<int> intensity(num_times * num_samples);

  thrust::fill(probe_design.begin() + 1, probe_design.end(), 1);
  thrust::fill(intensity.begin(), intensity.end(), 0);

  thrust::device_vector<double> t_array_device(5);
  thrust::copy(&t_array[0], &t_array[5], t_array_device.begin());

  curandState_t *rand_states;
  cudaMalloc(( void ** ) &rand_states, num_samples * sizeof(curandState_t));
  CUDACHKERR();

  init_rand_states<<<num_samples, 1>>>(rand_states);
  CUDACHKERR();

  int *X;
  cudaMalloc(( void ** ) &X, num_samples * num_rib_max * sizeof(int));
  CUDACHKERR();
  initialize_ribosome_locations<<<num_samples, 32, 0>>>(num_rib_max, X);
  CUDACHKERR();
  size_t shared_mem_size = 2 * sizeof(double) // for the two uniform random numbers
      + 2 * sizeof(double) // for time and stepsize
      + num_rib_max * sizeof(double) // for propensities
      + num_rib_max * sizeof(int) // for ribosome locations
      + num_rib_max * sizeof(int) // temporary space to copy ribosome locations (when shifting)
      + sizeof(int) // amount to shift
      + sizeof(int) // current intensity
      + sizeof(int) // time array index to output the intesnity to
  ;
  update_state<<<num_samples, 32, shared_mem_size>>>(num_times,
                                                     thrust::raw_pointer_cast(&t_array_device[0]),
                                                     n_excl,
                                                     gene_len,
                                                     num_rib_max,
                                                     X,
                                                     rand_states,
                                                     thrust::raw_pointer_cast(&rates[0]),
                                                     thrust::raw_pointer_cast(&probe_design[0]),
                                                     thrust::raw_pointer_cast(&intensity[0]));
  CUDACHKERR();

  int X_host[num_samples][num_rib_max];
  cudaMemcpy(( void * ) X_host, ( void * ) X, num_samples * num_rib_max * sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();

  thrust::host_vector<int> intensity_host(num_samples * num_times);
  thrust::copy(intensity.begin(), intensity.end(), intensity_host.begin());

  for (int i{0}; i < num_samples; ++i) {
    for (int j{0}; j < num_rib_max; ++j) {
      std::cout << X_host[i][j] << " ";
    }
    std::cout << "\n";
    for (int j{0}; j < num_times; ++j) {
      std::cout << intensity_host[i * num_times + j] << " ";
    }
    std::cout << "\n";
  }

  cudaFree(X);
  CUDACHKERR();
  cudaFree(rand_states);
  CUDACHKERR();
  return 0;
}