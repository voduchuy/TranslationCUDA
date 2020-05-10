//
// Created by huy on 12/14/19.
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

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

__device__
int draw_uniforms(curandState_t *rstate, double *rn) {
  if (threadIdx.x == 0) {
    rn[0] = curand_uniform_double(rstate);
    rn[1] = curand_uniform_double(rstate);
  }
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

__global__
void update_state(double t_final, int num_excl, int gene_len, int num_rib, int *X, curandState_t *rstates) {
  const uint &thread_id = threadIdx.x;
  const uint &sample_id = blockIdx.x;

  extern __shared__ double wsp[];
  // Partition the shared memory into appropriate arrays
  double *rn = wsp;
  double *t_now_ptr = wsp + 2;
  double *stepsize_ptr = t_now_ptr + 1;
  double *propensities = stepsize_ptr + 1;
  int *x_shared = ( int * ) (propensities + num_rib);

  double &t_now = *t_now_ptr;
  double &stepsize = *stepsize_ptr;

  // INITIALIZARION
  // Init time
  if (thread_id == 0) {
    *t_now_ptr = 0.0;
  }

  // Copy initial ribosome locations to shared memory
  uint n_per_thread = num_rib / blockDim.x;
  uint idx_start = sample_id * num_rib + thread_id * n_per_thread;
  for (int i{0}; i < n_per_thread; ++i) {
    x_shared[i] = X[idx_start + i];
  }
  if (thread_id < num_rib % blockDim.x) {
    x_shared[blockDim.x * n_per_thread + thread_id] = X[sample_id * num_rib + blockDim.x * n_per_thread + thread_id];
  }

  int idx;
  // STEPPING
  while (t_now < t_final) {
    // compute propensities
    for (int i{0}; i < n_per_thread; ++i) {
      propensities[i] = 1.0*(x_shared[(i-1)%num_rib] - x_shared[i] > num_excl);//(x_shared[i]==0) + x_shared[i];
    }
    if (thread_id < num_rib % blockDim.x) {
      idx = blockDim.x*n_per_thread + thread_id;
      propensities[idx] = 1.0*(x_shared[(idx-1)%num_rib] - x_shared[idx] > num_excl);//(x_shared[blockDim.x*n_per_thread+thread_id]==0) +
          //x_shared[blockDim.x * n_per_thread + thread_id];
    }

    // transform the propensities array to its cumsum array
    if (thread_id == 0){
      for (int i{1}; i < num_rib; ++i){
        propensities[i] += propensities[i-1];
      }
    }

    // determine stepsize and the next ribosome to move
    if (thread_id == 0) {
      draw_uniforms(rstates + sample_id, rn);
      stepsize = -1.0*log(rn[0])/propensities[num_rib-1];

      double tmp = 0.0;
      for (int i{0}; i < num_rib;++i){
        tmp += propensities[i];
        if (tmp > rn[1]*propensities[num_rib-1]){
          x_shared[i] = (x_shared[i]+1) % (gene_len+1);
          break;
        }
      }
      t_now += stepsize;
    }
  }

  // COPY FINAL RIBOSOMES LOCATIONS TO GLOBAL MEMORY
  for (int i{0}; i < n_per_thread; ++i) {
    X[idx_start + i] = x_shared[i];
  }
  if (thread_id < num_rib % blockDim.x) {
    X[sample_id * num_rib + blockDim.x * n_per_thread + thread_id] = x_shared[blockDim.x * n_per_thread + thread_id] ;
  }
}

int main(int argc, char **argv) {
  const int num_rib_max = 4;
  const int num_times = 100;
  const int num_samples = 10;
  const int n_excl = 3;
  const double t_final = 10.0;

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
  ;
  update_state<<<num_samples, 32, shared_mem_size>>>(t_final, n_excl, 100, num_rib_max, X, rand_states);
  CUDACHKERR();

  int X_host[num_samples * num_rib_max];
  cudaMemcpy(( void * ) X_host, ( void * ) X, num_samples * num_rib_max * sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();
  for (int i{0}; i < num_samples * num_rib_max; ++i) {
    std::cout << X_host[i] << "\n";
  }

  cudaFree(X);
  CUDACHKERR();
  cudaFree(rand_states);
  CUDACHKERR();
  return 0;
}