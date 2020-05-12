//
// Created by huy on 5/10/20.
//

#ifndef TRANSLATIONCUDA_SRC_CUTRANSKERNELS_H_
#define TRANSLATIONCUDA_SRC_CUTRANSKERNELS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <curand_kernel.h>
#include "CuTransTemplatedKernels.h"
#include "cub.cuh"

namespace ssit{
__global__
void init_rand_states(curandState_t *rstates, int seed);

__device__
void draw_two_uniforms(curandState_t *rstate, double *rn);

__device__
void shift_arrays(const int to_shift, const uint n, int *x_shared, int *x_shared_copy);

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
                  int *intensity);
}
#endif //TRANSLATIONCUDA_SRC_CUTRANSKERNELS_H_
