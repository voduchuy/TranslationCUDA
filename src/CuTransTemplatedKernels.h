//
// Created by huy on 5/11/20.
//

#ifndef TRANSLATIONCUDA_SRC_CUTRANSTEMPLATEDKERNELS_H_
#define TRANSLATIONCUDA_SRC_CUTRANSTEMPLATEDKERNELS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <curand_kernel.h>
#include "CuTransKernels.h"

namespace ssit{
/**
 * @brief Replicate an array stored in device memory.
 * @tparam T typename (e.g., double, int...)
 * @param n length of the source array.
 * @param x_src pointer to the source array, i.e. x_src[0] is the first element of the source.
 * @param x_dest pointer to the destination array collection. The execution block j will copy x_src into the j-th block of x_dest.
 */
template<typename T>
__global__
void _replicate_array(int n, T* x_src, T* x_dest){
  const uint &thread_id = threadIdx.x;
  const uint &block_id = blockIdx.x;

  uint n1 = n / blockDim.x;
  for (int k{0}; k < n1; ++k){
    x_dest[n*block_id + k*blockDim.x + thread_id] = x_src[k*blockDim.x + thread_id];
  }
  if (thread_id < n % blockDim.x){
    x_dest[n*block_id + n1*blockDim.x + thread_id] = x_src[n1*blockDim.x + thread_id];
  }
}

template __global__ void _replicate_array<int>(int n, int* x_src, int* x_dest);
}
#endif //TRANSLATIONCUDA_SRC_CUTRANSTEMPLATEDKERNELS_H_
