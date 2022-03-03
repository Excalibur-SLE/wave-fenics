#include "mass_kernel.hpp"
#include <cstdint>

template <typename T>
static __global__ void _mass_apply(const T* xe, const T* phi, const T* detJ, T* ye,
                                   int ndofs) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T _xq[512];
  __shared__ T _phi[4096];

  for (int i = threadIdx.x; i < 4096; i += blockDim.x)
    _phi[i] = phi[i];
  __syncthreads();

  int dof_id = threadIdx.x % ndofs;
  int element_id = threadIdx.x / ndofs;
  const T* _x = xe + blockIdx.x * blockDim.x + element_id * ndofs;

  T result = 0;
  for (int i = 0; i < ndofs; i++) {
    result += _x[i] * _phi[dof_id * ndofs + i];
  }

  _xq[threadIdx.x] = result * detJ[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();

  result = 0;
  for (int i = 0; i < ndofs; i++) {
    result += _xq[element_id + i] * _phi[dof_id * ndofs + i];
  }
  ye[gid] = result;
}

template <typename T>
void mass_apply(int Ne, const T* xe, const T* phi, const T* detJ, T* ye, int ndofs,
                int block_size) {
  const int num_blocks = (Ne + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _mass_apply<<<dimGrid, dimBlock>>>(xe, phi, detJ, ye, ndofs);
  cudaDeviceSynchronize();
}

//-----------------------------------------------------------------------------
template void mass_apply<double>(int Ne, const double* xe, const double* phi,
                                 const double* detJ, double* ye, int ndofs,
                                 int block_size);
template void mass_apply<float>(int Ne, const float* xe, const float* phi,
                                const float* detJ, float* ye, int ndofs, int block_size);
//-----------------------------------------------------------------------------