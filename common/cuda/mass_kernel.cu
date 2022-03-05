#include "mass_kernel.hpp"
#include <cstdint>

#define NDOFS 8
#define NQUADS 8
#define NUM_ELEMENTS 64

template <typename T>
static __global__ void _mass_apply(const T* xe, const T* phi, const T* detJ, T* ye) {
  int element = blockIdx.y * blockDim.y + threadIdx.y;

  // Allocate shared memory
  __shared__ T _xe[NUM_ELEMENTS][NDOFS];
  __shared__ T _xq[NUM_ELEMENTS][NQUADS];

  __shared__ T _phiT[NQUADS][NQUADS];
  __shared__ T _phi[NQUADS][NQUADS];

  // Prepare basis functions (load to shared memory)
  // Load Phi^T to shared memory
  for (int i = threadIdx.y; i < NQUADS; i += blockDim.y){
    for (int j = threadIdx.x; j < NDOFS; j += blockDim.x){
      _phi[i][j] = phi[i * NDOFS + j];
      _phiT[j][i] = phi[i * NDOFS + j];
    }
  }

  if (threadIdx.x < NDOFS)
    _xe[threadIdx.y][blockIdx.x] = xe[element * NDOFS + threadIdx.x];

  __syncthreads();

  // Evaluate coefficients at quadrature points
  for (int i = threadIdx.x; i < NQUADS; i += blockDim.x){
    T wq = 0;
    for (int j = 0; j < NDOFS; j++)
      wq += _xe[threadIdx.y][j] * _phiT[j][threadIdx.x];
    _xq[threadIdx.y][i] = detJ[element * NQUADS + threadIdx.x] * wq;
  }

  __syncthreads();

    T yi = 0;
    for (int i = 0; i < NQUADS; i++) {
      yi += _xq[threadIdx.y][i] * _phi[i][threadIdx.x];
    ye[element * NDOFS + threadIdx.x] = yi;
  }
}

template <typename T>
void mass_apply(int Ne, const T* xe, const T* phi, const T* detJ, T* ye) {
  int block_size = 512;
  const int num_blocks = (Ne + block_size - 1) / block_size;
  dim3 dimBlock(NDOFS, NUM_ELEMENTS);
  dim3 dimGrid(1, num_blocks);
  _mass_apply<<<dimGrid, dimBlock>>>(xe, phi, detJ, ye);
  cudaDeviceSynchronize();
}

//-----------------------------------------------------------------------------
template void mass_apply<double>(int Ne, const double* xe, const double* phi,
                                 const double* detJ, double* ye);
template void mass_apply<float>(int Ne, const float* xe, const float* phi,
                                const float* detJ, float* ye);
//-----------------------------------------------------------------------------