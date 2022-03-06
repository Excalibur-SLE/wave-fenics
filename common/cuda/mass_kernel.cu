#include "mass_kernel.hpp"
#include <cstdint>

#define NDOFS 8
#define NQUADS 8
#define NUM_ELEMENTS 64

template <typename T>
static __global__ void _mass_apply(std::int32_t num_elements, const T* xe, const T* phi,
                                   const T* detJ, T* ye) {
  int element = blockIdx.y * blockDim.y + threadIdx.y;
  int dof_id = element * blockDim.x + threadIdx.x;

  // Allocate shared memory
  __shared__ T _xe[NUM_ELEMENTS][NDOFS];
  __shared__ T _xq[NUM_ELEMENTS][NQUADS];
  __shared__ T _phi[NQUADS][NQUADS];

  if (element < num_elements) {
    // Prepare basis functions (load to shared memory)
    // Load Phi^T to shared memory
    for (int i = threadIdx.y; i < NQUADS; i += blockDim.y)
      for (int j = threadIdx.x; j < NDOFS; j += blockDim.x)
        _phi[j][i] = phi[i * NDOFS + j];

    _xe[threadIdx.y][threadIdx.x] = xe[dof_id];
    __syncthreads();

    // Evaluate coefficients at quadrature points
    for (int q = threadIdx.x; q < NQUADS; q += blockDim.x) {
      T wq = 0;
      for (int j = 0; j < NDOFS; j++)
        wq += _xe[threadIdx.y][j] * _phi[j][q];

      _xq[threadIdx.y][q] = detJ[element * NQUADS + q] * wq;
    }

    __syncthreads();
    
    // Prepare basis functions (load to shared memory)
    // Load Phi^T to shared memory
    for (int i = threadIdx.y; i < NQUADS; i += blockDim.y)
      for (int j = threadIdx.x; j < NDOFS; j += blockDim.x)
        _phi[i][j] = phi[i * NDOFS + j];

    __syncthreads();

    T yi = 0;
    for (int iq = 0; iq < NQUADS; iq++) {
      yi += _xq[threadIdx.y][iq] * _phi[iq][threadIdx.x];
    }

    ye[dof_id] = yi;
  }
}

template <typename T>
void mass_apply(int Ne, const T* xe, const T* phi, const T* detJ, T* ye) {
  int block_size = NDOFS * NUM_ELEMENTS;
  const int num_blocks = (Ne + block_size - 1) / block_size;
  dim3 dimBlock(NDOFS, NUM_ELEMENTS);
  dim3 dimGrid(1, num_blocks);
  _mass_apply<<<dimGrid, dimBlock>>>(Ne / NDOFS, xe, phi, detJ, ye);
  cudaDeviceSynchronize();
}

//-----------------------------------------------------------------------------
template void mass_apply<double>(int Ne, const double* xe, const double* phi,
                                 const double* detJ, double* ye);
template void mass_apply<float>(int Ne, const float* xe, const float* phi,
                                const float* detJ, float* ye);
//-----------------------------------------------------------------------------