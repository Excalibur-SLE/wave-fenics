#include "mass_kernel.hpp"
#include <cstdint>

#define NDOFS 27

template <typename T>
static __global__ void _mass_apply(std::int32_t num_elements, const T* xe, const T* phi,
                                   const T* detJ, T* ye) {
  int id = blockIdx.x * NDOFS + threadIdx.x;
  T _phi[NDOFS];
  __shared__ T _xe[32];
  __shared__ T _xq[32];

  if (threadIdx.x < NDOFS) {
    _xe[threadIdx.x] = xe[id];

    // Prepare basis functions (load to shared memory)
    // Load Phi^T to shared memory
    for (int j = 0; j < NDOFS; j++)
      _phi[j] = phi[threadIdx.x * NDOFS + j];
  }

  __syncthreads();

  // Evaluate coefficients at quadrature points
  if (threadIdx.x < NDOFS) {
    T wq = 0.;
    for (int j = 0; j < NDOFS; j++)
      wq += _xe[j] * _phi[j];

    _xq[threadIdx.x] = detJ[id] * wq;

    // Prepare basis functions (load to shared memory)
    // Load Phi^T to shared memory
    for (int j = 0; j < NDOFS; j++)
      _phi[j] = phi[j * NDOFS + threadIdx.x];

    __syncthreads();

    T yi = 0;
    for (int iq = 0; iq < NDOFS; iq++)
      yi += _xq[iq] * _phi[iq];

    ye[id] = yi;
  }
}

template <typename T>
void mass_apply(int num_elements, const T* xe, const T* phi, const T* detJ, T* ye) {
  int block_size = 32;
  const int num_blocks = num_elements;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _mass_apply<<<dimGrid, dimBlock>>>(num_elements, xe, phi, detJ, ye);
}

//-----------------------------------------------------------------------------
template void mass_apply<double>(int Ne, const double* xe, const double* phi,
                                 const double* detJ, double* ye);
template void mass_apply<float>(int Ne, const float* xe, const float* phi,
                                const float* detJ, float* ye);
//-----------------------------------------------------------------------------