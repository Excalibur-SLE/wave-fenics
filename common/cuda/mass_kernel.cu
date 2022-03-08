#include "mass_kernel.hpp"
#include <cstdint>

#define NDOFS 27

template <typename T>
static __global__ void _mass_apply(std::int32_t num_elements, const T* xe, const T* phi,
                                   const T* detJ, T* ye) {

  __shared__ T _phi[NDOFS][NDOFS];
  __shared__ T _xe[NDOFS];
  __shared__ T _xq[NDOFS];

  if (threadIdx.x < NDOFS) {
#pragma unroll
    for (int j = 0; j < NDOFS; j++) {
      _phi[j][threadIdx.x] = phi[threadIdx.x * NDOFS + j];
    }

    for (int block = blockIdx.x; block < num_elements; block += gridDim.x) {
      int id = block * NDOFS + threadIdx.x;
      _xe[threadIdx.x] = xe[id];

      // Evaluate coefficients at quadrature points
      T wq = 0.;
#pragma unroll
      for (int j = 0; j < NDOFS; j++)
        wq += _xe[j] * _phi[j][threadIdx.x];

      _xq[threadIdx.x] = detJ[id] * wq;

      T yi = 0;
#pragma unroll
      for (int iq = 0; iq < NDOFS; iq++)
        yi += _xq[iq] * _phi[threadIdx.x][iq];

      ye[id] = yi;
    }
  }
}

template <typename T>
void mass_apply(int num_elements, const T* xe, const T* phi, const T* detJ, T* ye) {
  int block_size = 32;
  const int num_blocks = num_elements / 8;
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