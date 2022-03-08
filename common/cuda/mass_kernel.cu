#include "mass_kernel.hpp"
#include <cstdint>
#include <iostream>

template <typename T, int ndofs, int bs>
static __global__ void _mass_apply_shm(std::int32_t num_elements, const T* xe,
                                       const T* phi, const T* detJ, T* ye) {
  int element = blockIdx.x * blockDim.x + threadIdx.x;
  int id = element * ndofs + threadIdx.y;

  // Allocate shared memory
  __shared__ T _xe[ndofs][bs];
  __shared__ T _xq[ndofs][bs];
  __shared__ T _phi[ndofs][ndofs];

  if (element < num_elements && threadIdx.y < ndofs) {
    // Prepare basis functions (load to shared memory)
    // Load Phi^T to shared memory
    for (int i = threadIdx.x; i < ndofs; i += blockDim.x)
      for (int j = threadIdx.y; j < ndofs; j += blockDim.y)
        _phi[j][i] = phi[i * ndofs + j];

    _xe[threadIdx.x][threadIdx.y] = xe[id];
    __syncthreads();

    // Evaluate coefficients at quadrature points
    T wq = 0;
    for (int j = 0; j < ndofs; j++)
      wq += _xe[j][threadIdx.x] * _phi[threadIdx.y][j];

    _xq[threadIdx.y][threadIdx.x] = detJ[id] * wq;

    for (int i = threadIdx.x; i < ndofs; i += blockDim.x)
      for (int j = threadIdx.y; j < ndofs; j += blockDim.y)
        _phi[i][j] = phi[i * ndofs + j];

    T yi = 0;
    for (int iq = 0; iq < ndofs; iq++) {
      yi += _xq[iq][threadIdx.x] * _phi[threadIdx.y][iq];
    }
    ye[id] = yi;
  }
}
// ------------------------------------------------------------------//
template <typename T, int ndofs>
static __global__ void _mass_apply(std::int32_t num_elements, const T* xe, const T* phi,
                                   const T* detJ, T* ye) {
  __shared__ T _phi[ndofs][ndofs];
  __shared__ T _xe[ndofs];
  __shared__ T _xq[ndofs];

  if (threadIdx.x < ndofs) {
#pragma unroll
    for (int j = 0; j < ndofs; j++) {
      _phi[threadIdx.x][j] = phi[j * ndofs + threadIdx.x];
    }

    for (int block = blockIdx.x; block < num_elements; block += gridDim.x) {
      int id = block * ndofs + threadIdx.x;
      _xe[threadIdx.x] = xe[id];

      // Evaluate coefficients at quadrature points
      T wq = 0.;
#pragma unroll
      for (int j = 0; j < ndofs; j++)
        wq += _xe[j] * _phi[threadIdx.x][j];

      _xq[threadIdx.x] = detJ[id] * wq;

      T yi = 0;
#pragma unroll
      for (int iq = 0; iq < ndofs; iq++)
        yi += _xq[iq] * _phi[iq][threadIdx.x];

      ye[id] = yi;
    }
  }
}
// ------------------------------------------------------------------//
template <typename T, int ndofs>
void mass_apply(int num_elements, const T* xe, const T* phi, const T* detJ, T* ye) {
  bool simple = true;
  if (simple) {
    int block_size = 32 * ((ndofs + 32 - 1) / 32);
    const int num_blocks = num_elements / 8;
    dim3 dimBlock(block_size);
    dim3 dimGrid(num_blocks);
    _mass_apply<T, ndofs><<<dimGrid, dimBlock>>>(num_elements, xe, phi, detJ, ye);
  } else {
    constexpr int cells_per_block = 8;
    int dimx = 32 * ((ndofs + 32 - 1) / 32);
    int block_size = dimx * cells_per_block;
    const int num_blocks = (num_elements * dimx + block_size - 1) / block_size;
    dim3 dimBlock(cells_per_block, dimx);
    dim3 dimGrid(num_blocks, 1);
    std::cout << num_blocks << std::endl;
    cudaDeviceSynchronize();
    _mass_apply_shm<T, ndofs, cells_per_block>
        <<<dimGrid, dimBlock>>>(num_elements, xe, phi, detJ, ye);
  }
}
//-----------------------------------------------------------------------------
template void mass_apply<double, 8>(int Ne, const double* xe, const double* phi,
                                    const double* detJ, double* ye);
template void mass_apply<double, 27>(int Ne, const double* xe, const double* phi,
                                     const double* detJ, double* ye);
template void mass_apply<double, 64>(int Ne, const double* xe, const double* phi,
                                     const double* detJ, double* ye);
//-----------------------------------------------------------------------------