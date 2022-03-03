#include "transform.hpp"
#include <cstdint>

//-----------------------------------------------------------------------------
template <typename T>
static __global__ void _transform1(const int N, const T* in, T* detJ, T* out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[gid] * detJ[gid];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void transform1(const std::int32_t N, const T* in, T* detJ, T* out, int block_size) {
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _transform1<<<dimGrid, dimBlock>>>(N, in, detJ, out);
  cudaDeviceSynchronize();
}
//-----------------------------------------------------------------------------
template void transform1<double>(const std::int32_t N, const double* in, double* detJ,
                                 double* out, int block_size);
template void transform1<float>(const std::int32_t N, const float* in, float* detJ,
                                float* out, int block_size);
//-----------------------------------------------------------------------------
