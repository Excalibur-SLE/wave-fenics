#include "transform.hpp"

static __global__ void _transform1(const int N, const double* in, double* detJ,
                                   double* out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[gid] * detJ[gid];
  }
}

void transform1(const int N, const double* in, double* detJ, double* out,
                int block_size) {
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _transform1<<<dimGrid, dimBlock>>>(N, in, detJ, out);
  cudaDeviceSynchronize();
}