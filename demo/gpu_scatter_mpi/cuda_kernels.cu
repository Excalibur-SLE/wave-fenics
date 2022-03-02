#include "cuda_kernels.hpp"

static
__global__ void _gather(const int N, const int* indices, const double* in, double* out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[indices[gid]];
  }
}

void gather(const int N, const int* indices, const double* in, double* out)
{
  // FIXME: make thread block size tunable
  const int block_size = 128;
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  // FIXME: make data is on-device
  _gather<<<dimGrid, dimBlock>>>(N, indices, in, out);
  cudaDeviceSynchronize();
}
