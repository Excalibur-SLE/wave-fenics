
#include "scatter.hpp"

static __global__ void _gather(const int N, const std::int32_t* __restrict__ indices,
                               const double* __restrict__ in, double* __restrict__ out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[indices[gid]];
  }
}

// The 64-bit floating-point version of atomicAdd() is only supported by devices of
// compute capability 6.x and higher.
static __global__ void _scatter(const int N, const int32_t* indices, const double* in,
                                double* out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    // atomicAdd(&out[indices[gid]], in[gid]);
  }
}

void gather(const int N, const int* indices, const double* in, double* out,
            int block_size) {
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _gather<<<dimGrid, dimBlock>>>(N, indices, in, out);
  cudaDeviceSynchronize();
}

void scatter(const int N, const int* indices, const double* in, double* out,
             int block_size) {
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _scatter<<<dimGrid, dimBlock>>>(N, indices, in, out);
  cudaDeviceSynchronize();
}