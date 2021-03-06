#include "scatter.hpp"

//-----------------------------------------------------------------------------
template <typename T>
static __global__ void _gather(const int N, const std::int32_t* __restrict__ indices,
                               const T* __restrict__ in, T* __restrict__ out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[indices[gid]];
  }
}
//-----------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

__device__ double _atomicAdd(double* address, const double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
//-----------------------------------------------------------------------------
static __global__ void _scatter(std::int32_t N, const int32_t* __restrict__ indices,
                                const double* __restrict__ in, double* __restrict__ out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    _atomicAdd(&out[indices[gid]], in[gid]);
  }
}
#endif
//-----------------------------------------------------------------------------
template <typename T>
static __global__ void _scatter(std::int32_t N, const int32_t* __restrict__ indices,
                                const T* __restrict__ in, T* __restrict__ out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    atomicAdd(&out[indices[gid]], in[gid]);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void gather(std::int32_t N, const std::int32_t* indices, const T* in, T* out,
            int block_size) {
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _gather<<<dimGrid, dimBlock>>>(N, indices, in, out);
  cudaDeviceSynchronize();
}
//-----------------------------------------------------------------------------
template <typename T>
void scatter(std::int32_t N, const std::int32_t* indices, const T* in, T* out,
             int block_size) {
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _scatter<<<dimGrid, dimBlock>>>(N, indices, in, out);
  cudaDeviceSynchronize();
}
//-----------------------------------------------------------------------------
template void gather<double>(std::int32_t, const std::int32_t*, const double*, double*,
                             int);
template void gather<float>(std::int32_t, const std::int32_t*, const float*, float*, int);
template void scatter<double>(std::int32_t, const std::int32_t*, const double*, double*,
                              int);
template void scatter<float>(std::int32_t, const std::int32_t*, const float*, float*,
                             int);
//-----------------------------------------------------------------------------
