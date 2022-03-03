#pragma once

#include <cuda_runtime.h>

/// Compute action of D for the Mass matrix
/// in[c, q] = xq[c, q] * detJ[c, q]
template <typename T>
void transform1(const int N, const T* in, T* detJ, T* out, int block_size);