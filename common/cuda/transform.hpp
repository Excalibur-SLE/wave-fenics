#pragma once

#include <cuda_runtime.h>

/// Compute action of D for the Mass matrix
/// in[c, q] = xq[c, q] * detJ[c, q]
void transform1(const int N, const double* in, double* detJ, double* out, int block_size);