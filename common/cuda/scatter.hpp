#pragma once

#include <cuda_runtime.h>

// out[indices[i]] += in[i];
void scatter(const int N, const int* indices, const double* in, double* out,
             int block_size);

// in[i] = y[out[i]];
void gather(const int N, const int* indices, const double* in, double* out,
            int block_size);