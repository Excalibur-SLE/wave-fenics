#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/// out[indices[i]] += in[i];
void scatter(const int N, const std::int32_t* indices, const double* in, double* out,
             int block_size);

/// in[i] = out[indices[i]];
void gather(const int N, const std::int32_t* indices, const double* in, double* out,
            int block_size);