#pragma once

#include <cuda_runtime.h>

void gather(const int N, const int* indices, const double* in, double* out);
