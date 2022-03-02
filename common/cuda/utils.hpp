// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include "cublas_v2.h"
#include "cuda/allocator.hpp"
#include <cuda_runtime.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>

namespace utils {
//-----------------------------------------------------------------------------
template <typename C>
void assert_cuda(C e) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CUDA ERROR: " + std::to_string(e));
}
//-----------------------------------------------------------------------------
/// Set device to be used for GPU executions for a given process.
/// The number of processes should match the number of available devices.
int set_device(MPI_Comm comm) {

  int rank = dolfinx::MPI::rank(comm);
  int mpi_size = dolfinx::MPI::size(comm);

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);

  if (num_devices < mpi_size) {
    throw std::runtime_error("The number of MPI processes should be less or equal the "
                             "number of available devices ("
                             + std::to_string(num_devices) + ").");
  }
  cudaSetDevice(rank);

  return rank;
}
//-----------------------------------------------------------------------------
void output_device_info() {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    double bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    std::cout << "Device Number: " << i << std::endl;
    std::cout << "\tDevice name: " << prop.name << std::endl;
    std::cout << "\tShared memory available per block (kB): "
              << prop.sharedMemPerBlock / 1e3 << std::endl;
    std::cout << "\tGlobal memory available (GB): " << prop.totalGlobalMem / 1e9
              << std::endl;
    std::cout << "\tPeak Memory Bandwidth (GB/s): " << bandwidth << std::endl;
  }
  std::cout << std::endl;
}
} // namespace utils