// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include "cuda/allocator.hpp"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>

namespace utils {

template <typename C>
void assert_cuda(C e) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CUDA ERROR: " + std::to_string(e));
}

/// Set device to be used for GPU executions for a given process.
/// The number of processes should match the number of available devices.
int set_device(MPI_Comm comm) {

  int rank = dolfinx::MPI::rank(comm);
  int mpi_size = dolfinx::MPI::size(comm);

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);

  if (num_devices != mpi_size && mpi_size != 1) {
    throw std::runtime_error("The number of MPI processes should be less or equal the "
                             "number of available devices ("
                             + std::to_string(num_devices) + ").");
  }
  cudaSetDevice(rank);

  return rank;
}

template <typename T>
auto create_distributed_vector(MPI_Comm comm, std::size_t global_size, T value = T{0},
                               int block_size = 1) {

  int size = dolfinx::MPI::size(comm);
  int rank = dolfinx::MPI::rank(comm);

  // Partition data
  std::int32_t size_local = static_cast<std::int32_t>(global_size / size);
  std::int32_t remainder = static_cast<std::int32_t>(global_size % size);

  if (rank < remainder)
    size_local++;

  assert(std::numeric_limits<std::int32_t>::max() > size_local);

  // Create simple indexmap with no overlap
  auto map = std::make_shared<dolfinx::common::IndexMap>(comm, size_local);

  CUDA::allocator<T> allocator{};
  dolfinx::la::Vector<T, decltype(allocator)> vec(map, block_size, allocator);

  T* data = vec.mutable_array().data();

  cudaMemset(data, value, size_local * sizeof(T));

  return vec;
}

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

} // namespace CUDA