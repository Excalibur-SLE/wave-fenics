// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "streaming.hpp"
#include "VectorUpdater.hpp"
#include <cuda_runtime.h>

#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>

namespace {
template <typename T>
inline T mpi_reduce(MPI_Comm comm, T value) {
  int mpi_size = dolfinx::MPI::size(comm);
  if (mpi_size == 1)
    return value;
  else {
    T output = 0;
    MPI_Allreduce(&value, &output, 1, dolfinx::MPI::mpi_type<T>(), MPI_SUM, comm);
    return output;
  }
}
} // namespace

using namespace dolfinx;
using namespace linalg::CUDA;

namespace device {
/// Solve problem A.x = b with Conjugate Gradient method
/// @param queue SYCL queue
/// @param x Solution Vector
/// @param b RHS Vector
/// @param matvec_function Function that provides the operator action
/// @param kmax Maxmimum number of iterations
template <typename T, typename Alloc>
int cg(cublasHandle_t handle, la::Vector<T, Alloc>& x, const la::Vector<T, Alloc>& b,
       std::function<void(const la::Vector<T, Alloc>&, la::Vector<T, Alloc>&)>
           matvec_function,
       int kmax = 50, double rtol = 1e-8) {

  MPI_Comm comm = x.map()->comm(common::IndexMap::Direction::forward);
  int mpi_size = dolfinx::MPI::size(comm);
  //  int rank = dolfinx::MPI::rank(comm);

  //  int M = b.map()->size_local();

  // Create cuda managed allocator
  Alloc allocator;

  // Allocate auxiliary vectors
  LOG(INFO) << "Allocate vectors";
  la::Vector<T, Alloc> r(b.map(), b.bs(), allocator);
  la::Vector<T, Alloc> y(b.map(), b.bs(), allocator);
  la::Vector<T, Alloc> p(b.map(), b.bs(), allocator);

  LOG(INFO) << "Call copy";
  copy<la::Vector<T, Alloc>>(handle, b, p);
  copy<la::Vector<T, Alloc>>(handle, b, r);
  //   cudaMemAdvise()

  LOG(INFO) << "Sq norm";
  T rnorm0_local = squared_norm<la::Vector<T, Alloc>>(handle, r);

  LOG(INFO) << "reduce";
  T rnorm0  = mpi_reduce<T>(comm, rnorm0_local);

  LOG(INFO) << "Vector updater";
  VectorUpdater<double, CUDA::allocator<double>> vu(p);
  
  // Iterations of CG
  const T rtol2 = rtol * rtol;
  T rnorm = rnorm0;
  int k = 0;
  while (k < kmax) {
    ++k;

    // Update ghosts before MatVec
    if (mpi_size > 1)
      {
	LOG(INFO) << "Update forward";
	p.scatter_fwd();
	// vu.update_fwd(p);
      }

    // MatVec
    // y = A.p;
    LOG(INFO) << "Update forward";
    matvec_function(p, y);

    LOG(INFO) << "Inner product";

    // Calculate alpha = r.r/p.y
    T pdoty_local = inner_product<la::Vector<T, Alloc>>(handle, p, y);
    T pdoty = mpi_reduce<T>(comm, pdoty_local);

    const T alpha = rnorm / pdoty;

    // Update x and r
    // x = x + alpha*p
    // r = r - alpha*y
    axpy<T, la::Vector<T, Alloc>>(handle, alpha, p, x);
    axpy<T, la::Vector<T, Alloc>>(handle, -alpha, y, r);

    // Update rnorm
    T rnorm_new_local = squared_norm<la::Vector<T, Alloc>>(handle, r);
    T rnorm_new = mpi_reduce<T>(comm, rnorm_new_local);

    const T beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p.
    // p = beta*p + r
    scale<T, la::Vector<T, Alloc>>(handle, beta, p);
    axpy<T, la::Vector<T, Alloc>>(handle, 1, p, r);
  }

  return k;
}
} // namespace device
