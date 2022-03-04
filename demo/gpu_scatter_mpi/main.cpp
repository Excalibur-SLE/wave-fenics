// Copyright (C) 2022 Chris Richardson and Athena Elafrou
// SPDX-License-Identifier:    MIT

#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>
#include <memory>

#include <cuda_profiler_api.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/la.hpp>
#include <cuda/scatter.hpp>
#include <cuda/utils.hpp>

#include "VectorUpdater.hpp"

using namespace dolfinx;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "size", po::value<std::size_t>()->default_value(32))(
      "degree", po::value<int>()->default_value(1));

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 0;
  }

  const std::size_t Nx = vm["size"].as<std::size_t>();
  const int degree = vm["degree"].as<int>();

  //  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    // MPI
    MPI_Comm mpi_comm{MPI_COMM_WORLD};
    MPI_Comm local_comm;
    MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL, &local_comm);
    
    int gpu_rank = utils::set_device(local_comm);
    int mpi_rank = dolfinx::MPI::rank(mpi_comm);
    MPI_Datatype data_type = dolfinx::MPI::mpi_type<double>();

    // Read mesh and mesh tags
    std::array<std::array<double, 3>, 2> p
        = {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    std::array<std::size_t, 3> n = {Nx, Nx, Nx};
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        mpi_comm, p, n, mesh::CellType::hexahedron, mesh::GhostMode::none));

    // Create a Basix continuous Lagrange element of given degree
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), degree,
        basix::element::lagrange_variant::equispaced, false);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(mesh, e, 1));
    auto idxmap = V->dofmap()->index_map;

    // Assemble RHS vector
    LOG(INFO) << "Allocate vector";
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);

    // Fill with rank values
    x.set((double)mpi_rank);

    VectorUpdater vu(x);

    // Prefetch data to gpu
    linalg::prefetch(gpu_rank, x);

    // Start profiling
    cudaProfilerStart();

    vu.update_fwd(x);

    // Ghost region should fill up with external ranks with float values
    // the same as ghost_owner_rank in the index_map.

    auto ghost_owner = x.map()->ghost_owner_rank();
    auto w = x.array();
    const int size_local = x.map()->size_local();
    const int num_ghosts = x.map()->num_ghosts();
    for (int i = 0; i < num_ghosts; ++i)
      assert((int)w[size_local + i] == ghost_owner[i]);

    // Fill up ghost region with ones and clear the rest.
    x.set(0.0);
    std::fill(x.mutable_array().data() + size_local,
              x.mutable_array().data() + size_local + num_ghosts, 1.0);

    vu.update_rev(x);

    // End profiling
    cudaProfilerStop();

    
    double sum
        = std::accumulate(x.array().data(), x.array().data() + size_local, 0.0);

    double gl_sum;
    MPI_Reduce(&sum, &gl_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int gh_sum;
    MPI_Reduce(&num_ghosts, &gh_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0)
    {
      LOG(INFO) << "gh_sum and gl_sum should be the same";
      LOG(INFO) << "gl_sum = " << gl_sum;
      LOG(INFO) << "gh_sum = " << gh_sum;
      assert(gl_sum == gh_sum);
    }

    // Now some timings

    dolfinx::common::Timer tcuda("Fwd CUDA-MPI");
    LOG(INFO) << "CUDA MPI updates";
    {
      for (int i = 0; i < 10000; ++i)
        vu.update_fwd(x);
    }
    tcuda.stop();

    dolfinx::common::Timer tcpu("Fwd CPU-MPI");
    LOG(INFO) << "CPU MPI updates";
    {
      for (int i = 0; i < 10000; ++i)
        x.scatter_fwd();
    }
    tcpu.stop();

    dolfinx::common::Timer tcuda2("Rev CUDA-MPI");
    LOG(INFO) << "CUDA MPI rev updates";
    {
      for (int i = 0; i < 10000; ++i)
        vu.update_rev(x);
    }
    tcuda2.stop();

    dolfinx::common::Timer tcpu2("Rev CPU-MPI");
    LOG(INFO) << "CPU MPI rev updates";
    {
      for (int i = 0; i < 10000; ++i)
        x.scatter_rev(dolfinx::common::IndexMap::Mode::add);
    }
    tcpu2.stop();

  }

  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});

  common::subsystem::finalize_mpi();
  return 0;
}
