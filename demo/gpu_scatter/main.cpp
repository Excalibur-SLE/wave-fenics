#include <basix/e-lagrange.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/la.hpp>

using namespace dolfinx;

int degree = 1;
std::size_t Nx = 32;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    // MPI
    MPI_Comm mpi_comm{MPI_COMM_WORLD};
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);
    int mpi_size;
    MPI_Comm_size(mpi_comm, &mpi_size);

    int numGpus = 0;
    cudaGetDeviceCount(&numGpus);
    std::cout << numGpus << " " << mpi_size << std::endl;
    cudaSetDevice(rank);

    if (numGpus < mpi_size && mpi_size != 1) {
      throw std::runtime_error("The number of MPI processes should be less or equal the "
                               "number of available devices.");
    }

    // Read mesh and mesh tags
    std::array<std::array<double, 3>, 2> p = {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    std::array<std::size_t, 3> n = {Nx, Nx, Nx};
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        mpi_comm, p, n, mesh::CellType::hexahedron, mesh::GhostMode::none));

    // Create a Basix continuous Lagrange element of degree 1
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), 1,
        basix::element::lagrange_variant::equispaced, true);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(mesh, e, 1));
    auto idxmap = V->dofmap()->index_map;

    // Assemble RHS vector
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);

    // prefetch data to gpu
    linalg::prefetch(rank, x);

    // Scatter forward (owner to ghost -> one to many map)
    x.scatter_fwd();

    // Scatter reverse (ghosts to owners -> many to one map)
    x.scatter_rev(dolfinx::common::IndexMap::Mode::add);
  }

  common::subsystem::finalize_mpi();
  return 0;
}