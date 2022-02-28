#include <basix/e-lagrange.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>
#include <boost/program_options.hpp>

#include <cuda_profiler_api.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/la.hpp>
#include <cuda/utils.hpp>

using namespace dolfinx;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
                                                      "size", po::value<int>()->default_value(32)),
    "degree", po::value<int>()->default_value(1));

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 0;
  }

  const int Nx = vm["size"].as<int>();
  const int degree = vm["degree"].as<int>();

  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    // MPI
    MPI_Comm mpi_comm{MPI_COMM_WORLD};
    int rank = utils::set_device(mpi_comm);

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

    cudaProfilerStart();

    // prefetch data to gpu
    linalg::prefetch(rank, x);

    // Scatter forward (owner to ghost -> one to many map)
    x.scatter_fwd();

    // Scatter reverse (ghosts to owners -> many to one map)
    x.scatter_rev(dolfinx::common::IndexMap::Mode::add);

    cudaProfilerStop();
  }

  common::subsystem::finalize_mpi();
  return 0;
}
