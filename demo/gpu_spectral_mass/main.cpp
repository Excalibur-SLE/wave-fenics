#include <boost/program_options.hpp>
#include <basix/e-lagrange.h>
#include <cmath>
#include <iostream>

#include <cuda_profiler_api.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/la.hpp>
#include <cuda/spectral_mass.hpp>
#include <cuda/utils.hpp>

using namespace dolfinx;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help, h", "print usage message")(
    "size", po::value<std::size_t>()->default_value(32))(
    "degree", po::value<int>()->default_value(2));

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help")){
    std::cout << desc << "\n";
    return 0;
  }

  const std::size_t Nx = vm["size"].as<std::size_t>();
  const int degree = vm["degree"].as<int>();

  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {

    // Create mesh and function space
    std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
        mesh::create_box(MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {Nx, Nx, Nx},
                         mesh::CellType::hexahedron, mesh::GhostMode::none));

    // Create a Basix continuous Lagrange element of a given degree
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), degree,
        basix::element::lagrange_variant::equispaced, false);

    // Create a scalar function space
    std::shared_ptr<fem::FunctionSpace> V 
        = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(mesh, e, 1));

    auto index_map = V->dofmap()->index_map;
    auto bs = V->dofmap()->index_map_bs();
    int ncells = mesh->topology().index_map(3)->size_local();
    int ndofs = e.dim();

    // Create input and output array
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> u(index_map, bs, allocator);
    la::Vector<double, decltype(allocator)> m(index_map, bs, allocator);
    std::fill(u.mutable_array().begin(), u.mutable_array().end(), 1.0);

    linalg::prefetch(0, u);
    linalg::prefetch(0, m);

    // Create mass operator
    SpectralMassOperator<double> op(V, degree);

    // Apply operator
    double t = MPI_Wtime();
    op.apply(u, m);
    t = MPI_Wtime() - t;

    std::cout << "Number of cells: " << op.num_cells() << "\n";
    std::cout << "Number of quads: " << op.num_quads() << "\n";
    std::cout << "Elapsed time: " << t << "\n";
    std::cout << "DOF/s: " << V->dofmap()->index_map->size_local() / t << "\n";
  }

  common::subsystem::finalize_mpi();
  return 0;
}
