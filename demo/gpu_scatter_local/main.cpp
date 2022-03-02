#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

#include <cuda_profiler_api.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/la.hpp>
#include <cuda/scatter.hpp>
#include <cuda/utils.hpp>

using namespace dolfinx;
namespace po = boost::program_options;

void assert_cublas(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(" Unable to allocate memoy - cublas error");
}

int main(int argc, char* argv[]) {

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "size", po::value<std::size_t>()->default_value(32))(
      "degree", po::value<int>()->default_value(1));

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  const std::size_t Nx = vm["size"].as<std::size_t>();
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

    // Create a Basix continuous Lagrange element of given degree
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), degree,
        basix::element::lagrange_variant::equispaced, true);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(mesh, e, 1));
    auto idxmap = V->dofmap()->index_map;

    // Assemble RHS vector
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);
    std::iota(x.mutable_array().begin(), x.mutable_array().end(), 0);

    int ncells = mesh->topology().index_map(3)->size_local();
    int ndofs = e.dim();

    std::int32_t Ne = ncells * ndofs;

    cuda::array<double> xe(Ne);
    cuda::array<std::int32_t> dofmap(Ne);
    const std::vector<std::int32_t>& dof_array = V->dofmap()->list().array();
    dofmap.set(dof_array);

    // prefetch data to gpu
    linalg::prefetch(rank, x);
    gather(Ne, dofmap.data(), x.array().data(), xe.data(), 512);

    auto h_xe = xe.copy_to_host();

    for (int i = 0; i < Ne; i++) {
      assert(h_xe[i] == double(dof_array[i]));
    }
  }

  common::subsystem::finalize_mpi();
  return 0;
}
