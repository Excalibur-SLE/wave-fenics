#include <basix/e-lagrange.h>
#include <basix/quadrature.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_profiler_api.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/la.hpp>
#include <cuda/mass.hpp>
#include <cuda/scatter.hpp>
#include <cuda/transform.hpp>
#include <cuda/utils.hpp>
#include <operators.hpp>

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
      "degree", po::value<int>()->default_value(1))(
      "check", po::value<bool>()->default_value(false));

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
  const bool check = vm["check"].as<bool>();

  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    // MPI
    MPI_Comm mpi_comm{MPI_COMM_WORLD};
    int rank = utils::set_device(mpi_comm);

    // Create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Read mesh and mesh tags
    std::array<std::array<double, 3>, 2> p = {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    std::array<std::size_t, 3> n = {Nx, Nx, Nx};
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        mpi_comm, p, n, mesh::CellType::hexahedron, mesh::GhostMode::none));

    // Create a Basix continuous Lagrange element of given degree
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), degree,
        basix::element::lagrange_variant::gll_warped, false);

    // Create a scalar function space
    std::shared_ptr<fem::FunctionSpace> V
        = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(mesh, e, 1));
    auto idxmap = V->dofmap()->index_map;

    int ncells = mesh->topology().index_map(3)->size_local();
    int ndofs = e.dim();

    fem::Function<double> u(V);
    // Interpolate sin(2 \pi x[0]) in the scalar Lagrange finite element space
    constexpr double PI = xt::numeric_constants<double>::PI;
    u.interpolate([PI](auto&& x) { return PI * xt::row(x, 0); });

    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);
    la::Vector<double, decltype(allocator)> y(idxmap, 1, allocator);
    std::fill(x.mutable_array().begin(), x.mutable_array().end(), 1);

    linalg::prefetch(0, x);
    linalg::prefetch(0, y);

    auto quad = basix::quadrature::type::gll;
    int qdegree = (degree > 1) ? degree + 1 : degree;
    MassOperator<double> op(V, e, quad, qdegree);

    double t = MPI_Wtime();
    op.apply(x, y);
    t = MPI_Wtime() - t;

    if (check) {
      la::Vector<double> x(idxmap, 1);
      la::Vector<double> y1(idxmap, 1);
      std::fill(x.mutable_array().begin(), x.mutable_array().end(), 1);
      MassOperatorCPU<double> cpu_op(V, degree);
      double t = MPI_Wtime();
      cpu_op(x, y1);
      t = MPI_Wtime() - t;
      std::cout << "Y norm: " << y1.norm() << std::endl;

      for (int i = 0; i < x.array().size(); i++) {
        double err = y1.array()[i] - y.array()[i];
        if (std::abs(err) > 1e-8)
          std::cout << y1.array()[i] - y.array()[i] << " ";
      }
      std::cout << "\n#Elapsed Time: " << t << std::endl;
    }

    std::cout << "X norm: " << x.norm() << std::endl;
    std::cout << "Y norm: " << y.norm() << std::endl;
    std::cout << "Number of cells: " << op.num_cells();
    std::cout << "\nNumber of dofs: " << op.num_dofs();
    std::cout << "\nNumber of quads: " << op.num_quads();
    std::cout << "\n#Elapsed Time: " << t;
    std::cout << "\nDOF/s: " << V->dofmap()->index_map->size_local() / t;
    std::cout << std::endl;
  }

  common::subsystem::finalize_mpi();
  return 0;
}
