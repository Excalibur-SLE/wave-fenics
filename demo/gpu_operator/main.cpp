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
#include <cuda/scatter.hpp>
#include <cuda/transform.hpp>
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
        basix::element::lagrange_variant::equispaced, true);

    // Create a scalar function space
    std::shared_ptr<fem::FunctionSpace> V
        = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(mesh, e, 1));
    auto idxmap = V->dofmap()->index_map;

    int ncells = mesh->topology().index_map(3)->size_local();
    int ndofs = e.dim();

    fem::Function<double> u(V);
    // Interpolate sin(2 \pi x[0]) in the scalar Lagrange finite element space
    constexpr double PI = xt::numeric_constants<double>::PI;
    u.interpolate([PI](auto&& x) { return xt::sin(2 * PI * xt::row(x, 0)); });

    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);
    auto uarray = u.x()->array();
    std::copy(uarray.begin(), uarray.end(), x.mutable_array().begin());

    // =====================================
    // Tabulate basis functions at quadrature points
    // 1 - Tabulate quadrature points and weights
    int q = 2 * degree + 2; // Quadrature degree
    auto cell = basix::cell::type::hexahedron;
    auto quad = basix::quadrature::type::gauss_jacobi;
    auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);

    std::int32_t nquads = weights.size();
    std::int32_t Ne = ncells * ndofs;
    std::int32_t Nq = ncells * nquads;

    // 2 - Tabulate basis functions
    xt::xtensor<double, 4> basis = e.tabulate(0, points);
    xt::xtensor<double, 2> _phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
    cuda::array<double> phi(ndofs * nquads);
    phi.set(_phi);
    _phi = xt::transpose(_phi);
    cuda::array<double> phiT(ndofs * nquads);
    phiT.set(_phi);

    // =====================================
    // Get dofmap for the gather/scatter
    // Copy dofmap data to device
    cuda::array<std::int32_t> dofmap(Ne);
    const std::vector<std::int32_t>& dof_array = V->dofmap()->list().array();
    dofmap.set(dof_array);

    // =====================================
    // Compute determinant of Jacobian
    // TODO: precompute jacobian function in perecomputation.hpp
    cuda::array<double> detJ(Nq);

    // Allocate memory for working arrays on device
    cuda::array<double> ue(Ne);
    cuda::array<double> uq(Nq);
    cuda::array<double> xe(Ne);

    // =====================================
    // Apply gather operator Ue = G u
    // Ue <- u[dofmap]
    // From global dof vector to element based dof vector
    gather(ue.size(), dofmap.data(), x.array().data(), ue.data(), 512);

    double alpha = 1;
    double beta = 0;

    // =====================================
    // Apply operator B^T D B to Ue
    double t = MPI_Wtime();
    // Uq^ = B Ue^T
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nquads, ncells, ndofs, &alpha,
                phiT.data(), nquads, ue.data(), ndofs, &beta, uq.data(), nquads);
    // Uq = detJ .* Uq
    transform1(Ne, uq.data(), detJ.data(), uq.data(), 512);
    // Uq = detJ .* Uq
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ndofs, ncells, nquads, &alpha,
                phi.data(), ndofs, uq.data(), nquads, &beta, xe.data(), ndofs);
    cudaDeviceSynchronize();
    t = MPI_Wtime() - t;

    // =====================================
    // Apply scatter operator
    // x[dofmap] <- Xe
    // From element based dof vector to global dof vector
    // scatter(ue.size(), dofmap.data(), x.array().data(), ue.data(), 512);

    std::cout << "Number of cells: " << ncells;
    std::cout << "\nNumber of dofs: " << ndofs;
    std::cout << "\nNumber of quads: " << nquads;
    std::cout << "\n#FLOPs: " << ((4 * ncells * nquads * ndofs) + Ne) / t;
    std::cout << "\nDOF/s: " << V->dofmap()->index_map->size_local() / t;
    std::cout << std::endl;
  }

  common::subsystem::finalize_mpi();
  return 0;
}
