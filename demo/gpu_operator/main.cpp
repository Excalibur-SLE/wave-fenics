#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_profiler_api.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/la.hpp>
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

    int ncells = mesh->topology().index_map(3)->size_local();
    int ndofs = e.dim();

    // xe [ncells times ndofs];
    // xq [ncells times ndofs];

    // Create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::int32_t Ne = ncells * ndofs;
    double *xe, *xq, *ue, *phi, *detJ;

    assert_cublas(cudaMalloc((void**)&xe, Ne * sizeof(double)));
    cudaMemset(&xe, 0.5, Ne * sizeof(double));

    assert_cublas(cudaMalloc((void**)&xq, Ne * sizeof(double)));
    cudaMemset(&xq, 0., Ne * sizeof(double));

    assert_cublas(cudaMalloc((void**)&ue, Ne * sizeof(double)));
    cudaMemset(&ue, 0., Ne * sizeof(double));

    assert_cublas(cudaMalloc((void**)&detJ, Ne * sizeof(double)));
    cudaMemset(&detJ, 0.5, Ne * sizeof(double));

    unsigned int size_phi = ndofs * ndofs;
    unsigned int mem_size_phi = sizeof(double) * size_phi;
    assert_cublas(cudaMalloc((void**)&phi, mem_size_phi));

    double alpha = 1;
    double beta = 0;

    double t = MPI_Wtime();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncells, ndofs, ndofs, &alpha, xe,
                ncells, phi, ndofs, &beta, xq, ncells);
    transform1(Ne, xq, detJ, xq, 512);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncells, ndofs, ndofs, &alpha, xq,
                ncells, phi, ndofs, &beta, ue, ncells);
    cudaDeviceSynchronize();
    t = MPI_Wtime() - t;

    std::cout << "Number of cells: " << ncells;
    std::cout << "\nNumber of dofs: " << ndofs;
    std::cout << "\n#GFLOPs: " << (4 * ncells * ndofs * ndofs) / t / 1e9;
    std::cout << "\nDOF/s: " << V->dofmap()->index_map->size_local() / t;
    std::cout << std::endl;
  }

  common::subsystem::finalize_mpi();
  return 0;
}
