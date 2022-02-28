#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

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

    cudaProfilerStart();

    // prefetch data to gpu
    linalg::prefetch(rank, x);

    // Scatter forward (owner to ghost -> one to many map)
    const dolfinx::graph::AdjacencyList<std::int32_t>& shared_indices = x.map()->scatter_fwd_indices();

    const std::vector<std::int32_t>& indices = shared_indices->array();
    std::vector<double> send_buffer(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
      send_buffer[i] = local_data[indices[i]];

    // Recv displacements and sizes
    std::vector<std::int32_t> displs_recv_fwd = {???};
    std::vector<std::int32_t> sizes_recv_fwd(displs_recv_fwd.size() - 1);
    std::adjacent_difference(displs_recv_fwd.begin(), displs_recv_fwd.end(), sizes_recv_fwd.begin());

    std::vector<double> recv_buffer(displs_recv_fwd.back());

    // Send displacements and sizes
    const std::vector<std::int32_t>& displs_send_fwd = shared_indices->offsets();
    std::vector<std::int32_t> sizes_send_fwd(displs_send_fwd.size() - 1);
    std::adjacent_difference(displs_send_fwd.begin(), displs_send_fwd.end(), sizes_send_fwd.begin());

    // Start send/receive
    MPI_Neighbor_alltoallv(send_buffer.data(), sizes_send_fwd.data(),
                           displs_send_fwd.data(), dolfinx::MPI::mpi_type<double>();
                           recv_buffer.data(), sizes_recv_fwd.data(),
                           displs_recv_fwd.data(), dolfinx::MPI::mpi_type<double>();
                           x.map()->comm(dolfinx::common::IndexMap::Direction::forward));

    // Copy into ghost area ("remote_data")
    const std::vector<std::int32_t>& ghost_pos_recv_fwd = x.map()->scatter_fwd_ghost_positions();
    assert(remote_data.size() == ghost_pos_recv_fwd.size());
    for (std::size_t i = 0; i < ghost_pos_recv_fwd.size(); ++i)
      remote_data[i] = buffer_recv[ghost_pos_recv_fwd[i]];
    
    cudaProfilerStop();
  }

  common::subsystem::finalize_mpi();
  return 0;
}
