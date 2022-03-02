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
#include <cuda/utils.hpp>

#include <cuda_kernels.hpp>

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
    std::array<std::array<double, 3>, 2> p
        = {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};
    std::array<std::size_t, 3> n = {Nx, Nx, Nx};
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        mpi_comm, p, n, mesh::CellType::hexahedron, mesh::GhostMode::none));

    // Create a Basix continuous Lagrange element of given degree
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), degree,
        basix::element::lagrange_variant::equispaced, true);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(mesh, e, 1));
    auto idxmap = V->dofmap()->index_map;

    // Assemble RHS vector
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);

    // Recv displacements and sizes
    const std::vector<std::int32_t>& displs_recv_fwd = x.map()->scatter_fwd_receive_offsets();
    std::vector<std::int32_t> sizes_recv_fwd(displs_recv_fwd.size());
    std::adjacent_difference(displs_recv_fwd.begin(), displs_recv_fwd.end(),
                             sizes_recv_fwd.begin());
    cuda::array<std::int32_t> d_sizes_recv_fwd(sizes_recv_fwd.size());
    cuda::array<std::int32_t> d_displs_recv_fwd(displs_recv_fwd.size());
    d_sizes_recv_fwd.set(sizes_recv_fwd);
    d_displs_recv_fwd.set(displs_recv_fwd);

    // Send displacements and sizes
    const std::vector<std::int32_t>& displs_send_fwd = shared_indices.offsets();
    std::vector<std::int32_t> sizes_send_fwd(displs_send_fwd.size());
    std::adjacent_difference(displs_send_fwd.begin(), displs_send_fwd.end(),
                             sizes_send_fwd.begin());
    cuda::array<std::int32_t> d_sizes_send_fwd(sizes_send_fwd.size());
    cuda::array<std::int32_t> d_displs_send_fwd(displs_send_fwd.size());
    d_sizes_send_fwd.set(sizes_send_fwd);
    d_displs_send_fwd.set(displs_send_fwd);

    const dolfinx::graph::AdjacencyList<std::int32_t>& shared_indices
        = x.map()->scatter_fwd_indices();
    const std::vector<std::int32_t>& indices = shared_indices.array();
    cuda::array<std::int32_t> d_indices(indices.size());
    d_indices.set(indices);
    const std::vector<std::int32_t>& ghost_pos_recv_fwd
        = x.map()->scatter_fwd_ghost_positions();
    cuda::array<std::int32_t> d_ghost_pos_recv_fwd(ghost_pos_recv_fwd.size());
    d_ghost_pos_recv_fwd.set(ghost_pos_recv_fwd);

    cuda::array<double> d_send_buffer(indices.size());
    cuda::array<double> d_recv_buffer(displs_recv_fwd.back());

    // Find my neighbors
    MPI_Comm comm
        = x.map()->comm(dolfinx::common::IndexMap::Direction::forward);
    int num_recv_neighbors, num_send_neighbors;
    int weighted;
    MPI_Dist_graph_neighbors_count(comm, &num_recv_neighbors,
                                   &num_send_neighbors, &weighted);
    std::vector<int> recv_neighbors(num_recv_neighbors);
    std::vector<int> send_neighbors(num_send_neighbors);
    MPI_Dist_graph_neighbors(comm, num_recv_neighbors, recv_neighbors.data(),
                             nullptr, num_send_neighbors, send_neighbors.data(),
                             nullptr);

    // Prefetch data to gpu
    linalg::prefetch(rank, x);

    // Start profiling
    cudaProfilerStart();

    // Scatter forward (owner to ghost -> one to many map)
    xtl::span<const double> local_data = x.array();
    gather(indices.size(), d_indices, local_data.data(), d_send_buffer);

    // Start send/receive
    MPI_Request* req = new MPI_Request[num_recv_neighbors];
    for (int i = 0; i < num_recv_neighbors; ++i) {
      MPI_Irecv(d_recv_buffer + d_displs_recv_fwd[i], d_sizes_recv_fwd[i + 1],
                dolfinx::MPI::mpi_type<double>(), recv_neighbors[i], 0, comm,
                &(req[i]));
    }

    for (int i = 0; i < num_send_neighbors; ++i) {
      MPI_Send(d_send_buffer + d_displs_send_fwd[i], d_sizes_send_fwd[i + 1],
               dolfinx::MPI::mpi_type<double>(), send_neighbors[i], 0, comm);
    }

    MPI_Waitall(num_recv_neighbors, req, MPI_STATUSES_IGNORE);

    // Copy into ghost area ("remote_data")
    xtl::span<double> remote_data(x.mutable_array().data()
                                      + x.map()->size_local(),
                                  x.map()->num_ghosts());
    gather(ghost_pos_recv_fwd.size(), d_ghost_pos_recv_fwd, d_recv_buffer,
           remote_data.data());

    // End profiling
    cudaProfilerStop();
  }

  common::subsystem::finalize_mpi();
  return 0;
}
