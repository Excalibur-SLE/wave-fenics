#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <dolfinx.h>
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

using namespace dolfinx;
namespace po = boost::program_options;

class VectorUpdater
{
public:
  VectorUpdater(std::shared_ptr<const dolfinx::common::IndexMap> index_map)
  {
    // Compute recv displacements and sizes
    displs_recv_fwd = index_map->scatter_fwd_receive_offsets();
    sizes_recv_fwd.resize(displs_recv_fwd.size());
    std::adjacent_difference(displs_recv_fwd.begin(), displs_recv_fwd.end(),
                             sizes_recv_fwd.begin());

    // Compute send displacements and sizes
    const dolfinx::graph::AdjacencyList<std::int32_t>& shared_indices
        = index_map->scatter_fwd_indices();
    displs_send_fwd = shared_indices.offsets();
    sizes_send_fwd.resize(displs_send_fwd.size());
    std::adjacent_difference(displs_send_fwd.begin(), displs_send_fwd.end(),
                             sizes_send_fwd.begin());

    // Copy indices to device
    const std::vector<std::int32_t>& indices = shared_indices.array();
    d_indices = std::make_unique<cuda::array<std::int32_t>>(indices.size());
    d_indices->set(indices);

    // Copy ghost_pos_recv_fwd to device
    const std::vector<std::int32_t>& ghost_pos_recv_fwd
        = index_map->scatter_fwd_ghost_positions();
    d_ghost_pos_recv_fwd = std::make_unique<cuda::array<std::int32_t>>(
        ghost_pos_recv_fwd.size());
    d_ghost_pos_recv_fwd->set(ghost_pos_recv_fwd);

    // Allocate device buffers for send and receive
    d_send_buffer = std::make_unique<cuda::array<double>>(indices.size());
    d_recv_buffer
        = std::make_unique<cuda::array<double>>(displs_recv_fwd.back());

    // Find my neighbors in forward communicator
    fwd_comm = index_map->comm(dolfinx::common::IndexMap::Direction::forward);
    int num_recv_neighbors, num_send_neighbors;
    int weighted;
    MPI_Dist_graph_neighbors_count(fwd_comm, &num_recv_neighbors,
                                   &num_send_neighbors, &weighted);
    fwd_recv_neighbors.resize(num_recv_neighbors);
    fwd_send_neighbors.resize(num_send_neighbors);
    MPI_Dist_graph_neighbors(fwd_comm, num_recv_neighbors,
			     fwd_recv_neighbors.data(), nullptr,
        num_send_neighbors, fwd_send_neighbors.data(), nullptr);

    // Find my neighbors in reverse communicator
    rev_comm = index_map->comm(dolfinx::common::IndexMap::Direction::reverse);
    int num_rev_recv_neighbors, num_rev_send_neighbors;
    MPI_Dist_graph_neighbors_count(rev_comm, &num_rev_recv_neighbors,
                                   &num_rev_send_neighbors, &weighted);
    rev_recv_neighbors.resize(num_rev_recv_neighbors);
    rev_send_neighbors.resize(num_rev_send_neighbors);
    MPI_Dist_graph_neighbors(
        rev_comm, num_rev_recv_neighbors, rev_recv_neighbors.data(), nullptr,
        num_rev_send_neighbors, rev_send_neighbors.data(), nullptr);
  }

  void update_fwd(la::Vector<double, CUDA::allocator<double>>& x)
  {

    MPI_Datatype data_type = dolfinx::MPI::mpi_type<double>();

    // Set thread block size for CUDA kernels
    const int cuda_block_size = 512;

    // Step 1: pack send buffer
    xtl::span<const double> x_local_const = x.array();
    gather(d_indices->size(), d_indices->data(), x_local_const.data(),
           d_send_buffer->data(), cuda_block_size);

    // Step 2: begin scatter
    std::vector<MPI_Request> req(fwd_recv_neighbors.size());
    for (std::size_t i = 0; i < req.size(); ++i)
    {
      MPI_Irecv(d_recv_buffer->data() + displs_recv_fwd[i],
                sizes_recv_fwd[i + 1], data_type, fwd_recv_neighbors[i], 0,
                fwd_comm, &(req[i]));
    }

    for (std::size_t i = 0; i < req.size(); ++i)
    {
      MPI_Send(d_send_buffer->data() + displs_send_fwd[i], sizes_send_fwd[i + 1],
               data_type, fwd_send_neighbors[i], 0, fwd_comm);
    }

    MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);

    // Step 3: copy into ghost area
    xtl::span<double> x_remote(x.mutable_array().data() + x.map()->size_local(),
                               x.map()->num_ghosts());
    gather(d_ghost_pos_recv_fwd->size(), d_ghost_pos_recv_fwd->data(),
           d_recv_buffer->data(), x_remote.data(), cuda_block_size);
  }

  void update_rev(la::Vector<double, CUDA::allocator<double>>& x)
  {
    MPI_Datatype data_type = dolfinx::MPI::mpi_type<double>();

    // Set thread block size for CUDA kernels
    const int cuda_block_size = 512;    

    // Scatter reverse (ghosts to owners -> many to one map)
    // _buffer_recv_fwd is send_buffer
    // _buffer_send_fwd is recv_buffer
    // So swap send_buffer and recv_buffer from scatter_fwd

    // Step 1: pack send buffer
    xtl::span<const double> x_remote_const(
        x.array().data() + x.map()->size_local(), x.map()->num_ghosts());
    // FIXME without atomics
    scatter(d_indices->size(), d_ghost_pos_recv_fwd->data(),
            x_remote_const.data(), d_recv_buffer->data(), cuda_block_size);

    // Step 2: begin scatter
    std::vector<MPI_Request> req(rev_recv_neighbors.size());   
    for (std::size_t i = 0; i < req.size(); ++i)
    {
      MPI_Irecv(d_send_buffer->data() + displs_send_fwd[i],
                sizes_send_fwd[i + 1], data_type, rev_recv_neighbors[i], 0,
                rev_comm, &(req[i]));
    }

    for (std::size_t i = 0; i < req.size(); ++i)
    {
      MPI_Send(d_recv_buffer->data() + displs_recv_fwd[i],
               sizes_recv_fwd[i + 1], data_type, rev_send_neighbors[i], 0,
               rev_comm);
    }

    MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);

    // Step 3: copy/accumulate into owned part of the vector
    xtl::span<double> x_local(x.mutable_array());
    scatter(d_indices->size(), d_indices->data(), d_send_buffer->data(),
            x_local.data(), cuda_block_size);
  }

private:
  MPI_Comm fwd_comm, rev_comm;

  // Displacements and sizes
  std::vector<std::int32_t> displs_recv_fwd, sizes_recv_fwd;
  std::vector<std::int32_t> displs_send_fwd, sizes_send_fwd;

  // On-device arrays for scatter/gather indices
  std::unique_ptr<cuda::array<std::int32_t>> d_ghost_pos_recv_fwd, d_indices;
  // On-device arrays for communicated data
  std::unique_ptr<cuda::array<double>> d_send_buffer, d_recv_buffer;

  // Neighbor lists in both directions
  std::vector<int> fwd_recv_neighbors, fwd_send_neighbors;
  std::vector<int> rev_recv_neighbors, rev_send_neighbors;
};

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
    int rank = utils::set_device(mpi_comm);
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

    std::cout << "num ghosts = " << idxmap->num_ghosts() << "\n";

    
    // Assemble RHS vector
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> x(idxmap, 1, allocator);

    VectorUpdater vu(x.map());

    // Prefetch data to gpu
    linalg::prefetch(rank, x);

    // Start profiling
    cudaProfilerStart();

    vu.update_fwd(x);

    vu.update_rev(x);

    // End profiling
    cudaProfilerStop();
  }

  common::subsystem::finalize_mpi();
  return 0;
}
