// Copyright 2022 Chris Richardson and Athena Elafrou
// MIT licence

#include <dolfinx/common/log.h>
#include <iostream>
#include <memory>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/scatter.hpp>
#include <cuda/utils.hpp>

using namespace dolfinx;

class VectorUpdater
{
public:
  VectorUpdater(std::shared_ptr<const dolfinx::common::IndexMap> index_map) : cuda_block_size(512)
  {
    LOG(INFO) << "Vector Updater";
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

    LOG(INFO) << "Copy to device";
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

    LOG(INFO) << "Get neighbourhood info";
    // Find my neighbors in forward communicator
    fwd_comm = index_map->comm(dolfinx::common::IndexMap::Direction::forward);
    int num_recv_neighbors, num_send_neighbors;
    int weighted;
    MPI_Dist_graph_neighbors_count(fwd_comm, &num_recv_neighbors,
                                   &num_send_neighbors, &weighted);
    fwd_recv_neighbors.resize(num_recv_neighbors);
    fwd_send_neighbors.resize(num_send_neighbors);
    std::vector<int> weights_recv(num_recv_neighbors);
    std::vector<int> weights_send(num_send_neighbors);
    MPI_Dist_graph_neighbors(fwd_comm, num_recv_neighbors,
			     fwd_recv_neighbors.data(), weights_recv.data(),
			     num_send_neighbors, fwd_send_neighbors.data(), weights_send.data());

    // Find my neighbors in reverse communicator
    rev_comm = index_map->comm(dolfinx::common::IndexMap::Direction::reverse);
    int num_rev_recv_neighbors, num_rev_send_neighbors;
    MPI_Dist_graph_neighbors_count(rev_comm, &num_rev_recv_neighbors,
                                   &num_rev_send_neighbors, &weighted);
    rev_recv_neighbors.resize(num_rev_recv_neighbors);
    rev_send_neighbors.resize(num_rev_send_neighbors);
    weights_recv.resize(num_rev_recv_neighbors);
    weights_send.resize(num_rev_send_neighbors);
    MPI_Dist_graph_neighbors(
			     rev_comm, num_rev_recv_neighbors, rev_recv_neighbors.data(),weights_recv.data(),
			     num_rev_send_neighbors, rev_send_neighbors.data(), weights_send.data());
  }

  void update_fwd(la::Vector<double, CUDA::allocator<double>>& x)
  {
    MPI_Datatype data_type = dolfinx::MPI::mpi_type<double>();

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

    // Scatter reverse (ghosts to owners -> many to one map)
    // _buffer_recv_fwd is send_buffer
    // _buffer_send_fwd is recv_buffer
    // So swap send_buffer and recv_buffer from scatter_fwd

    // Step 1: pack send buffer
    xtl::span<const double> x_remote_const(
        x.array().data() + x.map()->size_local(), x.map()->num_ghosts());
    gather(d_ghost_pos_recv_fwd->size(), d_ghost_pos_recv_fwd->data(),
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

  int cuda_block_size;

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
