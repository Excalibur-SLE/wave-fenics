// Copyright (C) 2022 Chris Richardson and Athena Elafrou
// SPDX-License-Identifier:    MIT

#include <dolfinx/common/log.h>
#include <dolfinx/la/Vector.h>
#include <iostream>
#include <memory>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/scatter.hpp>

using namespace dolfinx;

/// Updater for distributed dolfinx::la::Vector<T> which has been
/// allocated on GPU with AllocatorT
/// It does not hold a copy of the Vector or the IndexMap, just takes
/// information from the IndexMap which is needed for update.

template <class T, class AllocatorT>
class VectorUpdater
{
public:
  /// Create an updater object, related to the indexmap of Vector x
  /// @param x A dolfinx::la::Vector allocated on GPU with CUDA USM
  VectorUpdater(dolfinx::la::Vector<T, AllocatorT>& x)
      : cuda_block_size(512), data_type(dolfinx::MPI::mpi_type<T>())
  {
    // Get the index map
    std::shared_ptr<const dolfinx::common::IndexMap> index_map = x.map();

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
    d_send_buffer = std::make_unique<cuda::array<T>>(indices.size());
    d_recv_buffer = std::make_unique<cuda::array<T>>(displs_recv_fwd.back());

    LOG(INFO) << "Get neighbourhood info";
    // Find my neighbors in forward communicator
    MPI_Comm_dup(index_map->comm(dolfinx::common::IndexMap::Direction::forward),
                 &fwd_comm);

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
                             num_send_neighbors, fwd_send_neighbors.data(),
                             weights_send.data());

    // Find my neighbors in reverse communicator
    MPI_Comm_dup(index_map->comm(dolfinx::common::IndexMap::Direction::reverse),
                 &rev_comm);
    int num_rev_recv_neighbors, num_rev_send_neighbors;
    MPI_Dist_graph_neighbors_count(rev_comm, &num_rev_recv_neighbors,
                                   &num_rev_send_neighbors, &weighted);
    rev_recv_neighbors.resize(num_rev_recv_neighbors);
    rev_send_neighbors.resize(num_rev_send_neighbors);
    weights_recv.resize(num_rev_recv_neighbors);
    weights_send.resize(num_rev_send_neighbors);
    MPI_Dist_graph_neighbors(rev_comm, num_rev_recv_neighbors,
                             rev_recv_neighbors.data(), weights_recv.data(),
                             num_rev_send_neighbors, rev_send_neighbors.data(),
                             weights_send.data());
  }

  // Free up comms
  ~VectorUpdater()
  {
    MPI_Comm_free(&fwd_comm);
    MPI_Comm_free(&rev_comm);
  }

  void update_fwd_begin(const la::Vector<T, AllocatorT>& x)
  {
    // Step 1: pack send buffer
    xtl::span<const T> x_local_const = x.array();
    gather(d_indices->size(), d_indices->data(), x_local_const.data(),
           d_send_buffer->data(), cuda_block_size);

    // Step 2: begin scatter
    requests.resize(fwd_recv_neighbors.size());
    for (std::size_t i = 0; i < fwd_recv_neighbors.size(); ++i)
    {
      int status = MPI_Irecv(
          d_recv_buffer->data() + displs_recv_fwd[i], sizes_recv_fwd[i + 1],
          data_type, fwd_recv_neighbors[i], 0, fwd_comm, &(requests[i]));
      assert(status == MPI_SUCCESS);
    }

    for (std::size_t i = 0; i < fwd_send_neighbors.size(); ++i)
    {
      int status = MPI_Send(d_send_buffer->data() + displs_send_fwd[i],
                            sizes_send_fwd[i + 1], data_type,
                            fwd_send_neighbors[i], 0, fwd_comm);
      assert(status == MPI_SUCCESS);
    }
  }

  void update_fwd_end(la::Vector<T, AllocatorT>& x)
  {
    int status
        = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    assert(status == MPI_SUCCESS);

    // Step 3: copy into ghost area
    xtl::span<T> x_remote(x.mutable_array().data() + x.map()->size_local(),
                          x.map()->num_ghosts());
    gather(d_ghost_pos_recv_fwd->size(), d_ghost_pos_recv_fwd->data(),
           d_recv_buffer->data(), x_remote.data(), cuda_block_size);
  }

  /// Do a forward scatter, sending locally owned values to the ghost region on
  /// remote processes
  /// @param x Vector to update
  void update_fwd(const la::Vector<T, AllocatorT>& x)
  {
    update_fwd_begin(x);
    update_fwd_end(x);
  }

  /// Do a reverse scatter, accumulating ghost values from remote processes into
  /// owned values on the local process
  /// @param x Vector to update
  void update_rev_begin(const la::Vector<T, AllocatorT>& x)
  {
    // Scatter reverse (ghosts to owners -> many to one map)
    // _buffer_recv_fwd is send_buffer
    // _buffer_send_fwd is recv_buffer
    // So swap send_buffer and recv_buffer from scatter_fwd

    // Step 1: pack send buffer
    xtl::span<const T> x_remote_const(x.array().data() + x.map()->size_local(),
                                      x.map()->num_ghosts());
    gather(d_ghost_pos_recv_fwd->size(), d_ghost_pos_recv_fwd->data(),
           x_remote_const.data(), d_recv_buffer->data(), cuda_block_size);

    // Step 2: begin scatter
    requests.resize(rev_recv_neighbors.size());
    for (std::size_t i = 0; i < rev_recv_neighbors.size(); ++i)
    {
      int status = MPI_Irecv(
          d_send_buffer->data() + displs_send_fwd[i], sizes_send_fwd[i + 1],
          data_type, rev_recv_neighbors[i], 0, rev_comm, &(requests[i]));
      assert(status == MPI_SUCCESS);
    }

    for (std::size_t i = 0; i < rev_send_neighbors.size(); ++i)
    {
      int status = MPI_Send(d_recv_buffer->data() + displs_recv_fwd[i],
                            sizes_recv_fwd[i + 1], data_type,
                            rev_send_neighbors[i], 0, rev_comm);
      assert(status == MPI_SUCCESS);
    }
  }

  void update_rev_end(la::Vector<T, AllocatorT>& x)
  {
    int status
        = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    assert(status == MPI_SUCCESS);

    // Step 3: copy/accumulate into owned part of the vector
    xtl::span<T> x_local(x.mutable_array());
    scatter(d_indices->size(), d_indices->data(), d_send_buffer->data(),
            x_local.data(), cuda_block_size);
  }

  /// Do a reverse scatter, accumulating ghost values from remote processes into
  /// owned values on the local process
  /// @param x Vector to update
  void update_rev(const la::Vector<T, AllocatorT>& x)
  {
    update_rev_begin(x);
    update_rev_end(x);
  }

private:
  // Parameters and MPI comms
  int cuda_block_size;
  MPI_Datatype data_type;
  MPI_Comm fwd_comm, rev_comm;

  std::vector<MPI_Request> requests;

  // Displacements and sizes
  std::vector<std::int32_t> displs_recv_fwd, sizes_recv_fwd;
  std::vector<std::int32_t> displs_send_fwd, sizes_send_fwd;

  // On-device arrays for scatter/gather indices
  std::unique_ptr<cuda::array<std::int32_t>> d_ghost_pos_recv_fwd, d_indices;
  // On-device arrays for communicated data
  std::unique_ptr<cuda::array<T>> d_send_buffer, d_recv_buffer;

  // Neighbor lists in both directions
  std::vector<int> fwd_recv_neighbors, fwd_send_neighbors;
  std::vector<int> rev_recv_neighbors, rev_send_neighbors;
};
