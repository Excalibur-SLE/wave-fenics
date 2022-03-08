// Copyright (C) 2021 Igor A. Baratta, Chris Richardson
// SPDX-License-Identifier:    MIT

#pragma once

#include <bit>
#include <cfloat>
#include <cmath>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <numeric>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
namespace benchmark::utils {
//-------------------------------------------------------------------------//
// Returns True if an unsigned integer is power of two, and false otherwise
bool is_power_of_two(const std::uint32_t x) { return (x & (x - 1)) == 0; }

//-------------------------------------------------------------------------//
// Returns log2 of an unsigned integer
// Note: this avoids floating point conversions
constexpr std::uint32_t log2(std::uint32_t x) noexcept {
  constexpr auto _Nd = 32;
  return _Nd - std::__countl_zero(x) - 1;
}

//-------------------------------------------------------------------------//
// Decompose 2^x into 2^x0 * 2^x1 * 2^x2
// where x = x0 + x1 + x2.
xt::xtensor<int, 1> decompose3d(int x) {
  xt::xtensor<int, 1> Nx = xt::zeros<int>({3});
  auto dv = std::div(x, 3);

  for (int i = 0; i < 3; i++) {
    Nx[i] = std::pow(2, dv.quot);
    if (i < dv.rem)
      Nx[i] *= 2;
  }

  return Nx;
}
//-------------------------------------------------------------------------//
// Compute the cartesian index (Ix, Iy, Iz) for each rank in the MPI
// communitator
xt::xtensor<std::size_t, 2> compute_cartesian_indices(xt::xtensor<int, 1> procs) {
  int size = std::accumulate(procs.begin(), procs.end(), 1, std::multiplies<int>());

  xt::xtensor<std::size_t, 2> indices = xt::zeros<int>({size, 3});

  for (int i = 0; i < size; i++) {
    indices(i, 2) = i % procs[2];
    indices(i, 1) = (i / procs[2]) % procs[1];
    indices(i, 0) = i / (procs[2] * procs[1]);
  }
  return indices;
}
//-------------------------------------------------------------------------//
std::pair<graph::AdjacencyList<std::int32_t>, xt::xtensor<std::size_t, 2>>
compute_cartesian_topology(MPI_Comm comm, xt::xtensor<int, 1> px) {
  // Compute strides
  std::array<std::size_t, 3> stride = {0};
  stride[0] = px[1] * px[2];
  stride[1] = px[1];
  stride[2] = 1;

  std::size_t size = std::accumulate(px.begin(), px.end(), 1, std::multiplies<int>());
  xt::xtensor<std::size_t, 2> cart_indices = compute_cartesian_indices(px);

  // Count number of neighbors (by facet)
  std::vector<std::int32_t> counter(size);
  for (std::size_t i = 0; i < size; i++) {
    for (std::size_t j = 0; j < 3; j++) {
      if (cart_indices(i, j) > 0)
        counter[i]++;
      if (cart_indices(i, j) < std::size_t(px[j] - 1))
        counter[i]++;
    }
  }
  std::vector<std::int32_t> offsets(size + 1);
  std::partial_sum(counter.begin(), counter.end(), offsets.begin() + 1);
  std::vector<std::int32_t> data(offsets.back());
  std::fill(counter.begin(), counter.end(), 0);

  for (std::size_t i = 0; i < size; i++) {
    xt::xtensor<double, 1> row = xt::row(cart_indices, i);
    xt::xtensor<double, 1> neighbor;
    for (std::size_t j = 0; j < 3; j++) {
      if (cart_indices(i, j) > 0) {
        neighbor = row;
        neighbor[j] -= 1;
        int pos = offsets[i] + counter[i]++;
        data[pos]
            = std::inner_product(neighbor.begin(), neighbor.end(), stride.begin(), 0);
      }
      if (cart_indices(i, j) < std::size_t(px[j] - 1)) {
        neighbor = row;
        neighbor[j] += 1;
        int pos = offsets[i] + counter[i]++;
        data[pos]
            = std::inner_product(neighbor.begin(), neighbor.end(), stride.begin(), 0);
      }
    }
  }
  return {graph::AdjacencyList<std::int32_t>(data, offsets), cart_indices};
}

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> create_geom(MPI_Comm comm,
                                   const std::array<std::array<double, 3>, 2>& p,
                                   xt::xtensor<std::size_t, 1> n) {
  // Extract data
  const std::array<double, 3>& p0 = p[0];
  const std::array<double, 3>& p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array range_p = dolfinx::MPI::local_range(dolfinx::MPI::rank(comm), n_points,
                                                 dolfinx::MPI::size(comm));

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);
  const double z0 = std::min(p0[2], p1[2]);
  const double z1 = std::max(p0[2], p1[2]);

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);
  const double e = z0;
  const double f = z1;
  const double ef = (f - e) / static_cast<double>(nz);

  if (std::abs(x0 - x1) < 2.0 * DBL_EPSILON or std::abs(y0 - y1) < 2.0 * DBL_EPSILON
      or std::abs(z0 - z1) < 2.0 * DBL_EPSILON) {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 || ny < 1 || nz < 1) {
    throw std::runtime_error(
        "BoxMesh has non-positive number of vertices in some dimension");
  }

  xt::xtensor<double, 2> geom({static_cast<std::size_t>(range_p[1] - range_p[0]), 3});
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  std::array<double, 3> point;
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v) {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const double z = e + ef * static_cast<double>(iz);
    const double y = c + cd * static_cast<double>(iy);
    const double x = a + ab * static_cast<double>(ix);
    point = {x, y, z};
    for (std::size_t i = 0; i < 3; i++)
      geom(v - range_p[0], i) = point[i];
  }

  return geom;
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> partition(MPI_Comm comm,
                                             xt::xtensor<std::size_t, 2>& cart_indices,
                                             xt::xtensor<std::size_t, 2>& cells_cart,
                                             xt::xtensor<int, 1>& P,
                                             xt::xtensor<int, 1>& N) {
  int rank = dolfinx::MPI::rank(comm);
  xt::xtensor<int, 1> Nx = N / P;

  xt::xtensor<int, 2> neighbors({3, 2});
  neighbors.fill(-1);
  xt::xtensor<int, 1> index = xt::row(cart_indices, rank);
  // Compute strides
  std::array<int, 3> strides = {P[1] * P[2], P[2], 1};
  xt::xtensor<int, 1> neighbor;
  for (int i = 0; i < 3; i++) {
    if (index(i) > 0) {
      xt::xtensor<int, 1> neighbor = xt::row(cart_indices, rank);
      neighbor[i] -= 1;
      neighbors(i, 0)
          = std::inner_product(neighbor.begin(), neighbor.end(), strides.begin(), 0);
    }
    if (index(i) < P(i) - 1) {
      xt::xtensor<int, 1> neighbor = xt::row(cart_indices, rank);
      neighbor(i) += 1;
      neighbors(i, 1)
          = std::inner_product(neighbor.begin(), neighbor.end(), strides.begin(), 0);
    }
  }

  xt::xtensor<int, 1> counter = xt::ones<int>({cells_cart.shape(0)});
  for (std::size_t c = 0; c < counter.size(); c++) {
    xt::xtensor<int, 1> cell = xt::row(cells_cart, c);
    for (int j = 0; j < 3; j++) {
      if (cell(j) == 0)
        if (neighbors(j, 0) >= 0)
          counter[c]++;
      if (cell(j) == Nx(j) - 1)
        if (neighbors(j, 1) >= 0)
          counter[c]++;
    }
  }

  std::vector<std::int32_t> offsets(counter.size() + 1);
  std::partial_sum(counter.begin(), counter.end(), offsets.begin() + 1);
  std::vector<std::int32_t> data(offsets.back(), rank);
  std::fill(counter.begin(), counter.end(), 1);

  for (std::size_t c = 0; c < counter.size(); c++) {
    xt::xtensor<int, 1> cell = xt::row(cells_cart, c);
    for (int j = 0; j < 3; j++) {
      if (cell(j) == 0) {
        if (neighbors(j, 0) >= 0) {
          int pos = offsets[c] + counter[c]++;
          data[pos] = neighbors(j, 0);
        }
      }
      if (cell(j) == Nx(j) - 1) {
        if (neighbors(j, 1) >= 0) {
          int pos = offsets[c] + counter[c]++;
          data[pos] = neighbors(j, 1);
        }
      }
    }
  }

  return graph::AdjacencyList<std::int32_t>(data, offsets);
}
} // namespace benchmark::utils

using namespace benchmark::utils;

namespace benchmark {
/// Create a uniform unit cube mesh.
/// The total number of cells E = 2^s
/// Expects number of processes to be a poer of two, and P <= E
mesh::Mesh create_hex_mesh(MPI_Comm comm, const int s) {
  common::Timer t0("~create mesh");
  common::Timer t1("~compute mesh data");

  std::uint32_t mpi_size = dolfinx::MPI::size(comm);
  int rank = dolfinx::MPI::rank(comm);

  // Compute number of cells in each direction N = [Nx, Ny, Nz]
  xt::xtensor<int, 1> N = decompose3d(s);
  std::array<std::array<double, 3>, 2> p = {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}};

  bool is_pow2 = is_power_of_two(mpi_size);
  if (!is_pow2) {
    LOG(WARNING)
        << "Number of processes is not a power of two, falling back to dolfinx mesh";
    std::array<std::size_t, 3> n
        = {std::size_t(N[0]), std::size_t(N[1]), std::size_t(N[2])};
    return mesh::create_box(comm, p, n, mesh::CellType::hexahedron,
                            mesh::GhostMode::none);
  }

  if (s < 3)
    throw std::runtime_error("\n s should be at least 3.");

  // Compute number of processes in each direction P = [Px, Py, Pz]
  int t = log2(mpi_size);
  xt::xtensor<int, 1> P = decompose3d(t);

  // Compute MPI process topology
  auto [graph, cart_indices] = compute_cartesian_topology(comm, P);

  // Number of cells per process in each direction
  xt::xtensor<int, 1> Nx = N / P;

  // Compute cell indices (cartesian) and destination ranks
  xt::xtensor<std::size_t, 2> cells_cart = compute_cartesian_indices(Nx);
  auto dest = partition(comm, cart_indices, cells_cart, P, N);

  // Compute topology
  std::size_t nc = xt::prod<int>(Nx, 0)[0];
  xt::xtensor<std::int64_t, 2> cells({nc, 8});
  xt::xtensor<std::size_t, 1> cell_offset = xt::row(cart_indices, rank) * Nx;
  cells_cart = cells_cart + cell_offset;
  for (std::size_t i = 0; i < nc; i++) {
    auto cell_idx = xt::row(cells_cart, i);
    const std::int64_t ix = cell_idx(0);
    const std::int64_t iy = cell_idx(1);
    const std::int64_t iz = cell_idx(2);

    const std::int64_t v0 = (iz * (N[1] + 1) + iy) * (N[0] + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (N[0] + 1);
    const std::int64_t v3 = v1 + (N[0] + 1);
    const std::int64_t v4 = v0 + (N[0] + 1) * (N[1] + 1);
    const std::int64_t v5 = v1 + (N[0] + 1) * (N[1] + 1);
    const std::int64_t v6 = v2 + (N[0] + 1) * (N[1] + 1);
    const std::int64_t v7 = v3 + (N[0] + 1) * (N[1] + 1);

    xt::xtensor<std::int64_t, 1> cell = {v0, v1, v2, v3, v4, v5, v6, v7};

    xt::view(cells, i, xt::all()) = cell;
  }

  fem::CoordinateElement element(mesh::CellType::hexahedron, 1);
  auto [data, offset] = graph::create_adjacency_data(cells);

  auto geom = create_geom(comm, p, N);

  auto partitioner = [=](MPI_Comm mpi_comm, int nparts, int tdim,
                         const graph::AdjacencyList<std::int64_t>& cells,
                         mesh::GhostMode ghost_mode) { return dest; };

  t1.stop();
  return mesh::create_mesh(
      comm, graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, mesh::GhostMode::none, partitioner);
}
} // namespace benchmark