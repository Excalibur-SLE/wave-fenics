#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;

/// Compute geometric data
/// @param[in] mesh
/// @param[in] p degree of basis function
/// @param[out] G geometrical factor
/// @param[out] detJ determinant of Jacobian
std::pair<xt::xtensor<double, 4>, xt::xtensor<double, 2>>
precompute_geometric_data(std::shared_ptr<const mesh::Mesh> mesh, int p) {
  // Get geometrical and topological data
  const mesh::Geometry& geometry = mesh->geometry();
  const mesh::Topology& topology = mesh->topology();
  const fem::CoordinateElement& cmap = geometry.cmap();

  const std::size_t tdim = topology.dim();
  const std::size_t gdim = geometry.dim();
  const std::size_t ncells = mesh->topology().index_map(tdim)->size_local();

  const xt::xtensor<double, 2> x
      = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(),
                  std::vector{geometry.x().size() / 3, std::size_t(3)});
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_nodes = x_dofmap.num_links(0);

  // Create map between basis degree and quadrature degree
  std::map<int, int> qd;
  qd[2] = 3;
  qd[3] = 4;
  qd[4] = 6;
  qd[5] = 8;
  qd[6] = 10;
  qd[7] = 12;
  qd[8] = 14;
  qd[9] = 16;
  qd[10] = 18;

  // Tabulate quadrature points and weights
  auto cell = basix::cell::type::hexahedron;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, qd[p]);
  const std::size_t nq = weights.size();

  // Tabulate coordinate map basis functions and clamp -1, 0, 1 values
  xt::xtensor<double, 4> table = cmap.tabulate(1, points);
  xt::filtration(table, xt::isclose(table, -1.0)) = -1.0;
  xt::filtration(table, xt::isclose(table, 0.0)) = 0.0;
  xt::filtration(table, xt::isclose(table, 1.0)) = 1.0;
  xt::xtensor<double, 2> phi = xt::view(table, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi = xt::view(table, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  // Create placeholder for geometrical data
  xt::xtensor<double, 4> G({ncells, nq, tdim, gdim});
  xt::xtensor<double, 2> J({tdim, gdim}), J_inv({tdim, gdim});
  xt::xtensor<double, 2> detJ({ncells, nq});
  xt::xtensor<double, 2> coords({num_nodes, gdim});

  // Compute geometrical data for each quadrature point
  tcb::span<const int> x_dofs;
  for (std::size_t c = 0; c < ncells; c++) {
    // Get cell coordinates/geometry
    x_dofs = x_dofmap.links(c);

    // Copying x to coords
    for (std::size_t i = 0; i < x_dofs.size(); i++) {
      std::copy_n(xt::row(x, x_dofs[i]).begin(), 3, xt::row(coords, i).begin());
    }

    // Computing geometrical factor G
    J.fill(0.0);
    J_inv.fill(0.0);
    for (std::size_t q = 0; q < nq; q++) {
      // Computing the entries of the Jacobian matrix J
      xt::view(J, 0, 0) = xt::sum(xt::view(coords, xt::all(), 0) * xt::view(dphi, 0, q, xt::all()));
      xt::view(J, 0, 1) = xt::sum(xt::view(coords, xt::all(), 0) * xt::view(dphi, 1, q, xt::all()));
      xt::view(J, 0, 2) = xt::sum(xt::view(coords, xt::all(), 0) * xt::view(dphi, 2, q, xt::all()));
      xt::view(J, 1, 0) = xt::sum(xt::view(coords, xt::all(), 1) * xt::view(dphi, 0, q, xt::all()));
      xt::view(J, 1, 1) = xt::sum(xt::view(coords, xt::all(), 1) * xt::view(dphi, 1, q, xt::all()));
      xt::view(J, 1, 2) = xt::sum(xt::view(coords, xt::all(), 1) * xt::view(dphi, 2, q, xt::all()));
      xt::view(J, 2, 0) = xt::sum(xt::view(coords, xt::all(), 2) * xt::view(dphi, 0, q, xt::all()));
      xt::view(J, 2, 1) = xt::sum(xt::view(coords, xt::all(), 2) * xt::view(dphi, 1, q, xt::all()));
      xt::view(J, 2, 2) = xt::sum(xt::view(coords, xt::all(), 2) * xt::view(dphi, 2, q, xt::all()));

      // Computing the absolute value of the determinant of the Jacobian
      // matrix |J| scaled by the quadrature weights
      detJ(c, q) = std::fabs(dolfinx::math::det(J)) * weights[q];
      dolfinx::math::inv(J, J_inv);

      // Computing the geometrical factor G = J^{-1} * J^{-T} * |J|
      dolfinx::math::dot(J_inv * detJ(c, q), xt::transpose(J_inv),
                         xt::view(G, c, q, xt::all(), xt::all()));
    }
  }

  // Clamp -1, 0, 1 values
  xt::filtration(G, xt::isclose(G, -1.0)) = -1.0;
  xt::filtration(G, xt::isclose(G, 0.0)) = 0.0;
  xt::filtration(G, xt::isclose(G, 1.0)) = 1.0;

  return {G, detJ};
}