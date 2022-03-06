// Copyright (C) 2022 Adeeb Arif Kor
// SPDX-License-Identifier:   MIT

#pragma once

#include "permute.hpp"
#include "precompute.hpp"
#include "transform.hpp"

#include <vector>

#include <basix/finite-element.h>
#include <basix/quadrature.h>

// Helper functions
#include <cuda/allocator.hpp>
#include <cuda/array.hpp>
#include <cuda/scatter.hpp>
#include <cuda/transform.hpp>
#include <cuda/utils.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------//
template <typename T>
class SpectralMassOperator {
public:
  SpectralMassOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree) {
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();

    dolfinx::common::Timer t0("~setup phase");

    int tdim = mesh->topology().dim();
    _num_cells = mesh->topology().index_map(tdim)->size_local();
    _num_dofs = (bdegree + 1) * (bdegree + 1) * (bdegree + 1);

    auto _dofmap = V->dofmap()->list().array();
    std::vector<int> _perm_dofmap(_dofmap.size());
    reorder_dofmap(_perm_dofmap, _dofmap, bdegree); // get tensor product dofmap
    perm_dofmap = std::make_unique<cuda::array<std::int32_t>>(_perm_dofmap.size());
    perm_dofmap->set(_perm_dofmap);

    // Create a map between basis degree and quadrature degree
    std::map<int, int> qdegree;
    qdegree[2] = 3;
    qdegree[3] = 4;
    qdegree[4] = 6;
    qdegree[5] = 8;
    qdegree[6] = 10;
    qdegree[7] = 12;

    // Tabulate quadrature points and weights
    auto cell_type = basix::cell::type::hexahedron;
    auto quad_type = basix::quadrature::type::gll;
    auto [points, weights]
        = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[bdegree]);
    _num_quads = weights.size();

    // Get the determinant of the Jacobian
    xt::xtensor<double, 4> J = compute_jacobian(mesh, points);
    xt::xtensor<T, 2> _detJ = compute_jacobian_determinant(J);
    for (std::size_t i = 0; i < _detJ.shape(0); i++) {
      for (std::size_t j = 0; j < _detJ.shape(1); j++) {
        _detJ(i, j) = _detJ(i, j) * weights[j];
      }
    }

    detJ = std::make_unique<cuda::array<T>>(_detJ.size());
    detJ->set(_detJ);

    // Create buffer
    std::size_t Ne = _num_cells * _num_dofs;
    xe = std::make_unique<cuda::array<T>>(Ne);
    ye = std::make_unique<cuda::array<T>>(Ne);
  }

  ~SpectralMassOperator() = default;

  std::size_t num_quads() const { return _num_quads; }
  std::size_t num_cells() const { return _num_cells; };
  std::size_t num_dofs() const { return _num_dofs; }
  double flops() const { return 4 * _num_cells * _num_quads * _num_dofs; };

  /// Compute y = Ax for diagonal mass matrix A
  template <typename Vector>
  void apply(const Vector& x, Vector& y) {

    gather(perm_dofmap->size(), perm_dofmap->data(), x.array().data(), xe->data(), 512);
    transform1(perm_dofmap->size(), xe->data(), detJ->data(), xe->data(), 512);
    scatter(perm_dofmap->size(), perm_dofmap->data(), xe->data(), y.mutable_array().data(), 512);
  }

private:
  std::unique_ptr<cuda::array<T>> detJ;
  std::unique_ptr<cuda::array<std::int32_t>> perm_dofmap;
  std::unique_ptr<cuda::array<T>> xe;
  std::unique_ptr<cuda::array<T>> ye;

  std::size_t _num_cells;
  std::size_t _num_dofs;
  std::size_t _num_quads;
};