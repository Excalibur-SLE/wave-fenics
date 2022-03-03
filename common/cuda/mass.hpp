// Copyright (C) 2022 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/finite-element.h>
#include <cstdint>
#include <cuda/array.hpp>

#include "precompute.hpp"

using namespace dolfinx;

//-------------------------------------------------------//
template <typename T>
class MassOperator {
public:
  MassOperator(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
               basix::FiniteElement element, basix::quadrature::type quad_type, int qd) {

    dolfinx::common::Timer t0("~setup phase");
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    std::size_t ncells = mesh->topology().index_map(tdim)->size_local();

    auto cell = element.cell_type();

    // Tabulate quadrature points and weights
    auto [points, weights] = basix::quadrature::make_quadrature(quad_type, cell, qd);

    // Compute determinant of jacobian at quadrature points
    xt::xtensor<double, 4> J = compute_jacobian(mesh, points);
    xt::xtensor<T, 2> _detJ = compute_jacobian_determinant(J);
    for (std::size_t i = 0; i < _detJ.shape(0); i++)
      for (std::size_t j = 0; j < _detJ.shape(1); j++)
        _detJ(i, j) = _detJ(i, j) * weights[j];

    detJ = std::make_unique<cuda::array<T>>(_detJ.size());
    detJ->set(_detJ);

    // Tabulate basis functions
    xt::xtensor<double, 4> basis = element.tabulate(0, points);
    xt::xtensor<T, 2> _phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
    phi = std::make_unique<cuda::array<T>>(_phi.size());
    phi->set(_phi);

    // Create a buffer with the dofmap
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
    const std::vector<std::int32_t>& dof_array = dofmap->list().array();
    assert(ncells == dofmap->list().num_nodes());
    assert(_phi.shape(1) == size_t(dofmap->list().num_links(0)));

    dofarray = std::make_unique<cuda::array<std::int32_t>>(dof_array.size());
    dofarray->set(dof_array);

    int nquads = weights.size();
    int ndofs = element.dim();
    std::size_t Ne = ncells * ndofs;
    xe = std::make_unique<cuda::array<T>>(Ne);
    ye = std::make_unique<cuda::array<T>>(Ne);
  }

  ~MassOperator() = default;

private:
  std::unique_ptr<cuda::array<T>> detJ;
  std::unique_ptr<cuda::array<T>> phi;
  std::unique_ptr<cuda::array<std::int32_t>> dofarray;
  std::unique_ptr<cuda::array<T>> xe;
  std::unique_ptr<cuda::array<T>> ye;
};
