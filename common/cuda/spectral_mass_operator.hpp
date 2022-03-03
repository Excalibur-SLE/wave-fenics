#pragma once

#include "permute.hpp"
#include "precomputation.hpp"
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

template <typename T>
class MassOperator {
private:
  std::int32_t _ncells, _ndofs, _nquads, Nq, Ne;
  std::int32_t _dofmap_size;
  std::vector<int> _perm_dofmap;
  xt::xtensor<double, 2> _detJ, _phi;

public:
  MassOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree) : _perm_dofmap(0) {
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    auto _dofmap = V->dofmap()->list().array();
    _dofmap_size = _dofmap.size();
    std::cout << _dofmap_size << std::endl;
    _perm_dofmap.resize(_dofmap_size);
    reorder_dofmap(_perm_dofmap, _dofmap, bdegree); // get tensor product dofmap
    _ncells = mesh->topology().index_map(tdim)->size_local();
    _ndofs = (bdegree + 1) * (bdegree + 1) * (bdegree + 1);
    Ne = _ncells * _ndofs;
    std::cout << Ne << std::endl;

    // Create a map between basis degree and quadrature degree
    std::map<int, int> qdegree;
    qdegree[2] = 3;
    qdegree[3] = 4;
    qdegree[4] = 6;
    qdegree[5] = 8;
    qdegree[6] = 10;
    qdegree[7] = 12;

    // Get the determinant of the Jacobian
    auto Jdata = precompute_geometric_data(mesh, bdegree);
    _detJ = std::get<1>(Jdata);

    // Tabulate basis
    auto family = basix::element::family::P;
    auto cell = basix::cell::type::hexahedron;
    auto quad = basix::quadrature::type::gll;
    auto variant = basix::element::lagrange_variant::gll_warped;
    auto [qpoints, qweights]
        = basix::quadrature::make_quadrature(quad, cell, qdegree[bdegree]);
    auto element = basix::create_element(family, cell, bdegree, variant);
    auto table = element.tabulate(1, qpoints);
    _phi = xt::view(table, 0, xt::all(), xt::all(), 0);
    _nquads = qweights.size();
    Nq = _ncells * _nquads;

    // Clamp -1, 0, 1 values
    xt::filtration(_phi, xt::isclose(_phi, -1.0)) = -1.0;
    xt::filtration(_phi, xt::isclose(_phi, 0.0)) = 0.0;
    xt::filtration(_phi, xt::isclose(_phi, 1.0)) = 1.0;
  }

  template <typename Allocator>
  void operator()(const la::Vector<T, Allocator>& x, la::Vector<T, Allocator>& y) {
    // Allocate to device
    cuda::array<std::int32_t> perm_dofmap(Ne);
    perm_dofmap.set(_perm_dofmap);

    // cuda::array<double> phi(_ndofs * _nquads);
    // phi.set(_phi);

    cuda::array<double> detJ(Nq);
    detJ.set(_detJ);
    cuda::array<double> x_array(_ncells * _ndofs);

    std::cout << "Operates!!!" << std::endl;

    // gather operator
    gather(x_array.size(), perm_dofmap.data(), x.array().data(), x_array.data(), 512);

    // transform operator
    transform1(Ne, x_array.data(), detJ.data(), x_array.data(), 512);

    // scatter operator
    scatter(x_array.size(), perm_dofmap.data(), x_array.data(), y.mutable_array().data(),
            512);

    std::cout << "Complete!!!" << std::endl;
  }
};