// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include "precompute.hpp"

#include <dolfinx.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/Vector.h>
#include <memory>
#include <petscmat.h>
#include <xtensor/xtensor.hpp>

#ifdef USE_LIBXSMM
#include <libxsmm_source.h>
template <typename T>
using kernel_type = libxsmm_mmfunction<T>;
#endif

using namespace dolfinx;

//--------------------------------------------------------------------------------//
template <typename T>
class MatFreeOperator {
public:
  MatFreeOperator(std::shared_ptr<fem::Form<T>> L) : _form{L}, _coefficients{} {
    dolfinx::common::Timer t0("~setup phase");

    if (_form->rank() != 1)
      throw std::runtime_error("Matrirx Free operator rank should be 1.");

    int tdim = _form->mesh()->topology().dim();
    std::int32_t num_cells = _form->mesh()->topology().index_map(tdim)->size_local();
    std::int32_t ndofs_cell = _form->function_spaces()[0]->dofmap()->cell_dofs(0).size();
    _coeffs = xt::zeros<T>({num_cells, ndofs_cell});
    _coefficients[{dolfinx::fem::IntegralType::cell, -1}]
        = {xtl::span<T>(_coeffs.data(), _coeffs.size()), _coeffs.shape(1)};
  }

  // Compute y = Ax with matrix-free operator
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    xtl::span<const T> x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();
    auto& dofmap = _form->function_spaces()[0]->dofmap()->list();

    for (std::size_t cell = 0; cell < _coeffs.shape(0); ++cell) {
      auto cell_dofs = dofmap.links(cell);
      for (std::size_t i = 0; i < _coeffs.shape(1); i++) {
        _coeffs(cell, i) = x_array[cell_dofs[i]];
      }
    }
    dolfinx::fem::assemble_vector<T>(y_array, *_form, tcb::make_span(constants),
                                     _coefficients);
  }

private:
  std::shared_ptr<fem::Form<T>> _form;
  std::map<std::pair<dolfinx::fem::IntegralType, int>, std::pair<xtl::span<const T>, int>>
      _coefficients;
  xt::xtensor<T, 2> _coeffs;
  std::vector<T> constants;
};

//--------------------------------------------------------------------------------//
template <typename T>
class PETScOperator {
public:
  PETScOperator(std::shared_ptr<fem::Form<T>> a,
                const std::vector<std::shared_ptr<const fem::DirichletBC<T>>>& bcs)
      : _mat(la::petsc::Matrix(fem::petsc::create_matrix(*a), false)) {
    dolfinx::common::Timer t0("~setup phase");

    static_assert(std::is_same<T, double>(), "Type mismatch");

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    MatZeroEntries(_mat.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(_mat.mat(), ADD_VALUES), *a,
                         bcs);
    MatAssemblyBegin(_mat.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(_mat.mat(), MAT_FINAL_ASSEMBLY);

    // Copy ghosts to a different vector with PETSC_INT data
    std::shared_ptr<const fem::FunctionSpace> V = a->function_spaces()[0];
    auto&& dof_ghosts = V->dofmap()->index_map->ghosts();
    _ghosts.resize(dof_ghosts.size());
    std::copy(dof_ghosts.begin(), dof_ghosts.end(), _ghosts.begin());

    _comm = V->mesh()->comm();
  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {

    const PetscInt local_size = x.map()->size_local();
    const PetscInt global_size = x.map()->size_global();
    const PetscInt num_ghosts = _ghosts.size();
    const PetscInt* ghosts = _ghosts.data();

    // Creates a parallel vector with ghost padding on each processor
    VecCreateGhostWithArray(_comm, local_size, global_size, num_ghosts, ghosts,
                            x.array().data(), &_x_petsc);
    VecCreateGhostWithArray(_comm, local_size, global_size, num_ghosts, ghosts,
                            y.mutable_array().data(), &_y_petsc);

    // Actual matrix vector multiplication
    MatMult(_mat.mat(), _x_petsc, _y_petsc);
  }

private:
  std::vector<PetscInt> _ghosts;
  Vec _x_petsc = nullptr;
  Vec _y_petsc = nullptr;
  la::petsc::Matrix _mat;
  MPI_Comm _comm;
};

// --------------------------------------------------------------------------------//
template <typename T>
class EAOperator {
public:
  EAOperator(std::shared_ptr<fem::Form<T>> a,
             const std::vector<std::shared_ptr<const fem::DirichletBC<T>>>& bcs) {
    dolfinx::common::Timer t0("~setup phase");

    _element_tensor = assemble_element_tensor(a, bcs);

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    std::shared_ptr<const fem::FunctionSpace> V = a->function_spaces()[0];
    auto&& dofarray = V->dofmap()->list().array();
    int ndofs = _element_tensor.shape(1);
    int ncells = _element_tensor.shape(0);
    _xi.resize(ndofs, 0);
    _yi.resize(ndofs, 0);

#ifdef USE_LIBXSMM
    libxsmm_init();
    kernel = kernel_type<T>(LIBXSMM_GEMM_FLAG_NONE, ndofs, 1, ndofs, 1.0, 0.0);
#endif

    _dof_array = xt::zeros<std::int32_t>({ncells, ndofs});
    _dof_array.assign(xt::adapt(dofarray, {ncells, ndofs}));
  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    const auto& x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();

    // Zero output vector
    std::fill(y_array.begin(), y_array.end(), 0);

    // Allocate temporary data
    int ndofs = _element_tensor.shape(1);

    T* data = _element_tensor.data();

    // Compute Ae*be = xe at the element level
    int ncells = _element_tensor.shape(0);
    for (int c = 0; c < ncells; c++) {
      for (int i = 0; i < ndofs; i++)
        _xi[i] = x_array[_dof_array(c, i)];

      // xi = Ae * bi
      int offset = ndofs * ndofs;
      T* ae = data + c * offset;

#ifdef USE_LIBXSMM
      kernel(ae, _xi.data(), _yi.data());
#else
      std::fill_n(_yi.data(), ndofs, 0);
      for (int i = 0; i < ndofs; i++)
        for (int j = 0; j < ndofs; j++)
          _yi[i] += ae[i * ndofs + j] * _xi[j];
#endif

      for (int i = 0; i < ndofs; i++)
        y_array[_dof_array(c, i)] += _yi[i];
    }
  }

private:
  xt::xtensor<T, 3> _element_tensor;
  xt::xtensor<std::int32_t, 2> _dof_array;
  std::vector<T> _xi;
  std::vector<T> _yi;

#ifdef USE_LIBXSMM
  kernel_type<T> kernel;
#endif
};
// --------------------------------------------------------------------------------//