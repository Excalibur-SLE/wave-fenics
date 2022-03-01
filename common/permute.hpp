#pragma once

#include <vector>

#include <basix/finite-element.h>

/// Reorder dofmap into tensor product order
/// @param[in] in_arr Input dofmap
/// @param[in] p Degree of basis function
/// @param[out] out_arr Output dofmap
void reorder_dofmap(std::vector<int>& out_arr, std::vector<int>& in_arr, int& p) {
  // Tabulate quadrature points and weights
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::hexahedron;
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);

  std::vector<int> perm = std::get<1>(element.get_tensor_product_representation()[0]);
  int ncells = in_arr.size() / perm.size();

  int idx = 0;
  for (int c = 1; c < ncells + 1; c++) {
    for (auto& i : perm) {
      out_arr[idx] = in_arr[c * i];
      idx++;
    }
  }
}