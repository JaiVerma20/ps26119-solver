// simplex_scaling.cpp - see simplex_scaling.h. Origin: gpuopt src/simplex/scaling.cpp (c192dd0).
#include "simplex/simplex_scaling.h"

#include <algorithm>
#include <cmath>

namespace ps26119::simplex {
namespace {

double power_of_two(double v) { return std::exp2(std::round(std::log2(v))); }

}  // namespace

Scaling compute_scaling(const la::SparseMatrixCSC& A, int passes) {
  const int m = A.num_rows, n = A.num_cols;
  Scaling s;
  s.row.assign(m, 1.0);
  s.col.assign(n, 1.0);
  std::vector<double> lo(std::max(m, n)), hi(std::max(m, n));

  for (int pass = 0; pass < passes; ++pass) {
    // Rows: r_i = 1 / sqrt(min_j |a_ij c_j| * max_j |a_ij c_j|)
    std::fill(lo.begin(), lo.begin() + m, INFINITY);
    std::fill(hi.begin(), hi.begin() + m, 0.0);
    for (int j = 0; j < n; ++j) {
      for (int k = A.col_start[j]; k < A.col_start[j + 1]; ++k) {
        const double v = std::fabs(A.value[k]) * s.col[j];
        if (v == 0.0) continue;  // explicit zeros (C API, .lpm, presolve) carry no scale
        const int i = A.row_index[k];
        lo[i] = std::min(lo[i], v);
        hi[i] = std::max(hi[i], v);
      }
    }
    for (int i = 0; i < m; ++i) {
      if (hi[i] > 0.0) s.row[i] = 1.0 / std::sqrt(lo[i] * hi[i]);
    }
    // Columns: c_j = 1 / sqrt(min_i |r_i a_ij| * max_i |r_i a_ij|)
    for (int j = 0; j < n; ++j) {
      double l = INFINITY, h = 0.0;
      for (int k = A.col_start[j]; k < A.col_start[j + 1]; ++k) {
        const double v = std::fabs(A.value[k]) * s.row[A.row_index[k]];
        if (v == 0.0) continue;
        l = std::min(l, v);
        h = std::max(h, v);
      }
      if (h > 0.0) s.col[j] = 1.0 / std::sqrt(l * h);
    }
  }

  // Round rows to powers of two, then equilibrate columns (max entry ~ 1).
  for (double& r : s.row) r = power_of_two(r);
  for (int j = 0; j < n; ++j) {
    double h = 0.0;
    for (int k = A.col_start[j]; k < A.col_start[j + 1]; ++k) {
      h = std::max(h, std::fabs(A.value[k]) * s.row[A.row_index[k]]);
    }
    s.col[j] = h > 0.0 ? power_of_two(1.0 / h) : 1.0;
  }
  return s;
}

double scaled_dynamism(const la::SparseMatrixCSC& A, const Scaling& s) {
  double lo = INFINITY, hi = 0.0;
  for (int j = 0; j < A.num_cols; ++j) {
    for (int k = A.col_start[j]; k < A.col_start[j + 1]; ++k) {
      const double v = std::fabs(A.value[k]) * s.row[A.row_index[k]] * s.col[j];
      if (v == 0.0) continue;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  return hi > 0.0 ? hi / lo : 1.0;
}

}  // namespace ps26119::simplex
