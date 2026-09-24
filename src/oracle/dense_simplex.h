// dense_simplex.h — dense bounded-variable two-phase primal simplex in double-double.
//
// TEST ORACLE ONLY. Simple, slow, correct: dense tableau, Bland's rule, double-double
// (≈32 digits) arithmetic, periodic refactorisation from the original data. Use it to
// check other engines on models up to a few hundred rows.
//
// Formulation (see dense_simplex.cpp for the full derivation):
//   variables  x_j (structural, n), s_i (logical, m), r_i (artificial, m)
//   rows       a_iᵀx − s_i + d_i r_i = 0,   d_i ∈ {−1, +1}
//   bounds     l_j ≤ x_j ≤ u_j,  rl_i ≤ s_i ≤ ru_i,  0 ≤ r_i (≤ 0 in phase 2)
//   phase 1    minimise Σ r_i          phase 2  minimise sense·cᵀx
//
// Output follows include/ps26119/solution.h (z = c − Aᵀy, HiGHS sign convention).
#pragma once

#include <cstdint>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::oracle {

struct DenseSimplexOptions {
  std::int64_t iteration_limit = 10'000'000;
  double time_limit = 3600.0;  // seconds
  int refactor_every = 100;    // pivots between refactorisations from the original data
  int verbosity = 0;
  // Refuse models whose dense tableau would exceed this many entries (32 bytes each).
  std::int64_t max_tableau_entries = 8'000'000;
};

Solution solve_dense_simplex(const Model& model, const DenseSimplexOptions& options = {});

}  // namespace ps26119::oracle
