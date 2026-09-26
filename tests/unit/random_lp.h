// random_lp.h - random LP generator shared by the stress and differential tests.
//
// Every kind of row (E, L, G, ranged) and bound (default, boxed, MI, free,
// fixed, lower-only). 70% of the models are built around a point x0 inside
// the bounds, so they are feasible by construction.
#pragma once
// Origin: gpuopt tests/random_lp.hpp (Shivanshu Vats, c192dd0).

#include <random>
#include <string>
#include <vector>

#include "la/csc.h"

namespace ps26119::testing {

using la::build_csc;
using la::Triplet;

// Puts a CSC matrix into a Model (the gpuopt LpProblem held it as a member).
inline void set_matrix(Model& lp, const la::SparseMatrixCSC& A) {
  lp.num_rows = A.num_rows;
  lp.num_cols = A.num_cols;
  lp.col_start = A.col_start;
  lp.row_index = A.row_index;
  lp.value = A.value;
}

struct RandomLp {
  Model lp;
  bool known_feasible = false;
  bool all_boxed = true;
};

inline RandomLp random_lp(std::mt19937& rng, int max_rows = 8, int max_cols = 10, double density = 0.5) {
  std::uniform_int_distribution<int> dim_m(1, max_rows), dim_n(1, max_cols), coef(-5, 5), kind(0, 5);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  RandomLp out;
  Model& lp = out.lp;
  const int m = dim_m(rng);
  const int n = dim_n(rng);
  lp.sense = unit(rng) < 0.5 ? 1 : -1;
  lp.obj_offset = coef(rng);

  std::vector<Triplet> t;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (unit(rng) < density) {
        const int v = coef(rng);
        if (v != 0) t.push_back({i, j, static_cast<double>(v)});
      }
    }
  }
  set_matrix(lp, build_csc(m, n, t));
  lp.is_integer.assign(n, 0);
  for (int j = 0; j < n; ++j) lp.col_names.push_back("c" + std::to_string(j));
  for (int i = 0; i < m; ++i) lp.row_names.push_back("r" + std::to_string(i));

  // Column bounds and a reference point x0 inside them.
  std::vector<double> x0(n);
  for (int j = 0; j < n; ++j) {
    lp.obj.push_back(coef(rng));
    const double base = coef(rng);
    const double width = 1 + coef(rng) + 5;  // 1..11
    switch (kind(rng)) {
      case 0: lp.col_lower.push_back(0); lp.col_upper.push_back(kInf); x0[j] = unit(rng) * 3; break;
      case 1: lp.col_lower.push_back(base); lp.col_upper.push_back(base + width); x0[j] = base + unit(rng) * width; break;
      case 2: lp.col_lower.push_back(-kInf); lp.col_upper.push_back(base); x0[j] = base - unit(rng) * 3; break;
      case 3: lp.col_lower.push_back(-kInf); lp.col_upper.push_back(kInf); x0[j] = base; break;
      case 4: lp.col_lower.push_back(base); lp.col_upper.push_back(base); x0[j] = base; break;
      default: lp.col_lower.push_back(base); lp.col_upper.push_back(kInf); x0[j] = base + unit(rng) * 3; break;
    }
    if (lp.col_lower[j] == -kInf || lp.col_upper[j] == kInf) out.all_boxed = false;
  }

  // Row bounds: around A x0 (feasible by construction) or fully random.
  out.known_feasible = unit(rng) < 0.7;
  std::vector<double> ax0;
  ax0 = lp.row_activity(x0);
  for (int i = 0; i < m; ++i) {
    const double centre = out.known_feasible ? ax0[i] : coef(rng) * 2.0;
    const double slack = out.known_feasible ? unit(rng) * 4 : 0.0;
    switch (kind(rng)) {
      case 0: lp.row_lower.push_back(centre); lp.row_upper.push_back(centre); break;          // E
      case 1: lp.row_lower.push_back(-kInf); lp.row_upper.push_back(centre + slack); break;   // L
      case 2: lp.row_lower.push_back(centre - slack); lp.row_upper.push_back(kInf); break;    // G
      case 3: lp.row_lower.push_back(centre - slack); lp.row_upper.push_back(centre + slack + 1); break;  // ranged
      default: lp.row_lower.push_back(-kInf); lp.row_upper.push_back(centre + slack); break;  // L
    }
  }
  return out;
}

}  // namespace ps26119::testing
