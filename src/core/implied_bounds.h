// implied_bounds.h — rigorous variable bounds implied by the constraints (feasibility-based
// bound tightening, FBBT), rounded OUTWARD so that every feasible x satisfies them exactly.
//
// For a row with a finite upper bound ru_i and a_ij > 0:
//     x_j ≤ (ru_i − Σ_{k≠j} min_{x_k∈[l_k,u_k]} a_ik x_k) / a_ij
// (and symmetric cases for a_ij < 0 and for the lower bound rl_i), provided every other
// contribution is finite. Passes repeat while bounds improve noticeably.
//
// Rounding: activities are accumulated in double-double; each new bound is loosened by an
// absolute slack 1e-9·(Σ|terms|/|a_ij| + |bound|) — many orders of magnitude above the
// double-double rounding error — before being rounded outward. Looser is still valid.
//
// Used by the certified dual bound (safe_bound.h): a column that is bounded only on one side
// can still get a finite certified term once it has a finite implied bound. Also the first
// building block of a presolve.
// Citation: e.g. T. Achterberg et al., "Presolve reductions in mixed integer programming",
// INFORMS J. Comput. 32 (2020), §3 (bound strengthening). Our code.
#pragma once

#include <vector>

#include "ps26119/model.h"

namespace ps26119 {

struct ImpliedBounds {
  std::vector<double> lower, upper;  // never tighter than any feasible point allows
  int tightened = 0;                 // number of infinite model bounds made finite
};

ImpliedBounds implied_bounds(const Model& model, int max_passes = 20);

}  // namespace ps26119
