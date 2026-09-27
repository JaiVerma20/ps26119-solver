// safe_bound.h — a certified (rounding-error-proof) bound on the optimal objective from ANY
// row multipliers y (CLAUDE.md §9 idea 5: certified output from a first-order iterate).
//
// Citation: A. Neumaier, O. Shcherbina, "Safe bounds in linear and mixed-integer linear
// programming", Math. Prog. 99 (2004) — the construction below is theirs; our code.
//
// Weak duality for min c̃ᵀx s.t. rl ≤ Ax ≤ ru, l ≤ x ≤ u (min form, c̃ = sense·c): for any y
// and z = c̃ − Aᵀy, every feasible x satisfies
//     c̃ᵀx = yᵀ(Ax) + zᵀx ≥ Σ_i min_{s∈[rl_i,ru_i]} y_i s + Σ_j min_{x_j∈[l_j,u_j]} z_j x_j  =: L(y).
// L(y) needs no dual feasibility; a term is −∞ only when y_i or z_j "points at" an infinite
// bound. Rigour: z_j is computed in double-double with exact products (two_prod) and an
// explicit bound on the accumulated rounding error, giving an interval [z_lo, z_hi]; the term
// min over z ∈ [z_lo,z_hi] is attained at an endpoint (the inner min is concave in z). All
// sums are carried in double-double with error bounds and the result is rounded outward.
// Multipliers pointing at an infinite row bound are set to 0 first (valid for any y).
// Columns bounded on one side only would make almost every bound −∞ (a basic column's
// reduced cost is 0 ± rounding); for those we use rigorous implied bounds from the
// constraints (implied_bounds.h), computed only when needed.
// Invariant: for MIN models `bound` ≤ true optimum; for MAX models `bound` ≥ true optimum
// (both include obj_offset). Only an infeasible LP has no finite valid bound… or a bound of
// ∓∞ is returned when a term is unbounded (reported, never faked).
#pragma once

#include <vector>

#include "ps26119/model.h"

namespace ps26119 {

struct SafeBound {
  bool finite = false;
  double bound = 0.0;          // see invariant above; ±inf when not finite
  int unbounded_row_terms = 0;  // rows whose multiplier points at an infinite bound
  int unbounded_col_terms = 0;  // columns whose reduced-cost interval points at an infinite bound
};

// y: row multipliers in the ORIGINAL objective sense (same convention as Solution::y).
// use_implied_bounds = false skips the bound-propagation refinement (O(passes·nnz)); the
// result is still rigorous, only possibly −∞. Used when a time limit is already exhausted.
SafeBound certified_dual_bound(const Model& model, const std::vector<double>& y,
                               bool use_implied_bounds = true);

}  // namespace ps26119
