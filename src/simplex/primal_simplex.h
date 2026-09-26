// primal_simplex.h - Layer 3: bounded revised primal simplex.
//
// Origin: gpuopt include/gpuopt/simplex/primal_simplex.hpp (Shivanshu Vats,
// shivanshu24-code/gpu_optimization@c192dd0). Algorithm unchanged; it now reads the
// canonical Model and returns the canonical Solution (engine "simplex").
// References: Maros, Computational Techniques of the Simplex Method (2003);
// Forrest & Goldfarb, Math. Prog. 57 (1992) (Devex); Harris, Math. Prog. 5 (1973);
// Bland, Math. Oper. Res. 2 (1977).
//
// Solves  min/max c^T x  s.t.  L <= A x <= U,  l <= x <= u  using the sparse
// LU of Layer 2. Every row gets a logical (slack) variable, so the problem
// becomes [A I] z = 0 with bounds on every z_j, and the initial basis is the
// identity. Main ingredients:
//
//   * composite phase 1: while some basic variable violates a bound, the
//     simplex minimises the sum of infeasibilities; once feasible it
//     switches to the real objective (phase 2). Feasibility lost to round-off
//     is repaired the same way;
//   * Devex pricing (approximate steepest edge) or Dantzig pricing;
//   * Harris two-pass ratio test: allows bound violations up to the primal
//     tolerance in order to pivot on a larger, safer |alpha|;
//   * bound flips when the entering variable reaches its own opposite bound;
//   * cost perturbation against degeneracy, removed before the end, with
//     Bland's rule as a fallback when progress stalls;
//   * geometric scaling, periodic refactorization and recomputation of the
//     primal values from scratch.
//
// Integrality flags are ignored (the LP relaxation is solved).
#pragma once

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::simplex {

enum class Pricing { kDantzig, kDevex };

struct SimplexOptions {
  long long max_iterations = 50'000'000;
  double time_limit_seconds = 3600.0;
  double primal_tolerance = 1e-7;   // bound violation accepted as feasible (scaled model)
  double dual_tolerance = 1e-7;     // reduced cost accepted as optimal (scaled model)
  double pivot_tolerance = 1e-7;    // |alpha| below this is never pivoted on
  Pricing pricing = Pricing::kDevex;
  bool scale = true;
  bool perturb = true;
  int log_every = 0;  // print a progress line every N iterations (0 = silent)
};

Solution solve_primal_simplex(const Model& lp, const SimplexOptions& options = {});

}  // namespace ps26119::simplex
