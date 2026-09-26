// dense_tableau.h - the second, independent reference ("oracle") LP solver.
//
// Origin: gpuopt include/gpuopt/dense_oracle.hpp (Shivanshu Vats, c192dd0).
// ps26119 has two independently written oracles: the double-double bounded simplex
// (oracle/dense_simplex.h, canonical, `--algorithm oracle`) and this double-precision
// two-phase Bland tableau (test-only cross-check). They share no numerical code.
//
// A deliberately slow, textbook, two-phase dense tableau simplex using
// Bland's rule. Its only job is to be unarguably correct on small problems,
// so that every fast engine we build later can be differentially tested
// against it. Never optimise this file; keep it readable.
//
// Integrality flags are ignored: a MILP is solved as its LP relaxation.
#pragma once

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::oracle {

struct DenseTableauOptions {
  long long max_iterations = 1'000'000;
  double pivot_tolerance = 1e-9;        // smallest |entry| accepted as a pivot
  double optimality_tolerance = 1e-9;   // reduced cost below -tol => can improve
  double feasibility_tolerance = 1e-9;  // relative phase-1 residual => infeasible
  // Refuse problems whose tableau would exceed this many doubles (~8 bytes each).
  long long max_tableau_entries = 50'000'000;
};

Solution solve_dense_tableau(const Model& lp, const DenseTableauOptions& options = {});

}  // namespace ps26119::oracle
