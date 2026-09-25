// dense_oracle.hpp - the reference ("oracle") LP solver.
//
// A deliberately slow, textbook, two-phase dense tableau simplex using
// Bland's rule. Its only job is to be unarguably correct on small problems,
// so that every fast engine we build later can be differentially tested
// against it. Never optimise this file; keep it readable.
//
// Integrality flags are ignored: a MILP is solved as its LP relaxation.
#pragma once

#include "gpuopt/problem.hpp"
#include "gpuopt/result.hpp"

namespace gpuopt {

struct DenseOracleOptions {
  long long max_iterations = 1'000'000;
  double pivot_tolerance = 1e-9;        // smallest |entry| accepted as a pivot
  double optimality_tolerance = 1e-9;   // reduced cost below -tol => can improve
  double feasibility_tolerance = 1e-9;  // relative phase-1 residual => infeasible
  // Refuse problems whose tableau would exceed this many doubles (~8 bytes each).
  long long max_tableau_entries = 50'000'000;
};

SolveResult solve_dense_oracle(const LpProblem& lp, const DenseOracleOptions& options = {});

}  // namespace gpuopt
