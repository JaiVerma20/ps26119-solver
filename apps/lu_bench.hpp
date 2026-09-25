// lu_bench.hpp - Layer 2 benchmark on the constraint matrix of a real model.
#pragma once

#include "gpuopt/problem.hpp"

namespace gpuopt {

struct LuBenchOptions {
  int trials = 3;      // random bases per structural fraction
  int updates = 100;   // PFI updates in the update test
  unsigned seed = 1;
};

// Builds simplex-style bases (m columns chosen among the n structural columns
// of A and the m slack columns), factorizes them and verifies FTRAN / BTRAN
// by their residuals. Prints a report; returns 0 if every check passed.
int run_lu_bench(const LpProblem& lp, const LuBenchOptions& options);

}  // namespace gpuopt
