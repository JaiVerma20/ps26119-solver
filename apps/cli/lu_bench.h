// lu_bench.h - Layer 2 benchmark on the constraint matrix of a real model (`ps26119 lu-bench`).
// Origin: gpuopt apps/lu_bench.hpp (Shivanshu Vats, c192dd0).
#pragma once

#include "ps26119/model.h"

namespace ps26119::cli {

struct LuBenchOptions {
  int trials = 3;      // random bases per structural fraction
  int updates = 100;   // PFI updates in the update test
  unsigned seed = 1;
};

// Builds simplex-style bases (m columns chosen among the n structural columns
// of A and the m slack columns), factorizes them and verifies FTRAN / BTRAN
// by their residuals. Prints a report; returns 0 if every check passed.
int run_lu_bench(const Model& lp, const LuBenchOptions& options);

}  // namespace ps26119::cli
