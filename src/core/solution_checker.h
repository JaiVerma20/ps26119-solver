// solution_checker.h - independent optimality certificate (in-process).
//
// Origin: gpuopt include/gpuopt/solution_checker.hpp (Shivanshu Vats, c192dd0).
// Role in ps26119: the gate every Optimal LP answer of solve() must pass before it is
// reported (src/core/solve.cpp). tools/verify.py implements the same conditions with a
// different reader (highspy) and is the EXTERNAL verifier; the two are kept independent.
// Default tolerances are the verifier tolerances in include/ps26119/tolerances.h.
//
// Never trust a solver that says "optimal". This checker takes only the
// problem, a primal vector x and a row dual vector y, and verifies from
// first principles that
//   1. x is primal feasible          (rows and bounds satisfied),
//   2. y is dual feasible            (correct signs of y and d = c - A^T y),
//   3. complementary slackness holds (no dual weight on inactive bounds),
//   4. primal objective == dual objective (zero duality gap).
// Together these prove x is optimal, independently of how it was found.
#pragma once

#include <string>
#include <vector>

#include "ps26119/model.h"
#include "ps26119/tolerances.h"

namespace ps26119 {

struct CheckTolerances {
  double primal = tol::kVerifyPrimal;  // scaled bound / row violation
  double dual = tol::kVerifyDual;      // scaled wrong-sign dual value
  double gap = tol::kVerifyGap;        // relative duality gap
};

struct CheckReport {
  double max_primal_violation = 0.0;
  double max_dual_violation = 0.0;
  double max_complementarity = 0.0;
  double primal_objective = 0.0;
  double dual_objective = 0.0;
  double relative_gap = 0.0;

  bool primal_ok = false;
  bool dual_ok = false;
  bool gap_ok = false;

  bool passed() const { return primal_ok && dual_ok && gap_ok; }
  std::string summary() const;
};

CheckReport check_solution(const Model& lp, const std::vector<double>& x,
                           const std::vector<double>& row_dual,
                           const CheckTolerances& tolerances = {});

}  // namespace ps26119
