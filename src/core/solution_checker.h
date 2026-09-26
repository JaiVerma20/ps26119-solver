// solution_checker.hpp - independent optimality certificate.
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

#include "gpuopt/problem.hpp"

namespace gpuopt {

struct CheckTolerances {
  double primal = 1e-6;  // scaled bound / row violation
  double dual = 1e-6;    // scaled wrong-sign dual value
  double gap = 1e-6;     // relative duality gap
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

CheckReport check_solution(const LpProblem& lp, const std::vector<double>& x,
                           const std::vector<double>& row_dual,
                           const CheckTolerances& tolerances = {});

}  // namespace gpuopt
