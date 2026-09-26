// solution.h — the Solution contract (CLAUDE.md §7).
//
// Dual sign convention (same as HiGHS, stated in the ORIGINAL objective sense):
//   z = c − Aᵀy.
//   For a MIN problem: y_i ≥ 0 when row i sits at its lower bound, y_i ≤ 0 at its upper
//   bound; z_j ≥ 0 when x_j sits at its lower bound, z_j ≤ 0 at its upper bound.
//   For a MAX problem all these signs flip.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ps26119 {

enum class Status {
  Optimal,
  Infeasible,
  Unbounded,
  IterationLimit,
  TimeLimit,
  NumericalError,
  NotSolved,
};

const char* to_string(Status s);
// Parses the names produced by to_string; returns false on an unknown name.
bool status_from_string(const std::string& s, Status& out);

// CLI exit codes (CLAUDE.md §7).
int exit_code(Status s);  // 0 optimal, 1 limit/infeasible/unbounded, 5 numerical/not solved
inline constexpr int kExitReadError = 3;

struct Solution {
  Status status = Status::NotSolved;
  std::vector<double> x;             // size n (empty if no point)
  std::vector<double> row_activity;  // size m, A x
  std::vector<double> y;             // size m, row duals
  std::vector<double> z;             // size n, reduced costs c − Aᵀy
  double objective = std::numeric_limits<double>::quiet_NaN();  // cᵀx + obj_offset
  double dual_objective = std::numeric_limits<double>::quiet_NaN();
  double primal_residual = std::numeric_limits<double>::quiet_NaN();  // relative, see engine docs
  double dual_residual = std::numeric_limits<double>::quiet_NaN();
  double gap = std::numeric_limits<double>::quiet_NaN();
  std::int64_t iterations = 0;
  double seconds = 0.0;
  double setup_seconds = 0.0;  // part of `seconds` spent before the first iteration (scaling, ‖A‖, upload)
  // First-order engines record when the relative KKT error first dropped below the "fast"
  // tolerance (1e-4) while continuing to the requested one. -1 = never reached / not tracked.
  std::int64_t iterations_to_fast = -1;
  double seconds_to_fast = -1.0;
  // Certified bound on the optimal objective from y (src/core/safe_bound.h): a guaranteed
  // lower bound for MIN models, upper bound for MAX, immune to rounding. ±inf when the
  // multipliers point at an infinite bound; NaN when not computed (no y).
  double certified_bound = std::numeric_limits<double>::quiet_NaN();
  // First-order engines: primal weight ω at the end (pass it back as
  // Options::warm_primal_weight when warm-starting a re-solve). NaN for other engines.
  double primal_weight = std::numeric_limits<double>::quiet_NaN();
  // In-process verification of an Optimal LP answer on the ORIGINAL model
  // (src/core/solution_checker.h, verifier tolerances from tolerances.h): "PASS", "FAIL",
  // or empty when not applicable (not Optimal, MILP). An Optimal that fails is demoted to
  // NumericalError, except for a first-order run whose requested tolerance is not tighter
  // than the verifier's 1e-6 (it stays Optimal at ITS tolerance and `check` says FAIL).
  // Measures: primal max viol/(1+|bound|), dual max wrong-sign/max(1,‖c‖∞), relative gap.
  std::string check;
  double check_primal = std::numeric_limits<double>::quiet_NaN();
  double check_dual = std::numeric_limits<double>::quiet_NaN();
  double check_gap = std::numeric_limits<double>::quiet_NaN();
  std::string engine;     // "oracle", "pdlp", "r2hpdhg", "simplex", ...
  std::string precision;  // "dd", "fp64", "mixed"
  std::string model_fingerprint;
  std::string message;  // human readable detail (why NumericalError, etc.)
};

}  // namespace ps26119
