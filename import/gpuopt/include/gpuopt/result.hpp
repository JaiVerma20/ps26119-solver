// result.hpp - the common result every engine returns.
//
// One status enum and one result struct shared by the oracle, the sparse
// simplex, the IPM and PDLP, so the benchmark harness scores all of them
// identically.
#pragma once

#include <limits>
#include <string>
#include <vector>

namespace gpuopt {

enum class SolveStatus {
  kNotSolved,
  kOptimal,
  kInfeasible,
  kUnbounded,
  kIterationLimit,
  kNumericalError,
};

const char* to_string(SolveStatus status);

struct SolveResult {
  SolveStatus status = SolveStatus::kNotSolved;
  double objective = std::numeric_limits<double>::quiet_NaN();

  // Filled when status == kOptimal. All vectors are in the ORIGINAL problem
  // space and sense:
  //   x             primal values                              (num_cols)
  //   row_activity  A x                                        (num_rows)
  //   row_dual      y, multipliers of the rows                 (num_rows)
  //   col_dual      reduced costs d = obj - A^T y              (num_cols)
  std::vector<double> x;
  std::vector<double> row_activity;
  std::vector<double> row_dual;
  std::vector<double> col_dual;

  long long iterations = 0;
  double seconds = 0.0;
  std::string message;
};

}  // namespace gpuopt
