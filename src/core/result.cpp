#include "gpuopt/result.hpp"

namespace gpuopt {

const char* to_string(SolveStatus status) {
  switch (status) {
    case SolveStatus::kNotSolved: return "NOT_SOLVED";
    case SolveStatus::kOptimal: return "OPTIMAL";
    case SolveStatus::kInfeasible: return "INFEASIBLE";
    case SolveStatus::kUnbounded: return "UNBOUNDED";
    case SolveStatus::kIterationLimit: return "ITERATION_LIMIT";
    case SolveStatus::kNumericalError: return "NUMERICAL_ERROR";
  }
  return "UNKNOWN";
}

}  // namespace gpuopt
