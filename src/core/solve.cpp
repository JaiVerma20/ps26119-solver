// solve.cpp — the solve() dispatcher. Validates the model, routes to an engine, stamps
// provenance (fingerprint, engine, precision, wall time) into the Solution.
#include <chrono>

#include "ps26119/solve.h"

namespace ps26119 {

Solution solve(const Model& model, const Options& options) {
  const auto t0 = std::chrono::steady_clock::now();
  Solution sol;
  sol.model_fingerprint = model.fingerprint_hex();
  sol.precision = to_string(options.precision);
  sol.engine = to_string(options.algorithm);

  if (auto err = model.validate(); !err.empty()) {
    sol.status = Status::NotSolved;
    sol.message = "invalid model: " + err;
    return sol;
  }

  sol.status = Status::NotSolved;
  sol.message = "no engine implemented yet";

  sol.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return sol;
}

}  // namespace ps26119
