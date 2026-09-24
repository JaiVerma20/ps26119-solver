// solve.cpp — the solve() dispatcher. Validates the model, routes to an engine, stamps
// provenance (fingerprint, engine, precision, wall time) into the Solution.
#include <chrono>

#include "ps26119/solve.h"

#include "oracle/dense_simplex.h"

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

  switch (options.algorithm) {
    case Algorithm::Oracle: {
      oracle::DenseSimplexOptions o;
      o.iteration_limit = options.iteration_limit;
      o.time_limit = options.time_limit;
      o.verbosity = options.verbosity;
      sol = oracle::solve_dense_simplex(model, o);
      break;
    }
    default:
      sol.status = Status::NotSolved;
      sol.message = std::string("engine not implemented: ") + to_string(options.algorithm);
      break;
  }
  sol.model_fingerprint = model.fingerprint_hex();

  sol.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return sol;
}

}  // namespace ps26119
