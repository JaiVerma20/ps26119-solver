// solve.cpp — the solve() dispatcher. Validates the model, routes to an engine, stamps
// provenance (fingerprint, engine, precision, wall time) into the Solution.
#include <chrono>
#include <exception>
#include <new>

#include "ps26119/solve.h"

#include "oracle/dense_simplex.h"
#include "pdhg/pdlp.h"
#include "pdhg/r2hpdhg.h"

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

  if ((!options.warm_x.empty() && static_cast<int>(options.warm_x.size()) != model.num_cols) ||
      (!options.warm_y.empty() && static_cast<int>(options.warm_y.size()) != model.num_rows)) {
    sol.status = Status::NotSolved;
    sol.message = "warm start vectors have the wrong size";
    return sol;
  }

  try {
    switch (options.algorithm) {
      case Algorithm::Oracle: {
        oracle::DenseSimplexOptions o;
        o.iteration_limit = options.iteration_limit;
        o.time_limit = options.time_limit;
        o.verbosity = options.verbosity;
        sol = oracle::solve_dense_simplex(model, o);
        break;
      }
      case Algorithm::Pdlp:
      case Algorithm::Auto:
      case Algorithm::R2hpdhg: {
        pdhg::EngineOptions eo = pdhg::engine_options_from(options);
        if (auto bad = pdhg::apply_engine_params(options, eo); !bad.empty()) {
          sol.status = Status::NotSolved;
          sol.message = "unknown engine parameter '" + bad + "'";
          break;
        }
        sol = options.algorithm == Algorithm::Pdlp ? pdhg::solve_pdlp(model, eo) : pdhg::solve_r2hpdhg(model, eo);
        break;
      }
    }
  } catch (const std::bad_alloc&) {
    sol = Solution{};
    sol.status = Status::NotSolved;
    sol.message = "out of memory";
  } catch (const std::exception& e) {  // e.g. CUDA runtime errors; never let them escape
    sol = Solution{};
    sol.status = Status::NumericalError;
    sol.message = e.what();
  }
  sol.model_fingerprint = model.fingerprint_hex();

  sol.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return sol;
}

}  // namespace ps26119
