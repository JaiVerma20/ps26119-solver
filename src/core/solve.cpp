// solve.cpp — the solve() dispatcher. Validates the model, routes to an engine, stamps
// provenance (fingerprint, engine, precision, wall time) into the Solution.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <new>

#include "ps26119/solve.h"

#include "core/presolve.h"
#include "core/safe_bound.h"
#include "pdhg/backend.h"
#include "la/parallel.h"
#include "oracle/dense_simplex.h"
#include "pdhg/pdlp.h"
#include "pdhg/r2hpdhg.h"

namespace ps26119 {

namespace {
Solution solve_direct(const Model& model, const Options& options);

bool first_order(Algorithm a) { return a != Algorithm::Oracle; }

// fp64 KKT of a solution on the ORIGINAL model (termination.h definitions).
pdhg::KktStats original_kkt(const Model& M, const Solution& s) {
  const la::Csr<double> A = la::csr_from_model(M), At = la::csr_transpose_from_model(M);
  const pdhg::LpView lp{M.num_rows, M.num_cols, M.sense, M.obj.data(), M.col_lower.data(), M.col_upper.data(),
                        M.row_lower.data(), M.row_upper.data()};
  std::vector<double> ymin(s.y);
  for (double& v : ymin) v *= M.sense;
  return pdhg::kkt_general(A, At, lp, s.x, ymin);
}
}  // namespace

// Presolve wrapper: reduce, solve the reduced model, postsolve, and re-check optimality on the
// ORIGINAL model; if a first-order answer misses the tolerance there, polish it by a
// warm-started solve of the original (so presolve never weakens what "Optimal" means).
Solution solve(const Model& model, const Options& options) {
  // Invalid input (model or warm-start sizes) is reported by solve_direct before any
  // presolve work touches it (mapping a wrong-sized warm start would read out of bounds).
  const bool bad_warm = (!options.warm_x.empty() && static_cast<int>(options.warm_x.size()) != model.num_cols) ||
                        (!options.warm_y.empty() && static_cast<int>(options.warm_y.size()) != model.num_rows);
  if (!options.presolve || bad_warm || !model.validate().empty()) return solve_direct(model, options);
  const auto t0 = std::chrono::steady_clock::now();
  PresolveResult pr = presolve(model);
  if (pr.outcome == PresolveResult::Outcome::Unchanged) return solve_direct(model, options);
  Options inner = options;
  inner.presolve = false;
  {  // judge the reduced solve with the ORIGINAL model's normalization (see termination.h)
    double b2 = 0, c2 = 0;
    for (int i = 0; i < model.num_rows; ++i) {
      const double lo = std::isfinite(model.row_lower[i]) ? std::fabs(model.row_lower[i]) : 0.0;
      const double up = std::isfinite(model.row_upper[i]) ? std::fabs(model.row_upper[i]) : 0.0;
      b2 += std::max(lo, up) * std::max(lo, up);
    }
    for (double c : model.obj) c2 += c * c;
    inner.kkt_b_norm = std::sqrt(b2);
    inner.kkt_c_norm = std::sqrt(c2);
    // reduced objective (without offsets) = original − Σ c_j x_fixed; shift back, min form
    inner.kkt_obj_shift = model.sense * (pr.reduced.obj_offset - model.obj_offset);
  }
  if (pr.outcome == PresolveResult::Outcome::Infeasible) {
    Solution s;
    s.status = Status::Infeasible;
    s.engine = std::string("presolve+") + to_string(options.algorithm);
    s.message = "presolve: " + pr.message;
    s.model_fingerprint = model.fingerprint_hex();
    return s;
  }
  // map a warm start into the reduced space
  if (!options.warm_x.empty()) {
    inner.warm_x.clear();
    for (int j : pr.col_map) inner.warm_x.push_back(options.warm_x[j]);
  }
  if (!options.warm_y.empty()) {
    inner.warm_y.clear();
    for (int i : pr.row_map) inner.warm_y.push_back(options.warm_y[i]);
  }
  Solution red;
  if (pr.reduced.num_cols == 0 && pr.reduced.num_rows == 0) {
    red.status = Status::Optimal;
    red.engine = "presolve";
  } else {
    red = solve_direct(pr.reduced, inner);
  }
  Solution post = postsolve(model, pr, red);
  const std::string note = "presolve removed " + std::to_string(pr.removed_rows) + " rows, " +
                           std::to_string(pr.removed_cols) + " cols";
  post.message = post.message.empty() ? note : post.message + "; " + note;
  post.model_fingerprint = model.fingerprint_hex();
  auto passes = [&](const Solution& s) {
    return !s.x.empty() && !s.y.empty() && original_kkt(model, s).converged(options.tolerance);
  };
  if (post.status == Status::Optimal && first_order(options.algorithm) && !passes(post)) {
    // The reduced problem's relative KKT uses different norms, so a point that met the
    // tolerance there can narrowly miss it on the original. Step 1: tighten the reduced
    // solve 100×, warm-started in the REDUCED space (same problem, so the warm start is
    // good). Step 2, if still short: a plain cold solve of the original (never worse than
    // no presolve). A warm-started solve of the original was tried first and was fragile
    // (stocfor2: 4.8M iterations vs 39k cold).
    Options tight = inner;
    tight.tolerance = std::max(options.tolerance * 1e-2, 1e-13);
    tight.warm_x = red.x;
    tight.warm_y = red.y;
    const Solution red2 = solve_direct(pr.reduced, tight);
    Solution post2 = postsolve(model, pr, red2);
    post2.iterations += red.iterations;
    if (post2.status == Status::Optimal && passes(post2)) {
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.0e", tight.tolerance);
      post2.message = note + "; reduced model re-solved to " + buf + " to pass on the original";
      post = post2;
    } else {
      Solution cold = solve_direct(model, inner);
      cold.iterations += red.iterations + red2.iterations;
      cold.message += std::string(cold.message.empty() ? "" : "; ") + note +
                      "; postsolved point missed the tolerance, fell back to solving the original";
      post = cold;
    }
  } else if (post.status == Status::Optimal && first_order(options.algorithm) && post.x.empty()) {
    post.status = Status::NumericalError;  // never claim Optimal without a point
  }
  if (static_cast<int>(post.y.size()) == model.num_rows && model.num_rows + model.num_cols > 0)
    post.certified_bound = certified_dual_bound(model, post.y).bound;
  post.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return post;
}

namespace {

Solution solve_direct(const Model& model, const Options& options) {
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

  la::ThreadPool::instance().set_threads(options.threads);
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
  if (static_cast<int>(sol.y.size()) == model.num_rows && model.num_rows + model.num_cols > 0)
    sol.certified_bound = certified_dual_bound(model, sol.y).bound;

  sol.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return sol;
}

}  // namespace

}  // namespace ps26119
