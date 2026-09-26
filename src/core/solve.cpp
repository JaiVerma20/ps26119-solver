// solve.cpp — the solve() dispatcher. Validates the model, routes to an engine, stamps
// provenance (fingerprint, engine, precision, wall time) into the Solution, and gates every
// Optimal LP answer through the in-process checker on the ORIGINAL model (gate() below):
// every engine consumes the same Model and returns the same Solution.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>

#include "ps26119/solve.h"

#include "core/certificates.h"
#include "core/gate.h"
#include "core/presolve.h"
#include "core/safe_bound.h"
#include "core/solution_checker.h"
#include "pdhg/backend.h"
#include "ps26119/tolerances.h"
#include "la/parallel.h"
#include "mip/branch_and_bound.h"
#include "oracle/dense_simplex.h"
#include "pdhg/pdlp.h"
#include "pdhg/r2hpdhg.h"
#include "simplex/primal_simplex.h"

namespace ps26119 {

namespace {
Solution solve_direct(const Model& model, const Options& options);

bool first_order(Algorithm a) { return a == Algorithm::Auto || a == Algorithm::Pdlp || a == Algorithm::R2hpdhg; }

// Simplex options from the generic Options (limits + simplex_* engine knobs). Returns the
// name of an unknown or non-simplex knob, or "" on success.
std::string simplex_options_from(const Options& o, simplex::SimplexOptions& so) {
  so.time_limit_seconds = o.time_limit;
  so.max_iterations = o.iteration_limit;
  so.log_every = o.verbosity >= 2 ? 1000 : 0;
  for (const auto& [name, v] : o.engine_params) {
    if (name == "simplex_pricing") so.pricing = v != 0 ? simplex::Pricing::kDevex : simplex::Pricing::kDantzig;
    else if (name == "simplex_scale") so.scale = v != 0;
    else if (name == "simplex_perturb") so.perturb = v != 0;
    else if (name == "simplex_primal_tolerance") so.primal_tolerance = v;
    else if (name == "simplex_dual_tolerance") so.dual_tolerance = v;
    else return name;
  }
  return {};
}

// A model whose bounds cross is infeasible; the crossing itself is the certificate.
Solution trivially_infeasible(const Model& model, const Options& options, const std::string& why) {
  Solution s;
  s.status = Status::Infeasible;
  s.engine = to_string(options.algorithm);
  s.precision = to_string(options.precision);
  s.model_fingerprint = model.fingerprint_hex();
  s.message = "bounds cross: " + why;
  return s;
}

bool has_integers(const Model& m) {
  for (auto v : m.is_integer)
    if (v) return true;
  return false;
}

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

namespace {
Solution solve_impl(const Model& model, const Options& options);

bool treat_as_mip(const Model& m, const Options& o) { return has_integers(m) && !o.relax_integrality; }

}  // namespace

namespace core {
// The verification gate (CLAUDE.md §5.4: a wrong answer is worse than no answer). An Optimal
// LP answer is re-checked on the ORIGINAL model by the in-process checker with the verifier
// tolerances. Exact engines (simplex, oracle) and first-order runs that asked for
// verifier-grade accuracy are demoted to NumericalError when the check fails. MILP answers
// are verified by the branch-and-bound / presolve safety net instead (integrality, rows).
void gate(const Model& model, const Options& options, Solution& s) {
  if (treat_as_mip(model, options)) return;
  if (s.status == Status::Infeasible || s.status == Status::Unbounded) {
    // Claims other than Optimal are verified too (core/certificates.h).
    s.objective = s.dual_objective = std::numeric_limits<double>::quiet_NaN();
    auto note = [&](const std::string& t) { s.message += std::string(s.message.empty() ? "" : "; ") + t; };
    if (s.status == Status::Infeasible && !model.crossed_bounds().empty()) {
      s.check = "PASS";  // the certificate is the crossing itself, re-derived from the model
      return;
    }
    const CertificateCheck c = s.status == Status::Infeasible
                                   ? check_infeasibility_certificate(model, s.dual_ray)
                                   : check_unboundedness_certificate(model, s.x, s.primal_ray);
    if (!c.present) {
      s.check.clear();
      note("not certified (the engine returned no certificate)");
    } else if (c.passed) {
      s.check = "PASS";
      note(c.detail);
    } else {
      s.check = "FAIL";
      note(std::string(to_string(s.status)) + " withdrawn, " + c.detail);
      s.status = Status::NumericalError;
    }
    return;
  }
  if (s.status != Status::Optimal) return;
  if (static_cast<int>(s.x.size()) != model.num_cols || static_cast<int>(s.y.size()) != model.num_rows) {
    s.status = Status::NumericalError;
    s.message += std::string(s.message.empty() ? "" : "; ") + "engine reported Optimal without primal and dual vectors";
    return;
  }
  const CheckReport rep = check_solution(model, s.x, s.y);
  s.check = rep.passed() ? "PASS" : "FAIL";
  s.check_primal = rep.max_primal_violation;
  s.check_dual = rep.max_dual_violation;
  s.check_gap = rep.relative_gap;
  if (std::isnan(s.dual_objective)) s.dual_objective = rep.dual_objective;
  if (std::isnan(s.primal_residual)) s.primal_residual = rep.max_primal_violation;
  if (std::isnan(s.dual_residual)) s.dual_residual = rep.max_dual_violation;
  if (std::isnan(s.gap)) s.gap = rep.relative_gap;
  if (rep.passed()) return;
  // Same boundary as pdhg::KktStats::converged(): a first-order tolerance strictly tighter
  // than the verifier's promises verifier-grade per-row accuracy; 1e-6 and looser do not.
  const bool claims_verifier_grade = !first_order(options.algorithm) || options.tolerance < tol::kVerifyPrimal;
  const std::string what = "in-process check on the original model: " + rep.summary();
  if (claims_verifier_grade) {
    s.status = Status::NumericalError;
    s.message += std::string(s.message.empty() ? "" : "; ") + "Optimal withdrawn, " + what;
  } else {
    s.message += std::string(s.message.empty() ? "" : "; ") + what + " (requested tolerance is looser than the verifier's)";
  }
}
}  // namespace core

Solution solve(const Model& model, const Options& options) {
  if (options.algorithm != Algorithm::Auto) {
    Solution s = solve_impl(model, options);
    core::gate(model, options, s);
    return s;
  }
  // Auto: resolve the engine once, on the ORIGINAL model (DECISIONS #29).
  Options resolved = options;
  const double work = static_cast<double>(std::max(model.num_rows, 1)) * static_cast<double>(model.nnz());
  // Warm starts and first-order knobs only exist for the first-order engines: honour them.
  bool first_order_request = !options.warm_x.empty() || !options.warm_y.empty() || options.warm_primal_weight > 0;
  for (const auto& kv : options.engine_params)
    first_order_request = first_order_request || (kv.first.rfind("simplex_", 0) != 0 && kv.first.rfind("mip_", 0) != 0);
  resolved.algorithm = !first_order_request && work <= kAutoSimplexWork ? Algorithm::Simplex : Algorithm::R2hpdhg;
  Solution s = solve_impl(model, resolved);
  core::gate(model, resolved, s);
  if (!treat_as_mip(model, options)) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "auto: %s (rows*nnz = %.2g)", to_string(resolved.algorithm), work);
    s.message = s.message.empty() ? buf : std::string(buf) + "; " + s.message;
  }
  return s;
}

namespace {
// Presolve wrapper: reduce, solve the reduced model, postsolve, and re-check optimality on the
// ORIGINAL model; if a first-order answer misses the tolerance there, polish it by a
// warm-started solve of the original (so presolve never weakens what "Optimal" means).
Solution solve_impl(const Model& model, const Options& options) {
  // Invalid input (model or warm-start sizes) is reported by solve_direct before any
  // presolve work touches it (mapping a wrong-sized warm start would read out of bounds).
  const bool bad_warm = (!options.warm_x.empty() && static_cast<int>(options.warm_x.size()) != model.num_cols) ||
                        (!options.warm_y.empty() && static_cast<int>(options.warm_y.size()) != model.num_rows);
  if (!options.presolve || bad_warm || !model.validate().empty()) return solve_direct(model, options);
  if (auto why = model.crossed_bounds(); !why.empty()) return trivially_infeasible(model, options, why);
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
  const bool lp_model = !(has_integers(model) && !options.relax_integrality);
  // Certificates must be in the ORIGINAL model's space (the gate checks them there), and a
  // presolve infeasibility verdict carries none: for an LP, re-derive Infeasible / Unbounded
  // by solving the original directly (rare, so the extra cost is small; it also cross-checks
  // presolve's own verdict).
  auto on_original = [&](const Solution& why, const std::string& what) {
    Options direct = options;
    direct.presolve = false;
    Solution d = solve_direct(model, direct);
    d.iterations += why.iterations;
    d.message += std::string(d.message.empty() ? "" : "; ") + "presolve: " + what +
                 "; re-solved the original model for an original-space certificate";
    return d;
  };
  if (pr.outcome == PresolveResult::Outcome::Infeasible && lp_model) {
    Solution none;
    return on_original(none, "reduction found infeasibility (" + pr.message + ")");
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
  if (lp_model && (red.status == Status::Infeasible || red.status == Status::Unbounded))
    return on_original(red, std::string("reduced model ") + to_string(red.status));
  Solution post = postsolve(model, pr, red);
  const std::string note = "presolve removed " + std::to_string(pr.removed_rows) + " rows, " +
                           std::to_string(pr.removed_cols) + " cols";
  post.message = post.message.empty() ? note : post.message + "; " + note;
  post.model_fingerprint = model.fingerprint_hex();
  auto passes = [&](const Solution& s) {
    return static_cast<int>(s.x.size()) == model.num_cols && static_cast<int>(s.y.size()) == model.num_rows &&
           original_kkt(model, s).converged(options.tolerance);
  };
  const bool is_mip = has_integers(model) && !options.relax_integrality;
  if (is_mip && (post.status == Status::Optimal || !post.x.empty())) {
    // Safety net: a MILP answer that went through presolve is re-verified on the ORIGINAL
    // model (bounds, rows, integrality). Never report a point that fails as Optimal.
    bool ok = static_cast<int>(post.x.size()) == model.num_cols;
    if (ok) {
      const auto ax = model.row_activity(post.x);
      auto in = [](double v, double lo, double up) {
        return v >= lo - tol::kMipFeasibility * (1 + std::fabs(lo)) && v <= up + tol::kMipFeasibility * (1 + std::fabs(up));
      };
      for (int j = 0; ok && j < model.num_cols; ++j) {
        ok = in(post.x[j], model.col_lower[j], model.col_upper[j]);
        if (ok && model.is_integer[j]) ok = std::fabs(post.x[j] - std::round(post.x[j])) <= tol::kMipIntegrality;
      }
      for (int i = 0; ok && i < model.num_rows; ++i) ok = in(ax[i], model.row_lower[i], model.row_upper[i]);
    }
    if (!ok && post.status == Status::Optimal) {
      post.status = Status::NumericalError;
      post.message += "; postsolved MILP point failed the check on the original model";
    }
  }
  if (!is_mip && post.status == Status::Optimal && first_order(options.algorithm) && !passes(post)) {
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
  } else if (!is_mip && post.status == Status::Optimal && !first_order(options.algorithm) &&
             !check_solution(model, post.x, post.y).passed()) {
    // Exact engine: a postsolved vertex that fails on the original model is not trusted;
    // solve the original without presolve instead (never worse than no presolve).
    Solution cold = solve_direct(model, inner);
    cold.iterations += red.iterations;
    cold.message += std::string(cold.message.empty() ? "" : "; ") + note +
                    "; postsolved point failed the check on the original, solved the original instead";
    post = cold;
  } else if (!is_mip && post.status == Status::Optimal && static_cast<int>(post.x.size()) != model.num_cols) {
    post.status = Status::NumericalError;  // never claim Optimal without a point (a model may have 0 columns)
  }
  if (static_cast<int>(post.y.size()) == model.num_rows && model.num_rows + model.num_cols > 0)
    post.certified_bound = certified_dual_bound(model, post.y).bound;
  post.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return post;
}

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
  if (auto why = model.crossed_bounds(); !why.empty()) return trivially_infeasible(model, options, why);

  la::ThreadPool::instance().set_threads(options.threads);
  try {
    if (has_integers(model) && !options.relax_integrality) {
      // Never return an LP relaxation as if it were the MILP optimum (CLAUDE.md §5.4).
      mip::BranchAndBoundOptions bo;
      bo.time_limit = options.time_limit;
      bo.node_limit = options.iteration_limit;
      bo.verbosity = options.verbosity;
      for (const auto& [name, v] : options.engine_params) {
        if (name == "mip_node_solver") {
          bo.node_solver = v != 0 ? mip::NodeSolver::Oracle : mip::NodeSolver::Simplex;
        } else {
          sol.status = Status::NotSolved;
          sol.message = "unknown MILP engine parameter '" + name + "' (known: mip_node_solver)";
          return sol;
        }
      }
      sol = mip::solve_branch_and_bound(model, bo);
      sol.model_fingerprint = model.fingerprint_hex();
      sol.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
      case Algorithm::Simplex: {
        simplex::SimplexOptions so;
        if (auto bad = simplex_options_from(options, so); !bad.empty()) {
          sol.status = Status::NotSolved;
          sol.message = "unknown simplex engine parameter '" + bad + "'";
          break;
        }
        sol = simplex::solve_primal_simplex(model, so);
        if (!options.warm_x.empty() || !options.warm_y.empty())
          sol.message += std::string(sol.message.empty() ? "" : "; ") + "warm start ignored (no simplex basis warm start yet)";
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
  if (has_integers(model)) {  // options.relax_integrality: say what was solved
    sol.message = std::string("LP relaxation (integrality ignored)") + (sol.message.empty() ? "" : "; ") + sol.message;
  }
  if (static_cast<int>(sol.y.size()) == model.num_rows && model.num_rows + model.num_cols > 0)
    sol.certified_bound = certified_dual_bound(model, sol.y).bound;

  sol.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return sol;
}

}  // namespace

}  // namespace ps26119
