// engine.cpp — shared machinery for the first-order engines (see engine.h).
#include "pdhg/engine.h"

#include <cmath>
#include <cstdio>
#include <string>

#include "ps26119/tolerances.h"

namespace ps26119::pdhg {

EngineOptions engine_options_from(const Options& o) {
  EngineOptions e;
  e.tolerance = o.tolerance;
  e.time_limit = o.time_limit;
  e.iteration_limit = o.iteration_limit;
  e.check_every = o.termination_check_every > 0 ? o.termination_check_every : 64;
  e.precision = o.precision;
  e.use_gpu = o.use_gpu;
  e.verbosity = o.verbosity;
  e.scaling.ruiz_iterations = o.ruiz_iterations;
  e.scaling.pock_chambolle = o.pock_chambolle;
  return e;
}

// ------------------------------------------------------------------ precision policy
PrecisionPolicy::PrecisionPolicy(Backend& b, const EngineOptions& opt)
    : b_(b), mixed_(opt.precision == Precision::Mixed), target_(opt.tolerance) {
  b_.set_precision(mixed_ ? Precision::Mixed : Precision::Fp64);
}

bool PrecisionPolicy::on_check(double rel_kkt, std::int64_t iteration) {
  if (!mixed_ || b_.precision() == Precision::Fp64) return false;
  if (rel_kkt < 0.9 * best_) {
    best_ = rel_kkt;
    checks_since_best_ = 0;
  } else {
    ++checks_since_best_;
  }
  // Promote when fp32 has done what fp32 can: the switch level is reached (and the target
  // is tighter than it), or progress has stalled for 10 checks.
  const bool reached = rel_kkt <= tol::kMixedPrecisionSwitch && target_ < tol::kMixedPrecisionSwitch;
  const bool stalled = checks_since_best_ >= 10;
  if (reached || stalled) {
    b_.set_precision(Precision::Fp64);
    switched_at_ = iteration;
    return true;
  }
  return false;
}

// ------------------------------------------------------------------ infeasibility
RayTest ray_test(const ScaledProblem& sp, const std::vector<double>& dxs, const std::vector<double>& dys) {
  const Model& M = *sp.original;
  RayTest t;
  // ---- dual ray (certifies primal infeasibility)
  std::vector<double> r(sp.m), g(sp.n);
  double rmax = 0;
  for (int i = 0; i < sp.m; ++i) {
    double v = sp.row_scale[i] * dys[i] / sp.obj_scale;
    if (!std::isfinite(M.row_lower[i])) v = std::min(v, 0.0);
    if (!std::isfinite(M.row_upper[i])) v = std::max(v, 0.0);
    r[i] = v;
    rmax = std::max(rmax, std::fabs(v));
  }
  if (rmax > 0) {
    sp.At_orig.multiply<double>(r.data(), g.data());
    double obj = 0, viol = 0;
    for (int i = 0; i < sp.m; ++i) obj += r[i] > 0 ? M.row_lower[i] * r[i] : (r[i] < 0 ? M.row_upper[i] * r[i] : 0.0);
    for (int j = 0; j < sp.n; ++j) {
      const double lam = -g[j], lo = M.col_lower[j], up = M.col_upper[j];
      if (lam > 0) {
        if (std::isfinite(lo)) obj += lo * lam;
        else viol = std::max(viol, lam);
      } else if (lam < 0) {
        if (std::isfinite(up)) obj += up * lam;
        else viol = std::max(viol, -lam);
      }
    }
    const double scale = std::max(rmax, viol);
    t.dual_ray_objective = obj / scale;
    t.dual_ray_violation = viol / scale;
    t.primal_infeasible = t.dual_ray_objective > 0 &&
                          t.dual_ray_violation <= tol::kFirstOrderInfeasible * t.dual_ray_objective;
  }
  // ---- primal ray (certifies dual infeasibility)
  std::vector<double> d(sp.n), ad(sp.m);
  double dmax = 0;
  for (int j = 0; j < sp.n; ++j) {
    double v = sp.col_scale[j] * dxs[j] / sp.bound_scale;
    if (std::isfinite(M.col_lower[j])) v = std::max(v, 0.0);
    if (std::isfinite(M.col_upper[j])) v = std::min(v, 0.0);
    d[j] = v;
    dmax = std::max(dmax, std::fabs(v));
  }
  if (dmax > 0) {
    sp.A_orig.multiply<double>(d.data(), ad.data());
    double cd = 0, viol = 0;
    for (int j = 0; j < sp.n; ++j) cd += M.sense * M.obj[j] * d[j];
    for (int i = 0; i < sp.m; ++i) {
      if (std::isfinite(M.row_lower[i])) viol = std::max(viol, -ad[i]);
      if (std::isfinite(M.row_upper[i])) viol = std::max(viol, ad[i]);
    }
    const double scale = std::max(dmax, viol);
    t.primal_ray_objective = cd / scale;
    t.primal_ray_violation = viol / scale;
    t.dual_infeasible = t.primal_ray_objective < 0 &&
                        t.primal_ray_violation <= tol::kFirstOrderInfeasible * -t.primal_ray_objective;
  }
  return t;
}

Status EngineContext::check_infeasibility(int dx, int dy, const KktStats& current, std::string& message) {
  std::vector<double> dxs, dys;
  backend_->download(dx, dxs);
  backend_->download(dy, dys);
  const RayTest t = ray_test(sp_, dxs, dys);
  if (t.primal_infeasible) {
    message = "primal infeasible: dual ray certificate (objective " + std::to_string(t.dual_ray_objective) +
              ", violation " + std::to_string(t.dual_ray_violation) + ")";
    return Status::Infeasible;
  }
  if (t.dual_infeasible && current.rel_primal() <= tol::kVerifyPrimal) {
    message = "unbounded: primal ray certificate (cᵀd " + std::to_string(t.primal_ray_objective) +
              ", violation " + std::to_string(t.primal_ray_violation) + ") at a primal-feasible iterate";
    return Status::Unbounded;
  }
  return Status::NotSolved;
}

// ------------------------------------------------------------------ context
EngineContext::EngineContext(const Model& model, const EngineOptions& opt, const char* engine_name)
    : model_(model), opt_(opt), name_(engine_name), t0_(std::chrono::steady_clock::now()) {
  sp_ = make_scaled_problem(model, opt.scaling);
  backend_ = make_backend(opt.use_gpu, error_);
  if (!backend_) return;
  backend_->setup(sp_);
  const double norm = la::estimate_norm2(sp_.A, sp_.At);
  eta_ = norm > 0 ? 0.998 / norm : 1.0;
  backend_->sync();
  setup_seconds_ = elapsed();
}

double EngineContext::initial_primal_weight() const {
  if (opt_.scaling.bound_objective_rescaling) return 1.0;
  double c2 = 0, b2 = 0;
  for (double v : sp_.c) c2 += v * v;
  for (int i = 0; i < sp_.m; ++i) {
    const double lo = std::isfinite(sp_.row_lower[i]) ? std::fabs(sp_.row_lower[i]) : 0.0;
    const double up = std::isfinite(sp_.row_upper[i]) ? std::fabs(sp_.row_upper[i]) : 0.0;
    b2 += std::max(lo, up) * std::max(lo, up);
  }
  return (std::sqrt(c2) + 1.0) / (std::sqrt(b2) + 1.0);
}

double EngineContext::elapsed() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
}

bool EngineContext::record(const KktStats& k, std::int64_t iteration) {
  const double e = k.rel_kkt();
  if (fast_iterations_ < 0 && e <= tol::kFirstOrderFast) {
    fast_iterations_ = iteration;
    fast_seconds_ = elapsed();
  }
  if (opt_.verbosity >= 2)
    std::fprintf(stderr, "%s it %8lld  t %7.2fs  pobj %+.10e  dobj %+.10e  rp %.2e  rd %.2e  gap %.2e  %s\n", name_,
                 static_cast<long long>(iteration), elapsed(), k.primal_obj, k.dual_obj, k.rel_primal(), k.rel_dual(),
                 k.rel_gap(), backend_->precision() == Precision::Mixed ? "fp32" : "fp64");
  return k.converged(opt_.tolerance);
}

Solution EngineContext::finish(Status status, int xs, int ys, std::int64_t iterations, const std::string& message) {
  Solution sol;
  sol.engine = name_;
  sol.precision = to_string(opt_.precision);
  sol.iterations = iterations;
  sol.message = message;
  std::vector<double> xsd, ysd, y_min;
  backend_->download(xs, xsd);
  backend_->download(ys, ysd);
  sp_.unscale_primal(xsd, sol.x);
  sp_.unscale_dual(ysd, y_min);
  const KktStats k = kkt_on_original(sp_, sol.x, y_min);
  if (!k.finite()) status = Status::NumericalError;
  if (status == Status::Optimal && !k.converged(opt_.tolerance)) status = Status::NumericalError;  // never over-claim

  const double sense = model_.sense;
  sol.y.resize(sp_.m);
  for (int i = 0; i < sp_.m; ++i) sol.y[i] = sense * y_min[i];
  sol.z = model_.obj;  // z = c − Aᵀy (original sense)
  for (int j = 0; j < sp_.n; ++j)
    for (int p = model_.col_start[j]; p < model_.col_start[j + 1]; ++p) sol.z[j] -= model_.value[p] * sol.y[model_.row_index[p]];
  sol.row_activity = model_.row_activity(sol.x);
  sol.objective = model_.objective_value(sol.x);
  sol.dual_objective = sense * k.dual_obj + model_.obj_offset;
  sol.primal_residual = k.rel_primal();
  sol.dual_residual = k.rel_dual();
  sol.gap = k.rel_gap();
  sol.status = status;
  sol.iterations_to_fast = fast_iterations_;
  sol.seconds_to_fast = fast_seconds_;
  sol.seconds = elapsed();
  sol.setup_seconds = setup_seconds_;
  return sol;
}

}  // namespace ps26119::pdhg
