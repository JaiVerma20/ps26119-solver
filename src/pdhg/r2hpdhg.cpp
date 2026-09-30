// r2hpdhg.cpp — restarted Halpern PDHG with reflection. Equations and citations: r2hpdhg.h.
#include "pdhg/r2hpdhg.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <limits>

namespace ps26119::pdhg {

Solution solve_r2hpdhg(const Model& model, const EngineOptions& opt) {
  EngineContext ctx(model, opt, "r2hpdhg");
  if (!ctx.ok()) {
    Solution s;
    s.engine = "r2hpdhg";
    s.status = Status::NotSolved;
    s.message = ctx.error();
    return s;
  }
  Backend& b = ctx.backend();
  PrecisionPolicy policy(b, opt);
  const double eta = ctx.eta(), rho = opt.reflection;
  double omega = ctx.initial_primal_weight();

  // Halpern iterate z, anchor z0, PDHG output ẑ = T(z), reflections z̄ = 2ẑ − z, work.
  int x = b.create(Space::Primal), x0 = b.create(Space::Primal), xh = b.create(Space::Primal);
  int xb = b.create(Space::Primal), aty = b.create(Space::Primal), dx = b.create(Space::Primal);
  int y = b.create(Space::Dual), y0 = b.create(Space::Dual), yh = b.create(Space::Dual);
  int yb = b.create(Space::Dual), ax = b.create(Space::Dual), dy = b.create(Space::Dual);

  b.fill(dx, 0.0);
  b.fill(aty, 0.0);
  b.primal_step(dx, aty, 0.0, x);  // x = proj_X(0)
  b.fill(y, 0.0);
  const bool warm = ctx.apply_warm_start(x, y, policy);
  ctx.note_primal_weight(omega);
  if (warm) {  // the warm point may already be good enough (e.g. an unchanged re-solve)
    const KktStats kw = ctx.kkt(x, y);
    if (kw.finite() && ctx.record(kw, 0)) return ctx.finish(Status::Optimal, x, y, 0, "warm start already optimal");
  }
  b.copy(x0, x);
  b.copy(y0, y);
  b.copy(xh, x);
  b.copy(yh, y);

  PrimalWeightPid pid(omega);

  // Fixed-point residual r(z) from the last step's x̄, x̂, ȳ, ŷ (Δ = x̄ − x̂ = x̂ − z).
  auto fixed_point_residual = [&]() {
    b.copy(dx, xb);
    b.axpby(-1.0, xh, 1.0, dx);
    b.copy(dy, yb);
    b.axpby(-1.0, yh, 1.0, dy);
    b.spmv(dx, ax);  // ax is free here: it is recomputed before its next use
    const double px = b.dot(dx, dx), py = b.dot(dy, dy), cross = b.dot(dy, ax);
    const double r2 = omega * px + py / omega + 2.0 * eta * cross;
    return std::sqrt(std::max(r2, 0.0));
  };

  const int K = opt.check_every;
  std::int64_t it = 0, inner = 0;
  double r0 = -1.0, r_last = std::numeric_limits<double>::infinity();
  std::string msg = warm ? "warm start" : "";

  for (;;) {
    const double tau = eta / omega, sigma = eta * omega;
    for (int s = 0; s < K; ++s) {
      const double w = static_cast<double>(inner + 1) / static_cast<double>(inner + 2);
      const bool need_residual = (s == K - 1) || inner == 0;
      b.r2h_primal_fused(y, aty, x, x0, tau, w, rho, xh, xb);
      b.r2h_dual_fused(xb, ax, y, y0, sigma, w, rho, yh, need_residual ? yb : -1);
      if (inner == 0) r0 = fixed_point_residual();  // residual at the epoch's anchor
      ++inner;
      ++it;
      // The clock is read every iteration (negligible next to the SpMVs): checking only every
      // K iterations let a 3 s limit run to 12 s on a 1e6-row model (K iterations + setup).
      if (s + 1 < K && ctx.out_of_time()) {
        const KktStats kt = ctx.kkt(xh, yh);
        if (kt.finite() && ctx.record(kt, it)) return ctx.finish(Status::Optimal, xh, yh, it, msg);
        return ctx.finish(Status::TimeLimit, xh, yh, it, msg);
      }
    }

    const double r = fixed_point_residual();
    const KktStats k = ctx.kkt(xh, yh);
    if (!k.finite() || !std::isfinite(r)) return ctx.finish(Status::NumericalError, xh, yh, it, "non-finite iterate");
    if (ctx.record(k, it)) return ctx.finish(Status::Optimal, xh, yh, it, msg);
    if (it >= opt.iteration_limit) return ctx.finish(Status::IterationLimit, xh, yh, it, msg);
    if (ctx.out_of_time()) return ctx.finish(Status::TimeLimit, xh, yh, it, msg);
    if (it >= 4 * K) {  // dx, dy still hold T(z) − z from fixed_point_residual()
      std::string why;
      Status st = ctx.check_infeasibility(dx, dy, k, why);
      if (st == Status::NotSolved && inner >= K && (it / K) % 4 == 0) {
        // Second candidate: the drift since the epoch's anchor, z − z0. For an infeasible or
        // unbounded LP it also tends to the infimal displacement direction and is much less
        // noisy than a single T(z) − z (Applegate, Lubin, Hinder, Math. Program. 2023, use
        // iterate differences / normalized iterates the same way). Every 4th check only: it
        // costs another download + two fp64 SpMVs (adlittle + objective cut: 56M -> 84k
        // iterations; refinery T8760, feasible: iteration count unchanged).
        b.copy(dx, xh);
        b.axpby(-1.0, x0, 1.0, dx);
        b.copy(dy, yh);
        b.axpby(-1.0, y0, 1.0, dy);
        st = ctx.check_infeasibility(dx, dy, k, why);
        if (st != Status::NotSolved) why += " (drift since the restart anchor)";
      }
      if (st != Status::NotSolved) return ctx.finish(st, xh, yh, it, why);
    }

    const bool promoted = policy.on_check(k.rel_kkt(), it);
    if (promoted) msg += std::string(msg.empty() ? "" : "; ") + "fp32 -> fp64 at iteration " + std::to_string(it);
    // Ratio tests are only meaningful when both residuals are positive and finite: at an
    // extreme primal weight one of the steps can freeze in floating point, T(z) = z exactly,
    // and r = r0 = 0 would otherwise trigger a "sufficient decay" restart at every check.
    const bool ratios_ok = r0 > 0 && std::isfinite(r0) && r > 0;
    const bool restart = promoted ||
                         (ratios_ok && (r <= opt.restart_sufficient * r0 ||
                                        (r <= opt.restart_necessary * r0 && r > r_last))) ||
                         static_cast<double>(inner) >= opt.restart_artificial * static_cast<double>(it);
    r_last = r;
    if (!restart) continue;

    // ---- PID primal weight
    const double dxn = b.diff_norm2(xh, x0), dyn = b.diff_norm2(yh, y0);
    const double xn = opt.pid_noise_rel > 0 ? b.norm2(xh) : 0.0, yn = opt.pid_noise_rel > 0 ? b.norm2(yh) : 0.0;
    pid.update(opt, dxn, dyn, xn, yn, k.rel_primal(), k.rel_dual());
    omega = pid.omega;

    ctx.note_primal_weight(omega);
    if (opt.verbosity >= 2)
      std::fprintf(stderr, "  restart at %lld (inner %lld): r/r0 %.3f  omega %.4e  dx %.3e dy %.3e  |x| %.3e |y| %.3e\n",
                   static_cast<long long>(it), static_cast<long long>(inner), r / r0, omega, dxn, dyn, xn, yn);
    // ---- restart at T(z)
    b.copy(x, xh);
    b.copy(x0, xh);
    b.copy(y, yh);
    b.copy(y0, yh);
    inner = 0;
    r_last = std::numeric_limits<double>::infinity();
  }
}

void PrimalWeightPid::update(const EngineOptions& opt, double dxn, double dyn, double xn, double yn,
                             double rel_primal, double rel_dual) {
  // The residual-ratio guard only applies when both residuals are positive: a residual
  // that is exactly 0 (e.g. every column boxed ⇒ dual residual ≡ 0, as on recipe) is
  // not a sign of trouble, and treating it as one freezes ω forever.
  const bool both_pos = rel_dual > 0 && rel_primal > 0;
  const double ratio = both_pos ? rel_dual / rel_primal : 1.0;
  // Rounding-noise floor (r2hpdhg.h): a displacement this small carries no information about
  // the distance to the solution.
  // Not applied while an iterate is diverging (its norm more than doubles between restarts):
  // on an infeasible or unbounded LP one side drifts along a ray while the other converges,
  // and there the ω runaway is what speeds up the ray (and hence the certificate).
  const bool diverging = xn > 2.0 * (1.0 + last_xn) || yn > 2.0 * (1.0 + last_yn);
  last_xn = xn;
  last_yn = yn;
  const bool above_noise =
      diverging || (dxn > opt.pid_noise_rel * (1.0 + xn) && dyn > opt.pid_noise_rel * (1.0 + yn));
  if (above_noise && dxn > 1e-16 && dyn > 1e-16 && dxn < 1e12 && dyn < 1e12 && ratio > 1e-8 && ratio < 1e8) {
    const double e = std::log(dyn) - std::log(dxn) - std::log(omega);
    integral = opt.pid_integral_decay * integral + e;
    // damped: log ω moves by at most pid_max_log_step per restart
    const double step = opt.pid_kp * e + opt.pid_ki * integral + opt.pid_kd * (e - last_error);
    omega *= std::exp(std::clamp(step, -opt.pid_max_log_step, opt.pid_max_log_step));
    last_error = e;
  } else {
    omega = best_omega;
    integral = last_error = 0.0;
  }
  const double balance = std::fabs(std::log10(ratio));
  if (both_pos && balance < best_balance) {
    best_balance = balance;
    best_omega = omega;
  }
}

}  // namespace ps26119::pdhg
