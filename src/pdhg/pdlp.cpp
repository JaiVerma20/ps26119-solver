// pdlp.cpp — PDLP-style restarted PDHG. Math, derivation and citations: pdlp.h.
#include "pdhg/pdlp.h"

#include <cmath>
#include <utility>

namespace ps26119::pdhg {

Solution solve_pdlp(const Model& model, const EngineOptions& opt) {
  EngineContext ctx(model, opt, "pdlp");
  if (!ctx.ok()) {
    Solution s;
    s.engine = "pdlp";
    s.status = Status::NotSolved;
    s.message = ctx.error();
    return s;
  }
  Backend& b = ctx.backend();
  PrecisionPolicy policy(b, opt);
  const double eta = ctx.eta();
  double omega = ctx.initial_primal_weight();

  // iterate, next iterate, averages, restart anchors, work vectors
  int x = b.create(Space::Primal), xn = b.create(Space::Primal), xa = b.create(Space::Primal);
  int x0 = b.create(Space::Primal), xbar = b.create(Space::Primal), aty = b.create(Space::Primal);
  int y = b.create(Space::Dual), yn = b.create(Space::Dual), ya = b.create(Space::Dual);
  int y0 = b.create(Space::Dual), ax = b.create(Space::Dual);

  // Start at x = proj_X(0), y = 0: one primal step with c = 0 is not available, so project
  // via primal_step with τ = 0 on a zero vector.
  b.fill(xbar, 0.0);
  b.fill(aty, 0.0);
  b.primal_step(xbar, aty, 0.0, x);
  b.fill(y, 0.0);
  b.copy(x0, x);
  b.copy(y0, y);
  b.copy(xa, x);
  b.copy(ya, y);

  KktStats k0 = b.kkt(x, y);
  ctx.record(k0, 0);
  double restart_kkt = k0.rel_kkt(), last_candidate = 1e300;
  std::int64_t it = 0, inner = 0;
  const int K = opt.check_every;
  std::string msg;

  for (;;) {
    const double tau = eta / omega, sigma = eta * omega;
    for (int s = 0; s < K; ++s) {
      b.spmv_t(y, aty);
      b.primal_step(x, aty, tau, xn);
      b.copy(xbar, xn);
      b.axpby(-1.0, x, 2.0, xbar);  // x̄ = 2x⁺ − x
      b.spmv(xbar, ax);
      b.dual_step(y, ax, sigma, yn);
      std::swap(x, xn);
      std::swap(y, yn);
      ++inner;
      const double wa = 1.0 / static_cast<double>(inner);
      b.axpby(wa, x, 1.0 - wa, xa);
      b.axpby(wa, y, 1.0 - wa, ya);
    }
    it += K;

    const KktStats kc = b.kkt(x, y), ka = b.kkt(xa, ya);
    const bool use_avg = ka.finite() && ka.rel_kkt() < kc.rel_kkt();
    const KktStats& kk = use_avg ? ka : kc;
    const int cx = use_avg ? xa : x, cy = use_avg ? ya : y;
    if (!kk.finite()) return ctx.finish(Status::NumericalError, x, y, it, "non-finite KKT statistics");
    if (ctx.record(kk, it)) return ctx.finish(Status::Optimal, cx, cy, it, msg);
    if (it >= opt.iteration_limit) return ctx.finish(Status::IterationLimit, cx, cy, it, msg);
    if (ctx.out_of_time()) return ctx.finish(Status::TimeLimit, cx, cy, it, msg);
    if (it >= 4 * K) {  // Δ = z_k − z_{k−1}; xn/yn hold z_{k−1} after the swap
      b.axpby(1.0, x, -1.0, xn);  // xn ← x − xn   (xn is overwritten next step anyway)
      b.axpby(1.0, y, -1.0, yn);
      std::string why;
      const Status st = ctx.check_infeasibility(xn, yn, kc, why);
      if (st != Status::NotSolved) return ctx.finish(st, x, y, it, why);
    }

    const double e = kk.rel_kkt();
    const bool promoted = policy.on_check(e, it);
    if (promoted) msg = "fp32 -> fp64 at iteration " + std::to_string(it);
    const bool restart = promoted || e <= 0.2 * restart_kkt ||
                         (e <= opt.pdlp_restart_necessary * restart_kkt && e > last_candidate) ||
                         static_cast<double>(inner) >= 0.36 * static_cast<double>(it);
    last_candidate = e;
    if (restart) {
      const double dx = b.diff_norm2(cx, x0), dy = b.diff_norm2(cy, y0);
      if (dx > 1e-10 && dy > 1e-10) {
        const double th = opt.pdlp_primal_weight_smoothing;
        omega = std::exp(th * std::log(dy / dx) + (1.0 - th) * std::log(omega));
      }
      if (cx != x) b.copy(x, cx);
      if (cy != y) b.copy(y, cy);
      b.copy(x0, x);
      b.copy(y0, y);
      b.copy(xa, x);
      b.copy(ya, y);
      inner = 0;
      restart_kkt = e;
      last_candidate = 1e300;
    }
  }
}

}  // namespace ps26119::pdhg
