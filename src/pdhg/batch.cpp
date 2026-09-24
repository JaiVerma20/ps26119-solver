// batch.cpp — batched r²HPDHG: K LPs sharing one matrix, solved in lockstep with SpMM.
//
// Math per scenario is exactly r2hpdhg.h (reflected Halpern PDHG, constant step, fixed-point
// restarts, PID primal weight, relative-KKT termination with verifier-grade checks). What
// is shared: Ã = R A C (Ruiz + Pock–Chambolle scaling depends on A only), η = 0.998/‖Ã‖₂,
// and every sparse product. What is per scenario: the bound/objective scalars (bs_k, os_k),
// the scaled vectors, ω_k, the Halpern epoch, restart and PID state, the termination test.
//
// Layout: an n×K block is stored row-major, x[j·K + k], so the SpMM inner loop over the K
// scenarios is contiguous (vectorises) and each nonzero of Ã is read once per iteration.
//
// Difference from the single-LP engine (documented, not hidden): the epoch's reference
// residual r⁰_k is taken at the first check after a restart (K_check iterations later)
// instead of right after the restart, so that all scenarios share the extra SpMM that the
// residual needs. Restart criteria are otherwise identical.
//
// Invariants: frozen (converged / failed) scenarios are never updated again; every reported
// Optimal passed KktStats::converged(tolerance) on the ORIGINAL scenario data in fp64.
#include "ps26119/batch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "pdhg/backend.h"
#include "pdhg/engine.h"
#include "la/parallel.h"
#include "pdhg/scaling.h"
#include "ps26119/tolerances.h"

namespace ps26119 {
namespace {

using la::Csr;

// Y (rows × K) = M · X (cols × K), interleaved layout.
void spmm(const Csr<double>& M, const std::vector<double>& X, std::vector<double>& Y, int K) {
  la::parallel_for(
      M.rows,
      [&](std::int64_t b, std::int64_t e) {
        std::vector<double> acc(K);
        for (std::int64_t i = b; i < e; ++i) {
          std::fill(acc.begin(), acc.end(), 0.0);
          for (std::int64_t p = M.row_ptr[i]; p < M.row_ptr[i + 1]; ++p) {
            const double a = M.val[p];
            const double* x = &X[static_cast<std::size_t>(M.col[p]) * K];
            for (int k = 0; k < K; ++k) acc[k] += a * x[k];
          }
          std::copy(acc.begin(), acc.end(), &Y[static_cast<std::size_t>(i) * K]);
        }
      },
      1024);
}

inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct ScenarioData {  // original space, original sense
  std::vector<double> c, cl, cu, rl, ru;
};

}  // namespace

std::vector<Solution> solve_batch(const Model& base, const std::vector<Scenario>& scenarios, const Options& options) {
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  const int K = static_cast<int>(scenarios.size());
  std::vector<Solution> out(K);
  for (auto& s : out) {
    s.engine = "r2hpdhg-batch";
    s.precision = "fp64";
    s.model_fingerprint = base.fingerprint_hex();
  }
  auto fail_all = [&](const std::string& why) {
    for (auto& s : out) s.status = Status::NotSolved, s.message = why;
    return out;
  };
  if (K == 0) return out;
  if (auto err = base.validate(); !err.empty()) return fail_all("invalid base model: " + err);
  pdhg::EngineOptions eo = pdhg::engine_options_from(options);
  if (auto bad = pdhg::apply_engine_params(options, eo); !bad.empty()) return fail_all("unknown engine parameter '" + bad + "'");
  if (options.use_gpu) return fail_all("batch solve is CPU-only in this version");
  la::ThreadPool::instance().set_threads(options.threads);

  const int m = base.num_rows, n = base.num_cols;
  // ---- scenario data (original space) with validation
  std::vector<ScenarioData> D(K);
  for (int k = 0; k < K; ++k) {
    const Scenario& s = scenarios[k];
    auto pick = [](const std::vector<double>& v, const std::vector<double>& dflt, std::size_t len, bool& ok) {
      if (v.empty()) return dflt;
      ok = ok && v.size() == len;
      return v;
    };
    bool ok = true;
    D[k].c = pick(s.obj, base.obj, n, ok);
    D[k].cl = pick(s.col_lower, base.col_lower, n, ok);
    D[k].cu = pick(s.col_upper, base.col_upper, n, ok);
    D[k].rl = pick(s.row_lower, base.row_lower, m, ok);
    D[k].ru = pick(s.row_upper, base.row_upper, m, ok);
    for (int j = 0; ok && j < n; ++j) ok = D[k].cl[j] <= D[k].cu[j] && std::isfinite(D[k].c[j]);
    for (int i = 0; ok && i < m; ++i) ok = D[k].rl[i] <= D[k].ru[i];
    if (!ok) {
      out[k].status = Status::NotSolved;
      out[k].message = "scenario has wrong vector sizes or lower > upper";
    }
  }

  // ---- shared scaling of A (bound/objective scalars are per scenario, below)
  pdhg::ScalingOptions so = eo.scaling;
  so.bound_objective_rescaling = false;
  const pdhg::ScaledProblem sp = pdhg::make_scaled_problem(base, so);
  const double norm = la::estimate_norm2(sp.A, sp.At);
  const double eta = norm > 0 ? 0.998 / norm : 1.0;
  const double setup_shared = elapsed();

  // ---- per-scenario scaled vectors, interleaved
  const std::size_t NK = static_cast<std::size_t>(n) * K, MK = static_cast<std::size_t>(m) * K;
  std::vector<double> C(NK), L(NK), U(NK), RL(MK), RU(MK), bs(K, 1.0), os(K, 1.0);
  for (int k = 0; k < K; ++k) {
    if (eo.scaling.bound_objective_rescaling) {
      double b2 = 0, c2 = 0;
      for (int i = 0; i < m; ++i) {
        const double lo = std::isfinite(D[k].rl[i]) ? std::fabs(D[k].rl[i]) : 0.0;
        const double up = std::isfinite(D[k].ru[i]) ? std::fabs(D[k].ru[i]) : 0.0;
        const double b = sp.row_scale[i] * std::max(lo, up);
        b2 += b * b;
      }
      for (int j = 0; j < n; ++j) {
        const double c = sp.col_scale[j] * D[k].c[j];
        c2 += c * c;
      }
      bs[k] = 1.0 / (std::sqrt(b2) + 1.0);
      os[k] = 1.0 / (std::sqrt(c2) + 1.0);
    }
    for (int j = 0; j < n; ++j) {
      const std::size_t q = static_cast<std::size_t>(j) * K + k;
      C[q] = os[k] * sp.col_scale[j] * base.sense * D[k].c[j];
      L[q] = bs[k] * D[k].cl[j] / sp.col_scale[j];
      U[q] = bs[k] * D[k].cu[j] / sp.col_scale[j];
    }
    for (int i = 0; i < m; ++i) {
      const std::size_t q = static_cast<std::size_t>(i) * K + k;
      RL[q] = bs[k] * sp.row_scale[i] * D[k].rl[i];
      RU[q] = bs[k] * sp.row_scale[i] * D[k].ru[i];
    }
  }

  // ---- iterates
  std::vector<double> X(NK), X0(NK), XH(NK), XB(NK), ATY(NK), DX(NK);
  std::vector<double> Y(MK, 0.0), Y0(MK, 0.0), YH(MK, 0.0), YB(MK, 0.0), AX(MK), DY(MK), ADX(MK);
  for (std::size_t q = 0; q < NK; ++q) X[q] = clampd(0.0, L[q], U[q]);
  X0 = X;
  XH = X;

  std::vector<double> omega(K, 1.0), best_omega(K, 1.0), best_balance(K, 1e300), pid_int(K, 0.0), pid_last(K, 0.0);
  std::vector<double> r0(K, -1.0), r_last(K, std::numeric_limits<double>::infinity());
  std::vector<std::int64_t> inner(K, 0), fast_it(K, -1);
  std::vector<double> fast_s(K, -1.0);
  std::vector<char> active(K);
  for (int k = 0; k < K; ++k) active[k] = out[k].message.empty();  // invalid scenarios carry a message
  const double rho = eo.reflection, a2 = 2 * rho, b2 = 1 - 2 * rho;
  const int Kc = eo.check_every;
  std::int64_t it = 0;

  // Extract scenario k as original-space x, y_min.
  auto unscale = [&](int k, std::vector<double>& x, std::vector<double>& y) {
    x.resize(n);
    y.resize(m);
    for (int j = 0; j < n; ++j)
      x[j] = clampd(sp.col_scale[j] * XH[static_cast<std::size_t>(j) * K + k] / bs[k], D[k].cl[j], D[k].cu[j]);
    for (int i = 0; i < m; ++i) y[i] = sp.row_scale[i] * YH[static_cast<std::size_t>(i) * K + k] / os[k];
  };
  auto finish = [&](int k, Status st, const pdhg::KktStats& kk, const std::vector<double>& x,
                    const std::vector<double>& ymin, const std::string& msg) {
    Solution& s = out[k];
    s.status = st;
    s.message = msg;
    s.x = x;
    s.y.resize(m);
    for (int i = 0; i < m; ++i) s.y[i] = base.sense * ymin[i];
    s.z = D[k].c;
    for (int j = 0; j < n; ++j)
      for (int p = base.col_start[j]; p < base.col_start[j + 1]; ++p) s.z[j] -= base.value[p] * s.y[base.row_index[p]];
    s.row_activity = base.row_activity(x);
    double obj = base.obj_offset;
    for (int j = 0; j < n; ++j) obj += D[k].c[j] * x[j];
    s.objective = obj;
    s.dual_objective = base.sense * kk.dual_obj + base.obj_offset;
    s.primal_residual = kk.rel_primal();
    s.dual_residual = kk.rel_dual();
    s.gap = kk.rel_gap();
    s.iterations = it;
    s.seconds = elapsed();
    s.setup_seconds = setup_shared;
    s.iterations_to_fast = fast_it[k];
    s.seconds_to_fast = fast_s[k];
    s.primal_weight = omega[k];
    active[k] = 0;
  };

  std::vector<double> xk, yk;
  while (std::any_of(active.begin(), active.end(), [](char a) { return a != 0; })) {
    for (int s = 0; s < Kc; ++s) {
      spmm(sp.At, Y, ATY, K);
      for (int j = 0; j < n; ++j) {
        const std::size_t base_q = static_cast<std::size_t>(j) * K;
        for (int k = 0; k < K; ++k) {
          if (!active[k]) continue;
          const std::size_t q = base_q + k;
          const double tau = eta / omega[k];
          const double w = static_cast<double>(inner[k] + 1) / static_cast<double>(inner[k] + 2);
          const double h = clampd(X[q] - tau * (C[q] - ATY[q]), L[q], U[q]);
          XH[q] = h;
          XB[q] = 2 * h - X[q];
          X[q] = w * (a2 * h + b2 * X[q]) + (1 - w) * X0[q];
        }
      }
      spmm(sp.A, XB, AX, K);
      for (int i = 0; i < m; ++i) {
        const std::size_t base_q = static_cast<std::size_t>(i) * K;
        for (int k = 0; k < K; ++k) {
          if (!active[k]) continue;
          const std::size_t q = base_q + k;
          const double sigma = eta * omega[k];
          const double w = static_cast<double>(inner[k] + 1) / static_cast<double>(inner[k] + 2);
          const double v = AX[q] - Y[q] / sigma;
          const double h = Y[q] - sigma * AX[q] + sigma * clampd(v, RL[q], RU[q]);
          YH[q] = h;
          YB[q] = 2 * h - Y[q];
          Y[q] = w * (a2 * h + b2 * Y[q]) + (1 - w) * Y0[q];
        }
      }
      for (int k = 0; k < K; ++k)
        if (active[k]) ++inner[k];
    }
    it += Kc;

    // ---- fixed-point residuals of all scenarios (one shared SpMM)
    for (std::size_t q = 0; q < NK; ++q) DX[q] = XB[q] - XH[q];
    for (std::size_t q = 0; q < MK; ++q) DY[q] = YB[q] - YH[q];
    spmm(sp.A, DX, ADX, K);
    std::vector<double> px(K, 0.0), py(K, 0.0), cr(K, 0.0);
    for (std::size_t q = 0; q < NK; ++q) px[q % K] += DX[q] * DX[q];
    for (std::size_t q = 0; q < MK; ++q) py[q % K] += DY[q] * DY[q], cr[q % K] += DY[q] * ADX[q];

    const bool over_iter = it >= eo.iteration_limit, over_time = elapsed() > eo.time_limit;
    for (int k = 0; k < K; ++k) {
      if (!active[k]) continue;
      unscale(k, xk, yk);
      const pdhg::LpView lp{m, n, base.sense, D[k].c.data(), D[k].cl.data(), D[k].cu.data(), D[k].rl.data(),
                            D[k].ru.data()};
      const pdhg::KktStats kk = pdhg::kkt_general(sp.A_orig, sp.At_orig, lp, xk, yk);
      if (!kk.finite()) {
        finish(k, Status::NumericalError, kk, xk, yk, "non-finite iterate");
        continue;
      }
      if (fast_it[k] < 0 && kk.rel_kkt() <= tol::kFirstOrderFast) fast_it[k] = it, fast_s[k] = elapsed();
      if (kk.converged(eo.tolerance)) {
        finish(k, Status::Optimal, kk, xk, yk, "");
        continue;
      }
      if (over_iter || over_time) {
        finish(k, over_iter ? Status::IterationLimit : Status::TimeLimit, kk, xk, yk, "");
        continue;
      }
      const double r2 = omega[k] * px[k] + py[k] / omega[k] + 2.0 * eta * cr[k];
      const double r = std::sqrt(std::max(r2, 0.0));
      if (r0[k] < 0) {  // first check of the epoch: reference residual
        r0[k] = r;
        r_last[k] = r;
        if (!(static_cast<double>(inner[k]) >= eo.restart_artificial * static_cast<double>(it))) continue;
      }
      const bool ratios_ok = r0[k] > 0 && std::isfinite(r0[k]) && r > 0;
      const bool restart = (ratios_ok && (r <= eo.restart_sufficient * r0[k] ||
                                          (r <= eo.restart_necessary * r0[k] && r > r_last[k]))) ||
                           static_cast<double>(inner[k]) >= eo.restart_artificial * static_cast<double>(it);
      r_last[k] = r;
      if (!restart) continue;
      // PID primal weight (same rule as r2hpdhg.cpp)
      double dxn = 0, dyn = 0;
      for (int j = 0; j < n; ++j) {
        const std::size_t q = static_cast<std::size_t>(j) * K + k;
        dxn += (XH[q] - X0[q]) * (XH[q] - X0[q]);
      }
      for (int i = 0; i < m; ++i) {
        const std::size_t q = static_cast<std::size_t>(i) * K + k;
        dyn += (YH[q] - Y0[q]) * (YH[q] - Y0[q]);
      }
      dxn = std::sqrt(dxn);
      dyn = std::sqrt(dyn);
      const bool both_pos = kk.rel_dual() > 0 && kk.rel_primal() > 0;
      const double ratio = both_pos ? kk.rel_dual() / kk.rel_primal() : 1.0;
      if (dxn > 1e-16 && dyn > 1e-16 && dxn < 1e12 && dyn < 1e12 && ratio > 1e-8 && ratio < 1e8) {
        const double e = std::log(dyn) - std::log(dxn) - std::log(omega[k]);
        pid_int[k] = eo.pid_integral_decay * pid_int[k] + e;
        const double step = eo.pid_kp * e + eo.pid_ki * pid_int[k] + eo.pid_kd * (e - pid_last[k]);
        omega[k] *= std::exp(std::clamp(step, -eo.pid_max_log_step, eo.pid_max_log_step));
        pid_last[k] = e;
      } else {
        omega[k] = best_omega[k];
        pid_int[k] = pid_last[k] = 0.0;
      }
      if (both_pos && std::fabs(std::log10(ratio)) < best_balance[k]) {
        best_balance[k] = std::fabs(std::log10(ratio));
        best_omega[k] = omega[k];
      }
      for (int j = 0; j < n; ++j) {
        const std::size_t q = static_cast<std::size_t>(j) * K + k;
        X[q] = X0[q] = XH[q];
      }
      for (int i = 0; i < m; ++i) {
        const std::size_t q = static_cast<std::size_t>(i) * K + k;
        Y[q] = Y0[q] = YH[q];
      }
      inner[k] = 0;
      r0[k] = -1.0;
      r_last[k] = std::numeric_limits<double>::infinity();
    }
    if (eo.verbosity >= 2) {
      int act = 0;
      for (char a : active) act += a;
      std::fprintf(stderr, "batch it %lld  t %.2fs  active %d/%d\n", static_cast<long long>(it), elapsed(), act, K);
    }
  }
  return out;
}

}  // namespace ps26119
