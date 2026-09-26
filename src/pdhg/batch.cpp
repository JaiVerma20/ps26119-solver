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
// The epoch's reference residual r⁰_k is measured right after the first step of the epoch,
// exactly as in the single-LP engine: whenever any scenario has just (re)started, one extra
// SpMM (shared by all scenarios) computes the fixed-point residuals. (A first version took
// r⁰ at the next check instead; on refinery price scenarios that cost 1.7× the iterations.)
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

// Y (rows × W) = M · X (cols × W), interleaved layout; W known at compile time.
template <int W>
void spmm_fixed(const Csr<double>& M, const double* X, double* Y) {
  la::parallel_for(
      M.rows,
      [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t i = b; i < e; ++i) {
          double acc[W] = {};
          for (std::int64_t p = M.row_ptr[i]; p < M.row_ptr[i + 1]; ++p) {
            const double a = M.val[p];
            const double* x = X + static_cast<std::size_t>(M.col[p]) * W;
            for (int k = 0; k < W; ++k) acc[k] += a * x[k];
          }
          double* y = Y + static_cast<std::size_t>(i) * W;
          for (int k = 0; k < W; ++k) y[k] = acc[k];
        }
      },
      1024);
}

void spmm_generic(const Csr<double>& M, const double* X, double* Y, int W) {
  la::parallel_for(
      M.rows,
      [&](std::int64_t b, std::int64_t e) {
        std::vector<double> acc(W);
        for (std::int64_t i = b; i < e; ++i) {
          std::fill(acc.begin(), acc.end(), 0.0);
          for (std::int64_t p = M.row_ptr[i]; p < M.row_ptr[i + 1]; ++p) {
            const double a = M.val[p];
            const double* x = X + static_cast<std::size_t>(M.col[p]) * W;
            for (int k = 0; k < W; ++k) acc[k] += a * x[k];
          }
          std::copy(acc.begin(), acc.end(), Y + static_cast<std::size_t>(i) * W);
        }
      },
      1024);
}

void spmm(const Csr<double>& M, const std::vector<double>& X, std::vector<double>& Y, int W) {
  switch (W) {
    case 1: return spmm_fixed<1>(M, X.data(), Y.data());
    case 2: return spmm_fixed<2>(M, X.data(), Y.data());
    case 3: return spmm_fixed<3>(M, X.data(), Y.data());
    case 4: return spmm_fixed<4>(M, X.data(), Y.data());
    case 5: return spmm_fixed<5>(M, X.data(), Y.data());
    case 6: return spmm_fixed<6>(M, X.data(), Y.data());
    case 7: return spmm_fixed<7>(M, X.data(), Y.data());
    case 8: return spmm_fixed<8>(M, X.data(), Y.data());
    case 12: return spmm_fixed<12>(M, X.data(), Y.data());
    case 16: return spmm_fixed<16>(M, X.data(), Y.data());
    default: return spmm_generic(M, X.data(), Y.data(), W);
  }
}

// Keeps the slots listed in `keep` (in order) of an interleaved rows × W array.
void compact(std::vector<double>& v, std::size_t rows, int W, const std::vector<int>& keep) {
  const int W2 = static_cast<int>(keep.size());
  std::vector<double> nv(rows * W2);
  for (std::size_t r = 0; r < rows; ++r)
    for (int t = 0; t < W2; ++t) nv[r * W2 + t] = v[r * W + keep[t]];
  v.swap(nv);
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
    // A rejected scenario may have wrong-sized vectors: never read them (found by ASan).
    if (!out[k].message.empty()) continue;
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

  // ---- iterates, in SLOTS: only unfinished scenarios occupy a column of the interleaved
  // arrays; when scenarios finish, the arrays are compacted so later iterations only pay
  // for the scenarios still running. slot → scenario id: sid[t].
  std::vector<int> sid;
  for (int k = 0; k < K; ++k)
    if (out[k].message.empty()) sid.push_back(k);  // invalid scenarios carry a message
  {
    std::vector<int> keep(sid.begin(), sid.end());
    for (auto* v : {&C, &L, &U}) compact(*v, n, K, keep);
    for (auto* v : {&RL, &RU}) compact(*v, m, K, keep);
  }
  int W = static_cast<int>(sid.size());
  std::vector<double> X(static_cast<std::size_t>(n) * W), X0, XH, XB(X.size()), ATY(X.size()), DX(X.size());
  std::vector<double> Y(static_cast<std::size_t>(m) * W, 0.0), Y0, YH, YB(Y.size()), AX(Y.size()), DY(Y.size()),
      ADX(Y.size());
  for (std::size_t q = 0; q < X.size(); ++q) X[q] = clampd(0.0, L[q], U[q]);
  X0 = X;
  XH = X;
  Y0 = Y;
  YH = Y;

  // per-scenario state (indexed by scenario id)
  std::vector<double> omega(K, 1.0), best_omega(K, 1.0), best_balance(K, 1e300), pid_int(K, 0.0), pid_last(K, 0.0);
  std::vector<double> r0(K, -1.0), r_last(K, std::numeric_limits<double>::infinity());
  std::vector<std::int64_t> inner(K, 0), fast_it(K, -1);
  std::vector<double> fast_s(K, -1.0);
  const double rho = eo.reflection, a2 = 2 * rho, b2 = 1 - 2 * rho;
  const int Kc = eo.check_every;
  std::int64_t it = 0;
  // per-slot step coefficients, refreshed every iteration
  std::vector<double> ctau, csig, cw;

  // Extract the scenario in slot t as original-space x, y_min.
  auto unscale = [&](int t, std::vector<double>& x, std::vector<double>& y) {
    const int k = sid[t];
    x.resize(n);
    y.resize(m);
    for (int j = 0; j < n; ++j)
      x[j] = clampd(sp.col_scale[j] * XH[static_cast<std::size_t>(j) * W + t] / bs[k], D[k].cl[j], D[k].cu[j]);
    for (int i = 0; i < m; ++i) y[i] = sp.row_scale[i] * YH[static_cast<std::size_t>(i) * W + t] / os[k];
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
  };

  // Fixed-point residuals of all slots from the last step's x̄, x̂, ȳ, ŷ (one SpMM).
  std::vector<double> resid;
  auto compute_residuals = [&]() {
    for (std::size_t q = 0; q < X.size(); ++q) DX[q] = XB[q] - XH[q];
    for (std::size_t q = 0; q < Y.size(); ++q) DY[q] = YB[q] - YH[q];
    spmm(sp.A, DX, ADX, W);
    std::vector<double> px(W, 0.0), py(W, 0.0), cr(W, 0.0);
    for (std::size_t q = 0; q < X.size(); ++q) px[q % W] += DX[q] * DX[q];
    for (std::size_t q = 0; q < Y.size(); ++q) py[q % W] += DY[q] * DY[q], cr[q % W] += DY[q] * ADX[q];
    resid.assign(W, 0.0);
    for (int t = 0; t < W; ++t) {
      const double om = omega[sid[t]];
      resid[t] = std::sqrt(std::max(om * px[t] + py[t] / om + 2.0 * eta * cr[t], 0.0));
    }
  };

  std::vector<double> xk, yk;
  while (W > 0) {
    for (int step = 0; step < Kc; ++step) {
      ctau.resize(W);
      csig.resize(W);
      cw.resize(W);
      for (int t = 0; t < W; ++t) {
        const int k = sid[t];
        ctau[t] = eta / omega[k];
        csig[t] = eta * omega[k];
        cw[t] = static_cast<double>(inner[k] + 1) / static_cast<double>(inner[k] + 2);
      }
      spmm(sp.At, Y, ATY, W);
      la::parallel_for(n, [&](std::int64_t jb, std::int64_t je) {
        for (std::int64_t j = jb; j < je; ++j) {
          const std::size_t base_q = static_cast<std::size_t>(j) * W;
          for (int t = 0; t < W; ++t) {
            const std::size_t q = base_q + t;
            const double h = clampd(X[q] - ctau[t] * (C[q] - ATY[q]), L[q], U[q]);
            XH[q] = h;
            XB[q] = 2 * h - X[q];
            X[q] = cw[t] * (a2 * h + b2 * X[q]) + (1 - cw[t]) * X0[q];
          }
        }
      });
      spmm(sp.A, XB, AX, W);
      la::parallel_for(m, [&](std::int64_t ib, std::int64_t ie) {
        for (std::int64_t i = ib; i < ie; ++i) {
          const std::size_t base_q = static_cast<std::size_t>(i) * W;
          for (int t = 0; t < W; ++t) {
            const std::size_t q = base_q + t;
            const double sg = csig[t];
            const double v = AX[q] - Y[q] / sg;
            const double h = Y[q] - sg * AX[q] + sg * clampd(v, RL[q], RU[q]);
            YH[q] = h;
            YB[q] = 2 * h - Y[q];
            Y[q] = cw[t] * (a2 * h + b2 * Y[q]) + (1 - cw[t]) * Y0[q];
          }
        }
      });
      // epoch start: reference residual r⁰ right after the first step (as in r2hpdhg.cpp)
      bool any_new = false;
      for (int t = 0; t < W; ++t) any_new = any_new || inner[sid[t]] == 0;
      if (any_new) {
        compute_residuals();
        for (int t = 0; t < W; ++t)
          if (inner[sid[t]] == 0) r0[sid[t]] = resid[t];
      }
      for (int t = 0; t < W; ++t) ++inner[sid[t]];
    }
    it += Kc;

    compute_residuals();  // at the check
    const bool over_iter = it >= eo.iteration_limit, over_time = elapsed() > eo.time_limit;
    std::vector<int> keep;
    for (int t = 0; t < W; ++t) {
      const int k = sid[t];
      unscale(t, xk, yk);
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
      keep.push_back(t);
      const double r = resid[t];
      const bool ratios_ok = r0[k] > 0 && std::isfinite(r0[k]) && r > 0;
      const bool restart = (ratios_ok && (r <= eo.restart_sufficient * r0[k] ||
                                          (r <= eo.restart_necessary * r0[k] && r > r_last[k]))) ||
                           static_cast<double>(inner[k]) >= eo.restart_artificial * static_cast<double>(it);
      r_last[k] = r;
      if (!restart) continue;
      // PID primal weight (same rule as r2hpdhg.cpp)
      double dxn = 0, dyn = 0;
      for (int j = 0; j < n; ++j) {
        const std::size_t q = static_cast<std::size_t>(j) * W + t;
        dxn += (XH[q] - X0[q]) * (XH[q] - X0[q]);
      }
      for (int i = 0; i < m; ++i) {
        const std::size_t q = static_cast<std::size_t>(i) * W + t;
        dyn += (YH[q] - Y0[q]) * (YH[q] - Y0[q]);
      }
      dxn = std::sqrt(dxn);
      dyn = std::sqrt(dyn);
      const bool both_pos = kk.rel_dual() > 0 && kk.rel_primal() > 0;
      const double ratio = both_pos ? kk.rel_dual() / kk.rel_primal() : 1.0;
      if (dxn > 1e-16 && dyn > 1e-16 && dxn < 1e12 && dyn < 1e12 && ratio > 1e-8 && ratio < 1e8) {
        const double e = std::log(dyn) - std::log(dxn) - std::log(omega[k]);
        pid_int[k] = eo.pid_integral_decay * pid_int[k] + e;
        const double stp = eo.pid_kp * e + eo.pid_ki * pid_int[k] + eo.pid_kd * (e - pid_last[k]);
        omega[k] *= std::exp(std::clamp(stp, -eo.pid_max_log_step, eo.pid_max_log_step));
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
        const std::size_t q = static_cast<std::size_t>(j) * W + t;
        X[q] = X0[q] = XH[q];
      }
      for (int i = 0; i < m; ++i) {
        const std::size_t q = static_cast<std::size_t>(i) * W + t;
        Y[q] = Y0[q] = YH[q];
      }
      inner[k] = 0;
      r_last[k] = std::numeric_limits<double>::infinity();
    }
    if (static_cast<int>(keep.size()) != W) {  // some scenarios finished: compact
      const std::size_t nn = static_cast<std::size_t>(n), mm = static_cast<std::size_t>(m);
      for (auto* v : {&C, &L, &U, &X, &X0, &XH, &XB, &ATY, &DX}) compact(*v, nn, W, keep);
      for (auto* v : {&RL, &RU, &Y, &Y0, &YH, &YB, &AX, &DY, &ADX}) compact(*v, mm, W, keep);
      std::vector<int> nsid;
      for (int t : keep) nsid.push_back(sid[t]);
      sid.swap(nsid);
      W = static_cast<int>(sid.size());
    }
    if (eo.verbosity >= 2)
      std::fprintf(stderr, "batch it %lld  t %.2fs  running %d/%d\n", static_cast<long long>(it), elapsed(), W, K);
  }
  return out;
}

}  // namespace ps26119
