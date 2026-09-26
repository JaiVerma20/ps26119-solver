// dense_simplex.cpp — dense bounded two-phase primal simplex in double-double (TEST ORACLE).
//
// References (textbook material, our own code):
//   V. Chvátal, "Linear Programming" (1983), ch. 8 (bounded variables), ch. 3 (Bland).
//   R. G. Bland, "New finite pivoting rules for the simplex method", Math. OR 2 (1977).
//   R. J. Vanderbei, "Linear Programming: Foundations and Extensions", ch. 9 (bounds).
//
// ---------------------------------------------------------------------------------------
// Formulation
//   The Model is  min sense·cᵀx  s.t. rl ≤ Ax ≤ ru, l ≤ x ≤ u.  We add one logical per row,
//   s_i = a_iᵀx, and one artificial per row, so every row reads
//        a_iᵀx − s_i + d_i r_i = 0 .
//   Columns of the constraint matrix M = [A | −I | D] are indexed k = 0..N−1 with
//   N = n + 2m: structurals first, then logicals (n+i), then artificials (n+m+i).
//
// Starting point
//   Every structural is nonbasic at a finite bound (lower if finite, else upper, else 0
//   for a free column). Let act_i = a_iᵀx. If rl_i ≤ act_i ≤ ru_i the logical s_i is basic
//   (feasible) and r_i is nonbasic fixed at 0 forever. Otherwise s_i is nonbasic at the
//   violated bound b_i and r_i is basic with d_i = sign(b_i − act_i), so r_i = |b_i − act_i|.
//   The starting basis matrix is diagonal with entries −1 or d_i: trivially invertible.
//
// Phase 1 minimises Σ r_i. Artificials that leave the basis are fixed at 0. If the phase-1
//   optimum exceeds the feasibility tolerance the model is Infeasible. Artificials still
//   basic (at 0, e.g. redundant equality rows) get bounds [0,0]; the ratio test then keeps
//   them at 0 and pivots them out when possible.
// Phase 2 minimises sense·cᵀx from the phase-1 basis.
//
// Iteration (one pivot)
//   Tableau T = B⁻¹M (m×N, dd). Basic values satisfy B x_B = −N x_N.
//   Reduced costs d_k = cost_k − c_Bᵀ T_k.
//   Entering (Bland): the SMALLEST index k that is nonbasic, not fixed, and improving:
//      at lower and d_k < −tol,  at upper and d_k > tol,  free (at 0) and |d_k| > tol.
//   Direction δ = −sign(d_k). Moving x_k by δt changes x_B by −δ t T_k.
//   Ratio test: t = min over basic rows with |T_ik| > pivot tol of the distance to the
//   bound being approached, and the entering variable's own range u_k − l_k (bound flip).
//   Ties in t go to the bound flip first, then (Bland) to the smallest basic var index.
//   t = ∞ in phase 2 ⇒ Unbounded.
//   Bland's rule guarantees termination in exact arithmetic; dd makes the rounding
//   perturbation ~1e-30, far below the tolerances, and we refactorise periodically.
//
// Duals
//   The logical s_i has column −e_i, so its reduced cost is d_{s_i} = c_Bᵀ B⁻¹ e_i = y_i.
//   Hence y is read off the logical reduced costs and z_j = d_j = c̃_j − (Aᵀy)_j. At the
//   optimum of the min-form problem: y_i ≥ 0 when s_i is at rl_i, ≤ 0 at ru_i (same for z).
//   We report y, z in the ORIGINAL objective sense: y_out = sense·y, z_out = sense·z.
//
// Invariants
//   * basis[r] is the variable basic in row r; pos[k] = r or −1 when nonbasic.
//   * every nonbasic variable sits exactly at a bound (or at 0 when free).
//   * after refactor(): T and x_B are recomputed from the ORIGINAL data (no drift).
//   * Optimal is reported only after a final refactorisation confirms primal and dual
//     feasibility; otherwise NumericalError (CLAUDE.md §5.4).
// ---------------------------------------------------------------------------------------
#include "oracle/dense_simplex.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

#include "la/dd.h"
#include "ps26119/tolerances.h"

namespace ps26119::oracle {
namespace {

using la::dd;

enum class NbState : std::uint8_t { Basic, AtLower, AtUpper, AtZero };

class DenseSimplex {
 public:
  DenseSimplex(const Model& model, const DenseSimplexOptions& opt)
      : M_(model), opt_(opt), m_(model.num_rows), n_(model.num_cols), N_(model.num_cols + 2 * model.num_rows) {}

  Solution run();

 private:
  // ---- data
  const Model& M_;
  const DenseSimplexOptions& opt_;
  int m_, n_, N_;
  std::vector<double> lo_, up_;     // bounds per variable (artificials: [0,inf) or [0,0])
  std::vector<double> dsign_;       // d_i per row
  std::vector<dd> cost_;            // current phase costs
  std::vector<dd> T_;               // m × N, row-major
  std::vector<dd> xb_;              // basic values per row
  std::vector<dd> val_;             // values of all variables (valid for nonbasic)
  std::vector<int> basis_, pos_;
  std::vector<NbState> state_;
  std::int64_t iterations_ = 0;
  std::chrono::steady_clock::time_point t0_;
  std::string message_;

  dd& t(int i, int k) { return T_[static_cast<std::size_t>(i) * N_ + k]; }

  // Column k of M = [A | −I | D] as (row, value) pairs.
  template <class F>
  void for_column(int k, F&& f) const {
    if (k < n_) {
      for (int p = M_.col_start[k]; p < M_.col_start[k + 1]; ++p) f(M_.row_index[p], M_.value[p]);
    } else if (k < n_ + m_) {
      f(k - n_, -1.0);
    } else {
      f(k - n_ - m_, dsign_[k - n_ - m_]);
    }
  }

  bool is_fixed(int k) const { return lo_[k] == up_[k]; }
  double elapsed() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

  void setup();
  bool refactor();  // false if the basis matrix is numerically singular
  dd reduced_cost(int k);
  enum class PhaseResult { Optimal, Unbounded, IterationLimit, TimeLimit, NumericalError };
  PhaseResult iterate();
  void pivot(int row, int k);
  double max_basic_infeasibility();
};

void DenseSimplex::setup() {
  lo_.assign(N_, 0.0);
  up_.assign(N_, 0.0);
  dsign_.assign(m_, 1.0);
  val_.assign(N_, dd(0.0));
  state_.assign(N_, NbState::AtLower);
  basis_.assign(m_, -1);
  pos_.assign(N_, -1);
  for (int j = 0; j < n_; ++j) {
    lo_[j] = M_.col_lower[j];
    up_[j] = M_.col_upper[j];
    if (std::isfinite(lo_[j])) {
      val_[j] = lo_[j];
      state_[j] = NbState::AtLower;
    } else if (std::isfinite(up_[j])) {
      val_[j] = up_[j];
      state_[j] = NbState::AtUpper;
    } else {
      val_[j] = 0.0;
      state_[j] = NbState::AtZero;
    }
  }
  // activity of the starting point, in dd
  std::vector<dd> act(m_, dd(0.0));
  for (int j = 0; j < n_; ++j)
    if (val_[j].hi != 0.0)
      for (int p = M_.col_start[j]; p < M_.col_start[j + 1]; ++p) act[M_.row_index[p]] += val_[j] * M_.value[p];

  for (int i = 0; i < m_; ++i) {
    const int s = n_ + i, r = n_ + m_ + i;
    lo_[s] = M_.row_lower[i];
    up_[s] = M_.row_upper[i];
    lo_[r] = 0.0;
    const bool feasible = act[i] >= dd(M_.row_lower[i]) && act[i] <= dd(M_.row_upper[i]);
    if (feasible) {
      basis_[i] = s;
      pos_[s] = i;
      state_[s] = NbState::Basic;
      up_[r] = 0.0;  // artificial never used
      val_[r] = 0.0;
      state_[r] = NbState::AtLower;
    } else {
      const double b = act[i] < dd(M_.row_lower[i]) ? M_.row_lower[i] : M_.row_upper[i];
      val_[s] = b;
      state_[s] = b == M_.row_lower[i] ? NbState::AtLower : NbState::AtUpper;
      dsign_[i] = (dd(b) - act[i]).hi > 0 ? 1.0 : -1.0;
      up_[r] = kInf;
      basis_[i] = r;
      pos_[r] = i;
      state_[r] = NbState::Basic;
    }
  }
  xb_.assign(m_, dd(0.0));
  T_.assign(static_cast<std::size_t>(m_) * N_, dd(0.0));
}

// Rebuild T = B⁻¹M and x_B = −B⁻¹ N x_N from the original data by dense Gaussian
// elimination with partial pivoting on [B | M | rhs] in double-double.
bool DenseSimplex::refactor() {
  const int W = N_ + 1;  // columns: M then rhs
  std::vector<dd> B(static_cast<std::size_t>(m_) * m_, dd(0.0));
  std::vector<dd> R(static_cast<std::size_t>(m_) * W, dd(0.0));
  for (int r = 0; r < m_; ++r) for_column(basis_[r], [&](int i, double v) { B[static_cast<std::size_t>(i) * m_ + r] = v; });
  for (int k = 0; k < N_; ++k) for_column(k, [&](int i, double v) { R[static_cast<std::size_t>(i) * W + k] = v; });
  // rhs = −Σ_{nonbasic k} M_k val_k
  for (int k = 0; k < N_; ++k) {
    if (state_[k] == NbState::Basic || val_[k].hi == 0.0) continue;
    for_column(k, [&](int i, double v) { R[static_cast<std::size_t>(i) * W + N_] -= val_[k] * v; });
  }
  for (int c = 0; c < m_; ++c) {
    int p = c;
    for (int i = c + 1; i < m_; ++i)
      if (la::abs(B[static_cast<std::size_t>(i) * m_ + c]) > la::abs(B[static_cast<std::size_t>(p) * m_ + c])) p = i;
    const dd piv = B[static_cast<std::size_t>(p) * m_ + c];
    if (std::fabs(piv.hi) < 1e-14) return false;
    if (p != c) {
      for (int k = 0; k < m_; ++k) std::swap(B[static_cast<std::size_t>(p) * m_ + k], B[static_cast<std::size_t>(c) * m_ + k]);
      for (int k = 0; k < W; ++k) std::swap(R[static_cast<std::size_t>(p) * W + k], R[static_cast<std::size_t>(c) * W + k]);
    }
    const dd inv = dd(1.0) / piv;
    for (int k = c; k < m_; ++k) B[static_cast<std::size_t>(c) * m_ + k] *= inv;
    for (int k = 0; k < W; ++k) R[static_cast<std::size_t>(c) * W + k] *= inv;
    for (int i = 0; i < m_; ++i) {
      if (i == c) continue;
      const dd f = B[static_cast<std::size_t>(i) * m_ + c];
      if (f.hi == 0.0) continue;
      for (int k = c; k < m_; ++k) B[static_cast<std::size_t>(i) * m_ + k] -= f * B[static_cast<std::size_t>(c) * m_ + k];
      for (int k = 0; k < W; ++k) {
        const dd& rc = R[static_cast<std::size_t>(c) * W + k];
        if (rc.hi != 0.0) R[static_cast<std::size_t>(i) * W + k] -= f * rc;
      }
    }
  }
  // Row c of the reduced system corresponds to basis position c (B columns were basis order).
  for (int r = 0; r < m_; ++r) {
    for (int k = 0; k < N_; ++k) t(r, k) = R[static_cast<std::size_t>(r) * W + k];
    xb_[r] = R[static_cast<std::size_t>(r) * W + N_];
    val_[basis_[r]] = xb_[r];
  }
  return true;
}

dd DenseSimplex::reduced_cost(int k) {
  dd d = cost_[k];
  for (int r = 0; r < m_; ++r) {
    const dd& cb = cost_[basis_[r]];
    if (cb.hi != 0.0) {
      const dd& a = t(r, k);
      if (a.hi != 0.0) d -= cb * a;
    }
  }
  return d;
}

void DenseSimplex::pivot(int row, int k) {
  const dd inv = dd(1.0) / t(row, k);
  for (int c = 0; c < N_; ++c)
    if (t(row, c).hi != 0.0) t(row, c) *= inv;
  t(row, k) = 1.0;
  for (int i = 0; i < m_; ++i) {
    if (i == row) continue;
    const dd f = t(i, k);
    if (f.hi == 0.0) continue;
    for (int c = 0; c < N_; ++c) {
      const dd& rc = t(row, c);
      if (rc.hi != 0.0) t(i, c) -= f * rc;
    }
    t(i, k) = 0.0;
  }
}

double DenseSimplex::max_basic_infeasibility() {
  double worst = 0.0;
  for (int r = 0; r < m_; ++r) {
    const int k = basis_[r];
    const double below = (dd(lo_[k]) - xb_[r]).hi / (1.0 + std::fabs(lo_[k]));
    const double above = (xb_[r] - dd(up_[k])).hi / (1.0 + std::fabs(up_[k]));
    if (std::isfinite(lo_[k])) worst = std::max(worst, below);
    if (std::isfinite(up_[k])) worst = std::max(worst, above);
  }
  return worst;
}

DenseSimplex::PhaseResult DenseSimplex::iterate() {
  const double ftol = tol::kOracleFeasibility;
  const double ptol = tol::kOraclePivot;
  int since_refactor = 0;
  for (;;) {
    if (iterations_ >= opt_.iteration_limit) return PhaseResult::IterationLimit;
    if ((iterations_ & 63) == 0 && elapsed() > opt_.time_limit) return PhaseResult::TimeLimit;
    if (since_refactor >= opt_.refactor_every) {
      if (!refactor()) return PhaseResult::NumericalError;
      since_refactor = 0;
    }

    // ---- pricing: Bland's rule (smallest eligible index)
    int enter = -1;
    dd d_enter;
    for (int k = 0; k < N_; ++k) {
      if (state_[k] == NbState::Basic || is_fixed(k)) continue;
      const dd d = reduced_cost(k);
      const double tolk = ftol * (1.0 + std::fabs(cost_[k].hi));
      const bool improving = (state_[k] == NbState::AtLower && d.hi < -tolk) ||
                             (state_[k] == NbState::AtUpper && d.hi > tolk) ||
                             (state_[k] == NbState::AtZero && std::fabs(d.hi) > tolk);
      if (improving) {
        enter = k;
        d_enter = d;
        break;
      }
    }
    if (enter < 0) return PhaseResult::Optimal;
    const double delta = d_enter.hi < 0 ? 1.0 : -1.0;

    // ---- ratio test
    dd t_best(kInf);
    int leave_row = -1;
    bool leave_to_upper = false;
    if (std::isfinite(lo_[enter]) && std::isfinite(up_[enter])) t_best = dd(up_[enter]) - dd(lo_[enter]);  // bound flip
    for (int r = 0; r < m_; ++r) {
      const dd& a = t(r, enter);
      if (std::fabs(a.hi) <= ptol) continue;
      const int k = basis_[r];
      const double rate = -delta * a.hi;  // sign of d x_B[r] / dt
      dd ratio;
      bool to_upper;
      if (rate < 0) {
        if (!std::isfinite(lo_[k])) continue;
        ratio = (xb_[r] - dd(lo_[k])) / la::abs(a);
        to_upper = false;
      } else {
        if (!std::isfinite(up_[k])) continue;
        ratio = (dd(up_[k]) - xb_[r]) / la::abs(a);
        to_upper = true;
      }
      if (ratio.hi < 0) ratio = 0.0;  // basic var marginally outside its bound
      bool better;
      if (!la::isfinite(t_best)) {
        better = true;  // (dd arithmetic with ∞ would produce NaN below)
      } else {
        const double eps = 1e-26 * std::max(1.0, std::fabs(t_best.hi));
        better = (ratio < t_best - dd(eps)) ||
                 (leave_row >= 0 && la::abs(ratio - t_best).hi <= eps && k < basis_[leave_row]);
      }
      if (better) {
        t_best = ratio;
        leave_row = r;
        leave_to_upper = to_upper;
      }
    }
    if (!la::isfinite(t_best)) return PhaseResult::Unbounded;

    // ---- update values
    const dd step = t_best * delta;
    for (int r = 0; r < m_; ++r) {
      const dd& a = t(r, enter);
      if (a.hi != 0.0) xb_[r] -= step * a;
    }
    ++iterations_;
    if (leave_row < 0) {  // bound flip, no basis change
      if (state_[enter] == NbState::AtLower) {
        state_[enter] = NbState::AtUpper;
        val_[enter] = up_[enter];
      } else {
        state_[enter] = NbState::AtLower;
        val_[enter] = lo_[enter];
      }
      continue;
    }
    const int leave = basis_[leave_row];
    const dd enter_value = val_[enter] + step;
    // leaving variable goes exactly to the bound it hit
    state_[leave] = leave_to_upper ? NbState::AtUpper : NbState::AtLower;
    val_[leave] = leave_to_upper ? up_[leave] : lo_[leave];
    if (is_fixed(leave)) state_[leave] = NbState::AtLower;
    pos_[leave] = -1;
    pivot(leave_row, enter);
    basis_[leave_row] = enter;
    pos_[enter] = leave_row;
    state_[enter] = NbState::Basic;
    xb_[leave_row] = enter_value;
    val_[enter] = enter_value;
    ++since_refactor;
    if (opt_.verbosity >= 2 && iterations_ % 100 == 0)
      std::fprintf(stderr, "oracle it %lld enter %d leave %d\n", static_cast<long long>(iterations_), enter, leave);
  }
}

Solution DenseSimplex::run() {
  t0_ = std::chrono::steady_clock::now();
  Solution sol;
  sol.engine = "oracle";
  sol.precision = "dd";
  auto finish = [&](Status st, const std::string& msg) {
    sol.status = st;
    sol.message = msg;
    sol.iterations = iterations_;
    sol.seconds = elapsed();
    return sol;
  };
  if (static_cast<std::int64_t>(m_) * N_ > opt_.max_tableau_entries)
    return finish(Status::NotSolved, "model too large for the dense oracle");

  setup();
  if (!refactor()) return finish(Status::NumericalError, "singular starting basis");

  // ---------------- phase 1
  cost_.assign(N_, dd(0.0));
  bool need_phase1 = false;
  for (int i = 0; i < m_; ++i) {
    cost_[n_ + m_ + i] = 1.0;
    if (state_[n_ + m_ + i] == NbState::Basic) need_phase1 = true;
  }
  if (need_phase1) {
    auto r = iterate();
    if (r == PhaseResult::IterationLimit) return finish(Status::IterationLimit, "phase 1");
    if (r == PhaseResult::TimeLimit) return finish(Status::TimeLimit, "phase 1");
    if (r != PhaseResult::Optimal) return finish(Status::NumericalError, "phase 1 failed");
    if (!refactor()) return finish(Status::NumericalError, "singular basis after phase 1");
    double bmax = 0.0;
    for (int i = 0; i < m_; ++i) {
      if (std::isfinite(M_.row_lower[i])) bmax = std::max(bmax, std::fabs(M_.row_lower[i]));
      if (std::isfinite(M_.row_upper[i])) bmax = std::max(bmax, std::fabs(M_.row_upper[i]));
    }
    double infeas = 0.0;
    for (int i = 0; i < m_; ++i) {
      const int k = n_ + m_ + i;
      infeas = std::max(infeas, std::fabs(val_[k].hi));
    }
    if (infeas > tol::kOracleFeasibility * (1.0 + bmax)) {
      sol.x.clear();
      return finish(Status::Infeasible, "phase 1 optimum > 0");
    }
  }
  // fix all artificials at zero
  for (int i = 0; i < m_; ++i) {
    const int k = n_ + m_ + i;
    up_[k] = 0.0;
    if (state_[k] != NbState::Basic) {
      val_[k] = 0.0;
      state_[k] = NbState::AtLower;
    }
  }

  // ---------------- phase 2
  cost_.assign(N_, dd(0.0));
  for (int j = 0; j < n_; ++j) cost_[j] = static_cast<double>(M_.sense) * M_.obj[j];
  for (int attempt = 0;; ++attempt) {
    auto r = iterate();
    if (r == PhaseResult::IterationLimit) return finish(Status::IterationLimit, "phase 2");
    if (r == PhaseResult::TimeLimit) return finish(Status::TimeLimit, "phase 2");
    if (r == PhaseResult::Unbounded) return finish(Status::Unbounded, "unbounded ray found in phase 2");
    if (r == PhaseResult::NumericalError) return finish(Status::NumericalError, "singular basis");
    // Final confirmation from the original data.
    if (!refactor()) return finish(Status::NumericalError, "singular final basis");
    if (max_basic_infeasibility() > tol::kOracleFeasibility)
      return finish(Status::NumericalError, "basic solution infeasible after refactorisation");
    bool dual_ok = true;
    for (int k = 0; k < N_ && dual_ok; ++k) {
      if (state_[k] == NbState::Basic || is_fixed(k)) continue;
      const double d = reduced_cost(k).hi, tolk = tol::kOracleFeasibility * (1.0 + std::fabs(cost_[k].hi));
      if ((state_[k] == NbState::AtLower && d < -tolk) || (state_[k] == NbState::AtUpper && d > tolk) ||
          (state_[k] == NbState::AtZero && std::fabs(d) > tolk))
        dual_ok = false;
    }
    if (dual_ok) break;
    if (attempt >= 3) return finish(Status::NumericalError, "dual infeasible after refactorisation");
  }

  // ---------------- extract the solution
  const double sense = M_.sense;
  sol.x.resize(n_);
  for (int j = 0; j < n_; ++j) sol.x[j] = val_[j].to_double();
  sol.y.resize(m_);
  for (int i = 0; i < m_; ++i) sol.y[i] = sense * reduced_cost(n_ + i).to_double();
  // z = c − Aᵀy recomputed in dd from the reported y (exact reduced costs of the basis).
  sol.z.resize(n_);
  for (int j = 0; j < n_; ++j) sol.z[j] = sense * reduced_cost(j).to_double();

  std::vector<dd> act(m_, dd(0.0));
  dd obj = M_.obj_offset;
  for (int j = 0; j < n_; ++j) {
    obj += val_[j] * M_.obj[j];
    for (int p = M_.col_start[j]; p < M_.col_start[j + 1]; ++p) act[M_.row_index[p]] += val_[j] * M_.value[p];
  }
  sol.row_activity.resize(m_);
  for (int i = 0; i < m_; ++i) sol.row_activity[i] = act[i].to_double();
  sol.objective = obj.to_double();

  // Dual objective (min form): Σ y_i·(active row bound) + Σ z_j·(active column bound).
  dd dobj = 0.0;
  auto bterm = [&](double v, double lo, double up) {
    if (v > 0 && std::isfinite(lo)) dobj += dd(v) * lo;
    if (v < 0 && std::isfinite(up)) dobj += dd(v) * up;
  };
  for (int i = 0; i < m_; ++i) bterm(sense * sol.y[i], M_.row_lower[i], M_.row_upper[i]);
  for (int j = 0; j < n_; ++j) bterm(sense * sol.z[j], M_.col_lower[j], M_.col_upper[j]);
  sol.dual_objective = sense * dobj.to_double() + M_.obj_offset;
  const double p = sense * (sol.objective - M_.obj_offset), d = dobj.to_double();
  sol.gap = std::fabs(p - d) / (1.0 + std::fabs(p) + std::fabs(d));
  sol.primal_residual = max_basic_infeasibility();
  sol.dual_residual = 0.0;
  return finish(Status::Optimal, "");
}

}  // namespace

Solution solve_dense_simplex(const Model& model, const DenseSimplexOptions& options) {
  DenseSimplex s(model, options);
  return s.run();
}

}  // namespace ps26119::oracle
