// primal_simplex.cpp - bounded revised primal simplex on top of SparseLU.
// Origin: gpuopt src/simplex/primal_simplex.cpp (Shivanshu Vats, c192dd0); see primal_simplex.h.
//
// Internal (scaled) form:   min c^T z   s.t.  [A I] z = 0,   l <= z <= u
//   z_j, j <  n : structural variables (scaled by the column factors)
//   z_j, j >= n : logical of row i = j - n, equal to -(scaled a_i x); its
//                 bounds are the negated, scaled row bounds.
//
// One iteration:
//   1. phase   : phase 1 if any basic variable violates a bound by more than
//                the primal tolerance (costs -1 below / +1 above), else phase 2
//   2. duals   : y = B^{-T} c_B,  d_j = c_j - a_j^T y   for nonbasic j
//   3. pricing : entering q = attractive d_j with the best Devex score d_j^2 / w_j
//   4. FTRAN   : alpha = B^{-1} a_q  (how the basics move with q)
//   5. ratio   : Harris two-pass test; or a bound flip of q; or unbounded
//   6. update  : move the primal values, swap q into the basis (PFI update),
//                update Devex weights from the pivot row
// Any "done" verdict (optimal / infeasible / unbounded) is re-checked with a
// fresh factorization before it is believed.

#include "simplex/primal_simplex.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "la/sparse_lu.h"
#include "simplex/simplex_scaling.h"

namespace ps26119::simplex {
using la::build_basis_matrix;
using la::SparseLU;
using la::SparseMatrixCSC;

namespace {

enum class VarStatus : char { kBasic, kAtLower, kAtUpper, kFree };

class PrimalSimplex {
 public:
  PrimalSimplex(const Model& lp, const SimplexOptions& opt) : lp_(lp), A_orig_(la::csc_from_model(lp)), opt_(opt) {}

  Solution run();

 private:
  // ------------------------------------------------------------- setup
  void build_internal_problem();
  void initial_basis();
  void perturb_costs();

  // ----------------------------------------------------- linear algebra
  template <class F>
  void for_column(int j, F f) const {
    if (j < n_) {
      for (int k = A_.col_start[j]; k < A_.col_start[j + 1]; ++k) f(A_.row_index[k], A_.value[k]);
    } else {
      f(j - n_, 1.0);
    }
  }
  double column_dot(int j, const std::vector<double>& v) const {
    double s = 0.0;
    for_column(j, [&](int i, double a) { s += a * v[i]; });
    return s;
  }
  void refactor();
  void compute_primal();
  void compute_duals(bool phase1);

  // ---------------------------------------------------------- iteration
  bool below(int v) const { return x_[v] < lower_[v] - ptol_; }
  bool above(int v) const { return x_[v] > upper_[v] + ptol_; }
  bool any_infeasible() const;
  int choose_entering(int& dir) const;

  struct Ratio {
    enum Kind { kPivot, kFlip, kUnbounded } kind = kUnbounded;
    int row = -1;
    double theta = 0.0;
    double leave_value = 0.0;
  };
  Ratio ratio_test(int q, int dir) const;
  bool pivot(int q, int dir, const Ratio& ratio);
  void update_devex(int q, int r);
  void make_nonbasic(int v);

  // ----------------------------------------------------------- results
  double internal_objective() const;
  Solution finish(Status status, const std::string& note);
  double elapsed() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

  const Model& lp_;
  const SparseMatrixCSC A_orig_;  // unscaled copy of the model's A
  const SimplexOptions& opt_;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();

  // Largest primal / dual violation of the current point measured in the
  // ORIGINAL (unscaled) model; used to confirm optimality.
  void unscaled_violations(double& primal, double& dual) const;

  int m_ = 0, n_ = 0, nt_ = 0;
  double sense_ = 1.0;
  double cost_scale_ = 1.0;           // internal costs = cost_scale_ * scaled costs
  double ptol_ = 0.0, dtol_ = 0.0;    // working tolerances (tightened if needed)
  int tightenings_ = 0;
  Scaling scale_;
  SparseMatrixCSC A_;  // scaled
  std::vector<double> lower_, upper_, cost_, cost_orig_;

  std::vector<int> basis_, pos_;  // position -> variable, variable -> position (-1 if nonbasic)
  std::vector<VarStatus> status_;
  std::vector<double> x_, y_, d_, weight_, alpha_, rho_, row_alpha_;
  SparseLU lu_;

  bool fresh_ = false;      // no updates since the last refactorization
  bool perturbed_ = false;
  bool bland_ = false;      // anti-stalling mode
  long long iterations_ = 0, phase1_iterations_ = 0, flips_ = 0, refactors_ = 0;
  long long degenerate_run_ = 0, bland_episodes_ = 0, devex_resets_ = 0;
};

// =================================================================== setup

void PrimalSimplex::build_internal_problem() {
  m_ = lp_.num_rows;
  n_ = lp_.num_cols;
  nt_ = n_ + m_;
  sense_ = lp_.sense;

  if (opt_.scale) {
    scale_ = compute_scaling(A_orig_);
  } else {
    scale_.row.assign(m_, 1.0);
    scale_.col.assign(n_, 1.0);
  }
  A_ = A_orig_;
  for (int j = 0; j < n_; ++j) {
    for (int k = A_.col_start[j]; k < A_.col_start[j + 1]; ++k) {
      A_.value[k] *= scale_.row[A_.row_index[k]] * scale_.col[j];
    }
  }

  lower_.resize(nt_);
  upper_.resize(nt_);
  cost_orig_.assign(nt_, 0.0);
  for (int j = 0; j < n_; ++j) {
    const double c = scale_.col[j];
    lower_[j] = lp_.col_lower[j] / c;  // +-inf stays +-inf
    upper_[j] = lp_.col_upper[j] / c;
    cost_orig_[j] = sense_ * lp_.obj[j] * c;
  }
  for (int i = 0; i < m_; ++i) {
    const double r = scale_.row[i];
    lower_[n_ + i] = -lp_.row_upper[i] * r;
    upper_[n_ + i] = -lp_.row_lower[i] * r;
  }
  // Cost scaling: bring the largest |cost| to about 1 (power of two, exact),
  // so the dual tolerance means the same thing on every model.
  double cmax = 0.0;
  for (int j = 0; j < n_; ++j) cmax = std::max(cmax, std::fabs(cost_orig_[j]));
  cost_scale_ = (opt_.scale && cmax > 0.0) ? std::exp2(std::round(std::log2(1.0 / cmax))) : 1.0;
  for (int j = 0; j < n_; ++j) cost_orig_[j] *= cost_scale_;
  cost_ = cost_orig_;
  ptol_ = opt_.primal_tolerance;
  dtol_ = opt_.dual_tolerance;
}

void PrimalSimplex::unscaled_violations(double& primal, double& dual) const {
  primal = dual = 0.0;
  for (int v = 0; v < nt_; ++v) {
    // original value = factor * internal value
    const double factor = v < n_ ? scale_.col[v] : 1.0 / scale_.row[v - n_];
    const double viol = std::max({0.0, lower_[v] - x_[v], x_[v] - upper_[v]}) * factor;
    double bound = 0.0;  // size of the finite bounds, for a relative measure (as the checker uses)
    if (std::isfinite(lower_[v])) bound = std::max(bound, std::fabs(lower_[v] * factor));
    if (std::isfinite(upper_[v])) bound = std::max(bound, std::fabs(upper_[v] * factor));
    primal = std::max(primal, viol / (1.0 + bound));
    if (pos_[v] >= 0 || lower_[v] == upper_[v]) continue;
    // original reduced cost = internal / (factor * cost_scale)
    const double d = d_[v] / (factor * cost_scale_);
    double wrong = 0.0;
    switch (status_[v]) {
      case VarStatus::kAtLower: wrong = std::max(0.0, -d); break;
      case VarStatus::kAtUpper: wrong = std::max(0.0, d); break;
      case VarStatus::kFree: wrong = std::fabs(d); break;
      case VarStatus::kBasic: break;
    }
    dual = std::max(dual, wrong);
  }
}

void PrimalSimplex::initial_basis() {
  basis_.resize(m_);
  pos_.assign(nt_, -1);
  status_.assign(nt_, VarStatus::kAtLower);
  x_.assign(nt_, 0.0);
  for (int i = 0; i < m_; ++i) {
    basis_[i] = n_ + i;
    pos_[n_ + i] = i;
    status_[n_ + i] = VarStatus::kBasic;
  }
  for (int j = 0; j < n_; ++j) make_nonbasic(j);
  weight_.assign(nt_, 1.0);
}

// Puts a nonbasic variable at its bound nearest to its current value (or
// leaves a free variable where it is).
void PrimalSimplex::make_nonbasic(int v) {
  const double l = lower_[v], u = upper_[v], val = x_[v];
  if (l == -kInf && u == kInf) {
    status_[v] = VarStatus::kFree;
  } else if (u == kInf || (l != -kInf && std::fabs(val - l) <= std::fabs(u - val))) {
    status_[v] = VarStatus::kAtLower;
    x_[v] = l;
  } else {
    status_[v] = VarStatus::kAtUpper;
    x_[v] = u;
  }
}

// Small random cost changes break the ties that make degenerate models stall.
// The sign keeps each nonbasic variable's reduced cost on its "correct" side.
void PrimalSimplex::perturb_costs() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> unit(1.0, 2.0);
  for (int j = 0; j < n_; ++j) {
    if (lower_[j] == upper_[j] || status_[j] == VarStatus::kFree) continue;
    const double delta = (5e-7 + 1e-6 * std::fabs(cost_orig_[j])) * unit(rng);
    cost_[j] = cost_orig_[j] + (status_[j] == VarStatus::kAtUpper ? -delta : delta);
  }
  perturbed_ = true;
}

// ========================================================== linear algebra

void PrimalSimplex::refactor() {
  lu_.factorize(build_basis_matrix(A_, basis_));
  // Dependent columns were swapped for the slacks of uncovered rows.
  for (auto [p, row] : lu_.replaced_columns()) {
    const int old_var = basis_[p];
    const int new_var = n_ + row;
    basis_[p] = new_var;
    pos_[new_var] = p;
    status_[new_var] = VarStatus::kBasic;
    pos_[old_var] = -1;
    make_nonbasic(old_var);
  }
  ++refactors_;
  fresh_ = true;
  compute_primal();
}

// x_B = -B^{-1} N x_N, from scratch (removes accumulated drift).
void PrimalSimplex::compute_primal() {
  std::vector<double> rhs(m_, 0.0);
  for (int j = 0; j < nt_; ++j) {
    if (pos_[j] >= 0 || x_[j] == 0.0) continue;
    const double xj = x_[j];
    for_column(j, [&](int i, double a) { rhs[i] -= a * xj; });
  }
  lu_.ftran(rhs);
  for (int i = 0; i < m_; ++i) x_[basis_[i]] = rhs[i];
}

// y = B^{-T} c_B and d_j = c_j - a_j^T y for nonbasic j. Phase 1 costs are the
// gradient of the sum of infeasibilities.
void PrimalSimplex::compute_duals(bool phase1) {
  y_.assign(m_, 0.0);
  for (int i = 0; i < m_; ++i) {
    const int v = basis_[i];
    y_[i] = phase1 ? (below(v) ? -1.0 : above(v) ? 1.0 : 0.0) : cost_[v];
  }
  lu_.btran(y_);
  d_.assign(nt_, 0.0);
  for (int j = 0; j < nt_; ++j) {
    if (pos_[j] >= 0) continue;
    d_[j] = (phase1 ? 0.0 : cost_[j]) - column_dot(j, y_);
  }
}

// ================================================================ iteration

bool PrimalSimplex::any_infeasible() const {
  for (int v : basis_) {
    if (below(v) || above(v)) return true;
  }
  return false;
}

// Returns the entering variable (or -1 if none is attractive) and its
// direction: +1 to increase, -1 to decrease.
int PrimalSimplex::choose_entering(int& dir) const {
  const double tol = dtol_;
  int best = -1;
  double best_score = 0.0;
  for (int j = 0; j < nt_; ++j) {
    if (pos_[j] >= 0 || lower_[j] == upper_[j]) continue;
    const double dj = d_[j];
    int dj_dir = 0;
    switch (status_[j]) {
      case VarStatus::kAtLower: if (dj < -tol) dj_dir = 1; break;
      case VarStatus::kAtUpper: if (dj > tol) dj_dir = -1; break;
      case VarStatus::kFree: if (std::fabs(dj) > tol) dj_dir = dj < 0 ? 1 : -1; break;
      case VarStatus::kBasic: break;
    }
    if (dj_dir == 0) continue;
    if (bland_) {  // smallest index
      dir = dj_dir;
      return j;
    }
    const double score = opt_.pricing == Pricing::kDevex ? dj * dj / weight_[j] : std::fabs(dj);
    if (score > best_score) {
      best_score = score;
      best = j;
      dir = dj_dir;
    }
  }
  return best;
}

// Harris two-pass ratio test on x_B(theta) = x_B - theta * dir * alpha.
// Phase-1 aware: a basic variable below its lower bound blocks when it
// reaches that lower bound (it becomes feasible); one moving further away
// does not block (its cost is already in d_q).
PrimalSimplex::Ratio PrimalSimplex::ratio_test(int q, int dir) const {
  const double ptol = ptol_;
  const double harris_tol = bland_ ? 0.0 : ptol;
  Ratio res;

  // For each candidate row: the bound it moves towards, or none.
  auto target = [&](int i, double rate, double& bound) {
    const int v = basis_[i];
    if (rate > 0) {
      if (above(v)) return false;
      bound = below(v) ? lower_[v] : upper_[v];
      return bound != kInf;
    }
    if (below(v)) return false;
    bound = above(v) ? upper_[v] : lower_[v];
    return bound != -kInf;
  };

  // Pass 1: the largest step allowed with bounds relaxed by the tolerance.
  double theta_max = kInf;
  for (int i = 0; i < m_; ++i) {
    const double a = alpha_[i];
    if (std::fabs(a) <= opt_.pivot_tolerance) continue;
    const double rate = -dir * a;
    double bound;
    if (!target(i, rate, bound)) continue;
    const double relaxed = rate > 0 ? bound + harris_tol : bound - harris_tol;
    theta_max = std::min(theta_max, (relaxed - x_[basis_[i]]) / rate);
  }

  const double span = upper_[q] - lower_[q];  // inf unless q is boxed
  if (theta_max == kInf && span == kInf) return res;  // unbounded direction

  // Pass 2: among rows blocking within theta_max, the largest |alpha|.
  // (With Bland's rule: the smallest ratio, ties to the smallest variable index.)
  double best_abs = -1.0, best_theta = kInf;
  for (int i = 0; i < m_; ++i) {
    const double a = alpha_[i];
    if (std::fabs(a) <= opt_.pivot_tolerance) continue;
    const double rate = -dir * a;
    double bound;
    if (!target(i, rate, bound)) continue;
    const double t = (bound - x_[basis_[i]]) / rate;
    if (t > theta_max) continue;
    bool take;
    if (bland_) {
      take = res.row < 0 || t < best_theta - 1e-12 ||
             (t <= best_theta + 1e-12 && basis_[i] < basis_[res.row]);
    } else {
      take = std::fabs(a) > best_abs;
    }
    if (take) {
      best_abs = std::fabs(a);
      best_theta = t;
      res.row = i;
      res.leave_value = bound;
    }
  }

  if (span != kInf && (res.row < 0 || span <= best_theta)) {
    res.kind = Ratio::kFlip;
    res.theta = span;
    res.row = -1;
    return res;
  }
  if (res.row < 0) return res;  // only possible through round-off; treated as unbounded
  res.kind = Ratio::kPivot;
  res.theta = std::max(0.0, best_theta);
  return res;
}

// Devex reference weights (Forrest & Goldfarb): needs the pivot row
// alpha_r = e_r^T B^{-1} [A I], computed before the basis changes.
void PrimalSimplex::update_devex(int q, int r) {
  rho_.assign(m_, 0.0);
  rho_[r] = 1.0;
  lu_.btran(rho_);
  const double arq = alpha_[r];
  const double wq = weight_[q];
  bool reset = false;
  for (int j = 0; j < nt_; ++j) {
    if (pos_[j] >= 0 || j == q) continue;
    const double arj = column_dot(j, rho_);
    if (arj == 0.0) continue;
    const double ratio = arj / arq;
    const double w = ratio * ratio * wq;
    if (w > weight_[j]) weight_[j] = w;
    if (weight_[j] > 1e6) reset = true;
  }
  weight_[basis_[r]] = std::max(wq / (arq * arq), 1.0);
  if (reset) {
    std::fill(weight_.begin(), weight_.end(), 1.0);
    ++devex_resets_;
  }
}

// Applies the step. Returns false if the LU update was refused (the basis
// change is then completed by a refactorization).
bool PrimalSimplex::pivot(int q, int dir, const Ratio& ratio) {
  const double step = dir * ratio.theta;
  x_[q] += step;
  for (int i = 0; i < m_; ++i) x_[basis_[i]] -= step * alpha_[i];

  if (ratio.kind == Ratio::kFlip) {
    x_[q] = dir > 0 ? upper_[q] : lower_[q];
    status_[q] = dir > 0 ? VarStatus::kAtUpper : VarStatus::kAtLower;
    ++flips_;
    return true;
  }

  const int r = ratio.row;
  const int leaving = basis_[r];
  if (opt_.pricing == Pricing::kDevex) update_devex(q, r);

  x_[leaving] = ratio.leave_value;  // exactly on the bound it reached
  status_[leaving] = (ratio.leave_value == lower_[leaving]) ? VarStatus::kAtLower : VarStatus::kAtUpper;
  pos_[leaving] = -1;
  basis_[r] = q;
  pos_[q] = r;
  status_[q] = VarStatus::kBasic;

  if (!lu_.update(r, alpha_)) {
    refactor();
    return false;
  }
  fresh_ = false;
  return true;
}

double PrimalSimplex::internal_objective() const {
  double obj = 0.0;
  for (int j = 0; j < n_; ++j) obj += cost_orig_[j] * x_[j];
  obj /= cost_scale_;
  return obj;
}

// ================================================================== driver

Solution PrimalSimplex::run() {
  if (const std::string err = lp_.validate(); !err.empty()) {
    return finish(Status::NumericalError, "invalid problem: " + err);
  }
  for (int j = 0; j < lp_.num_cols; ++j) {
    if (lp_.col_lower[j] > lp_.col_upper[j]) return finish(Status::Infeasible, "column bounds cross");
  }
  for (int i = 0; i < lp_.num_rows; ++i) {
    if (lp_.row_lower[i] > lp_.row_upper[i]) return finish(Status::Infeasible, "row bounds cross");
  }

  build_internal_problem();
  initial_basis();
  if (opt_.perturb) perturb_costs();
  refactor();

  const long long stall_limit = 100 + 2LL * m_;
  while (true) {
    if (iterations_ >= opt_.max_iterations) return finish(Status::IterationLimit, "iteration limit");
    if ((iterations_ & 63) == 0 && elapsed() > opt_.time_limit_seconds) {
      return finish(Status::IterationLimit, "time limit");
    }
    if (lu_.needs_refactor()) refactor();

    const bool phase1 = any_infeasible();
    compute_duals(phase1);
    int dir = 0;
    const int q = choose_entering(dir);

    if (opt_.log_every > 0 && iterations_ % opt_.log_every == 0) {
      double infeas = 0.0;
      for (int v : basis_) infeas += std::max({0.0, lower_[v] - x_[v], x_[v] - upper_[v]});
      std::printf("  iter %9lld  phase %d  objective % .10e  infeasibility %.2e  %.2fs\n", iterations_,
                  phase1 ? 1 : 2, sense_ * internal_objective() + lp_.obj_offset, infeas, elapsed());
    }

    if (q < 0) {
      if (!fresh_) {  // confirm with a fresh factorization and fresh primal values
        refactor();
        continue;
      }
      if (phase1) return finish(Status::Infeasible, "phase 1 cannot reduce the infeasibility");
      if (perturbed_) {  // optimal for the perturbed costs: clean up with the true ones
        cost_ = cost_orig_;
        perturbed_ = false;
        continue;
      }
      // Optimal in the scaled model. Scaling can magnify violations when they
      // are mapped back, so confirm in original units and, if needed, keep
      // iterating with tighter tolerances.
      double primal_viol, dual_viol;
      unscaled_violations(primal_viol, dual_viol);
      if ((primal_viol > opt_.primal_tolerance || dual_viol > opt_.dual_tolerance) && tightenings_ < 4) {
        ptol_ = std::max(ptol_ * 0.1, 1e-12);
        dtol_ = std::max(dtol_ * 0.1, 1e-12);
        ++tightenings_;
        continue;
      }
      return finish(Status::Optimal, "");
    }

    alpha_.assign(m_, 0.0);
    for_column(q, [&](int i, double a) { alpha_[i] = a; });
    lu_.ftran(alpha_);

    const Ratio ratio = ratio_test(q, dir);
    if (ratio.kind == Ratio::kUnbounded) {
      if (!fresh_) {
        refactor();
        continue;
      }
      if (perturbed_) {
        cost_ = cost_orig_;
        perturbed_ = false;
        continue;
      }
      if (phase1) return finish(Status::NumericalError, "unbounded direction in phase 1");
      return finish(Status::Unbounded, "");
    }

    pivot(q, dir, ratio);
    ++iterations_;
    if (phase1) ++phase1_iterations_;

    // Anti-stalling: after a long run of zero-length steps use Bland's rule
    // (which cannot cycle) until a step makes real progress.
    if (ratio.theta <= 1e-12) {
      if (++degenerate_run_ > stall_limit && !bland_) {
        bland_ = true;
        ++bland_episodes_;
      }
    } else {
      degenerate_run_ = 0;
      bland_ = false;
    }
  }
}

Solution PrimalSimplex::finish(Status status, const std::string& note) {
  Solution res;
  res.engine = "simplex";
  res.precision = "fp64";
  res.status = status;
  res.iterations = iterations_;
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "phase-1 iters %lld, bound flips %lld, refactorizations %lld, Bland episodes %lld, "
                "Devex resets %lld, tolerance tightenings %d",
                phase1_iterations_, flips_, refactors_, bland_episodes_, devex_resets_, tightenings_);
  res.message = note.empty() ? buf : note + "; " + buf;
  if (std::any_of(lp_.is_integer.begin(), lp_.is_integer.end(), [](auto v) { return v != 0; })) res.message = "integrality ignored: LP relaxation solved; " + res.message;

  if (status == Status::Optimal) {
    compute_duals(false);  // true (unperturbed) costs
    res.x.resize(n_);
    for (int j = 0; j < n_; ++j) {
      // Nonbasic variables sit exactly on their original bounds.
      if (status_[j] == VarStatus::kAtLower) res.x[j] = lp_.col_lower[j];
      else if (status_[j] == VarStatus::kAtUpper) res.x[j] = lp_.col_upper[j];
      else res.x[j] = x_[j] * scale_.col[j];
    }
    res.y.resize(m_);
    for (int i = 0; i < m_; ++i) res.y[i] = sense_ * scale_.row[i] * y_[i] / cost_scale_;
    A_orig_.multiply(res.x, res.row_activity);
    A_orig_.multiply_transpose(res.y, res.z);
    for (int j = 0; j < n_; ++j) res.z[j] = lp_.obj[j] - res.z[j];
    res.objective = lp_.objective_value(res.x);
  }
  res.seconds = elapsed();
  return res;
}

}  // namespace

Solution solve_primal_simplex(const Model& lp, const SimplexOptions& options) {
  PrimalSimplex solver(lp, options);
  return solver.run();
}

}  // namespace ps26119::simplex
