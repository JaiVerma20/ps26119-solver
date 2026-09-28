// ranging.cpp — sensitivity ranging (cost and right-hand side) at an optimal vertex.
//
// Method: the classical post-optimal analysis of a basic optimal solution (Dantzig 1963, §12;
// Chvátal, "Linear Programming" 1983, ch. 10; Murty 1983 §3.13). Everything is done in
// minimisation form, c' = sense·c, on [A I] z = 0 with z = (x, s), s_i = −(A x)_i, so that
// s_i ∈ [−row_upper_i, −row_lower_i] (the layout the revised simplex uses).
//
//   1. Basis from the solution: variables strictly between their bounds must be basic; la::SparseLU
//      completes their columns with slack (unit) columns. The recomputed basic values must
//      reproduce x (primal feasibility), else ranging is refused. Then degenerate pivots (step 0,
//      Bland's rule — the point never moves) turn it into a basis whose reduced costs
//      d = c' − Aᵀπ, Bᵀπ = c'_B, have the optimal signs; if a wrong-sign variable could move
//      without a degenerate variable blocking it, the point is not optimal: refused, never a guess.
//   2. Cost of a nonbasic column j: the basis stays optimal while d_j keeps its sign, so
//      c'_j ∈ [c'_j − d_j, +∞) at a lower bound, (−∞, c'_j − d_j] at an upper bound.
//   3. Cost of a basic column in position r: ρ = B⁻ᵀ e_r, α_k = ρᵀ a_k over the nonbasic k; a change
//      δ gives d_k − δ α_k, which must keep each nonbasic sign — a ratio test for [δ_min, δ_max].
//   4. Right-hand side of a binding row i: moving its bound b by Δ moves s_i by −Δ and the basic
//      values by Δ·B⁻¹e_i; a ratio test keeps every basic variable within its bounds (and a ranged
//      row's bound on the right side of its other bound). The objective changes by y_i·Δ.
//      A row whose slack is basic is not binding: its (finite upper, else lower) bound may move
//      up to the activity and without limit the other way (binding = 0, e.g. [activity, +∞) for
//      an upper bound — the "allowable increase / decrease" of a spreadsheet sensitivity report).
//
// Invariants: never throws; refuses non-vertex (interior) solutions, infeasible or non-optimal
// reconstructed bases and models above options.max_rows; ranges are returned in the model's own
// objective sense. Under primal degeneracy a range may be one-sided at 0 (as in every solver).
#include "ps26119/ranging.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "la/csc.h"
#include "la/sparse_lu.h"

namespace ps26119 {
namespace {

constexpr double kInfD = std::numeric_limits<double>::infinity();

struct Var {
  double lo, up, val;
};

bool at_value(double v, double b, double tol) { return std::isfinite(b) && std::fabs(v - b) <= tol * (1.0 + std::fabs(b)); }

}  // namespace

RangingResult compute_ranging(const Model& model, const Solution& sol, const RangingOptions& options) {
  RangingResult R;
  const int m = model.num_rows, n = model.num_cols;
  if (sol.status != Status::Optimal) { R.message = "ranging needs an Optimal solution"; return R; }
  if (static_cast<int>(sol.x.size()) != n || static_cast<int>(sol.y.size()) != m || static_cast<int>(sol.z.size()) != n) {
    R.message = "the solution has no primal / dual vectors of the model's size";
    return R;
  }
  if (std::any_of(model.is_integer.begin(), model.is_integer.end(), [](std::uint8_t v) { return v != 0; })) {
    R.message = "ranging is defined for LPs (this model has integer columns)";
    return R;
  }
  if (m > options.max_rows) {
    std::ostringstream os;
    os << "model too large for full ranging (" << m << " rows > " << options.max_rows << ")";
    R.message = os.str();
    return R;
  }
  const double sense = model.sense;
  const la::SparseMatrixCSC A = la::csc_from_model(model);

  // activity and the variables of [A I] z = 0
  std::vector<double> act = sol.row_activity;
  if (static_cast<int>(act.size()) != m) act = model.row_activity(sol.x);
  std::vector<Var> v(n + m);
  for (int j = 0; j < n; ++j) v[j] = {model.col_lower[j], model.col_upper[j], sol.x[j]};
  for (int i = 0; i < m; ++i) v[n + i] = {-model.row_upper[i], -model.row_lower[i], -act[i]};
  double cmax = 0.0;
  std::vector<double> cp(n + m, 0.0);  // min-form costs; slacks cost 0
  for (int j = 0; j < n; ++j) { cp[j] = sense * model.obj[j]; cmax = std::max(cmax, std::fabs(cp[j])); }
  const double dtol = 1e-7 * (1.0 + cmax);

  auto at_lo = [&](int k) { return at_value(v[k].val, v[k].lo, 1e-7); };
  auto at_up = [&](int k) { return at_value(v[k].val, v[k].up, 1e-7); };

  // 1. the basis: every variable strictly between its bounds is basic (a vertex has at most m of
  //    them, with independent columns). The LU completes them: the empty placeholder columns are
  //    replaced by the unit (slack) columns of the rows the interior columns leave uncovered.
  std::vector<int> basic;
  basic.reserve(m);
  std::vector<char> is_basic(n + m, 0);
  int interior = 0;
  for (int k = 0; k < n + m; ++k) {
    // a free variable at 0 may stay nonbasic (the dual repair below brings it in if it must)
    const bool free_at_zero = !std::isfinite(v[k].lo) && !std::isfinite(v[k].up) && std::fabs(v[k].val) <= 1e-9;
    if (!at_lo(k) && !at_up(k) && !free_at_zero) {
      ++interior;
      if (static_cast<int>(basic.size()) < m) { basic.push_back(k); is_basic[k] = 1; }
    }
  }
  if (interior > m) {
    std::ostringstream os;
    os << "not a vertex: " << interior << " variables strictly between their bounds for " << m
       << " rows (an interior first-order answer; ranging needs the simplex)";
    R.message = os.str();
    return R;
  }
  const int n_interior = static_cast<int>(basic.size());
  la::SparseLU lu;
  {
    la::SparseMatrixCSC B0 = la::build_basis_matrix(A, basic);
    B0.num_cols = m;
    while (static_cast<int>(B0.col_start.size()) < m + 1) B0.col_start.push_back(static_cast<int>(B0.value.size()));
    basic.resize(m, -1);
    lu.factorize(B0);
  }
  for (const auto& [pos, row] : lu.replaced_columns()) {
    if (pos < n_interior || is_basic[n + row]) {
      R.message = "the solution is not a basic solution (the columns of the variables strictly inside their bounds are dependent)";
      return R;
    }
    basic[pos] = n + row;
    is_basic[n + row] = 1;
  }
  if (std::find(basic.begin(), basic.end(), -1) != basic.end()) {
    R.message = "the basis could not be completed";
    return R;
  }

  // basic values from the nonbasic ones: B x_B = −N z_N — they must reproduce the solution
  std::vector<double> xb(m, 0.0);
  for (int k = 0; k < n + m; ++k) {
    if (is_basic[k] || v[k].val == 0.0) continue;
    if (k < n) {
      for (int p = A.col_start[k]; p < A.col_start[k + 1]; ++p) xb[A.row_index[p]] -= A.value[p] * v[k].val;
    } else {
      xb[k - n] -= v[k].val;
    }
  }
  lu.ftran(xb);
  for (int r = 0; r < m; ++r) {
    const int k = basic[r];
    const double scale = 1.0 + std::fabs(v[k].val);
    if (std::fabs(xb[r] - v[k].val) > 1e-6 * scale ||
        (std::isfinite(v[k].lo) && xb[r] < v[k].lo - 1e-6 * (1.0 + std::fabs(v[k].lo))) ||
        (std::isfinite(v[k].up) && xb[r] > v[k].up + 1e-6 * (1.0 + std::fabs(v[k].up)))) {
      R.message = "the reconstructed basis does not reproduce a feasible solution";
      return R;
    }
  }

  // 2. an OPTIMAL basis for this point: reduced costs d = c' − [A I]ᵀπ, Bᵀπ = c'_B. While a
  //    nonbasic variable has the wrong sign, pivot it in against a degenerate basic variable that
  //    blocks at step 0 (Bland's rule: smallest index enters and leaves, so no cycling). The point
  //    never moves; if no degenerate variable blocks, the point is not optimal — refused.
  auto fixed = [&](int k) { return v[k].lo == v[k].up; };
  std::vector<double> pi(m), d(n + m, 0.0), col_in(m);
  const int max_pivots = 20 * (m - n_interior) + 100;
  for (int pivots = 0;; ++pivots) {
    for (int r = 0; r < m; ++r) pi[r] = cp[basic[r]];
    lu.btran(pi);
    for (int j = 0; j < n; ++j) {
      double s = cp[j];
      for (int p = A.col_start[j]; p < A.col_start[j + 1]; ++p) s -= A.value[p] * pi[A.row_index[p]];
      d[j] = s;
    }
    for (int i = 0; i < m; ++i) d[n + i] = -pi[i];
    int enter = -1;
    double dir = 0.0;  // +1: the entering variable would increase, −1: decrease
    for (int k = 0; k < n + m && enter < 0; ++k) {
      if (is_basic[k] || fixed(k)) continue;
      const bool lo = at_lo(k), up = at_up(k);
      if (lo && !up && d[k] < -dtol) { enter = k; dir = 1.0; }
      else if (up && !lo && d[k] > dtol) { enter = k; dir = -1.0; }
      else if (!lo && !up && std::fabs(d[k]) > dtol) { enter = k; dir = d[k] < 0 ? 1.0 : -1.0; }
    }
    if (enter < 0) break;  // dual feasible: an optimal basis of this point
    if (pivots >= max_pivots) {
      R.message = "no optimal basis found for this point within the degenerate-pivot limit";
      return R;
    }
    std::fill(col_in.begin(), col_in.end(), 0.0);
    if (enter < n) {
      for (int p = A.col_start[enter]; p < A.col_start[enter + 1]; ++p) col_in[A.row_index[p]] = A.value[p];
    } else {
      col_in[enter - n] = 1.0;
    }
    lu.ftran(col_in);  // x_B moves by −t·dir·α
    int leave = -1;
    for (int r = 0; r < m; ++r) {
      const double change = -dir * col_in[r];
      if (std::fabs(change) <= 1e-9) continue;
      const int b = basic[r];
      const bool blocks = fixed(b) || (at_lo(b) && change < 0) || (at_up(b) && change > 0);
      if (blocks && (leave < 0 || b < basic[leave])) leave = r;
    }
    if (leave < 0) {
      R.message = "the reconstructed basis is not dual feasible (the solution is not an optimal vertex at this tolerance)";
      return R;
    }
    is_basic[basic[leave]] = 0;
    basic[leave] = enter;
    is_basic[enter] = 1;
    lu.factorize(la::build_basis_matrix(A, basic));
    if (!lu.replaced_columns().empty()) {
      R.message = "a degenerate pivot produced a singular basis";
      return R;
    }
  }
  for (int r = 0; r < m; ++r)
    if (at_lo(basic[r]) || at_up(basic[r])) ++R.degenerate_basics;

  // nonbasic k -> constraint on a change t of its reduced cost direction (helper for 3.)
  auto ratio_limits = [&](const std::vector<double>& alpha_of, double& tmin, double& tmax) {
    tmin = -kInfD;
    tmax = kInfD;
    for (int k = 0; k < n + m; ++k) {
      if (is_basic[k] || fixed(k)) continue;
      const double a = alpha_of[k];
      if (std::fabs(a) <= 1e-12) continue;
      const bool lo = at_lo(k), up = at_up(k);
      const double q = d[k] / a;  // d_k − t a = 0 at t = q
      if (!lo && !up) { tmin = std::max(tmin, 0.0); tmax = std::min(tmax, 0.0); continue; }  // free at 0
      const bool need_nonneg = lo;  // at lower: d ≥ 0; at upper: d ≤ 0
      if ((need_nonneg && a > 0) || (!need_nonneg && a < 0)) tmax = std::min(tmax, q);
      else tmin = std::max(tmin, q);
    }
    tmax = std::max(tmax, 0.0);
    tmin = std::min(tmin, 0.0);
  };

  // 3. + 4. cost ranging (min form, then back to the model's sense)
  R.cols.resize(n);
  std::vector<int> pos_of(n + m, -1);
  for (int r = 0; r < m; ++r) pos_of[basic[r]] = r;
  std::vector<double> rho(m), alpha(n + m);
  for (int j = 0; j < n; ++j) {
    CostRange& c = R.cols[j];
    c.col = j;
    c.cost = model.obj[j];
    c.reduced_cost = sense * d[j];  // this basis's reduced cost
    c.basic = is_basic[j];
    double dmin, dmax;  // change of c'_j keeping the basis optimal
    if (!is_basic[j]) {
      if (fixed(j)) { dmin = -kInfD; dmax = kInfD; }
      else if (at_lo(j) && !at_up(j)) { dmin = -std::max(d[j], 0.0); dmax = kInfD; }
      else if (at_up(j) && !at_lo(j)) { dmin = -kInfD; dmax = -std::min(d[j], 0.0); }
      else { dmin = 0.0; dmax = 0.0; }
    } else {
      std::fill(rho.begin(), rho.end(), 0.0);
      rho[pos_of[j]] = 1.0;
      lu.btran(rho);
      for (int k = 0; k < n; ++k) {
        if (is_basic[k]) { alpha[k] = 0.0; continue; }
        double s = 0.0;
        for (int p = A.col_start[k]; p < A.col_start[k + 1]; ++p) s += A.value[p] * rho[A.row_index[p]];
        alpha[k] = s;
      }
      for (int i = 0; i < m; ++i) alpha[n + i] = is_basic[n + i] ? 0.0 : rho[i];
      ratio_limits(alpha, dmin, dmax);
    }
    const double lo_p = cp[j] + dmin, up_p = cp[j] + dmax;
    if (sense > 0) { c.lower = lo_p; c.upper = up_p; }
    else { c.lower = -up_p; c.upper = -lo_p; }
  }

  // 4. right-hand-side ranging
  R.rows.resize(m);
  std::vector<double> col(m);
  for (int i = 0; i < m; ++i) {
    RhsRange& rr = R.rows[i];
    rr.row = i;
    rr.activity = act[i];
    rr.dual = sense * pi[i];  // this basis's dual: the objective moves by it across the range (unique only if non-degenerate)
    const int k = n + i;
    const double rl = model.row_lower[i], ru = model.row_upper[i];
    if (is_basic[k]) {
      // basic slack: the basic values do not depend on this row's bounds, so the basis stays while
      // s_i = −activity stays within [−ru, −rl]: an upper bound may move within [activity, +∞), a
      // lower bound within (−∞, activity]. An equality row with a (degenerate) basic slack is
      // binding with the one-point range [b, b].
      if (rl == ru) {
        rr.binding = 1;
        rr.bound = rr.lower = rr.upper = ru;
      } else if (std::isfinite(ru)) {
        rr.binding = 0;
        rr.bound = ru;
        rr.lower = std::min(act[i], ru);
        rr.upper = kInfD;
      } else if (std::isfinite(rl)) {
        rr.binding = 0;
        rr.bound = rl;
        rr.lower = -kInfD;
        rr.upper = std::max(act[i], rl);
      } else {  // free row
        rr.binding = 0;
        rr.bound = act[i];
        rr.lower = -kInfD;
        rr.upper = kInfD;
      }
      continue;
    }
    // s_i at its lower bound −row_upper  -> the row's upper bound binds; at −row_lower -> the lower
    const bool upper_binds = at_lo(k);
    rr.binding = upper_binds ? 1 : -1;
    rr.bound = upper_binds ? model.row_upper[i] : model.row_lower[i];
    std::fill(col.begin(), col.end(), 0.0);
    col[i] = 1.0;
    lu.ftran(col);  // Δx_B = Δ · col when the bound moves by Δ
    double dlo = -kInfD, dhi = kInfD;
    for (int r = 0; r < m; ++r) {
      const double a = col[r];
      if (std::fabs(a) <= 1e-12) continue;
      const Var& bv = v[basic[r]];
      const double x = v[basic[r]].val;
      // keep lo ≤ x + Δ·a ≤ up
      if (std::isfinite(bv.up)) {
        if (a > 0) dhi = std::min(dhi, (bv.up - x) / a);
        else dlo = std::max(dlo, (bv.up - x) / a);
      }
      if (std::isfinite(bv.lo)) {
        if (a > 0) dlo = std::max(dlo, (bv.lo - x) / a);
        else dhi = std::min(dhi, (bv.lo - x) / a);
      }
    }
    if (rl != ru) {  // a ranged row's bound stays on its side of the other bound
      if (upper_binds && std::isfinite(rl)) dlo = std::max(dlo, rl - ru);
      if (!upper_binds && std::isfinite(ru)) dhi = std::min(dhi, ru - rl);
    }
    dlo = std::min(dlo, 0.0);
    dhi = std::max(dhi, 0.0);
    rr.lower = rr.bound + dlo;
    rr.upper = rr.bound + dhi;
  }
  R.ok = true;
  if (R.degenerate_basics) {
    std::ostringstream os;
    os << R.degenerate_basics << " basic variable(s) at a bound: primal-degenerate vertex, some ranges are one-sided at 0";
    R.message = os.str();
  }
  return R;
}

}  // namespace ps26119
