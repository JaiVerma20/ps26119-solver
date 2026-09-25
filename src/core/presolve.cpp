// presolve.cpp — see presolve.h.
#include "core/presolve.h"

#include <cmath>
#include <limits>

#include "la/csr.h"

namespace ps26119 {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kFeasTol = 1e-9;  // relative feasibility tolerance for presolve-time checks

bool within(double v, double lo, double up) {
  return v >= lo - kFeasTol * (1 + std::fabs(lo)) && v <= up + kFeasTol * (1 + std::fabs(up));
}

}  // namespace

PresolveResult presolve(const Model& M) {
  PresolveResult r;
  const int m = M.num_rows, n = M.num_cols;
  const la::Csr<double> A = la::csr_from_model(M);  // row access
  std::vector<double> rl = M.row_lower, ru = M.row_upper, cl = M.col_lower, cu = M.col_upper;
  std::vector<char> row_alive(m, 1), col_alive(n, 1);
  std::vector<int> row_cnt(m), col_cnt(n);
  for (int i = 0; i < m; ++i) row_cnt[i] = static_cast<int>(A.row_ptr[i + 1] - A.row_ptr[i]);
  for (int j = 0; j < n; ++j) col_cnt[j] = M.col_start[j + 1] - M.col_start[j];
  r.fixed_value.assign(n, kNaN);
  double offset = M.obj_offset;
  // Integer columns: bounds are always rounded INWARD (a fractional bound on an integer
  // column is equivalent to its rounded value). Without this, a column tightened to x ≥ 0.5
  // and then removed would be fixed at 0.5 — found on MIPLIB egout (wrong "optimum").
  auto is_int = [&](int j) { return j < static_cast<int>(M.is_integer.size()) && M.is_integer[j] != 0; };
  auto round_int_bounds = [&](int j) {
    if (!is_int(j)) return true;
    if (std::isfinite(cl[j])) cl[j] = std::ceil(cl[j] - 1e-9);
    if (std::isfinite(cu[j])) cu[j] = std::floor(cu[j] + 1e-9);
    return cl[j] <= cu[j];
  };
  for (int j = 0; j < n; ++j)
    if (!round_int_bounds(j)) {
      r.outcome = PresolveResult::Outcome::Infeasible;
      r.message = "integer column " + std::to_string(j) + " has no integer value within its bounds";
      return r;
    }
  auto infeasible = [&](const std::string& why) {
    r.outcome = PresolveResult::Outcome::Infeasible;
    r.message = why;
    return r;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    // ---- rows: empty (R1) and singleton (R4)
    for (int i = 0; i < m; ++i) {
      if (!row_alive[i]) continue;
      if (row_cnt[i] == 0) {
        if (!within(0.0, rl[i], ru[i])) return infeasible("empty row " + std::to_string(i) + " excludes 0");
        row_alive[i] = 0;
        ++r.removed_rows;
        changed = true;
      } else if (row_cnt[i] == 1) {
        int j = -1;
        double a = 0;
        for (std::int64_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
          if (col_alive[A.col[p]]) j = A.col[p], a = A.val[p];
        if (j < 0 || a == 0.0) continue;
        const double lo = a > 0 ? rl[i] / a : ru[i] / a;  // ±inf propagate correctly
        const double hi = a > 0 ? ru[i] / a : rl[i] / a;
        const bool set_lower = lo > cl[j], set_upper = hi < cu[j];
        double nl = set_lower ? lo : cl[j], nu = set_upper ? hi : cu[j];
        if (nl > nu) {
          if (nl - nu > kFeasTol * (1 + std::fabs(nl))) return infeasible("singleton row " + std::to_string(i) + " conflicts with the bounds of its column");
          nl = nu = 0.5 * (nl + nu);
        }
        cl[j] = nl;
        cu[j] = nu;
        if (!round_int_bounds(j)) return infeasible("singleton row " + std::to_string(i) + " leaves integer column " + std::to_string(j) + " no integer value");
        r.singletons.push_back({i, j, a, set_lower, set_upper});
        r.tightened_bounds += set_lower + set_upper;
        row_alive[i] = 0;
        --col_cnt[j];
        ++r.removed_rows;
        changed = true;
      }
    }
    // ---- columns: fixed (R2) and empty (R3)
    for (int j = 0; j < n; ++j) {
      if (!col_alive[j]) continue;
      double v;
      if (cl[j] == cu[j]) {
        v = cl[j];
      } else if (col_cnt[j] == 0) {
        const double c = M.sense * M.obj[j];
        v = c > 0 ? cl[j] : (c < 0 ? cu[j] : std::min(std::max(0.0, cl[j]), cu[j]));
        if (!std::isfinite(v)) continue;  // unbounded direction: leave it to the engine
      } else {
        continue;
      }
      for (int p = M.col_start[j]; p < M.col_start[j + 1]; ++p) {
        const int i = M.row_index[p];
        if (!row_alive[i]) continue;
        const double t = M.value[p] * v;
        if (std::isfinite(rl[i])) rl[i] -= t;
        if (std::isfinite(ru[i])) ru[i] -= t;
        --row_cnt[i];
      }
      offset += M.obj[j] * v;
      r.fixed_value[j] = v;
      col_alive[j] = 0;
      ++r.removed_cols;
      changed = true;
    }
  }

  if (r.removed_rows == 0 && r.removed_cols == 0) {
    r.outcome = PresolveResult::Outcome::Unchanged;
    return r;
  }
  // ---- build the reduced model
  Model& R = r.reduced;
  std::vector<int> new_row(m, -1);
  for (int i = 0; i < m; ++i)
    if (row_alive[i]) new_row[i] = static_cast<int>(r.row_map.size()), r.row_map.push_back(i);
  for (int j = 0; j < n; ++j)
    if (col_alive[j]) r.col_map.push_back(j);
  R.name = M.name;
  R.sense = M.sense;
  R.obj_offset = offset;
  R.num_rows = static_cast<int>(r.row_map.size());
  R.num_cols = static_cast<int>(r.col_map.size());
  R.col_start.push_back(0);
  for (int j : r.col_map) {
    R.obj.push_back(M.obj[j]);
    R.col_lower.push_back(cl[j]);
    R.col_upper.push_back(cu[j]);
    for (int p = M.col_start[j]; p < M.col_start[j + 1]; ++p) {
      const int i = M.row_index[p];
      if (new_row[i] >= 0) R.row_index.push_back(new_row[i]), R.value.push_back(M.value[p]);
    }
    R.col_start.push_back(static_cast<int>(R.value.size()));
    if (!M.col_names.empty()) R.col_names.push_back(M.col_names[j]);
    if (!M.is_integer.empty()) R.is_integer.push_back(M.is_integer[j]);
  }
  for (int i : r.row_map) {
    R.row_lower.push_back(rl[i]);
    R.row_upper.push_back(ru[i]);
    if (!M.row_names.empty()) R.row_names.push_back(M.row_names[i]);
  }
  r.outcome = PresolveResult::Outcome::Reduced;
  return r;
}

Solution postsolve(const Model& M, const PresolveResult& r, const Solution& red) {
  Solution s = red;
  const int m = M.num_rows, n = M.num_cols;
  s.x.assign(n, 0.0);
  s.y.assign(m, 0.0);
  const bool have_x = red.x.size() == r.col_map.size();
  const bool have_y = red.y.size() == r.row_map.size();
  for (int j = 0; j < n; ++j)
    if (!std::isnan(r.fixed_value[j])) s.x[j] = r.fixed_value[j];
  if (have_x)
    for (std::size_t c = 0; c < r.col_map.size(); ++c) s.x[r.col_map[c]] = red.x[c];
  if (have_y)
    for (std::size_t q = 0; q < r.row_map.size(); ++q) s.y[r.row_map[q]] = red.y[q];
  // Singleton rows take over the reduced cost of their column when the column sits on the
  // bound the row created (reverse order: the last row that tightened a bound owns it).
  for (auto it = r.singletons.rbegin(); it != r.singletons.rend(); ++it) {
    const int j = it->col;
    double z = M.obj[j];
    for (int p = M.col_start[j]; p < M.col_start[j + 1]; ++p) z -= M.value[p] * s.y[M.row_index[p]];
    const double zm = M.sense * z;
    if ((zm > 0 && it->set_lower) || (zm < 0 && it->set_upper)) s.y[it->row] = z / it->a;
  }
  // z = c − Aᵀy and the activities on the ORIGINAL model
  s.z = M.obj;
  for (int j = 0; j < n; ++j)
    for (int p = M.col_start[j]; p < M.col_start[j + 1]; ++p) s.z[j] -= M.value[p] * s.y[M.row_index[p]];
  s.row_activity = M.row_activity(s.x);
  s.objective = M.objective_value(s.x);
  if (!have_x) s.x.clear(), s.row_activity.clear();
  if (!have_y) s.y.clear(), s.z.clear();
  return s;
}

}  // namespace ps26119
