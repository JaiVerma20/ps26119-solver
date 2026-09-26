// Origin: gpuopt src/oracle/dense_oracle.cpp (Shivanshu Vats, c192dd0); see dense_tableau.h.
// dense_oracle.cpp - two-phase dense tableau simplex with Bland's rule.
//
// Pipeline
// --------
// 1. Standard form. The general problem
//        min c^T x   s.t.  L <= A x <= U,  l <= x <= u
//    is rewritten as
//        min c'^T z  s.t.  M z = b,  z >= 0
//    by giving every row i an activity variable r_i = a_i x with bounds
//    [L_i, U_i] (so every row becomes a_i x - r_i = 0) and then removing
//    every bound with one of four textbook substitutions:
//        l == u            x = l               (fixed: becomes a constant)
//        l finite          x = l + z           (+ row z + s = u - l if u finite)
//        only u finite     x = u - z
//        free              x = z+ - z-
// 2. Phase 1. Flip rows so b >= 0, add one artificial per row, minimise the
//    sum of artificials. A positive optimum proves the LP infeasible.
// 3. Phase 2. Minimise c'^T z starting from the phase-1 basis. Artificials
//    are never allowed to re-enter.
// 4. Recover x from z, and the row duals y from the reduced costs of the
//    artificial columns (they hold -c_B^T B^{-1}).
//
// Bland's rule (smallest-index entering column, smallest-index leaving row
// among ties) guarantees termination without cycling in exact arithmetic.
// That, not speed, is why it is used here.

#include "oracle/dense_tableau.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace ps26119::oracle {
namespace {

// How one variable of the extended problem (x_j or r_i) maps onto
// standard-form columns.
struct VarMap {
  enum Kind { kFixed, kShift, kNegate, kSplit } kind = kFixed;
  double base = 0.0;  // l, u or the fixed value
  int col = -1;       // z column
  int col2 = -1;      // z- column for kSplit

  double value(const std::vector<double>& z) const {
    switch (kind) {
      case kFixed: return base;
      case kShift: return base + z[col];
      case kNegate: return base - z[col];
      case kSplit: return z[col] - z[col2];
    }
    return 0.0;
  }
};

struct StandardForm {
  int m = 0;  // rows
  int n = 0;  // columns
  std::vector<double> M;  // m x n, row-major
  std::vector<double> b;
  std::vector<double> cost;
  std::vector<VarMap> maps;  // one per extended variable: n_orig columns, then m_orig rows

  double& at(int i, int j) { return M[static_cast<size_t>(i) * n + j]; }
};

// Builds the standard form. Returns false (with a reason) if some bound pair
// is already contradictory, which proves infeasibility without any pivoting.
bool build_standard_form(const Model& lp, const DenseTableauOptions& opt, StandardForm& sf,
                         std::string& reason) {
  const int n_orig = lp.num_cols;
  const int m_orig = lp.num_rows;
  const int n_ext = n_orig + m_orig;
  const double sense = static_cast<double>(lp.sense);

  auto bounds = [&](int v) {
    return v < n_orig ? std::pair{lp.col_lower[v], lp.col_upper[v]}
                      : std::pair{lp.row_lower[v - n_orig], lp.row_upper[v - n_orig]};
  };
  auto name = [&](int v) {
    // Model names are optional; fall back to the index.
    if (v < n_orig) return "column '" + (lp.col_names.empty() ? std::to_string(v) : lp.col_names[v]) + "'";
    const int i = v - n_orig;
    return "row '" + (lp.row_names.empty() ? std::to_string(i) : lp.row_names[i]) + "'";
  };

  // Pass 1: classify every variable and count standard-form columns / rows.
  sf.maps.assign(n_ext, {});
  int n_std = 0;
  int ub_rows = 0;
  for (int v = 0; v < n_ext; ++v) {
    const auto [l, u] = bounds(v);
    if (l > u || l == kInf || u == -kInf) {
      reason = name(v) + " has contradictory bounds";
      return false;
    }
    VarMap& vm = sf.maps[v];
    if (l == -kInf && u == kInf) {
      vm.kind = VarMap::kSplit;
      vm.col = n_std++;
      vm.col2 = n_std++;
    } else if (l == u) {
      vm.kind = VarMap::kFixed;
      vm.base = l;
    } else if (l != -kInf) {
      vm.kind = VarMap::kShift;
      vm.base = l;
      vm.col = n_std++;
      if (u != kInf) {
        ++n_std;  // slack for z + s = u - l
        ++ub_rows;
      }
    } else {
      vm.kind = VarMap::kNegate;
      vm.base = u;
      vm.col = n_std++;
    }
  }

  sf.m = m_orig + ub_rows;
  sf.n = n_std;
  const long long entries = static_cast<long long>(sf.m + 1) * (sf.n + sf.m + 1);
  if (entries > opt.max_tableau_entries) {
    reason = "problem too large for the dense oracle (" + std::to_string(entries) + " tableau entries)";
    return false;
  }
  sf.M.assign(static_cast<size_t>(sf.m) * sf.n, 0.0);
  sf.b.assign(sf.m, 0.0);
  sf.cost.assign(sf.n, 0.0);

  // Pass 2: fill. Extended column of x_j is A's column j; of r_i it is -e_i.
  int next_ub_row = m_orig;
  for (int v = 0; v < n_ext; ++v) {
    std::vector<std::pair<int, double>> entries_v;
    double c = 0.0;
    if (v < n_orig) {
      for (int k = lp.col_start[v]; k < lp.col_start[v + 1]; ++k) {
        entries_v.push_back({lp.row_index[k], lp.value[k]});
      }
      c = sense * lp.obj[v];
    } else {
      entries_v.push_back({v - n_orig, -1.0});
    }

    const VarMap& vm = sf.maps[v];
    const auto [l, u] = bounds(v);
    switch (vm.kind) {
      case VarMap::kFixed:
        for (auto [i, a] : entries_v) sf.b[i] -= a * vm.base;
        break;
      case VarMap::kShift:
        for (auto [i, a] : entries_v) {
          sf.at(i, vm.col) = a;
          sf.b[i] -= a * vm.base;
        }
        sf.cost[vm.col] = c;
        if (u != kInf) {
          const int r = next_ub_row++;
          sf.at(r, vm.col) = 1.0;
          sf.at(r, vm.col + 1) = 1.0;  // slack column allocated right after
          sf.b[r] = u - l;
        }
        break;
      case VarMap::kNegate:
        for (auto [i, a] : entries_v) {
          sf.at(i, vm.col) = -a;
          sf.b[i] -= a * vm.base;
        }
        sf.cost[vm.col] = -c;
        break;
      case VarMap::kSplit:
        for (auto [i, a] : entries_v) {
          sf.at(i, vm.col) = a;
          sf.at(i, vm.col2) = -a;
        }
        sf.cost[vm.col] = c;
        sf.cost[vm.col2] = -c;
        break;
    }
  }
  return true;
}

// Dense simplex tableau with one artificial column per row.
//   columns [0, n)      standard-form variables z
//   columns [n, n+m)    artificials
//   column  n+m         right-hand side
//   row m               reduced costs; T[m][rhs] = -(objective value)
class Tableau {
 public:
  enum class Outcome { kOptimal, kUnbounded, kIterationLimit };

  Tableau(const StandardForm& sf, const DenseTableauOptions& opt)
      : m_(sf.m), n_(sf.n), width_(sf.n + sf.m + 1), rhs_(sf.n + sf.m), opt_(opt) {
    T_.assign(static_cast<size_t>(m_ + 1) * width_, 0.0);
    basis_.resize(m_);
    sign_.resize(m_);
    for (int i = 0; i < m_; ++i) {
      sign_[i] = sf.b[i] < 0.0 ? -1.0 : 1.0;
      for (int j = 0; j < n_; ++j) at(i, j) = sign_[i] * sf.M[static_cast<size_t>(i) * n_ + j];
      at(i, n_ + i) = 1.0;
      at(i, rhs_) = sign_[i] * sf.b[i];
      basis_[i] = n_ + i;
    }
  }

  // Phase 1 objective: sum of artificials, priced out against the identity basis.
  void set_phase1_objective() {
    std::fill(row(m_), row(m_) + width_, 0.0);
    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < n_; ++j) at(m_, j) -= at(i, j);
      at(m_, rhs_) -= at(i, rhs_);
    }
  }

  // Phase 2 objective: real costs, priced out against the current basis.
  void set_phase2_objective(const std::vector<double>& cost) {
    std::fill(row(m_), row(m_) + width_, 0.0);
    for (int j = 0; j < n_; ++j) at(m_, j) = cost[j];
    for (int i = 0; i < m_; ++i) {
      const double cb = basis_[i] < n_ ? cost[basis_[i]] : 0.0;
      if (cb == 0.0) continue;
      for (int j = 0; j < width_; ++j) at(m_, j) -= cb * at(i, j);
    }
  }

  // Runs simplex iterations. Only columns j < n may enter (artificials never re-enter).
  Outcome run(long long& iterations) {
    while (true) {
      if (iterations >= opt_.max_iterations) return Outcome::kIterationLimit;

      // Bland entering rule: the first column with a negative reduced cost.
      int q = -1;
      for (int j = 0; j < n_; ++j) {
        if (at(m_, j) < -opt_.optimality_tolerance) {
          q = j;
          break;
        }
      }
      if (q < 0) return Outcome::kOptimal;

      // Ratio test. Among (near-)ties, Bland picks the smallest basic index.
      int r = -1;
      double best = 0.0;
      for (int i = 0; i < m_; ++i) {
        const double a = at(i, q);
        if (a <= opt_.pivot_tolerance) continue;
        const double ratio = at(i, rhs_) / a;
        const double tie = 1e-12 * (1.0 + std::fabs(best));
        if (r < 0 || ratio < best - tie) {
          r = i;
          best = ratio;
        } else if (ratio <= best + tie && basis_[i] < basis_[r]) {
          r = i;
          best = std::min(best, ratio);
        }
      }
      if (r < 0) return Outcome::kUnbounded;

      pivot(r, q);
      ++iterations;
    }
  }

  // After phase 1: pivot zero-level artificials out of the basis where possible.
  // A row where that is impossible is linearly dependent on the others; its
  // artificial stays basic at zero forever, which is harmless.
  void drive_out_artificials() {
    for (int i = 0; i < m_; ++i) {
      if (basis_[i] < n_) continue;
      int best_j = -1;
      double best_abs = opt_.pivot_tolerance;
      for (int j = 0; j < n_; ++j) {
        if (std::fabs(at(i, j)) > best_abs) {
          best_abs = std::fabs(at(i, j));
          best_j = j;
        }
      }
      if (best_j >= 0) pivot(i, best_j);
    }
  }

  double objective() const { return -at(m_, rhs_); }
  double max_abs_rhs() const {
    double v = 0.0;
    for (int i = 0; i < m_; ++i) v = std::max(v, std::fabs(at(i, rhs_)));
    return v;
  }

  std::vector<double> primal() const {
    std::vector<double> z(n_, 0.0);
    for (int i = 0; i < m_; ++i) {
      if (basis_[i] < n_) z[basis_[i]] = at(i, rhs_);
    }
    return z;
  }

  // Duals w of the original (unflipped) standard-form rows: the reduced cost
  // of artificial k is -w_k (in the flipped system), so w_k = -sign_k * T[m][n+k].
  std::vector<double> duals() const {
    std::vector<double> w(m_);
    for (int k = 0; k < m_; ++k) w[k] = -sign_[k] * at(m_, n_ + k);
    return w;
  }

 private:
  double* row(int i) { return &T_[static_cast<size_t>(i) * width_]; }
  double& at(int i, int j) { return T_[static_cast<size_t>(i) * width_ + j]; }
  double at(int i, int j) const { return T_[static_cast<size_t>(i) * width_ + j]; }

  // Gauss-Jordan pivot on (r, q): column q becomes the unit vector e_r.
  void pivot(int r, int q) {
    double* pr = row(r);
    const double p = pr[q];
    for (int j = 0; j < width_; ++j) pr[j] /= p;
    pr[q] = 1.0;
    for (int i = 0; i <= m_; ++i) {
      if (i == r) continue;
      double* ri = row(i);
      const double f = ri[q];
      if (f == 0.0) continue;
      for (int j = 0; j < width_; ++j) ri[j] -= f * pr[j];
      ri[q] = 0.0;
      // Round-off can push a zero basic value to -1e-16; snap it back so the
      // next ratio test never sees a spurious negative step.
      if (i < m_ && ri[rhs_] < 0.0 && ri[rhs_] > -1e-11) ri[rhs_] = 0.0;
    }
    basis_[r] = q;
  }

  int m_, n_, width_, rhs_;
  const DenseTableauOptions& opt_;
  std::vector<double> T_;
  std::vector<int> basis_;
  std::vector<double> sign_;
};

}  // namespace

Solution solve_dense_tableau(const Model& lp, const DenseTableauOptions& options) {
  const auto start = std::chrono::steady_clock::now();
  Solution result;
  result.engine = "tableau";
  result.precision = "fp64";
  auto finish = [&](Status status, std::string message) {
    result.status = status;
    if (!message.empty()) {
      result.message = result.message.empty() ? message : result.message + "; " + message;
    }
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
  };

  if (const std::string err = lp.validate(); !err.empty()) {
    return finish(Status::NumericalError, "invalid problem: " + err);
  }
  if (std::any_of(lp.is_integer.begin(), lp.is_integer.end(), [](auto v) { return v != 0; })) result.message = "integrality ignored: LP relaxation solved";

  StandardForm sf;
  std::string reason;
  if (!build_standard_form(lp, options, sf, reason)) {
    const bool too_large = reason.find("too large") != std::string::npos;
    return finish(too_large ? Status::NotSolved : Status::Infeasible, reason);
  }

  Tableau tableau(sf, options);
  const double b_scale = 1.0 + tableau.max_abs_rhs();

  // Phase 1: find a feasible basis.
  tableau.set_phase1_objective();
  if (tableau.run(result.iterations) == Tableau::Outcome::kIterationLimit) {
    return finish(Status::IterationLimit, "iteration limit reached in phase 1");
  }
  if (tableau.objective() > options.feasibility_tolerance * b_scale) {
    return finish(Status::Infeasible, "phase 1 optimum is positive");
  }
  tableau.drive_out_artificials();

  // Phase 2: optimise the real objective.
  tableau.set_phase2_objective(sf.cost);
  switch (tableau.run(result.iterations)) {
    case Tableau::Outcome::kIterationLimit:
      return finish(Status::IterationLimit, "iteration limit reached in phase 2");
    case Tableau::Outcome::kUnbounded:
      return finish(Status::Unbounded, "");
    case Tableau::Outcome::kOptimal:
      break;
  }

  // Map the standard-form solution back to the original space.
  const int n = lp.num_cols;
  const int m = lp.num_rows;
  const std::vector<double> z = tableau.primal();
  const std::vector<double> w = tableau.duals();
  const double sense = static_cast<double>(lp.sense);

  result.x.resize(n);
  for (int j = 0; j < n; ++j) result.x[j] = sf.maps[j].value(z);
  // Row i of the standard form is a_i x - r_i = 0, so its dual is y_i
  // (for the minimisation form; flip back for a maximisation problem).
  result.y.resize(m);
  for (int i = 0; i < m; ++i) result.y[i] = sense * w[i];
  result.row_activity = lp.row_activity(result.x);
  result.z.assign(n, 0.0);
  for (int j = 0; j < n; ++j) {
    double aty = 0.0;
    for (int k = lp.col_start[j]; k < lp.col_start[j + 1]; ++k) aty += lp.value[k] * result.y[lp.row_index[k]];
    result.z[j] = lp.obj[j] - aty;
  }
  result.objective = lp.objective_value(result.x);

  for (double v : result.x) {
    if (!std::isfinite(v)) return finish(Status::NumericalError, "non-finite primal value");
  }
  return finish(Status::Optimal, "");
}

}  // namespace ps26119::oracle
