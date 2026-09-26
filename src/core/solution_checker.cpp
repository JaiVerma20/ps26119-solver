// solution_checker.cpp - verifies an LP optimum from first principles.
//
// Everything is evaluated in minimisation form (a maximisation problem is
// negated first). With d = c - A^T y, the conditions for optimality of
//     min c^T x   s.t.  L <= A x <= U,  l <= x <= u
// are
//   primal feasibility   L <= A x <= U  and  l <= x <= u
//   dual feasibility     y_i >= 0 if only L_i is finite, y_i <= 0 if only U_i
//                        is finite, y_i = 0 for a free row; same for d_j with
//                        the column bounds
//   zero duality gap     c^T x  ==  sum_i (y_i > 0 ? y_i L_i : y_i U_i)
//                                 + sum_j (d_j > 0 ? d_j l_j : d_j u_j)
// Primal + dual feasibility + zero gap is a complete optimality proof.

#include "gpuopt/solution_checker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gpuopt {
namespace {

struct DualTerm {
  double violation;     // wrong-sign part of the multiplier
  double contribution;  // multiplier times the active bound
  double complementarity;
};

// Multiplier `mult` attached to a quantity with bounds [lo, up] and value `val`.
DualTerm evaluate_multiplier(double mult, double lo, double up, double val) {
  DualTerm t{0.0, 0.0, 0.0};
  if (mult > 0.0) {
    if (lo == -kInf) {
      t.violation = mult;  // positive multiplier needs a finite lower bound
    } else {
      t.contribution = mult * lo;
      t.complementarity = mult * std::fabs(val - lo);
    }
  } else if (mult < 0.0) {
    if (up == kInf) {
      t.violation = -mult;  // negative multiplier needs a finite upper bound
    } else {
      t.contribution = mult * up;
      t.complementarity = -mult * std::fabs(up - val);
    }
  }
  return t;
}

double bound_violation(double val, double lo, double up) {
  double v = 0.0;
  if (lo != -kInf) v = std::max(v, (lo - val) / (1.0 + std::fabs(lo)));
  if (up != kInf) v = std::max(v, (val - up) / (1.0 + std::fabs(up)));
  return v;
}

}  // namespace

CheckReport check_solution(const LpProblem& lp, const std::vector<double>& x,
                           const std::vector<double>& row_dual, const CheckTolerances& tol) {
  CheckReport rep;
  const int m = lp.num_rows();
  const int n = lp.num_cols();
  const double sense = lp.sense == ObjSense::kMinimize ? 1.0 : -1.0;

  // Primal feasibility.
  std::vector<double> ax;
  lp.A.multiply(x, ax);
  for (int i = 0; i < m; ++i) {
    rep.max_primal_violation =
        std::max(rep.max_primal_violation, bound_violation(ax[i], lp.row_lower[i], lp.row_upper[i]));
  }
  for (int j = 0; j < n; ++j) {
    rep.max_primal_violation =
        std::max(rep.max_primal_violation, bound_violation(x[j], lp.col_lower[j], lp.col_upper[j]));
  }

  // Dual feasibility and dual objective, in minimisation form.
  std::vector<double> y(m);
  for (int i = 0; i < m; ++i) y[i] = sense * row_dual[i];
  std::vector<double> aty;
  lp.A.multiply_transpose(y, aty);

  double dual_obj_min = 0.0;
  double primal_obj_min = 0.0;
  double c_scale = 1.0;
  for (int j = 0; j < n; ++j) c_scale = std::max(c_scale, std::fabs(lp.obj[j]));

  for (int i = 0; i < m; ++i) {
    const DualTerm t = evaluate_multiplier(y[i], lp.row_lower[i], lp.row_upper[i], ax[i]);
    rep.max_dual_violation = std::max(rep.max_dual_violation, t.violation / c_scale);
    rep.max_complementarity = std::max(rep.max_complementarity, t.complementarity);
    dual_obj_min += t.contribution;
  }
  for (int j = 0; j < n; ++j) {
    const double c = sense * lp.obj[j];
    const double d = c - aty[j];
    const DualTerm t = evaluate_multiplier(d, lp.col_lower[j], lp.col_upper[j], x[j]);
    rep.max_dual_violation = std::max(rep.max_dual_violation, t.violation / c_scale);
    rep.max_complementarity = std::max(rep.max_complementarity, t.complementarity);
    dual_obj_min += t.contribution;
    primal_obj_min += c * x[j];
  }

  // Back to the problem's own sense, including the constant term.
  rep.primal_objective = sense * primal_obj_min + lp.obj_offset;
  rep.dual_objective = sense * dual_obj_min + lp.obj_offset;
  rep.relative_gap = std::fabs(rep.primal_objective - rep.dual_objective) /
                     (1.0 + std::fabs(rep.primal_objective) + std::fabs(rep.dual_objective));

  rep.primal_ok = rep.max_primal_violation <= tol.primal;
  rep.dual_ok = rep.max_dual_violation <= tol.dual;
  rep.gap_ok = rep.relative_gap <= tol.gap;
  return rep;
}

std::string CheckReport::summary() const {
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "primal viol %.1e  dual viol %.1e  compl %.1e  gap %.1e  => %s",
                max_primal_violation, max_dual_violation, max_complementarity, relative_gap,
                passed() ? "PASS" : "FAIL");
  return buf;
}

}  // namespace gpuopt
