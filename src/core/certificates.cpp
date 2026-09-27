// certificates.cpp — see certificates.h.
#include "core/certificates.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core/safe_bound.h"
#include "core/solution_checker.h"
#include "ps26119/tolerances.h"

namespace ps26119 {

CertificateCheck check_infeasibility_certificate(const Model& model, const std::vector<double>& r) {
  CertificateCheck c;
  if (static_cast<int>(r.size()) != model.num_rows || model.num_rows == 0) {
    c.detail = "no dual ray";
    return c;
  }
  c.present = true;
  for (double v : r) {
    if (!std::isfinite(v)) {
      c.detail = "dual ray is not finite";
      return c;
    }
  }
  Model zero = model;  // same feasible set, objective 0: its certified bound is L₀(r)
  zero.sense = 1;
  zero.obj_offset = 0.0;
  std::fill(zero.obj.begin(), zero.obj.end(), 0.0);
  zero.is_integer.clear();
  const SafeBound sb = certified_dual_bound(zero, r);
  char buf[200];
  if (sb.finite && sb.bound > 0.0) {
    c.passed = c.rigorous = true;
    c.measure = sb.bound;
    std::snprintf(buf, sizeof buf, "Farkas certificate: certified L0(r) = %.3g > 0 (rounding-proof)", sb.bound);
    c.detail = buf;
    return c;
  }
  // Stage 2: tolerance test (fp64), same normalisation as the first-order engines.
  const int m = model.num_rows, n = model.num_cols;
  double rmax = 0.0, obj = 0.0, viol = 0.0;
  for (int i = 0; i < m; ++i) {
    rmax = std::max(rmax, std::fabs(r[i]));
    if (r[i] > 0) {
      if (std::isfinite(model.row_lower[i])) obj += r[i] * model.row_lower[i];
      else viol = std::max(viol, r[i]);
    } else if (r[i] < 0) {
      if (std::isfinite(model.row_upper[i])) obj += r[i] * model.row_upper[i];
      else viol = std::max(viol, -r[i]);
    }
  }
  std::vector<double> atr(n, 0.0);
  for (int j = 0; j < n; ++j)
    for (int k = model.col_start[j]; k < model.col_start[j + 1]; ++k) atr[j] += model.value[k] * r[model.row_index[k]];
  for (int j = 0; j < n; ++j) {
    const double lam = -atr[j];
    if (lam > 0) {
      if (std::isfinite(model.col_lower[j])) obj += lam * model.col_lower[j];
      else viol = std::max(viol, lam);
    } else if (lam < 0) {
      if (std::isfinite(model.col_upper[j])) obj += lam * model.col_upper[j];
      else viol = std::max(viol, -lam);
    }
  }
  const double scale = std::max(rmax, viol);
  c.measure = scale > 0 ? obj / scale : 0.0;
  c.violation = scale > 0 ? viol / scale : 0.0;
  c.passed = std::isfinite(c.measure) && c.measure > 0 && c.violation <= tol::kVerifyRay * c.measure;
  std::snprintf(buf, sizeof buf,
                "Farkas certificate: rigorous L0(r) = %.3g; tolerance test L0/s = %.3g, violation/s = %.3g (%s)",
                sb.bound, c.measure, c.violation,
                c.passed ? "infeasibility shown within tolerance, not rounding-proof" : "FAIL");
  c.detail = buf;
  return c;
}

CertificateCheck check_unboundedness_certificate(const Model& model, const std::vector<double>& x,
                                                 const std::vector<double>& ray) {
  CertificateCheck c;
  const int m = model.num_rows, n = model.num_cols;
  if (static_cast<int>(ray.size()) != n || static_cast<int>(x.size()) != n || n == 0) {
    c.detail = "no primal ray / point";
    return c;
  }
  c.present = true;
  double dmax = 0.0, cmax = 0.0;
  for (int j = 0; j < n; ++j) {
    if (!std::isfinite(ray[j]) || !std::isfinite(x[j])) {
      c.detail = "ray or point is not finite";
      return c;
    }
    dmax = std::max(dmax, std::fabs(ray[j]));
    cmax = std::max(cmax, std::fabs(model.obj[j]));
  }
  if (dmax == 0.0) {
    c.detail = "zero ray";
    return c;
  }
  // 1. x is feasible (the same primal test the gate applies to an Optimal answer).
  const CheckReport rep = check_solution(model, x, std::vector<double>(m, 0.0));
  // 2. d (scaled to ‖d‖∞ = 1) decreases the objective and lies in the recession cone.
  std::vector<double> d(n);
  double cd = 0.0, viol = 0.0;
  for (int j = 0; j < n; ++j) {
    d[j] = ray[j] / dmax;
    cd += model.sense * model.obj[j] * d[j];
    if (std::isfinite(model.col_lower[j])) viol = std::max(viol, -d[j]);
    if (std::isfinite(model.col_upper[j])) viol = std::max(viol, d[j]);
  }
  const std::vector<double> ad = model.row_activity(d);
  for (int i = 0; i < m; ++i) {
    if (std::isfinite(model.row_lower[i])) viol = std::max(viol, -ad[i]);
    if (std::isfinite(model.row_upper[i])) viol = std::max(viol, ad[i]);
  }
  c.measure = -cd;
  c.violation = viol;
  const bool decreases = -cd >= tol::kVerifyRay * std::max(1.0, cmax);
  const bool in_cone = viol <= tol::kVerifyRay * (-cd);
  c.passed = rep.primal_ok && decreases && in_cone;
  char buf[200];
  std::snprintf(buf, sizeof buf, "unbounded ray: -c'd = %.3g, cone violation %.3g, point primal violation %.3g (%s)",
                -cd, viol, rep.max_primal_violation, c.passed ? "unboundedness proven" : "FAIL");
  c.detail = buf;
  return c;
}

}  // namespace ps26119
