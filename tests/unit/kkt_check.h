// kkt_check.h — test helper: independent KKT check of a Solution against a Model
// (same definitions as tools/verify.py, re-implemented in C++ for unit tests).
#pragma once

#include <algorithm>
#include <cmath>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::test {

struct Kkt {
  double primal = 0, dual = 0, gap = 0;  // relative, as in tools/verify.py
  double objective = 0;
};

inline Kkt kkt(const Model& m, const Solution& s) {
  Kkt r;
  const double sense = m.sense;
  const auto ax = m.row_activity(s.x);
  auto viol = [](double v, double lo, double up) {
    const double below = lo - v, above = v - up;
    if (below > 0) return below / (1 + std::fabs(lo));
    if (above > 0) return above / (1 + std::fabs(up));
    return 0.0;
  };
  for (int j = 0; j < m.num_cols; ++j) r.primal = std::max(r.primal, viol(s.x[j], m.col_lower[j], m.col_upper[j]));
  for (int i = 0; i < m.num_rows; ++i) r.primal = std::max(r.primal, viol(ax[i], m.row_lower[i], m.row_upper[i]));
  // z = c − Aᵀy recomputed from y
  std::vector<double> z(m.obj);
  for (int j = 0; j < m.num_cols; ++j)
    for (int k = m.col_start[j]; k < m.col_start[j + 1]; ++k) z[j] -= m.value[k] * s.y[m.row_index[k]];
  double cmax = 0, dv = 0, p = 0, d = 0;
  for (double c : m.obj) cmax = std::max(cmax, std::fabs(c));
  auto dual_part = [&](double v, double lo, double up) {
    v *= sense;
    if (v > 0) {
      if (std::isfinite(lo)) d += v * lo;
      else dv = std::max(dv, v);
    } else if (v < 0) {
      if (std::isfinite(up)) d += v * up;
      else dv = std::max(dv, -v);
    }
  };
  for (int i = 0; i < m.num_rows; ++i) dual_part(s.y[i], m.row_lower[i], m.row_upper[i]);
  for (int j = 0; j < m.num_cols; ++j) dual_part(z[j], m.col_lower[j], m.col_upper[j]);
  for (int j = 0; j < m.num_cols; ++j) p += sense * m.obj[j] * s.x[j];
  r.dual = dv / (1 + cmax);
  r.gap = std::fabs(p - d) / (1 + std::fabs(p) + std::fabs(d));
  r.objective = m.objective_value(s.x);
  return r;
}

}  // namespace ps26119::test
