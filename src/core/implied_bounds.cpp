// implied_bounds.cpp — see implied_bounds.h.
#include "core/implied_bounds.h"

#include <cmath>
#include <limits>

#include "la/csr.h"
#include "la/dd.h"

namespace ps26119 {
namespace {

using la::dd;
constexpr double kInfD = std::numeric_limits<double>::infinity();
constexpr double kSlack = 1e-9;

double loosen_up(double v, double scale) {
  return std::nextafter(v + kSlack * (scale + std::fabs(v)), kInfD);
}
double loosen_down(double v, double scale) {
  return std::nextafter(v - kSlack * (scale + std::fabs(v)), -kInfD);
}

}  // namespace

ImpliedBounds implied_bounds(const Model& M, int max_passes) {
  ImpliedBounds r;
  r.lower = M.col_lower;
  r.upper = M.col_upper;
  const la::Csr<double> A = la::csr_from_model(M);  // row access
  auto& lo = r.lower;
  auto& up = r.upper;
  for (int pass = 0; pass < max_passes; ++pass) {
    bool changed = false;
    for (int i = 0; i < M.num_rows; ++i) {
      const bool has_up = std::isfinite(M.row_upper[i]), has_lo = std::isfinite(M.row_lower[i]);
      if (!has_up && !has_lo) continue;
      // min and max activity with counts of infinite contributions
      dd amin = 0.0, amax = 0.0;
      double mag = 0.0;
      int ninf_min = 0, ninf_max = 0, jinf_min = -1, jinf_max = -1;
      for (std::int64_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
        const int j = A.col[p];
        const double a = A.val[p];
        const double bmin = a > 0 ? lo[j] : up[j];  // bound giving the minimum of a·x_j
        const double bmax = a > 0 ? up[j] : lo[j];
        if (std::isfinite(bmin)) {
          const dd t = la::mul_exact(a, bmin);
          amin += t;
          mag += std::fabs(t.to_double());
        } else {
          ++ninf_min, jinf_min = j;
        }
        if (std::isfinite(bmax)) {
          const dd t = la::mul_exact(a, bmax);
          amax += t;
          mag += std::fabs(t.to_double());
        } else {
          ++ninf_max, jinf_max = j;
        }
      }
      for (std::int64_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
        const int j = A.col[p];
        const double a = A.val[p];
        const double scale = mag / std::fabs(a);
        // residual min activity of the other columns (finite only if they all are)
        if (has_up && (ninf_min == 0 || (ninf_min == 1 && jinf_min == j))) {
          const double bmin = a > 0 ? lo[j] : up[j];
          dd rest = amin;
          if (ninf_min == 0) rest -= la::mul_exact(a, bmin);
          const double v = ((dd(M.row_upper[i]) - rest) / dd(a)).to_double();  // a x_j ≤ ru − rest
          if (a > 0) {
            const double nb = loosen_up(v, scale);
            if (nb < up[j] - 1e-7 * (1 + std::fabs(nb))) up[j] = nb, changed = true;
          } else {
            const double nb = loosen_down(v, scale);
            if (nb > lo[j] + 1e-7 * (1 + std::fabs(nb))) lo[j] = nb, changed = true;
          }
        }
        if (has_lo && (ninf_max == 0 || (ninf_max == 1 && jinf_max == j))) {
          const double bmax = a > 0 ? up[j] : lo[j];
          dd rest = amax;
          if (ninf_max == 0) rest -= la::mul_exact(a, bmax);
          const double v = ((dd(M.row_lower[i]) - rest) / dd(a)).to_double();  // a x_j ≥ rl − rest
          if (a > 0) {
            const double nb = loosen_down(v, scale);
            if (nb > lo[j] + 1e-7 * (1 + std::fabs(nb))) lo[j] = nb, changed = true;
          } else {
            const double nb = loosen_up(v, scale);
            if (nb < up[j] - 1e-7 * (1 + std::fabs(nb))) up[j] = nb, changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }
  for (int j = 0; j < M.num_cols; ++j) {
    if (!std::isfinite(M.col_lower[j]) && std::isfinite(lo[j])) ++r.tightened;
    if (!std::isfinite(M.col_upper[j]) && std::isfinite(up[j])) ++r.tightened;
  }
  return r;
}

}  // namespace ps26119
