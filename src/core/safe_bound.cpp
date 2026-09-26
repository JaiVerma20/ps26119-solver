// safe_bound.cpp — see safe_bound.h for the math.
//
// Error model (conservative): a double-double addition has relative error ≤ 2⁻¹⁰⁴ of its
// result's magnitude (the "accurate" dd add has ≈ 2·2⁻¹⁰⁶); products of two doubles are
// exact by two_prod unless they underflow, for which we add an absolute 2⁻⁹⁶⁰ slack per term.
// Hence after accumulating k terms t_1..t_k the error is ≤ k·2⁻¹⁰⁴·Σ|t| + k·2⁻⁹⁶⁰. We double
// these for margin. Converting a dd v to a double lower bound uses nextafter(double(v − e), −∞)
// (one full ulp covers the round-to-nearest conversion).
#include "core/safe_bound.h"

#include <cmath>
#include <limits>

#include "core/implied_bounds.h"
#include "la/dd.h"

namespace ps26119 {
namespace {

using la::dd;

constexpr double kRel = 2.0 * 4.930380657631324e-32;  // 2 · 2⁻¹⁰⁴
constexpr double kAbs = 2.0 * 1.0e-289;               // ≥ 2 · 2⁻⁹⁶⁰, generous underflow slack

// A value that is provably exact (no error, representable as one double) is not widened.
double down(const dd& v, double err) {
  if (err == 0.0 && v.lo == 0.0) return v.hi;
  const double d = (v - dd(err)).to_double();
  return std::nextafter(d, -std::numeric_limits<double>::infinity());
}
double up(const dd& v, double err) {
  if (err == 0.0 && v.lo == 0.0) return v.hi;
  const double d = (v + dd(err)).to_double();
  return std::nextafter(d, std::numeric_limits<double>::infinity());
}
// two_prod is exact unless the product is subnormal/underflows; flag that case.
bool may_underflow(const dd& p) { return p.hi != 0.0 && std::fabs(p.hi) < 1e-280; }

// Exact sum of doubles as a nonoverlapping expansion (Shewchuk, "Adaptive precision
// floating-point arithmetic and fast robust geometric predicates", 1997: GROW-EXPANSION with
// zero elimination). Knuth's two-sum is error-free barring overflow, so `e` represents the sum
// exactly. Returns false (caller falls back) on non-finite values or a too-long expansion.
bool grow_expansion(std::vector<double>& e, double b, std::vector<double>& scratch) {
  if (b == 0.0) return true;
  if (!std::isfinite(b)) return false;
  scratch.clear();
  double q = b;
  for (double ei : e) {
    const double sum = q + ei;
    const double bv = sum - q;
    const double av = sum - bv;
    const double err = (q - av) + (ei - bv);
    q = sum;
    if (err != 0.0) scratch.push_back(err);
  }
  if (!std::isfinite(q)) return false;
  if (q != 0.0) scratch.push_back(q);
  e.swap(scratch);
  return e.size() <= 64;
}

// Accumulates dd terms with a running error bound.
struct SafeSum {
  dd sum = 0.0;
  double abs_sum = 0.0;
  int count = 0;
  bool minus_inf = false;
  void add(const dd& t) {
    sum += t;
    abs_sum += std::fabs(t.to_double());
    ++count;
  }
  double error() const { return (count + 2) * (kRel * abs_sum * 1.0000001 + kAbs); }
};

// min over x ∈ [lo, up] of z·x for a fixed z (exact product as dd); sets `inf` when −∞.
dd term(double z, double lo, double hi, bool& inf) {
  if (z > 0) {
    if (!std::isfinite(lo)) return inf = true, dd(0.0);
    return la::mul_exact(z, lo);
  }
  if (z < 0) {
    if (!std::isfinite(hi)) return inf = true, dd(0.0);
    return la::mul_exact(z, hi);
  }
  return dd(0.0);
}

}  // namespace

namespace {

SafeBound bound_with(const Model& M, const std::vector<double>& y_orig, const std::vector<double>& col_lo,
                     const std::vector<double>& col_up) {
  SafeBound r;
  const int m = M.num_rows, n = M.num_cols;
  const double sense = M.sense;
  if (static_cast<int>(y_orig.size()) != m) return r;
  SafeSum L;
  // ---- row terms: min over s ∈ [rl, ru] of y_i s  (y in min form: sense·y, exact sign flip)
  for (int i = 0; i < m; ++i) {
    const double yi = sense * y_orig[i];
    bool inf = false;
    const dd t = term(yi, M.row_lower[i], M.row_upper[i], inf);
    if (inf) ++r.unbounded_row_terms;
    else L.add(t);
  }
  // ---- column terms with z_j enclosed in an interval
  std::vector<double> ex, scratch;
  ex.reserve(16);
  scratch.reserve(17);
  for (int j = 0; j < n; ++j) {
    dd z = sense * M.obj[j];
    double mag = 0.0;  // Σ|terms| except the first (the first assignment is exact)
    int cnt = 0;
    bool uf = false;
    for (int p = M.col_start[j]; p < M.col_start[j + 1]; ++p) {
      const dd prod = la::mul_exact(M.value[p], sense * y_orig[M.row_index[p]]);
      if (prod.hi == 0.0 && prod.lo == 0.0 && M.value[p] * y_orig[M.row_index[p]] == 0.0) continue;  // exact 0
      z -= prod;
      mag += std::fabs(prod.to_double()) + std::fabs(z.to_double());
      uf = uf || may_underflow(prod);
      ++cnt;
    }
    const double ez = cnt == 0 ? 0.0 : (cnt + 2) * (kRel * mag * 1.0000001 + (uf ? kAbs : 0.0));
    double zlo = down(z, ez), zhi = up(z, ez);
    if (cnt > 0 && !uf) {
      // Tighter, still rigorous: the exact value of z as an expansion. Exact cancellation gives
      // exactly 0 (a FREE column then contributes 0 instead of −∞), a one-component expansion
      // is an exact double. Otherwise enclose the dd sum of the components.
      ex.clear();
      bool ok = grow_expansion(ex, sense * M.obj[j], scratch);
      for (int p = M.col_start[j]; ok && p < M.col_start[j + 1]; ++p) {
        const dd prod = la::mul_exact(M.value[p], sense * y_orig[M.row_index[p]]);
        ok = grow_expansion(ex, -prod.hi, scratch) && grow_expansion(ex, -prod.lo, scratch);
      }
      if (ok) {
        if (ex.empty()) {
          zlo = zhi = 0.0;
        } else if (ex.size() == 1) {
          zlo = zhi = ex[0];
        } else {
          dd v = 0.0;
          for (double c : ex) v += c;  // smallest first
          const double ev = static_cast<double>(ex.size() + 2) * kRel * std::fabs(v.to_double()) * 1.0000001;
          zlo = std::max(zlo, down(v, ev));
          zhi = std::min(zhi, up(v, ev));
        }
      }
    }
    bool inf_lo = false, inf_hi = false;
    const dd tlo = term(zlo, col_lo[j], col_up[j], inf_lo);
    const dd thi = term(zhi, col_lo[j], col_up[j], inf_hi);
    if (inf_lo || inf_hi) {
      ++r.unbounded_col_terms;
      continue;
    }
    L.add(tlo < thi ? tlo : thi);
  }
  if (r.unbounded_row_terms > 0 || r.unbounded_col_terms > 0) {
    r.finite = false;
    r.bound = sense > 0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    return r;
  }
  // L bounds min c̃ᵀx from below. MIN: optimum ≥ L + offset. MAX: max cᵀx = −min(−cᵀx) ≤ −L.
  const double err = L.error() + kAbs;
  r.finite = true;
  if (sense > 0) {
    r.bound = down(L.sum + dd(M.obj_offset), err + kRel * std::fabs(M.obj_offset) * 4);
  } else {
    r.bound = up(-L.sum + dd(M.obj_offset), err + kRel * std::fabs(M.obj_offset) * 4);
  }
  return r;
}

}  // namespace

SafeBound certified_dual_bound(const Model& M, const std::vector<double>& y_in) {
  if (static_cast<int>(y_in.size()) != M.num_rows) return SafeBound{};
  // The bound is valid for ANY multipliers, so first zero the ones that point at an infinite
  // row bound (typically rounding-sized wrong signs on inactive rows); z is recomputed from
  // the modified y, so nothing is lost in rigour.
  std::vector<double> y = y_in;
  for (int i = 0; i < M.num_rows; ++i) {
    const double ym = M.sense * y[i];
    if ((ym > 0 && !std::isfinite(M.row_lower[i])) || (ym < 0 && !std::isfinite(M.row_upper[i]))) y[i] = 0.0;
  }
  SafeBound r = bound_with(M, y, M.col_lower, M.col_upper);
  if (r.unbounded_col_terms > 0 && r.unbounded_row_terms == 0) {
    // Columns bounded on one side only: use rigorous implied bounds (implied_bounds.h).
    const ImpliedBounds ib = implied_bounds(M);
    r = bound_with(M, y, ib.lower, ib.upper);
  }
  return r;
}

}  // namespace ps26119
