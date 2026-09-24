// dd.h — double-double arithmetic (≈106-bit significand, ≈32 decimal digits).
//
// Purpose: extra precision for the test oracle and for residual / certificate
// computations. On Apple Silicon `long double` is the same 64-bit type as `double`, so
// we never rely on it (CLAUDE.md §3).
//
// A value is the unevaluated sum hi + lo with |lo| ≤ ulp(hi)/2 (normalised).
//
// Citations:
//   T. J. Dekker, "A floating-point technique for extending the available precision",
//     Numer. Math. 18 (1971) — two_prod splitting, dd add/mul/div.
//   D. E. Knuth, TAOCP vol. 2, §4.2.2 — error-free two_sum.
//   Y. Hida, X. S. Li, D. H. Bailey, "Library for double-double and quad-double
//     arithmetic" (QD, 2000/2008) — the accurate ("IEEE") dd+dd addition and the
//     three-step long division follow the approach described there (our own code).
//   A. H. Karp, P. Markstein, "High-precision division and square root" (1997) — sqrt.
//
// Invariants / assumptions:
//   * IEEE-754 binary64 with round-to-nearest; must NOT be compiled with -ffast-math
//     (checked below). FP contraction is harmless: two_prod uses std::fma explicitly and
//     two_sum contains no multiplications.
//   * Relative error of +, −, × is ≲ 4·2⁻¹⁰⁶ (≈ 5e-32); ÷ and sqrt ≲ 1e-31. Inf/NaN
//     propagate through hi (lo is then meaningless).
#pragma once

#include <cmath>

#if defined(__FAST_MATH__)
#error "dd.h requires IEEE semantics: do not compile with -ffast-math"
#endif

namespace ps26119::la {

// s + e == a + b exactly, s = fl(a + b).
inline constexpr void two_sum(double a, double b, double& s, double& e) {
  s = a + b;
  const double bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// Requires |a| ≥ |b| (or a == 0). s + e == a + b exactly.
inline constexpr void quick_two_sum(double a, double b, double& s, double& e) {
  s = a + b;
  e = b - (s - a);
}

// p + e == a * b exactly (barring overflow/underflow), p = fl(a * b).
inline void two_prod(double a, double b, double& p, double& e) {
  p = a * b;
  e = std::fma(a, b, -p);
}

struct dd {
  double hi = 0.0;
  double lo = 0.0;

  constexpr dd() = default;
  constexpr dd(double h) : hi(h), lo(0.0) {}  // NOLINT: implicit on purpose (exact)
  constexpr dd(int v) : hi(static_cast<double>(v)), lo(0.0) {}  // NOLINT
  constexpr dd(double h, double l) : hi(h), lo(l) {}

  // Normalise an arbitrary pair.
  static constexpr dd from_sum(double a, double b) {
    dd r;
    two_sum(a, b, r.hi, r.lo);
    return r;
  }

  explicit constexpr operator double() const { return hi + lo; }
  constexpr double to_double() const { return hi + lo; }

  constexpr dd operator-() const { return {-hi, -lo}; }

  dd& operator+=(const dd& b);
  dd& operator-=(const dd& b) { return *this += -b; }
  dd& operator*=(const dd& b);
  dd& operator/=(const dd& b);
};

// ------------------------------------------------------------------ addition
inline dd operator+(const dd& a, const dd& b) {
  // Accurate addition: two_sum on both halves, then renormalise twice.
  double s1, s2, t1, t2;
  two_sum(a.hi, b.hi, s1, s2);
  two_sum(a.lo, b.lo, t1, t2);
  s2 += t1;
  quick_two_sum(s1, s2, s1, s2);
  s2 += t2;
  dd r;
  quick_two_sum(s1, s2, r.hi, r.lo);
  return r;
}

inline dd operator+(const dd& a, double b) {
  double s1, s2;
  two_sum(a.hi, b, s1, s2);
  s2 += a.lo;
  dd r;
  quick_two_sum(s1, s2, r.hi, r.lo);
  return r;
}
inline dd operator+(double a, const dd& b) { return b + a; }
inline dd operator-(const dd& a, const dd& b) { return a + (-b); }
inline dd operator-(const dd& a, double b) { return a + (-b); }
inline dd operator-(double a, const dd& b) { return (-b) + a; }

// ------------------------------------------------------------------ multiplication
inline dd operator*(const dd& a, const dd& b) {
  double p1, p2;
  two_prod(a.hi, b.hi, p1, p2);
  p2 += a.hi * b.lo + a.lo * b.hi;
  dd r;
  quick_two_sum(p1, p2, r.hi, r.lo);
  return r;
}

inline dd operator*(const dd& a, double b) {
  double p1, p2;
  two_prod(a.hi, b, p1, p2);
  p2 += a.lo * b;
  dd r;
  quick_two_sum(p1, p2, r.hi, r.lo);
  return r;
}
inline dd operator*(double a, const dd& b) { return b * a; }

// Exact product of two doubles as a dd.
inline dd mul_exact(double a, double b) {
  dd r;
  two_prod(a, b, r.hi, r.lo);
  return r;
}

// ------------------------------------------------------------------ division
inline dd operator/(const dd& a, const dd& b) {
  // Long division with three partial quotients; each remainder is formed in dd.
  const double q1 = a.hi / b.hi;
  if (!std::isfinite(q1)) return dd(q1);
  dd r = a - b * q1;
  const double q2 = r.hi / b.hi;
  r -= b * q2;
  const double q3 = r.hi / b.hi;
  dd q;
  quick_two_sum(q1, q2, q.hi, q.lo);
  return q + q3;
}
inline dd operator/(const dd& a, double b) { return a / dd(b); }
inline dd operator/(double a, const dd& b) { return dd(a) / b; }

inline dd& dd::operator+=(const dd& b) { return *this = *this + b; }
inline dd& dd::operator*=(const dd& b) { return *this = *this * b; }
inline dd& dd::operator/=(const dd& b) { return *this = *this / b; }

// ------------------------------------------------------------------ comparisons
// Normalised values compare lexicographically on (hi, lo).
inline constexpr bool operator==(const dd& a, const dd& b) { return a.hi == b.hi && a.lo == b.lo; }
inline constexpr bool operator<(const dd& a, const dd& b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline constexpr bool operator>(const dd& a, const dd& b) { return b < a; }
inline constexpr bool operator<=(const dd& a, const dd& b) { return !(b < a); }
inline constexpr bool operator>=(const dd& a, const dd& b) { return !(a < b); }
inline constexpr bool operator!=(const dd& a, const dd& b) { return !(a == b); }

// ------------------------------------------------------------------ functions
inline dd abs(const dd& a) { return a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0) ? -a : a; }

inline dd sqrt(const dd& a) {
  // Karp's trick: x = 1/sqrt(a.hi) in double, then sqrt(a) ≈ a·x + (a − (a·x)²)·x/2.
  if (a.hi <= 0.0) return dd(a.hi == 0.0 ? 0.0 : std::nan(""));
  const double x = 1.0 / std::sqrt(a.hi);
  const double ax = a.hi * x;
  const dd ax2 = mul_exact(ax, ax);
  const double corr = (a - ax2).hi * (x * 0.5);
  return dd::from_sum(ax, corr);
}

inline bool isfinite(const dd& a) { return std::isfinite(a.hi); }

}  // namespace ps26119::la
