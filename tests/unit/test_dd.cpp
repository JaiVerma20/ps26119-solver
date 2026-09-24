// test_dd.cpp — double-double arithmetic: identities that fail in plain double but hold
// in dd, and ~1e-30 relative accuracy on cases with known exact answers.
#include <gtest/gtest.h>

#include <cmath>

#include "la/dd.h"

using ps26119::la::dd;
namespace la = ps26119::la;

namespace {
// |a − b| as a double (computed in dd, so it is accurate even when tiny).
double diff(const dd& a, const dd& b) { return std::fabs(la::abs(a - b).to_double()); }
}  // namespace

TEST(DoubleDouble, ErrorFreeTransforms) {
  double s, e;
  la::two_sum(1.0, 1e-20, s, e);
  EXPECT_EQ(s, 1.0);
  EXPECT_EQ(e, 1e-20);
  double p;
  const double a = 1.0 + std::ldexp(1.0, -30);
  la::two_prod(a, a, p, e);  // (1+2^-30)^2 = 1 + 2^-29 + 2^-60
  EXPECT_EQ(p, 1.0 + std::ldexp(1.0, -29));
  EXPECT_EQ(e, std::ldexp(1.0, -60));
}

TEST(DoubleDouble, AbsorptionThatPlainDoubleLoses) {
  // (1e16 + 1) − 1e16: plain double gives 0 (1e16+1 rounds to 1e16), dd gives 1.
  volatile double big = 1e16;
  EXPECT_EQ((big + 1.0) - big, 0.0);
  EXPECT_EQ(((dd(1e16) + 1.0) - 1e16).to_double(), 1.0);

  // 1 + 2^-80 − 1 = 2^-80 (below double epsilon, inside dd precision).
  const double tiny = std::ldexp(1.0, -80);
  EXPECT_EQ(1.0 + tiny - 1.0, 0.0);
  EXPECT_EQ(((dd(1.0) + tiny) - 1.0).to_double(), tiny);
}

TEST(DoubleDouble, ProductBeyondDoublePrecision) {
  // (1 + 2^-40)(1 − 2^-40) = 1 − 2^-80 exactly; plain double rounds it to exactly 1.
  const double e = std::ldexp(1.0, -40);
  EXPECT_EQ((1.0 + e) * (1.0 - e), 1.0);
  const dd p = (dd(1.0) + e) * (dd(1.0) - e);
  EXPECT_EQ(p.hi, 1.0);
  EXPECT_EQ(p.lo, -std::ldexp(1.0, -80));
}

TEST(DoubleDouble, TenthsSumToOne) {
  // Summing 0.1 ten times in double is not 1. In dd, summing the dd value of 1/10 ten
  // times is 1 to ~1e-31.
  double s = 0.0;
  for (int i = 0; i < 10; ++i) s += 0.1;
  EXPECT_NE(s, 1.0);
  const dd tenth = dd(1.0) / dd(10.0);
  dd t = 0.0;
  for (int i = 0; i < 10; ++i) t += tenth;
  EXPECT_LT(diff(t, dd(1.0)), 1e-30);
}

TEST(DoubleDouble, DivisionKnownExact) {
  // 1/3: hi = fl(1/3) = 6004799503160661·2^-54, so lo = 1/3 − hi = 2^-54/3 exactly
  // (3·6004799503160661 = 2^54 − 1).
  const dd q = dd(1.0) / dd(3.0);
  EXPECT_EQ(q.hi, 1.0 / 3.0);
  EXPECT_NEAR(q.lo, std::ldexp(1.0 / 3.0, -54), 1e-48);
  // x / y * y == x to ~1e-31 relative for awkward values (no subnormals: dd loses bits there)
  for (double x : {1.0, 7.0, 1e-200, 123456.789, -2.5e17}) {
    for (double y : {3.0, 7.0, 0.1, 1e-5, 9.999999999e10}) {
      const dd r = (dd(x) / y) * y;
      EXPECT_LT(diff(r, dd(x)) / std::fabs(x), 1e-30) << x << " / " << y;
    }
  }
}

TEST(DoubleDouble, SqrtTwo) {
  const dd s = la::sqrt(dd(2.0));
  EXPECT_EQ(s.hi, std::sqrt(2.0));
  EXPECT_LT(diff(s * s, dd(2.0)) / 2.0, 1e-31);
  // Known dd value of sqrt(2): lo = -9.667293313452913e-17 (to double precision).
  EXPECT_NEAR(s.lo, -9.6672933134529135e-17, 1e-31);
  EXPECT_EQ(la::sqrt(dd(0.0)).to_double(), 0.0);
  EXPECT_TRUE(std::isnan(la::sqrt(dd(-1.0)).hi));
  // perfect squares are exact
  EXPECT_EQ(la::sqrt(dd(144.0)), dd(12.0));
}

TEST(DoubleDouble, PiByMachin) {
  // π = 16 atan(1/5) − 4 atan(1/239), atan by its Taylor series in dd.
  auto atan_inv = [](int n) {
    const dd x = dd(1.0) / dd(n);
    const dd x2 = x * x;
    dd term = x, sum = x;
    for (int k = 1; k < 60; ++k) {
      term *= x2;
      const dd t = term / dd(2 * k + 1);
      sum = (k % 2) ? sum - t : sum + t;
    }
    return sum;
  };
  const dd pi = dd(16.0) * atan_inv(5) - dd(4.0) * atan_inv(239);
  // π as double-double: hi = 3.141592653589793, lo = 1.2246467991473532e-16.
  EXPECT_EQ(pi.hi, 3.141592653589793);
  EXPECT_NEAR(pi.lo, 1.2246467991473532e-16, 1e-30);
}

TEST(DoubleDouble, ComparisonsAndAbs) {
  const dd a = dd(1.0) + std::ldexp(1.0, -70);
  const dd b = dd(1.0);
  EXPECT_TRUE(a > b);
  EXPECT_TRUE(b < a);
  EXPECT_TRUE(a >= a);
  EXPECT_TRUE(b != a);
  EXPECT_EQ(la::abs(-a), a);
  EXPECT_TRUE(la::abs(dd(0.0, -1e-40)) > dd(0.0));
  EXPECT_EQ(static_cast<double>(a), 1.0);
}

TEST(DoubleDouble, CancellationInDotProduct) {
  // x·y with catastrophic cancellation: [1e20, 1, -1e20]·[1, 1, 1] = 1.
  const double x[] = {1e20, 1.0, -1e20};
  double plain = 0.0;
  dd acc = 0.0;
  for (double v : x) {
    plain += v;
    acc += v;
  }
  EXPECT_EQ(plain, 0.0);
  EXPECT_EQ(acc.to_double(), 1.0);
}
