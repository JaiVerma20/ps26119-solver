// test_safe_bound.cpp — the certified (Neumaier–Shcherbina) bound must NEVER exceed the true
// optimum of a MIN model (never be below it for MAX), for ANY multipliers y, and must be
// tight when y is optimal.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "core/implied_bounds.h"
#include "core/safe_bound.h"
#include "io/lpm_reader.h"
#include "model_builder.h"
#include "ps26119/solve.h"

using namespace ps26119;

namespace {
std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }
const char* kNetlib[] = {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe"};

Solution oracle(const Model& m) {
  Options o;
  o.algorithm = Algorithm::Oracle;
  return solve(m, o);
}
}  // namespace

// With optimal duals the bound is finite unless a basic column (reduced cost exactly 0 in
// theory, 0 ± rounding in practice) has no finite bound even after bound propagation — then
// no rounding-proof certificate exists for that term and −∞ is the honest answer
// (adlittle, blend, share2b in this set). When finite, it must be tight.
TEST(SafeBound, TightAtOptimalDualsOnSmallNetlib) {
  int finite = 0;
  for (const char* name : kNetlib) {
    SCOPED_TRACE(name);
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    const Solution s = oracle(m);
    ASSERT_EQ(s.status, Status::Optimal);
    const SafeBound b = certified_dual_bound(m, s.y);
    EXPECT_EQ(s.certified_bound, b.bound);  // solve() stamps it
    if (!b.finite) {
      EXPECT_EQ(b.bound, -kInf);
      continue;
    }
    ++finite;
    EXPECT_LE(b.bound, s.objective + 1e-9 * (1 + std::fabs(s.objective)));  // oracle optimum is itself rounded
    EXPECT_NEAR(b.bound, s.objective, 1e-9 * (1 + std::fabs(s.objective)));
  }
  EXPECT_GE(finite, 7);
}

TEST(SafeBound, ImpliedBoundsContainTheOptimum) {
  for (const char* name : kNetlib) {
    SCOPED_TRACE(name);
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    const Solution s = oracle(m);
    const ImpliedBounds ib = implied_bounds(m);
    for (int j = 0; j < m.num_cols; ++j) {
      EXPECT_LE(ib.lower[j], s.x[j] + 1e-9 * (1 + std::fabs(s.x[j]))) << j;
      EXPECT_GE(ib.upper[j], s.x[j] - 1e-9 * (1 + std::fabs(s.x[j]))) << j;
      EXPECT_GE(ib.lower[j], m.col_lower[j]);  // never looser than the model
      EXPECT_LE(ib.upper[j], m.col_upper[j]);
    }
  }
}

TEST(SafeBound, ValidForRandomMultipliers) {
  // Any y gives a valid bound: perturb the optimal y heavily and check bound ≤ optimum.
  std::mt19937_64 rng(11);
  std::normal_distribution<double> N(0, 1);
  int finite = 0;
  for (const char* name : {"afiro", "sc50a", "blend", "recipe", "kb2"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    const Solution s = oracle(m);
    for (int t = 0; t < 50; ++t) {
      std::vector<double> y = s.y;
      const double scale = std::pow(10.0, -8 + t % 9);  // 1e-8 … 1 relative noise
      for (double& v : y) v += scale * (1 + std::fabs(v)) * N(rng);
      const SafeBound b = certified_dual_bound(m, y);
      if (!b.finite) continue;
      ++finite;
      EXPECT_LE(b.bound, s.objective + 1e-9 * (1 + std::fabs(s.objective))) << name << " trial " << t;
    }
  }
  EXPECT_GT(finite, 20);
}

TEST(SafeBound, MaxSenseIsAnUpperBound) {
  Model m;
  ASSERT_TRUE(io::read_lpm(data("hand/max_ranged.lpm"), m).ok);
  const Solution s = oracle(m);
  const SafeBound b = certified_dual_bound(m, s.y);
  ASSERT_TRUE(b.finite);
  EXPECT_GE(b.bound, 21.5 - 1e-12);
  EXPECT_NEAR(b.bound, 21.5, 1e-9);
  // wrong multipliers still give a valid (weaker) upper bound when finite
  std::vector<double> y = s.y;
  y[0] += 0.7;
  const SafeBound w = certified_dual_bound(m, y);
  if (w.finite) EXPECT_GE(w.bound, 21.5 - 1e-12);
}

TEST(SafeBound, InfiniteBoundsAreReportedNotFaked) {
  // min x s.t. x - y >= 0 ; x >= 0, y free: y's reduced cost must be exactly 0.
  Model m = test::make_model({1, 0}, {{1, -1}}, {0}, {kInf}, {0, -kInf}, {kInf, kInf});
  const SafeBound good = certified_dual_bound(m, {0.0});
  EXPECT_TRUE(good.finite);
  EXPECT_LE(good.bound, 0.0);
  const SafeBound bad = certified_dual_bound(m, {0.5});  // z_y = 0.5 on a free column (y ≥ 0 is right-signed)
  EXPECT_FALSE(bad.finite);
  EXPECT_EQ(bad.unbounded_col_terms, 1);
  EXPECT_EQ(bad.bound, -kInf);
}
