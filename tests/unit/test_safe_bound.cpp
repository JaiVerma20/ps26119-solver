// test_safe_bound.cpp — the certified (Neumaier–Shcherbina) bound must NEVER exceed the true
// optimum of a MIN model (never be below it for MAX), for ANY multipliers y, and must be
// tight when y is optimal.
#include <gtest/gtest.h>
#include "la/dd.h"
#include "la/csr.h"
#include <limits>
#include <cstring>

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

// Exact reduced costs (expansion arithmetic): an exactly cancelling z_j on a FREE column
// contributes 0, not −inf (before, any column with terms got a nonzero error interval), and a
// cancellation that plain doubles get wrong is handled exactly. The final sum is still
// rounded outward, so the bound is ≤ the true value 0 but within 1e-20 of it.
TEST(SafeBound, ExactCancellationOnFreeColumns) {
  // min 0·x + 0·y s.t. x + y = 1 (row 0), x + y = 1 (row 1); x, y free; y = (0.5, -0.5):
  // z = -(0.5 - 0.5) = 0 exactly -> bound = 0.5·1 + (-0.5)·1 = 0.
  Model m = test::make_model({0, 0}, {{1, 1}, {1, 1}}, {1, 1}, {1, 1}, {-kInf, -kInf}, {kInf, kInf});
  SafeBound b = certified_dual_bound(m, {0.5, -0.5});
  ASSERT_TRUE(b.finite);
  EXPECT_LE(b.bound, 0.0);  // valid: the final sum is rounded outward
  EXPECT_GT(b.bound, -1e-20);  // and tight
  // Coefficients 1e16 and 1: z = 1 - (1e16·1 + 1·1 - 1e16·1) = 0 exactly with y = (1, 1, -1),
  // although the naive left-to-right double sum 1e16 + 1 - 1e16 gives 0 for the inner part.
  Model big = test::make_model({1}, {{1e16}, {1}, {1e16}}, {0, 0, 0}, {0, 0, 0}, {-kInf}, {kInf});
  b = certified_dual_bound(big, {1, 1, -1});
  ASSERT_TRUE(b.finite);  // free column, z exactly 0
  EXPECT_LE(b.bound, 0.0);
  EXPECT_GT(b.bound, -1e-20);
  // A z that is NOT zero on a free column must still give −inf (never a fake finite bound).
  b = certified_dual_bound(m, {0.5, -0.4999999999999999});
  EXPECT_FALSE(b.finite);
}

namespace {
// The previous full-scan implementation of implied_bounds (every row, every pass), kept here as
// the reference: the worklist version must return bit-identical bounds.
ImpliedBounds implied_bounds_full_scan(const Model& M, int max_passes = 20) {
  using la::dd;
  const double inf = std::numeric_limits<double>::infinity();
  auto loosen_up = [&](double v, double s) { return std::nextafter(v + 1e-9 * (s + std::fabs(v)), inf); };
  auto loosen_down = [&](double v, double s) { return std::nextafter(v - 1e-9 * (s + std::fabs(v)), -inf); };
  ImpliedBounds r;
  r.lower = M.col_lower;
  r.upper = M.col_upper;
  const la::Csr<double> A = la::csr_from_model(M);
  auto& lo = r.lower;
  auto& up = r.upper;
  for (int pass = 0; pass < max_passes; ++pass) {
    bool changed = false;
    for (int i = 0; i < M.num_rows; ++i) {
      const bool has_up = std::isfinite(M.row_upper[i]), has_lo = std::isfinite(M.row_lower[i]);
      if (!has_up && !has_lo) continue;
      dd amin = 0.0, amax = 0.0;
      double mag = 0.0;
      int ninf_min = 0, ninf_max = 0, jinf_min = -1, jinf_max = -1;
      for (std::int64_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
        const int j = A.col[p];
        const double a = A.val[p];
        const double bmin = a > 0 ? lo[j] : up[j], bmax = a > 0 ? up[j] : lo[j];
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
        if (has_up && (ninf_min == 0 || (ninf_min == 1 && jinf_min == j))) {
          dd rest = amin;
          if (ninf_min == 0) rest -= la::mul_exact(a, a > 0 ? lo[j] : up[j]);
          const double v = ((dd(M.row_upper[i]) - rest) / dd(a)).to_double();
          if (a > 0) {
            const double nb = loosen_up(v, scale);
            if (nb < up[j] - 1e-7 * (1 + std::fabs(nb))) up[j] = nb, changed = true;
          } else {
            const double nb = loosen_down(v, scale);
            if (nb > lo[j] + 1e-7 * (1 + std::fabs(nb))) lo[j] = nb, changed = true;
          }
        }
        if (has_lo && (ninf_max == 0 || (ninf_max == 1 && jinf_max == j))) {
          dd rest = amax;
          if (ninf_max == 0) rest -= la::mul_exact(a, a > 0 ? up[j] : lo[j]);
          const double v = ((dd(M.row_lower[i]) - rest) / dd(a)).to_double();
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
  return r;
}
}  // namespace

TEST(SafeBound, ImpliedBoundsWorklistIsBitIdenticalToFullScan) {
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  int nontrivial = 0;
  auto compare = [&](const Model& m, const std::string& what) {
    const ImpliedBounds a = implied_bounds(m), b = implied_bounds_full_scan(m);
    for (int j = 0; j < m.num_cols; ++j) {
      ASSERT_EQ(std::memcmp(&a.lower[j], &b.lower[j], sizeof(double)), 0) << what << " lower " << j;
      ASSERT_EQ(std::memcmp(&a.upper[j], &b.upper[j], sizeof(double)), 0) << what << " upper " << j;
      nontrivial += a.lower[j] != m.col_lower[j] || a.upper[j] != m.col_upper[j];
    }
  };
  for (int t = 0; t < 300; ++t) {
    const int mr = 2 + t % 9, n = 2 + (t / 9) % 9;
    std::vector<std::vector<double>> rows(mr, std::vector<double>(n, 0.0));
    for (auto& r : rows)
      for (double& v : r) v = rng() % 3 == 0 ? 0.0 : U(rng) * 5;
    std::vector<double> rl(mr), ru(mr), cl(n), cu(n);
    for (int i = 0; i < mr; ++i) {
      const int k = rng() % 3;
      rl[i] = k == 1 ? -kInf : U(rng) * 4 - 2;
      ru[i] = k == 0 ? kInf : rl[i] == -kInf ? U(rng) * 4 + 1 : rl[i] + 1 + U(rng);
    }
    for (int j = 0; j < n; ++j) {
      const int k = rng() % 4;
      cl[j] = k == 3 ? -kInf : 0.0;
      cu[j] = k == 0 ? 3.0 : kInf;
    }
    compare(test::make_model(std::vector<double>(n, 1.0), rows, rl, ru, cl, cu), "random " + std::to_string(t));
  }
  for (const char* name : {"afiro", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe", "kb2"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(std::string(PS26119_SOURCE_DIR) + "/data/netlib_small/" + name + ".lpm", m).ok);
    compare(m, name);
  }
  EXPECT_GT(nontrivial, 100);  // the comparison actually covers tightened bounds
}
