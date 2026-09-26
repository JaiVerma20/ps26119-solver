// test_oracle.cpp — the dense double-double simplex oracle: 12 hand-written LPs with
// known answers, randomized LPs checked by an independent KKT certificate, and the small
// Netlib set (afiro = −464.75314286) through the .lpm bridge.
#include <gtest/gtest.h>

#include <random>
#include <cmath>

#include "io/lpm_reader.h"
#include "kkt_check.h"
#include "model_builder.h"
#include "oracle/dense_simplex.h"
#include "ps26119/solve.h"

using namespace ps26119;
using test::make_model;

namespace {

Solution oracle_solve(const Model& m) {
  Options o;
  o.algorithm = Algorithm::Oracle;
  return solve(m, o);
}

void expect_optimal(const Model& m, const Solution& s, double obj, double tol = 1e-9) {
  ASSERT_EQ(s.status, Status::Optimal) << s.message;
  EXPECT_NEAR(s.objective, obj, tol * (1 + std::fabs(obj)));
  const auto k = test::kkt(m, s);
  EXPECT_LT(k.primal, 1e-12);
  EXPECT_LT(k.dual, 1e-12);
  EXPECT_LT(k.gap, 1e-12);
}

std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }

}  // namespace

// 1. Textbook (Wyndor Glass, Hillier & Lieberman): max 3x + 5y, x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18.
TEST(Oracle, WyndorMaxWithShadowPrices) {
  Model m = make_model({3, 5}, {{1, 0}, {0, 2}, {3, 2}}, {-kInf, -kInf, -kInf}, {4, 12, 18}, {0, 0},
                       {kInf, kInf}, -1);
  auto s = oracle_solve(m);
  expect_optimal(m, s, 36);
  EXPECT_NEAR(s.x[0], 2, 1e-12);
  EXPECT_NEAR(s.x[1], 6, 1e-12);
  // shadow prices in the max sense (HiGHS convention): 0, 1.5, 1
  EXPECT_NEAR(s.y[0], 0, 1e-12);
  EXPECT_NEAR(s.y[1], 1.5, 1e-12);
  EXPECT_NEAR(s.y[2], 1, 1e-12);
}

// 2. Infeasible: x + y ≤ 1 and x + y ≥ 3.
TEST(Oracle, Infeasible) {
  Model m = make_model({1, 1}, {{1, 1}, {1, 1}}, {-kInf, 3}, {1, kInf}, {0, 0}, {kInf, kInf});
  EXPECT_EQ(oracle_solve(m).status, Status::Infeasible);
}

// 3. Infeasible through column bounds: x ≥ 5 but row x ≤ 3.
TEST(Oracle, InfeasibleByBounds) {
  Model m = make_model({1}, {{1}}, {-kInf}, {3}, {5}, {kInf});
  EXPECT_EQ(oracle_solve(m).status, Status::Infeasible);
}

// 4. Unbounded: min −x s.t. x − y ≤ 1, x, y ≥ 0.
TEST(Oracle, Unbounded) {
  Model m = make_model({-1, 0}, {{1, -1}}, {-kInf}, {1}, {0, 0}, {kInf, kInf});
  EXPECT_EQ(oracle_solve(m).status, Status::Unbounded);
}

// 5. Unbounded with free variables: min x s.t. x − y = 0, x, y free.
TEST(Oracle, UnboundedFree) {
  Model m = make_model({1, 0}, {{1, -1}}, {0}, {0}, {-kInf, -kInf}, {kInf, kInf});
  EXPECT_EQ(oracle_solve(m).status, Status::Unbounded);
}

// 6. Degenerate: Beale's example, which cycles under Dantzig's rule without anti-cycling.
//    min −3/4 x1 + 20 x2 − 1/2 x3 + 6 x4
//    s.t. 1/4 x1 − 8 x2 − x3 + 9 x4 ≤ 0;  1/2 x1 − 12 x2 − 1/2 x3 + 3 x4 ≤ 0;  x3 ≤ 1;  x ≥ 0
//    optimum −5/4 at x = (1, 0, 1, 0).
TEST(Oracle, BealeCyclingExample) {
  Model m = make_model({-0.75, 20, -0.5, 6},
                       {{0.25, -8, -1, 9}, {0.5, -12, -0.5, 3}, {0, 0, 1, 0}},
                       {-kInf, -kInf, -kInf}, {0, 0, 1}, {0, 0, 0, 0}, {kInf, kInf, kInf, kInf});
  auto s = oracle_solve(m);
  expect_optimal(m, s, -1.25);
}

// 7. Free variables: min t s.t. t ≥ x − 3, t ≥ 3 − x, x and t free → 0 at x = 3.
TEST(Oracle, FreeVariablesAbsoluteValue) {
  Model m = make_model({0, 1}, {{-1, 1}, {1, 1}}, {-3, 3}, {kInf, kInf}, {-kInf, -kInf}, {kInf, kInf});
  auto s = oracle_solve(m);
  expect_optimal(m, s, 0);
  EXPECT_NEAR(s.x[0], 3, 1e-12);
}

// 8. Equalities: min x1 + 2x2 + 3x3 s.t. x1 + x2 + x3 = 6, x1 − x2 = 0 → 9 at (3, 3, 0).
TEST(Oracle, Equalities) {
  Model m = make_model({1, 2, 3}, {{1, 1, 1}, {1, -1, 0}}, {6, 0}, {6, 0}, {0, 0, 0}, {kInf, kInf, kInf});
  auto s = oracle_solve(m);
  expect_optimal(m, s, 9);
  EXPECT_NEAR(s.x[2], 0, 1e-12);
}

// 9. Ranged row + upper bound: min −x − y s.t. 1 ≤ x + 2y ≤ 4, 0 ≤ x ≤ 3, y ≥ 0 → −3.5.
TEST(Oracle, RangedRow) {
  Model m = make_model({-1, -1}, {{1, 2}}, {1}, {4}, {0, 0}, {3, kInf});
  auto s = oracle_solve(m);
  expect_optimal(m, s, -3.5);
  EXPECT_NEAR(s.x[0], 3, 1e-12);
  EXPECT_NEAR(s.x[1], 0.5, 1e-12);
}

// 10. Negative bounds, upper-bounded column with lower −inf:
//     min x s.t. x + y ≥ −5, −10 ≤ x ≤ 10, y ≤ 2 → −7 at (−7, 2).
TEST(Oracle, NegativeAndUpperOnlyBounds) {
  Model m = make_model({1, 0}, {{1, 1}}, {-5}, {kInf}, {-10, -kInf}, {10, 2});
  auto s = oracle_solve(m);
  expect_optimal(m, s, -7);
  EXPECT_NEAR(s.x[1], 2, 1e-12);
}

// 11. Redundant equalities (rank-deficient A): an artificial stays basic at zero.
//     min x − y s.t. x + y = 2, 2x + 2y = 4, x, y ≥ 0 → −2.
TEST(Oracle, RedundantEqualities) {
  Model m = make_model({1, -1}, {{1, 1}, {2, 2}}, {2, 4}, {2, 4}, {0, 0}, {kInf, kInf});
  auto s = oracle_solve(m);
  expect_optimal(m, s, -2);
}

// 12. No rows at all, and a fixed column: min x − y + 2w, 0 ≤ x ≤ 1, 0 ≤ y ≤ 2, w = 1.5 → 1.
TEST(Oracle, NoRowsAndFixedColumn) {
  Model m = make_model({1, -1, 2}, {}, {}, {}, {0, 0, 1.5}, {1, 2, 1.5});
  auto s = oracle_solve(m);
  expect_optimal(m, s, 1);
}

// Max sense + RANGES on an E row + FR column + objective offset (data/hand/max_ranged.mps,
// converted by the highspy bridge). Optimum 21.5 (HiGHS).
TEST(Oracle, MaxRangedFromBridge) {
  Model m;
  auto r = io::read_lpm(data("hand/max_ranged.lpm"), m);
  ASSERT_TRUE(r.ok) << r.error;
  auto s = oracle_solve(m);
  expect_optimal(m, s, 21.5);
  EXPECT_NEAR(s.dual_objective, 21.5, 1e-9);
}

// Randomized: feasible LPs built around a random point, bounded boxes. The oracle must
// return a KKT certificate (checked independently here).
TEST(Oracle, RandomLpsSatisfyKkt) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> U(-1, 1);
  std::uniform_int_distribution<int> kind(0, 3);
  int solved = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const int m = 1 + trial % 7, n = 1 + (trial / 7) % 9;
    std::vector<std::vector<double>> rows(m, std::vector<double>(n, 0.0));
    std::vector<double> x0(n), c(n), cl(n), cu(n), rl(m), ru(m);
    for (int j = 0; j < n; ++j) {
      x0[j] = 3 * U(rng);
      c[j] = std::round(10 * U(rng));
      cl[j] = kind(rng) == 0 ? -kInf : x0[j] - 1 - std::fabs(U(rng));
      cu[j] = kind(rng) == 0 ? kInf : x0[j] + 1 + std::fabs(U(rng));
      if (kind(rng) == 0) cl[j] = cu[j] = x0[j];  // some fixed at the construction point
    }
    for (int i = 0; i < m; ++i) {
      double act = 0;
      for (int j = 0; j < n; ++j) {
        if (U(rng) > 0.2) rows[i][j] = std::round(5 * U(rng));
        act += rows[i][j] * x0[j];
      }
      switch (kind(rng)) {
        case 0: rl[i] = ru[i] = act; break;                                       // equality
        case 1: rl[i] = -kInf; ru[i] = act + std::fabs(U(rng)); break;            // ≤
        case 2: rl[i] = act - std::fabs(U(rng)); ru[i] = kInf; break;             // ≥
        default: rl[i] = act - std::fabs(U(rng)); ru[i] = act + std::fabs(U(rng));  // ranged
      }
    }
    Model mod = make_model(c, rows, rl, ru, cl, cu, trial % 2 ? 1 : -1);
    auto s = oracle_solve(mod);
    ASSERT_NE(s.status, Status::Infeasible) << "trial " << trial;  // feasible by construction
    ASSERT_NE(s.status, Status::NumericalError) << "trial " << trial << " " << s.message;
    if (s.status == Status::Unbounded) continue;  // possible with infinite bounds
    ASSERT_EQ(s.status, Status::Optimal) << "trial " << trial;
    const auto k = test::kkt(mod, s);
    EXPECT_LT(k.primal, 1e-12) << "trial " << trial;
    EXPECT_LT(k.dual, 1e-12) << "trial " << trial;
    EXPECT_LT(k.gap, 1e-12) << "trial " << trial;
    ++solved;
  }
  EXPECT_GT(solved, 120);
}

// The small Netlib set through the .lpm bridge, against the published optima.
TEST(Oracle, SmallNetlibMatchesPublishedOptima) {
  const std::pair<const char*, double> cases[] = {
      {"afiro", -4.6475314286E+02},  {"sc50a", -6.4575077059E+01},    {"sc50b", -7.0000000000E+01},
      {"kb2", -1.7499001299E+03},    {"adlittle", 2.2549496316E+05},  {"blend", -3.0812149846E+01},
      {"share2b", -4.1573224074E+02}, {"sc105", -5.2202061212E+01},   {"stocfor1", -4.1131976219E+04},
      {"recipe", -2.6661600000E+02}};
  for (const auto& [name, opt] : cases) {
    Model m;
    auto r = io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m);
    ASSERT_TRUE(r.ok) << name << ": " << r.error;
    auto s = oracle_solve(m);
    SCOPED_TRACE(name);
    expect_optimal(m, s, opt, 1e-10);
  }
}
