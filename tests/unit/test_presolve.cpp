// test_presolve.cpp — presolve reductions and postsolve: the postsolved solution must be
// optimal for the ORIGINAL model (independent KKT check), with correct dual signs.
#include <gtest/gtest.h>

#include <cmath>

#include "core/presolve.h"
#include "io/lpm_reader.h"
#include "kkt_check.h"
#include "model_builder.h"
#include "ps26119/solve.h"

using namespace ps26119;
using test::make_model;

namespace {
std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }

Solution run(const Model& m, Algorithm a, bool pre) {
  Options o;
  o.algorithm = a;
  o.presolve = pre;
  return solve(m, o);
}
}  // namespace

TEST(Presolve, EachReductionAndCascade) {
  // rows: r0 empty (0 ≤ 0 ≤ 1); r1 singleton 2·x0 ≤ 6 (x0 ≤ 3); r2 singleton −x1 ≤ −1 (x1 ≥ 1);
  //       r3 x0 + x1 + x2 ≥ 2;  r4 singleton x3 = 4 (fixes x3 → cascades);  r5 x2 + x3 ≤ 10
  // cols: x0 ∈ [0,10], x1 ∈ [0,10], x2 ∈ [0,5], x3 ∈ [0,10], x4 empty with c > 0 (→ lower 2)
  Model m = make_model({-1, 1, 1, 0, 3},
                       {{0, 0, 0, 0, 0}, {2, 0, 0, 0, 0}, {0, -1, 0, 0, 0}, {1, 1, 1, 0, 0}, {0, 0, 0, 1, 0},
                        {0, 0, 1, 1, 0}},
                       {0, -kInf, -kInf, 2, 4, -kInf}, {1, 6, -1, kInf, 4, 10}, {0, 0, 0, 0, 2},
                       {10, 10, 5, 10, 7});
  PresolveResult r = presolve(m);
  ASSERT_EQ(r.outcome, PresolveResult::Outcome::Reduced);
  EXPECT_GE(r.removed_rows, 4);  // r0, r1, r2, r4
  EXPECT_GE(r.removed_cols, 2);  // x3 (fixed by r4) and x4 (empty)
  const Solution ref = run(m, Algorithm::Oracle, false);
  for (Algorithm a : {Algorithm::Oracle, Algorithm::R2hpdhg}) {
    const Solution s = run(m, a, true);
    ASSERT_EQ(s.status, Status::Optimal) << s.message;
    EXPECT_NEAR(s.objective, ref.objective, 1e-7 * (1 + std::fabs(ref.objective)));
    const auto k = test::kkt(m, s);
    EXPECT_LE(k.primal, 1e-6);
    EXPECT_LE(k.dual, 1e-6);
    EXPECT_LE(k.gap, 1e-6);
    EXPECT_NE(s.message.find("presolve"), std::string::npos);
  }
}

TEST(Presolve, DetectsInfeasibility) {
  // empty row excluding 0
  Model a = make_model({1}, {{0}, {1}}, {1, -kInf}, {2, 5}, {0}, {10});
  EXPECT_EQ(presolve(a).outcome, PresolveResult::Outcome::Infeasible);
  EXPECT_EQ(run(a, Algorithm::R2hpdhg, true).status, Status::Infeasible);
  // singleton row x ≥ 5 against column bound x ≤ 3
  Model b = make_model({1}, {{1}}, {5}, {kInf}, {0}, {3});
  EXPECT_EQ(presolve(b).outcome, PresolveResult::Outcome::Infeasible);
}

TEST(Presolve, DualsMatchWithoutPresolve) {
  // Wyndor with its first two constraints as singleton rows: unique duals (0, 1.5, 1).
  Model m = make_model({3, 5}, {{1, 0}, {0, 2}, {3, 2}}, {-kInf, -kInf, -kInf}, {4, 12, 18}, {0, 0},
                       {kInf, kInf}, -1);
  const Solution s = run(m, Algorithm::Oracle, true);
  ASSERT_EQ(s.status, Status::Optimal) << s.message;
  EXPECT_NEAR(s.objective, 36, 1e-9);
  EXPECT_NEAR(s.y[0], 0.0, 1e-9);
  EXPECT_NEAR(s.y[1], 1.5, 1e-9);  // singleton row 2y ≤ 12 owns the reduced cost of y
  EXPECT_NEAR(s.y[2], 1.0, 1e-9);
  EXPECT_NEAR(s.z[1], 0.0, 1e-9);
}

TEST(Presolve, SmallNetlibSameOptimumAndOriginalKkt) {
  for (const char* name :
       {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe"}) {
    SCOPED_TRACE(name);
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    const Solution ref = run(m, Algorithm::Oracle, false);
    for (Algorithm a : {Algorithm::Oracle, Algorithm::R2hpdhg}) {
      SCOPED_TRACE(to_string(a));
      const Solution s = run(m, a, true);
      ASSERT_EQ(s.status, Status::Optimal) << s.message;
      EXPECT_NEAR(s.objective, ref.objective, 1e-6 * (1 + std::fabs(ref.objective)));
      const auto k = test::kkt(m, s);
      EXPECT_LE(k.primal, 1e-6);
      EXPECT_LE(k.dual, 1e-6);
      EXPECT_LE(k.gap, 1e-6);
    }
  }
}
