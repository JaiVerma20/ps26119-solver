// test_presolve.cpp — presolve reductions and postsolve: the postsolved solution must be
// optimal for the ORIGINAL model (independent KKT check), with correct dual signs.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <random>

#include "core/certificates.h"
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

TEST(Presolve, FollowUpSolvesShareTheCallersBudget) {
  // x_i + x_{i+1} >= 2 (cyclic, i < 10) and sum x_i + z <= 7 with z fixed at 2: infeasible (the
  // pair rows need sum x >= 10), not visible to presolve's row tests, and phase 1 needs several
  // pivots. Presolve removes z (Reduced), the reduced solve finds infeasibility and the original
  // is re-solved for a certificate. That re-solve must get only the iterations the reduced solve
  // left: the TOTAL never exceeds the caller's limit (it used to be up to twice the limit).
  const int k = 10;
  std::vector<std::vector<double>> rows(k + 1, std::vector<double>(k + 1, 0.0));
  std::vector<double> rl(k + 1, 2.0), ru(k + 1, kInf);
  for (int i = 0; i < k; ++i) rows[i][i] = rows[i][(i + 1) % k] = 1.0;
  for (int j = 0; j <= k; ++j) rows[k][j] = 1.0;
  rl[k] = -kInf;
  ru[k] = 7.0;
  std::vector<double> c(k + 1, 1.0), cl(k + 1, 0.0), cu(k + 1, kInf);
  c[k] = 0.0;
  cl[k] = cu[k] = 2.0;
  const Model m = make_model(c, rows, rl, ru, cl, cu);
  ASSERT_EQ(presolve(m).outcome, PresolveResult::Outcome::Reduced);
  bool saw_limit = false, saw_certified = false;
  for (std::int64_t n = 0; n <= 40; ++n) {
    Options o;
    o.algorithm = Algorithm::Simplex;
    o.iteration_limit = n;
    const Solution s = solve(m, o);
    EXPECT_LE(s.iterations, n) << "limit " << n << ": " << s.message;
    ASSERT_TRUE(s.status == Status::Infeasible || s.status == Status::IterationLimit) << to_string(s.status);
    if (s.status == Status::Infeasible) {
      EXPECT_EQ(s.check, "PASS") << s.message;
      saw_certified = true;
    } else {
      saw_limit = true;
    }
  }
  EXPECT_TRUE(saw_limit);
  EXPECT_TRUE(saw_certified);
}

TEST(Presolve, FarkasPostsolveMovesSingletonBoundsOntoTheirRows) {
  // r0: x <= 1 (singleton row -> bound x <= 1), r1: x + y + w >= 5 with y in [0, 1] and w fixed
  // at 2. Presolve removes r0 and w; the reduced model x, y in [0, 1], x + y >= 3 is infeasible
  // with r = 1, whose λ_x = -1 points at the bound that r0 created.
  const Model m = make_model({0, 0, 0}, {{1, 0, 0}, {1, 1, 1}}, {-kInf, 5}, {1, kInf}, {0, 0, 2}, {kInf, 1, 2});
  const PresolveResult pr = presolve(m);
  ASSERT_EQ(pr.outcome, PresolveResult::Outcome::Reduced);
  ASSERT_EQ(pr.reduced.num_rows, 1);
  const std::vector<double> ray = postsolve_farkas(m, pr, {1.0});  // reduced: r = 1 on x + y >= 3
  ASSERT_EQ(ray.size(), 2u);
  EXPECT_DOUBLE_EQ(ray[1], 1.0);
  EXPECT_DOUBLE_EQ(ray[0], -1.0);  // λ_x = -1 pointed at the bound x <= 1 created by r0
  const CertificateCheck c = check_infeasibility_certificate(m, ray);
  EXPECT_TRUE(c.passed) << c.detail;
  EXPECT_TRUE(c.rigorous) << c.detail;
  // end to end: no re-solve of the original, certificate verified there
  Options o;
  o.algorithm = Algorithm::Simplex;
  const Solution s = solve(m, o);
  EXPECT_EQ(s.status, Status::Infeasible);
  EXPECT_EQ(s.check, "PASS") << s.message;
  EXPECT_NE(s.message.find("postsolved to the original rows"), std::string::npos) << s.message;
}

TEST(Presolve, FarkasPostsolveKeepsEveryReducedCertificateValid) {
  // Random infeasible LPs with singleton rows and fixed columns: whenever the reduced model's
  // Farkas vector passes on the REDUCED model, its postsolved image must pass on the ORIGINAL.
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  int checked = 0;
  for (int trial = 0; trial < 600; ++trial) {
    const int m = 3 + trial % 4, n = 3 + (trial / 4) % 4;
    std::vector<std::vector<double>> rows(m, std::vector<double>(n, 0.0));
    std::vector<double> rl(m), ru(m), cl(n), cu(n);
    for (int i = 0; i < m; ++i) {
      if (i < 2) {  // singleton rows
        rows[i][rng() % n] = std::round(U(rng) * 16) / 4 + (rng() % 2 ? 0.5 : -0.5);
      } else {
        for (double& v : rows[i]) v = (rng() % 3 == 0) ? 0.0 : std::round(U(rng) * 16) / 4;
      }
      const int kind = rng() % 3;
      const double b = std::round(U(rng) * 12) / 4;
      rl[i] = kind == 1 ? -kInf : b;
      ru[i] = kind == 0 ? kInf : kind == 1 ? b : b + 1;
    }
    for (int j = 0; j < n; ++j) {
      const int kind = rng() % 4;
      cl[j] = kind == 2 ? -kInf : 0.0;
      cu[j] = kind == 0 ? 2.0 : kind == 3 ? 0.0 : kInf;  // kind 3: fixed at 0
    }
    const Model model = make_model(std::vector<double>(n, 0.0), rows, rl, ru, cl, cu);
    const PresolveResult pr = presolve(model);
    if (pr.outcome != PresolveResult::Outcome::Reduced) continue;
    Options o;
    o.algorithm = Algorithm::Simplex;
    o.presolve = false;
    const Solution red = solve(pr.reduced, o);
    if (red.status != Status::Infeasible || red.check != "PASS") continue;
    ++checked;
    const CertificateCheck c = check_infeasibility_certificate(model, postsolve_farkas(model, pr, red.dual_ray));
    EXPECT_TRUE(c.passed) << "trial " << trial << ": " << c.detail;
  }
  EXPECT_GE(checked, 30);
  std::printf("reduced certificates postsolved and re-checked: %d\n", checked);
}
