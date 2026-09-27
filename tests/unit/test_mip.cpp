// test_mip.cpp — prototype branch-and-bound: optima equal brute-force enumeration.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <algorithm>

#include "core/certificates.h"
#include "io/lpm_reader.h"
#include "mip/branch_and_bound.h"
#include "model_builder.h"
#include "ps26119/solve.h"

using namespace ps26119;
using test::make_model;

namespace {

// Brute force: all integer points in the box (pure integer models), feasibility checked
// exactly; returns +inf (min) if none feasible. Objective in the model's own sense.
double brute_force_pure(const Model& m, bool& feasible) {
  const int n = m.num_cols;
  std::vector<int> lo(n), hi(n), x(n);
  for (int j = 0; j < n; ++j) lo[j] = static_cast<int>(std::ceil(m.col_lower[j])), hi[j] = static_cast<int>(std::floor(m.col_upper[j]));
  double best = m.sense > 0 ? INFINITY : -INFINITY;
  feasible = false;
  x = lo;
  for (;;) {
    std::vector<double> xd(x.begin(), x.end());
    const auto ax = m.row_activity(xd);
    bool ok = true;
    for (int i = 0; i < m.num_rows && ok; ++i) ok = ax[i] >= m.row_lower[i] - 1e-9 && ax[i] <= m.row_upper[i] + 1e-9;
    if (ok) {
      feasible = true;
      const double v = m.objective_value(xd);
      best = m.sense > 0 ? std::min(best, v) : std::max(best, v);
    }
    int j = 0;
    while (j < n && ++x[j] > hi[j]) x[j] = lo[j], ++j;
    if (j == n) break;
  }
  return best;
}

}  // namespace

TEST(Mip, KnapsackMatchesEnumeration) {
  const std::vector<double> v = {12, 7, 9, 4, 11, 6, 8, 5, 10, 3, 7, 9}, w = {5, 3, 4, 2, 6, 3, 4, 2, 5, 1, 4, 5};
  Model m = make_model(v, {w}, {-kInf}, {20}, std::vector<double>(12, 0.0), std::vector<double>(12, 1.0), -1);
  m.is_integer.assign(12, 1);
  bool feas;
  const double ref = brute_force_pure(m, feas);
  const Solution s = solve(m);
  ASSERT_EQ(s.status, Status::Optimal) << s.message;
  EXPECT_NEAR(s.objective, ref, 1e-9);
  for (double xj : s.x) EXPECT_EQ(xj, std::round(xj));
  EXPECT_NE(s.engine.find("branch-and-bound"), std::string::npos);
}

TEST(Mip, RandomPureIntegerProgramsMatchEnumeration) {
  std::mt19937_64 rng(5);
  std::uniform_int_distribution<int> coef(-4, 5), rhs(0, 12);
  int optimal = 0, infeasible = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const int n = 2 + trial % 5, m = 1 + trial % 4;
    std::vector<double> c(n), cl(n, 0.0), cu(n, 3.0), rl(m), ru(m);
    std::vector<std::vector<double>> rows(m, std::vector<double>(n));
    for (int j = 0; j < n; ++j) c[j] = coef(rng);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) rows[i][j] = coef(rng);
      rl[i] = trial % 3 == 0 ? rhs(rng) - 6.0 : -kInf;
      ru[i] = rhs(rng) + (trial % 7 == 0 ? -15.0 : 0.0);  // some infeasible ones
      if (rl[i] > ru[i]) std::swap(rl[i], ru[i]);
    }
    Model mod = make_model(c, rows, rl, ru, cl, cu, trial % 2 ? 1 : -1);
    mod.is_integer.assign(n, 1);
    bool feas;
    const double ref = brute_force_pure(mod, feas);
    const Solution s = solve(mod);
    if (!feas) {
      EXPECT_EQ(s.status, Status::Infeasible) << "trial " << trial;
      ++infeasible;
      continue;
    }
    ASSERT_EQ(s.status, Status::Optimal) << "trial " << trial << ": " << s.message;
    EXPECT_NEAR(s.objective, ref, 1e-9) << "trial " << trial;
    ++optimal;
  }
  EXPECT_GT(optimal, 30);
  EXPECT_GT(infeasible, 0);
}

TEST(Mip, MixedIntegerMatchesEnumerationOverIntegerPart) {
  // x0, x1 integer in [0,4]; x2, x3 continuous. Reference: enumerate the 25 integer
  // assignments and solve each LP in the continuous part with the oracle.
  std::mt19937_64 rng(9);
  std::uniform_real_distribution<double> U(-3, 3);
  for (int trial = 0; trial < 10; ++trial) {
    std::vector<double> c = {std::round(U(rng) * 3), std::round(U(rng) * 3), U(rng), U(rng)};
    std::vector<std::vector<double>> rows = {{1, 2, 1, 0}, {3, -1, 0, 1}, {U(rng), U(rng), 1, 1}};
    Model m = make_model(c, rows, {-kInf, -2, -5}, {7.5, 8.2, 6}, {0, 0, 0, -1}, {4, 4, 3.5, 2.5});
    m.is_integer = {1, 1, 0, 0};
    double ref = INFINITY;
    for (int a = 0; a <= 4; ++a)
      for (int b = 0; b <= 4; ++b) {
        Model f = m;
        f.is_integer.clear();
        f.col_lower[0] = f.col_upper[0] = a;
        f.col_lower[1] = f.col_upper[1] = b;
        Options o;
        o.algorithm = Algorithm::Oracle;
        const Solution s = solve(f, o);
        if (s.status == Status::Optimal) ref = std::min(ref, s.objective);
      }
    const Solution s = solve(m);
    if (!std::isfinite(ref)) {
      EXPECT_EQ(s.status, Status::Infeasible);
      continue;
    }
    ASSERT_EQ(s.status, Status::Optimal) << s.message;
    EXPECT_NEAR(s.objective, ref, 1e-8 * (1 + std::fabs(ref))) << "trial " << trial;
  }
}

TEST(Mip, RelaxationIsLabelledAndLargeModelsAreRefused) {
  Model m = make_model({-1, -1}, {{2, 2}}, {-kInf}, {3}, {0, 0}, {kInf, kInf});
  m.is_integer = {1, 1};
  const Solution mip = solve(m);
  ASSERT_EQ(mip.status, Status::Optimal);
  EXPECT_NEAR(mip.objective, -1.0, 1e-9);  // integer optimum
  Options o;
  o.relax_integrality = true;
  const Solution lp = solve(m, o);
  EXPECT_NEAR(lp.objective, -1.5, 1e-6);  // LP relaxation
  EXPECT_NE(lp.message.find("LP relaxation"), std::string::npos);
  // size guard of the dense oracle node solver
  mip::BranchAndBoundOptions bo;
  bo.node_solver = mip::NodeSolver::Oracle;
  bo.max_tableau_entries = 1;  // this 1×2 model needs 1·(2+2) = 4 entries
  EXPECT_EQ(mip::solve_branch_and_bound(m, bo).status, Status::NotSolved);
  // the sparse simplex node solver (default) has no dense size limit
  bo.node_solver = mip::NodeSolver::Simplex;
  const Solution sx = mip::solve_branch_and_bound(m, bo);
  EXPECT_EQ(sx.status, Status::Optimal);
  EXPECT_NEAR(sx.objective, -1.0, 1e-9);
}

// Both node solvers against each other on random mixed-integer programs (in addition to the
// enumeration tests above, which run with the default simplex node solver).
TEST(Mip, SimplexAndOracleNodeSolversAgree) {
  std::mt19937_64 rng(77);
  std::uniform_int_distribution<int> coef(-5, 6), rhs(2, 20), kind(0, 2);
  int compared = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const int m = 3 + trial % 4, n = 5 + trial % 5;
    std::vector<std::vector<double>> A(m, std::vector<double>(n));
    for (auto& r : A)
      for (double& v : r) v = coef(rng);
    std::vector<double> c(n), rl(m, -kInf), ru(m), cl(n, 0.0), cu(n, 6.0);
    for (double& v : c) v = coef(rng);
    for (double& v : ru) v = rhs(rng);
    Model mdl = make_model(c, A, rl, ru, cl, cu, trial % 2 ? 1 : -1);
    mdl.is_integer.assign(n, 0);
    for (int j = 0; j < n; ++j) mdl.is_integer[j] = kind(rng) != 0;  // ~2/3 integer
    Options a, b;
    a.engine_params = {{"mip_node_solver", 0}};
    b.engine_params = {{"mip_node_solver", 1}};
    const Solution sa = solve(mdl, a), sb = solve(mdl, b);
    ASSERT_EQ(sa.status, sb.status) << "trial " << trial << ": " << sa.message << " | " << sb.message;
    if (sa.status == Status::Optimal) {
      ++compared;
      EXPECT_NEAR(sa.objective, sb.objective, 1e-7 * (1 + std::fabs(sb.objective))) << "trial " << trial;
      EXPECT_EQ(sa.engine, "branch-and-bound(simplex)");
      EXPECT_EQ(sb.engine, "branch-and-bound(oracle)");
    }
  }
  EXPECT_GT(compared, 30);
  Options bad;
  bad.engine_params = {{"reflection", 1}};
  Model k = make_model({-1, -1}, {{2, 2}}, {-kInf}, {3}, {0, 0}, {5, 5});
  k.is_integer = {1, 1};
  EXPECT_EQ(solve(k, bad).status, Status::NotSolved);
}

// Regression (MIPLIB egout): presolve must round integer bounds inward. Here a singleton
// row gives x0 ≥ 0.5; x0 then becomes an empty column. Without inward rounding presolve
// fixed x0 = 0.5 and reported a better-than-possible "optimum".
TEST(Mip, PresolveRoundsIntegerBoundsInward) {
  // min 10·x0 + x1  s.t.  2·x0 ≥ 1 (singleton),  x1 ≥ 0.3 (singleton);  x0 ∈ Z ∩ [0,5], x1 ∈ [0,5]
  Model m = make_model({10, 1}, {{2, 0}, {0, 1}}, {1, 0.3}, {kInf, kInf}, {0, 0}, {5, 5});
  m.is_integer = {1, 0};
  for (bool pre : {true, false}) {
    Options o;
    o.presolve = pre;
    const Solution s = solve(m, o);
    ASSERT_EQ(s.status, Status::Optimal) << s.message;
    EXPECT_NEAR(s.objective, 10.3, 1e-9) << (pre ? "presolve" : "no presolve");
    EXPECT_EQ(s.x[0], 1.0);
  }
  // integer column with no integer value in its (tightened) bounds → infeasible
  Model inf = make_model({1}, {{4}}, {1}, {3}, {0}, {1});  // 0.25 ≤ x ≤ 0.75
  inf.is_integer = {1};
  EXPECT_EQ(solve(inf).status, Status::Infeasible);
}

TEST(Mip, PseudocostBranchingSolvesGt2WithinANodeBudget) {
  // MIPLIB 3 gt2 (29 x 188, all integer). With most-fractional branching it was not solved in
  // 190k nodes / 60 s; pseudocost branching proves the optimum 21166 (HiGHS) in a few thousand.
  Model m;
  ASSERT_TRUE(io::read_lpm(std::string(PS26119_SOURCE_DIR) + "/data/mip_small/gt2.lpm", m).ok);
  Options o;
  o.iteration_limit = 20000;  // node limit for branch-and-bound
  const Solution s = solve(m, o);
  ASSERT_EQ(s.status, Status::Optimal) << s.message;
  EXPECT_NEAR(s.objective, 21166.0, 1e-6 * 21166.0);
  EXPECT_LT(s.iterations, 20000);
}

TEST(Mip, DivingFindsAnIncumbentWhereRoundingFails) {
  // max x1 + x2 + x3, 2(x1 + x2 + x3) <= 5, x binary. Root LP: two ones and a 0.5; rounding it
  // gives (1,1,1), infeasible. The dive fixes the integral columns, tries x3 = 1 (infeasible),
  // flips to x3 = 0 and finds the optimum 2 — within a budget of ONE branch-and-bound node.
  Model m = make_model({1, 1, 1}, {{2, 2, 2}}, {-kInf}, {5}, {0, 0, 0}, {1, 1, 1});
  m.sense = -1;
  m.is_integer = {1, 1, 1};
  Options o;
  o.iteration_limit = 1;
  o.presolve = false;
  const Solution s = solve(m, o);
  ASSERT_EQ(s.x.size(), 3u) << to_string(s.status) << ": " << s.message;
  EXPECT_NEAR(s.objective, 2.0, 1e-9);
  EXPECT_NE(s.message.find("dives"), std::string::npos) << s.message;
}

TEST(Mip, UnboundedRelaxationIsDecidedNotAssumed) {
  // An unbounded LP relaxation means "unbounded OR infeasible". It used to be reported as
  // Unbounded unconditionally.
  // (a) unbounded: min -x, x - y <= 0.5, x, y integer >= 0 -> integer point + ray, certified.
  Model u = make_model({-1, 0}, {{1, -1}}, {-kInf}, {0.5}, {0, 0}, {kInf, kInf});
  u.is_integer = {1, 1};
  Options o;
  const Solution su = solve(u, o);
  ASSERT_EQ(su.status, Status::Unbounded) << su.message;
  EXPECT_EQ(su.check, "PASS") << su.message;
  ASSERT_EQ(su.x.size(), 2u);
  for (double v : su.x) EXPECT_EQ(v, std::round(v));
  // (a') the same through presolve: a fixed column w (in the row) and a singleton row y >= 1
  Model up = make_model({-1, 0, 0}, {{1, -1, 1}, {0, 1, 0}}, {-kInf, 1}, {2.5, kInf}, {0, 0, 2}, {kInf, kInf, 2});
  up.is_integer = {1, 1, 1};
  const Solution sp = solve(up, o);
  ASSERT_EQ(sp.status, Status::Unbounded) << sp.message;
  ASSERT_EQ(sp.primal_ray.size(), 3u);  // in the ORIGINAL space
  EXPECT_EQ(sp.primal_ray[2], 0.0);
  EXPECT_TRUE(check_unboundedness_certificate(up, sp.x, sp.primal_ray).passed);
  EXPECT_NE(sp.message.find("postsolved to the original model"), std::string::npos) << sp.message;
  // (b) infeasible although the relaxation is unbounded: 2y + 2z = 1 has no integer solution,
  //     x (integer, free objective direction) makes the relaxation unbounded.
  Model inf = make_model({-1, 0, 0}, {{0, 2, 2}}, {1}, {1}, {0, 0, 0}, {kInf, 5, 5});
  inf.is_integer = {1, 1, 1};
  const Solution si = solve(inf, o);
  EXPECT_EQ(si.status, Status::Infeasible) << si.message;
}
