// Tests for the dense reference oracle and the solution checker.
// Every expected optimum below was derived by hand and cross-checked with an
// independent solver (SciPy/HiGHS, used only as a comparison baseline).
// Origin: gpuopt tests/test_dense_oracle.cpp (Shivanshu Vats, c192dd0); ported to GoogleTest.
#include <gtest/gtest.h>

#include <string>

#include "oracle/dense_tableau.h"
#include "io/mps_parser.h"
#include "core/solution_checker.h"
#include "unit/random_lp.h"

using namespace ps26119;
using namespace ps26119::io;
using namespace ps26119::la;
using namespace ps26119::oracle;

namespace {

const std::string kData = std::string(PS26119_SOURCE_DIR) + "/data";

Solution solve_file(const std::string& file, Model* out = nullptr) {
  const auto r = read_mps_file(kData + "/examples/" + file);
  if (out) *out = r.problem;
  return solve_dense_tableau(r.problem);
}

// Solves, then insists the independent checker certifies the optimum.
void expect_certified_optimum(const std::string& file, double expected) {
  Model lp;
  const Solution res = solve_file(file, &lp);
  EXPECT_TRUE(res.status == Status::Optimal);
  if (res.status != Status::Optimal) return;
  EXPECT_NEAR(res.objective, expected, 1e-9 * (1.0 + std::fabs(expected)));
  const CheckReport rep = check_solution(lp, res.x, res.y);
  if (!rep.passed()) std::printf("  %s: %s\n", file.c_str(), rep.summary().c_str());
  EXPECT_TRUE(rep.passed());
}

// Builds a dense-input LP for quick hand-written cases.
Model make_lp(int m, int n, const std::vector<double>& dense_rowmajor) {
  Model lp;
  std::vector<Triplet> t;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const double v = dense_rowmajor[static_cast<size_t>(i) * n + j];
      if (v != 0.0) t.push_back({i, j, v});
    }
  }
  ps26119::testing::set_matrix(lp, build_csc(m, n, t));
  lp.obj.assign(n, 0.0);
  lp.col_lower.assign(n, 0.0);
  lp.col_upper.assign(n, kInf);
  lp.row_lower.assign(m, -kInf);
  lp.row_upper.assign(m, kInf);
  lp.is_integer.assign(n, 0);
  for (int i = 0; i < m; ++i) lp.row_names.push_back("r" + std::to_string(i));
  for (int j = 0; j < n; ++j) lp.col_names.push_back("c" + std::to_string(j));
  return lp;
}

}  // namespace

TEST(DenseTableau, solves_maximisation_example) {
  expect_certified_optimum("tiny_max.mps", 11.0);
  const Solution res = solve_file("tiny_max.mps");
  EXPECT_NEAR(res.x[0], 3.0, 1e-9);
  EXPECT_NEAR(res.x[1], 1.0, 1e-9);
}

TEST(DenseTableau, solves_model_with_ranges_free_fixed_and_negative_bounds) {
  expect_certified_optimum("features.mps", -6.5);
}

TEST(DenseTableau, terminates_on_beale_cycling_example) { expect_certified_optimum("beale_cycling.mps", -0.05); }

TEST(DenseTableau, solves_fixed_format_model) { expect_certified_optimum("fixed_format.mps", -180.0); }

TEST(DenseTableau, solves_lp_relaxation_of_mip) {
  expect_certified_optimum("knapsack_mip.mps", 23.5);
  EXPECT_TRUE(!solve_file("knapsack_mip.mps").message.empty());  // says integrality ignored
}

TEST(DenseTableau, detects_infeasibility) {
  EXPECT_TRUE(solve_file("infeasible.mps").status == Status::Infeasible);
}

TEST(DenseTableau, detects_unboundedness) {
  EXPECT_TRUE(solve_file("unbounded.mps").status == Status::Unbounded);
}

TEST(DenseTableau, contradictory_bounds_are_infeasible_without_pivoting) {
  Model lp = make_lp(1, 1, {1.0});
  lp.col_lower[0] = 3.0;
  lp.col_upper[0] = 2.0;
  const Solution res = solve_dense_tableau(lp);
  EXPECT_TRUE(res.status == Status::Infeasible);
  EXPECT_EQ(res.iterations, 0);
}

TEST(DenseTableau, handles_redundant_equality_rows) {
  // x + y = 2 written twice, 2x + 2y = 4 once more; min x + 2y  ->  x = 2, obj 2.
  Model lp = make_lp(3, 2, {1, 1, 1, 1, 2, 2});
  lp.obj = {1.0, 2.0};
  lp.row_lower = lp.row_upper = {2.0, 2.0, 4.0};
  const Solution res = solve_dense_tableau(lp);
  EXPECT_TRUE(res.status == Status::Optimal);
  EXPECT_NEAR(res.objective, 2.0, 1e-12);
  EXPECT_TRUE(check_solution(lp, res.x, res.y).passed());
}

TEST(DenseTableau, handles_free_variables_and_empty_constraints) {
  // No rows at all: min x - y with x in [1, 5], y in [-3, 4]  ->  1 - 4 = -3.
  Model lp = make_lp(0, 2, {});
  lp.obj = {1.0, -1.0};
  lp.col_lower = {1.0, -3.0};
  lp.col_upper = {5.0, 4.0};
  Solution res = solve_dense_tableau(lp);
  EXPECT_TRUE(res.status == Status::Optimal);
  EXPECT_NEAR(res.objective, -3.0, 1e-12);

  // Free variable pinned by an equality: min y  s.t.  y - x = 0,  x >= -7,  y free  ->  -7.
  Model lp2 = make_lp(1, 2, {-1.0, 1.0});
  lp2.obj = {0.0, 1.0};
  lp2.col_lower = {-7.0, -kInf};
  lp2.row_lower = lp2.row_upper = {0.0};
  res = solve_dense_tableau(lp2);
  EXPECT_TRUE(res.status == Status::Optimal);
  EXPECT_NEAR(res.objective, -7.0, 1e-12);
  EXPECT_TRUE(check_solution(lp2, res.x, res.y).passed());
}

TEST(DenseTableau, checker_rejects_a_wrong_answer) {
  Model lp;
  const Solution res = solve_file("tiny_max.mps", &lp);
  // Feasible but sub-optimal point x = (2, 1) with the optimal duals: gap must show.
  EXPECT_TRUE(!check_solution(lp, {2.0, 1.0}, res.y).passed());
  // Infeasible point.
  EXPECT_TRUE(!check_solution(lp, {4.0, 4.0}, res.y).primal_ok);
  // Wrong-sign duals.
  EXPECT_TRUE(!check_solution(lp, res.x, {-1.0, -1.0}).dual_ok);
}

TEST(DenseTableau, iteration_limit_is_reported) {
  const auto r = read_mps_file(kData + "/examples/beale_cycling.mps");
  DenseTableauOptions opt;
  opt.max_iterations = 1;
  EXPECT_TRUE(solve_dense_tableau(r.problem, opt).status == Status::IterationLimit);
}

