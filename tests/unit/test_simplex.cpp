// Tests for the Layer 3 revised primal simplex.
//
// The key test is differential: thousands of random LPs are solved by both
// the simplex and the dense oracle. The status must agree, optimal objectives
// must agree, and every simplex optimum must pass the independent checker.
#include <cstdio>
#include <random>
#include <string>

#include "gpuopt/dense_oracle.hpp"
#include "gpuopt/mps_reader.hpp"
#include "gpuopt/simplex/primal_simplex.hpp"
#include "gpuopt/solution_checker.hpp"
#include "random_lp.hpp"
#include "test_framework.hpp"

using namespace gpuopt;

namespace {

const std::string kData = GPUOPT_DATA_DIR;

void expect_file_optimum(const std::string& path, double expected, const SimplexOptions& opt = {}) {
  const LpProblem lp = read_mps_file(kData + path).problem;
  const SolveResult res = solve_primal_simplex(lp, opt);
  EXPECT_TRUE(res.status == SolveStatus::kOptimal);
  if (res.status != SolveStatus::kOptimal) {
    std::printf("  %s: status %s (%s)\n", path.c_str(), to_string(res.status), res.message.c_str());
    return;
  }
  EXPECT_NEAR(res.objective, expected, 1e-8 * (1.0 + std::fabs(expected)));
  const CheckReport rep = check_solution(lp, res.x, res.row_dual);
  if (!rep.passed()) std::printf("  %s: %s\n", path.c_str(), rep.summary().c_str());
  EXPECT_TRUE(rep.passed());
}

struct DiffStats {
  int optimal = 0, infeasible = 0, unbounded = 0, mismatches = 0;
};

// Solves random LPs with both engines and counts disagreements.
DiffStats differential(int trials, unsigned seed, int max_rows, int max_cols, double density,
                       const SimplexOptions& opt) {
  std::mt19937 rng(seed);
  DiffStats s;
  for (int t = 0; t < trials; ++t) {
    const auto r = testing::random_lp(rng, max_rows, max_cols, density);
    const SolveResult ref = solve_dense_oracle(r.lp);
    const SolveResult got = solve_primal_simplex(r.lp, opt);
    bool ok = got.status == ref.status;
    if (ok && got.status == SolveStatus::kOptimal) {
      ok = std::fabs(got.objective - ref.objective) <= 1e-7 * (1.0 + std::fabs(ref.objective)) &&
           check_solution(r.lp, got.x, got.row_dual).passed();
    }
    if (!ok) {
      ++s.mismatches;
      if (s.mismatches <= 5) {
        std::printf("  trial %d (%dx%d): oracle %s %.10g, simplex %s %.10g  [%s]\n", t, r.lp.num_rows(),
                    r.lp.num_cols(), to_string(ref.status), ref.objective, to_string(got.status),
                    got.objective, got.message.c_str());
      }
    }
    if (ref.status == SolveStatus::kOptimal) ++s.optimal;
    if (ref.status == SolveStatus::kInfeasible) ++s.infeasible;
    if (ref.status == SolveStatus::kUnbounded) ++s.unbounded;
  }
  return s;
}

void expect_agreement(const char* label, int trials, unsigned seed, int max_rows, int max_cols, double density,
                      const SimplexOptions& opt = {}) {
  const DiffStats s = differential(trials, seed, max_rows, max_cols, density, opt);
  std::printf("  %-28s %d LPs: %d optimal, %d infeasible, %d unbounded, %d mismatches\n", label, trials,
              s.optimal, s.infeasible, s.unbounded, s.mismatches);
  EXPECT_EQ(s.mismatches, 0);
}

}  // namespace

TEST(solves_the_example_models) {
  expect_file_optimum("/examples/tiny_max.mps", 11.0);
  expect_file_optimum("/examples/features.mps", -6.5);
  expect_file_optimum("/examples/fixed_format.mps", -180.0);
  expect_file_optimum("/examples/knapsack_mip.mps", 23.5);
  expect_file_optimum("/examples/refinery_blend.mps", 1800.0);
  expect_file_optimum("/examples/beale_cycling.mps", -0.05);
}

TEST(detects_infeasible_and_unbounded) {
  EXPECT_TRUE(solve_primal_simplex(read_mps_file(kData + "/examples/infeasible.mps").problem).status ==
              SolveStatus::kInfeasible);
  EXPECT_TRUE(solve_primal_simplex(read_mps_file(kData + "/examples/unbounded.mps").problem).status ==
              SolveStatus::kUnbounded);
}

TEST(solves_netlib_afiro_and_adlittle) {
  expect_file_optimum("/netlib/afiro.mps", -464.75314285714);
  expect_file_optimum("/netlib/adlittle.mps", 225494.96316);
}

TEST(agrees_with_oracle_on_small_random_lps) {
  expect_agreement("small (<=8 x 10)", 5000, 1, 8, 10, 0.5);
}

TEST(agrees_with_oracle_on_medium_random_lps) {
  expect_agreement("medium (<=40 x 60, sparse)", 400, 2, 40, 60, 0.15);
}

TEST(agrees_with_oracle_under_every_option_combination) {
  SimplexOptions dantzig;
  dantzig.pricing = Pricing::kDantzig;
  expect_agreement("Dantzig pricing", 1000, 3, 10, 12, 0.5, dantzig);
  SimplexOptions raw;
  raw.scale = false;
  raw.perturb = false;
  expect_agreement("no scaling, no perturbation", 1000, 4, 10, 12, 0.5, raw);
}

TEST_MAIN()
