// Tests for the Layer 3 revised primal simplex.
//
// The key test is differential: thousands of random LPs are solved by both
// the simplex and the dense oracle. The status must agree, optimal objectives
// must agree, and every simplex optimum must pass the independent checker.
// Origin: gpuopt tests/test_simplex.cpp (Shivanshu Vats, c192dd0); ported to GoogleTest.
#include <gtest/gtest.h>

#include <cstdio>
#include <random>
#include <string>

#include "oracle/dense_tableau.h"
#include "io/mps_parser.h"
#include "simplex/primal_simplex.h"
#include "core/solution_checker.h"
#include "unit/random_lp.h"

using namespace ps26119;
using namespace ps26119::io;
using namespace ps26119::la;
using namespace ps26119::oracle;
using namespace ps26119::simplex;

namespace {

const std::string kData = std::string(PS26119_SOURCE_DIR) + "/data";

void expect_file_optimum(const std::string& path, double expected, const SimplexOptions& opt = {}) {
  const Model lp = read_mps_file(kData + path).problem;
  const Solution res = solve_primal_simplex(lp, opt);
  EXPECT_TRUE(res.status == Status::Optimal);
  if (res.status != Status::Optimal) {
    std::printf("  %s: status %s (%s)\n", path.c_str(), to_string(res.status), res.message.c_str());
    return;
  }
  EXPECT_NEAR(res.objective, expected, 1e-8 * (1.0 + std::fabs(expected)));
  const CheckReport rep = check_solution(lp, res.x, res.y);
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
    const auto r = ps26119::randlp::random_lp(rng, max_rows, max_cols, density);
    const Solution ref = solve_dense_tableau(r.lp);
    const Solution got = solve_primal_simplex(r.lp, opt);
    bool ok = got.status == ref.status;
    if (ok && got.status == Status::Optimal) {
      ok = std::fabs(got.objective - ref.objective) <= 1e-7 * (1.0 + std::fabs(ref.objective)) &&
           check_solution(r.lp, got.x, got.y).passed();
    }
    if (!ok) {
      ++s.mismatches;
      if (s.mismatches <= 5) {
        std::printf("  trial %d (%dx%d): oracle %s %.10g, simplex %s %.10g  [%s]\n", t, r.lp.num_rows,
                    r.lp.num_cols, to_string(ref.status), ref.objective, to_string(got.status),
                    got.objective, got.message.c_str());
      }
    }
    if (ref.status == Status::Optimal) ++s.optimal;
    if (ref.status == Status::Infeasible) ++s.infeasible;
    if (ref.status == Status::Unbounded) ++s.unbounded;
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

TEST(Simplex, solves_the_example_models) {
  expect_file_optimum("/examples/tiny_max.mps", 11.0);
  expect_file_optimum("/examples/features.mps", -6.5);
  expect_file_optimum("/examples/fixed_format.mps", -180.0);
  expect_file_optimum("/examples/knapsack_mip.mps", 23.5);
  expect_file_optimum("/examples/refinery_blend.mps", 1800.0);
  expect_file_optimum("/examples/beale_cycling.mps", -0.05);
}

TEST(Simplex, detects_infeasible_and_unbounded) {
  EXPECT_TRUE(solve_primal_simplex(read_mps_file(kData + "/examples/infeasible.mps").problem).status ==
              Status::Infeasible);
  EXPECT_TRUE(solve_primal_simplex(read_mps_file(kData + "/examples/unbounded.mps").problem).status ==
              Status::Unbounded);
}

TEST(Simplex, solves_netlib_afiro_and_adlittle) {
  expect_file_optimum("/netlib_small/afiro.mps", -464.75314285714);
  expect_file_optimum("/netlib_small/adlittle.mps", 225494.96316);
}

TEST(Simplex, agrees_with_oracle_on_small_random_lps) {
  expect_agreement("small (<=8 x 10)", 5000, 1, 8, 10, 0.5);
}

TEST(Simplex, agrees_with_oracle_on_medium_random_lps) {
  expect_agreement("medium (<=40 x 60, sparse)", 400, 2, 40, 60, 0.15);
}

TEST(Simplex, agrees_with_oracle_under_every_option_combination) {
  SimplexOptions dantzig;
  dantzig.pricing = Pricing::kDantzig;
  expect_agreement("Dantzig pricing", 1000, 3, 10, 12, 0.5, dantzig);
  SimplexOptions raw;
  raw.scale = false;
  raw.perturb = false;
  expect_agreement("no scaling, no perturbation", 1000, 4, 10, 12, 0.5, raw);
}


// Integration audit finding S1: whatever happens with an unreachable tolerance (a limit,
// NumericalError), the result is never an Optimal whose unscaled violations exceed it.
TEST(Simplex, never_optimal_above_the_requested_tolerance) {
  const Model lp = read_mps_file(kData + "/netlib_small/adlittle.mps").problem;
  for (double tol : {1e-13, 1e-15, 1e-300}) {
    SimplexOptions opt;
    opt.primal_tolerance = tol;
    opt.dual_tolerance = tol;
    opt.max_iterations = 20000;
    opt.time_limit_seconds = 5.0;
    const Solution res = solve_primal_simplex(lp, opt);
    if (res.status == Status::Optimal) {
      const CheckReport rep = check_solution(lp, res.x, res.y, {tol, tol, 1.0});
      EXPECT_TRUE(rep.primal_ok && rep.dual_ok) << "tol " << tol << ": " << rep.summary();
    }
  }
}

// Audit finding S2: the time limit is reported as TimeLimit, not IterationLimit.
TEST(Simplex, time_limit_is_reported_as_time_limit) {
  const Model lp = read_mps_file(kData + "/netlib_small/adlittle.mps").problem;
  SimplexOptions opt;
  opt.time_limit_seconds = 0.0;
  const Solution res = solve_primal_simplex(lp, opt);
  EXPECT_EQ(res.status, Status::TimeLimit) << res.message;
}
