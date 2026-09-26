// Randomised stress test of the oracle.
//
// Thousands of small random LPs with every kind of row and bound. For each:
//   * if the LP was constructed around a known feasible point, the oracle
//     must never answer INFEASIBLE;
//   * if every variable is boxed, it must never answer UNBOUNDED;
//   * every OPTIMAL answer must be certified by the independent checker
//     (primal feasible, dual feasible, zero duality gap).
// Later, the same generator drives differential tests of the sparse simplex
// against this oracle.
// Origin: gpuopt tests/test_oracle_random.cpp (Shivanshu Vats, c192dd0); ported to GoogleTest.
#include <gtest/gtest.h>

#include <random>
#include <string>
#include <cstdio>

#include "oracle/dense_tableau.h"
#include "core/solution_checker.h"
#include "unit/random_lp.h"

using namespace ps26119;
using namespace ps26119::oracle;

using ps26119::randlp::random_lp;
using ps26119::randlp::RandomLp;

TEST(DenseTableauRandom, random_lps_are_certified) {
  constexpr int kTrials = 5000;
  std::mt19937 rng(20260924);
  int optimal = 0, infeasible = 0, unbounded = 0, other = 0, certified = 0;

  for (int trial = 0; trial < kTrials; ++trial) {
    const RandomLp r = random_lp(rng);
    const Solution res = solve_dense_tableau(r.lp);
    switch (res.status) {
      case Status::Optimal: {
        ++optimal;
        const CheckReport rep = check_solution(r.lp, res.x, res.y);
        if (rep.passed()) {
          ++certified;
        } else {
          std::printf("  trial %d not certified: %s\n", trial, rep.summary().c_str());
        }
        break;
      }
      case Status::Infeasible:
        ++infeasible;
        if (r.known_feasible) std::printf("  trial %d: feasible LP reported INFEASIBLE\n", trial);
        EXPECT_TRUE(!r.known_feasible);
        break;
      case Status::Unbounded:
        ++unbounded;
        if (r.all_boxed) std::printf("  trial %d: boxed LP reported UNBOUNDED\n", trial);
        EXPECT_TRUE(!r.all_boxed);
        break;
      default:
        ++other;
        std::printf("  trial %d: status %s\n", trial, to_string(res.status));
        break;
    }
  }

  std::printf("  %d random LPs: %d optimal (%d certified), %d infeasible, %d unbounded, %d other\n",
              kTrials, optimal, certified, infeasible, unbounded, other);
  EXPECT_EQ(certified, optimal);
  EXPECT_EQ(other, 0);
  EXPECT_TRUE(optimal > kTrials / 3);  // the generator must produce a meaningful mix
  EXPECT_TRUE(infeasible > 0);
  EXPECT_TRUE(unbounded > 0);
}

