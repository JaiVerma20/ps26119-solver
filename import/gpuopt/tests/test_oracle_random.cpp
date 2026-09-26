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
#include <random>
#include <string>

#include "gpuopt/dense_oracle.hpp"
#include "gpuopt/solution_checker.hpp"
#include "random_lp.hpp"
#include "test_framework.hpp"

using namespace gpuopt;

using gpuopt::testing::random_lp;
using gpuopt::testing::RandomLp;

TEST(random_lps_are_certified) {
  constexpr int kTrials = 5000;
  std::mt19937 rng(20260924);
  int optimal = 0, infeasible = 0, unbounded = 0, other = 0, certified = 0;

  for (int trial = 0; trial < kTrials; ++trial) {
    const RandomLp r = random_lp(rng);
    const SolveResult res = solve_dense_oracle(r.lp);
    switch (res.status) {
      case SolveStatus::kOptimal: {
        ++optimal;
        const CheckReport rep = check_solution(r.lp, res.x, res.row_dual);
        if (rep.passed()) {
          ++certified;
        } else {
          std::printf("  trial %d not certified: %s\n", trial, rep.summary().c_str());
        }
        break;
      }
      case SolveStatus::kInfeasible:
        ++infeasible;
        if (r.known_feasible) std::printf("  trial %d: feasible LP reported INFEASIBLE\n", trial);
        EXPECT_TRUE(!r.known_feasible);
        break;
      case SolveStatus::kUnbounded:
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

TEST_MAIN()
