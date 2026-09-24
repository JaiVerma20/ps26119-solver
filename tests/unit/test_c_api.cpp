// test_c_api.cpp — the C API: solve through it, error codes, enum parity with Status.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "ps26119/ps26119.h"
#include "ps26119/solution.h"

namespace {
// Wyndor (max 3x + 5y; x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18) in CSC.
const double kC[] = {3, 5};
const double kCl[] = {0, 0}, kCu[] = {HUGE_VAL, HUGE_VAL};
const double kRl[] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL}, kRu[] = {4, 12, 18};
const int kStart[] = {0, 2, 4};
const int kIdx[] = {0, 2, 1, 2};
const double kVal[] = {1, 3, 2, 2};
}  // namespace

TEST(CApi, StatusCodesMatchCppEnum) {
  using ps26119::Status;
  EXPECT_EQ(PS26119_OPTIMAL, static_cast<int>(Status::Optimal));
  EXPECT_EQ(PS26119_INFEASIBLE, static_cast<int>(Status::Infeasible));
  EXPECT_EQ(PS26119_UNBOUNDED, static_cast<int>(Status::Unbounded));
  EXPECT_EQ(PS26119_ITERATION_LIMIT, static_cast<int>(Status::IterationLimit));
  EXPECT_EQ(PS26119_TIME_LIMIT, static_cast<int>(Status::TimeLimit));
  EXPECT_EQ(PS26119_NUMERICAL_ERROR, static_cast<int>(Status::NumericalError));
  EXPECT_EQ(PS26119_NOT_SOLVED, static_cast<int>(Status::NotSolved));
}

TEST(CApi, SolvesWyndorWithEveryAlgorithm) {
  for (int alg : {PS26119_ALG_ORACLE, PS26119_ALG_PDLP, PS26119_ALG_R2HPDHG, PS26119_ALG_AUTO}) {
    ps26119_options o;
    ps26119_default_options(&o);
    o.algorithm = alg;
    ps26119_result r;
    double x[2], y[3], z[2];
    const int st = ps26119_solve_lp(3, 2, -1, 0.0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, &o, &r, x, y, z);
    ASSERT_EQ(st, PS26119_OPTIMAL) << "alg " << alg << ": " << r.message;
    EXPECT_NEAR(r.objective, 36, 1e-6);
    EXPECT_NEAR(x[0], 2, 1e-5);
    EXPECT_NEAR(x[1], 6, 1e-5);
    EXPECT_NEAR(y[1], 1.5, 1e-5);
    EXPECT_NEAR(y[2], 1.0, 1e-5);
    EXPECT_GT(std::strlen(r.engine), 0u);
  }
}

TEST(CApi, NullOptionsAndOutputsAreAllowed) {
  ps26119_result r;
  EXPECT_EQ(ps26119_solve_lp(3, 2, -1, 1.0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, nullptr, &r, nullptr, nullptr,
                             nullptr),
            PS26119_OPTIMAL);
  EXPECT_NEAR(r.objective, 37, 1e-6);  // offset included
}

TEST(CApi, InvalidArgumentsNeverCrash) {
  ps26119_result r;
  EXPECT_EQ(ps26119_solve_lp(3, 2, -1, 0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, nullptr, nullptr, nullptr, nullptr,
                             nullptr),
            PS26119_INVALID_ARGUMENT);
  EXPECT_EQ(ps26119_solve_lp(-1, 2, 1, 0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, nullptr, &r, nullptr, nullptr,
                             nullptr),
            PS26119_INVALID_ARGUMENT);
  EXPECT_EQ(ps26119_solve_lp(3, 2, 1, 0, nullptr, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, nullptr, &r, nullptr, nullptr,
                             nullptr),
            PS26119_INVALID_ARGUMENT);
  const int bad_idx[] = {0, 7, 1, 2};  // row index out of range
  EXPECT_EQ(ps26119_solve_lp(3, 2, 1, 0, kC, kCl, kCu, kRl, kRu, kStart, bad_idx, kVal, nullptr, &r, nullptr, nullptr,
                             nullptr),
            PS26119_INVALID_ARGUMENT);
  EXPECT_NE(std::strstr(r.message, "invalid model"), nullptr);
  ps26119_options o;
  ps26119_default_options(&o);
  o.algorithm = 42;
  EXPECT_EQ(ps26119_solve_lp(3, 2, 1, 0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, &o, &r, nullptr, nullptr, nullptr),
            PS26119_INVALID_ARGUMENT);
}

TEST(CApi, Version) { EXPECT_STRNE(ps26119_version(), ""); }
