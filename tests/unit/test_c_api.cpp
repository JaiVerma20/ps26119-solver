// test_c_api.cpp — the C API: solve through it, error codes, enum parity with Status.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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
  for (int alg : {PS26119_ALG_ORACLE, PS26119_ALG_PDLP, PS26119_ALG_R2HPDHG, PS26119_ALG_AUTO, PS26119_ALG_SIMPLEX}) {
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
    EXPECT_GE(r.certified_bound, 36 - 1e-9);  // MAX model: certified upper bound on the optimum
    EXPECT_LT(r.certified_bound, 36 + 1e-4);
    EXPECT_GT(std::strlen(r.engine), 0u);
    EXPECT_EQ(r.check, 1) << "alg " << alg;  // in-process verification PASS
  }
}

TEST(CApi, SolveMpsFile) {
  const std::string afiro = std::string(PS26119_SOURCE_DIR) + "/data/netlib_small/afiro.mps";
  const std::string out = testing::TempDir() + "capi_afiro.sol";
  ps26119_options o;
  ps26119_default_options(&o);
  o.algorithm = PS26119_ALG_SIMPLEX;
  ps26119_result r;
  ASSERT_EQ(ps26119_solve_mps(afiro.c_str(), &o, &r, out.c_str()), PS26119_OPTIMAL) << r.message;
  EXPECT_NEAR(r.objective, -464.75314286, 1e-6);
  EXPECT_EQ(r.check, 1);
  EXPECT_STREQ(r.engine, "simplex");
  FILE* f = std::fopen(out.c_str(), "r");
  ASSERT_NE(f, nullptr);
  std::fclose(f);
  std::remove(out.c_str());
  EXPECT_EQ(ps26119_solve_mps("/no/such/file.mps", nullptr, &r, nullptr), PS26119_INVALID_ARGUMENT);
  EXPECT_NE(std::strstr(r.message, "read error"), nullptr);
  EXPECT_EQ(ps26119_solve_mps(nullptr, nullptr, &r, nullptr), PS26119_INVALID_ARGUMENT);
  EXPECT_EQ(ps26119_solve_mps(afiro.c_str(), nullptr, nullptr, nullptr), PS26119_INVALID_ARGUMENT);
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

TEST(CApi, WarmStartFromOptimum) {
  ps26119_options o;
  ps26119_default_options(&o);
  o.algorithm = PS26119_ALG_R2HPDHG;
  ps26119_result cold, warm;
  double x[2], y[3];
  ASSERT_EQ(ps26119_solve_lp(3, 2, -1, 0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, &o, &cold, x, y, nullptr),
            PS26119_OPTIMAL);
  o.warm_x = x;
  o.warm_y = y;
  ASSERT_EQ(ps26119_solve_lp(3, 2, -1, 0, kC, kCl, kCu, kRl, kRu, kStart, kIdx, kVal, &o, &warm, nullptr, nullptr,
                             nullptr),
            PS26119_OPTIMAL);
  EXPECT_LE(warm.iterations, cold.iterations);
  EXPECT_NEAR(warm.objective, 36, 1e-6);
}

TEST(CApi, SolveLpExReturnsCertificates) {
  // x + y <= 1, x + y >= 3, x, y >= 0: infeasible; the Farkas vector is (-1, 1) up to scale.
  const double inf = HUGE_VAL;
  const double c[2] = {1, 1}, cl[2] = {0, 0}, cu[2] = {inf, inf}, rl[2] = {-inf, 3}, ru[2] = {1, inf};
  const int cs[3] = {0, 2, 4}, ri[4] = {0, 1, 0, 1};
  const double v[4] = {1, 1, 1, 1};
  ps26119_options opt;
  ps26119_default_options(&opt);
  opt.algorithm = PS26119_ALG_SIMPLEX;
  ps26119_result res;
  double ray[2] = {NAN, NAN}, pray[2] = {NAN, NAN};
  ASSERT_EQ(ps26119_solve_lp_ex(2, 2, 1, 0.0, c, cl, cu, rl, ru, cs, ri, v, &opt, &res, nullptr, nullptr, nullptr,
                                ray, pray),
            PS26119_INFEASIBLE);
  EXPECT_EQ(res.check, 1) << res.message;
  EXPECT_LT(ray[0], 0);
  EXPECT_GT(ray[1], 0);
  EXPECT_NEAR(ray[0], -ray[1], 1e-12 * std::fabs(ray[1]));
  EXPECT_TRUE(std::isnan(pray[0]));  // no primal ray for an Infeasible verdict
  // min -x, x - y <= 0: unbounded along a ray with d_x > 0, starting from a feasible x
  const double c2[2] = {-1, 0}, rl2[1] = {-inf}, ru2[1] = {0};
  const int cs2[3] = {0, 1, 2}, ri2[2] = {0, 0};
  const double v2[2] = {1, -1};
  double x[2] = {NAN, NAN};
  ASSERT_EQ(ps26119_solve_lp_ex(1, 2, 1, 0.0, c2, cl, cu, rl2, ru2, cs2, ri2, v2, &opt, &res, x, nullptr, nullptr,
                                nullptr, pray),
            PS26119_UNBOUNDED);
  EXPECT_EQ(res.check, 1) << res.message;
  EXPECT_GT(pray[0], 0);
  EXPECT_FALSE(std::isnan(x[0]));
}

TEST(CApi, InvalidOptionsAreRejectedNotIgnored) {
  // negative / NaN limits used to be replaced silently by the defaults (a 1-hour time limit)
  const double inf = HUGE_VAL;
  const double c[1] = {1}, cl[1] = {0}, cu[1] = {inf}, rl[1] = {1}, ru[1] = {inf};
  const int cs[2] = {0, 1}, ri[1] = {0};
  const double v[1] = {1};
  ps26119_result res;
  auto run = [&](const ps26119_options& o) {
    return ps26119_solve_lp(1, 1, 1, 0.0, c, cl, cu, rl, ru, cs, ri, v, &o, &res, nullptr, nullptr, nullptr);
  };
  ps26119_options o;
  ps26119_default_options(&o);
  EXPECT_EQ(run(o), PS26119_OPTIMAL);
  o.time_limit = 0;  // 0 = default: allowed
  o.iteration_limit = 0;
  o.tolerance = 0;
  EXPECT_EQ(run(o), PS26119_OPTIMAL);
  const auto bad = [&](auto mutate, const char* what) {
    ps26119_options b;
    ps26119_default_options(&b);
    mutate(b);
    EXPECT_EQ(run(b), PS26119_INVALID_ARGUMENT) << what;
    EXPECT_GT(std::strlen(res.message), 0u) << what;
  };
  bad([](ps26119_options& b) { b.time_limit = -1; }, "negative time");
  bad([](ps26119_options& b) { b.time_limit = NAN; }, "NaN time");
  bad([](ps26119_options& b) { b.tolerance = -1e-8; }, "negative tolerance");
  bad([](ps26119_options& b) { b.tolerance = NAN; }, "NaN tolerance");
  bad([](ps26119_options& b) { b.iteration_limit = -5; }, "negative iterations");
  bad([](ps26119_options& b) { b.threads = -3; }, "negative threads");
  bad([](ps26119_options& b) { b.precision = 7; }, "unknown precision");
  bad([](ps26119_options& b) { b.algorithm = 99; }, "unknown algorithm");
  // the MPS entry point validates the same way and survives a NULL path
  ps26119_options b;
  ps26119_default_options(&b);
  b.time_limit = -1;
  EXPECT_EQ(ps26119_solve_mps(PS26119_SOURCE_DIR "/data/netlib_small/afiro.mps", &b, &res, nullptr),
            PS26119_INVALID_ARGUMENT);
  EXPECT_EQ(ps26119_solve_mps(nullptr, nullptr, &res, nullptr), PS26119_INVALID_ARGUMENT);
}

TEST(CApi, NonFiniteWarmStartIsRejected) {
  // an inf/NaN warm start used to reach the iteration and end as NumericalError
  const double inf = HUGE_VAL;
  const double c[1] = {1}, cl[1] = {0}, cu[1] = {inf}, rl[1] = {1}, ru[1] = {inf};
  const int cs[2] = {0, 1}, ri[1] = {0};
  const double v[1] = {1};
  const double wx[1] = {NAN}, wy[1] = {0};
  ps26119_options o;
  ps26119_default_options(&o);
  o.algorithm = PS26119_ALG_R2HPDHG;
  o.warm_x = wx;
  o.warm_y = wy;
  ps26119_result res;
  EXPECT_EQ(ps26119_solve_lp(1, 1, 1, 0.0, c, cl, cu, rl, ru, cs, ri, v, &o, &res, nullptr, nullptr, nullptr),
            PS26119_NOT_SOLVED);
  EXPECT_NE(std::string(res.message).find("non-finite"), std::string::npos) << res.message;
}
