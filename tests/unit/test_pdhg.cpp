// test_pdhg.cpp — first-order engines (PDLP-style PDHG and r²HPDHG, fp64 and mixed):
// backend algebra, scaling round trips, dual signs against the double-double oracle,
// the small Netlib set, and honest limit statuses.
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

#include "io/lpm_reader.h"
#include "kkt_check.h"
#include "model_builder.h"
#include "pdhg/backend.h"
#include "pdhg/scaling.h"
#include "ps26119/solve.h"

using namespace ps26119;
using test::make_model;

namespace {

std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }

Solution run(const Model& m, Algorithm a, Precision p, double tol = 1e-8) {
  Options o;
  o.algorithm = a;
  o.precision = p;
  o.tolerance = tol;
  o.time_limit = 60;
  return solve(m, o);
}

Solution oracle(const Model& m) {
  Options o;
  o.algorithm = Algorithm::Oracle;
  return solve(m, o);
}

// Wyndor (max) — unique primal and dual solution, so signs of y are pinned down.
Model wyndor() {
  return make_model({3, 5}, {{1, 0}, {0, 2}, {3, 2}}, {-kInf, -kInf, -kInf}, {4, 12, 18}, {0, 0}, {kInf, kInf}, -1);
}

// min with ≥ rows, equality, ranged row and bounded/free columns.
Model mixed_rows() {
  return make_model({2, 3, -1, 0.5}, {{1, 1, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}, {1, -1, 1, 1}},
                    {2, 1, 1, -3}, {kInf, 1, 4, 3}, {0, 0, -1, -kInf}, {10, 10, 2, kInf});
}

}  // namespace

// ------------------------------------------------------------------ backend algebra
TEST(PdhgBackend, StepsMatchDefinitionsInBothPrecisions) {
  Model m = mixed_rows();
  pdhg::ScalingOptions so;
  so.ruiz_iterations = 0;
  so.pock_chambolle = false;
  so.bound_objective_rescaling = false;
  auto sp = pdhg::make_scaled_problem(m, so);  // identity scaling: Ã = A
  for (Precision prec : {Precision::Fp64, Precision::Mixed}) {
    auto b = pdhg::make_cpu_backend();
    b->setup(sp);
    b->set_precision(prec);
    int x = b->create(pdhg::Space::Primal), g = b->create(pdhg::Space::Primal), xh = b->create(pdhg::Space::Primal);
    int y = b->create(pdhg::Space::Dual), ax = b->create(pdhg::Space::Dual), yh = b->create(pdhg::Space::Dual);
    b->upload(x, {1, -2, 5, 0.5});
    b->upload(y, {0.5, -1, 2, 0});
    b->spmv_t(y, g);
    b->primal_step(x, g, 0.1, xh);
    std::vector<double> gv, xv;
    b->download(g, gv);
    b->download(xh, xv);
    // Aᵀy by hand: columns (1,0,1,1), (1,1,0,-1), (0,1,0,1), (0,0,1,1)
    EXPECT_NEAR(gv[0], 0.5 + 2 + 0, 1e-6);
    EXPECT_NEAR(gv[1], 0.5 - 1 - 0, 1e-6);
    const double tol = prec == Precision::Fp64 ? 1e-15 : 1e-6;
    const std::vector<double> xin = {1, -2, 5, 0.5};
    for (int j = 0; j < 4; ++j) {
      const double raw = xin[j] - 0.1 * (m.obj[j] - gv[j]);
      EXPECT_NEAR(xv[j], std::clamp(raw, m.col_lower[j], m.col_upper[j]), tol * 10) << j;
    }
    b->spmv(x, ax);
    b->dual_step(y, ax, 0.7, yh);
    std::vector<double> axv, yv, yin = {0.5, -1, 2, 0};
    b->download(ax, axv);
    b->download(yh, yv);
    for (int i = 0; i < 4; ++i) {
      const double proj = std::clamp(axv[i] - yin[i] / 0.7, m.row_lower[i], m.row_upper[i]);
      EXPECT_NEAR(yv[i], yin[i] - 0.7 * axv[i] + 0.7 * proj, tol * 100) << i;
    }
    // r2h with w = 1, ρ = ½ is exactly one PDHG step (plain Halpern weight 1, no reflection)
    int x0 = b->create(pdhg::Space::Primal), xb = b->create(pdhg::Space::Primal);
    b->upload(x, {1, -2, 5, 0.5});
    b->copy(x0, x);
    b->r2h_primal(x, x0, g, 0.1, 1.0, 0.5, xh, xb);
    std::vector<double> xnew;
    b->download(x, xnew);
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(xnew[j], xv[j], tol * 10);
    // reductions accumulate in fp64
    EXPECT_NEAR(b->dot(x0, x0), 1 + 4 + 25 + 0.25, 1e-12);
  }
}

TEST(PdhgScaling, UnscaleInvertsScale) {
  Model m = mixed_rows();
  auto sp = pdhg::make_scaled_problem(m, {});
  // Ã = R A C entry by entry
  auto A = la::csr_from_model(m);
  for (int i = 0; i < sp.m; ++i)
    for (std::int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
      EXPECT_NEAR(sp.A.val[k], sp.row_scale[i] * A.val[k] * sp.col_scale[A.col[k]], 1e-14);
  // A feasible original x maps to a feasible scaled x̃ and back.
  std::vector<double> x = {1.5, 0.5, 0.5, 1.0}, xs(4), back;
  for (int j = 0; j < 4; ++j) xs[j] = sp.bound_scale * x[j] / sp.col_scale[j];
  sp.unscale_primal(xs, back);
  for (int j = 0; j < 4; ++j) EXPECT_NEAR(back[j], x[j], 1e-14);
  for (int j = 0; j < 4; ++j) {
    EXPECT_GE(xs[j], sp.col_lower[j] - 1e-15);
    EXPECT_LE(xs[j], sp.col_upper[j] + 1e-15);
  }
}

// ------------------------------------------------------------------ against the oracle
class PdhgEngines : public ::testing::TestWithParam<std::tuple<Algorithm, Precision>> {};

TEST_P(PdhgEngines, DualSignsMatchOracleOnWyndor) {
  const auto [alg, prec] = GetParam();
  Model m = wyndor();
  auto ref = oracle(m);
  auto s = run(m, alg, prec);
  ASSERT_EQ(s.status, Status::Optimal) << s.message;
  EXPECT_NEAR(s.objective, 36, 1e-6);
  for (int i = 0; i < m.num_rows; ++i) EXPECT_NEAR(s.y[i], ref.y[i], 1e-5) << "row " << i;
  for (int j = 0; j < m.num_cols; ++j) EXPECT_NEAR(s.z[j], ref.z[j], 1e-5) << "col " << j;
}

TEST_P(PdhgEngines, HandLpsAgreeWithOracle) {
  const auto [alg, prec] = GetParam();
  std::vector<Model> models = {
      mixed_rows(),
      make_model({-1, -1}, {{1, 2}}, {1}, {4}, {0, 0}, {3, kInf}),                        // ranged
      make_model({1, 2, 3}, {{1, 1, 1}, {1, -1, 0}}, {6, 0}, {6, 0}, {0, 0, 0}, {kInf, kInf, kInf}),
      make_model({0, 1}, {{-1, 1}, {1, 1}}, {-3, 3}, {kInf, kInf}, {-kInf, -kInf}, {kInf, kInf}),  // free
      make_model({-0.75, 20, -0.5, 6}, {{0.25, -8, -1, 9}, {0.5, -12, -0.5, 3}, {0, 0, 1, 0}},
                 {-kInf, -kInf, -kInf}, {0, 0, 1}, {0, 0, 0, 0}, {kInf, kInf, kInf, kInf}),     // Beale
      make_model({1, 0}, {{1, 1}}, {-5}, {kInf}, {-10, -kInf}, {10, 2}),
  };
  for (std::size_t t = 0; t < models.size(); ++t) {
    const Model& m = models[t];
    auto ref = oracle(m);
    ASSERT_EQ(ref.status, Status::Optimal);
    auto s = run(m, alg, prec);
    ASSERT_EQ(s.status, Status::Optimal) << "model " << t << ": " << s.message;
    EXPECT_NEAR(s.objective, ref.objective, 1e-6 * (1 + std::fabs(ref.objective))) << "model " << t;
    const auto k = test::kkt(m, s);
    EXPECT_LT(k.primal, 1e-6) << "model " << t;
    EXPECT_LT(k.dual, 1e-6) << "model " << t;
    EXPECT_LT(k.gap, 1e-6) << "model " << t;
  }
}

TEST_P(PdhgEngines, SmallNetlibAgreesWithOracleAndPassesKkt) {
  const auto [alg, prec] = GetParam();
  for (const char* name :
       {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    auto ref = oracle(m);
    auto s = run(m, alg, prec);
    SCOPED_TRACE(name);
    ASSERT_EQ(s.status, Status::Optimal) << s.message;
    EXPECT_NEAR(s.objective, ref.objective, 1e-6 * (1 + std::fabs(ref.objective)));
    const auto k = test::kkt(m, s);  // verifier-grade, per element
    EXPECT_LE(k.primal, 1e-6);
    EXPECT_LE(k.dual, 1e-6);
    EXPECT_LE(k.gap, 1e-6);
    EXPECT_GT(s.iterations_to_fast, 0);
    EXPECT_LE(s.iterations_to_fast, s.iterations);
  }
}

TEST_P(PdhgEngines, IterationLimitIsNotOptimal) {
  const auto [alg, prec] = GetParam();
  Model m;
  ASSERT_TRUE(io::read_lpm(data("netlib_small/share2b.lpm"), m).ok);
  Options o;
  o.algorithm = alg;
  o.precision = prec;
  o.iteration_limit = 128;
  auto s = solve(m, o);
  EXPECT_EQ(s.status, Status::IterationLimit);
  EXPECT_EQ(s.x.size(), static_cast<std::size_t>(m.num_cols));  // best point still reported
}

INSTANTIATE_TEST_SUITE_P(All, PdhgEngines,
                         ::testing::Combine(::testing::Values(Algorithm::Pdlp, Algorithm::R2hpdhg),
                                            ::testing::Values(Precision::Fp64, Precision::Mixed)),
                         [](const auto& info) {
                           return std::string(to_string(std::get<0>(info.param))) + "_" +
                                  to_string(std::get<1>(info.param));
                         });

TEST(PdhgEngines, GpuRequestWithoutCudaIsNotSolved) {
#if !defined(PS26119_HAVE_CUDA)
  Options o;
  o.algorithm = Algorithm::R2hpdhg;
  o.use_gpu = true;
  auto s = solve(wyndor(), o);
  EXPECT_EQ(s.status, Status::NotSolved);
  EXPECT_NE(s.message.find("CUDA"), std::string::npos);
#endif
}

// ------------------------------------------------------------------ infeasibility detection
TEST_P(PdhgEngines, DetectsInfeasibleAndUnbounded) {
  const auto [alg, prec] = GetParam();
  struct Case {
    const char* name;
    Model m;
    Status expect;
  };
  std::vector<Case> cases = {
      {"x+y<=1 & x+y>=3", make_model({1, 1}, {{1, 1}, {1, 1}}, {-kInf, 3}, {1, kInf}, {0, 0}, {kInf, kInf}),
       Status::Infeasible},
      {"x>=5 bound, x<=3 row", make_model({1}, {{1}}, {-kInf}, {3}, {5}, {kInf}), Status::Infeasible},
      {"equalities x+y=1, x+y=2 (free)",
       make_model({1, 2}, {{1, 1}, {1, 1}}, {1, 2}, {1, 2}, {-kInf, -kInf}, {kInf, kInf}), Status::Infeasible},
      {"min -x, x-y<=1", make_model({-1, 0}, {{1, -1}}, {-kInf}, {1}, {0, 0}, {kInf, kInf}), Status::Unbounded},
      {"max x+y, x-y<=2, y<=x+1 ... x+ y unbounded",
       make_model({1, 1}, {{1, -1}, {-1, 1}}, {-kInf, -kInf}, {2, 1}, {0, 0}, {kInf, kInf}, -1), Status::Unbounded},
  };
  for (auto& c : cases) {
    Options o;
    o.algorithm = alg;
    o.precision = prec;
    o.iteration_limit = 200000;
    o.time_limit = 30;
    auto s = solve(c.m, o);
    EXPECT_EQ(s.status, c.expect) << c.name << ": got " << to_string(s.status) << " — " << s.message;
    // the oracle agrees on every case
    EXPECT_EQ(oracle(c.m).status, c.expect) << c.name;
  }
}

TEST_P(PdhgEngines, NoFalseCertificatesOnFeasibleModels) {
  const auto [alg, prec] = GetParam();
  for (const char* name : {"afiro", "kb2", "share2b", "stocfor1"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    auto s = run(m, alg, prec);
    EXPECT_EQ(s.status, Status::Optimal) << name << ": " << s.message;
  }
}
