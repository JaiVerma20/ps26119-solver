// test_pdhg.cpp — first-order engines (PDLP-style PDHG and r²HPDHG, fp64 and mixed):
// backend algebra, scaling round trips, dual signs against the double-double oracle,
// the small Netlib set, and honest limit statuses.
#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>

#include "io/lpm_reader.h"
#include "kkt_check.h"
#include "la/parallel.h"
#include "model_builder.h"
#include "pdhg/backend.h"
#include "pdhg/engine.h"
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

// ------------------------------------------------------------------ warm start
TEST_P(PdhgEngines, WarmStartFromOptimumStopsAtFirstCheck) {
  const auto [alg, prec] = GetParam();
  for (const char* name : {"stocfor1", "share2b", "adlittle"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    auto cold = run(m, alg, Precision::Fp64);
    ASSERT_EQ(cold.status, Status::Optimal);
    Options o;
    o.algorithm = alg;
    o.precision = prec;
    o.warm_x = cold.x;
    o.warm_y = cold.y;
    auto warm = solve(m, o);
    ASSERT_EQ(warm.status, Status::Optimal) << name << ": " << warm.message;
    EXPECT_LT(warm.iterations, cold.iterations) << name;
    EXPECT_LE(warm.iterations, 4 * o.termination_check_every) << name;
    EXPECT_NE(warm.message.find("warm start"), std::string::npos);
    EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * (1 + std::fabs(cold.objective)));
  }
}

TEST(PdhgWarmStart, WrongSizeIsNotSolved) {
  Options o;
  o.algorithm = Algorithm::R2hpdhg;
  o.warm_x = {1.0};  // model has 2 columns
  auto s = solve(wyndor(), o);
  EXPECT_EQ(s.status, Status::NotSolved);
  EXPECT_NE(s.message.find("warm start"), std::string::npos);
}

TEST(PdhgWarmStart, PerturbedObjectiveNeedsFewerIterations) {
  // what-if: 1% change in the objective of share2b; warm start from the base solution
  Model base;
  ASSERT_TRUE(io::read_lpm(data("netlib_small/stocfor1.lpm"), base).ok);
  auto s0 = run(base, Algorithm::R2hpdhg, Precision::Fp64);
  ASSERT_EQ(s0.status, Status::Optimal);
  Model changed = base;
  for (std::size_t j = 0; j < changed.obj.size(); j += 3) changed.obj[j] *= 1.01;
  auto cold = run(changed, Algorithm::R2hpdhg, Precision::Fp64);
  Options o;
  o.algorithm = Algorithm::R2hpdhg;
  o.warm_x = s0.x;
  o.warm_y = s0.y;
  auto warm = solve(changed, o);
  ASSERT_EQ(cold.status, Status::Optimal);
  ASSERT_EQ(warm.status, Status::Optimal);
  EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * (1 + std::fabs(cold.objective)));
  EXPECT_LT(warm.iterations, cold.iterations);
}

TEST(PdhgEngineParams, KnownAndUnknownNames) {
  Options o;
  o.algorithm = Algorithm::R2hpdhg;
  o.engine_params = {{"reflection", 0.5}, {"pid_max_log_step", 1e9}};
  auto s = solve(wyndor(), o);
  EXPECT_EQ(s.status, Status::Optimal) << s.message;
  EXPECT_NEAR(s.objective, 36, 1e-6);
  o.engine_params = {{"no_such_knob", 1}};
  s = solve(wyndor(), o);
  EXPECT_EQ(s.status, Status::NotSolved);
  EXPECT_NE(s.message.find("no_such_knob"), std::string::npos);
}

TEST(PdhgScaling, GeometricMeanIsAdaptive) {
  // afiro: entries span ~10^1.4 → not applied by default; forced with threshold 0
  Model m;
  ASSERT_TRUE(io::read_lpm(data("netlib_small/afiro.lpm"), m).ok);
  pdhg::ScalingOptions so;
  auto sp = pdhg::make_scaled_problem(m, so);
  EXPECT_FALSE(sp.geometric_mean_applied);
  EXPECT_NEAR(sp.log10_range, 1.4, 0.1);
  so.geometric_mean_min_log10_range = 0;
  auto sp2 = pdhg::make_scaled_problem(m, so);
  EXPECT_TRUE(sp2.geometric_mean_applied);
  // still a valid scaling: Ã = R A C
  auto A = la::csr_from_model(m);
  for (int i = 0; i < sp2.m; ++i)
    for (std::int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
      EXPECT_NEAR(sp2.A.val[k], sp2.row_scale[i] * A.val[k] * sp2.col_scale[A.col[k]], 1e-12 * (1 + std::fabs(sp2.A.val[k])));
  // and solving with it forced still gives the optimum
  Options o;
  o.algorithm = Algorithm::R2hpdhg;
  o.engine_params = {{"geometric_mean_min_log10_range", 0}};
  auto s = solve(m, o);
  ASSERT_EQ(s.status, Status::Optimal);
  EXPECT_NEAR(s.objective, -464.75314286, 1e-6);
}

// Determinism across thread counts (la/parallel.h): identical iterations and bit-identical
// solutions for 1 and 4 threads, on a model large enough to actually use the pool. Also a
// stress test of the pool's dispatch protocol (many short parallel calls).
TEST(PdhgThreads, BitIdenticalForAnyThreadCount) {
  Model m;
  ASSERT_TRUE(io::read_lpm(data("netlib_small/stocfor1.lpm"), m).ok);
  // replicate stocfor1 block-diagonally so vectors exceed the parallel thresholds
  const int copies = 400;
  Model big;
  big.num_rows = m.num_rows * copies;
  big.num_cols = m.num_cols * copies;
  big.col_start = {0};
  for (int c = 0; c < copies; ++c) {
    for (int j = 0; j < m.num_cols; ++j) {
      big.obj.push_back(m.obj[j] * (1.0 + 0.001 * c));
      big.col_lower.push_back(m.col_lower[j]);
      big.col_upper.push_back(m.col_upper[j]);
      for (int p = m.col_start[j]; p < m.col_start[j + 1]; ++p) {
        big.row_index.push_back(m.row_index[p] + c * m.num_rows);
        big.value.push_back(m.value[p]);
      }
      big.col_start.push_back(static_cast<int>(big.value.size()));
    }
    big.row_lower.insert(big.row_lower.end(), m.row_lower.begin(), m.row_lower.end());
    big.row_upper.insert(big.row_upper.end(), m.row_upper.begin(), m.row_upper.end());
  }
  ASSERT_EQ(big.validate(), "");
  Options o;
  o.algorithm = Algorithm::R2hpdhg;  // explicit: this tests the first-order engine's thread pool
  o.tolerance = 1e-6;
  o.threads = 1;
  auto a = solve(big, o);
  o.threads = 4;
  auto b = solve(big, o);
  ASSERT_EQ(a.status, Status::Optimal) << a.message;
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(a.x, b.x);
  EXPECT_EQ(a.y, b.y);
  EXPECT_EQ(a.objective, b.objective);
}

namespace {
// A large LP with every row and bound kind (sizes above the parallel thresholds of la/parallel.h)
// and a point (x, y) that is neither feasible nor optimal, so every KKT term is non-trivial.
struct BigPoint {
  Model m;
  std::vector<double> x, y;
};
BigPoint big_random_point(int m, int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<int> row(0, m - 1), kind(0, 4);
  BigPoint b;
  Model& M = b.m;
  M.num_rows = m;
  M.num_cols = n;
  M.col_start = {0};
  for (int j = 0; j < n; ++j) {
    M.obj.push_back(u(rng) * 10);
    const int k = kind(rng);
    M.col_lower.push_back(k == 0 || k == 3 ? -kInf : u(rng) - 1);
    M.col_upper.push_back(k == 1 || k == 3 ? kInf : u(rng) + 1);
    for (int e = 0; e < 3; ++e) {
      M.row_index.push_back(row(rng));
      M.value.push_back(u(rng) * 100);
    }
    std::sort(M.row_index.end() - 3, M.row_index.end());
    if (M.row_index.end()[-1] == M.row_index.end()[-2] || M.row_index.end()[-2] == M.row_index.end()[-3]) {
      M.row_index.resize(M.row_index.size() - 3);  // keep the column simple: one entry
      M.value.resize(M.value.size() - 3);
      M.row_index.push_back(row(rng));
      M.value.push_back(u(rng) * 100);
    }
    M.col_start.push_back(static_cast<int>(M.value.size()));
  }
  for (int i = 0; i < m; ++i) {
    const int k = kind(rng);
    const double r = u(rng) * 50;
    M.row_lower.push_back(k == 0 ? -kInf : r);
    M.row_upper.push_back(k == 1 ? kInf : (k == 2 ? r : r + 5));
  }
  for (int j = 0; j < n; ++j) b.x.push_back(std::clamp(u(rng) * 3, M.col_lower[j], M.col_upper[j]));
  for (int i = 0; i < m; ++i) b.y.push_back(u(rng));
  return b;
}
}  // namespace

TEST(PdhgThreads, KktAndRayTestAreThreadInvariantAndMatchSerialDefinition) {
  // kkt_general and ray_test run at every termination check; they are parallel reductions over
  // a FIXED chunking (la::parallel_reduce), so any thread count must give bit-identical stats.
  BigPoint b = big_random_point(90000, 120000, 7);
  ASSERT_EQ(b.m.validate(), "");
  pdhg::ScaledProblem sp = pdhg::make_scaled_problem(b.m, pdhg::ScalingOptions{});
  // y_min sign-feasible, as for a PDHG output
  for (int i = 0; i < b.m.num_rows; ++i) {
    if (!std::isfinite(b.m.row_lower[i])) b.y[i] = std::min(b.y[i], 0.0);
    if (!std::isfinite(b.m.row_upper[i])) b.y[i] = std::max(b.y[i], 0.0);
  }
  la::ThreadPool::instance().set_threads(1);
  const pdhg::KktStats k1 = pdhg::kkt_on_original(sp, b.x, b.y);
  const pdhg::RayTest r1 = pdhg::ray_test(sp, b.x, b.y);
  la::ThreadPool::instance().set_threads(4);
  pdhg::KktWorkspace ws;
  const pdhg::KktStats k4 = pdhg::kkt_on_original(sp, b.x, b.y, &ws);
  const pdhg::KktStats k4b = pdhg::kkt_on_original(sp, b.x, b.y, &ws);  // reused workspace
  const pdhg::RayTest r4 = pdhg::ray_test(sp, b.x, b.y);
  la::ThreadPool::instance().set_threads(1);
  for (const pdhg::KktStats* k : {&k4, &k4b}) {
    EXPECT_EQ(k1.primal_residual, k->primal_residual);
    EXPECT_EQ(k1.dual_residual, k->dual_residual);
    EXPECT_EQ(k1.primal_obj, k->primal_obj);
    EXPECT_EQ(k1.dual_obj, k->dual_obj);
    EXPECT_EQ(k1.b_norm, k->b_norm);
    EXPECT_EQ(k1.c_norm, k->c_norm);
    EXPECT_EQ(k1.primal_max_rel, k->primal_max_rel);
    EXPECT_EQ(k1.dual_max_rel, k->dual_max_rel);
  }
  EXPECT_EQ(r1.dual_ray_objective, r4.dual_ray_objective);
  EXPECT_EQ(r1.dual_ray_violation, r4.dual_ray_violation);
  EXPECT_EQ(r1.primal_ray_objective, r4.primal_ray_objective);
  EXPECT_EQ(r1.primal_ray_violation, r4.primal_ray_violation);

  // Serial textbook evaluation of termination.h (only rounding may differ).
  const Model& M = b.m;
  std::vector<double> ax = M.row_activity(b.x), aty(M.num_cols, 0.0);
  for (int j = 0; j < M.num_cols; ++j)
    for (int p = M.col_start[j]; p < M.col_start[j + 1]; ++p) aty[j] += M.value[p] * b.y[M.row_index[p]];
  double rp2 = 0, rd2 = 0, p = 0, d = 0;
  for (int i = 0; i < M.num_rows; ++i) {
    const double v = std::max(M.row_lower[i] - ax[i], 0.0) + std::max(ax[i] - M.row_upper[i], 0.0);
    rp2 += v * v;
    if (b.y[i] > 0) d += b.y[i] * M.row_lower[i];
    if (b.y[i] < 0) d += b.y[i] * M.row_upper[i];
  }
  for (int j = 0; j < M.num_cols; ++j) {
    const double lam = M.obj[j] - aty[j], lo = M.col_lower[j], up = M.col_upper[j];
    const double z = std::isfinite(lo) && std::isfinite(up) ? lam
                     : std::isfinite(lo)                    ? std::max(lam, 0.0)
                     : std::isfinite(up)                    ? std::min(lam, 0.0)
                                                            : 0.0;
    rd2 += (lam - z) * (lam - z);
    p += M.obj[j] * b.x[j];
    if (z > 0) d += z * lo;
    if (z < 0) d += z * up;
  }
  auto near = [](double a, double e) { return std::fabs(a - e) <= 1e-11 * (1 + std::fabs(e)); };
  EXPECT_TRUE(near(k1.primal_residual, std::sqrt(rp2))) << k1.primal_residual << " vs " << std::sqrt(rp2);
  EXPECT_TRUE(near(k1.dual_residual, std::sqrt(rd2))) << k1.dual_residual << " vs " << std::sqrt(rd2);
  EXPECT_TRUE(near(k1.primal_obj, p)) << k1.primal_obj << " vs " << p;
  EXPECT_TRUE(near(k1.dual_obj, d)) << k1.dual_obj << " vs " << d;
}
