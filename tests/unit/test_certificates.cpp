// test_certificates.cpp — checks of Infeasible / Unbounded claims (core/certificates.h) and
// their use by the gate: good certificates pass, bad or missing ones never certify a claim.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <thread>
#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <cstdio>
#include <vector>

#include "core/certificates.h"
#include "core/gate.h"
#include "io/lpm_reader.h"
#include "pdhg/engine.h"
#include "pdhg/scaling.h"
#include "model_builder.h"
#include "ps26119/solve.h"

using namespace ps26119;
using ps26119::test::make_model;

namespace {
const double kNan = std::numeric_limits<double>::quiet_NaN();
// x + y <= 1, x + y >= 3, x, y >= 0: infeasible; r = (-1, 1) gives L0 = -1 + 3 = 2 > 0.
Model infeasible() { return make_model({1, 1}, {{1, 1}, {1, 1}}, {-kInf, 3}, {1, kInf}, {0, 0}, {kInf, kInf}); }
// min -x s.t. x - y <= 1, x, y >= 0: unbounded along d = (1, 1).
Model unbounded() { return make_model({-1, 0}, {{1, -1}}, {-kInf}, {1}, {0, 0}, {kInf, kInf}); }
}  // namespace

TEST(Certificates, FarkasRigorousPassAndFailures) {
  const Model m = infeasible();
  CertificateCheck c = check_infeasibility_certificate(m, {-1, 1});
  EXPECT_TRUE(c.present);
  EXPECT_TRUE(c.passed);
  EXPECT_TRUE(c.rigorous);
  EXPECT_NEAR(c.measure, 2.0, 1e-12);
  EXPECT_FALSE(check_infeasibility_certificate(m, {1, -1}).passed);  // wrong sign
  EXPECT_FALSE(check_infeasibility_certificate(m, {0, 0}).passed);   // zero ray
  EXPECT_FALSE(check_infeasibility_certificate(m, {kNan, 1}).passed);
  EXPECT_FALSE(check_infeasibility_certificate(m, {1}).present);     // wrong size
  // A FEASIBLE model admits no certificate, whatever r is offered.
  const Model feas = make_model({1, 1}, {{1, 1}, {1, 1}}, {-kInf, 0.5}, {1, kInf}, {0, 0}, {kInf, kInf});
  for (double a : {-1.0, 1.0, 2.0})
    for (double b : {-1.0, 1.0, 3.0}) EXPECT_FALSE(check_infeasibility_certificate(feas, {a, b}).passed);
}

TEST(Certificates, FarkasToleranceStageOnFreeColumns) {
  // x + y = 1, x + y = 2, x, y free: a floating-point ray leaves A^T r = O(1e-16) on free
  // columns, so the rigorous bound is -inf; the tolerance stage decides and says so.
  const Model m = make_model({1, 2}, {{1, 1}, {1, 1}}, {1, 2}, {1, 2}, {-kInf, -kInf}, {kInf, kInf});
  const CertificateCheck exact = check_infeasibility_certificate(m, {-1, 1});
  EXPECT_TRUE(exact.passed);
  EXPECT_TRUE(exact.rigorous);  // exact cancellation: rigorous
  const CertificateCheck noisy = check_infeasibility_certificate(m, {-1, 1 + 1e-15});
  EXPECT_TRUE(noisy.passed);
  EXPECT_FALSE(noisy.rigorous);
  EXPECT_FALSE(check_infeasibility_certificate(m, {-1, 1 + 1e-3}).passed);  // real violation
}

TEST(Certificates, UnboundedRayChecks) {
  const Model m = unbounded();
  EXPECT_TRUE(check_unboundedness_certificate(m, {0, 0}, {1, 1}).passed);
  EXPECT_TRUE(check_unboundedness_certificate(m, {0, 0}, {5, 5}).passed);     // scale-free
  EXPECT_FALSE(check_unboundedness_certificate(m, {2, 0}, {1, 1}).passed);    // infeasible point (row: 2 > 1)
  EXPECT_FALSE(check_unboundedness_certificate(m, {0, 0}, {1, 0}).passed);    // leaves the cone (row grows)
  EXPECT_FALSE(check_unboundedness_certificate(m, {0, 0}, {-1, -1}).passed);  // not improving / leaves bounds
  EXPECT_FALSE(check_unboundedness_certificate(m, {0, 0}, {0, 1}).passed);    // improves nothing
  EXPECT_FALSE(check_unboundedness_certificate(m, {0, 0}, {0, 0}).passed);
  // MAX sense: max x s.t. x - y <= 1 is unbounded along (1, 1) as well
  Model mx = make_model({1, 0}, {{1, -1}}, {-kInf}, {1}, {0, 0}, {kInf, kInf}, -1);
  EXPECT_TRUE(check_unboundedness_certificate(mx, {0, 0}, {1, 1}).passed);
  EXPECT_FALSE(check_unboundedness_certificate(mx, {0, 0}, {-1, -1}).passed);
}

TEST(Certificates, GateCertifiesDemotesOrFlags) {
  const Model m = infeasible();
  Options o;
  o.algorithm = Algorithm::Simplex;
  Solution s;
  s.status = Status::Infeasible;
  s.dual_ray = {-1, 1};
  core::gate(m, o, s);
  EXPECT_EQ(s.status, Status::Infeasible);
  EXPECT_EQ(s.check, "PASS");
  EXPECT_TRUE(std::isnan(s.objective));
  Solution bad;
  bad.status = Status::Infeasible;
  bad.dual_ray = {1, -1};
  core::gate(m, o, bad);
  EXPECT_EQ(bad.status, Status::NumericalError);
  EXPECT_EQ(bad.check, "FAIL");
  Solution none;
  none.status = Status::Infeasible;
  core::gate(m, o, none);
  EXPECT_EQ(none.status, Status::Infeasible);  // kept, but explicitly not certified
  EXPECT_TRUE(none.check.empty());
  EXPECT_NE(none.message.find("not certified"), std::string::npos);
  // a fabricated Unbounded claim on a model that is actually bounded
  const Model bounded = make_model({-1, 0}, {{1, -1}}, {-kInf}, {1}, {0, 0}, {kInf, 3});
  Solution fake;
  fake.status = Status::Unbounded;
  fake.x = {0, 0};
  fake.primal_ray = {1, 1};  // y is capped at 3: d leaves the recession cone
  core::gate(bounded, o, fake);
  EXPECT_EQ(fake.status, Status::NumericalError);
}

// End to end: every LP engine that returns certificates has them verified on these models,
// with and without presolve.
TEST(Certificates, EnginesCertifyTheirVerdicts) {
  for (Algorithm a : {Algorithm::Simplex, Algorithm::R2hpdhg, Algorithm::Pdlp}) {
    for (bool presolve : {false, true}) {
      Options o;
      o.algorithm = a;
      o.presolve = presolve;
      const Solution si = solve(infeasible(), o);
      EXPECT_EQ(si.status, Status::Infeasible) << to_string(a) << " " << si.message;
      EXPECT_EQ(si.check, "PASS") << to_string(a) << " " << si.message;
      EXPECT_EQ(si.dual_ray.size(), 2u);
      const Solution su = solve(unbounded(), o);
      EXPECT_EQ(su.status, Status::Unbounded) << to_string(a) << " " << su.message;
      EXPECT_EQ(su.check, "PASS") << to_string(a) << " " << su.message;
      EXPECT_TRUE(std::isnan(su.objective));
    }
  }
}

TEST(Certificates, SimplexTightensPhaseOneUntilTheFarkasRayPasses) {
  // Phase 1 stops when no reduced cost beats the dual tolerance, so with a loose tolerance the
  // phase-1 duals can leave wrong-sign reduced costs on columns with an infinite bound and the
  // Farkas vector fails the check (seen on d2q06c + objective cut at the default 1e-7). The
  // simplex must then tighten and keep pivoting, never return an uncertified Infeasible. The
  // loose tolerance below forces that path on random infeasible LPs with free / one-sided columns.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  int infeasible_models = 0, tightened = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int m = 3 + trial % 5, n = 3 + (trial / 5) % 5;
    std::vector<std::vector<double>> rows(m, std::vector<double>(n, 0.0));
    for (auto& r : rows)
      for (double& v : r) v = (rng() % 3 == 0) ? 0.0 : std::round(U(rng) * 40) / 8;
    std::vector<double> rl(m), ru(m), cl(n), cu(n), c(n);
    for (int i = 0; i < m; ++i) {
      const int kind = rng() % 3;
      const double b = std::round(U(rng) * 16) / 4;
      rl[i] = kind == 1 ? -kInf : b;
      ru[i] = kind == 0 ? kInf : kind == 1 ? b : b + 1;
    }
    for (int j = 0; j < n; ++j) {
      const int kind = rng() % 3;
      cl[j] = kind == 2 ? -kInf : 0.0;
      cu[j] = kind == 0 ? 2.0 : kInf;
      c[j] = U(rng);
    }
    const Model model = make_model(c, rows, rl, ru, cl, cu);
    Options ref;
    ref.algorithm = Algorithm::Simplex;
    ref.presolve = false;
    if (solve(model, ref).status != Status::Infeasible) continue;
    ++infeasible_models;
    Options loose = ref;
    loose.engine_params = {{"simplex_dual_tolerance", 0.2}};
    const Solution s = solve(model, loose);
    // An uncertified Infeasible is demoted to NumericalError by the gate: that is what the
    // tightening must prevent here.
    EXPECT_NE(s.status, Status::NumericalError) << "trial " << trial << ": " << s.message;
    if (s.status != Status::Infeasible) continue;  // a limit is allowed
    EXPECT_EQ(s.check, "PASS") << "trial " << trial << ": " << s.message;
    if (s.message.find("tolerance tightenings 0") == std::string::npos) ++tightened;
  }
  EXPECT_GE(infeasible_models, 20);
  EXPECT_GE(tightened, 1) << "the loose tolerance never exercised the tightening path";
  std::printf("infeasible models %d, phase-1 tightenings needed on %d\n", infeasible_models, tightened);
}

namespace {
// The model plus one "objective cut" row  sense·cᵀx ≤ sense·(f* − offset) − δ: infeasible by LP
// duality, and its Farkas certificate is essentially the optimal dual — a realistic, "barely"
// infeasible LP (same construction as bench/netlib_infeasible_cut.py).
Model with_objective_cut(const Model& m, double fstar, double delta) {
  Model c = m;
  const int r = m.num_rows;
  c.col_start.assign(1, 0);
  c.row_index.clear();
  c.value.clear();
  for (int j = 0; j < m.num_cols; ++j) {
    for (int k = m.col_start[j]; k < m.col_start[j + 1]; ++k) {
      c.row_index.push_back(m.row_index[k]);
      c.value.push_back(m.value[k]);
    }
    if (m.obj[j] != 0.0) {
      c.row_index.push_back(r);
      c.value.push_back(m.obj[j]);
    }
    c.col_start.push_back(static_cast<int>(c.value.size()));
  }
  if (m.sense > 0) {
    c.row_lower.push_back(-kInf);
    c.row_upper.push_back(fstar - m.obj_offset - delta);
  } else {
    c.row_lower.push_back(fstar - m.obj_offset + delta);
    c.row_upper.push_back(kInf);
  }
  if (!c.row_names.empty()) c.row_names.push_back("OBJCUT");
  c.num_rows = r + 1;
  return c;
}
}  // namespace

TEST(Certificates, FirstOrderEnginesDetectBarelyInfeasibleNetlibCuts) {
  // Each of the 10 small Netlib LPs with an objective cut 1e-4 (1 + |f*|) below its optimum.
  // Detection tests two directions: T(z) − z and the drift z − z0 since the restart anchor.
  // With T(z) − z alone, adlittle + cut took 56M iterations; the budget here is 3M in total.
  for (Algorithm engine : {Algorithm::R2hpdhg, Algorithm::Pdlp}) {
  int certified = 0;
  for (const char* name :
       {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe"}) {
    SCOPED_TRACE(std::string(name) + " " + to_string(engine));
    Model m;
    ASSERT_TRUE(io::read_lpm(std::string(PS26119_SOURCE_DIR) + "/data/netlib_small/" + name + ".lpm", m).ok);
    Options ref;
    ref.algorithm = Algorithm::Simplex;
    const Solution opt = solve(m, ref);
    ASSERT_EQ(opt.status, Status::Optimal);
    const Model cut = with_objective_cut(m, opt.objective, 1e-4 * (1 + std::fabs(opt.objective)));
    Options o;
    o.algorithm = engine;
    o.iteration_limit = 3'000'000;
    const Solution s = solve(cut, o);
    EXPECT_NE(s.status, Status::NumericalError) << s.message;
    EXPECT_NE(s.status, Status::Optimal) << s.message;
    if (s.status == Status::Infeasible) {
      EXPECT_EQ(s.check, "PASS") << s.message;
      ++certified;
    }
    std::printf("  %-8s %-9s %-15s %10lld it\n", to_string(engine), name, to_string(s.status),
                static_cast<long long>(s.iterations));
  }
  EXPECT_GE(certified, 9) << to_string(engine);
  }
}

TEST(Certificates, FirstOrderUnboundedWithoutAFeasibleIterate) {
  // Random 7x7 LP (crosscheck_random_mps.py seed 2027, model 1932): unbounded, but the PDHG
  // iterate drifts along the ray and never became primal-feasible to 1e-6 — 100M iterations
  // without a verdict. Now: stop at the first valid ray, get a feasible point from a
  // zero-objective solve, and let the gate check point + ray.
  const Model m = [] {
    Model x = make_model({5, -1, 4, -5, 0, 0, -5},
                         {{0, 5, 0, 0, 0, 2, 0},
                          {0, 5, -5, 0, 0, 0, -1},
                          {0, 0, 0, -1, 0, 0, 5},
                          {-3, 0, 0, 0, 0, -3, 4},
                          {0, 4, -1, -5, 0, 0, 0},
                          {4, 0, 0, 0, 0, -4, 4},
                          {0, 0, 0, 0, -5, -2, 0}},
                         {-kInf, -kInf, -15, -21, 24, -kInf, -kInf}, {31, 23, kInf, kInf, 24, -24, -9},
                         {0, 4, 1, -kInf, 0, 1, -kInf}, {kInf, 9, kInf, 2, kInf, 4, kInf});
    x.sense = -1;
    return x;
  }();
  for (Algorithm a : {Algorithm::R2hpdhg, Algorithm::Pdlp}) {
    for (bool presolve : {false, true}) {
      Options o;
      o.algorithm = a;
      o.presolve = presolve;
      o.iteration_limit = 200000;
      const Solution s = solve(m, o);
      EXPECT_EQ(s.status, Status::Unbounded) << to_string(a) << " " << s.message;
      EXPECT_EQ(s.check, "PASS") << to_string(a) << " " << s.message;
    }
  }
  // primal AND dual infeasible (x has an improving ray, but y <= -1 with y >= 0): the answer
  // must be a certified Infeasible, whichever direction the engine finds first
  const Model both = make_model({-1, 0}, {{0, 1}}, {-kInf}, {-1}, {0, 0}, {kInf, kInf});
  for (Algorithm a : {Algorithm::R2hpdhg, Algorithm::Pdlp, Algorithm::Simplex}) {
    Options o;
    o.algorithm = a;
    o.presolve = false;
    const Solution s = solve(both, o);
    EXPECT_EQ(s.status, Status::Infeasible) << to_string(a) << " " << s.message;
    EXPECT_EQ(s.check, "PASS") << to_string(a) << " " << s.message;
  }
}

TEST(Certificates, ZeroMeasureFarkasVectorIsRejected) {
  // Random MPS seed 99, model 2004 (max -x0 + x2 + x3; 5 x0 - x2 = 49, x0 in [4, 10], x2 in [1, 2]
  // ...): FEASIBLE (optimum -6). r = (9.36e-4, 0, 0) has L0 = 49r - 50r + r = 0 exactly, but
  // +2e-15 in fp64 with no violation, and the tolerance test used to accept it — a wrong
  // Infeasible that passed the gate. L0 must be a meaningful fraction of its summands.
  Model m = make_model({-1, 0, 1, 1}, {{5, 0, -1, 0}, {0, 0, 0, -5}, {-2, 0, -5, 0}}, {49, -16, -26}, {49, -15, kInf},
                       {4, 0, 1, 0}, {10, kInf, 2, 3});
  m.sense = -1;
  const CertificateCheck c = check_infeasibility_certificate(m, {0.00093634084528417731, 0, 0});
  EXPECT_FALSE(c.passed) << c.detail;
  for (Algorithm a : {Algorithm::R2hpdhg, Algorithm::Pdlp, Algorithm::Simplex}) {
    for (bool presolve : {false, true}) {
      Options o;
      o.algorithm = a;
      o.presolve = presolve;
      const Solution s = solve(m, o);
      EXPECT_EQ(s.status, Status::Optimal) << to_string(a) << " " << s.message;
      EXPECT_NEAR(s.objective, -6.0, 1e-6) << to_string(a);
    }
  }
}

TEST(Certificates, EngineRayTestIgnoresRoundingNoise) {
  // pdhg::ray_test must not accept a direction whose objective is rounding noise relative to
  // its summands (the engines stop at the first accepted ray). Scaling off: vectors pass as is.
  pdhg::ScalingOptions none;
  none.geometric_mean_iterations = 0;
  none.ruiz_iterations = 0;
  none.pock_chambolle = false;
  none.bound_objective_rescaling = false;
  {  // primal side: c = (0.3, -0.1, -0.2), free columns, d = (1, 1, 1): cᵀd = -2.8e-17 exactly
    const Model m = make_model({0.3, -0.1, -0.2}, {{1, 0, 0}}, {-kInf}, {kInf}, {-kInf, -kInf, -kInf}, {kInf, kInf, kInf});
    const pdhg::ScaledProblem sp = pdhg::make_scaled_problem(m, none);
    EXPECT_FALSE(pdhg::ray_test(sp, {1, 1, 1}, {0}).dual_infeasible);
    EXPECT_TRUE(pdhg::ray_test(sp, {-1, 0, 0}, {0}).dual_infeasible);  // a real ray: cᵀd = -0.3
  }
  {  // dual side: seed 99 #2004 (feasible) with its zero-measure r; and a real Farkas vector
    const Model m = make_model({-1, 0, 1, 1}, {{5, 0, -1, 0}, {0, 0, 0, -5}, {-2, 0, -5, 0}}, {49, -16, -26},
                               {49, -15, kInf}, {4, 0, 1, 0}, {10, kInf, 2, 3});
    const pdhg::ScaledProblem sp = pdhg::make_scaled_problem(m, none);
    EXPECT_FALSE(pdhg::ray_test(sp, {0, 0, 0, 0}, {0.00093634084528417731, 0, 0}).primal_infeasible);
    const Model inf = infeasible();  // ScaledProblem keeps a pointer to its model
    const pdhg::ScaledProblem si = pdhg::make_scaled_problem(inf, none);
    EXPECT_TRUE(pdhg::ray_test(si, {0, 0}, {-1, 1}).primal_infeasible);
  }
}

TEST(Limits, FirstOrderEnginesStopNearTheTimeLimit) {
  // 60k x 60k random covering LP (min cᵀx, A x >= 1, x >= 0, 5 entries per column): far from
  // converged after 0.3 s. The engines used to read the clock only every 64 iterations, and the
  // certified bound's refinement then ran for seconds (a 3 s limit took 32 s at 1e6 rows).
  // Margin: setup is not interruptible, and the sanitizer CI jobs are several times slower.
  const int n = 60000;
  std::mt19937 rng(3);
  Model m;
  m.num_rows = m.num_cols = n;
  m.col_start.push_back(0);
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < 5; ++k) {
      m.row_index.push_back(static_cast<int>(rng() % n));
      m.value.push_back(1.0 + (rng() % 100) / 50.0);
    }
    std::sort(m.row_index.end() - 5, m.row_index.end());
    for (int k = 1; k < 5; ++k)  // no duplicate rows within a column
      if (m.row_index[m.row_index.size() - 5 + k] <= m.row_index[m.row_index.size() - 6 + k])
        m.row_index[m.row_index.size() - 5 + k] = -1;
    // drop the duplicates marked -1
    std::vector<int> ri(m.row_index.end() - 5, m.row_index.end());
    std::vector<double> va(m.value.end() - 5, m.value.end());
    m.row_index.resize(m.row_index.size() - 5);
    m.value.resize(m.value.size() - 5);
    for (int k = 0; k < 5; ++k)
      if (ri[k] >= 0) m.row_index.push_back(ri[k]), m.value.push_back(va[k]);
    m.col_start.push_back(static_cast<int>(m.value.size()));
  }
  m.obj.resize(n);
  for (double& c : m.obj) c = 1.0 + (rng() % 1000) / 100.0;
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, kInf);
  m.row_lower.assign(n, 1.0);
  m.row_upper.assign(n, kInf);
  ASSERT_TRUE(m.validate().empty()) << m.validate();
  for (Algorithm a : {Algorithm::R2hpdhg, Algorithm::Pdlp}) {
    Options o;
    o.algorithm = a;
    o.time_limit = 0.3;
    const auto t0 = std::chrono::steady_clock::now();
    const Solution s = solve(m, o);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_TRUE(s.status == Status::TimeLimit || s.status == Status::Optimal) << to_string(s.status);
    EXPECT_LT(wall, o.time_limit + s.setup_seconds + 3.0) << to_string(a) << ": " << s.iterations << " iterations";
  }
}

TEST(Concurrency, ParallelSolvesFromSeveralThreadsMatchSequentialResults) {
  // Library users may call solve() from several threads. The thread pool is process-wide; it
  // used to take a new job while another was running (and set_threads could stop workers in
  // use) when threads > 1. Results must be bit-identical to sequential runs, whatever the
  // interleaving and per-call thread counts. (The TSan CI job runs this too.)
  std::vector<Model> models;
  for (const char* name : {"afiro", "sc50a", "blend", "share2b", "stocfor1", "recipe"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(std::string(PS26119_SOURCE_DIR) + "/data/netlib_small/" + name + ".lpm", m).ok);
    models.push_back(std::move(m));
  }
  auto options = [](int k) {
    Options o;
    o.algorithm = k % 3 == 2 ? Algorithm::Simplex : Algorithm::R2hpdhg;
    o.threads = 1 + k % 4;
    return o;
  };
  std::vector<Solution> ref;
  for (std::size_t i = 0; i < models.size(); ++i) ref.push_back(solve(models[i], options(static_cast<int>(i))));
  std::vector<std::vector<Solution>> got(4);
  std::vector<std::thread> pool;
  for (int t = 0; t < 4; ++t)
    pool.emplace_back([&, t] {
      for (int rep = 0; rep < 2; ++rep)
        for (std::size_t i = 0; i < models.size(); ++i)
          got[t].push_back(solve(models[(i + t) % models.size()], options(static_cast<int>((i + t) % models.size()))));
    });
  for (auto& th : pool) th.join();
  for (int t = 0; t < 4; ++t)
    for (std::size_t k = 0; k < got[t].size(); ++k) {
      const std::size_t i = (k % models.size() + t) % models.size();
      const Solution& s = got[t][k];
      EXPECT_EQ(s.status, ref[i].status) << t << "/" << k;
      EXPECT_EQ(s.iterations, ref[i].iterations) << t << "/" << k;
      ASSERT_EQ(s.x.size(), ref[i].x.size());
      EXPECT_EQ(std::memcmp(s.x.data(), ref[i].x.data(), s.x.size() * sizeof(double)), 0) << t << "/" << k;
    }
}
