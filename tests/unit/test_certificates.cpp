// test_certificates.cpp — checks of Infeasible / Unbounded claims (core/certificates.h) and
// their use by the gate: good certificates pass, bad or missing ones never certify a claim.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <cstdio>
#include <vector>

#include "core/certificates.h"
#include "core/gate.h"
#include "io/lpm_reader.h"
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
