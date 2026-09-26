// test_certificates.cpp — checks of Infeasible / Unbounded claims (core/certificates.h) and
// their use by the gate: good certificates pass, bad or missing ones never certify a claim.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "core/certificates.h"
#include "core/gate.h"
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
