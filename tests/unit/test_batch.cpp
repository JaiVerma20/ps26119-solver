// test_batch.cpp — batched r²HPDHG (K LPs sharing A): every scenario's answer matches an
// independent single solve and passes the verifier-grade KKT check on its own data.
#include <gtest/gtest.h>

#include <cmath>

#include "io/lpm_reader.h"
#include "kkt_check.h"
#include "ps26119/batch.h"
#include "ps26119/solve.h"

using namespace ps26119;

namespace {
std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }

Model with(const Model& base, const Scenario& s) {
  Model m = base;
  if (!s.obj.empty()) m.obj = s.obj;
  if (!s.col_lower.empty()) m.col_lower = s.col_lower;
  if (!s.col_upper.empty()) m.col_upper = s.col_upper;
  if (!s.row_lower.empty()) m.row_lower = s.row_lower;
  if (!s.row_upper.empty()) m.row_upper = s.row_upper;
  return m;
}
}  // namespace

TEST(Batch, ScenariosMatchSingleSolves) {
  for (const char* name : {"afiro", "sc50a", "stocfor1", "share2b"}) {
    SCOPED_TRACE(name);
    Model base;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), base).ok);
    std::vector<Scenario> sc(4);
    // 0: unchanged; 1: objective +/-5% pattern; 2: row bounds loosened 2%; 3: both
    sc[1].obj = base.obj;
    for (std::size_t j = 0; j < base.obj.size(); ++j) sc[1].obj[j] *= 1.0 + 0.05 * ((j % 3) - 1.0);
    sc[2].row_lower = base.row_lower;
    sc[2].row_upper = base.row_upper;
    for (std::size_t i = 0; i < base.row_upper.size(); ++i) {
      if (std::isfinite(sc[2].row_upper[i]) && sc[2].row_lower[i] != sc[2].row_upper[i])
        sc[2].row_upper[i] += 0.02 * (1 + std::fabs(sc[2].row_upper[i]));
    }
    sc[3] = sc[2];
    sc[3].obj = sc[1].obj;
    Options o;
    auto sols = solve_batch(base, sc, o);
    ASSERT_EQ(sols.size(), sc.size());
    for (std::size_t k = 0; k < sc.size(); ++k) {
      SCOPED_TRACE(k);
      const Model mk = with(base, sc[k]);
      ASSERT_EQ(sols[k].status, Status::Optimal) << sols[k].message;
      const auto single = solve(mk, o);
      ASSERT_EQ(single.status, Status::Optimal);
      EXPECT_NEAR(sols[k].objective, single.objective, 1e-6 * (1 + std::fabs(single.objective)));
      const auto kk = test::kkt(mk, sols[k]);
      EXPECT_LE(kk.primal, 1e-6);
      EXPECT_LE(kk.dual, 1e-6);
      EXPECT_LE(kk.gap, 1e-6);
      EXPECT_EQ(sols[k].engine, "r2hpdhg-batch");
    }
  }
}

TEST(Batch, BadScenarioFailsAloneAndMaxSenseWorks) {
  Model base;
  ASSERT_TRUE(io::read_lpm(data("hand/max_ranged.lpm"), base).ok);  // MAX model, optimum 21.5
  std::vector<Scenario> sc(3);
  sc[1].obj = {1.0};  // wrong size
  sc[2].col_upper = base.col_upper;
  sc[2].col_upper[0] = 2.0;  // tighter bound on x
  auto sols = solve_batch(base, sc);
  EXPECT_EQ(sols[0].status, Status::Optimal) << sols[0].message;
  EXPECT_NEAR(sols[0].objective, 21.5, 1e-6);
  EXPECT_EQ(sols[1].status, Status::NotSolved);
  EXPECT_EQ(sols[2].status, Status::Optimal) << sols[2].message;
  Options o;
  o.algorithm = Algorithm::Oracle;
  const auto ref = solve(with(base, sc[2]), o);
  EXPECT_NEAR(sols[2].objective, ref.objective, 1e-6 * (1 + std::fabs(ref.objective)));
}

TEST(Batch, EmptyAndUnknownKnob) {
  Model base;
  ASSERT_TRUE(io::read_lpm(data("netlib_small/afiro.lpm"), base).ok);
  EXPECT_TRUE(solve_batch(base, {}).empty());
  Options o;
  o.engine_params = {{"nope", 1}};
  auto s = solve_batch(base, std::vector<Scenario>(2), o);
  EXPECT_EQ(s[0].status, Status::NotSolved);
}

// Each scenario's solution names the model it solved (base matrix + the scenario's objective
// and bounds), not the base model: the solution file's `model` line is what verify.py checks.
TEST(Batch, EachSolutionCarriesItsOwnScenarioFingerprint) {
  Model base;
  ASSERT_TRUE(io::read_lpm(data("netlib_small/afiro.lpm"), base).ok);
  std::vector<Scenario> sc(4);
  sc[1].obj = base.obj;
  sc[1].obj[0] += 1.0;
  sc[2].col_upper = base.col_upper;
  sc[2].col_upper[1] = 50.0;
  sc[3].row_lower = base.row_lower;  // equal to the base's: same model, same fingerprint
  auto sols = solve_batch(base, sc);
  ASSERT_EQ(sols.size(), sc.size());
  for (std::size_t k = 0; k < sc.size(); ++k) {
    SCOPED_TRACE(k);
    EXPECT_EQ(sols[k].model_fingerprint, with(base, sc[k]).fingerprint_hex());
  }
  EXPECT_EQ(sols[0].model_fingerprint, base.fingerprint_hex());
  EXPECT_EQ(sols[3].model_fingerprint, base.fingerprint_hex());
  EXPECT_NE(sols[1].model_fingerprint, base.fingerprint_hex());
  EXPECT_NE(sols[2].model_fingerprint, base.fingerprint_hex());
  EXPECT_NE(sols[1].model_fingerprint, sols[2].model_fingerprint);
  // also when the run fails before iterating (unknown engine knob)
  Options o;
  o.engine_params = {{"nope", 1}};
  auto bad = solve_batch(base, sc, o);
  EXPECT_EQ(bad[1].model_fingerprint, sols[1].model_fingerprint);
}
