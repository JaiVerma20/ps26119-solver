// test_model.cpp — hand-built Model invariants, fingerprint stability, status strings.
#include <gtest/gtest.h>

#include "model_builder.h"
#include "ps26119/solve.h"
#include "ps26119/version.h"
#include <cmath>
#include <limits>

using namespace ps26119;
using ps26119::test::make_model;

namespace {

// min -x0 - 2 x1  s.t.  x0 + x1 <= 4,  1 <= x0 - x1 (ranged: [1, 3]),  0 <= x <= 10
Model small() {
  return make_model({-1, -2}, {{1, 1}, {1, -1}}, {-kInf, 1}, {4, 3}, {0, 0}, {10, 10});
}

}  // namespace

TEST(Model, HandBuiltIsValid) {
  Model m = small();
  EXPECT_EQ(m.validate(), "");
  EXPECT_EQ(m.num_rows, 2);
  EXPECT_EQ(m.num_cols, 2);
  EXPECT_EQ(m.nnz(), 4u);
  EXPECT_EQ(m.col_start.size(), 3u);
}

TEST(Model, DetectsBrokenInvariants) {
  {
    Model m = small();
    m.col_start[1] = 5;  // not monotone / wrong nnz
    EXPECT_NE(m.validate(), "");
  }
  {
    Model m = small();
    m.row_index[0] = 7;
    EXPECT_NE(m.validate(), "");
  }
  {
    Model m = small();
    m.row_index[1] = m.row_index[0];  // duplicate within column 0
    EXPECT_NE(m.validate(), "");
  }
  {
    // Crossed bounds are valid data describing an infeasible model (integration decision,
    // docs/DECISIONS.md #27): validate() accepts them, crossed_bounds() reports them and
    // solve() returns Infeasible without running an engine.
    Model m = small();
    m.col_lower[0] = 11;  // lower > upper
    EXPECT_EQ(m.validate(), "");
    EXPECT_NE(m.crossed_bounds(), "");
    EXPECT_EQ(solve(m).status, Status::Infeasible);
    Model r = small();
    r.row_lower[0] = r.row_upper[0] + 1;
    EXPECT_NE(r.crossed_bounds(), "");
    EXPECT_EQ(solve(r).status, Status::Infeasible);
    EXPECT_EQ(small().crossed_bounds(), "");
  }
  {
    Model m = small();
    m.row_lower[1] = kInf;
    EXPECT_NE(m.validate(), "");
  }
  {
    Model m = small();
    m.col_upper[1] = -kInf;
    EXPECT_NE(m.validate(), "");
  }
  {
    Model m = small();
    m.sense = 0;
    EXPECT_NE(m.validate(), "");
  }
  {
    Model m = small();
    m.obj.pop_back();
    EXPECT_NE(m.validate(), "");
  }
  {
    Model m = small();
    m.value[2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NE(m.validate(), "");
  }
}

TEST(Model, EmptyModelIsValid) {
  Model m;
  m.col_start = {0};
  EXPECT_EQ(m.validate(), "");
}

TEST(Model, FingerprintIsStableAndSensitive) {
  Model a = small();
  Model b = small();
  EXPECT_EQ(a.fingerprint(), b.fingerprint());
  EXPECT_EQ(a.fingerprint_hex().size(), 16u);

  b.row_names[0] = "renamed";  // names are not hashed
  EXPECT_EQ(a.fingerprint(), b.fingerprint());

  b.value[3] = std::nextafter(b.value[3], 0.0);  // one ulp changes it
  EXPECT_NE(a.fingerprint(), b.fingerprint());

  Model c = small();
  c.sense = -1;
  EXPECT_NE(a.fingerprint(), c.fingerprint());

  Model d = small();
  d.is_integer = {0, 1};
  EXPECT_NE(a.fingerprint(), d.fingerprint());

  Model e = small();
  e.is_integer = {0, 0};  // explicit all-continuous == empty
  EXPECT_EQ(a.fingerprint(), e.fingerprint());

  Model f = small();
  f.obj_offset = -0.0;  // -0 hashes like +0
  EXPECT_EQ(a.fingerprint(), f.fingerprint());
}

TEST(Model, ObjectiveAndActivity) {
  Model m = small();
  m.obj_offset = 5;
  std::vector<double> x = {3, 1};
  EXPECT_DOUBLE_EQ(m.objective_value(x), 5 - 3 - 2);
  auto ax = m.row_activity(x);
  EXPECT_DOUBLE_EQ(ax[0], 4);
  EXPECT_DOUBLE_EQ(ax[1], 2);
}

TEST(Status, RoundTripsAndExitCodes) {
  for (Status s : {Status::Optimal, Status::Infeasible, Status::Unbounded, Status::IterationLimit,
                   Status::TimeLimit, Status::NumericalError, Status::NotSolved}) {
    Status back;
    ASSERT_TRUE(status_from_string(to_string(s), back));
    EXPECT_EQ(back, s);
  }
  EXPECT_EQ(exit_code(Status::Optimal), 0);
  EXPECT_EQ(exit_code(Status::Infeasible), 1);
  EXPECT_EQ(exit_code(Status::TimeLimit), 1);
  EXPECT_EQ(exit_code(Status::NumericalError), 5);
}

TEST(Solve, InvalidModelIsNotSolved) {
  Model m = small();
  m.col_upper[0] = std::nan("");  // invalid data (crossed bounds are Infeasible, see above)
  Solution s = solve(m);
  EXPECT_EQ(s.status, Status::NotSolved);
  EXPECT_NE(s.message.find("invalid model"), std::string::npos);
  EXPECT_EQ(s.model_fingerprint, m.fingerprint_hex());
}

TEST(Version, HasName) { EXPECT_STRNE(kProductName, ""); }
