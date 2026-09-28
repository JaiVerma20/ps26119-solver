// test_ranging.cpp — sensitivity ranging (src/core/ranging.cpp):
//   * a textbook LP with published ranges (Hillier & Lieberman, "Introduction to Operations
//     Research", the Wyndor Glass problem: max 3x1 + 5x2, x1 ≤ 4, 2x2 ≤ 12, 3x1 + 2x2 ≤ 18);
//   * a property test on the committed small Netlib LPs, by re-solving: moving a cost INSIDE its
//     range keeps the old optimal x optimal; moving a binding bound inside its range changes the
//     optimum by exactly dual × Δ;
//   * refusals: MILP, a non-optimal status, a point that is not a basic optimal solution.
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "io/lpm_reader.h"
#include "ps26119/ranging.h"
#include "ps26119/solve.h"

using namespace ps26119;

namespace {
std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }

Solution simplex(const Model& m) {
  Options o;
  o.algorithm = Algorithm::Simplex;
  return solve(m, o);
}

double objective_at(const Model& m, const std::vector<double>& x) {
  double s = m.obj_offset;
  for (int j = 0; j < m.num_cols; ++j) s += m.obj[j] * x[j];
  return s;
}

Model wyndor() {
  Model m;
  m.name = "wyndor";
  m.num_rows = 3;
  m.num_cols = 2;
  m.sense = -1;
  m.obj = {3, 5};
  m.col_lower = {0, 0};
  m.col_upper = {kInf, kInf};
  m.row_lower = {-kInf, -kInf, -kInf};
  m.row_upper = {4, 12, 18};
  m.col_start = {0, 2, 4};
  m.row_index = {0, 2, 1, 2};
  m.value = {1, 3, 2, 2};
  return m;
}
}  // namespace

TEST(Ranging, TextbookWyndorGlass) {
  const Model m = wyndor();
  const Solution s = simplex(m);
  ASSERT_EQ(s.status, Status::Optimal);
  ASSERT_NEAR(s.objective, 36.0, 1e-9);
  const RangingResult r = compute_ranging(m, s);
  ASSERT_TRUE(r.ok) << r.message;
  // costs: c1 in [0, 7.5], c2 in [2, +inf)
  EXPECT_NEAR(r.cols[0].lower, 0.0, 1e-9);
  EXPECT_NEAR(r.cols[0].upper, 7.5, 1e-9);
  EXPECT_NEAR(r.cols[1].lower, 2.0, 1e-9);
  EXPECT_TRUE(std::isinf(r.cols[1].upper) && r.cols[1].upper > 0);
  EXPECT_TRUE(r.cols[0].basic && r.cols[1].basic);
  // rows: x1 <= 4 not binding; 2 x2 <= 12 in [6, 18]; 3 x1 + 2 x2 <= 18 in [12, 24]
  EXPECT_EQ(r.rows[0].binding, 0);  // x1 = 2: the bound 4 may fall to 2, rise without limit
  EXPECT_NEAR(r.rows[0].bound, 4.0, 1e-12);
  EXPECT_NEAR(r.rows[0].lower, 2.0, 1e-9);
  EXPECT_TRUE(std::isinf(r.rows[0].upper) && r.rows[0].upper > 0);
  EXPECT_EQ(r.rows[1].binding, 1);
  EXPECT_NEAR(r.rows[1].lower, 6.0, 1e-9);
  EXPECT_NEAR(r.rows[1].upper, 18.0, 1e-9);
  EXPECT_NEAR(r.rows[1].dual, 1.5, 1e-9);
  EXPECT_EQ(r.rows[2].binding, 1);
  EXPECT_NEAR(r.rows[2].lower, 12.0, 1e-9);
  EXPECT_NEAR(r.rows[2].upper, 24.0, 1e-9);
  EXPECT_NEAR(r.rows[2].dual, 1.0, 1e-9);
  EXPECT_EQ(r.degenerate_basics, 0);

  // just beyond the c1 range the old point is no longer optimal (nondegenerate here)
  Model beyond = m;
  beyond.obj[0] = 7.5 + 0.1;
  const Solution sb = simplex(beyond);
  ASSERT_EQ(sb.status, Status::Optimal);
  EXPECT_GT(sb.objective, objective_at(beyond, s.x) + 1e-6);
}

TEST(Ranging, InsideTheRangesTheOptimumBehavesAsPredicted) {
  int costs_checked = 0, rhs_checked = 0;
  for (const char* name : {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe"}) {
    SCOPED_TRACE(name);
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    const Solution s = simplex(m);
    ASSERT_EQ(s.status, Status::Optimal);
    const RangingResult r = compute_ranging(m, s);
    ASSERT_TRUE(r.ok) << r.message;
    const double scale = 1.0 + std::fabs(s.objective);

    // costs: a point strictly inside [lower, upper]; the old x must stay optimal
    const int stride = std::max(1, m.num_cols / 12);
    for (int j = 0; j < m.num_cols; j += stride) {
      const CostRange& c = r.cols[j];
      double t;
      if (std::isfinite(c.upper) && c.upper > c.cost + 1e-7) t = c.cost + 0.5 * (c.upper - c.cost);
      else if (std::isfinite(c.lower) && c.lower < c.cost - 1e-7) t = c.cost - 0.5 * (c.cost - c.lower);
      else if (std::isinf(c.upper) && c.upper > 0) t = c.cost + 1.0 + std::fabs(c.cost);
      else if (std::isinf(c.lower) && c.lower < 0) t = c.cost - 1.0 - std::fabs(c.cost);
      else continue;  // degenerate range [c, c]
      Model mm = m;
      mm.obj[j] = t;
      const Solution ss = simplex(mm);
      ASSERT_EQ(ss.status, Status::Optimal) << "col " << j;
      EXPECT_NEAR(ss.objective, objective_at(mm, s.x), 1e-6 * (1.0 + std::fabs(ss.objective))) << "col " << j << " cost " << t;
      ++costs_checked;
    }
    // binding rows: move the binding bound inside its range; the optimum moves by dual × Δ
    int nonbinding = 0;
    for (const RhsRange& rr : r.rows) {
      if (rr.binding != 0) continue;
      if (std::isinf(rr.lower) == std::isinf(rr.upper) || nonbinding++ >= 6) continue;  // free rows
      // tighten the non-binding bound halfway to the activity: the optimum does not move
      const bool upper_bound = std::isinf(rr.upper);  // an upper bound has the range [activity, +inf)
      const double target = 0.5 * (rr.bound + (upper_bound ? rr.lower : rr.upper));
      Model mm = m;
      (upper_bound ? mm.row_upper : mm.row_lower)[rr.row] = target;
      const Solution ss = simplex(mm);
      ASSERT_EQ(ss.status, Status::Optimal) << "row " << rr.row;
      EXPECT_NEAR(ss.objective, s.objective, 1e-6 * scale) << "non-binding row " << rr.row;
      ++rhs_checked;
    }
    for (const RhsRange& rr : r.rows) {
      if (rr.binding == 0) continue;
      double delta;
      if (std::isfinite(rr.upper) && rr.upper > rr.bound + 1e-7) delta = 0.5 * std::min(rr.upper - rr.bound, 1.0 + std::fabs(rr.bound));
      else if (std::isfinite(rr.lower) && rr.lower < rr.bound - 1e-7) delta = -0.5 * std::min(rr.bound - rr.lower, 1.0 + std::fabs(rr.bound));
      else if (std::isinf(rr.upper)) delta = 0.5 * (1.0 + std::fabs(rr.bound));
      else continue;
      Model mm = m;
      const bool equality = m.row_lower[rr.row] == m.row_upper[rr.row];
      if (equality || rr.binding > 0) mm.row_upper[rr.row] += delta;
      if (equality || rr.binding < 0) mm.row_lower[rr.row] += delta;
      const Solution ss = simplex(mm);
      ASSERT_EQ(ss.status, Status::Optimal) << "row " << rr.row;
      EXPECT_NEAR(ss.objective, s.objective + rr.dual * delta, 1e-6 * scale) << "row " << rr.row << " delta " << delta;
      ++rhs_checked;
    }
  }
  EXPECT_GT(costs_checked, 60);
  EXPECT_GT(rhs_checked, 60);
}

TEST(Ranging, RefusesWhatItCannotRange) {
  Model mip;
  ASSERT_TRUE(io::read_lpm(data("mip_small/gt2.lpm"), mip).ok);
  Solution fake;
  fake.status = Status::Optimal;
  fake.x.assign(mip.num_cols, 0.0);
  fake.z.assign(mip.num_cols, 0.0);
  fake.y.assign(mip.num_rows, 0.0);
  EXPECT_FALSE(compute_ranging(mip, fake).ok);

  const Model m = wyndor();
  Solution s = simplex(m);
  Solution limit = s;
  limit.status = Status::TimeLimit;
  EXPECT_FALSE(compute_ranging(m, limit).ok);

  Solution moved = s;  // not a basic optimal solution any more
  moved.x[0] -= 1.0;
  const RangingResult r = compute_ranging(m, moved);
  EXPECT_FALSE(r.ok);
  EXPECT_FALSE(r.message.empty());

  RangingOptions small;
  small.max_rows = 2;
  EXPECT_FALSE(compute_ranging(m, s, small).ok);
}
