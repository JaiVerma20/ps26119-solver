// Tests for the Layer 1 MPS reader.
#include <string>

#include "gpuopt/mps_reader.hpp"
#include "test_framework.hpp"

using namespace gpuopt;

namespace {

const std::string kData = GPUOPT_DATA_DIR;

int row_of(const LpProblem& lp, const std::string& name) {
  for (int i = 0; i < lp.num_rows(); ++i) {
    if (lp.row_names[i] == name) return i;
  }
  return -1;
}

int col_of(const LpProblem& lp, const std::string& name) {
  for (int j = 0; j < lp.num_cols(); ++j) {
    if (lp.col_names[j] == name) return j;
  }
  return -1;
}

// Coefficient A(row, col), 0 if absent.
double coef(const LpProblem& lp, int row, int col) {
  for (int k = lp.A.col_start[col]; k < lp.A.col_start[col + 1]; ++k) {
    if (lp.A.row_index[k] == row) return lp.A.value[k];
  }
  return 0.0;
}

}  // namespace

TEST(reads_tiny_max_model) {
  const auto r = read_mps_file(kData + "/examples/tiny_max.mps");
  const LpProblem& lp = r.problem;
  EXPECT_TRUE(lp.validate().empty());
  EXPECT_EQ(lp.name, std::string("TINY_MAX"));
  EXPECT_TRUE(lp.sense == ObjSense::kMaximize);
  EXPECT_EQ(lp.objective_name, std::string("PROFIT"));
  EXPECT_EQ(lp.num_rows(), 2);
  EXPECT_EQ(lp.num_cols(), 2);
  EXPECT_EQ(lp.A.nnz(), 4);
  EXPECT_NEAR(lp.obj[col_of(lp, "X")], 3.0, 0);
  EXPECT_NEAR(coef(lp, row_of(lp, "C2"), col_of(lp, "Y")), 3.0, 0);
  EXPECT_NEAR(lp.row_upper[row_of(lp, "C2")], 6.0, 0);
  EXPECT_TRUE(lp.row_lower[row_of(lp, "C2")] == -kInf);
  EXPECT_NEAR(lp.col_upper[col_of(lp, "X")], 3.0, 0);
  EXPECT_TRUE(lp.col_upper[col_of(lp, "Y")] == kInf);
  EXPECT_TRUE(r.warnings.empty());
}

TEST(reads_ranges_bounds_and_offset) {
  const auto r = read_mps_file(kData + "/examples/features.mps");
  const LpProblem& lp = r.problem;
  EXPECT_TRUE(lp.validate().empty());
  EXPECT_NEAR(lp.obj_offset, 10.0, 0);  // RHS on objective row is the negated constant

  // L row, G row, E row, E row with positive range, L row with range.
  EXPECT_TRUE(lp.row_lower[row_of(lp, "LIM1")] == -kInf);
  EXPECT_NEAR(lp.row_upper[row_of(lp, "LIM1")], 4.0, 0);
  EXPECT_NEAR(lp.row_lower[row_of(lp, "LIM2")], 1.0, 0);
  EXPECT_TRUE(lp.row_upper[row_of(lp, "LIM2")] == kInf);
  EXPECT_NEAR(lp.row_lower[row_of(lp, "MYEQN")], 7.0, 0);
  EXPECT_NEAR(lp.row_upper[row_of(lp, "MYEQN")], 7.0, 0);
  EXPECT_NEAR(lp.row_lower[row_of(lp, "RNGEQ")], 2.0, 0);
  EXPECT_NEAR(lp.row_upper[row_of(lp, "RNGEQ")], 5.0, 0);
  EXPECT_NEAR(lp.row_lower[row_of(lp, "RNGL")], -5.0, 0);
  EXPECT_NEAR(lp.row_upper[row_of(lp, "RNGL")], -1.0, 0);

  const int x2 = col_of(lp, "X2"), x3 = col_of(lp, "X3"), x4 = col_of(lp, "X4"), x5 = col_of(lp, "X5");
  EXPECT_TRUE(lp.col_lower[x2] == -kInf);
  EXPECT_NEAR(lp.col_upper[x2], 1.0, 0);
  EXPECT_NEAR(lp.col_lower[x3], -1.0, 0);
  EXPECT_NEAR(lp.col_upper[x3], 8.0, 0);
  EXPECT_NEAR(lp.col_lower[x4], 0.5, 0);
  EXPECT_NEAR(lp.col_upper[x4], 0.5, 0);
  EXPECT_TRUE(lp.col_lower[x5] == -kInf);  // negative UP with default lower bound
  EXPECT_NEAR(lp.col_upper[x5], -1.0, 0);
  EXPECT_EQ(r.warnings.size(), size_t{1});
}

TEST(falls_back_to_fixed_format_for_names_with_spaces) {
  const auto r = read_mps_file(kData + "/examples/fixed_format.mps");
  const LpProblem& lp = r.problem;
  EXPECT_TRUE(r.format_used == MpsFormat::kFixed);
  EXPECT_EQ(lp.name, std::string("FIXED FORMAT DEMO"));
  EXPECT_TRUE(row_of(lp, "MACH A") >= 0);
  EXPECT_TRUE(col_of(lp, "PROD 2") >= 0);
  EXPECT_NEAR(coef(lp, row_of(lp, "MACH B"), col_of(lp, "PROD 1")), 2.0, 0);
  EXPECT_NEAR(lp.row_upper[row_of(lp, "MACH B")], 60.0, 0);

  MpsReadOptions strict_free;
  strict_free.format = MpsFormat::kFree;
  EXPECT_THROWS(read_mps_file(kData + "/examples/fixed_format.mps", strict_free));
}

TEST(reads_integer_markers_and_binary_bounds) {
  const auto r = read_mps_file(kData + "/examples/knapsack_mip.mps");
  const LpProblem& lp = r.problem;
  EXPECT_TRUE(lp.sense == ObjSense::kMaximize);  // "OBJSENSE MAX" on one line
  EXPECT_EQ(lp.num_integers(), 3);
  EXPECT_TRUE(lp.is_integer[col_of(lp, "A")]);
  EXPECT_TRUE(lp.is_integer[col_of(lp, "B")]);
  EXPECT_TRUE(lp.is_integer[col_of(lp, "C")]);  // integer via BV, outside the marker block
  EXPECT_NEAR(lp.col_upper[col_of(lp, "C")], 1.0, 0);
}

TEST(sums_duplicate_entries_and_accepts_missing_set_names) {
  const char* text =
      "NAME DUP\n"
      "ROWS\n"
      " N obj\n"
      " L r1\n"
      "COLUMNS\n"
      " x obj 1 r1 2\n"
      " x r1 3\n"       // duplicate (r1, x): summed to 5
      "RHS\n"
      " r1 10\n"        // no RHS set name
      "BOUNDS\n"
      " UP x 7\n"       // no bound set name
      " FR y_missing_is_error_if_used\n"
      "ENDATA\n";
  EXPECT_THROWS(read_mps_string(text));  // bound on an unknown column

  const char* ok_text =
      "NAME DUP\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1 r1 2\n x r1 3\n"
      "RHS\n r1 10\nBOUNDS\n UP x 7\nENDATA\n";
  const auto r = read_mps_string(ok_text);
  EXPECT_NEAR(coef(r.problem, 0, 0), 5.0, 0);
  EXPECT_NEAR(r.problem.row_upper[0], 10.0, 0);
  EXPECT_NEAR(r.problem.col_upper[0], 7.0, 0);
  EXPECT_EQ(r.warnings.size(), size_t{1});  // "summed 1 duplicate matrix entries"
}

TEST(parses_number_formats) {
  const char* text =
      "NAME NUM\nROWS\n N obj\n E r1\n G r2\nCOLUMNS\n"
      " x obj 1.5D+02 r1 .5\n"
      " x r2 +3.\n"
      "RHS\n rhs r1 -2e-1 r2 1e30\n"
      "RANGES\n rng r1 -4\n"
      "ENDATA\n";
  const auto r = read_mps_string(text);
  const LpProblem& lp = r.problem;
  EXPECT_NEAR(lp.obj[0], 150.0, 0);
  EXPECT_NEAR(coef(lp, 0, 0), 0.5, 0);
  EXPECT_NEAR(coef(lp, 1, 0), 3.0, 0);
  EXPECT_NEAR(lp.row_lower[0], -4.2, 1e-15);  // E row with negative range: [b-|R|, b]
  EXPECT_NEAR(lp.row_upper[0], -0.2, 1e-15);
  EXPECT_TRUE(lp.row_lower[1] == kInf);        // 1e30 is infinity
}

TEST(extra_free_rows_are_dropped) {
  const char* text =
      "NAME FREE\nROWS\n N obj\n N other\n L r1\nCOLUMNS\n"
      " x obj 1 other 99\n x r1 1\nRHS\n rhs r1 4 other 5\nENDATA\n";
  const auto r = read_mps_string(text);
  EXPECT_EQ(r.problem.num_rows(), 1);
  EXPECT_EQ(r.problem.A.nnz(), 1);
  EXPECT_NEAR(r.problem.obj[0], 1.0, 0);
}

TEST(reports_errors_with_line_numbers) {
  const char* unknown_row = "NAME E\nROWS\n N obj\nCOLUMNS\n x nosuchrow 1\nENDATA\n";
  try {
    read_mps_string(unknown_row, {MpsFormat::kFree});
    EXPECT_TRUE(false);
  } catch (const MpsParseError& e) {
    EXPECT_EQ(e.line(), 5);
  }
  EXPECT_THROWS(read_mps_string("NAME Q\nROWS\n N obj\nQUADOBJ\nENDATA\n"));
  EXPECT_THROWS(read_mps_string("NAME B\nROWS\n N obj\n X r1\nENDATA\n"));
  EXPECT_THROWS(read_mps_file(kData + "/examples/does_not_exist.mps"));
  EXPECT_THROWS(read_mps_file("model.mps.gz"));
}

TEST_MAIN()
