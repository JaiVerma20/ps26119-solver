// test_lpm.cpp — .lpm reader: sizes, cross-language fingerprint agreement, round trip,
// and malformed-input handling.
#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "io/lpm_reader.h"
#include "model_builder.h"

using namespace ps26119;

namespace {
std::string netlib(const std::string& name) {
  return std::string(PS26119_SOURCE_DIR) + "/data/netlib_small/" + name + ".lpm";
}
}  // namespace

TEST(Lpm, AfiroSizesFromMpsBridge) {
  Model m;
  auto r = io::read_lpm(netlib("afiro"), m);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(m.num_rows, 27);  // 28 MPS rows minus the objective row
  EXPECT_EQ(m.num_cols, 32);
  EXPECT_EQ(m.nnz(), 83u);
  EXPECT_EQ(m.sense, 1);
  EXPECT_EQ(m.row_names.size(), 27u);
  EXPECT_EQ(m.col_names.size(), 32u);
}

// The Python bridge (highspy) writes its own fingerprint of the model it read. The C++
// fingerprint of the model we load must be identical — two independent hash
// implementations over two independent parsers.
TEST(Lpm, FingerprintMatchesPythonForAllSmallNetlib) {
  for (const char* name : {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105",
                           "stocfor1", "recipe"}) {
    Model m;
    auto r = io::read_lpm(netlib(name), m);
    ASSERT_TRUE(r.ok) << name << ": " << r.error;
    ASSERT_FALSE(r.declared_fingerprint.empty()) << name;
    EXPECT_EQ(m.fingerprint_hex(), r.declared_fingerprint) << name;
  }
}

TEST(Lpm, WriteReadRoundTrip) {
  Model a = test::make_model({1.5, -2, 0.1}, {{1, 1, 0}, {0, -1, 3}}, {-kInf, 2}, {4, kInf},
                             {0, -kInf, -1}, {kInf, kInf, 1e30}, -1, 3.25);
  a.is_integer = {0, 1, 0};
  a.name = "hand";
  const std::string path = testing::TempDir() + "roundtrip.lpm";
  std::string err;
  ASSERT_TRUE(io::write_lpm(path, a, err)) << err;
  Model b;
  auto r = io::read_lpm(path, b);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_EQ(a.fingerprint(), b.fingerprint());
  EXPECT_EQ(r.declared_fingerprint, a.fingerprint_hex());
  EXPECT_EQ(b.name, "hand");
  EXPECT_EQ(b.row_names, a.row_names);
  EXPECT_EQ(b.col_upper[0], kInf);
  EXPECT_EQ(b.col_lower[1], -kInf);
  std::remove(path.c_str());
}

TEST(Lpm, RejectsMalformedInput) {
  Model m;
  EXPECT_FALSE(io::read_lpm_string("", m).ok);
  EXPECT_FALSE(io::read_lpm_string("LPM 2\nEND\n", m).ok);
  EXPECT_FALSE(io::read_lpm_string("LPM 1\nOBJ\n1\nEND\n", m).ok);  // sizes first
  EXPECT_FALSE(io::read_lpm_string("LPM 1\nROWS 0\nCOLS 1\nNNZ 0\nOBJ\nabc\nEND\n", m).ok);
  EXPECT_FALSE(io::read_lpm_string("LPM 1\nROWS 0\nCOLS 0\nNNZ 0\n", m).ok);  // no END
  // col_lower > col_upper is caught by validate()
  auto r = io::read_lpm_string(
      "LPM 1\nROWS 0\nCOLS 1\nNNZ 0\nOBJ\n1\nCOL_LOWER\n2\nCOL_UPPER\n1\nROW_LOWER\nROW_UPPER\n"
      "COL_START\n0 0\nROW_INDEX\nVALUE\nEND\n",
      m);
  EXPECT_FALSE(r.ok);
  EXPECT_NE(r.error.find("invalid model"), std::string::npos);
  // the minimal valid file
  r = io::read_lpm_string(
      "LPM 1\nROWS 0\nCOLS 1\nNNZ 0\nOBJ\n1\nCOL_LOWER\n0\nCOL_UPPER\ninf\nROW_LOWER\nROW_UPPER\n"
      "COL_START\n0 0\nROW_INDEX\nVALUE\nEND\n",
      m);
  EXPECT_TRUE(r.ok) << r.error;
}

#include "io/solution_reader.h"
#include "io/solution_writer.h"

TEST(SolutionFile, WriteReadRoundTrip) {
  Model m = test::make_model({1, 2}, {{1, 1}}, {1}, {kInf}, {0, 0}, {kInf, kInf});
  Solution s;
  s.status = Status::Optimal;
  s.x = {0.25, 0.75};
  s.z = {0, 1};
  s.y = {1.0 / 3.0};
  s.row_activity = {1.0};
  s.objective = 1.75;
  s.engine = "test";
  s.model_fingerprint = m.fingerprint_hex();
  const std::string path = testing::TempDir() + "rt.sol";
  std::string err;
  ASSERT_TRUE(io::write_solution(path, m, s, err)) << err;
  Solution r;
  ASSERT_TRUE(io::read_solution(path, r, err)) << err;
  EXPECT_EQ(r.status, Status::Optimal);
  EXPECT_EQ(r.x, s.x);  // %.17g round-trips exactly
  EXPECT_EQ(r.y, s.y);
  EXPECT_EQ(r.z, s.z);
  EXPECT_EQ(r.model_fingerprint, s.model_fingerprint);
  EXPECT_DOUBLE_EQ(r.objective, 1.75);
  std::remove(path.c_str());
}
