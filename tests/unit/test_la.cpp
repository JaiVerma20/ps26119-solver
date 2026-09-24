// test_la.cpp — sparse kernels and the ‖A‖₂ estimate that sets the PDHG step size.
// Regression: a stagnation-stopped power iteration under-estimated ‖A‖₂ on matrices with
// clustered top singular values (Netlib scrs8), making η‖A‖ > 1 and PDHG diverge. The
// estimate must never be below the true norm and must be tight.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "la/csr.h"

using namespace ps26119::la;

namespace {

Csr<double> from_dense(const std::vector<std::vector<double>>& d) {
  Csr<double> a;
  a.rows = static_cast<int>(d.size());
  a.cols = static_cast<int>(d[0].size());
  a.row_ptr.push_back(0);
  for (const auto& row : d) {
    for (int j = 0; j < a.cols; ++j)
      if (row[j] != 0) a.col.push_back(j), a.val.push_back(row[j]);
    a.row_ptr.push_back(static_cast<std::int64_t>(a.val.size()));
  }
  return a;
}

}  // namespace

TEST(Norm2, DiagonalWithClusteredTopSingularValues) {
  // σ = 1, 1 − 1e-4, 1 − 2e-4, ... : the regime where the old stopping rule stalled.
  const int n = 400;
  std::vector<std::vector<double>> d(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) d[i][i] = 1.0 - 1e-4 * i;
  auto a = from_dense(d);
  auto at = transpose(a);
  const double est = estimate_norm2(a, at);
  EXPECT_GE(est, 1.0);
  EXPECT_LE(est, 1.0 + 2e-4);
}

TEST(Norm2, RandomSparseNeverBelowTrueNorm) {
  // true norm from a long, tight power iteration on a dense copy
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> U(-1, 1);
  for (int trial = 0; trial < 10; ++trial) {
    const int m = 30 + trial * 7, n = 50 + trial * 5;
    std::vector<std::vector<double>> d(m, std::vector<double>(n, 0.0));
    for (auto& row : d)
      for (double& v : row)
        if (U(rng) > 0.7) v = U(rng);
    auto a = from_dense(d);
    auto at = transpose(a);
    // reference: power iteration to machine precision on AᵀA
    std::vector<double> v(n, 1.0), t(m), w(n);
    double lam = 0;
    for (int it = 0; it < 200000; ++it) {
      a.multiply(v.data(), t.data());
      at.multiply(t.data(), w.data());
      double nw = 0;
      for (double e : w) nw += e * e;
      nw = std::sqrt(nw);
      for (int j = 0; j < n; ++j) v[j] = w[j] / nw;
      if (std::fabs(nw - lam) < 1e-15 * nw) break;
      lam = nw;
    }
    const double truth = std::sqrt(lam);
    const double est = estimate_norm2(a, at);
    EXPECT_GE(est, truth * (1 - 1e-12)) << "trial " << trial;
    EXPECT_LE(est, truth * (1 + 1e-3)) << "trial " << trial;
  }
}

TEST(Norm2, ZeroAndTransposeRoundTrip) {
  Csr<double> z;
  z.rows = 3;
  z.cols = 2;
  z.row_ptr = {0, 0, 0, 0};
  EXPECT_EQ(estimate_norm2(z, transpose(z)), 0.0);
  auto a = from_dense({{1, 0, 2}, {0, 3, 0}});
  auto tt = transpose(transpose(a));
  EXPECT_EQ(tt.row_ptr, a.row_ptr);
  EXPECT_EQ(tt.col, a.col);
  EXPECT_EQ(tt.val, a.val);
}
