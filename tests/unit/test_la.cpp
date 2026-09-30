// test_la.cpp — sparse kernels and the ‖A‖₂ estimate that sets the PDHG step size.
// Regression: a stagnation-stopped power iteration under-estimated ‖A‖₂ on matrices with
// clustered top singular values (Netlib scrs8), making η‖A‖ > 1 and PDHG diverge. The
// estimate must never be below the true norm and must be tight.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <mutex>
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

TEST(CsrMultiply, NnzBalancedChunksCoverEveryRowOnceForSkewedShapes) {
  // Chunks are balanced by nnz + rows (wide matrices with a few long rows used to run on one
  // thread). Every row must be computed exactly once, bit-identically to the serial product.
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-1, 1);
  auto make = [&](const std::vector<int>& row_len, int cols) {
    Csr<double> a;
    a.rows = static_cast<int>(row_len.size());
    a.cols = cols;
    a.row_ptr = {0};
    for (int len : row_len) {
      for (int k = 0; k < len; ++k) a.col.push_back(static_cast<int>(rng() % cols)), a.val.push_back(u(rng));
      a.row_ptr.push_back(static_cast<std::int64_t>(a.col.size()));
    }
    return a;
  };
  std::vector<std::vector<int>> shapes = {
      std::vector<int>(25, 5000),                     // fit2d-like: 25 rows x 5000
      [] { std::vector<int> r(20000, 0); r[7] = 200000; return r; }(),  // one dense row, rest empty
      [] { std::vector<int> r(3000, 3); r[0] = 100000; r[2999] = 100000; return r; }(),
  };
  for (const auto& shape : shapes) {
    const Csr<double> a = make(shape, 50000);
    std::vector<double> x(a.cols);
    for (double& v : x) v = u(rng);
    std::vector<double> ref(a.rows, -7.0), par(a.rows, -7.0);
    ThreadPool::instance().set_threads(1);
    a.multiply<double>(x.data(), ref.data());
    for (int t : {2, 3, 4, 7}) {
      ThreadPool::instance().set_threads(t);
      std::fill(par.begin(), par.end(), -7.0);
      a.multiply<double>(x.data(), par.data());
      EXPECT_EQ(ref, par) << "threads " << t;
      std::vector<int> seen(a.rows, 0);  // for_rows: disjoint cover of [0, rows)
      std::mutex mu;
      a.for_rows([&](std::int64_t b, std::int64_t e) {
        std::lock_guard<std::mutex> lk(mu);
        for (std::int64_t i = b; i < e; ++i) ++seen[i];
      });
      EXPECT_EQ(std::count(seen.begin(), seen.end(), 1), a.rows) << "threads " << t;
    }
  }
  ThreadPool::instance().set_threads(1);
}
