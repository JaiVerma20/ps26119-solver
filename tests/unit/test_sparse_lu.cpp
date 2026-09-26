// Tests for the Layer 2 sparse LU (factorize, FTRAN, BTRAN, PFI update).
//
// Correctness is judged by residuals: x solves B x = b if ||B x - b|| is at
// round-off level relative to ||B|| ||x|| + ||b||. Small cases are also
// compared against a dense Gaussian elimination reference.
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>

#include "gpuopt/linalg/sparse_lu.hpp"
#include "gpuopt/mps_reader.hpp"
#include "test_framework.hpp"

using namespace gpuopt;

namespace {

const std::string kData = GPUOPT_DATA_DIR;

// Relative residual  ||B x - b||_inf / (||B||_inf ||x||_inf + ||b||_inf).
double ftran_residual(const SparseMatrixCSC& B, const std::vector<double>& x, const std::vector<double>& b) {
  std::vector<double> bx;
  B.multiply(x, bx);
  double r = 0, nb = 0, nx = 0, nB = 0;
  std::vector<double> row_sum(B.num_rows, 0.0);
  for (int j = 0; j < B.num_cols; ++j) {
    for (int k = B.col_start[j]; k < B.col_start[j + 1]; ++k) row_sum[B.row_index[k]] += std::fabs(B.value[k]);
  }
  for (int i = 0; i < B.num_rows; ++i) {
    r = std::max(r, std::fabs(bx[i] - b[i]));
    nb = std::max(nb, std::fabs(b[i]));
    nB = std::max(nB, row_sum[i]);
  }
  for (double v : x) nx = std::max(nx, std::fabs(v));
  return r / (nB * nx + nb + 1e-300);
}

double btran_residual(const SparseMatrixCSC& B, const std::vector<double>& y, const std::vector<double>& d) {
  std::vector<double> bty;
  B.multiply_transpose(y, bty);
  double r = 0, nd = 0, ny = 0, nB = 0;
  for (int j = 0; j < B.num_cols; ++j) {
    double col_sum = 0;
    for (int k = B.col_start[j]; k < B.col_start[j + 1]; ++k) col_sum += std::fabs(B.value[k]);
    nB = std::max(nB, col_sum);
    r = std::max(r, std::fabs(bty[j] - d[j]));
    nd = std::max(nd, std::fabs(d[j]));
  }
  for (double v : y) ny = std::max(ny, std::fabs(v));
  return r / (nB * ny + nd + 1e-300);
}

// Dense Gaussian elimination with partial pivoting (reference only).
std::vector<double> dense_solve(const SparseMatrixCSC& B, std::vector<double> b) {
  const int m = B.num_rows;
  std::vector<double> a = B.to_dense();
  for (int k = 0; k < m; ++k) {
    int p = k;
    for (int i = k + 1; i < m; ++i) {
      if (std::fabs(a[i * m + k]) > std::fabs(a[p * m + k])) p = i;
    }
    if (p != k) {
      for (int j = 0; j < m; ++j) std::swap(a[k * m + j], a[p * m + j]);
      std::swap(b[k], b[p]);
    }
    for (int i = k + 1; i < m; ++i) {
      const double f = a[i * m + k] / a[k * m + k];
      if (f == 0.0) continue;
      for (int j = k; j < m; ++j) a[i * m + j] -= f * a[k * m + j];
      b[i] -= f * b[k];
    }
  }
  std::vector<double> x(m);
  for (int k = m - 1; k >= 0; --k) {
    double s = b[k];
    for (int j = k + 1; j < m; ++j) s -= a[k * m + j] * x[j];
    x[k] = s / a[k * m + k];
  }
  return x;
}

// Random sparse m x m matrix that is nonsingular with probability ~1:
// a randomly permuted "diagonal" plus `extra` random entries per column.
SparseMatrixCSC random_sparse(int m, int extra, std::mt19937& rng) {
  std::uniform_int_distribution<int> row(0, m - 1);
  std::uniform_real_distribution<double> val(-10.0, 10.0);
  std::vector<int> perm(m);
  for (int i = 0; i < m; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<Triplet> t;
  for (int j = 0; j < m; ++j) {
    t.push_back({perm[j], j, val(rng) + (val(rng) > 0 ? 5.0 : -5.0)});
    for (int e = 0; e < extra; ++e) t.push_back({row(rng), j, val(rng)});
  }
  return build_csc(m, m, t);
}

std::vector<double> random_vector(int m, std::mt19937& rng) {
  std::uniform_real_distribution<double> val(-1.0, 1.0);
  std::vector<double> v(m);
  for (double& x : v) x = val(rng);
  return v;
}

// Factorizes B and checks FTRAN and BTRAN residuals on random right-hand sides.
void expect_accurate(const SparseMatrixCSC& B, std::mt19937& rng, double tol = 1e-12) {
  SparseLU lu;
  EXPECT_EQ(lu.factorize(B), B.num_rows);
  const std::vector<double> b = random_vector(B.num_rows, rng);
  std::vector<double> x = b;
  lu.ftran(x);
  const double rf = ftran_residual(B, x, b);
  const std::vector<double> d = random_vector(B.num_rows, rng);
  std::vector<double> y = d;
  lu.btran(y);
  const double rb = btran_residual(B, y, d);
  if (rf > tol || rb > tol) std::printf("  m=%d ftran %.2e btran %.2e\n", B.num_rows, rf, rb);
  EXPECT_TRUE(rf <= tol);
  EXPECT_TRUE(rb <= tol);
}

}  // namespace

TEST(solves_small_known_system) {
  // B = [[2, 1, 0], [0, 3, 1], [1, 0, 4]],  x = (1, 2, 3)  =>  b = (4, 9, 13)
  const SparseMatrixCSC B = build_csc(3, 3, {{0, 0, 2}, {2, 0, 1}, {0, 1, 1}, {1, 1, 3}, {1, 2, 1}, {2, 2, 4}});
  SparseLU lu;
  EXPECT_EQ(lu.factorize(B), 3);
  std::vector<double> x = {4, 9, 13};
  lu.ftran(x);
  EXPECT_NEAR(x[0], 1.0, 1e-14);
  EXPECT_NEAR(x[1], 2.0, 1e-14);
  EXPECT_NEAR(x[2], 3.0, 1e-14);
  // B^T y = d with y = (1, 1, 1)  =>  d = column sums = (3, 4, 5)
  std::vector<double> y = {3, 4, 5};
  lu.btran(y);
  for (double v : y) EXPECT_NEAR(v, 1.0, 1e-14);
}

TEST(identity_and_permutation_have_no_fill) {
  std::mt19937 rng(1);
  const int m = 50;
  std::vector<Triplet> t;
  std::vector<int> perm(m);
  for (int i = 0; i < m; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  for (int j = 0; j < m; ++j) t.push_back({perm[j], j, 1.0});
  const SparseMatrixCSC P = build_csc(m, m, t);
  SparseLU lu;
  EXPECT_EQ(lu.factorize(P), m);
  EXPECT_EQ(lu.stats().nnz_l, 0);
  EXPECT_EQ(lu.stats().nnz_u, m);
  expect_accurate(P, rng);
}

TEST(threshold_pivoting_avoids_tiny_pivot) {
  // Without the threshold test, Markowitz would happily pivot on 1e-14.
  const SparseMatrixCSC B = build_csc(2, 2, {{0, 0, 1e-14}, {1, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}});
  std::mt19937 rng(2);
  expect_accurate(B, rng, 1e-15);
}

TEST(matches_dense_reference_on_random_matrices) {
  std::mt19937 rng(3);
  double worst = 0.0;
  for (int trial = 0; trial < 300; ++trial) {
    const int m = 2 + trial % 60;
    const SparseMatrixCSC B = random_sparse(m, 1 + trial % 4, rng);
    SparseLU lu;
    if (lu.factorize(B) < m) continue;  // rare singular draw
    const std::vector<double> b = random_vector(m, rng);
    std::vector<double> x = b;
    lu.ftran(x);
    const std::vector<double> ref = dense_solve(B, b);
    double xmax = 1.0;
    for (double v : ref) xmax = std::max(xmax, std::fabs(v));
    for (int i = 0; i < m; ++i) worst = std::max(worst, std::fabs(x[i] - ref[i]) / xmax);
    EXPECT_TRUE(ftran_residual(B, x, b) < 1e-13);
  }
  std::printf("  worst relative difference to dense LU over 300 matrices: %.1e\n", worst);
  EXPECT_TRUE(worst < 1e-9);
}

TEST(singular_basis_is_completed_with_unit_columns) {
  // Column 2 = column 0 + column 1, and column 3 is empty: rank 2 of 4.
  const SparseMatrixCSC B = build_csc(4, 4, {{0, 0, 1}, {1, 0, 2}, {1, 1, 1}, {2, 1, 3},
                                             {0, 2, 1}, {1, 2, 3}, {2, 2, 3}});
  SparseLU lu;
  EXPECT_EQ(lu.factorize(B), 2);
  EXPECT_EQ(lu.replaced_columns().size(), size_t{2});

  // The factorization now represents B with those columns swapped for unit vectors.
  std::vector<int> basic = {0, 1, 2, 3};
  const int n = 4;
  for (auto [pos, row] : lu.replaced_columns()) basic[pos] = n + row;
  const SparseMatrixCSC completed = build_basis_matrix(B, basic);
  std::mt19937 rng(4);
  const std::vector<double> b = random_vector(4, rng);
  std::vector<double> x = b;
  lu.ftran(x);
  EXPECT_TRUE(ftran_residual(completed, x, b) < 1e-14);
  std::vector<double> y = b;
  lu.btran(y);
  EXPECT_TRUE(btran_residual(completed, y, b) < 1e-14);
}

TEST(pfi_updates_match_fresh_factorization) {
  std::mt19937 rng(5);
  const int m = 120, n = 300;
  // A structural matrix with a random basis of structurals and slacks.
  std::vector<Triplet> t;
  std::uniform_int_distribution<int> row(0, m - 1);
  std::uniform_real_distribution<double> val(-3.0, 3.0);
  for (int j = 0; j < n; ++j) {
    for (int e = 0; e < 4; ++e) t.push_back({row(rng), j, val(rng)});
  }
  const SparseMatrixCSC A = build_csc(m, n, t);
  std::vector<int> basic(m);
  for (int i = 0; i < m; ++i) basic[i] = n + i;  // start from the slack basis

  SparseLU lu;
  lu.factorize(build_basis_matrix(A, basic));
  int done = 0;
  for (int it = 0; it < 80; ++it) {
    const int q = std::uniform_int_distribution<int>(0, n - 1)(rng);
    if (std::find(basic.begin(), basic.end(), q) != basic.end()) continue;
    std::vector<double> alpha(m, 0.0);
    for (int k = A.col_start[q]; k < A.col_start[q + 1]; ++k) alpha[A.row_index[k]] = A.value[k];
    lu.ftran(alpha);
    int p = 0;  // most stable position, as a ratio test with a good tie-break would pick
    for (int i = 1; i < m; ++i) {
      if (std::fabs(alpha[i]) > std::fabs(alpha[p])) p = i;
    }
    if (!lu.update(p, alpha)) continue;
    basic[p] = q;
    ++done;
  }
  EXPECT_TRUE(done > 50);
  EXPECT_EQ(lu.stats().num_updates, done);

  const SparseMatrixCSC B = build_basis_matrix(A, basic);
  const std::vector<double> b = random_vector(m, rng);
  std::vector<double> x_updated = b;
  lu.ftran(x_updated);
  std::vector<double> y_updated = b;
  lu.btran(y_updated);
  EXPECT_TRUE(ftran_residual(B, x_updated, b) < 1e-11);
  EXPECT_TRUE(btran_residual(B, y_updated, b) < 1e-11);

  SparseLU fresh;
  EXPECT_EQ(fresh.factorize(B), m);
  std::vector<double> x_fresh = b;
  fresh.ftran(x_fresh);
  double diff = 0, scale = 1;
  for (int i = 0; i < m; ++i) {
    diff = std::max(diff, std::fabs(x_fresh[i] - x_updated[i]));
    scale = std::max(scale, std::fabs(x_fresh[i]));
  }
  std::printf("  %d PFI updates: max relative difference to fresh LU %.1e\n", done, diff / scale);
  EXPECT_TRUE(diff / scale < 1e-9);
}

// Uniformly random sparsity is a worst case for fill-in (real LP bases are far
// more structured), so this mainly checks accuracy at scale.
TEST(large_random_sparse_matrix) {
  std::mt19937 rng(6);
  const SparseMatrixCSC B = random_sparse(5000, 2, rng);
  SparseLU lu;
  EXPECT_EQ(lu.factorize(B), 5000);
  std::printf("  m=5000 nnz(B)=%lld nnz(L+U)=%lld factor %.3f s\n", lu.stats().nnz_basis,
              lu.stats().nnz_l + lu.stats().nnz_u, lu.stats().factor_seconds);
  expect_accurate(B, rng, 1e-11);
}

TEST(two_dimensional_grid_laplacian) {
  // 5-point Laplacian on a 60 x 60 grid (m = 3600): the textbook fill-in test.
  const int g = 60, m = g * g;
  std::vector<Triplet> t;
  for (int a = 0; a < g; ++a) {
    for (int b = 0; b < g; ++b) {
      const int i = a * g + b;
      t.push_back({i, i, 4.0});
      if (a > 0) t.push_back({i, i - g, -1.0});
      if (a + 1 < g) t.push_back({i, i + g, -1.0});
      if (b > 0) t.push_back({i, i - 1, -1.0});
      if (b + 1 < g) t.push_back({i, i + 1, -1.0});
    }
  }
  const SparseMatrixCSC B = build_csc(m, m, t);
  SparseLU lu;
  EXPECT_EQ(lu.factorize(B), m);
  const double fill = double(lu.stats().nnz_l + lu.stats().nnz_u) / lu.stats().nnz_basis;
  std::printf("  grid 60x60: nnz(B)=%lld nnz(L+U)=%lld (fill x%.1f, dense would be x%.0f) %.3f s\n",
              lu.stats().nnz_basis, lu.stats().nnz_l + lu.stats().nnz_u, fill,
              double(m) * m / lu.stats().nnz_basis, lu.stats().factor_seconds);
  std::mt19937 rng(7);
  expect_accurate(B, rng, 1e-12);
}

TEST(bases_from_afiro) {
  const auto r = read_mps_file(kData + "/netlib/afiro.mps");
  const SparseMatrixCSC& A = r.problem.A;
  const int m = A.num_rows, n = A.num_cols;
  std::mt19937 rng(8);
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<int> vars(n + m);
    for (int v = 0; v < n + m; ++v) vars[v] = v;
    std::shuffle(vars.begin(), vars.end(), rng);
    std::vector<int> basic(vars.begin(), vars.begin() + m);
    SparseLU lu;
    lu.factorize(build_basis_matrix(A, basic));
    for (auto [pos, row] : lu.replaced_columns()) basic[pos] = n + row;
    const SparseMatrixCSC B = build_basis_matrix(A, basic);
    const std::vector<double> b = random_vector(m, rng);
    std::vector<double> x = b;
    lu.ftran(x);
    EXPECT_TRUE(ftran_residual(B, x, b) < 1e-13);
    std::vector<double> y = b;
    lu.btran(y);
    EXPECT_TRUE(btran_residual(B, y, b) < 1e-13);
  }
}

TEST_MAIN()
