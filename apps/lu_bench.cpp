// lu_bench.cpp - measures the sparse LU on bases taken from a real LP.
//
// A simplex basis consists of m columns: some structural columns of A, the
// rest slack (unit) columns. For each basis we report
//   rank       found by the factorization (dependent columns are replaced by slacks)
//   fill       nnz(L + U) / nnz(B)
//   residuals  ||B x - b|| and ||B^T y - d||, relative to ||B|| ||x|| + ||b||
//   timings    factorization, FTRAN of a column of A, BTRAN of a unit vector
// plus an update test: a chain of PFI updates checked against a fresh factorization.

#include "lu_bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "gpuopt/linalg/sparse_lu.hpp"

namespace gpuopt {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

double inf_norm_rows(const SparseMatrixCSC& B) {
  std::vector<double> s(B.num_rows, 0.0);
  for (int k = 0; k < B.nnz(); ++k) s[B.row_index[k]] += std::fabs(B.value[k]);
  double v = 0;
  for (double x : s) v = std::max(v, x);
  return v;
}

double ftran_residual(const SparseMatrixCSC& B, const std::vector<double>& x, const std::vector<double>& b) {
  std::vector<double> bx;
  B.multiply(x, bx);
  double r = 0, nx = 0, nb = 0;
  for (size_t i = 0; i < b.size(); ++i) {
    r = std::max(r, std::fabs(bx[i] - b[i]));
    nb = std::max(nb, std::fabs(b[i]));
  }
  for (double v : x) nx = std::max(nx, std::fabs(v));
  return r / (inf_norm_rows(B) * nx + nb + 1e-300);
}

double btran_residual(const SparseMatrixCSC& B, const std::vector<double>& y, const std::vector<double>& d) {
  std::vector<double> bty;
  B.multiply_transpose(y, bty);
  double r = 0, ny = 0, nd = 0, nB = 0;
  for (int j = 0; j < B.num_cols; ++j) {
    double s = 0;
    for (int k = B.col_start[j]; k < B.col_start[j + 1]; ++k) s += std::fabs(B.value[k]);
    nB = std::max(nB, s);
    r = std::max(r, std::fabs(bty[j] - d[j]));
    nd = std::max(nd, std::fabs(d[j]));
  }
  for (double v : y) ny = std::max(ny, std::fabs(v));
  return r / (nB * ny + nd + 1e-300);
}

// Lower bound on the condition number: ||B|| ||x|| / ||b|| for x = B^{-1} b.
// Near 1e16 the basis is numerically singular in double precision.
double condition_estimate(const SparseMatrixCSC& B, const std::vector<double>& x, const std::vector<double>& b) {
  double nx = 0, nb = 0;
  for (double v : x) nx = std::max(nx, std::fabs(v));
  for (double v : b) nb = std::max(nb, std::fabs(v));
  return inf_norm_rows(B) * nx / std::max(nb, 1e-300);
}

std::vector<double> random_vector(int m, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(m);
  for (double& x : v) x = u(rng);
  return v;
}

// Textbook dense LU with partial pivoting, timed only (the comparison point).
// It does the full O(m^3) work: no zero-skipping, which would make it
// secretly a sparse method.
double dense_lu_seconds(const SparseMatrixCSC& B) {
  const int m = B.num_rows;
  std::vector<double> a = B.to_dense();
  const auto t = Clock::now();
  for (int k = 0; k < m; ++k) {
    int p = k;
    for (int i = k + 1; i < m; ++i) {
      if (std::fabs(a[static_cast<size_t>(i) * m + k]) > std::fabs(a[static_cast<size_t>(p) * m + k])) p = i;
    }
    if (a[static_cast<size_t>(p) * m + k] == 0.0) continue;
    if (p != k) {
      for (int j = 0; j < m; ++j) std::swap(a[static_cast<size_t>(k) * m + j], a[static_cast<size_t>(p) * m + j]);
    }
    const double* rk = &a[static_cast<size_t>(k) * m];
    for (int i = k + 1; i < m; ++i) {
      double* ri = &a[static_cast<size_t>(i) * m];
      const double f = ri[k] / rk[k];
      for (int j = k + 1; j < m; ++j) ri[j] -= f * rk[j];
    }
  }
  return seconds_since(t);
}

// m distinct variables: round(frac * m) random structural columns (at most n),
// the remaining positions filled with random slacks.
std::vector<int> random_basis(int m, int n, double frac, std::mt19937& rng) {
  std::vector<int> structurals(n), slacks(m);
  for (int j = 0; j < n; ++j) structurals[j] = j;
  for (int i = 0; i < m; ++i) slacks[i] = n + i;
  std::shuffle(structurals.begin(), structurals.end(), rng);
  std::shuffle(slacks.begin(), slacks.end(), rng);
  const int ns = std::min(n, static_cast<int>(std::lround(frac * m)));
  std::vector<int> basic(structurals.begin(), structurals.begin() + ns);
  basic.insert(basic.end(), slacks.begin(), slacks.begin() + (m - ns));
  std::shuffle(basic.begin(), basic.end(), rng);
  return basic;
}

struct BasisReport {
  bool ok = true;
  double worst_residual = 0.0;
};

// Factorizes one basis (completing it if singular) and prints one table row.
BasisReport measure_basis(const char* label, const SparseMatrixCSC& A, std::vector<int>& basic,
                          std::mt19937& rng, SparseLU& lu) {
  const int m = A.num_rows, n = A.num_cols;
  lu.factorize(build_basis_matrix(A, basic));
  for (auto [pos, row] : lu.replaced_columns()) basic[pos] = n + row;
  const SparseMatrixCSC B = build_basis_matrix(A, basic);  // what was actually factorized
  const SparseLUStats& s = lu.stats();

  const std::vector<double> b = random_vector(m, rng);
  std::vector<double> x = b;
  lu.ftran(x);
  const double rf = ftran_residual(B, x, b);
  std::vector<double> y = b;
  lu.btran(y);
  const double rb = btran_residual(B, y, b);

  // Realistic solve timings: FTRAN of structural columns, BTRAN of unit vectors.
  const int reps = 50;
  std::vector<double> v(m);
  auto t = Clock::now();
  for (int r = 0; r < reps; ++r) {
    std::fill(v.begin(), v.end(), 0.0);
    const int q = n > 0 ? static_cast<int>(rng() % n) : 0;
    if (n > 0) {
      for (int k = A.col_start[q]; k < A.col_start[q + 1]; ++k) v[A.row_index[k]] = A.value[k];
    } else {
      v[0] = 1.0;
    }
    lu.ftran(v);
  }
  const double ftran_us = 1e6 * seconds_since(t) / reps;
  t = Clock::now();
  for (int r = 0; r < reps; ++r) {
    std::fill(v.begin(), v.end(), 0.0);
    v[rng() % m] = 1.0;
    lu.btran(v);
  }
  const double btran_us = 1e6 * seconds_since(t) / reps;

  const double fill = double(s.nnz_l + s.nnz_u) / std::max(1, B.nnz());
  const bool ok = rf <= 1e-9 && rb <= 1e-9;
  std::printf("  %-16s %6d/%-6d %9d %10lld %6.2f %8.0e %10.2f   %8.1e %8.1e %9.1f %9.1f  %s\n", label,
              s.rank, m, B.nnz(), s.nnz_l + s.nnz_u, fill, condition_estimate(B, x, b),
              1e3 * s.factor_seconds, rf, rb, ftran_us, btran_us, ok ? "PASS" : "FAIL");
  return {ok, std::max(rf, rb)};
}

}  // namespace

int run_lu_bench(const LpProblem& lp, const LuBenchOptions& opt) {
  const SparseMatrixCSC& A = lp.A;
  const int m = A.num_rows, n = A.num_cols;
  std::mt19937 rng(opt.seed);
  bool all_ok = true;
  double worst = 0.0;

  std::printf("layer 2 sparse LU benchmark  (Markowitz + threshold pivoting, tau = %.2f)\n",
              SparseLUOptions{}.pivot_threshold);
  std::printf("  bases of size m = %d drawn from %d structural + %d slack columns\n\n", m, n, m);
  std::printf("  %-16s %13s %9s %10s %6s %8s %10s   %8s %8s %9s %9s\n", "basis", "rank", "nnz(B)",
              "nnz(L+U)", "fill", "cond >=", "factor ms", "ftran", "btran", "ftran us", "btran us");
  std::printf("  %-16s %13s %9s %10s %6s %8s %10s   %8s %8s\n", "", "", "", "", "", "", "", "resid",
              "resid");

  SparseLU lu;
  std::vector<int> basic(m);
  for (int i = 0; i < m; ++i) basic[i] = n + i;
  BasisReport rep = measure_basis("all slack", A, basic, rng, lu);
  all_ok &= rep.ok;
  worst = std::max(worst, rep.worst_residual);

  std::vector<int> densest_basis;
  for (double frac : {0.5, 0.8, 1.0}) {
    for (int t = 0; t < opt.trials; ++t) {
      basic = random_basis(m, n, frac, rng);
      char label[32];
      std::snprintf(label, sizeof label, "%3.0f%% struct #%d", 100 * frac, t + 1);
      rep = measure_basis(label, A, basic, rng, lu);
      all_ok &= rep.ok;
      worst = std::max(worst, rep.worst_residual);
      if (frac == 1.0 && t == 0) densest_basis = basic;
    }
  }
  std::printf("\n  rank < m means the random columns were linearly dependent; they were\n"
              "  replaced by slack columns, exactly as the simplex crash procedure will do.\n");

  // ---- update test: a chain of PFI updates vs a fresh factorization.
  // Like a real simplex, the residual is checked every kCheckEvery updates and
  // the basis is refactorized when it has drifted above kRefactorResidual.
  if (n > 0) {
    constexpr int kCheckEvery = 10;
    constexpr double kRefactorResidual = 1e-11;
    auto refactorize = [&]() {
      lu.factorize(build_basis_matrix(A, basic));
      for (auto [pos, row] : lu.replaced_columns()) basic[pos] = n + row;
    };
    basic = random_basis(m, n, 0.8, rng);
    refactorize();
    std::vector<char> in_basis(n + m, 0);
    for (int v : basic) in_basis[v] = 1;
    int done = 0, rejected = 0, scheduled = 0, residual_triggered = 0;
    const auto t = Clock::now();
    for (int attempt = 0; attempt < 20 * opt.updates && done < opt.updates; ++attempt) {
      const int q = static_cast<int>(rng() % n);
      if (in_basis[q]) continue;
      std::vector<double> alpha(m, 0.0);
      for (int k = A.col_start[q]; k < A.col_start[q + 1]; ++k) alpha[A.row_index[k]] = A.value[k];
      lu.ftran(alpha);
      // Leaving position: largest |alpha_p| (what a well-tuned ratio test tends to prefer).
      int p = 0;
      for (int i = 1; i < m; ++i) {
        if (std::fabs(alpha[i]) > std::fabs(alpha[p])) p = i;
      }
      if (!lu.update(p, alpha)) {
        ++rejected;
        continue;
      }
      in_basis[basic[p]] = 0;
      in_basis[q] = 1;
      basic[p] = q;
      ++done;
      bool refactor = false;
      if (lu.needs_refactor()) {  // update count or eta file size limit
        refactor = true;
        ++scheduled;
      } else if (done % kCheckEvery == 0) {
        const std::vector<double> probe = random_vector(m, rng);
        std::vector<double> xp = probe;
        lu.ftran(xp);
        if (ftran_residual(build_basis_matrix(A, basic), xp, probe) > kRefactorResidual) {
          refactor = true;
          ++residual_triggered;
        }
      }
      if (refactor) {
        refactorize();  // may swap dependent columns for slacks
        std::fill(in_basis.begin(), in_basis.end(), 0);
        for (int v : basic) in_basis[v] = 1;
      }
    }
    const double update_s = seconds_since(t);
    const SparseMatrixCSC B = build_basis_matrix(A, basic);
    const std::vector<double> b = random_vector(m, rng);
    std::vector<double> x_upd = b;
    lu.ftran(x_upd);
    const double resid = ftran_residual(B, x_upd, b);
    SparseLU fresh;
    fresh.factorize(B);
    std::vector<double> x_new = b;
    fresh.ftran(x_new);
    double diff = 0, scale = 1;
    for (int i = 0; i < m; ++i) {
      diff = std::max(diff, std::fabs(x_new[i] - x_upd[i]));
      scale = std::max(scale, std::fabs(x_new[i]));
    }
    // Pass/fail on the backward error (residual); the forward difference to a
    // fresh factorization is informative only, it scales with the condition number.
    const bool ok = resid <= 1e-9 && fresh.replaced_columns().empty();
    all_ok &= ok;
    std::printf("\n  update test: %d PFI updates (%d rejected as unstable), %.2f ms each incl. FTRAN\n", done,
                rejected, done ? 1e3 * update_s / done : 0.0);
    std::printf("    refactorizations: %d scheduled (eta file limit), %d triggered by the residual check\n"
                "    (residual checked every %d updates, limit %.0e)\n",
                scheduled, residual_triggered, kCheckEvery, kRefactorResidual);
    std::printf("    final residual %.1e, difference to fresh factorization %.1e  %s\n", resid, diff / scale,
                ok ? "PASS" : "FAIL");
  }

  // ---- dense comparison (only where it is affordable)
  if (!densest_basis.empty() && m >= 200 && m <= 2500) {
    SparseLU timing;
    timing.factorize(build_basis_matrix(A, densest_basis));
    const double dense_s = dense_lu_seconds(build_basis_matrix(A, densest_basis));
    std::printf("\n  dense LU on the same basis: %.2f ms  vs sparse LU %.2f ms  (%.0fx), dense storage %.1f MB\n",
                1e3 * dense_s, 1e3 * timing.stats().factor_seconds,
                dense_s / std::max(1e-9, timing.stats().factor_seconds), 8.0 * m * m / 1e6);
  } else if (m > 2500) {
    std::printf("\n  dense LU skipped: it would need %.0f MB and O(m^3) = %.1e flops\n", 8.0 * m * m / 1e6,
                double(m) * m * m / 1.5);
  }

  std::printf("\nLU-BENCH %s  m=%d  worst residual %.1e\n", all_ok ? "PASS" : "FAIL", m, worst);
  return all_ok ? 0 : 3;
}

}  // namespace gpuopt
