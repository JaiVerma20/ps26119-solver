// csr.cpp — CSR construction and power iteration.
#include "la/csr.h"

#include <cmath>

namespace ps26119::la {

Csr<double> csr_transpose_from_model(const Model& m) {
  Csr<double> r;
  r.rows = m.num_cols;
  r.cols = m.num_rows;
  r.row_ptr.assign(m.col_start.begin(), m.col_start.end());
  r.col = m.row_index;
  r.val = m.value;
  return r;
}

Csr<double> transpose(const Csr<double>& a) {
  Csr<double> t;
  t.rows = a.cols;
  t.cols = a.rows;
  t.row_ptr.assign(static_cast<std::size_t>(t.rows) + 1, 0);
  for (std::int64_t k = 0; k < a.nnz(); ++k) ++t.row_ptr[a.col[k] + 1];
  for (int i = 0; i < t.rows; ++i) t.row_ptr[i + 1] += t.row_ptr[i];
  t.col.resize(static_cast<std::size_t>(a.nnz()));
  t.val.resize(static_cast<std::size_t>(a.nnz()));
  std::vector<std::int64_t> next(t.row_ptr.begin(), t.row_ptr.end() - 1);
  for (int i = 0; i < a.rows; ++i)  // rows visited in order ⇒ columns of t sorted
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      const std::int64_t p = next[a.col[k]]++;
      t.col[p] = i;
      t.val[p] = a.val[k];
    }
  return t;
}

Csr<double> csr_from_model(const Model& m) { return transpose(csr_transpose_from_model(m)); }

namespace {

// Largest eigenvalue of the symmetric tridiagonal matrix (diag a[0..k), off-diag b[0..k-1))
// by bisection on Sturm sequence counts.
double tridiag_max_eig(const std::vector<double>& a, const std::vector<double>& b, int k) {
  double lo = 1e300, hi = -1e300;
  for (int i = 0; i < k; ++i) {  // Gershgorin interval
    const double r = (i > 0 ? std::fabs(b[i - 1]) : 0.0) + (i + 1 < k ? std::fabs(b[i]) : 0.0);
    lo = std::min(lo, a[i] - r);
    hi = std::max(hi, a[i] + r);
  }
  // count of eigenvalues < x
  auto count_below = [&](double x) {
    int c = 0;
    double d = 1.0;
    for (int i = 0; i < k; ++i) {
      const double bb = i > 0 ? b[i - 1] * b[i - 1] : 0.0;
      d = (a[i] - x) - (i > 0 ? bb / d : 0.0);
      if (d == 0.0) d = -1e-300;
      if (d < 0) ++c;
    }
    return c;
  };
  for (int it = 0; it < 200 && hi - lo > 1e-15 * std::max(1.0, std::fabs(hi)); ++it) {
    const double mid = 0.5 * (lo + hi);
    if (count_below(mid) == k) hi = mid;  // all eigenvalues below mid
    else lo = mid;
  }
  return hi;
}

}  // namespace

// ‖A‖₂ by Lanczos on AᵀA.
//
// History: the first version used plain power iteration stopped when the Rayleigh quotient
// changed by < 1e-6. On matrices whose top singular values cluster it stalled early and
// UNDER-estimated ‖A‖₂ (Netlib scrs8 after scaling: 0.99550 vs the true 0.99877), which
// made η‖A‖ = 1.0013 > 1 and the PDHG iterates diverged. Lanczos converges far faster for
// the extreme eigenvalue of a clustered spectrum (Chebyshev-type rate) at the same cost per
// step (one A·v and one Aᵀ·w). Plain three-term recurrence without reorthogonalisation:
// loss of orthogonality only duplicates converged Ritz values, it does not push the top Ritz
// value above λ_max (up to rounding). The top Ritz value never exceeds λ_max, so we stop
// when it has stagnated and return it inflated by `safety` (relative), which keeps
// τσ‖A‖² < 1 with the engines' 0.998 factor.
// Deterministic: fixed pseudo-random start vector.
double estimate_norm2(const Csr<double>& a, const Csr<double>& at, int max_iter, double rel_tol) {
  const int n = a.cols;
  if (n == 0 || a.nnz() == 0) return 0.0;
  const int K = std::min(std::max(max_iter, 20), 400);
  std::vector<double> v(n), vprev(n, 0.0), w(n), t(a.rows);
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  double nv = 0;
  for (int j = 0; j < n; ++j) {  // xorshift64* → uniform(−1, 1)
    state ^= state >> 12, state ^= state << 25, state ^= state >> 27;
    v[j] = static_cast<double>((state * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0 * 2.0 - 1.0;
    nv += v[j] * v[j];
  }
  nv = std::sqrt(nv);
  for (double& e : v) e /= nv;
  std::vector<double> alpha, beta;
  double theta = 0, theta_old = -1;
  int stagnant = 0;
  for (int k = 0; k < K; ++k) {
    a.multiply(v.data(), t.data());
    at.multiply(t.data(), w.data());
    double al = 0;
    for (int j = 0; j < n; ++j) al += w[j] * v[j];
    const double bprev = beta.empty() ? 0.0 : beta.back();
    double bn = 0;
    for (int j = 0; j < n; ++j) {
      w[j] -= al * v[j] + bprev * vprev[j];
      bn += w[j] * w[j];
    }
    bn = std::sqrt(bn);
    alpha.push_back(al);
    theta = tridiag_max_eig(alpha, beta, static_cast<int>(alpha.size()));
    if (bn <= 1e-14 * std::max(1.0, theta)) break;  // invariant subspace: θ is exact
    beta.push_back(bn);
    if (theta - theta_old <= rel_tol * theta) {
      if (++stagnant >= 5 && k >= 20) break;
    } else {
      stagnant = 0;
    }
    theta_old = theta;
    for (int j = 0; j < n; ++j) {
      vprev[j] = v[j];
      v[j] = w[j] / bn;
    }
  }
  constexpr double safety = 1e-4;  // relative inflation of the estimate
  return std::sqrt(std::max(theta, 0.0)) * (1.0 + safety);
}

}  // namespace ps26119::la
