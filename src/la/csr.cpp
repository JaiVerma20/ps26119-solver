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

double estimate_norm2(const Csr<double>& a, const Csr<double>& at, int max_iter, double rel_tol) {
  const int n = a.cols;
  if (n == 0 || a.nnz() == 0) return 0.0;
  // Deterministic, non-degenerate start: v_j = 1 + (j mod 7)/7.
  std::vector<double> v(n), w(a.rows), u(n);
  double nv = 0;
  for (int j = 0; j < n; ++j) {
    v[j] = 1.0 + static_cast<double>(j % 7) / 7.0;
    nv += v[j] * v[j];
  }
  nv = std::sqrt(nv);
  for (double& e : v) e /= nv;
  double lambda = 0.0;
  for (int it = 0; it < max_iter; ++it) {
    a.multiply(v.data(), w.data());
    at.multiply(w.data(), u.data());
    double nu = 0;
    for (double e : u) nu += e * e;
    nu = std::sqrt(nu);  // ≈ σ_max² once converged
    if (nu == 0.0) return 0.0;
    for (int j = 0; j < n; ++j) v[j] = u[j] / nu;
    if (std::fabs(nu - lambda) <= rel_tol * nu) {
      lambda = nu;
      break;
    }
    lambda = nu;
  }
  return std::sqrt(lambda);
}

}  // namespace ps26119::la
