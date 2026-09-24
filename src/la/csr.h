// csr.h — compressed sparse row matrices and the kernels the CPU engines need.
//
// Model stores A in CSC (col_start/row_index/value), which is exactly the CSR form of Aᵀ.
// The first-order engines need both A x and Aᵀ y as row-parallel products, so we keep
// two CSR matrices: A (m×n) and Aᵀ (n×m). Row-parallel SpMV writes each output entry
// once — deterministic and trivially parallel (and the layout the CUDA kernels use).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ps26119/model.h"

namespace ps26119::la {

template <class T>
struct Csr {
  int rows = 0, cols = 0;
  std::vector<std::int64_t> row_ptr;  // rows+1
  std::vector<int> col;               // nnz
  std::vector<T> val;                 // nnz

  std::int64_t nnz() const { return row_ptr.empty() ? 0 : row_ptr.back(); }

  // out = this · x   (row-parallel; accumulates in Acc)
  template <class Acc = T, class X, class Out>
  void multiply(const X* x, Out* out) const {
#if defined(PS26119_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < rows; ++i) {
      Acc s = 0;
      for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) s += static_cast<Acc>(val[k]) * static_cast<Acc>(x[col[k]]);
      out[i] = static_cast<Out>(s);
    }
  }

  template <class U>
  Csr<U> cast() const {
    Csr<U> r;
    r.rows = rows;
    r.cols = cols;
    r.row_ptr = row_ptr;
    r.col = col;
    r.val.assign(val.begin(), val.end());
    return r;
  }
};

// CSR of Aᵀ straight from the Model's CSC arrays (no copy of structure logic needed).
Csr<double> csr_transpose_from_model(const Model& m);
// CSR of A (m×n) by transposing the CSC.
Csr<double> csr_from_model(const Model& m);
// Transpose of a CSR matrix.
Csr<double> transpose(const Csr<double>& a);

// Largest singular value of A (Lanczos on AᵀA, deterministic start; see csr.cpp for why not
// power iteration). Slightly OVER-estimates (×(1+1e-4)) so step sizes stay stable.
// Returns 0 for an all-zero matrix. `at` must be the transpose of `a`. At most 400 steps.
double estimate_norm2(const Csr<double>& a, const Csr<double>& at, int max_iter = 400, double rel_tol = 1e-12);

}  // namespace ps26119::la
