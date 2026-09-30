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

#include "la/parallel.h"
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
  // Row-parallel over the shared thread pool (la/parallel.h); each out[i] is computed by one
  // thread in a fixed order, so the result does not depend on the thread count.
  // Work is split by nnz + rows, not by rows: a wide matrix with few long rows (Kennington
  // osa-*: 1–10k rows of ~130 entries; fit2d: 25 rows of ~5000) used to get one or two chunks
  // and ran A x on a single thread whatever the thread count.
  template <class Acc = T, class X, class Out>
  void multiply(const X* x, Out* out) const {
    auto body = [&](std::int64_t b, std::int64_t e) {
      for (std::int64_t i = b; i < e; ++i) {
        Acc s = 0;
        for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
          s += static_cast<Acc>(val[k]) * static_cast<Acc>(x[col[k]]);
        out[i] = static_cast<Out>(s);
      }
    };
    for_rows(body);
  }

  // Runs body(b, e) over row ranges covering [0, rows), balanced by nnz + rows over the thread
  // pool (one range per chunk; each row in exactly one range). Used by multiply() and by fused
  // kernels that consume a row's product immediately (pdhg/cpu_backend.cpp).
  template <class Body>
  void for_rows(Body&& body) const {
    ThreadPool& pool = ThreadPool::instance();
    const std::int64_t work = nnz() + rows;  // cost model: one unit per entry and per row
    constexpr std::int64_t kGrain = 32768;   // minimum work per chunk
    const int t = pool.threads();
    if (t <= 1 || work < 2 * kGrain || rows < 2) {
      body(std::int64_t{0}, static_cast<std::int64_t>(rows));
      return;
    }
    const int chunks = static_cast<int>(std::min<std::int64_t>({4 * t, work / kGrain, rows}));
    pool.run(chunks, [&](int c) { body(balanced_row(work * c / chunks), balanced_row(work * (c + 1) / chunks)); });
  }

  // First row i with row_ptr[i] + i >= target (row_ptr[i] + i is strictly increasing).
  std::int64_t balanced_row(std::int64_t target) const {
    std::int64_t lo = 0, hi = rows;
    while (lo < hi) {
      const std::int64_t mid = lo + (hi - lo) / 2;
      if (row_ptr[mid] + mid < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
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
