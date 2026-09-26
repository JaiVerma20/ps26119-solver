// sparse_lu.hpp - Layer 2: sparse LU factorization of a simplex basis.
//
// The revised simplex never inverts the basis matrix B. Instead it keeps a
// factorization  B = L U  (up to row/column permutations) and answers two
// questions every iteration:
//
//   FTRAN   solve  B x = b      (x = column of the entering variable, in
//                                basis-position space)
//   BTRAN   solve  B^T y = d    (y = simplex multipliers, in row space)
//
// Factorization: right-looking Gaussian elimination on the sparse active
// submatrix. Pivots are chosen by the Markowitz criterion (minimise
// (r_i - 1)(c_j - 1), the worst-case fill-in) restricted to entries that pass
// a threshold stability test |a_ij| >= tau * max_k |a_kj|. tau trades
// sparsity (small tau) against numerical stability (large tau).
//
// Updates: after a basis change the factors are updated with the product
// form (PFI): B_new^{-1} = E^{-1} B^{-1}, one eta vector per update, until
// needs_refactor() asks for a fresh factorization.
//
// Rank deficiency: if B is singular, the dependent columns are replaced by
// unit columns e_r of the uncovered rows (i.e. by slack variables), the
// factorization is completed, and replaced_columns() reports each swap so
// the simplex can update its basis accordingly.
#pragma once

#include <utility>
#include <vector>

#include "gpuopt/problem.hpp"

namespace gpuopt {

// Builds the m x basic.size() matrix whose column k is
//   column basic[k] of A          if basic[k] <  A.num_cols   (structural)
//   unit vector e_(basic[k] - n)  if basic[k] >= A.num_cols   (slack / logical)
SparseMatrixCSC build_basis_matrix(const SparseMatrixCSC& A, const std::vector<int>& basic);

struct SparseLUOptions {
  double pivot_threshold = 0.1;     // tau in the threshold test
  double pivot_tolerance = 1e-11;   // |pivot| at or below this counts as zero
  int search_limit = 8;             // Markowitz candidates examined before settling
  int max_updates = 100;            // refactorize after this many PFI updates
  double update_tolerance = 1e-9;   // reject an update whose |alpha_p| is smaller
};

struct SparseLUStats {
  int dim = 0;
  int rank = 0;               // rank found before dependent columns were replaced
  long long nnz_basis = 0;
  long long nnz_l = 0;        // off-diagonal entries of L
  long long nnz_u = 0;        // entries of U including the diagonal
  long long nnz_eta = 0;      // entries in the PFI eta file
  int num_updates = 0;
  double factor_seconds = 0.0;
};

class SparseLU {
 public:
  explicit SparseLU(SparseLUOptions options = {});

  // Factorizes a square matrix B. Returns the rank found; if it is below
  // B.num_rows the factorization is completed with unit columns, see below.
  int factorize(const SparseMatrixCSC& B);

  // (position, row) pairs: column `position` of B was linearly dependent and
  // has been replaced by the unit vector e_row. Empty when B was nonsingular.
  const std::vector<std::pair<int, int>>& replaced_columns() const { return replaced_; }

  // In place: rhs holds b (indexed by row) on entry, x (indexed by basis position) on exit.
  void ftran(std::vector<double>& rhs) const;
  // In place: rhs holds d (indexed by basis position) on entry, y (indexed by row) on exit.
  void btran(std::vector<double>& rhs) const;

  // Replaces the column at basis `position` by a new column a_q, given
  // alpha = B^{-1} a_q from ftran. Returns false (and changes nothing) if
  // |alpha[position]| is too small to pivot on safely.
  bool update(int position, const std::vector<double>& alpha);

  bool needs_refactor() const;
  const SparseLUStats& stats() const { return stats_; }

 private:
  SparseLUOptions opt_;
  SparseLUStats stats_;
  int m_ = 0;

  // Pivot sequence: step k eliminated row prow_[k] using basis position pcol_[k].
  std::vector<int> prow_, pcol_;
  std::vector<double> pval_;

  // L: one column eta per real pivot k < rank: rows l_index_, multipliers l_value_.
  std::vector<int> l_start_, l_index_;
  std::vector<double> l_value_;

  // U without its diagonal, stored twice:
  //   by pivot row (for BTRAN):   entries (basis position, value)
  //   by pivot column (for FTRAN): entries (pivot step k' of the row, value)
  std::vector<int> ur_start_, ur_index_;
  std::vector<double> ur_value_;
  std::vector<int> uc_start_, uc_index_;
  std::vector<double> uc_value_;

  // PFI eta file: eta e pivots on position eta_pos_[e] with value eta_pivot_[e].
  std::vector<int> eta_pos_, eta_start_{0}, eta_index_;
  std::vector<double> eta_pivot_, eta_value_;

  std::vector<std::pair<int, int>> replaced_;
  mutable std::vector<double> work_;  // scratch for ftran / btran (not thread safe)
};

}  // namespace gpuopt
