// csc.h - compressed sparse column matrix for the simplex / sparse LU layer.
//
// Origin: gpuopt (Shivanshu Vats, shivanshu24-code/gpu_optimization@c192dd0),
// include/gpuopt/problem.hpp. The LpProblem part of that file was replaced by the
// canonical ps26119::Model (include/ps26119/model.h); the matrix type stayed.
//
// This is a linear-algebra object (a basis matrix, a scaled copy of A), not a
// second model representation. csc_from_model() copies A out of a Model.
#pragma once

#include <vector>

#include "ps26119/model.h"

namespace ps26119::la {

// Compressed sparse column (CSC) matrix.
// The entries of column j live at positions [col_start[j], col_start[j+1])
// of row_index / value. Row indices inside a column are sorted and unique.
struct SparseMatrixCSC {
  int num_rows = 0;
  int num_cols = 0;
  std::vector<int> col_start{0};
  std::vector<int> row_index;
  std::vector<double> value;

  int nnz() const { return static_cast<int>(value.size()); }

  // Dense row-major copy (num_rows x num_cols). Only for small problems.
  std::vector<double> to_dense() const;

  // y = A x
  void multiply(const std::vector<double>& x, std::vector<double>& y) const;
  // y = A^T x
  void multiply_transpose(const std::vector<double>& x, std::vector<double>& y) const;
};

struct Triplet {
  int row;
  int col;
  double value;
};

// Builds a CSC matrix from (row, col, value) triplets. Duplicate (row, col)
// pairs are summed; entries that become exactly zero are dropped.
// If merged_duplicates is non-null it receives the number of merged pairs.
SparseMatrixCSC build_csc(int num_rows, int num_cols, std::vector<Triplet> triplets,
                          int* merged_duplicates = nullptr);

// Copy of the model's constraint matrix. Model::validate() guarantees unique row
// indices per column; they need not be sorted (no user of this type relies on it
// except build_csc's output, which is sorted).
SparseMatrixCSC csc_from_model(const Model& model);

}  // namespace ps26119::la
