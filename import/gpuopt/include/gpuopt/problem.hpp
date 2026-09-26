// problem.hpp - the canonical in-memory LP / MILP model.
//
// This struct is the frozen contract between the front end (MPS reader,
// presolve) and every solver engine (dense oracle, sparse simplex, IPM, PDLP).
//
//   minimize / maximize   obj^T x + obj_offset
//   subject to            row_lower <= A x <= row_upper
//                         col_lower <=   x <= col_upper
//                         x_j integer          for every j with is_integer[j]
//
// Infinite bounds are stored as +/- kInf. Equality rows have row_lower == row_upper.
#pragma once

#include <limits>
#include <string>
#include <vector>

namespace gpuopt {

inline constexpr double kInf = std::numeric_limits<double>::infinity();

enum class ObjSense { kMinimize, kMaximize };

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

struct LpProblem {
  std::string name;
  std::string objective_name;
  ObjSense sense = ObjSense::kMinimize;
  double obj_offset = 0.0;

  std::vector<double> obj;  // size num_cols
  SparseMatrixCSC A;        // num_rows x num_cols
  std::vector<double> row_lower, row_upper;
  std::vector<double> col_lower, col_upper;
  std::vector<char> is_integer;  // size num_cols, 0 or 1

  std::vector<std::string> row_names;
  std::vector<std::string> col_names;

  int num_rows() const { return A.num_rows; }
  int num_cols() const { return A.num_cols; }
  int num_integers() const;

  // obj^T x + obj_offset, in the problem's own sense.
  double objective_value(const std::vector<double>& x) const;

  // Empty string if every array has a consistent size and no bound is NaN;
  // otherwise a description of the first inconsistency found.
  std::string validate() const;
};

}  // namespace gpuopt
