#include "gpuopt/problem.hpp"

#include <algorithm>
#include <cmath>

namespace gpuopt {

std::vector<double> SparseMatrixCSC::to_dense() const {
  std::vector<double> dense(static_cast<size_t>(num_rows) * num_cols, 0.0);
  for (int j = 0; j < num_cols; ++j) {
    for (int k = col_start[j]; k < col_start[j + 1]; ++k) {
      dense[static_cast<size_t>(row_index[k]) * num_cols + j] = value[k];
    }
  }
  return dense;
}

void SparseMatrixCSC::multiply(const std::vector<double>& x, std::vector<double>& y) const {
  y.assign(num_rows, 0.0);
  for (int j = 0; j < num_cols; ++j) {
    const double xj = x[j];
    if (xj == 0.0) continue;
    for (int k = col_start[j]; k < col_start[j + 1]; ++k) y[row_index[k]] += value[k] * xj;
  }
}

void SparseMatrixCSC::multiply_transpose(const std::vector<double>& x,
                                         std::vector<double>& y) const {
  y.assign(num_cols, 0.0);
  for (int j = 0; j < num_cols; ++j) {
    double sum = 0.0;
    for (int k = col_start[j]; k < col_start[j + 1]; ++k) sum += value[k] * x[row_index[k]];
    y[j] = sum;
  }
}

SparseMatrixCSC build_csc(int num_rows, int num_cols, std::vector<Triplet> triplets,
                          int* merged_duplicates) {
  std::sort(triplets.begin(), triplets.end(), [](const Triplet& a, const Triplet& b) {
    return a.col != b.col ? a.col < b.col : a.row < b.row;
  });

  SparseMatrixCSC m;
  m.num_rows = num_rows;
  m.num_cols = num_cols;
  m.col_start.assign(num_cols + 1, 0);
  m.row_index.reserve(triplets.size());
  m.value.reserve(triplets.size());

  int merged = 0;
  size_t k = 0;
  for (int j = 0; j < num_cols; ++j) {
    m.col_start[j] = static_cast<int>(m.value.size());
    while (k < triplets.size() && triplets[k].col == j) {
      const int row = triplets[k].row;
      double sum = triplets[k].value;
      ++k;
      while (k < triplets.size() && triplets[k].col == j && triplets[k].row == row) {
        sum += triplets[k].value;
        ++merged;
        ++k;
      }
      if (sum != 0.0) {
        m.row_index.push_back(row);
        m.value.push_back(sum);
      }
    }
  }
  m.col_start[num_cols] = static_cast<int>(m.value.size());
  if (merged_duplicates) *merged_duplicates = merged;
  return m;
}

int LpProblem::num_integers() const {
  return static_cast<int>(std::count(is_integer.begin(), is_integer.end(), 1));
}

double LpProblem::objective_value(const std::vector<double>& x) const {
  double value = obj_offset;
  for (int j = 0; j < num_cols(); ++j) value += obj[j] * x[j];
  return value;
}

std::string LpProblem::validate() const {
  const size_t m = static_cast<size_t>(num_rows());
  const size_t n = static_cast<size_t>(num_cols());
  if (A.col_start.size() != n + 1) return "A.col_start has wrong size";
  if (A.row_index.size() != A.value.size()) return "A.row_index / A.value size mismatch";
  if (obj.size() != n) return "obj has wrong size";
  if (col_lower.size() != n || col_upper.size() != n) return "column bounds have wrong size";
  if (row_lower.size() != m || row_upper.size() != m) return "row bounds have wrong size";
  if (is_integer.size() != n) return "is_integer has wrong size";
  for (int r : A.row_index) {
    if (r < 0 || static_cast<size_t>(r) >= m) return "A has a row index out of range";
  }
  for (size_t j = 0; j < n; ++j) {
    if (std::isnan(col_lower[j]) || std::isnan(col_upper[j]) || std::isnan(obj[j])) {
      return "NaN in column data at column " + std::to_string(j);
    }
  }
  for (size_t i = 0; i < m; ++i) {
    if (std::isnan(row_lower[i]) || std::isnan(row_upper[i])) {
      return "NaN in row bounds at row " + std::to_string(i);
    }
  }
  return {};
}

}  // namespace gpuopt
