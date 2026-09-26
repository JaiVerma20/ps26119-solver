// csc.cpp - see csc.h. Origin: gpuopt src/core/problem.cpp (Shivanshu Vats, c192dd0).
#include "la/csc.h"

#include <algorithm>
#include <cmath>

namespace ps26119::la {

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

SparseMatrixCSC csc_from_model(const Model& model) {
  SparseMatrixCSC m;
  m.num_rows = model.num_rows;
  m.num_cols = model.num_cols;
  m.col_start = model.col_start;
  m.row_index = model.row_index;
  m.value = model.value;
  return m;
}

}  // namespace ps26119::la
