// model_builder.h — tiny helper for hand-writing LPs in tests (row-wise, dense-ish).
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ps26119/model.h"

namespace ps26119::test {

// Builds a Model from dense rows. Zero coefficients are dropped.
//   rows[i] = coefficients of row i (size n).
inline Model make_model(const std::vector<double>& c, const std::vector<std::vector<double>>& rows,
                        const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                        const std::vector<double>& col_lower, const std::vector<double>& col_upper,
                        int sense = 1, double offset = 0.0) {
  Model m;
  m.num_cols = static_cast<int>(c.size());
  m.num_rows = static_cast<int>(rows.size());
  m.sense = sense;
  m.obj_offset = offset;
  m.obj = c;
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.col_lower = col_lower;
  m.col_upper = col_upper;
  m.col_start.assign(1, 0);
  for (int j = 0; j < m.num_cols; ++j) {
    for (int i = 0; i < m.num_rows; ++i) {
      if (rows[i][j] != 0.0) {
        m.row_index.push_back(i);
        m.value.push_back(rows[i][j]);
      }
    }
    m.col_start.push_back(static_cast<int>(m.value.size()));
  }
  for (int i = 0; i < m.num_rows; ++i) m.row_names.push_back("r" + std::to_string(i));
  for (int j = 0; j < m.num_cols; ++j) m.col_names.push_back("x" + std::to_string(j));
  return m;
}

}  // namespace ps26119::test
