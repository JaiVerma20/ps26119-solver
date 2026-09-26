// sparse_lu.cpp - Markowitz / threshold sparse LU with FTRAN, BTRAN and PFI updates.
//
// Factorization works on the "active submatrix": the rows and columns not yet
// pivoted. It is stored twice so both directions are cheap:
//   by column  col_rows[j] / col_vals[j]   (indices + values)
//   by row     row_cols[i]                 (indices only)
// Each step k:
//   1. choose a pivot (r, c) by Markowitz cost among threshold-stable entries,
//      searching columns and rows in increasing order of their counts;
//   2. record U row k   = the active part of row r;
//      record L column k = multipliers a_ic / a_rc for the other rows i;
//   3. remove row r and column c from the active submatrix;
//   4. Schur-complement update a_ij -= l_i * u_rj, creating fill-in as needed.

#include "gpuopt/linalg/sparse_lu.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>

namespace gpuopt {

SparseMatrixCSC build_basis_matrix(const SparseMatrixCSC& A, const std::vector<int>& basic) {
  SparseMatrixCSC B;
  B.num_rows = A.num_rows;
  B.num_cols = static_cast<int>(basic.size());
  B.col_start.assign(1, 0);
  for (int v : basic) {
    if (v < A.num_cols) {
      for (int k = A.col_start[v]; k < A.col_start[v + 1]; ++k) {
        B.row_index.push_back(A.row_index[k]);
        B.value.push_back(A.value[k]);
      }
    } else {
      B.row_index.push_back(v - A.num_cols);
      B.value.push_back(1.0);
    }
    B.col_start.push_back(static_cast<int>(B.value.size()));
  }
  return B;
}

namespace {

// Items (rows or columns) kept in doubly linked lists bucketed by their
// current count, so the pivot search can visit the sparsest ones first.
class CountLists {
 public:
  void init(int num_items, int max_count) {
    head_.assign(max_count + 1, -1);
    next_.assign(num_items, -1);
    prev_.assign(num_items, -1);
    count_.assign(num_items, -1);
  }
  void insert(int item, int count) {
    count_[item] = count;
    prev_[item] = -1;
    next_[item] = head_[count];
    if (head_[count] >= 0) prev_[head_[count]] = item;
    head_[count] = item;
  }
  void remove(int item) {
    const int c = count_[item];
    if (c < 0) return;
    if (prev_[item] >= 0) next_[prev_[item]] = next_[item];
    else head_[c] = next_[item];
    if (next_[item] >= 0) prev_[next_[item]] = prev_[item];
    count_[item] = -1;
  }
  void move(int item, int count) {
    remove(item);
    insert(item, count);
  }
  int first(int count) const { return head_[count]; }
  int next(int item) const { return next_[item]; }

 private:
  std::vector<int> head_, next_, prev_, count_;
};

void erase_value(std::vector<int>& v, int x) {
  auto it = std::find(v.begin(), v.end(), x);
  if (it != v.end()) {
    *it = v.back();
    v.pop_back();
  }
}

}  // namespace

SparseLU::SparseLU(SparseLUOptions options) : opt_(options) {}

int SparseLU::factorize(const SparseMatrixCSC& B) {
  const auto start = std::chrono::steady_clock::now();
  const int m = B.num_rows;
  m_ = m;
  stats_ = {};
  stats_.dim = m;
  stats_.nnz_basis = B.nnz();
  replaced_.clear();
  prow_.clear();
  pcol_.clear();
  pval_.clear();
  l_start_.assign(1, 0);
  l_index_.clear();
  l_value_.clear();
  eta_pos_.clear();
  eta_start_.assign(1, 0);
  eta_index_.clear();
  eta_pivot_.clear();
  eta_value_.clear();

  // ---- load the active submatrix
  std::vector<std::vector<int>> col_rows(m), row_cols(m);
  std::vector<std::vector<double>> col_vals(m);
  for (int j = 0; j < m; ++j) {
    for (int k = B.col_start[j]; k < B.col_start[j + 1]; ++k) {
      if (B.value[k] == 0.0) continue;
      col_rows[j].push_back(B.row_index[k]);
      col_vals[j].push_back(B.value[k]);
      row_cols[B.row_index[k]].push_back(j);
    }
  }
  CountLists col_lists, row_lists;
  col_lists.init(m, m);
  row_lists.init(m, m);
  for (int j = 0; j < m; ++j) col_lists.insert(j, static_cast<int>(col_rows[j].size()));
  for (int i = 0; i < m; ++i) row_lists.insert(i, static_cast<int>(row_cols[i].size()));

  std::vector<double> col_max(m, 0.0);
  std::vector<char> col_max_valid(m, 0);
  auto column_max = [&](int j) {
    if (!col_max_valid[j]) {
      double v = 0.0;
      for (double a : col_vals[j]) v = std::max(v, std::fabs(a));
      col_max[j] = v;
      col_max_valid[j] = 1;
    }
    return col_max[j];
  };
  auto entry = [&](int i, int j) {
    const auto& rows = col_rows[j];
    for (size_t t = 0; t < rows.size(); ++t) {
      if (rows[t] == i) return col_vals[j][t];
    }
    return 0.0;
  };

  // Scratch for U rows (pivot-position index, value) collected per step.
  std::vector<std::vector<std::pair<int, double>>> urows;
  urows.reserve(m);
  std::vector<int> mark(m, -1);
  std::vector<char> row_done(m, 0), col_done(m, 0);

  // ---- Markowitz pivot search
  auto find_pivot = [&](int& best_r, int& best_c) {
    best_r = best_c = -1;
    long long best_cost = LLONG_MAX;
    double best_abs = 0.0;
    int searched = 0;
    auto consider = [&](int i, int j, double a, long long cost) {
      if (cost < best_cost || (cost == best_cost && a > best_abs)) {
        best_r = i;
        best_c = j;
        best_cost = cost;
        best_abs = a;
      }
    };
    for (int cnt = 1; cnt <= m; ++cnt) {
      // Any candidate with count >= cnt costs at least (cnt-1)^2.
      if (best_r >= 0 && best_cost <= static_cast<long long>(cnt - 1) * (cnt - 1)) return;

      for (int j = col_lists.first(cnt); j >= 0; j = col_lists.next(j)) {
        const double cmax = column_max(j);
        if (cmax > opt_.pivot_tolerance) {
          for (size_t t = 0; t < col_rows[j].size(); ++t) {
            const double a = std::fabs(col_vals[j][t]);
            if (a < opt_.pivot_threshold * cmax || a <= opt_.pivot_tolerance) continue;
            const int i = col_rows[j][t];
            consider(i, j, a, static_cast<long long>(row_cols[i].size() - 1) * (cnt - 1));
          }
        }
        if (best_r >= 0 && ++searched >= opt_.search_limit) return;
      }

      for (int i = row_lists.first(cnt); i >= 0; i = row_lists.next(i)) {
        for (int j : row_cols[i]) {
          const double cmax = column_max(j);
          const double a = std::fabs(entry(i, j));
          if (a < opt_.pivot_threshold * cmax || a <= opt_.pivot_tolerance) continue;
          consider(i, j, a, static_cast<long long>(cnt - 1) * (col_rows[j].size() - 1));
        }
        if (best_r >= 0 && ++searched >= opt_.search_limit) return;
      }
    }
  };

  // ---- elimination
  for (int step = 0; step < m; ++step) {
    int r, c;
    find_pivot(r, c);
    if (r < 0) break;  // no acceptable pivot left: B is (numerically) singular

    const double piv = entry(r, c);
    prow_.push_back(r);
    pcol_.push_back(c);
    pval_.push_back(piv);

    // U row: active part of row r, without the pivot.
    std::vector<std::pair<int, double>> urow;
    urow.reserve(row_cols[r].size());
    for (int j : row_cols[r]) {
      if (j != c) urow.push_back({j, entry(r, j)});
    }
    // L column: multipliers for the other active rows of column c.
    const int l_begin = static_cast<int>(l_index_.size());
    for (size_t t = 0; t < col_rows[c].size(); ++t) {
      const int i = col_rows[c][t];
      if (i == r) continue;
      l_index_.push_back(i);
      l_value_.push_back(col_vals[c][t] / piv);
    }
    const int l_end = static_cast<int>(l_index_.size());
    l_start_.push_back(l_end);

    // Remove row r and column c from the active submatrix.
    for (int j : row_cols[r]) {
      if (j == c) continue;
      auto& rows = col_rows[j];
      for (size_t t = 0; t < rows.size(); ++t) {
        if (rows[t] == r) {
          rows[t] = rows.back();
          rows.pop_back();
          col_vals[j][t] = col_vals[j].back();
          col_vals[j].pop_back();
          break;
        }
      }
    }
    for (int i : col_rows[c]) {
      if (i != r) erase_value(row_cols[i], c);
    }
    col_rows[c].clear();
    col_vals[c].clear();
    row_cols[r].clear();
    col_lists.remove(c);
    row_lists.remove(r);
    row_done[r] = col_done[c] = 1;

    // Schur complement update, column by column of the pivot row.
    for (const auto& [j, u] : urow) {
      auto& rows = col_rows[j];
      auto& vals = col_vals[j];
      for (size_t t = 0; t < rows.size(); ++t) mark[rows[t]] = static_cast<int>(t);
      for (int k = l_begin; k < l_end; ++k) {
        const int i = l_index_[k];
        const double delta = -l_value_[k] * u;
        if (mark[i] >= 0) {
          vals[mark[i]] += delta;
        } else {
          rows.push_back(i);  // fill-in
          vals.push_back(delta);
          row_cols[i].push_back(j);
        }
      }
      for (int i : rows) mark[i] = -1;
      col_max_valid[j] = 0;
      col_lists.move(j, static_cast<int>(rows.size()));
    }
    for (int k = l_begin; k < l_end; ++k) {
      const int i = l_index_[k];
      row_lists.move(i, static_cast<int>(row_cols[i].size()));
    }
    urows.push_back(std::move(urow));
  }

  const int rank = static_cast<int>(prow_.size());
  stats_.rank = rank;

  // ---- complete a singular factorization with unit columns
  if (rank < m) {
    std::vector<int> free_rows, free_cols;
    for (int i = 0; i < m; ++i) {
      if (!row_done[i]) free_rows.push_back(i);
    }
    for (int j = 0; j < m; ++j) {
      if (!col_done[j]) free_cols.push_back(j);
    }
    std::vector<char> is_replaced(m, 0);
    for (size_t t = 0; t < free_cols.size(); ++t) {
      replaced_.push_back({free_cols[t], free_rows[t]});
      is_replaced[free_cols[t]] = 1;
      prow_.push_back(free_rows[t]);
      pcol_.push_back(free_cols[t]);
      pval_.push_back(1.0);
      urows.emplace_back();
    }
    // The replaced columns no longer contain their old entries.
    for (auto& urow : urows) {
      urow.erase(std::remove_if(urow.begin(), urow.end(),
                                [&](const auto& e) { return is_replaced[e.first]; }),
                 urow.end());
    }
  }

  // ---- U by row (BTRAN) and by column (FTRAN)
  std::vector<int> step_of_col(m);
  for (int k = 0; k < m; ++k) step_of_col[pcol_[k]] = k;
  ur_start_.assign(1, 0);
  ur_index_.clear();
  ur_value_.clear();
  std::vector<int> col_count(m + 1, 0);
  for (int k = 0; k < m; ++k) {
    for (const auto& [j, u] : urows[k]) {
      ur_index_.push_back(j);
      ur_value_.push_back(u);
      ++col_count[step_of_col[j] + 1];
    }
    ur_start_.push_back(static_cast<int>(ur_index_.size()));
  }
  uc_start_.assign(m + 1, 0);
  for (int k = 0; k < m; ++k) uc_start_[k + 1] = uc_start_[k] + col_count[k + 1];
  uc_index_.assign(ur_index_.size(), 0);
  uc_value_.assign(ur_index_.size(), 0.0);
  std::vector<int> fill_pos(uc_start_.begin(), uc_start_.end() - 1);
  for (int k = 0; k < m; ++k) {
    for (int t = ur_start_[k]; t < ur_start_[k + 1]; ++t) {
      const int dest = fill_pos[step_of_col[ur_index_[t]]]++;
      uc_index_[dest] = k;
      uc_value_[dest] = ur_value_[t];
    }
  }

  stats_.nnz_l = static_cast<long long>(l_index_.size());
  stats_.nnz_u = static_cast<long long>(ur_index_.size()) + m;
  stats_.factor_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  work_.assign(m, 0.0);
  return rank;
}

void SparseLU::ftran(std::vector<double>& rhs) const {
  const int rank = static_cast<int>(l_start_.size()) - 1;
  // L^{-1}: apply the column etas in pivot order (row space).
  for (int k = 0; k < rank; ++k) {
    const double v = rhs[prow_[k]];
    if (v == 0.0) continue;
    for (int t = l_start_[k]; t < l_start_[k + 1]; ++t) rhs[l_index_[t]] -= l_value_[t] * v;
  }
  // U^{-1}: back substitution by columns, result in position space.
  std::fill(work_.begin(), work_.end(), 0.0);
  for (int k = m_ - 1; k >= 0; --k) {
    const double x = rhs[prow_[k]] / pval_[k];
    work_[pcol_[k]] = x;
    if (x == 0.0) continue;
    for (int t = uc_start_[k]; t < uc_start_[k + 1]; ++t) {
      rhs[prow_[uc_index_[t]]] -= uc_value_[t] * x;
    }
  }
  // PFI etas, oldest first.
  for (size_t e = 0; e < eta_pos_.size(); ++e) {
    const int p = eta_pos_[e];
    const double xp = work_[p] / eta_pivot_[e];
    work_[p] = xp;
    if (xp == 0.0) continue;
    for (int t = eta_start_[e]; t < eta_start_[e + 1]; ++t) work_[eta_index_[t]] -= eta_value_[t] * xp;
  }
  rhs.swap(work_);
}

void SparseLU::btran(std::vector<double>& rhs) const {
  const int rank = static_cast<int>(l_start_.size()) - 1;
  // PFI etas transposed, newest first (position space).
  for (size_t e = eta_pos_.size(); e-- > 0;) {
    const int p = eta_pos_[e];
    double s = rhs[p];
    for (int t = eta_start_[e]; t < eta_start_[e + 1]; ++t) s -= eta_value_[t] * rhs[eta_index_[t]];
    rhs[p] = s / eta_pivot_[e];
  }
  // U^{-T}: forward substitution by rows, result in row space.
  std::fill(work_.begin(), work_.end(), 0.0);
  for (int k = 0; k < m_; ++k) {
    const double w = rhs[pcol_[k]] / pval_[k];
    work_[prow_[k]] = w;
    if (w == 0.0) continue;
    for (int t = ur_start_[k]; t < ur_start_[k + 1]; ++t) rhs[ur_index_[t]] -= ur_value_[t] * w;
  }
  // L^{-T}: column etas transposed, in reverse pivot order.
  for (int k = rank - 1; k >= 0; --k) {
    double s = work_[prow_[k]];
    for (int t = l_start_[k]; t < l_start_[k + 1]; ++t) s -= l_value_[t] * work_[l_index_[t]];
    work_[prow_[k]] = s;
  }
  rhs.swap(work_);
}

bool SparseLU::update(int position, const std::vector<double>& alpha) {
  const double piv = alpha[position];
  if (std::fabs(piv) < opt_.update_tolerance) return false;
  eta_pos_.push_back(position);
  eta_pivot_.push_back(piv);
  for (int i = 0; i < m_; ++i) {
    if (i != position && alpha[i] != 0.0) {
      eta_index_.push_back(i);
      eta_value_.push_back(alpha[i]);
    }
  }
  eta_start_.push_back(static_cast<int>(eta_index_.size()));
  stats_.nnz_eta = static_cast<long long>(eta_index_.size());
  ++stats_.num_updates;
  return true;
}

bool SparseLU::needs_refactor() const {
  return stats_.num_updates >= opt_.max_updates ||
         stats_.nnz_eta > 2 * (stats_.nnz_l + stats_.nnz_u) + m_;
}

}  // namespace gpuopt
