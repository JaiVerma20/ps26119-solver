// model.cpp — Model invariants, fingerprint, and small helpers.
//
// Fingerprint: 64-bit FNV-1a over a canonical byte stream of every number in the model.
// Canonicalisation: doubles are hashed by bit pattern after mapping -0.0 to +0.0 (so two
// readers that differ only in the sign of a zero agree); all NaNs map to one pattern.
// Names are deliberately excluded — the fingerprint identifies the mathematical model.
// tools/lpm.py implements the identical hash in Python (keep them in sync).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include "ps26119/model.h"

namespace ps26119 {
namespace {

class Fnv1a {
 public:
  void bytes(const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
      h_ ^= b[i];
      h_ *= 0x100000001b3ULL;
    }
  }
  void u64(std::uint64_t v) { bytes(&v, sizeof v); }
  void i64(std::int64_t v) { bytes(&v, sizeof v); }
  void f64(double v) {
    if (v == 0.0) v = 0.0;  // -0.0 -> +0.0
    std::uint64_t bits;
    if (std::isnan(v)) {
      bits = 0x7ff8000000000000ULL;
    } else {
      std::memcpy(&bits, &v, sizeof bits);
    }
    u64(bits);
  }
  void f64s(const std::vector<double>& v) {
    u64(v.size());
    for (double d : v) f64(d);
  }
  std::uint64_t value() const { return h_; }

 private:
  std::uint64_t h_ = 0xcbf29ce484222325ULL;
};

}  // namespace

std::uint64_t Model::fingerprint() const {
  Fnv1a h;
  h.i64(num_rows);
  h.i64(num_cols);
  h.i64(sense);
  h.f64(obj_offset);
  h.f64s(obj);
  h.f64s(col_lower);
  h.f64s(col_upper);
  h.f64s(row_lower);
  h.f64s(row_upper);
  // Matrix: per column, the (row, value) pairs sorted by row, so that two readers that
  // store a column's entries in a different order still agree.
  h.u64(value.size());
  std::vector<std::pair<int, double>> col;
  for (int j = 0; j < num_cols; ++j) {
    col.clear();
    for (int k = col_start[j]; k < col_start[j + 1]; ++k) col.emplace_back(row_index[k], value[k]);
    std::sort(col.begin(), col.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    h.i64(static_cast<std::int64_t>(col.size()));
    for (const auto& [i, v] : col) {
      h.i64(i);
      h.f64(v);
    }
  }
  // Empty integrality == all continuous: hash n zeros either way.
  for (int j = 0; j < num_cols; ++j) {
    const bool integral = j < static_cast<int>(is_integer.size()) && is_integer[j] != 0;
    h.i64(integral ? 1 : 0);
  }
  return h.value();
}

std::string Model::fingerprint_hex() const {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fingerprint()));
  return buf;
}

std::string Model::validate() const {
  std::ostringstream err;
  const auto n = static_cast<std::size_t>(num_cols);
  const auto m = static_cast<std::size_t>(num_rows);
  if (num_rows < 0 || num_cols < 0) return "negative dimension";
  if (sense != 1 && sense != -1) return "sense must be +1 or -1";
  if (!std::isfinite(obj_offset)) return "obj_offset is not finite";
  if (obj.size() != n) return "obj has wrong size";
  if (col_lower.size() != n || col_upper.size() != n) return "column bounds have wrong size";
  if (row_lower.size() != m || row_upper.size() != m) return "row bounds have wrong size";
  if (col_start.size() != n + 1) return "col_start must have num_cols+1 entries";
  if (col_start[0] != 0) return "col_start[0] must be 0";
  if (row_index.size() != value.size()) return "row_index and value differ in size";
  if (static_cast<std::size_t>(col_start[n]) != value.size()) return "col_start[n] != nnz";
  if (!row_names.empty() && row_names.size() != m) return "row_names has wrong size";
  if (!col_names.empty() && col_names.size() != n) return "col_names has wrong size";
  if (!is_integer.empty() && is_integer.size() != n) return "is_integer has wrong size";

  std::vector<int> seen(m, -1);
  for (std::size_t j = 0; j < n; ++j) {
    if (col_start[j + 1] < col_start[j]) {
      err << "col_start not monotone at column " << j;
      return err.str();
    }
    if (!std::isfinite(obj[j])) {
      err << "objective coefficient of column " << j << " is not finite";
      return err.str();
    }
    for (int k = col_start[j]; k < col_start[j + 1]; ++k) {
      const int i = row_index[k];
      if (i < 0 || static_cast<std::size_t>(i) >= m) {
        err << "row index out of range in column " << j;
        return err.str();
      }
      if (seen[i] == static_cast<int>(j)) {
        err << "duplicate entry (" << i << "," << j << ")";
        return err.str();
      }
      seen[i] = static_cast<int>(j);
      if (!std::isfinite(value[k])) {
        err << "matrix entry (" << i << "," << j << ") is not finite";
        return err.str();
      }
    }
  }
  auto check_bounds = [&](const std::vector<double>& lo, const std::vector<double>& up,
                          const char* what) -> std::string {
    for (std::size_t k = 0; k < lo.size(); ++k) {
      std::ostringstream e;
      if (std::isnan(lo[k]) || std::isnan(up[k])) e << what << " bound " << k << " is NaN";
      else if (lo[k] == kInf) e << what << " lower bound " << k << " is +inf";
      else if (up[k] == -kInf) e << what << " upper bound " << k << " is -inf";
      if (!e.str().empty()) return e.str();
    }
    return {};
  };
  if (auto s = check_bounds(col_lower, col_upper, "column"); !s.empty()) return s;
  if (auto s = check_bounds(row_lower, row_upper, "row"); !s.empty()) return s;
  return {};
}

std::string Model::crossed_bounds() const {
  auto first = [](const std::vector<double>& lo, const std::vector<double>& up, const char* what) -> std::string {
    for (std::size_t k = 0; k < lo.size() && k < up.size(); ++k) {
      if (lo[k] > up[k]) {
        std::ostringstream e;
        e << what << " " << k << " has lower bound " << lo[k] << " > upper bound " << up[k];
        return e.str();
      }
    }
    return {};
  };
  if (auto s = first(col_lower, col_upper, "column"); !s.empty()) return s;
  return first(row_lower, row_upper, "row");
}

double Model::objective_value(const std::vector<double>& x) const {
  double v = obj_offset;
  for (int j = 0; j < num_cols; ++j) v += obj[j] * x[j];
  return v;
}

std::vector<double> Model::row_activity(const std::vector<double>& x) const {
  std::vector<double> ax(num_rows, 0.0);
  for (int j = 0; j < num_cols; ++j)
    for (int k = col_start[j]; k < col_start[j + 1]; ++k) ax[row_index[k]] += value[k] * x[j];
  return ax;
}

}  // namespace ps26119
