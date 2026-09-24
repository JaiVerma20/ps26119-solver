// model.h — the Model contract shared with the MPS reader author (CLAUDE.md §6).
// DO NOT change the meaning of any field without agreeing it with the reader author.
//
//   minimise   sense * (cᵀx) + obj_offset      (sense = +1 min, −1 max)
//   subject to row_lower ≤ A x ≤ row_upper
//              col_lower ≤ x   ≤ col_upper
//
// A is stored in CSC: col_start (n+1 entries), row_index / value (nnz entries).
// Infinity is std::numeric_limits<double>::infinity(). Row / column names are kept.
// is_integer marks MARKER-integer and BV/LI/UI columns (0/1 per column).
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ps26119 {

inline constexpr double kInf = std::numeric_limits<double>::infinity();

struct Model {
  std::string name;
  int num_rows = 0;
  int num_cols = 0;
  int sense = 1;            // +1 minimise, -1 maximise
  // Interpretation used everywhere in this code base (same as HiGHS / MPS practice):
  // the objective VALUE is  cᵀx + obj_offset, and `sense` only chooses min or max.
  // Internally engines minimise sense*(cᵀx); the offset never affects the argmin.
  double obj_offset = 0.0;
  std::vector<double> obj;  // c, size n

  std::vector<double> col_lower, col_upper;  // size n
  std::vector<double> row_lower, row_upper;  // size m

  std::vector<int> col_start;   // size n+1
  std::vector<int> row_index;   // size nnz
  std::vector<double> value;    // size nnz

  std::vector<std::string> row_names;  // size m (may be empty)
  std::vector<std::string> col_names;  // size n (may be empty)
  std::vector<std::uint8_t> is_integer;  // size n (may be empty = all continuous)

  std::size_t nnz() const { return value.size(); }

  // 64-bit FNV-1a hash of every number in the model (sizes, sense, offset, c, bounds,
  // CSC structure and values, integrality). Names are NOT hashed. -0.0 hashes like 0.0.
  // Two readers that produce the same Model produce the same fingerprint.
  std::uint64_t fingerprint() const;
  std::string fingerprint_hex() const;

  // Returns an empty string when all invariants hold, otherwise a description of the
  // first violated invariant (sizes, CSC monotonicity, index range, duplicate entries in a
  // column, NaNs, lower > upper, lower = +inf, upper = -inf, sense ∈ {+1,-1}).
  std::string validate() const;

  // Objective as the user sees it: cᵀx + obj_offset (sense only sets the direction).
  double objective_value(const std::vector<double>& x) const;
  // Row activity A x.
  std::vector<double> row_activity(const std::vector<double>& x) const;
};

}  // namespace ps26119
