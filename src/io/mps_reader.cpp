// mps_reader.cpp - MPS parser.
//
// An MPS file describes an LP column by column:
//
//   NAME          EXAMPLE
//   ROWS
//    N  COST                       <- objective row
//    L  LIM1                       <- a_i x <= b_i
//   COLUMNS
//       X1        COST         1.0   LIM1         1.0
//   RHS
//       RHS       LIM1         4.0
//   BOUNDS
//    UP BND       X1           4.0
//   ENDATA
//
// The parser works line by line. Lines starting in column 1 are section
// headers; indented lines are data for the current section. Every data line
// is first split into "fields" (by whitespace in free format, by column
// position in fixed format) and then handed to one handler per section.
// Coefficients are collected as (row, col, value) triplets and converted to
// compressed sparse column form at the end.
//
// Origin: gpuopt src/io/mps_reader.cpp (Shivanshu Vats, shivanshu24-code/gpu_optimization@c192dd0).
// Adapted to fill the canonical ps26119::Model; parsing logic unchanged.

#include "io/mps_parser.h"
#include "io/mps_reader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "la/csc.h"

namespace ps26119::io {

MpsParseError::MpsParseError(const std::string& message, int line)
    : std::runtime_error(line > 0 ? "line " + std::to_string(line) + ": " + message : message),
      line_(line) {}

using la::Triplet;

namespace {

enum class Section { kNone, kName, kObjSense, kObjName, kRows, kColumns, kRhs, kRanges, kBounds, kEndata };
enum class RowType : char { kE, kL, kG };

// Special values stored in the row-name map for rows that are not constraints.
constexpr int kObjectiveRow = -1;
constexpr int kDroppedFreeRow = -2;

std::string_view trim(std::string_view s) {
  size_t b = 0;
  size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::vector<std::string_view> split_ws(std::string_view s) {
  std::vector<std::string_view> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    const size_t start = i;
    while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i > start) out.push_back(s.substr(start, i - start));
  }
  return out;
}

std::string upper(std::string_view s) {
  std::string out(s);
  for (char& ch : out) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return out;
}

// Fixed-format field between 1-based inclusive columns [first, last].
std::string_view fixed_field(std::string_view line, int first, int last) {
  if (static_cast<int>(line.size()) < first) return {};
  const size_t start = static_cast<size_t>(first - 1);
  const size_t len = std::min(static_cast<size_t>(last - first + 1), line.size() - start);
  return trim(line.substr(start, len));
}

// Accepts "1.", ".5", "-3e2", "+4", Fortran-style "1.5D+03".
bool parse_number(std::string_view s, double& out) {
  std::string buf(s);
  if (!buf.empty() && buf[0] == '+') buf.erase(0, 1);
  for (char& ch : buf) {
    if (ch == 'd' || ch == 'D') ch = 'e';
  }
  if (buf.empty()) return false;
  // Portable replacement for std::from_chars(double), which Apple's libc++ does not
  // provide. Accepts the same language: [-]digits[.digits][(e|E)[+-]digits], or
  // inf / infinity / nan (any case); an out-of-range value is an error.
  std::string body = buf;
  for (char& ch : body) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  const bool neg = body[0] == '-';
  const std::string word = neg ? body.substr(1) : body;
  if (word == "inf" || word == "infinity") {
    out = neg ? -kInf : kInf;
    return true;
  }
  if (word == "nan") {
    out = std::nan("");
    return true;
  }
  if (word.empty() || !(std::isdigit(static_cast<unsigned char>(word[0])) || word[0] == '.')) return false;
  for (char ch : word) {
    if (!std::isdigit(static_cast<unsigned char>(ch)) && ch != '.' && ch != 'e' && ch != '+' && ch != '-') return false;
  }
  char* end = nullptr;
  errno = 0;
  out = std::strtod(buf.c_str(), &end);
  return errno != ERANGE && end == buf.c_str() + buf.size();
}

class Parser {
 public:
  Parser(MpsFormat format, const MpsReadOptions& options) : format_(format), opt_(options) {}

  MpsReadResult parse(std::string_view text) {
    size_t pos = 0;
    while (pos < text.size() && section_ != Section::kEndata) {
      size_t end = text.find('\n', pos);
      if (end == std::string_view::npos) end = text.size();
      std::string_view line = text.substr(pos, end - pos);
      pos = end + 1;
      ++line_no_;
      if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
      process_line(line);
    }
    return finish();
  }

 private:
  // ---------------------------------------------------------------- helpers
  [[noreturn]] void fail(const std::string& message) const { throw MpsParseError(message, line_no_); }

  void warn(const std::string& message) {
    warnings_.push_back("line " + std::to_string(line_no_) + ": " + message);
  }

  double number(std::string_view s) const {
    double v = 0.0;
    if (!parse_number(s, v)) fail("expected a number, found '" + std::string(s) + "'");
    return v;
  }

  double to_inf(double v) const {
    if (v >= opt_.infinity_threshold) return kInf;
    if (v <= -opt_.infinity_threshold) return -kInf;
    return v;
  }

  int find_row(std::string_view name) const {
    auto it = row_index_.find(std::string(name));
    if (it == row_index_.end()) fail("unknown row '" + std::string(name) + "'");
    return it->second;
  }

  int find_col(std::string_view name) const {
    auto it = col_index_.find(std::string(name));
    if (it == col_index_.end()) fail("unknown column '" + std::string(name) + "'");
    return it->second;
  }

  // Returns true if this data line belongs to the set we are reading.
  // MPS allows several RHS / RANGES / BOUNDS vectors; like every solver we
  // use the first one named and ignore the rest.
  bool accept_set(std::string& chosen, bool& chosen_flag, std::string_view set, const char* what) {
    if (!chosen_flag) {
      chosen = std::string(set);
      chosen_flag = true;
      return true;
    }
    if (set == chosen) return true;
    if (!ignored_set_warned_) {
      warn(std::string("ignoring additional ") + what + " set '" + std::string(set) + "'");
      ignored_set_warned_ = true;
    }
    return false;
  }

  // ----------------------------------------------------------- line dispatch
  void process_line(std::string_view line) {
    if (trim(line).empty() || line[0] == '*') return;
    const bool indented = line[0] == ' ' || line[0] == '\t';
    if (!indented && start_section(line)) return;

    switch (section_) {
      case Section::kObjSense: read_objsense(line); break;
      case Section::kObjName: objname_ = std::string(split_ws(line)[0]); break;
      case Section::kRows: read_row(line); break;
      case Section::kColumns: read_column(line); break;
      case Section::kRhs: read_rhs_or_range(line, /*is_range=*/false); break;
      case Section::kRanges: read_rhs_or_range(line, /*is_range=*/true); break;
      case Section::kBounds: read_bound(line); break;
      default: fail("data line outside of any section");
    }
  }

  // Returns false if the line is not a section header (only possible for an
  // unindented data line in free format).
  bool start_section(std::string_view line) {
    const auto tokens = split_ws(line);
    const std::string key = upper(tokens[0]);

    if (key == "NAME") {
      section_ = Section::kName;
      name_ = std::string(trim(trim(line).substr(4)));
    } else if (key == "OBJSENSE" || key == "OBJSENCE") {
      section_ = Section::kObjSense;
      if (tokens.size() > 1) set_sense(tokens[1]);
    } else if (key == "OBJNAME") {
      section_ = Section::kObjName;
      if (tokens.size() > 1) objname_ = std::string(tokens[1]);
    } else if (key == "ROWS") {
      section_ = Section::kRows;
    } else if (key == "COLUMNS") {
      if (row_index_.empty()) fail("COLUMNS section before any ROWS");
      section_ = Section::kColumns;
    } else if (key == "RHS") {
      section_ = Section::kRhs;
    } else if (key == "RANGES") {
      section_ = Section::kRanges;
    } else if (key == "BOUNDS") {
      section_ = Section::kBounds;
    } else if (key == "ENDATA") {
      section_ = Section::kEndata;
      seen_endata_ = true;
    } else if (key == "QUADOBJ" || key == "QMATRIX" || key == "QSECTION" || key == "QCMATRIX") {
      fail("section " + key + " (quadratic objective) is not supported yet - arrives with Layer 6 QP");
    } else if (key == "SOS" || key == "CSECTION" || key == "INDICATORS" || key == "CONES") {
      fail("section " + key + " is not supported");
    } else if (format_ == MpsFormat::kFree && section_ >= Section::kRows) {
      return false;  // unindented data line, tolerated in free format
    } else {
      fail("unknown section '" + std::string(tokens[0]) + "'");
    }
    return true;
  }

  // ---------------------------------------------------------------- OBJSENSE
  void set_sense(std::string_view token) {
    const std::string s = upper(token);
    if (s == "MAX" || s == "MAXIMIZE") {
      sense_ = -1;
    } else if (s == "MIN" || s == "MINIMIZE") {
      sense_ = 1;
    } else {
      fail("unknown OBJSENSE '" + std::string(token) + "'");
    }
  }

  void read_objsense(std::string_view line) { set_sense(split_ws(line)[0]); }

  // -------------------------------------------------------------------- ROWS
  //   free : <type> <name>
  //   fixed: type in columns 2-3, name in columns 5-12
  void read_row(std::string_view line) {
    std::string_view type_field, name_field;
    if (format_ == MpsFormat::kFree) {
      const auto t = split_ws(line);
      if (t.size() != 2) fail("ROWS line needs exactly 2 fields (type, name)");
      type_field = t[0];
      name_field = t[1];
    } else {
      type_field = fixed_field(line, 2, 3);
      name_field = fixed_field(line, 5, 12);
    }
    if (name_field.empty()) fail("ROWS line without a row name");
    const std::string name(name_field);
    if (row_index_.count(name)) fail("duplicate row name '" + name + "'");

    const std::string type = upper(type_field);
    if (type == "N") {
      const bool is_objective = !have_objective_ && (objname_.empty() || objname_ == name);
      if (is_objective) {
        row_index_[name] = kObjectiveRow;
        objective_name_ = name;
        have_objective_ = true;
      } else {
        row_index_[name] = kDroppedFreeRow;
        ++dropped_free_rows_;
      }
      return;
    }

    RowType rt;
    if (type == "E") {
      rt = RowType::kE;
    } else if (type == "L") {
      rt = RowType::kL;
    } else if (type == "G") {
      rt = RowType::kG;
    } else {
      fail("unknown row type '" + std::string(type_field) + "'");
    }
    row_index_[name] = static_cast<int>(row_names_.size());
    row_names_.push_back(name);
    row_types_.push_back(rt);
    rhs_.push_back(0.0);
    range_.push_back(0.0);
    has_range_.push_back(0);
  }

  // ----------------------------------------------------------------- COLUMNS
  //   free : <col> <row> <value> [<row> <value>]
  //   fixed: col 5-12, row 15-22, value 25-36, row 40-47, value 50-61
  //   integer blocks:  <name> 'MARKER' 'INTORG'  ...  <name> 'MARKER' 'INTEND'
  void read_column(std::string_view line) {
    if (line.find("'MARKER'") != std::string_view::npos) {
      const std::string u = upper(line);
      if (u.find("'INTORG'") != std::string::npos) {
        in_integer_block_ = true;
      } else if (u.find("'INTEND'") != std::string::npos) {
        in_integer_block_ = false;
      } else {
        fail("MARKER line without 'INTORG' or 'INTEND'");
      }
      return;
    }

    std::string_view col, row1, val1, row2, val2;
    if (format_ == MpsFormat::kFree) {
      const auto t = split_ws(line);
      if (t.size() != 3 && t.size() != 5) fail("COLUMNS line needs 3 or 5 fields");
      col = t[0];
      row1 = t[1];
      val1 = t[2];
      if (t.size() == 5) {
        row2 = t[3];
        val2 = t[4];
      }
    } else {
      col = fixed_field(line, 5, 12);
      row1 = fixed_field(line, 15, 22);
      val1 = fixed_field(line, 25, 36);
      row2 = fixed_field(line, 40, 47);
      val2 = fixed_field(line, 50, 61);
      if (col.empty() || row1.empty() || val1.empty()) fail("incomplete COLUMNS line");
    }

    const int j = column_for(col);
    add_coefficient(j, row1, number(val1));
    if (!row2.empty()) add_coefficient(j, row2, number(val2));
  }

  int column_for(std::string_view name) {
    if (current_col_ >= 0 && col_names_[current_col_] == name) return current_col_;
    const std::string key(name);
    auto it = col_index_.find(key);
    if (it != col_index_.end()) {
      warn("entries of column '" + key + "' are not contiguous");
      current_col_ = it->second;
      return current_col_;
    }
    current_col_ = static_cast<int>(col_names_.size());
    col_index_[key] = current_col_;
    col_names_.push_back(key);
    obj_.push_back(0.0);
    col_lower_.push_back(0.0);
    col_upper_.push_back(kInf);
    is_integer_.push_back(in_integer_block_ ? 1 : 0);
    lower_explicit_.push_back(0);
    upper_explicit_.push_back(0);
    return current_col_;
  }

  void add_coefficient(int col, std::string_view row_name, double value) {
    const int i = find_row(row_name);
    if (i == kObjectiveRow) {
      obj_[col] += value;
    } else if (i >= 0 && value != 0.0) {
      triplets_.push_back({i, col, value});
    }
    // kDroppedFreeRow: coefficients of extra N rows are discarded.
  }

  // ------------------------------------------------------------ RHS / RANGES
  //   free : [<set>] <row> <value> [<row> <value>]
  //   fixed: set 5-12, row 15-22, value 25-36, row 40-47, value 50-61
  void read_rhs_or_range(std::string_view line, bool is_range) {
    std::string_view set, row1, val1, row2, val2;
    if (format_ == MpsFormat::kFree) {
      const auto t = split_ws(line);
      size_t k = 0;
      if (t.size() == 3 || t.size() == 5) {
        set = t[0];
        k = 1;
      } else if (t.size() != 2 && t.size() != 4) {
        fail(std::string(is_range ? "RANGES" : "RHS") + " line needs 2 to 5 fields");
      }
      row1 = t[k];
      val1 = t[k + 1];
      if (t.size() - k == 4) {
        row2 = t[k + 2];
        val2 = t[k + 3];
      }
    } else {
      set = fixed_field(line, 5, 12);
      row1 = fixed_field(line, 15, 22);
      val1 = fixed_field(line, 25, 36);
      row2 = fixed_field(line, 40, 47);
      val2 = fixed_field(line, 50, 61);
      if (row1.empty() || val1.empty()) fail("incomplete RHS/RANGES line");
    }

    const bool use = is_range ? accept_set(range_set_, range_set_chosen_, set, "RANGES")
                              : accept_set(rhs_set_, rhs_set_chosen_, set, "RHS");
    if (!use) return;

    set_rhs_or_range(row1, number(val1), is_range);
    if (!row2.empty()) set_rhs_or_range(row2, number(val2), is_range);
  }

  void set_rhs_or_range(std::string_view row_name, double value, bool is_range) {
    const int i = find_row(row_name);
    if (i == kDroppedFreeRow) return;
    if (i == kObjectiveRow) {
      if (is_range) {
        warn("RANGES entry on the objective row ignored");
      } else {
        obj_offset_ = -value;  // MPS convention: RHS on the objective = -constant
      }
      return;
    }
    if (is_range) {
      range_[i] = value;
      has_range_[i] = 1;
    } else {
      rhs_[i] = value;
    }
  }

  // ------------------------------------------------------------------ BOUNDS
  //   free : <type> [<set>] <col> [<value>]
  //   fixed: type 2-3, set 5-12, col 15-22, value 25-36
  void read_bound(std::string_view line) {
    std::string_view type_field, set, col, val;
    if (format_ == MpsFormat::kFree) {
      const auto t = split_ws(line);
      if (t.size() < 2) fail("BOUNDS line too short");
      type_field = t[0];
      const std::string ty = upper(type_field);
      const bool needs_value = ty == "UP" || ty == "LO" || ty == "FX" || ty == "LI" || ty == "UI";
      const size_t rest = t.size() - 1;
      if (needs_value) {
        if (rest == 3) {
          set = t[1]; col = t[2]; val = t[3];
        } else if (rest == 2) {
          col = t[1]; val = t[2];
        } else {
          fail("BOUNDS line of type " + ty + " needs a column and a value");
        }
      } else {
        // FR / MI / PL / BV: the value is optional, so "BV X 1" and
        // "BV SET X" both have 3 tokens. Disambiguate by column name.
        double tmp;
        if (rest == 1) {
          col = t[1];
        } else if (rest == 2 && col_index_.count(std::string(t[1])) && parse_number(t[2], tmp)) {
          col = t[1]; val = t[2];
        } else if (rest == 2) {
          set = t[1]; col = t[2];
        } else if (rest == 3) {
          set = t[1]; col = t[2]; val = t[3];
        } else {
          fail("malformed BOUNDS line");
        }
      }
    } else {
      type_field = fixed_field(line, 2, 3);
      set = fixed_field(line, 5, 12);
      col = fixed_field(line, 15, 22);
      val = fixed_field(line, 25, 36);
    }

    if (!accept_set(bound_set_, bound_set_chosen_, set, "BOUNDS")) return;
    const int j = find_col(col);
    const std::string ty = upper(type_field);
    auto value = [&]() { return to_inf(number(val)); };

    if (ty == "UP" || ty == "UI") {
      const double v = value();
      col_upper_[j] = v;
      upper_explicit_[j] = 1;
      if (v < 0.0 && col_lower_[j] == 0.0 && !lower_explicit_[j]) {
        col_lower_[j] = -kInf;  // classic MPS rule for a negative upper bound
        warn("negative upper bound on '" + col_names_[j] + "' with default lower bound: lower set to -inf");
      }
      if (ty == "UI") is_integer_[j] = 1;
    } else if (ty == "LO" || ty == "LI") {
      col_lower_[j] = value();
      lower_explicit_[j] = 1;
      if (ty == "LI") is_integer_[j] = 1;
    } else if (ty == "FX") {
      const double v = value();
      col_lower_[j] = col_upper_[j] = v;
      lower_explicit_[j] = upper_explicit_[j] = 1;
    } else if (ty == "FR") {
      col_lower_[j] = -kInf;
      col_upper_[j] = kInf;
      lower_explicit_[j] = upper_explicit_[j] = 1;
    } else if (ty == "MI") {
      col_lower_[j] = -kInf;
      lower_explicit_[j] = 1;
    } else if (ty == "PL") {
      col_upper_[j] = kInf;
      upper_explicit_[j] = 1;
    } else if (ty == "BV") {
      col_lower_[j] = 0.0;
      col_upper_[j] = 1.0;
      lower_explicit_[j] = upper_explicit_[j] = 1;
      is_integer_[j] = 1;
    } else if (ty == "SC") {
      fail("semi-continuous (SC) bounds are not supported");
    } else {
      fail("unknown bound type '" + std::string(type_field) + "'");
    }
  }

  // ------------------------------------------------------------------ finish
  MpsReadResult finish() {
    if (!seen_endata_) warnings_.push_back("file ended without ENDATA");
    if (!have_objective_) warnings_.push_back("no objective (N) row found; objective is zero");
    if (dropped_free_rows_ > 0) {
      warnings_.push_back("dropped " + std::to_string(dropped_free_rows_) +
                          " extra free (N) row(s) that are not the objective");
    }

    MpsReadResult result;
    result.format_used = format_;
    Model& lp = result.problem;
    const int m = static_cast<int>(row_names_.size());
    const int n = static_cast<int>(col_names_.size());

    int merged = 0;
    la::SparseMatrixCSC A = la::build_csc(m, n, std::move(triplets_), &merged);
    lp.num_rows = m;
    lp.num_cols = n;
    lp.col_start = std::move(A.col_start);
    lp.row_index = std::move(A.row_index);
    lp.value = std::move(A.value);
    if (merged > 0) {
      warnings_.push_back("summed " + std::to_string(merged) + " duplicate matrix entries");
    }

    lp.name = name_;
    result.objective_name = objective_name_;
    lp.sense = sense_;
    lp.obj_offset = obj_offset_;
    lp.obj = std::move(obj_);
    lp.row_names = std::move(row_names_);
    lp.col_names = std::move(col_names_);
    lp.is_integer.assign(is_integer_.begin(), is_integer_.end());

    // Row bounds  L <= a_i x <= U  from the row type, RHS and RANGES:
    //   type  no range      range R
    //   E     [b, b]        R > 0: [b, b+|R|]   R < 0: [b-|R|, b]
    //   L     [-inf, b]     [b-|R|, b]
    //   G     [b, +inf]     [b, b+|R|]
    lp.row_lower.resize(m);
    lp.row_upper.resize(m);
    for (int i = 0; i < m; ++i) {
      const double b = to_inf(rhs_[i]);
      const double r = std::fabs(to_inf(range_[i]));
      double lo = 0.0, up = 0.0;
      switch (row_types_[i]) {
        case RowType::kE:
          lo = up = b;
          if (has_range_[i] && range_[i] > 0) up = b + r;
          if (has_range_[i] && range_[i] < 0) lo = b - r;
          break;
        case RowType::kL:
          lo = has_range_[i] ? b - r : -kInf;
          up = b;
          break;
        case RowType::kG:
          lo = b;
          up = has_range_[i] ? b + r : kInf;
          break;
      }
      lp.row_lower[i] = lo;
      lp.row_upper[i] = up;
    }

    for (int j = 0; j < n; ++j) {
      if (opt_.integer_default_binary && lp.is_integer[j] && !lower_explicit_[j] && !upper_explicit_[j]) {
        col_upper_[j] = 1.0;
      }
      if (col_lower_[j] > col_upper_[j]) {
        warnings_.push_back("column '" + lp.col_names[j] + "' has lower bound > upper bound");
      }
    }
    lp.col_lower = std::move(col_lower_);
    lp.col_upper = std::move(col_upper_);

    result.warnings = std::move(warnings_);
    return result;
  }

  // ------------------------------------------------------------------- state
  MpsFormat format_;
  const MpsReadOptions& opt_;
  int line_no_ = 0;
  Section section_ = Section::kNone;
  bool seen_endata_ = false;
  std::vector<std::string> warnings_;

  std::string name_;
  int sense_ = 1;  // +1 minimise, -1 maximise (Model::sense)
  std::string objname_;
  std::string objective_name_;
  bool have_objective_ = false;
  int dropped_free_rows_ = 0;
  double obj_offset_ = 0.0;

  std::unordered_map<std::string, int> row_index_;
  std::vector<std::string> row_names_;
  std::vector<RowType> row_types_;
  std::vector<double> rhs_, range_;
  std::vector<char> has_range_;

  std::unordered_map<std::string, int> col_index_;
  std::vector<std::string> col_names_;
  std::vector<double> obj_, col_lower_, col_upper_;
  std::vector<char> is_integer_, lower_explicit_, upper_explicit_;
  int current_col_ = -1;
  bool in_integer_block_ = false;
  std::vector<Triplet> triplets_;

  std::string rhs_set_, range_set_, bound_set_;
  bool rhs_set_chosen_ = false, range_set_chosen_ = false, bound_set_chosen_ = false;
  bool ignored_set_warned_ = false;
};

}  // namespace

MpsReadResult read_mps_string(std::string_view text, const MpsReadOptions& options) {
  if (options.format != MpsFormat::kAuto) return Parser(options.format, options).parse(text);

  // Auto: free format handles almost every modern file; fixed format is
  // needed only when names contain spaces.
  try {
    return Parser(MpsFormat::kFree, options).parse(text);
  } catch (const MpsParseError& free_error) {
    try {
      MpsReadResult result = Parser(MpsFormat::kFixed, options).parse(text);
      result.warnings.insert(result.warnings.begin(),
                             std::string("free-format parse failed (") + free_error.what() +
                                 "); file was read as fixed format");
      return result;
    } catch (const MpsParseError& fixed_error) {
      throw MpsParseError(std::string("free format: ") + free_error.what() +
                              " | fixed format: " + fixed_error.what(),
                          0);
    }
  }
}

MpsReadResult read_mps_file(const std::string& path, const MpsReadOptions& options) {
  if (path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz") == 0) {
    throw MpsParseError("gzip-compressed input is not supported yet; decompress the file first", 0);
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open file '" + path + "'");
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return read_mps_string(buffer.str(), options);
}

bool read_mps(const std::string& path, Model& model, std::string& error, std::vector<std::string>* warnings) {
  try {
    MpsReadResult r = read_mps_file(path);
    model = std::move(r.problem);
    if (warnings) *warnings = std::move(r.warnings);
    return true;
  } catch (const std::exception& e) {  // MpsParseError, cannot open, bad_alloc
    error = e.what();
    return false;
  }
}

}  // namespace ps26119::io
