// lpm_reader.cpp — .lpm reader/writer. Format spec: tools/lpm.py (module docstring).
//
// Invariants on success: the returned Model has consistent sizes and passes the
// structural checks in Model::validate() (the reader calls it and fails otherwise).
#include "io/lpm_reader.h"

#include <algorithm>
#include <limits>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>

namespace ps26119::io {
namespace {

class Scanner {
 public:
  explicit Scanner(const std::string& text) : s_(text) {}

  bool eof() const { return pos_ >= s_.size(); }

  // Next full line (without '\n'); returns false at end.
  bool line(std::string_view& out) {
    if (eof()) return false;
    const std::size_t e = s_.find('\n', pos_);
    const std::size_t end = e == std::string::npos ? s_.size() : e;
    out = std::string_view(s_).substr(pos_, end - pos_);
    if (!out.empty() && out.back() == '\r') out.remove_suffix(1);
    pos_ = end + 1;
    ++line_no_;
    return true;
  }

  // Next whitespace-separated token spanning lines.
  bool token(std::string_view& out) {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
      if (s_[pos_] == '\n') ++line_no_;
      ++pos_;
    }
    if (pos_ >= s_.size()) return false;
    const std::size_t b = pos_;
    while (pos_ < s_.size() && !std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    out = std::string_view(s_).substr(b, pos_ - b);
    return true;
  }

  // After reading tokens, skip the remainder of the current line.
  void finish_line() {
    while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
    if (pos_ < s_.size()) {
      ++pos_;
      ++line_no_;
    }
  }

  int line_no() const { return line_no_; }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;
  int line_no_ = 0;
};

bool parse_double(std::string_view t, double& v) {
  std::string tmp(t);
  char* end = nullptr;
  errno = 0;
  v = std::strtod(tmp.c_str(), &end);
  return end == tmp.c_str() + tmp.size() && !tmp.empty() && !std::isnan(v);
}

bool parse_int(std::string_view t, long long& v) {
  std::string tmp(t);
  char* end = nullptr;
  v = std::strtoll(tmp.c_str(), &end, 10);
  return end == tmp.c_str() + tmp.size() && !tmp.empty();
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
  return s;
}

}  // namespace

LpmReadResult read_lpm_string(const std::string& text, Model& model) {
  LpmReadResult r;
  model = Model{};
  Scanner sc(text);
  auto fail = [&](const std::string& msg) {
    r.ok = false;
    r.error = "line " + std::to_string(sc.line_no()) + ": " + msg;
    return r;
  };

  std::string_view ln;
  if (!sc.line(ln) || trim(ln) != "LPM 1") return fail("expected header 'LPM 1'");

  long long rows = -1, cols = -1, nnz = -1;
  bool have_integer = false;
  // Sections grow as values are actually read, with a bounded reserve: a bogus header count
  // (e.g. NNZ 99999999999) used to allocate gigabytes up front and get the process killed.
  constexpr long long kReserveCap = 1 << 22;
  auto read_doubles = [&](std::vector<double>& out, long long count) -> bool {
    out.clear();
    out.reserve(static_cast<std::size_t>(std::min(count, kReserveCap)));
    std::string_view t;
    double x = 0;
    for (long long k = 0; k < count; ++k) {
      if (!sc.token(t) || !parse_double(t, x)) return false;
      out.push_back(x);
    }
    sc.finish_line();
    return true;
  };
  auto read_ints = [&](std::vector<int>& out, long long count) -> bool {
    out.clear();
    out.reserve(static_cast<std::size_t>(std::min(count, kReserveCap)));
    std::string_view t;
    long long v;
    for (long long k = 0; k < count; ++k) {
      if (!sc.token(t) || !parse_int(t, v) || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
        return false;
      out.push_back(static_cast<int>(v));
    }
    sc.finish_line();
    return true;
  };
  auto sizes_known = [&] { return rows >= 0 && cols >= 0 && nnz >= 0; };

  bool ended = false;
  while (sc.line(ln)) {
    std::string_view t = trim(ln);
    if (t.empty()) continue;
    if (t.front() == '#') {
      constexpr std::string_view kFp = "# fingerprint ";
      if (t.substr(0, kFp.size()) == kFp) r.declared_fingerprint = std::string(trim(t.substr(kFp.size())));
      continue;
    }
    const std::size_t sp = t.find(' ');
    const std::string_view tag = t.substr(0, sp);
    const std::string_view rest = sp == std::string_view::npos ? std::string_view{} : trim(t.substr(sp + 1));
    if (tag == "END") {
      ended = true;
      break;
    } else if (tag == "NAME") {
      model.name = std::string(rest);
    } else if (tag == "SENSE") {
      if (rest == "MIN") model.sense = 1;
      else if (rest == "MAX") model.sense = -1;
      else return fail("SENSE must be MIN or MAX");
    } else if (tag == "ROWS" || tag == "COLS" || tag == "NNZ") {
      long long v;
      // the Model indexes with int (CSC col_start, row_index)
      if (!parse_int(rest, v) || v < 0 || v > std::numeric_limits<int>::max()) return fail("bad size");
      (tag == "ROWS" ? rows : tag == "COLS" ? cols : nnz) = v;
    } else if (tag == "OFFSET") {
      if (!parse_double(rest, model.obj_offset)) return fail("bad OFFSET");
    } else if (!sizes_known()) {
      return fail("ROWS/COLS/NNZ must precede section " + std::string(tag));
    } else if (tag == "OBJ") {
      if (!read_doubles(model.obj, cols)) return fail("bad OBJ section");
    } else if (tag == "COL_LOWER") {
      if (!read_doubles(model.col_lower, cols)) return fail("bad COL_LOWER section");
    } else if (tag == "COL_UPPER") {
      if (!read_doubles(model.col_upper, cols)) return fail("bad COL_UPPER section");
    } else if (tag == "ROW_LOWER") {
      if (!read_doubles(model.row_lower, rows)) return fail("bad ROW_LOWER section");
    } else if (tag == "ROW_UPPER") {
      if (!read_doubles(model.row_upper, rows)) return fail("bad ROW_UPPER section");
    } else if (tag == "COL_START") {
      if (!read_ints(model.col_start, cols + 1)) return fail("bad COL_START section");
    } else if (tag == "ROW_INDEX") {
      if (!read_ints(model.row_index, nnz)) return fail("bad ROW_INDEX section");
    } else if (tag == "VALUE") {
      if (!read_doubles(model.value, nnz)) return fail("bad VALUE section");
    } else if (tag == "INTEGER") {
      std::vector<int> v;
      if (!read_ints(v, cols)) return fail("bad INTEGER section");
      model.is_integer.assign(v.begin(), v.end());
      have_integer = true;
    } else if (tag == "ROW_NAMES" || tag == "COL_NAMES") {
      auto& names = tag == "ROW_NAMES" ? model.row_names : model.col_names;
      const long long count = tag == "ROW_NAMES" ? rows : cols;
      names.reserve(static_cast<std::size_t>(std::min(count, kReserveCap)));
      for (long long k = 0; k < count; ++k) {
        if (!sc.line(ln)) return fail("unexpected end of names");
        names.emplace_back(ln);
      }
    } else {
      return fail("unknown section '" + std::string(tag) + "'");
    }
  }
  if (!ended) return fail("missing END");
  if (!sizes_known()) return fail("missing ROWS/COLS/NNZ");
  model.num_rows = static_cast<int>(rows);
  model.num_cols = static_cast<int>(cols);
  if (model.col_start.empty() && cols == 0) model.col_start = {0};
  (void)have_integer;
  if (auto err = model.validate(); !err.empty()) return fail("invalid model: " + err);
  r.ok = true;
  return r;
}

LpmReadResult read_lpm(const std::string& path, Model& model) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    LpmReadResult r;
    r.error = "cannot open " + path;
    return r;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return read_lpm_string(ss.str(), model);
}

bool write_lpm(const std::string& path, const Model& m, std::string& error) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    error = "cannot open " + path + " for writing";
    return false;
  }
  auto num = [&](double v) {
    if (v == kInf) std::fputs("inf", f);
    else if (v == -kInf) std::fputs("-inf", f);
    else std::fprintf(f, "%.17g", v);
  };
  auto dvec = [&](const char* tag, const std::vector<double>& v) {
    std::fprintf(f, "%s\n", tag);
    for (std::size_t k = 0; k < v.size(); ++k) {
      num(v[k]);
      std::fputc((k % 8 == 7 || k + 1 == v.size()) ? '\n' : ' ', f);
    }
  };
  auto ivec = [&](const char* tag, const auto& v) {
    std::fprintf(f, "%s\n", tag);
    for (std::size_t k = 0; k < v.size(); ++k)
      std::fprintf(f, "%d%c", static_cast<int>(v[k]), (k % 16 == 15 || k + 1 == v.size()) ? '\n' : ' ');
  };
  std::fprintf(f, "LPM 1\n# fingerprint %s\nNAME %s\nSENSE %s\nROWS %d\nCOLS %d\nNNZ %zu\nOFFSET ",
               m.fingerprint_hex().c_str(), m.name.c_str(), m.sense == -1 ? "MAX" : "MIN", m.num_rows,
               m.num_cols, m.nnz());
  num(m.obj_offset);
  std::fputc('\n', f);
  dvec("OBJ", m.obj);
  dvec("COL_LOWER", m.col_lower);
  dvec("COL_UPPER", m.col_upper);
  dvec("ROW_LOWER", m.row_lower);
  dvec("ROW_UPPER", m.row_upper);
  ivec("COL_START", m.col_start);
  ivec("ROW_INDEX", m.row_index);
  dvec("VALUE", m.value);
  if (!m.is_integer.empty()) ivec("INTEGER", m.is_integer);
  if (!m.row_names.empty()) {
    std::fputs("ROW_NAMES\n", f);
    for (const auto& s : m.row_names) std::fprintf(f, "%s\n", s.c_str());
  }
  if (!m.col_names.empty()) {
    std::fputs("COL_NAMES\n", f);
    for (const auto& s : m.col_names) std::fprintf(f, "%s\n", s.c_str());
  }
  std::fputs("END\n", f);
  const bool ok = std::fclose(f) == 0;
  if (!ok) error = "write failed for " + path;
  return ok;
}

}  // namespace ps26119::io
