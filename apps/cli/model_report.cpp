// model_report.cpp — `ps26119 info` (model statistics) and `ps26119 print` (the model in
// algebraic form, exactly as the reader stored it, to compare with the MPS file).
//
// Origin: gpuopt apps/gpuopt_cli.cpp (Shivanshu Vats, c192dd0), functions print_statistics
// and print_model; adapted to the canonical Model (names are optional there).
#include "model_report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace ps26119::cli {
namespace {

std::string col_name(const Model& lp, int j) { return j < static_cast<int>(lp.col_names.size()) ? lp.col_names[j] : "x" + std::to_string(j); }
std::string row_name(const Model& lp, int i) { return i < static_cast<int>(lp.row_names.size()) ? lp.row_names[i] : "r" + std::to_string(i); }
bool is_int(const Model& lp, int j) { return j < static_cast<int>(lp.is_integer.size()) && lp.is_integer[j]; }

// Smallest and largest non-zero magnitude of a range of values.
struct MagRange {
  double lo = kInf, hi = 0.0;
  void add(double v) {
    const double a = std::fabs(v);
    if (a == 0.0 || a == kInf) return;
    lo = std::min(lo, a);
    hi = std::max(hi, a);
  }
  void print(const char* label) const {
    if (hi == 0.0) std::printf("  %-18s -\n", label);
    else std::printf("  %-18s [%.1e, %.1e]\n", label, lo, hi);
  }
};

std::string number_text(double v) {
  if (v == kInf) return "+inf";
  if (v == -kInf) return "-inf";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.10g", v);
  return buf;
}

// Appends " + 3 X" / " - X" style terms; the first term has no leading "+".
void append_term(std::string& out, double coef, const std::string& name, bool first) {
  if (coef < 0) out += first ? "-" : " - ";
  else if (!first) out += " + ";
  const double a = std::fabs(coef);
  if (a != 1.0) out += number_text(a) + " ";
  out += name;
}

// Prints the model exactly as the reader stored it, row by row, so it can be
// compared line by line with the MPS file.
}  // namespace

void print_model(const Model& lp) {
  constexpr int kMaxRows = 60;
  constexpr int kMaxTerms = 12;
  const int m = lp.num_rows;
  const int n = lp.num_cols;

  // CSC stores columns; build row lists to print constraints.
  std::vector<std::vector<std::pair<int, double>>> rows(m);
  for (int j = 0; j < n; ++j) {
    for (int k = lp.col_start[j]; k < lp.col_start[j + 1]; ++k) {
      rows[lp.row_index[k]].push_back({j, lp.value[k]});
    }
  }

  std::printf("parsed model\n");
  std::string objective;
  int terms = 0;
  for (int j = 0; j < n; ++j) {
    if (lp.obj[j] == 0.0) continue;
    if (terms++ == kMaxTerms) {
      objective += " + ...";
      break;
    }
    append_term(objective, lp.obj[j], col_name(lp, j), objective.empty());
  }
  if (objective.empty()) objective = "0";
  if (lp.obj_offset != 0.0) objective += (lp.obj_offset < 0 ? " - " : " + ") + number_text(std::fabs(lp.obj_offset));
  std::printf("  %s  %s\n",lp.sense > 0 ? "minimize" : "maximize", objective.c_str());

  std::printf("  subject to\n");
  for (int i = 0; i < std::min(m, kMaxRows); ++i) {
    std::string expr;
    for (size_t t = 0; t < rows[i].size(); ++t) {
      if (t == kMaxTerms) {
        expr += " + ...";
        break;
      }
      append_term(expr, rows[i][t].second, col_name(lp, rows[i][t].first), t == 0);
    }
    if (expr.empty()) expr = "0";
    const double lo = lp.row_lower[i], up = lp.row_upper[i];
    std::string text;
    if (lo == up) text = expr + " = " + number_text(lo);
    else if (lo == -kInf && up == kInf) text = expr + "  (free row)";
    else if (lo == -kInf) text = expr + " <= " + number_text(up);
    else if (up == kInf) text = expr + " >= " + number_text(lo);
    else text = number_text(lo) + " <= " + expr + " <= " + number_text(up);
    std::printf("    %-10s %s\n",(row_name(lp, i) + ":").c_str(), text.c_str());
  }
  if (m > kMaxRows) std::printf("    ... %d more rows\n",m - kMaxRows);

  std::printf("  bounds\n");
  for (int j = 0; j < std::min(n, kMaxRows); ++j) {
    const double lo = lp.col_lower[j], up = lp.col_upper[j];
    const std::string name = col_name(lp, j);
    std::string text;
    if (lo == up) text = name + " = " + number_text(lo) + "  (fixed)";
    else if (lo == -kInf && up == kInf) text = name + " free";
    else if (up == kInf) text = name + " >= " + number_text(lo);
    else if (lo == -kInf) text = name + " <= " + number_text(up);
    else text = number_text(lo) + " <= " + name + " <= " + number_text(up);
    if (is_int(lp, j)) text += "  (integer)";
    std::printf("    %s\n",text.c_str());
  }
  if (n > kMaxRows) std::printf("    ... %d more columns\n",n - kMaxRows);
  std::printf("\n");
}

void print_statistics(const Model& lp) {
  int eq = 0, le = 0, ge = 0, ranged = 0, free_rows = 0;
  for (int i = 0; i < lp.num_rows; ++i) {
    const double lo = lp.row_lower[i], up = lp.row_upper[i];
    if (lo == up) ++eq;
    else if (lo == -kInf && up == kInf) ++free_rows;
    else if (lo == -kInf) ++le;
    else if (up == kInf) ++ge;
    else ++ranged;
  }
  int integer = 0, binary = 0, free_cols = 0, fixed = 0, boxed = 0;
  for (int j = 0; j < lp.num_cols; ++j) {
    const double lo = lp.col_lower[j], up = lp.col_upper[j];
    if (is_int(lp, j)) (lo == 0.0 && up == 1.0) ? ++binary : ++integer;
    if (lo == -kInf && up == kInf) ++free_cols;
    else if (lo == up) ++fixed;
    else if (lo != -kInf && up != kInf) ++boxed;
  }
  MagRange mat, obj, rhs, bnd;
  for (double v : lp.value) mat.add(v);
  for (double v : lp.obj) obj.add(v);
  for (int i = 0; i < lp.num_rows; ++i) {
    rhs.add(lp.row_lower[i]);
    rhs.add(lp.row_upper[i]);
  }
  for (int j = 0; j < lp.num_cols; ++j) {
    bnd.add(lp.col_lower[j]);
    bnd.add(lp.col_upper[j]);
  }
  const double density = lp.num_rows && lp.num_cols
                             ? 100.0 * static_cast<int>(lp.nnz()) / (double(lp.num_rows) * lp.num_cols)
                             : 0.0;

  std::printf("model statistics\n");
  std::printf("  name               %s\n", lp.name.empty() ? "(none)" : lp.name.c_str());
  std::printf("  sense              %s\n", lp.sense > 0 ? "minimize" : "maximize");
  std::printf("  rows               %d  (E %d, L %d, G %d, ranged %d, free %d)\n", lp.num_rows,
              eq, le, ge, ranged, free_rows);
  std::printf("  columns            %d  (continuous %d, integer %d, binary %d)\n", lp.num_cols,
              lp.num_cols - integer - binary, integer, binary);
  std::printf("  column bounds      free %d, fixed %d, boxed %d\n", free_cols, fixed, boxed);
  std::printf("  nonzeros           %d  (density %.3f%%)\n", static_cast<int>(lp.nnz()), density);
  mat.print("|matrix coeff|");
  obj.print("|objective coeff|");
  rhs.print("|row bounds|");
  bnd.print("|column bounds|");
  if (mat.hi > 0.0) std::printf("  matrix dynamism    %.1e  (max/min coefficient ratio)\n", mat.hi / mat.lo);
  if (lp.obj_offset != 0.0) std::printf("  objective offset   %.10g\n", lp.obj_offset);
}

}  // namespace ps26119::cli
