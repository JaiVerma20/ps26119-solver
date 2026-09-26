// gpuopt - command-line front end.
//
//   gpuopt model.mps                 read, solve with the revised simplex, verify
//   gpuopt model.mps --method oracle solve with the dense reference oracle instead
//   gpuopt model.mps --info          read only, print model statistics
//   gpuopt model.mps --expect -464.7531428571
//
// Exit codes: 0 success, 1 usage / read error, 2 solver did not reach a
// definitive status, 3 verification failed (checker FAIL or --expect mismatch).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "gpuopt/dense_oracle.hpp"
#include "gpuopt/mps_reader.hpp"
#include "gpuopt/simplex/primal_simplex.hpp"
#include "gpuopt/solution_checker.hpp"
#include "lu_bench.hpp"

using namespace gpuopt;

namespace {

struct CliOptions {
  std::string path;
  MpsFormat format = MpsFormat::kAuto;
  bool info_only = false;
  bool print_model = false;
  bool lu_bench = false;
  LuBenchOptions lu;
  bool print_solution = false;
  bool print_duals = false;
  bool has_expect = false;
  double expect = 0.0;
  bool use_oracle = false;
  SimplexOptions simplex;
  long long max_iterations = 50'000'000;
};

void print_usage() {
  std::printf(
      "usage: gpuopt <model.mps> [options]\n"
      "  --info               read the model and print statistics only\n"
      "  --print-model        print the model as the reader understood it\n"
      "  --lu-bench           Layer 2: factorize bases built from the model's matrix\n"
      "  --trials N           random bases per structural fraction (default 3)\n"
      "  --updates N          PFI updates in the LU update test (default 100)\n"
      "  --format auto|free|fixed   MPS dialect (default auto)\n"
      "  --solution           print the non-zero primal values\n"
      "  --duals              print the non-zero row duals\n"
      "  --expect VALUE       compare the optimum with a known value (rel. tol 1e-6)\n"
      "  --method simplex|oracle    solver (default simplex; oracle = dense reference)\n"
      "  --pricing devex|dantzig    simplex pricing rule (default devex)\n"
      "  --no-scale           disable matrix scaling\n"
      "  --no-perturb         disable cost perturbation\n"
      "  --log N              print simplex progress every N iterations\n"
      "  --time-limit S       stop after S seconds (default 3600)\n"
      "  --max-iter N         iteration limit\n");
}

bool parse_args(int argc, char** argv, CliOptions& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
        return nullptr;
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      return false;
    } else if (a == "--info") {
      o.info_only = true;
    } else if (a == "--print-model") {
      o.print_model = true;
    } else if (a == "--lu-bench") {
      o.lu_bench = true;
    } else if (a == "--trials" || a == "--updates") {
      const char* v = next();
      if (!v) return false;
      (a == "--trials" ? o.lu.trials : o.lu.updates) = std::atoi(v);
    } else if (a == "--solution") {
      o.print_solution = true;
    } else if (a == "--duals") {
      o.print_duals = true;
    } else if (a == "--format") {
      const char* v = next();
      if (!v) return false;
      const std::string f = v;
      if (f == "auto") o.format = MpsFormat::kAuto;
      else if (f == "free") o.format = MpsFormat::kFree;
      else if (f == "fixed") o.format = MpsFormat::kFixed;
      else {
        std::fprintf(stderr, "error: unknown format '%s'\n", v);
        return false;
      }
    } else if (a == "--expect") {
      const char* v = next();
      if (!v) return false;
      o.has_expect = true;
      o.expect = std::strtod(v, nullptr);
    } else if (a == "--max-iter") {
      const char* v = next();
      if (!v) return false;
      o.max_iterations = std::strtoll(v, nullptr, 10);
    } else if (a == "--method") {
      const char* v = next();
      if (!v) return false;
      const std::string mth = v;
      if (mth != "simplex" && mth != "oracle") {
        std::fprintf(stderr, "error: unknown method '%s'\n", v);
        return false;
      }
      o.use_oracle = mth == "oracle";
    } else if (a == "--pricing") {
      const char* v = next();
      if (!v) return false;
      const std::string p = v;
      if (p != "devex" && p != "dantzig") {
        std::fprintf(stderr, "error: unknown pricing '%s'\n", v);
        return false;
      }
      o.simplex.pricing = p == "devex" ? Pricing::kDevex : Pricing::kDantzig;
    } else if (a == "--no-scale") {
      o.simplex.scale = false;
    } else if (a == "--no-perturb") {
      o.simplex.perturb = false;
    } else if (a == "--log" || a == "--time-limit") {
      const char* v = next();
      if (!v) return false;
      if (a == "--log") o.simplex.log_every = std::atoi(v);
      else o.simplex.time_limit_seconds = std::strtod(v, nullptr);
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
      return false;
    } else if (o.path.empty()) {
      o.path = a;
    } else {
      std::fprintf(stderr, "error: more than one input file given\n");
      return false;
    }
  }
  return !o.path.empty();
}

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
void print_model(const LpProblem& lp) {
  constexpr int kMaxRows = 60;
  constexpr int kMaxTerms = 12;
  const int m = lp.num_rows();
  const int n = lp.num_cols();

  // CSC stores columns; build row lists to print constraints.
  std::vector<std::vector<std::pair<int, double>>> rows(m);
  for (int j = 0; j < n; ++j) {
    for (int k = lp.A.col_start[j]; k < lp.A.col_start[j + 1]; ++k) {
      rows[lp.A.row_index[k]].push_back({j, lp.A.value[k]});
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
    append_term(objective, lp.obj[j], lp.col_names[j], objective.empty());
  }
  if (objective.empty()) objective = "0";
  if (lp.obj_offset != 0.0) objective += (lp.obj_offset < 0 ? " - " : " + ") + number_text(std::fabs(lp.obj_offset));
  std::printf("  %s  %s\n",lp.sense == ObjSense::kMinimize ? "minimize" : "maximize", objective.c_str());

  std::printf("  subject to\n");
  for (int i = 0; i < std::min(m, kMaxRows); ++i) {
    std::string expr;
    for (size_t t = 0; t < rows[i].size(); ++t) {
      if (t == kMaxTerms) {
        expr += " + ...";
        break;
      }
      append_term(expr, rows[i][t].second, lp.col_names[rows[i][t].first], t == 0);
    }
    if (expr.empty()) expr = "0";
    const double lo = lp.row_lower[i], up = lp.row_upper[i];
    std::string text;
    if (lo == up) text = expr + " = " + number_text(lo);
    else if (lo == -kInf && up == kInf) text = expr + "  (free row)";
    else if (lo == -kInf) text = expr + " <= " + number_text(up);
    else if (up == kInf) text = expr + " >= " + number_text(lo);
    else text = number_text(lo) + " <= " + expr + " <= " + number_text(up);
    std::printf("    %-10s %s\n",(lp.row_names[i] + ":").c_str(), text.c_str());
  }
  if (m > kMaxRows) std::printf("    ... %d more rows\n",m - kMaxRows);

  std::printf("  bounds\n");
  for (int j = 0; j < std::min(n, kMaxRows); ++j) {
    const double lo = lp.col_lower[j], up = lp.col_upper[j];
    const std::string& name = lp.col_names[j];
    std::string text;
    if (lo == up) text = name + " = " + number_text(lo) + "  (fixed)";
    else if (lo == -kInf && up == kInf) text = name + " free";
    else if (up == kInf) text = name + " >= " + number_text(lo);
    else if (lo == -kInf) text = name + " <= " + number_text(up);
    else text = number_text(lo) + " <= " + name + " <= " + number_text(up);
    if (lp.is_integer[j]) text += "  (integer)";
    std::printf("    %s\n",text.c_str());
  }
  if (n > kMaxRows) std::printf("    ... %d more columns\n",n - kMaxRows);
  std::printf("\n");
}

void print_statistics(const LpProblem& lp) {
  int eq = 0, le = 0, ge = 0, ranged = 0, free_rows = 0;
  for (int i = 0; i < lp.num_rows(); ++i) {
    const double lo = lp.row_lower[i], up = lp.row_upper[i];
    if (lo == up) ++eq;
    else if (lo == -kInf && up == kInf) ++free_rows;
    else if (lo == -kInf) ++le;
    else if (up == kInf) ++ge;
    else ++ranged;
  }
  int integer = 0, binary = 0, free_cols = 0, fixed = 0, boxed = 0;
  for (int j = 0; j < lp.num_cols(); ++j) {
    const double lo = lp.col_lower[j], up = lp.col_upper[j];
    if (lp.is_integer[j]) (lo == 0.0 && up == 1.0) ? ++binary : ++integer;
    if (lo == -kInf && up == kInf) ++free_cols;
    else if (lo == up) ++fixed;
    else if (lo != -kInf && up != kInf) ++boxed;
  }
  MagRange mat, obj, rhs, bnd;
  for (double v : lp.A.value) mat.add(v);
  for (double v : lp.obj) obj.add(v);
  for (int i = 0; i < lp.num_rows(); ++i) {
    rhs.add(lp.row_lower[i]);
    rhs.add(lp.row_upper[i]);
  }
  for (int j = 0; j < lp.num_cols(); ++j) {
    bnd.add(lp.col_lower[j]);
    bnd.add(lp.col_upper[j]);
  }
  const double density = lp.num_rows() && lp.num_cols()
                             ? 100.0 * lp.A.nnz() / (double(lp.num_rows()) * lp.num_cols())
                             : 0.0;

  std::printf("model statistics\n");
  std::printf("  name               %s\n", lp.name.empty() ? "(none)" : lp.name.c_str());
  std::printf("  sense              %s\n", lp.sense == ObjSense::kMinimize ? "minimize" : "maximize");
  std::printf("  rows               %d  (E %d, L %d, G %d, ranged %d, free %d)\n", lp.num_rows(),
              eq, le, ge, ranged, free_rows);
  std::printf("  columns            %d  (continuous %d, integer %d, binary %d)\n", lp.num_cols(),
              lp.num_cols() - integer - binary, integer, binary);
  std::printf("  column bounds      free %d, fixed %d, boxed %d\n", free_cols, fixed, boxed);
  std::printf("  nonzeros           %d  (density %.3f%%)\n", lp.A.nnz(), density);
  mat.print("|matrix coeff|");
  obj.print("|objective coeff|");
  rhs.print("|row bounds|");
  bnd.print("|column bounds|");
  if (mat.hi > 0.0) std::printf("  matrix dynamism    %.1e  (max/min coefficient ratio)\n", mat.hi / mat.lo);
  if (lp.obj_offset != 0.0) std::printf("  objective offset   %.10g\n", lp.obj_offset);
}

}  // namespace

int main(int argc, char** argv) {
  CliOptions opt;
  if (!parse_args(argc, argv, opt)) {
    print_usage();
    return 1;
  }

  std::printf("gpuopt 0.2  -  indigenous optimization solver\n\n");

  // ---- read
  MpsReadResult read;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    MpsReadOptions ro;
    ro.format = opt.format;
    read = read_mps_file(opt.path, ro);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error reading %s: %s\n", opt.path.c_str(), e.what());
    return 1;
  }
  const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const LpProblem& lp = read.problem;

  std::printf("reading    %s\n", opt.path.c_str());
  std::printf("           %d rows, %d columns, %d nonzeros, %s format, %.3f s\n", lp.num_rows(),
              lp.num_cols(), lp.A.nnz(), read.format_used == MpsFormat::kFixed ? "fixed" : "free",
              read_s);
  const size_t shown = std::min<size_t>(read.warnings.size(), 10);
  for (size_t k = 0; k < shown; ++k) std::printf("warning    %s\n", read.warnings[k].c_str());
  if (read.warnings.size() > shown) {
    std::printf("warning    ... %zu more\n", read.warnings.size() - shown);
  }
  std::printf("\n");

  if (opt.print_model) print_model(lp);
  if (opt.lu_bench) return run_lu_bench(lp, opt.lu);
  if (opt.info_only) {
    print_statistics(lp);
    return 0;
  }

  // ---- solve
  SolveResult res;
  if (opt.use_oracle) {
    DenseOracleOptions so;
    so.max_iterations = opt.max_iterations;
    std::printf("method     dense tableau simplex, Bland's rule (reference oracle)\n");
    res = solve_dense_oracle(lp, so);
  } else {
    SimplexOptions so = opt.simplex;
    so.max_iterations = opt.max_iterations;
    std::printf("method     revised primal simplex, %s pricing, sparse LU%s%s\n",
                so.pricing == Pricing::kDevex ? "Devex" : "Dantzig", so.scale ? ", scaled" : "",
                so.perturb ? ", perturbed" : "");
    res = solve_primal_simplex(lp, so);
  }

  std::printf("status     %s\n", to_string(res.status));
  if (!res.message.empty()) std::printf("note       %s\n", res.message.c_str());
  std::printf("iterations %lld\n", res.iterations);
  std::printf("time       %.3f s\n", res.seconds);
  if (res.status != SolveStatus::kOptimal) {
    const bool definitive =
        res.status == SolveStatus::kInfeasible || res.status == SolveStatus::kUnbounded;
    return definitive ? 0 : 2;
  }
  std::printf("objective  %.10e\n", res.objective);
  if (lp.obj_offset != 0.0) {
    std::printf("offset     %.10e  (constant term included above; without it: %.10e)\n", lp.obj_offset,
                res.objective - lp.obj_offset);
  }

  // ---- verify independently
  const CheckReport rep = check_solution(lp, res.x, res.row_dual);
  std::printf("checker    %s\n", rep.summary().c_str());
  std::printf("           primal objective %.10e   dual objective %.10e\n", rep.primal_objective,
              rep.dual_objective);
  int exit_code = rep.passed() ? 0 : 3;

  if (opt.has_expect) {
    const double rel = std::fabs(res.objective - opt.expect) / (1.0 + std::fabs(opt.expect));
    const bool ok = rel <= 1e-6;
    std::printf("expected   %.10e   relative error %.1e  => %s\n", opt.expect, rel,
                ok ? "MATCH" : "MISMATCH");
    if (!ok) exit_code = 3;
  }

  if (opt.print_solution) {
    std::printf("\nprimal solution (non-zero values)\n");
    for (int j = 0; j < lp.num_cols(); ++j) {
      if (std::fabs(res.x[j]) > 1e-12) std::printf("  %-16s %.10g\n", lp.col_names[j].c_str(), res.x[j]);
    }
  }
  if (opt.print_duals) {
    std::printf("\nrow duals (non-zero values)\n");
    for (int i = 0; i < lp.num_rows(); ++i) {
      if (std::fabs(res.row_dual[i]) > 1e-12) {
        std::printf("  %-16s %.10g\n", lp.row_names[i].c_str(), res.row_dual[i]);
      }
    }
  }
  return exit_code;
}
