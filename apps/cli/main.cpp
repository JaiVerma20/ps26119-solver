// main.cpp — the ps26119 command-line interface (a thin layer over the library).
//
//   ps26119 --version
//   ps26119 info  <file.mps|file.lpm>     model statistics            (from gpuopt)
//   ps26119 print <file.mps|file.lpm>     the model in algebraic form (from gpuopt)
//   ps26119 lu-bench <file> [--trials n] [--updates n]  sparse LU on bases of A (from gpuopt)
//   ps26119 solve <file.lpm|file.mps> [--algorithm auto|oracle|pdlp|r2hpdhg|simplex]
//           [--precision fp64|mixed] [--gpu] [--tol 1e-8] [--time-limit s]
//           [--iteration-limit n] [--out solution.sol] [-v|-vv|-vvv]
//
// Exit codes (CLAUDE.md §7): 0 optimal, 1 limit/infeasible/unbounded, 2 usage,
// 3 read error, 4 cannot write the output, 5 numerical error / not solved.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "io/lpm_reader.h"
#include "io/mps_reader.h"
#include "io/solution_reader.h"
#include "io/solution_writer.h"
#include "ps26119/batch.h"
#include "ps26119/ranging.h"
#include "ps26119/solve.h"
#include "ps26119/version.h"
#include "ps26119_build_info.h"
#include "lu_bench.h"
#include "model_report.h"

using namespace ps26119;

namespace {

bool write_ranging(const std::string& path, const Model& model, const RangingResult& rg);  // below

void usage(std::FILE* f) {
  std::fprintf(f,
               "%s %s — LP solver\n"
               "usage:\n"
               "  %s --version\n"
               "  %s solve <file.mps|file.lpm> [options]\n"
               "  %s info  <file.mps|file.lpm>            model statistics\n"
               "  %s print <file.mps|file.lpm>            the model as the reader understood it\n"
               "  %s lu-bench <file> [--trials n] [--updates n]   sparse LU on bases built from A\n"
               "  %s batch <base> <scenario>... [--out-dir d] [--tol e] [--time-limit s] [--set k=v]\n"
               "        scenarios share the base matrix; their objective and bounds may differ\n"
               "solve options:\n"
               "  --algorithm auto|simplex|r2hpdhg|pdlp|oracle   (default auto: simplex for small models,\n"
               "                                         r2hpdhg for large ones, --gpu or --warm)\n"
               "                                         simplex: revised primal simplex (vertex, exact duals)\n"
               "                                         oracle: dense double-double simplex, small models only\n"
               "  --precision fp64|mixed                 (first-order engines, default fp64)\n"
               "  --gpu                                  use the CUDA backend (CUDA builds only)\n"
               "  --threads <n>                          CPU threads (default 1, 0 = all cores; same results)\n"
               "  --no-presolve                          skip the (default) safe presolve + postsolve\n"
               "  --tol <eps>                            relative KKT tolerance, first-order engines (default 1e-8)\n"
               "  --time-limit <seconds>   --iteration-limit <n>\n"
               "  --out <file>                           write the solution file (tools/verify.py reads it)\n"
               "  --ranging <file>                       cost and right-hand-side ranging at the optimal vertex (CSV;\n"
               "                                         LPs solved by the simplex / oracle)\n"
               "  --warm <file>                          warm start from a previous solution file (same model shape)\n"
               "  --warm-weight                          with --warm: also reuse its primal weight (faster on some\n"
               "                                         re-solves, slower on others; see bench/warm_start.py)\n"
               "  --set name=value                       expert engine knob (see Options::engine_params), repeatable\n"
               "  -v | -vv | -vvv                        verbosity (-vvv: r2hpdhg per-check TRACE lines)\n"
               "exit codes: 0 optimal, 1 infeasible/unbounded/limit, 2 usage, 3 read error,\n"
               "            4 cannot write output, 5 numerical error / not solved\n",
               kProductName, kVersion, kProductName, kProductName, kProductName, kProductName, kProductName,
               kProductName);
}

constexpr int kExitWriteError = 4;

// Numeric flags parse completely or the command stops with a usage error (exit 2).
double parse_number(const char* flag, const char* s, bool positive_only) {
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  const bool ok = end != s && *end == '\0' && std::isfinite(v) && (positive_only ? v > 0 : v >= 0);
  if (!ok) {
    std::fprintf(stderr, "%s needs a %s number, got '%s'\n", flag, positive_only ? "positive" : "non-negative", s);
    std::exit(2);
  }
  return v;
}

int parse_threads(const char* s) {
  const double v = parse_number("--threads", s, false);
  if (v != std::floor(v) || v > 4096) {
    std::fprintf(stderr, "--threads needs an integer between 0 and 4096, got '%s'\n", s);
    std::exit(2);
  }
  return static_cast<int>(v);
}

// name=number for --set; exits with a usage error otherwise.
std::pair<std::string, double> parse_set(const std::string& kv) {
  const auto eq = kv.find('=');
  char* end = nullptr;
  const double v = eq == std::string::npos ? 0 : std::strtod(kv.c_str() + eq + 1, &end);
  if (eq == std::string::npos || eq == 0 || end == kv.c_str() + eq + 1 || *end != '\0') {
    std::fprintf(stderr, "--set expects name=number, got '%s'\n", kv.c_str());
    std::exit(2);
  }
  return {kv.substr(0, eq), v};
}

// Fails fast when the output file cannot be created, instead of after a long solve. A file
// this check creates is removed again (the solution is written only at the end).
bool output_writable(const std::string& path) {
  std::FILE* probe = std::fopen(path.c_str(), "r");
  const bool existed = probe != nullptr;
  if (probe) std::fclose(probe);
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return false;
  std::fclose(f);
  if (!existed) std::remove(path.c_str());
  return true;
}

bool ends_with(const std::string& s, const char* suf) {
  const std::size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Reads .lpm (our generated-model format) or .mps (the integrated reader). Reader warnings go
// to stderr. Returns false (message printed) on failure.
bool load_model(const std::string& path, Model& model) {
  if (ends_with(path, ".lpm")) {
    auto r = io::read_lpm(path, model);
    if (!r.ok) std::fprintf(stderr, "read error: %s\n", r.error.c_str());
    return r.ok;
  }
  if (ends_with(path, ".mps") || ends_with(path, ".MPS")) {
    std::string err;
    std::vector<std::string> warnings;
    if (!io::read_mps(path, model, err, &warnings)) {
      std::fprintf(stderr, "read error: %s\n", err.c_str());
      return false;
    }
    for (const auto& w : warnings) std::fprintf(stderr, "reader warning: %s\n", w.c_str());
    return true;
  }
  std::fprintf(stderr, "read error: unknown file type (expected .mps or .lpm)\n");
  return false;
}

int cmd_inspect(const std::string& what, int argc, char** argv) {
  if (argc < 1) {
    usage(stderr);
    return 2;
  }
  Model model;
  if (!load_model(argv[0], model)) return kExitReadError;
  if (what == "info") {
    cli::print_statistics(model);
    std::printf("  fingerprint        %s\n", model.fingerprint_hex().c_str());
    if (auto c = model.crossed_bounds(); !c.empty()) std::printf("  note               %s (infeasible)\n", c.c_str());
    return 0;
  }
  if (what == "print") {
    cli::print_model(model);
    return 0;
  }
  std::printf("model      %s  %d rows, %d columns, %zu nonzeros\n", model.name.c_str(), model.num_rows,
              model.num_cols, model.nnz());
  cli::LuBenchOptions lo;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if ((a == "--trials" || a == "--updates") && i + 1 < argc) {
      (a == "--trials" ? lo.trials : lo.updates) = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  return cli::run_lu_bench(model, lo);
}

int cmd_solve(int argc, char** argv) {
  std::string path, out, warm;
  std::string ranging_out;  // --ranging: cost / rhs ranging CSV (optimal LP at a vertex)
  bool warm_weight = false;
  auto positive = [](const char* flag, const char* s, double& v) { v = parse_number(flag, s, true); };
  Options opt;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--algorithm") {
      if (!algorithm_from_string(next(), opt.algorithm)) {
        std::fprintf(stderr, "unknown algorithm\n");
        return 2;
      }
    } else if (a == "--precision") {
      if (!precision_from_string(next(), opt.precision)) {
        std::fprintf(stderr, "unknown precision\n");
        return 2;
      }
    } else if (a == "--gpu") {
      opt.use_gpu = true;
    } else if (a == "--tol") {
      positive("--tol", next(), opt.tolerance);
    } else if (a == "--time-limit") {
      positive("--time-limit", next(), opt.time_limit);
    } else if (a == "--iteration-limit") {
      double v;
      positive("--iteration-limit", next(), v);
      opt.iteration_limit = static_cast<std::int64_t>(v);
    } else if (a == "--warm") {
      warm = next();
    } else if (a == "--threads") {
      opt.threads = parse_threads(next());
    } else if (a == "--presolve") {
      opt.presolve = true;
    } else if (a == "--no-presolve") {
      opt.presolve = false;
    } else if (a == "--warm-weight") {
      warm_weight = true;
    } else if (a == "--set") {
      opt.engine_params.push_back(parse_set(next()));
    } else if (a == "--out") {
      out = next();
    } else if (a == "--ranging") {
      ranging_out = next();
    } else if (a == "-v") {
      opt.verbosity = 1;
    } else if (a == "-vv") {
      opt.verbosity = 2;
    } else if (a == "-vvv") {  // -vv plus a per-check TRACE line (r²HPDHG diagnostics)
      opt.verbosity = 3;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    } else if (path.empty()) {
      path = a;
    } else {
      std::fprintf(stderr, "more than one input file\n");
      return 2;
    }
  }
  if (path.empty()) {
    usage(stderr);
    return 2;
  }
  if (!out.empty() && !output_writable(out)) {
    std::fprintf(stderr, "cannot write the solution file '%s'\n", out.c_str());
    return kExitWriteError;
  }
  if (!ranging_out.empty() && !output_writable(ranging_out)) {
    std::fprintf(stderr, "cannot write the ranging file '%s'\n", ranging_out.c_str());
    return kExitWriteError;
  }

  Model model;
  if (!load_model(path, model)) return kExitReadError;

  if (!warm.empty()) {
    Solution prev;
    std::string err;
    if (!io::read_solution(warm, prev, err)) {
      std::fprintf(stderr, "read error (warm start): %s\n", err.c_str());
      return kExitReadError;
    }
    for (const auto* v : {&prev.x, &prev.y})
      for (double a : *v)
        if (!std::isfinite(a)) {
          std::fprintf(stderr, "read error (warm start): %s contains non-finite values\n", warm.c_str());
          return kExitReadError;
        }
    if (static_cast<int>(prev.x.size()) != model.num_cols || static_cast<int>(prev.y.size()) != model.num_rows) {
      std::fprintf(stderr, "warm start %s does not match the model's size\n", warm.c_str());
      return 2;
    }
    opt.warm_x = std::move(prev.x);
    opt.warm_y = std::move(prev.y);
    if (warm_weight && prev.primal_weight > 0) opt.warm_primal_weight = prev.primal_weight;
  }
  const Solution sol = solve(model, opt);
  std::printf("model      %s  rows %d  cols %d  nnz %zu  fingerprint %s\n", model.name.c_str(), model.num_rows,
              model.num_cols, model.nnz(), sol.model_fingerprint.c_str());
  std::printf("status     %s\n", to_string(sol.status));
  std::printf("engine     %s (%s)\n", sol.engine.c_str(), sol.precision.c_str());
  std::printf("objective  %.12g\n", sol.objective);
  std::printf("residuals  primal %.2e  dual %.2e  gap %.2e\n", sol.primal_residual, sol.dual_residual, sol.gap);
  const bool claim = sol.status == Status::Infeasible || sol.status == Status::Unbounded;
  if (claim) {
    std::printf("check      %s\n", sol.check.empty() ? "NOT CERTIFIED (no certificate)"
                                                     : (sol.check + "  (" + (sol.status == Status::Infeasible ? "Farkas" : "ray") +
                                                        " certificate verified on the original model)").c_str());
  } else if (!sol.check.empty() && sol.engine.rfind("branch-and-bound", 0) == 0) {
    std::printf("check      %s  (in-process, original model: bounds, rows, integrality of the point)\n", sol.check.c_str());
  } else if (!sol.check.empty()) {
    std::printf("check      %s  (in-process, original model: primal %.1e  dual %.1e  gap %.1e)\n", sol.check.c_str(),
                sol.check_primal, sol.check_dual, sol.check_gap);
  }
  if (sol.certified_bound == sol.certified_bound && !claim) {  // a bound on the optimum: not for Infeasible/Unbounded
    if (std::isfinite(sol.certified_bound))
      std::printf("certified  %s %.12g  (rounding-proof %s bound from y)\n", model.sense > 0 ? "optimum >=" : "optimum <=",
                  sol.certified_bound, model.sense > 0 ? "lower" : "upper");
    else
      std::printf("certified  none  (y gives no finite rounding-proof bound)\n");
  }
  std::printf("iterations %lld   seconds %.3f   (setup %.3f)\n", static_cast<long long>(sol.iterations), sol.seconds,
              sol.setup_seconds);
  if (sol.iterations_to_fast >= 0)
    std::printf("to 1e-4    iterations %lld   seconds %.3f\n", static_cast<long long>(sol.iterations_to_fast),
                sol.seconds_to_fast);
  if (!sol.message.empty()) std::printf("message    %s\n", sol.message.c_str());
  if (!out.empty()) {
    std::string err;
    if (!io::write_solution(out, model, sol, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return kExitWriteError;
    }
  }
  if (!ranging_out.empty()) {
    const RangingResult rg = compute_ranging(model, sol);
    if (!rg.ok) {
      std::printf("ranging    refused: %s\n", rg.message.c_str());
    } else if (!write_ranging(ranging_out, model, rg)) {
      std::fprintf(stderr, "cannot write the ranging file '%s'\n", ranging_out.c_str());
      return kExitWriteError;
    } else {
      std::printf("ranging    %zu costs, %zu rows -> %s%s%s\n", rg.cols.size(), rg.rows.size(), ranging_out.c_str(),
                  rg.message.empty() ? "" : "  (", rg.message.empty() ? "" : (rg.message + ")").c_str());
    }
  }
  return exit_code(sol.status);
}

// a name as one CSV field: quoted (with doubled quotes) when it holds a comma, a quote or a blank
std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\" \t") == std::string::npos) return s;
  std::string q = "\"";
  for (char ch : s) q += ch == '"' ? std::string("\"\"") : std::string(1, ch);
  return q + "\"";
}

// --ranging: one CSV row per objective coefficient and per row (see include/ps26119/ranging.h)
bool write_ranging(const std::string& path, const Model& model, const RangingResult& rg) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return false;
  std::fprintf(f, "kind,index,name,value,status,lower,upper,dual_or_reduced_cost\n");
  for (const CostRange& c : rg.cols) {
    const std::string nm = c.col < static_cast<int>(model.col_names.size()) ? csv_field(model.col_names[c.col]) : "";
    std::fprintf(f, "cost,%d,%s,%.17g,%s,%.17g,%.17g,%.17g\n", c.col, nm.c_str(), c.cost, c.basic ? "basic" : "nonbasic", c.lower,
                 c.upper, c.reduced_cost);
  }
  for (const RhsRange& r : rg.rows) {
    const std::string nm = r.row < static_cast<int>(model.row_names.size()) ? csv_field(model.row_names[r.row]) : "";
    const char* st = r.binding > 0 ? "upper_binding" : r.binding < 0 ? "lower_binding" : "not_binding";
    std::fprintf(f, "rhs,%d,%s,%.17g,%s,%.17g,%.17g,%.17g\n", r.row, nm.c_str(), r.bound, st, r.lower, r.upper, r.dual);
  }
  return std::fclose(f) == 0;
}

int cmd_batch(int argc, char** argv) {
  std::vector<std::string> files;
  std::string out_dir;
  Options opt;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--out-dir") out_dir = next();
    else if (a == "--tol") opt.tolerance = parse_number("--tol", next(), true);
    else if (a == "--time-limit") opt.time_limit = parse_number("--time-limit", next(), true);
    else if (a == "--iteration-limit")
      opt.iteration_limit = static_cast<std::int64_t>(parse_number("--iteration-limit", next(), true));
    else if (a == "-v") opt.verbosity = 1;
    else if (a == "-vv") opt.verbosity = 2;
    else if (a == "-vvv") opt.verbosity = 3;
    else if (a == "--threads") opt.threads = parse_threads(next());
    else if (a == "--set") opt.engine_params.push_back(parse_set(next()));
    else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    } else {
      files.push_back(a);
    }
  }
  if (files.size() < 2 || !(opt.tolerance > 0) || !(opt.time_limit > 0)) {
    usage(stderr);
    return 2;
  }
  Model base;
  if (!load_model(files[0], base)) return kExitReadError;
  std::vector<Scenario> sc;
  std::vector<std::string> names, model_names;
  for (std::size_t f = 1; f < files.size(); ++f) {
    Model s;
    if (!load_model(files[f], s)) return kExitReadError;
    if (s.num_rows != base.num_rows || s.num_cols != base.num_cols || s.sense != base.sense ||
        s.col_start != base.col_start || s.row_index != base.row_index || s.value != base.value) {
      std::fprintf(stderr, "%s: matrix or sense differs from the base model (batch needs a shared A)\n",
                   files[f].c_str());
      return 2;
    }
    sc.push_back(Scenario{s.obj, s.col_lower, s.col_upper, s.row_lower, s.row_upper});
    const auto slash = files[f].find_last_of('/');
    std::string nm = files[f].substr(slash == std::string::npos ? 0 : slash + 1);
    names.push_back(nm.substr(0, nm.rfind('.')));
    model_names.push_back(s.name);
  }
  const auto sols = solve_batch(base, sc, opt);
  int worst = 0;
  for (std::size_t k = 0; k < sols.size(); ++k) {
    const Solution& s = sols[k];
    std::printf("%-28s %-14s obj %.12g  it %lld  t %.3fs%s%s\n", names[k].c_str(), to_string(s.status), s.objective,
                static_cast<long long>(s.iterations), s.seconds, s.message.empty() ? "" : "  ", s.message.c_str());
    worst = std::max(worst, exit_code(s.status));
    if (!out_dir.empty()) {
      // The file describes the scenario: its own name, and (from solve_batch) the fingerprint
      // of the scenario model. The matrix and the row / column names are the base's (shared).
      base.name = model_names[k];
      std::string err;
      if (!io::write_solution(out_dir + "/" + names[k] + ".sol", base, s, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        worst = std::max(worst, kExitWriteError);
      }
    }
  }
  return worst;
}

}  // namespace

int run_cli(int argc, char** argv);

// Last line of defence: an exception (e.g. out of memory while reading a huge file) ends the
// program with a message and exit code 5, never with std::terminate.
int main(int argc, char** argv) {
  try {
    return run_cli(argc, argv);
  } catch (const std::bad_alloc&) {
    std::fprintf(stderr, "fatal: out of memory\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
  }
  return 5;
}

int run_cli(int argc, char** argv) {
  if (argc < 2) {
    usage(stderr);
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "--version" || cmd == "version") {
    std::printf("%s %s (git %s)\n", kProductName, kVersion, PS26119_GIT_HASH);
    return 0;
  }
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    usage(stdout);
    return 0;
  }
  if (cmd == "solve") return cmd_solve(argc - 2, argv + 2);
  if (cmd == "batch") return cmd_batch(argc - 2, argv + 2);
  if (cmd == "info" || cmd == "print" || cmd == "lu-bench") return cmd_inspect(cmd, argc - 2, argv + 2);
  usage(stderr);
  return 2;
}
