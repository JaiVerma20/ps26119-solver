// main.cpp — the ps26119 command-line interface (a thin layer over the library).
//
//   ps26119 --version
//   ps26119 solve <file.lpm|file.mps> [--algorithm auto|oracle|pdlp|r2hpdhg]
//           [--precision fp64|mixed] [--gpu] [--tol 1e-8] [--time-limit s]
//           [--iteration-limit n] [--out solution.sol] [-v|-vv]
//
// Exit codes (CLAUDE.md §7): 0 optimal, 1 limit/infeasible/unbounded, 2 usage,
// 3 read error, 5 numerical error / not solved.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "io/lpm_reader.h"
#include "io/solution_reader.h"
#include "io/solution_writer.h"
#include "ps26119/solve.h"
#include "ps26119/version.h"
#ifdef PS26119_HAVE_MPS_READER
#include "io/mps_reader.h"
#endif

using namespace ps26119;

namespace {

void usage(std::FILE* f) {
  std::fprintf(f,
               "%s %s — LP solver\n"
               "usage:\n"
               "  %s --version\n"
               "  %s solve <file.lpm|file.mps> [options]\n"
               "options:\n"
               "  --algorithm auto|oracle|pdlp|r2hpdhg   (default auto)\n"
               "  --precision fp64|mixed                 (first-order engines, default fp64)\n"
               "  --gpu                                  use the CUDA backend (CUDA builds only)\n"
               "  --tol <eps>                            relative KKT tolerance (default 1e-8)\n"
               "  --time-limit <seconds>   --iteration-limit <n>\n"
               "  --out <file>                           write the solution file (tools/verify.py reads it)\n"
               "  --warm <file>                          warm start from a previous solution file (same model shape)\n"
               "  --warm-weight                          with --warm: also reuse its primal weight (faster on some\n"
               "                                         re-solves, slower on others; see bench/warm_start.py)\n"
               "  --set name=value                       expert engine knob (see Options::engine_params), repeatable\n"
               "  -v | -vv                               verbosity\n",
               kProductName, kVersion, kProductName, kProductName);
}

bool ends_with(const std::string& s, const char* suf) {
  const std::size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

int cmd_solve(int argc, char** argv) {
  std::string path, out, warm;
  bool warm_weight = false;
  // numeric flags must parse completely and be positive
  auto positive = [](const char* flag, const char* s, double& v) {
    char* end = nullptr;
    v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v > 0)) {
      std::fprintf(stderr, "%s needs a positive number, got '%s'\n", flag, s);
      std::exit(2);
    }
  };
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
    } else if (a == "--warm-weight") {
      warm_weight = true;
    } else if (a == "--set") {
      const std::string kv = next();
      const auto eq = kv.find('=');
      char* end = nullptr;
      const double v = eq == std::string::npos ? 0 : std::strtod(kv.c_str() + eq + 1, &end);
      if (eq == std::string::npos || end == kv.c_str() + eq + 1 || *end != '\0') {
        std::fprintf(stderr, "--set expects name=number, got '%s'\n", kv.c_str());
        return 2;
      }
      opt.engine_params.emplace_back(kv.substr(0, eq), v);
    } else if (a == "--out") {
      out = next();
    } else if (a == "-v") {
      opt.verbosity = 1;
    } else if (a == "-vv") {
      opt.verbosity = 2;
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

  Model model;
  if (ends_with(path, ".lpm")) {
    auto r = io::read_lpm(path, model);
    if (!r.ok) {
      std::fprintf(stderr, "read error: %s\n", r.error.c_str());
      return kExitReadError;
    }
  } else if (ends_with(path, ".mps") || ends_with(path, ".MPS")) {
#ifdef PS26119_HAVE_MPS_READER
    std::string err;
    if (!io::read_mps(path, model, err)) {
      std::fprintf(stderr, "read error: %s\n", err.c_str());
      return kExitReadError;
    }
#else
    std::fprintf(stderr,
                 "read error: the MPS reader is not in this build yet (owned by a teammate).\n"
                 "convert first:  python3 tools/mps_to_lpm.py %s\n",
                 path.c_str());
    return kExitReadError;
#endif
  } else {
    std::fprintf(stderr, "read error: unknown file type (expected .lpm or .mps)\n");
    return kExitReadError;
  }

  if (!warm.empty()) {
    Solution prev;
    std::string err;
    if (!io::read_solution(warm, prev, err)) {
      std::fprintf(stderr, "read error (warm start): %s\n", err.c_str());
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
      return 5;
    }
  }
  return exit_code(sol.status);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(stderr);
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "--version" || cmd == "version") {
    std::printf("%s %s\n", kProductName, kVersion);
    return 0;
  }
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    usage(stdout);
    return 0;
  }
  if (cmd == "solve") return cmd_solve(argc - 2, argv + 2);
  usage(stderr);
  return 2;
}
