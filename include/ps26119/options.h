// options.h — solver options passed to ps26119::solve().
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ps26119/tolerances.h"

namespace ps26119 {

enum class Algorithm {
  Auto,     // currently: r2hpdhg
  Oracle,   // dense double-double simplex — TEST ORACLE ONLY, small models
  Pdlp,     // restarted PDHG, PDLP-style (Applegate et al.)
  R2hpdhg,  // restarted Halpern PDHG with reflection (Lu & Yang; cuPDLPx)
};

enum class Precision {
  Fp64,   // everything in double
  Mixed,  // fp32 iterate + SpMV, fp64 residuals / restart decisions / final polish
};

const char* to_string(Algorithm a);
const char* to_string(Precision p);
bool algorithm_from_string(const std::string& s, Algorithm& out);
bool precision_from_string(const std::string& s, Precision& out);

struct Options {
  Algorithm algorithm = Algorithm::Auto;
  Precision precision = Precision::Fp64;
  bool use_gpu = false;                               // needs a CUDA build
  double tolerance = tol::kFirstOrderHigh;            // relative KKT target (first-order engines)
  double time_limit = 3600.0;                         // seconds, wall clock
  std::int64_t iteration_limit = 100'000'000;         // engine iterations
  int verbosity = 0;                                  // 0 silent, 1 summary, 2 progress
  // First-order engine knobs (defaults follow the cited papers; see src/pdhg/*.h).
  int ruiz_iterations = 10;
  bool pock_chambolle = true;
  int termination_check_every = 64;  // iterations between KKT evaluations
  // Warm start (first-order engines): a previous primal point x (size n) and row duals y
  // (size m, same sign convention as Solution::y). Either may be empty (then 0 is used for
  // that part); a size mismatch makes solve() return NotSolved. Typical use: re-solving
  // after a small data change (what-if, SLP step, rolling horizon) from the last solution.
  std::vector<double> warm_x, warm_y;
  // > 0: start from this ω (Solution::primal_weight). Opt-in: measured on refinery what-if
  // re-solves it helps some scenarios and hurts others (bench/warm_start.py).
  double warm_primal_weight = 0.0;
};

}  // namespace ps26119
