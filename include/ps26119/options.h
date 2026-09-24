// options.h — solver options passed to ps26119::solve().
#pragma once

#include <cstdint>
#include <string>

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
};

}  // namespace ps26119
