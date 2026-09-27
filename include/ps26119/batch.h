// batch.h — solve many LPs that share one constraint matrix in one pass (CLAUDE.md §9 idea 4).
//
// Typical use: crude-price or demand scenarios of one refinery plan, SLP steps, what-if
// studies. All scenarios share A (hence scaling, ‖A‖₂ and the sparse structure); each may
// change the objective and any bounds. The engine is r²HPDHG run in lockstep on all
// scenarios: every iteration reads A once and multiplies it with K vectors (SpMM), which
// is the memory-bandwidth win on CPUs and GPUs. Each scenario keeps its own primal weight,
// restarts and termination test, and freezes when it converges.
//
// Current limits: CPU, fp64, no infeasibility detection (a scenario that does not converge
// reports IterationLimit / TimeLimit), no warm start.
#pragma once

#include <vector>

#include "ps26119/model.h"
#include "ps26119/options.h"
#include "ps26119/solution.h"

namespace ps26119 {

// Differences from the base model; an empty vector means "same as the base".
struct Scenario {
  std::vector<double> obj;
  std::vector<double> col_lower, col_upper;
  std::vector<double> row_lower, row_upper;
};

// One Solution per scenario, in order. Each Solution's model_fingerprint is that of its
// scenario model (the base matrix with the scenario's objective and bounds). Uses options.tolerance, time_limit (for the whole
// batch), iteration_limit, termination_check_every, verbosity and engine_params.
std::vector<Solution> solve_batch(const Model& base, const std::vector<Scenario>& scenarios,
                                  const Options& options = {});

}  // namespace ps26119
