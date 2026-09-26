// gate.h — the verification gate applied by solve() to every answer (src/core/solve.cpp).
//
// An Optimal LP answer is re-checked on the ORIGINAL model by the in-process checker
// (core/solution_checker.h) with the verifier tolerances. Exact engines (simplex, oracle) and
// first-order runs whose tolerance is strictly tighter than the verifier's are demoted to
// NumericalError when the check fails; a looser first-order run keeps Optimal at its own
// tolerance and records check = "FAIL". MILP answers are not touched here (branch-and-bound
// and the presolve safety net verify integrality and feasibility themselves).
// Exposed for unit tests; not part of the public API.
#pragma once

#include "ps26119/model.h"
#include "ps26119/options.h"
#include "ps26119/solution.h"

namespace ps26119::core {

void gate(const Model& model, const Options& options, Solution& s);

}  // namespace ps26119::core
