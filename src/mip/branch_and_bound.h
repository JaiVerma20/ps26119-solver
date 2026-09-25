// branch_and_bound.h — prototype branch-and-bound for SMALL mixed-integer LPs.
//
// Node relaxations are solved by the dense double-double simplex oracle (exact to ~1e-30,
// slow, dense), so this is a correctness prototype for small models — not yet a MIP solver
// for MIPLIB-scale problems (that needs the sparse dual simplex with warm-started nodes).
//
// Algorithm (textbook; e.g. Wolsey, "Integer Programming", ch. 7):
//   * node = the model with tightened integer column bounds; LP relaxation by the oracle;
//   * prune a node if its LP is infeasible or its bound ≥ incumbent − gap tolerance;
//   * if the LP solution is integral (|x_j − round(x_j)| ≤ kMipIntegrality for integer j),
//     it is a new incumbent (x_j rounded exactly for integer j, re-checked for feasibility);
//   * else branch on the most fractional integer column: x_j ≤ ⌊v⌋ and x_j ≥ ⌈v⌉;
//   * node order: depth-first until the first incumbent, then best-bound;
//   * a simple rounding heuristic at every node tries to produce incumbents early.
// Status: Optimal when (incumbent − best bound) ≤ max(abs, rel·|incumbent|) with the
// tolerances in tolerances.h; Infeasible when the tree is exhausted without an incumbent;
// NodeLimit / TimeLimit → IterationLimit / TimeLimit with the incumbent (if any) reported.
// Invariant: an incumbent is only accepted after an exact feasibility re-check of the
// rounded point on the ORIGINAL model.
#pragma once

#include <cstdint>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::mip {

struct BranchAndBoundOptions {
  double time_limit = 3600.0;
  std::int64_t node_limit = 1'000'000;
  int verbosity = 0;
  // Dense oracle size guard (m·(n+2m) tableau entries); larger models are refused.
  std::int64_t max_tableau_entries = 2'000'000;
};

Solution solve_branch_and_bound(const Model& model, const BranchAndBoundOptions& options = {});

}  // namespace ps26119::mip
