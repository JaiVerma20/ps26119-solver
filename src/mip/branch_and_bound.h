// branch_and_bound.h — prototype branch-and-bound for small and medium mixed-integer LPs.
//
// Node relaxations: the sparse revised primal simplex (src/simplex, integrated from gpuopt;
// default) or the dense double-double oracle (NodeSolver::Oracle — exact, dense, small models
// only; also the fallback when a simplex node LP fails). Nodes are cold-started from the
// slack basis (no basis warm start yet: that needs basis I/O / a dual simplex).
// Pruning is rigorous: a node is pruned by the Neumaier–Shcherbina certified bound computed
// from its LP duals (core/safe_bound.h), which rounding cannot push above the true node
// optimum. Only when that bound is infinite is the LP objective itself used; such prunes are
// counted and reported in the message.
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
// Invariant: an incumbent is only accepted after a feasibility re-check of the rounded point
// on the ORIGINAL model (relative tolerance kMipFeasibility).
#pragma once

#include <cstdint>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::mip {

enum class NodeSolver { Simplex, Oracle };

struct BranchAndBoundOptions {
  double time_limit = 3600.0;
  std::int64_t node_limit = 1'000'000;
  int verbosity = 0;
  NodeSolver node_solver = NodeSolver::Simplex;
  // Dense oracle size guard (m·(n+2m) tableau entries): with NodeSolver::Oracle larger models
  // are refused; with Simplex the oracle is used as a fallback only below this size.
  std::int64_t max_tableau_entries = 2'000'000;
};

Solution solve_branch_and_bound(const Model& model, const BranchAndBoundOptions& options = {});

}  // namespace ps26119::mip
