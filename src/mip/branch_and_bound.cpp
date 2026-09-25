// branch_and_bound.cpp — see branch_and_bound.h.
#include "mip/branch_and_bound.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "oracle/dense_simplex.h"
#include "ps26119/tolerances.h"

namespace ps26119::mip {
namespace {

struct Node {
  std::vector<double> lo, up;  // bounds of the integer columns (indexed like `ints`)
  double bound;                // parent's LP bound (min form); refined when solved
  int depth;
};

// Feasibility of x on the original model (bounds, rows) with the verifier's relative test.
bool feasible(const Model& M, const std::vector<double>& x) {
  auto ok = [](double v, double lo, double up) {
    return v >= lo - tol::kMipFeasibility * (1 + std::fabs(lo)) && v <= up + tol::kMipFeasibility * (1 + std::fabs(up));
  };
  for (int j = 0; j < M.num_cols; ++j)
    if (!ok(x[j], M.col_lower[j], M.col_upper[j])) return false;
  const auto ax = M.row_activity(x);
  for (int i = 0; i < M.num_rows; ++i)
    if (!ok(ax[i], M.row_lower[i], M.row_upper[i])) return false;
  return true;
}

}  // namespace

Solution solve_branch_and_bound(const Model& M, const BranchAndBoundOptions& opt) {
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  Solution best;
  best.engine = "branch-and-bound(oracle)";
  best.precision = "dd";
  const double sense = M.sense;
  std::vector<int> ints;
  for (int j = 0; j < M.num_cols; ++j)
    if (j < static_cast<int>(M.is_integer.size()) && M.is_integer[j]) ints.push_back(j);

  const std::int64_t tableau = static_cast<std::int64_t>(M.num_rows) * (M.num_cols + 2 * M.num_rows);
  if (tableau > opt.max_tableau_entries) {
    best.status = Status::NotSolved;
    best.message = "MILP too large for the prototype branch-and-bound (dense oracle node solver; " +
                   std::to_string(M.num_rows) + " rows)";
    return best;
  }

  double incumbent = std::numeric_limits<double>::infinity();  // min form, without offset
  std::vector<double> inc_x;
  std::int64_t nodes = 0, lp_iterations = 0;
  double best_bound = -std::numeric_limits<double>::infinity();

  auto node_model = [&](const Node& nd) {
    Model m = M;
    for (std::size_t t = 0; t < ints.size(); ++t) {
      m.col_lower[ints[t]] = nd.lo[t];
      m.col_upper[ints[t]] = nd.up[t];
    }
    m.is_integer.clear();
    return m;
  };
  auto try_incumbent = [&](std::vector<double> x) {
    for (int j : ints) x[j] = std::round(x[j]);
    if (!feasible(M, x)) return;
    double obj = 0;
    for (int j = 0; j < M.num_cols; ++j) obj += sense * M.obj[j] * x[j];
    if (obj < incumbent) {
      incumbent = obj;
      inc_x = std::move(x);
      if (opt.verbosity >= 2) std::fprintf(stderr, "bb node %lld incumbent %.12g\n", static_cast<long long>(nodes), sense * incumbent + M.obj_offset);
    }
  };
  auto gap_closed = [&](double bound) {
    const double g = incumbent - bound;
    return g <= std::max(tol::kMipGapAbs, tol::kMipGapRel * std::fabs(incumbent));
  };

  // open nodes: DFS stack until an incumbent exists, then best-bound heap
  auto worse = [](const Node& a, const Node& b) { return a.bound > b.bound; };
  std::priority_queue<Node, std::vector<Node>, decltype(worse)> heap(worse);
  std::vector<Node> stack;
  Node root{{}, {}, -std::numeric_limits<double>::infinity(), 0};
  for (int j : ints) {
    root.lo.push_back(std::ceil(M.col_lower[j] - tol::kMipIntegrality));
    root.up.push_back(std::floor(M.col_upper[j] + tol::kMipIntegrality));
  }
  stack.push_back(root);
  Status limit = Status::Optimal;
  oracle::DenseSimplexOptions lpo;
  lpo.max_tableau_entries = opt.max_tableau_entries;

  while (!stack.empty() || !heap.empty()) {
    if (elapsed() > opt.time_limit) {
      limit = Status::TimeLimit;
      break;
    }
    if (nodes >= opt.node_limit) {
      limit = Status::IterationLimit;
      break;
    }
    Node nd;
    if (std::isfinite(incumbent) && !stack.empty()) {  // switch to best-bound once we have one
      for (auto& s : stack) heap.push(std::move(s));
      stack.clear();
    }
    if (!stack.empty()) {
      nd = std::move(stack.back());
      stack.pop_back();
    } else {
      nd = heap.top();
      heap.pop();
    }
    if (std::isfinite(incumbent) && gap_closed(nd.bound)) continue;  // pruned by the parent's bound
    bool empty_box = false;
    for (std::size_t t = 0; t < ints.size(); ++t) empty_box = empty_box || nd.lo[t] > nd.up[t];
    if (empty_box) continue;
    ++nodes;
    lpo.time_limit = std::max(1.0, opt.time_limit - elapsed());
    const Solution lp = oracle::solve_dense_simplex(node_model(nd), lpo);
    lp_iterations += lp.iterations;
    if (lp.status == Status::Infeasible) continue;
    if (lp.status != Status::Optimal) {  // unbounded relaxation or numerical trouble
      if (lp.status == Status::Unbounded && !std::isfinite(incumbent)) {
        best.status = Status::Unbounded;
        best.message = "LP relaxation unbounded (MILP unbounded or infeasible)";
        best.iterations = lp_iterations;
        best.seconds = elapsed();
        return best;
      }
      best.status = Status::NumericalError;
      best.message = "node LP failed: " + lp.message;
      return best;
    }
    const double bound = sense * (lp.objective - M.obj_offset);
    if (std::isfinite(incumbent) && gap_closed(bound)) continue;
    // most fractional integer column
    int branch = -1;
    double best_frac = tol::kMipIntegrality;
    for (std::size_t t = 0; t < ints.size(); ++t) {
      const double v = lp.x[ints[t]];
      const double f = std::fabs(v - std::round(v));
      if (f > best_frac) best_frac = f, branch = static_cast<int>(t);
    }
    try_incumbent(lp.x);  // integral LP point, or a rounding of it
    if (branch < 0) continue;  // integral: incumbent updated (if feasible after rounding)
    const double v = lp.x[ints[branch]];
    Node down = nd, upn = nd;
    down.up[branch] = std::floor(v);
    upn.lo[branch] = std::ceil(v);
    down.bound = upn.bound = bound;
    down.depth = upn.depth = nd.depth + 1;
    if (std::isfinite(incumbent)) {
      heap.push(std::move(down));
      heap.push(std::move(upn));
    } else {  // DFS: explore the side the LP leans to first
      if (v - std::floor(v) >= 0.5) stack.push_back(std::move(down)), stack.push_back(std::move(upn));
      else stack.push_back(std::move(upn)), stack.push_back(std::move(down));
    }
  }

  // best bound over the remaining open nodes (for the reported gap)
  best_bound = incumbent;
  for (const auto& s : stack) best_bound = std::min(best_bound, s.bound);
  while (!heap.empty()) best_bound = std::min(best_bound, heap.top().bound), heap.pop();
  best.iterations = nodes;
  best.seconds = elapsed();
  best.message = std::to_string(nodes) + " nodes, " + std::to_string(lp_iterations) + " simplex iterations";
  if (!std::isfinite(incumbent)) {
    best.status = limit == Status::Optimal ? Status::Infeasible : limit;
    return best;
  }
  best.status = limit;
  best.x = inc_x;
  best.row_activity = M.row_activity(inc_x);
  best.objective = M.objective_value(inc_x);
  best.dual_objective = sense * best_bound + M.obj_offset;  // best bound on the MILP optimum
  best.gap = std::fabs(incumbent - best_bound) / (1.0 + std::fabs(incumbent));
  best.primal_residual = 0;
  return best;
}

}  // namespace ps26119::mip
