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

#include "core/certificates.h"
#include "core/safe_bound.h"
#include "oracle/dense_simplex.h"
#include "ps26119/tolerances.h"
#include "simplex/primal_simplex.h"

namespace ps26119::mip {
namespace {

struct Node {
  std::vector<double> lo, up;  // bounds of the integer columns (indexed like `ints`)
  double bound;                // parent's LP bound (min form); refined when solved
  int depth;
  int branched = -1;           // index into `ints` of the branching that created this node
  int dir = 0;                 // -1 down, +1 up
  double dist = 0.0;           // distance the branching moved the parent's LP value
};

// Pseudocosts (Bénichou et al. 1971; Achterberg, Koch, Martin, "Branching rules revisited",
// Oper. Res. Lett. 33, 2005): average LP bound gain per unit of change, per column and
// direction, learned from the children actually solved. A column without history uses the
// average over all columns (1 when there is none yet), so the rule starts as most-fractional.
struct Pseudocosts {
  std::vector<double> sum[2];
  std::vector<int> cnt[2];
  double all_sum[2] = {0, 0};
  int all_cnt[2] = {0, 0};
  explicit Pseudocosts(std::size_t n) {
    for (int d = 0; d < 2; ++d) sum[d].assign(n, 0.0), cnt[d].assign(n, 0);
  }
  void record(int t, int dir, double gain_per_unit) {
    const int d = dir > 0;
    sum[d][t] += gain_per_unit, ++cnt[d][t];
    all_sum[d] += gain_per_unit, ++all_cnt[d];
  }
  double get(int t, int dir) const {
    const int d = dir > 0;
    if (cnt[d][t] > 0) return sum[d][t] / cnt[d][t];
    return all_cnt[d] > 0 ? all_sum[d] / all_cnt[d] : 1.0;
  }
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
  const bool use_oracle = opt.node_solver == NodeSolver::Oracle;
  best.engine = use_oracle ? "branch-and-bound(oracle)" : "branch-and-bound(simplex)";
  best.precision = use_oracle ? "dd" : "fp64";
  const double sense = M.sense;
  std::vector<int> ints;
  for (int j = 0; j < M.num_cols; ++j)
    if (j < static_cast<int>(M.is_integer.size()) && M.is_integer[j]) ints.push_back(j);

  const std::int64_t tableau = static_cast<std::int64_t>(M.num_rows) * (M.num_cols + 2 * M.num_rows);
  const bool oracle_fits = tableau <= opt.max_tableau_entries;
  if (use_oracle && !oracle_fits) {
    best.status = Status::NotSolved;
    best.message = "MILP too large for the prototype branch-and-bound (dense oracle node solver; " +
                   std::to_string(M.num_rows) + " rows)";
    return best;
  }

  Pseudocosts pc(ints.size());
  double incumbent = std::numeric_limits<double>::infinity();  // min form, without offset
  std::vector<double> inc_x;
  std::int64_t nodes = 0, lp_iterations = 0, uncertified_prunes = 0, oracle_fallbacks = 0;
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

  // Fractional diving (primal heuristic; e.g. Berthold, "Primal heuristics for MIP", 2006):
  // from a node's LP point, fix every integral integer column and the least fractional one to
  // their rounded values, re-solve, repeat. An infeasible LP flips the last rounding once, then
  // the dive stops; it also stops when the LP bound can no longer beat the incumbent. Only a
  // point that passes try_incumbent's check on the ORIGINAL model is ever kept.
  std::int64_t dive_lps = 0, dives = 0, dive_incumbents = 0;
  auto dive = [&](const Node& start, std::vector<double> x) {
    ++dives;
    Node d = start;
    for (int step = 0; step < static_cast<int>(ints.size()) + 1; ++step) {
      int pick = -1;
      double best = 2.0;
      for (std::size_t t = 0; t < ints.size(); ++t) {
        const double v = x[ints[t]], r = std::round(v), f = std::fabs(v - r);
        if (f <= tol::kMipIntegrality) {
          d.lo[t] = d.up[t] = std::clamp(r, d.lo[t], d.up[t]);
        } else if (f < best) {
          best = f, pick = static_cast<int>(t);
        }
      }
      if (pick < 0) {  // integral
        const double before = incumbent;
        try_incumbent(x);
        if (incumbent < before) ++dive_incumbents;
        return;
      }
      const double v = x[ints[pick]];
      bool solved = false;
      for (int attempt = 0; attempt < 2 && !solved; ++attempt) {
        const double r = attempt == 0 ? std::round(v) : (std::round(v) > v ? std::floor(v) : std::ceil(v));
        if (r < start.lo[pick] || r > start.up[pick]) continue;
        Node trial = d;
        trial.lo[pick] = trial.up[pick] = r;
        simplex::SimplexOptions so;
        so.time_limit_seconds = std::max(0.0, opt.time_limit - elapsed());
        const Solution lp = simplex::solve_primal_simplex(node_model(trial), so);
        ++dive_lps;
        lp_iterations += lp.iterations;
        if (lp.status != Status::Optimal) continue;
        if (std::isfinite(incumbent) && gap_closed(sense * (lp.objective - M.obj_offset))) return;
        d = std::move(trial);
        x = lp.x;
        solved = true;
      }
      if (!solved || elapsed() > opt.time_limit) return;
    }
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
    const Model nm = node_model(nd);
    Solution lp;
    if (!use_oracle) {
      simplex::SimplexOptions so;
      so.time_limit_seconds = std::max(1.0, opt.time_limit - elapsed());
      lp = simplex::solve_primal_simplex(nm, so);
      if (lp.status == Status::NumericalError && oracle_fits) {  // rare: retry exactly
        ++oracle_fallbacks;
        lp_iterations += lp.iterations;
        lpo.time_limit = std::max(1.0, opt.time_limit - elapsed());
        lp = oracle::solve_dense_simplex(nm, lpo);
      }
    } else {
      lpo.time_limit = std::max(1.0, opt.time_limit - elapsed());
      lp = oracle::solve_dense_simplex(nm, lpo);
    }
    lp_iterations += lp.iterations;
    if (lp.status == Status::TimeLimit) {
      limit = Status::TimeLimit;
      break;
    }
    if (lp.status == Status::Infeasible) continue;
    if (lp.status != Status::Optimal) {  // unbounded relaxation or numerical trouble
      if (lp.status == Status::Unbounded && !std::isfinite(incumbent)) {
        // An unbounded relaxation means "unbounded OR infeasible" — reporting Unbounded here
        // was an unproven claim. For rational data, if any integer-feasible point exists, the
        // integer hull has the same recession cone as the relaxation (Meyer 1974), so an
        // integer point + this relaxation's ray proves unboundedness. Find a point by solving
        // the zero-objective MILP (its relaxations are never unbounded) with what is left.
        best.iterations = lp_iterations;
        if (lp.primal_ray.size() != static_cast<std::size_t>(M.num_cols)) {
          best.status = Status::NotSolved;
          best.message = "LP relaxation unbounded (MILP unbounded or infeasible); the node solver returned no ray";
          best.seconds = elapsed();
          return best;
        }
        Model feas = M;
        std::fill(feas.obj.begin(), feas.obj.end(), 0.0);
        BranchAndBoundOptions fo = opt;
        fo.time_limit = std::max(0.0, opt.time_limit - elapsed());
        fo.node_limit = std::max<std::int64_t>(0, opt.node_limit - nodes);
        const Solution f = solve_branch_and_bound(feas, fo);
        best.iterations += f.iterations;
        if (f.status == Status::Optimal && f.x.size() == static_cast<std::size_t>(M.num_cols)) {
          const CertificateCheck c = check_unboundedness_certificate(M, f.x, lp.primal_ray);
          if (c.passed) {
            best.status = Status::Unbounded;
            best.x = f.x;
            best.row_activity = M.row_activity(f.x);
            best.primal_ray = lp.primal_ray;
            best.check = "PASS";
            best.message = "LP relaxation unbounded and an integer-feasible point exists (integer point + ray); " + c.detail;
          } else {
            best.status = Status::NumericalError;
            best.message = "LP relaxation unbounded, but the ray fails at the integer point: " + c.detail;
          }
        } else if (f.status == Status::Infeasible) {
          best.status = Status::Infeasible;
          best.message = "no integer-feasible point (the LP relaxation is unbounded, the tree was exhausted)";
        } else {
          best.status = f.status == Status::Optimal ? Status::NumericalError : f.status;
          best.message = std::string("LP relaxation unbounded; integer feasibility not decided (") + to_string(f.status) + ")";
        }
        best.objective = std::numeric_limits<double>::quiet_NaN();
        best.seconds = elapsed();
        return best;
      }
      best.status = Status::NumericalError;
      best.message = "node LP failed: " + lp.message;
      return best;
    }
    // Node bound for pruning (min form, no offset). The LP objective of an fp64 engine can
    // exceed the true node optimum by rounding; the certified bound from its duals cannot.
    double bound = sense * (lp.objective - M.obj_offset);
    bool certified = use_oracle;  // the dd oracle's objective is exact to far below the gap tolerance
    if (!use_oracle) {
      const SafeBound sb = certified_dual_bound(nm, lp.y);
      if (sb.finite) {
        bound = std::min(bound, sense * (sb.bound - M.obj_offset));
        certified = true;
      }
    }
    // learn from this node: the bound gain its branching caused, per unit of change
    if (nd.branched >= 0 && std::isfinite(nd.bound) && std::isfinite(bound) && nd.dist > 0)
      pc.record(nd.branched, nd.dir, std::max(0.0, bound - nd.bound) / nd.dist);
    if (std::isfinite(incumbent) && gap_closed(bound)) {
      if (!certified) ++uncertified_prunes;
      continue;
    }
    // pseudocost branching, product score
    int branch = -1;
    double best_score = -1.0;
    for (std::size_t t = 0; t < ints.size(); ++t) {
      const double v = lp.x[ints[t]];
      if (std::fabs(v - std::round(v)) <= tol::kMipIntegrality) continue;
      const double f = v - std::floor(v);
      const int ti = static_cast<int>(t);
      // ε on the per-unit gain (not on the product), so that with zero gains everywhere (pure
      // feasibility problems such as enigma) the score is ε²·f(1−f): most fractional again.
      const double score = std::max(pc.get(ti, -1), 1e-6) * f * std::max(pc.get(ti, +1), 1e-6) * (1.0 - f);
      if (score > best_score) best_score = score, branch = ti;
    }
    try_incumbent(lp.x);  // integral LP point, or a rounding of it
    if (branch < 0) continue;  // integral: incumbent updated (if feasible after rounding)
    // dive at the root, then every 100 nodes without an incumbent / 1000 with one, while the
    // dives' LPs stay below a fifth of all node LPs
    if (!use_oracle && (nodes == 1 || nodes % (std::isfinite(incumbent) ? 1000 : 100) == 0) &&
        dive_lps <= nodes / 5 + static_cast<std::int64_t>(ints.size()))
      dive(nd, lp.x);
    const double v = lp.x[ints[branch]];
    Node down = nd, upn = nd;
    down.up[branch] = std::floor(v);
    upn.lo[branch] = std::ceil(v);
    down.bound = upn.bound = bound;
    down.depth = upn.depth = nd.depth + 1;
    down.branched = upn.branched = branch;
    down.dir = -1, upn.dir = +1;
    down.dist = v - std::floor(v), upn.dist = std::ceil(v) - v;
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
  if (uncertified_prunes > 0)
    best.message += ", " + std::to_string(uncertified_prunes) + " prunes by an uncertified LP bound";
  if (oracle_fallbacks > 0) best.message += ", " + std::to_string(oracle_fallbacks) + " node LPs re-solved by the oracle";
  if (dives > 0)
    best.message += ", " + std::to_string(dives) + " dives (" + std::to_string(dive_lps) + " LPs, " +
                    std::to_string(dive_incumbents) + " incumbents)";
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
