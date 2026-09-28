// ranging.h — sensitivity ranging at an optimal vertex: for every objective coefficient, the
// interval over which the optimal basis (hence the optimal x) stays the same; for every row, the
// interval over which its binding bound can move with the same basis (the objective then changes
// linearly with the row's dual). The classical post-optimal analysis planners read next to the
// duals (e.g. "how far can the naphtha price fall before the plan changes?").
//
// Works from a Solution (x, row_activity, y, z) of the ORIGINAL model at a vertex — simplex or the
// dense oracle; an interior first-order answer is rejected with a message, never ranged. The basis
// is reconstructed from the solution and factorized with la::SparseLU; ranges are basis-dependent
// under degeneracy (as in every solver), and the basis actually used is checked for primal and
// dual feasibility first. See src/core/ranging.cpp for the method and its references.
#pragma once

#include <string>
#include <vector>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119 {

struct CostRange {
  int col = 0;
  double cost = 0.0;        // c_j as in the model
  double lower = 0.0;       // c_j may move within [lower, upper] (±inf allowed) with the same basis
  double upper = 0.0;
  bool basic = false;
  double reduced_cost = 0.0;  // of the basis used (see dual)
};

struct RhsRange {
  int row = 0;
  double activity = 0.0;    // (A x)_i
  double dual = 0.0;        // y_i of the basis used (the objective's change per unit of the bound within the
                            // range); equals the solver's y unless the optimum is dual-degenerate
  int binding = 0;          // -1 lower bound binding, +1 upper bound binding, 0 not binding
  double bound = 0.0;       // the binding bound; not binding: the finite upper bound, else the
                            // finite lower bound, else (free row) the activity
  double lower = 0.0;       // `bound` may move within [lower, upper] with the same basis (not binding:
  double upper = 0.0;       // an upper bound within [activity, +inf), a lower one within (-inf, activity])
};

struct RangingResult {
  bool ok = false;
  std::string message;      // why ranging was refused, or notes (e.g. degeneracy)
  std::vector<CostRange> cols;
  std::vector<RhsRange> rows;
  int degenerate_basics = 0;  // basic variables at a bound (ranges may be one-sided at 0)
};

struct RangingOptions {
  double tolerance = 1e-9;    // relative: "at a bound" / "zero reduced cost"
  int max_rows = 25000;       // above this, ranging (one FTRAN + one BTRAN per row) is refused
};

// sol must be Optimal for `model`. Never throws.
RangingResult compute_ranging(const Model& model, const Solution& sol, const RangingOptions& options = {});

}  // namespace ps26119
