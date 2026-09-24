// presolve.h — safe LP presolve reductions with primal AND dual postsolve.
//
// Reductions (repeated until nothing changes):
//   R1 empty row        — removed if 0 ∈ [rl, ru] (else the LP is infeasible); y_i = 0.
//   R2 fixed column     — l_j = u_j: substituted out (row bounds and objective offset shift).
//   R3 empty column     — set to its best finite bound for sense·c_j (removed only if that
//                         bound is finite, or c_j = 0); z_j = c_j.
//   R4 singleton row    — rl ≤ a·x_j ≤ ru becomes a bound on x_j; the row is removed.
// Postsolve: x of removed columns is restored; y of kept rows comes from the reduced solve;
// y of removed rows is 0 except singleton rows, which take over the column's reduced cost
// when the column sits on the bound that row created (so that z = c − Aᵀy keeps the right
// signs); finally z = c − Aᵀy is recomputed on the ORIGINAL model.
//
// Citations: E. D. Andersen, K. D. Andersen, "Presolving in linear programming", Math. Prog.
// 71 (1995); T. Achterberg et al., "Presolve reductions in MIP", INFORMS JoC 32 (2020).
// Our code. Invariants: a postsolved x satisfies the original bounds whenever the reduced x
// satisfied the reduced ones (exact arithmetic up to the substitutions' rounding); the
// reported Solution is always re-checked on the ORIGINAL model by the caller.
#pragma once

#include <string>
#include <vector>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119 {

struct PresolveResult {
  enum class Outcome { Reduced, Infeasible, Unchanged } outcome = Outcome::Unchanged;
  Model reduced;
  std::string message;  // reason for Infeasible
  int removed_rows = 0, removed_cols = 0, tightened_bounds = 0;

  // ---- postsolve data
  std::vector<int> row_map;  // reduced row → original row
  std::vector<int> col_map;  // reduced col → original col
  std::vector<double> fixed_value;  // original col → value when removed (NaN if kept)
  struct Singleton {
    int row, col;
    double a;
    bool set_lower, set_upper;  // did this row tighten the column's lower / upper bound?
  };
  std::vector<Singleton> singletons;  // in the order they were applied
};

PresolveResult presolve(const Model& model);

// Maps a solution of `r.reduced` back to the original model (x, y, z, row activity,
// objective). Status and statistics are copied; the caller re-checks optimality.
Solution postsolve(const Model& original, const PresolveResult& r, const Solution& reduced_solution);

}  // namespace ps26119
