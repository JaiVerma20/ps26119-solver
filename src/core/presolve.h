// presolve.h — safe LP presolve reductions with primal AND dual postsolve.
//
// Reductions (repeated until nothing changes):
//   R1 empty row        — removed if 0 ∈ [rl, ru] (else the LP is infeasible); y_i = 0.
//   R2 fixed column     — l_j = u_j: substituted out (row bounds and objective offset shift).
//   R3 empty column     — set to its best finite bound for sense·c_j (removed only if that
//                         bound is finite, or c_j = 0); z_j = c_j.
//   R4 singleton row    — rl ≤ a·x_j ≤ ru becomes a bound on x_j; the row is removed.
//   R5 redundant row    — the row's activity range over the model's OWN column bounds lies
//                         inside [rl, ru] even after the worst-case rounding of the range:
//                         removed, y_i = 0 (a Farkas ray gets 0 there too). Kennington osa-*:
//                         37 rows Σ a_j x_j ≥ 0 with a, x ≥ 0 carry ~39% of all nonzeros
//                         (osa-07/14: 30% fewer r²HPDHG iterations, ~2× faster). Bounds
//                         tightened by R4 are deliberately NOT used: chaining them removed rows
//                         whose loss slowed r²HPDHG badly (Netlib agg lost at 60 s, bnl2, agg3
//                         ×58); with the model's own bounds Netlib stays 85/85 (geomean +0.6%).
// After postsolve the caller (solve()) re-checks optimality on the ORIGINAL model; a
// first-order answer that narrowly misses is re-solved tighter in the reduced space, and
// failing that the original is solved without presolve.
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
  int redundant_rows = 0;  // of removed_rows: by R5

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

// Maps a Farkas vector of `r.reduced` (row multipliers proving it infeasible) to the original
// rows. Kept rows keep their multiplier; fixed and empty columns need nothing (their terms are
// identical in both models). A column bound created by a removed singleton row i (a·x_j) is
// moved onto that row, newest singleton first: if λ_j = −(Aᵀr)_j points at that bound,
// r_i = λ_j / a, which zeroes the column's term and contributes λ_j·(bound) through the row —
// the same value as in the reduced L₀. The caller checks the result on the original model.
std::vector<double> postsolve_farkas(const Model& original, const PresolveResult& r,
                                     const std::vector<double>& reduced_ray);

}  // namespace ps26119
