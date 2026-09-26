// tolerances.h — THE single place tolerances live (CLAUDE.md §5.3, §8).
// tools/verify.py parses this file with a regex, so keep the `kName = value;` form.
#pragma once

namespace ps26119::tol {

// First-order engines (PDHG family): relative KKT error, see src/pdhg/termination.h.
inline constexpr double kFirstOrderFast = 1e-4;
inline constexpr double kFirstOrderHigh = 1e-8;
// First-order infeasibility / unboundedness certificates (relative ray violation, see
// src/pdhg/engine.h). A status Infeasible/Unbounded from these engines needs a ray whose
// violation divided by its objective is below this.
inline constexpr double kFirstOrderInfeasible = 1e-8;
// Prototype branch-and-bound (src/mip): integrality of an integer column, feasibility of an
// incumbent (relative, after rounding its integer columns) and the gap needed to claim
// Optimal (absolute or relative to |incumbent|).
inline constexpr double kMipIntegrality = 1e-6;
inline constexpr double kMipFeasibility = 1e-9;
inline constexpr double kMipGapAbs = 1e-9;
inline constexpr double kMipGapRel = 1e-9;
// Mixed precision: fp32 iterates are promoted to fp64 once the relative KKT error reaches
// this level (when the target is tighter), or earlier if fp32 progress stalls.
inline constexpr double kMixedPrecisionSwitch = 1e-6;

// Dense double-double oracle.
inline constexpr double kOracleFeasibility = 1e-9;  // primal/dual feasibility in the simplex
inline constexpr double kOraclePivot = 1e-11;       // smallest acceptable |pivot|

// Independent verifier (tools/verify.py). All three are RELATIVE:
//   primal: max_i viol_i / (1 + |violated bound_i|)
//   dual:   max sign violation of y and z = c − Aᵀy, divided by (1 + ‖c‖∞)
//   gap:    |p − d| / (1 + |p| + |d|)
// Absolute values are reported alongside, but PASS/FAIL uses the relative ones.
inline constexpr double kVerifyPrimal = 1e-6;
inline constexpr double kVerifyDual = 1e-6;
inline constexpr double kVerifyGap = 1e-6;

// Unboundedness certificates (core/certificates.h, tools/verify.py): after scaling the ray d
// to ‖d‖∞ = 1, the objective must decrease, −(sense·c)ᵀd ≥ kVerifyRay·max(1, ‖c‖∞), and every
// recession-cone violation (rows: A d against the finite row bounds; columns: d against the
// finite bounds) must be ≤ kVerifyRay·(−(sense·c)ᵀd). Infeasibility certificates need no
// tolerance: they are checked with the rounding-proof bound of core/safe_bound.h.
inline constexpr double kVerifyRay = 1e-8;

// Agreement with a published optimum: |ours − ref| / (1 + |ref|).
inline constexpr double kVerifyReference = 1e-6;

}  // namespace ps26119::tol
