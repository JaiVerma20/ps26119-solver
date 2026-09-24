// tolerances.h — THE single place tolerances live (CLAUDE.md §5.3, §8).
// tools/verify.py parses this file with a regex, so keep the `kName = value;` form.
#pragma once

namespace ps26119::tol {

// First-order engines (PDHG family): relative KKT error, see src/pdhg/termination.h.
inline constexpr double kFirstOrderFast = 1e-4;
inline constexpr double kFirstOrderHigh = 1e-8;

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

// Agreement with a published optimum: |ours − ref| / (1 + |ref|).
inline constexpr double kVerifyReference = 1e-6;

}  // namespace ps26119::tol
