// termination.h — relative KKT error for the first-order engines, evaluated on the
// ORIGINAL (unscaled) problem in fp64, in min form (objective sense·cᵀx, no offset).
//
// Following PDLP (Applegate et al. 2021) and cuPDLPx (Lu, Peng, Yang 2025, §termination):
//   primal residual   r_p = ‖A x − proj_[rl,ru](A x)‖₂           (x is kept inside [l,u])
//   reduced costs     λ   = sense·c − Aᵀ y
//   bound duals       z_j = λ_j projected on the cone allowed by the bounds of x_j:
//                           both finite → λ_j;  only l finite → max(λ_j,0);
//                           only u finite → min(λ_j,0);  free → 0
//   dual residual     r_d = ‖λ − z‖₂
//   primal objective  p   = sense·cᵀx
//   dual objective    d   = Σ_i (y_i>0 ? rl_i : ru_i) y_i + Σ_j (z_j>0 ? l_j : u_j) z_j
//                           (y is kept dual-feasible by the projection, so infinite bounds
//                            never appear with a nonzero multiplier)
//   relative errors   r_p/(1+‖b‖₂),  r_d/(1+‖c‖₂),  |p−d|/(1+|p|+|d|)
//   ‖b‖₂ = 2-norm of the vector of max(|finite rl_i|, |finite ru_i|).
// The three L2 errors above are the literature definition (used for the "time to 1e-4 /
// 1e-8" comparisons). They are global norms, so a single row can still be violated by
// more than the verifier allows (seen on afiro/adlittle: L2-KKT 1e-8, one row 2e-6).
// Therefore we also track the verifier's per-element measures (tools/verify.py):
//   primal_max_rel = max_i viol_i / (1 + |violated bound_i|)   (rows and columns)
//   dual_max_rel   = max sign violation of y and of c − Aᵀy / (1 + ‖c‖∞)
// converged(ε): L2 relative KKT ≤ ε, and — whenever ε is tighter than the verifier
// tolerances — the verifier-grade conditions hold too. So an Optimal status from a
// first-order engine at 1e-8 is always one tools/verify.py accepts.
#pragma once

#include <algorithm>
#include <cmath>

#include "ps26119/tolerances.h"

namespace ps26119::pdhg {

struct KktStats {
  double primal_residual = 0, dual_residual = 0;
  double primal_obj = 0, dual_obj = 0;
  double b_norm = 0, c_norm = 0;
  double primal_max_rel = 0, dual_max_rel = 0;  // verifier-grade, per element
  // Added to both objectives in the gap's DENOMINATOR only: lets a presolved (reduced)
  // problem be judged with the original objective's magnitude (see Options::kkt_obj_shift).
  double obj_shift = 0;

  double rel_primal() const { return primal_residual / (1.0 + b_norm); }
  double rel_dual() const { return dual_residual / (1.0 + c_norm); }
  double rel_gap() const {
    return std::fabs(primal_obj - dual_obj) /
           (1.0 + std::fabs(primal_obj + obj_shift) + std::fabs(dual_obj + obj_shift));
  }
  double rel_kkt() const { return std::max({rel_primal(), rel_dual(), rel_gap()}); }
  bool converged(double eps) const {
    if (rel_kkt() > eps) return false;
    if (eps >= tol::kVerifyPrimal) return true;
    return primal_max_rel <= tol::kVerifyPrimal && dual_max_rel <= tol::kVerifyDual && rel_gap() <= tol::kVerifyGap;
  }
  bool finite() const {
    return std::isfinite(primal_residual) && std::isfinite(dual_residual) && std::isfinite(primal_obj) &&
           std::isfinite(dual_obj);
  }
};

}  // namespace ps26119::pdhg
