// pdlp.h — restarted PDHG for LP, PDLP-style (CPU or any Backend).
//
// Citations:
//   A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex problems with
//     applications to imaging", JMIV 40 (2011) — PDHG.
//   D. Applegate, M. Díaz, O. Hinder, H. Lu, M. Lubin, B. O'Donoghue, W. Schudy,
//     "Practical large-scale linear programming using primal-dual hybrid gradient",
//     NeurIPS 2021 (PDLP) — preconditioning, adaptive restarts, primal weight.
//   D. Applegate, O. Hinder, H. Lu, M. Lubin, "Faster first-order primal-dual methods for
//     linear programming using restarts and sharpness", Math. Prog. 2023 — restart scheme.
//   H. Lu, J. Yang, "cuPDLP.jl: A GPU implementation of restarted primal-dual hybrid
//     gradient for linear programming in Julia" (arXiv 2311.12180) — KKT-error based
//     restarts (β_sufficient 0.2, β_necessary 0.8, β_artificial 0.36), constant step.
//
// ---------------------------------------------------------------------------------------
// Derivation of the steps for the row-bounds form (scaled problem, always min):
//     min cᵀx   s.t.  Ax = s,  s ∈ S = [rl, ru],  x ∈ X = [l, u].
// Dualise only Ax = s with multiplier y:
//     L(x, s; y) = cᵀx − yᵀ(Ax − s),    x ∈ X, s ∈ S.
// Minimising over s ∈ S gives  min_s yᵀs = Σ_i (y_i > 0 ? rl_i : ru_i)·y_i  =: −p(y),
// so the saddle problem is  min_{x∈X} max_y  cᵀx − yᵀAx − p(y).
// PDHG (primal first, extrapolated primal in the dual step):
//   x⁺ = argmin_{x∈X}  cᵀx − yᵀAx + ‖x − x_k‖²/(2τ)  = proj_X( x_k − τ(c − Aᵀy_k) )
//   y⁺ = argmax_y  −yᵀA x̄ − p(y) − ‖y − y_k‖²/(2σ),   x̄ = 2x⁺ − x_k.
// Write −p(y) = min_{s∈S} yᵀs and swap min/max (convex–concave): for fixed s the inner
// maximiser is y = y_k + σ(s − A x̄); substituting and minimising over s ∈ S gives
//   s* = proj_S( A x̄ − y_k/σ ),   y⁺ = y_k − σ A x̄ + σ · proj_S( A x̄ − y_k/σ ).
// Sign check: if row i should sit at its LOWER bound (y_i > 0 at the optimum, the
// convention of include/ps26119/solution.h) then A x̄ ≈ rl_i and A x̄ − y_i/σ < rl_i, the
// projection returns rl_i and y⁺_i = y_i + σ(rl_i − (A x̄)_i) ≈ y_i: y stays positive.
// If ru_i = +∞ the formula gives y⁺_i = σ·max(0, y_i/σ − (Ax̄)_i + rl_i) ≥ 0, i.e. the
// dual is automatically sign-feasible. These signs are verified against the oracle by
// tests/unit/test_pdhg.cpp.
// Step sizes: τ = η/ω, σ = ηω with η = 0.998/‖A‖₂ (power iteration), so τσ‖A‖² < 1;
// ω is the primal weight.
//
// PDLP-style loop (this file's engine):
//   * iterate PDHG, keep the uniform running average (constant step ⇒ equal weights);
//   * every K iterations evaluate the relative KKT error (termination.h) of the current
//     and the average iterate; the better one is the restart candidate;
//   * restart to the candidate if  KKT ≤ 0.2·KKT_last_restart  (sufficient), or
//     KKT ≤ 0.8·KKT_last_restart and KKT > KKT_previous_check  (necessary + no progress),
//     or inner iterations ≥ 0.36·total iterations  (artificial);
//   * at a restart update the primal weight (PDLP):
//       ω ← exp( θ·log(Δy/Δx) + (1−θ)·log ω ),  θ = 0.5,
//     Δx = ‖x_new − x_last_restart‖₂, Δy = ‖y_new − y_last_restart‖₂ (skipped if tiny).
//   * terminate when the candidate's relative KKT error ≤ tolerance.
// Invariants: x̂ ∈ X always; ŷ is sign-feasible for infinite row bounds; every
// termination/restart decision is taken on fp64 KKT statistics of the ORIGINAL problem.
// ---------------------------------------------------------------------------------------
#pragma once

#include "pdhg/engine.h"

namespace ps26119::pdhg {

Solution solve_pdlp(const Model& model, const EngineOptions& opt);

}  // namespace ps26119::pdhg
