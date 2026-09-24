// r2hpdhg.h — restarted Halpern PDHG with reflection (r²HPDHG) for LP.
//
// Citations:
//   H. Lu, J. Yang, "Restarted Halpern PDHG for linear programming" (arXiv 2407.16144).
//   H. Lu, Z. Peng, J. Yang, "cuPDLPx: A further enhanced GPU-based first-order solver for
//     linear programming" (arXiv 2507.14051) — reflected Halpern update, constant step,
//     fixed-point-residual restarts, PID-controlled primal weight.
//   B. Halpern, "Fixed points of nonexpanding maps", Bull. AMS 73 (1967).
//   Default constants follow the public reference implementation (approach informed by
//   MIT-Lu-Lab/cuPDLPx src/solver.cu and src/utils.cu, Apache-2.0; our own code).
//
// ---------------------------------------------------------------------------------------
// Equations (scaled problem, z = (x, y), T = one PDHG step as derived in pdlp.h):
//   T(z) = (x̂, ŷ):   x̂ = proj_X( x − τ(c − Aᵀy) )
//                     ŷ = y − σA(2x̂ − x) + σ·proj_S( A(2x̂ − x) − y/σ )
//   Reflected Halpern iteration within one restart epoch (k = 0, 1, …, anchor z⁰):
//       z^{k+1} = (k+1)/(k+2) · [ (1+γ)·T(z^k) − γ·z^k ] + 1/(k+2) · z⁰
//   written with the reflection coefficient ρ ∈ [½, 1] (γ = 2ρ − 1):
//       (1+γ)T − γI = 2ρ·T + (1 − 2ρ)·I ;  ρ = 1 ⇒ full reflection 2T − I (cuPDLPx
//       default), ρ = ½ ⇒ plain Halpern.
//   Step sizes are constant: η = 0.998/‖A‖₂, τ = η/ω, σ = ηω.
//
//   Fixed-point residual in the PDHG norm (M = [[I/τ, Aᵀ],[A, I/σ]], see pdlp.h for the
//   sign of the coupling block), scaled by η:
//       r(z) = sqrt( ω‖Δx‖² + ‖Δy‖²/ω + 2η·Δyᵀ A Δx ),   Δ = T(z) − z.
//   Computed only every K iterations from x̄ = 2x̂ − x and ȳ = 2ŷ − y (Δ = x̄ − x̂ etc.),
//   so the fused kernels never need a copy of the previous iterate.
//
//   Restart (checked every K iterations; r⁰ = residual at the start of the epoch):
//       sufficient:  r ≤ 0.2·r⁰
//       necessary:   r ≤ 0.5·r⁰  and  r > r_previous_check
//       artificial:  k ≥ 0.36·(total iterations)
//   On restart the anchor AND the iterate become T(z) = (x̂, ŷ), k ← 0.
//
//   PID primal weight at each restart (Δx = ‖x̂ − x⁰‖, Δy = ‖ŷ − y⁰‖):
//       e     = log Δy − log Δx − log ω
//       I     ← 0.3·I + e
//       ω     ← ω · exp( K_P·e + K_I·I + K_D·(e − e_prev) ),  K_P 0.99, K_I 0.01, K_D 0
//   If a distance is degenerate (≤1e-16 or ≥1e12) or the primal/dual residual ratio is
//   extreme (outside [1e-8, 1e8]; checked only when both residuals are > 0) we fall
//   back to the best ω seen so far (the one with the most balanced relative primal and
//   dual residuals) and reset the controller.
//
//   Termination: relative KKT (termination.h) of T(z) = (x̂, ŷ) ≤ ε, evaluated every K
//   iterations in fp64 on the original problem.
// Invariants: x̂ ∈ X always; ŷ sign-feasible; x, y (Halpern iterates) may leave X, which
// is fine — only (x̂, ŷ) is ever reported.
// ---------------------------------------------------------------------------------------
#pragma once

#include "pdhg/engine.h"

namespace ps26119::pdhg {

Solution solve_r2hpdhg(const Model& model, const EngineOptions& opt);

}  // namespace ps26119::pdhg
