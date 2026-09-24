# Decision log

Decisions taken during development that a reviewer (or the team) may want to revisit.
Newest last. Each entry: what, why, evidence, how to undo.

## 2026-09-24

1. **Objective offset semantics.** `Model` doc says "minimise sense·(cᵀx) + obj_offset"; we
   interpret the objective VALUE as `cᵀx + obj_offset` with `sense` choosing min/max (HiGHS
   and MPS practice). *Why:* the literal formula would flip the offset's sign for MAX models.
   *Needs:* confirmation with the MPS-reader teammate (contract §6).
2. **Dual sign convention = HiGHS** (`z = c − Aᵀy` in the original sense). Verified by a MAX
   model with nonzero duals (`data/hand/max_ranged.mps`) passing `tools/verify.py` with
   HiGHS's own duals.
3. **Fingerprint ignores entry order within a column** (and names, and −0.0). *Why:* two
   correct MPS readers may store a column's entries in different orders.
4. **First-order "Optimal" at 1e-8 also requires verifier-grade per-row checks.** The
   literature L2 relative KKT alone let single rows be violated by 2e-6 on afiro/adlittle,
   which `verify.py` (1e-6 per row) rejected. Iteration counts at 1e-8 are therefore
   slightly higher than a pure-KKT stop would report.
5. **Termination and restart decisions are always fp64**, even in mixed precision; fp32 is
   promoted to fp64 at relative KKT 1e-6 or after 10 checks without 10% progress.
6. **PID guard change**: cuPDLPx's residual-ratio guard (skip PID when rel_dual/rel_primal is
   outside [1e-8, 1e8]) is applied only when both residuals are positive. *Evidence:*
   recipe (all columns boxed ⇒ dual residual ≡ 0) went from 55,296 to 1,088 iterations.

## 2026-09-25

7. **HiGHS reference runs in a killable subprocess.** HiGHS ran > 25 min past its 600 s
   `time_limit` on the T = 8760 refinery LP; `bench/scale.py` now hard-kills at cap + 30 s.
8. **Benchmarks run from a frozen copy of the binary, under `caffeinate`.** A rebuild during
   a run would mix code versions under one CSV hash; a sleeping/throttled laptop produced one
   2,427 s row for a ~130 s job. Both invalid CSVs were discarded, not committed.
9. **Warm start default = (x, y) only; reusing the primal weight is opt-in** (`--warm-weight`).
   *Evidence (refinery T=365, iterations vs cold):* with ω — price 2.11×, demand 0.63×,
   crude 0.85×; without ω — 1.02×, 0.89×, 0.85×. Robustness over best case.
10. **‖A‖₂ estimate: Lanczos instead of power iteration** (bug fix). Power iteration stopped
    on stagnation under-estimated ‖A‖₂ on scrs8 (0.99550 vs 0.99877), giving η‖A‖ = 1.0013 > 1
    and divergence. Lanczos + ×(1+1e-4) safety. scrs8, bnl1, greenbeb now solve.
11. **r²HPDHG safeguards (ours, beyond cuPDLPx):** restart ratio tests only with positive
    finite residuals; PID step |Δ log ω| ≤ ln 10 per restart. Net −6% iterations on small
    Netlib; within a few % on generated models.
12. **Infeasible/Unbounded only from checked certificates.** A primal ray alone proves only
    dual infeasibility, so Unbounded additionally needs a primal-feasible iterate; otherwise
    the engine keeps iterating to its limit rather than guess.
13. **GitHub:** the repo is ready for a private GitHub remote (`scripts/setup_github.sh`),
    but it was NOT created: that needs the owner's `gh auth login`.
