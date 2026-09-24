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
    Netlib; within a few % on generated models. Isolated with `--set pid_max_log_step=1e9`:
    neutral on most models, bnl1 6.45M vs 9.69M iterations with the clamp (kept).
12. **Infeasible/Unbounded only from checked certificates.** A primal ray alone proves only
    dual infeasibility, so Unbounded additionally needs a primal-feasible iterate; otherwise
    the engine keeps iterating to its limit rather than guess.
13. **GitHub:** the repo is ready for a private GitHub remote (`scripts/setup_github.sh`),
    but it was NOT created: that needs the owner's `gh auth login`.

## 2026-09-25 (overnight)

14. **Certified bound (Neumaier–Shcherbina) with two rigorous repairs**: multipliers that
    point at an infinite row bound are zeroed first (the bound is valid for ANY y), and
    columns bounded on one side get rigorous implied bounds by bound propagation
    (`src/core/implied_bounds.*`, outward-rounded). On the small Netlib set with optimal duals
    the bound is finite and tight (≤ 1e-9) for 7/10; the other 3 have a basic column with no
    finite implied bound, where no rounding-proof certificate exists — reported as −∞.
15. **Batched scenarios** use the same r²HPDHG per scenario, but the epoch reference residual
    r⁰ is taken at the first check after a restart (so all scenarios share one extra SpMM).
16. **Threads**: CPU kernels run on a small deterministic pool (no OpenMP; Apple clang lacks
    libomp). Results are bit-identical for any thread count (fixed chunking, fixed-order
    reductions); default is 1 thread. Reductions changed from one running sum to 64 fixed
    chunks, which changes rounding slightly vs. earlier CSVs.
17. **Adaptive geometric-mean scaling** (12 sweeps when max|a|/min|a| ≥ 10^4.5, else off).
    Full-Netlib A/B at 4f0db8c: off 81/93, always 85/93 (geomean −24% iterations), adaptive
    85/93 with the same geomean — and no 20–25% slowdown on the refinery/random models, which
    are well scaled (10^2.5 / 10^1.8). Tuned on Netlib: a simple, explainable rule, but a
    threshold is still a tuned constant (knob `geometric_mean_min_log10_range`).
18. **Thread-pool race (found by the benchmark, fixed).** With 10 threads, 1 run in ~6 crashed
    (exit 139): a late worker could join a job while the dispatcher rewrote it. Publishing,
    joining and retiring a job now all happen under one mutex; 42 stress runs + 20 repeats
    of the determinism test pass. Scaling runs made with the buggy binary were discarded.
19. **Benchmark provenance from the binary.** `ps26119 --version` prints the commit it was
    built from (`-dirty` if the tree had uncommitted changes); every bench labels its CSV
    with that, not with the working tree at run time.
20. **Batch = identical algorithm per scenario.** r⁰ is measured right after the epoch's first
    step (one shared SpMM when any scenario restarts); finished scenarios are compacted out.
    T=365 price scenarios: identical iterations to single solves; K=4 1.12× faster, K=8 0.83×
    — the T=365 matrix is cache-resident, so SpMM has little to save on CPU.
21. **Hard Netlib models are chaotic.** perold/nesm solve in ~0.45M iterations with one
    rounding pattern of the reductions and need > 8M with another (fixed 64-chunk sums vs one
    running sum). Counts of solved hard models within a time limit carry ± a few models of
    this kind of noise.
22. **Presolve judged by the original model's yardstick.** Substituting fixed columns changes
    the right-hand side, so the reduced problem's relative KKT (divided by 1+‖b‖) is not the
    original's (tuff: ‖b‖ = 0 originally, 1100 after presolve — a 1000× difference for the same
    absolute residual). The reduced solve therefore uses the ORIGINAL ‖b‖, ‖c‖ and objective
    magnitude for termination (Options::kkt_b_norm/kkt_c_norm/kkt_obj_shift). Safety net if a
    postsolved point still misses on the original: re-solve the reduced model 100× tighter,
    then fall back to solving the original without presolve. A warm-started polish of the
    original was tried first and was fragile (stocfor2 4.8M vs 39k iterations) — removed.
    Result at 01f2eec (full Netlib, 60 s): presolve 85 vs 83 solved, no losses, geomean
    iterations −6%, total time −5%; safety net used on 4 models. **Presolve is ON by default**
    (`--no-presolve` to disable).
