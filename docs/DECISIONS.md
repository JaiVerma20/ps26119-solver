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
23. **Presolve is not free everywhere.** On the hard refinery price scenario (T=8760) the cold
    solve needed > 224k iterations with presolve (TimeLimit at 600 s) vs 94k without; warm
    starts are unaffected (47k). Netlib overall still favours presolve (85 vs 83, no losses),
    so it stays on; `--no-presolve` exists for such models. Recorded in
    bench/results/warm-start-macbook-air-m4-8fd5170.csv.
24. **MILP models are never answered with their LP relaxation.** A model with integer columns
    goes to the prototype branch-and-bound (dense double-double oracle as node solver,
    depth-first then best-bound, most-fractional branching, rounding heuristic, no cuts);
    too-large models get NotSolved with a reason. `Options::relax_integrality` solves the
    relaxation explicitly and says so in the message. Tested against brute-force enumeration
    (knapsack, 60 pure-integer, 10 mixed-integer random programs).
25. **Batch at T=8760 (K=4) was slower (0.68×)**: all four scenarios optimal and verified,
    but the hardest one took 360k iterations in the batch vs 265k alone — rounding-level
    differences (reduction order) on a chaotic, very long run. At T=365 iteration counts
    matched exactly. SpMM batching remains primarily a GPU idea; CPU gains are modest.
26. **Wrong MILP answer found by the MIPLIB run and fixed (egout).** Presolve had no notion of
    integrality: a singleton row tightened a binary column to x ≥ 0.183…, the column became
    empty and was fixed at that fractional value, and the tree reported 468.15 "Optimal"
    (true optimum 568.1007) — caught by tools/verify.py. Fixes: presolve rounds integer
    bounds inward (conflict ⇒ Infeasible), and every MILP answer that went through presolve
    is re-verified on the ORIGINAL model (bounds, rows, integrality) before it may be called
    Optimal. Regression test: Mip.PresolveRoundsIntegerBoundsInward.

## 2026-09-26 (integration of gpuopt)

27. **Crossed bounds are valid, infeasible data** (Model contract §6, agreed by both code
    bases). `Model::validate()` used to reject `lower > upper` as an invalid model
    (NotSolved); the teammate's reader and engines treat it as a legal model that is
    Infeasible — as HiGHS does, and as MPS files allow (`LO 3` / `UP 2`). The crossing is a
    complete infeasibility certificate, so `solve()` now returns **Infeasible** (message
    "bounds cross: …") before presolve or any engine runs; batch scenarios likewise.
    `validate()` still rejects NaN, `lower = +inf`, `upper = −inf` and every structural
    error. Tests that used crossed bounds as their "invalid model" example now use a
    genuinely invalid one, and additionally assert Infeasible for crossed bounds.
28. **MILP node LPs by the sparse simplex, pruned by certified bounds.** The branch-and-bound
    node solver was the dense double-double oracle (exact, O(m(n+m)) per pivot, refused
    beyond 2M tableau entries). It is now the teammate's sparse primal simplex (cold start
    per node), with the oracle selectable (`--set mip_node_solver=1`) and used as the
    fallback if a node LP fails. Because an fp64 LP objective can exceed the true node
    optimum by rounding, nodes are pruned by the Neumaier–Shcherbina certified bound from
    the node duals (`core/safe_bound`); the plain LP value is used only when that bound is
    infinite, and such prunes are counted in the message. *Evidence (small MIPLIB 3, 60 s,
    M4, evaluation run at the integration branch):* optimal within the limit 10/14 (simplex)
    vs 8/14 (oracle); misc03 0.6 s vs > 60 s, p0201 3.5 s vs > 60 s, stein27 1.7 s vs 16.6 s;
    the two tiniest models are slightly faster with the oracle. All answers agree;
    `Mip.SimplexAndOracleNodeSolversAgree` compares both on 60 random MIPs. Still missing
    for real MIP performance: warm-started nodes (basis I/O or dual simplex), cuts,
    primal heuristics (bell3a/bell5 find no incumbent in 60 s).
29. **`Auto` = simplex for small models, r²HPDHG for large ones.** Measured at the integration
    branch (M4, 1 thread): full Netlib — simplex 92/93 solved and verified, r²HPDHG 85/93
    (60 s); refinery T=365 (17.9k rows) — simplex 12.5 s vs r²HPDHG 0.34 s; rand-10000 —
    simplex 278 s vs 0.20 s. Simplex work grows like rows·nnz (full pricing + Devex row every
    iteration), so Auto picks the simplex when rows·nnz ≤ 2·10⁸ (`kAutoSimplexWork`) and
    r²HPDHG otherwise; a warm start or a first-order knob always selects r²HPDHG (the simplex
    has no warm start yet). The chosen engine is stated in the message ("auto: simplex
    (rows*nnz = …)"). The threshold is a tuned constant (Netlib + generated models) — an
    explicit `--algorithm` always overrides it. An explicitly chosen simplex reports "warm
    start ignored" instead of silently dropping it.
30. **Simplex defaults kept: geometric scaling + Devex pricing** (teammate's choices, now
    measured on the full set). Full Netlib, 60 s, at `fc3f29c`: default 92/93 solved and
    verified (68.6 s total for the solved ones); no scaling 91/93 (`cycle` lost; pilot87 passes
    every check but is 1.1e-6 from HiGHS); Dantzig pricing 90/93. PDHG scaling decisions are
    separate (#17). CSVs: `netlib-full-simplex-fp64[-noscale|-dantzig]-macbook-air-m4-fc3f29c.csv`.

## 2026-09-27 (overnight)

31. **Infeasible / Unbounded are verified claims too** (`core/certificates.h`). The gate used to
    re-check only Optimal; the other two verdicts rested on the engines' own tests.
    - *Infeasible*: a Farkas vector r (row multipliers) with L₀(r) > 0, where L₀ is the
      zero-objective Lagrangian bound, evaluated first with the rounding-proof machinery of
      `core/safe_bound` (rigorous); only if that cannot decide (a floating-point ray whose Aᵀr is
      1e-16 instead of 0 on a free column gives −∞) the PDLP-style tolerance test
      (violation ≤ 1e-8·L₀ after normalising). Which stage passed is reported.
    - *Unbounded*: a feasible point (verifier-grade, per row) and a ray d in the recession cone
      with (sense·c)ᵀd < 0 (`tol::kVerifyRay` = 1e-8).
    - Engines: r²HPDHG/PDLP return the rays they already computed; the simplex returns ρ = R y
      from its phase-1 duals (for Bᵀy = c_B, c_B = ±1 on the violated basics, the internal
      certificate −Mᵀy gives min over the box = Σ infeasibilities > 0; in original variables that
      is L₀(ρ) with ρᵢ = rowscaleᵢ·yᵢ) and the entering direction at an unbounded ratio test.
    - Gate: a failing certificate → NumericalError; a missing one → status kept, `check` empty,
      message "not certified" (e.g. the dense oracles, which return none). With presolve, an LP
      Infeasible/Unbounded verdict (or a presolve infeasibility) is re-derived by solving the
      original model, so the certificate lives in the original space; this also cross-checks
      presolve's verdict. Objective is NaN for both statuses.
    - Found on the way: r²HPDHG claimed Unbounded when the L2-relative primal residual was ≤ 1e-6
      although one row was violated by 1.04e-6 (random MPS model #119); the claim now also needs
      the per-row verifier test, as Optimal does. The certified bound gave −∞ for exactly
      cancelling reduced costs on free columns; fixed with exact expansion arithmetic (`1f2102c`).
    - *Evidence*: Differential.* — every Infeasible/Unbounded verdict certified: simplex
      2192/2192, r²HPDHG 1032/1032 (1,800 random LPs), 0 mismatches, 0 extra first-order limits.
    - *Independent check* (`tools/verify.py`): L₀ in exact rational arithmetic, with column bounds
      implied by iterated one-row propagation where a multiplier points at an infinite bound. Each
      implied bound is exact for one step and then rounded outward to a double, so it stays valid
      and denominators cannot grow across passes (unrounded, wood1p + objective cut ran > 10 min in
      gcd; rounded: 1.8 s, same verdict).
    - *Presolve*: a reduced-model Farkas vector is mapped to the original rows by
      `postsolve_farkas` (kept rows keep r_i; a column bound created by a removed singleton row
      moves onto that row, r_row = λ_j / a — exact) and checked on the original; the original is
      re-solved only if that fails. Follow-up solves get only the remaining time/iteration budget.
    - *First-order detection*: r²HPDHG also tests the drift z − z0 since the restart anchor (every
      4th check). Small Netlib + objective cut, 3M-iteration budget: 10/10 certified (3/10 with
      T(z) − z alone); adlittle + cut 56M → 13k iterations with both changes.

32. **Pseudocost branching in the MILP prototype** (`mip/branch_and_bound.cpp`). Most-fractional
    branching could not solve gt2 in 190k nodes; pseudocosts (per column and direction, learned
    from solved children, product score, column average for columns without history) solve it
    in ~3.4k nodes (`Mip.PseudocostBranchingSolvesGt2WithinANodeBudget`). The ε floor is on the
    per-unit gain, not on the product: with zero gains everywhere (enigma, a pure feasibility
    model) the rule must fall back to most fractional — an ε on the product made every score
    equal and branched in column order (enigma 0.8 s → 17 s, fixed → 3.7 s).
    Also a fractional-diving heuristic (root, then every 100 nodes without / 1000 with an
    incumbent; dive LPs capped at a fifth of node LPs). Scratch runs, 60 s: pk1 incumbent 30 → 18,
    bell5 final incumbent slightly worse (8966493 vs 8966406), others unchanged
    (`Mip.DivingFindsAnIncumbentWhereRoundingFails`). *Evidence*: `miplib3-macbook-air-m4-b04f2d8.csv`,
    300 s: 12/14 proven optimal and verified (was 10/14 at `eb90bbf`): gt2 0.7 s, bell3a 134 s;
    pk1 (incumbent 12, optimum 11) and bell5 (gap 0.02%) at the limit.

33. **First-order Unbounded: stop at the first valid ray, find the feasible point separately.**
    The engines required a primal-feasible *iterate* before claiming Unbounded, but on an
    unbounded LP the iterate drifts along the ray: random 7x7 model (seed 2027 #1932) ran 100M
    iterations with a valid ray from iteration ~2000. Now the engine stops at the first ray that
    passes the ray test; if its iterate is not feasible, the dispatcher solves the zero-objective
    model (never unbounded) with the same engine and the remaining budget: feasible → point + ray
    go to the gate; infeasible → that certified Infeasible is the answer; limit → honest limit.
    Random MPS cross-check vs HiGHS, 2000 models, seed 2027: r2hpdhg 1999 → 2000/2000, PDLP
    1969 → 2000/2000 (`Certificates.FirstOrderUnboundedWithoutAFeasibleIterate`).

34. **Soundness fix: a Farkas vector must prove more than rounding noise.** Random MPS cross-check
    (seed 99, model 2004, 3x4, FEASIBLE, optimum −6): r²HPDHG and PDLP returned Infeasible and
    the certificate passed the gate AND verify.py. r had L0 = 0 exactly (a zero-cost dual
    direction; 49r − 50r + r), +2e-15 in fp64, no violation; the rigorous stage correctly gave
    −8e-18, but the tolerance fallback accepted any L0 > 0. Now (gate, engine ray test, verify.py)
    L0 must exceed 1e-8 × Σ|summands| (inside λ_j too), and verify.py never lets the tolerance
    test override a conclusive exact L0 ≤ 0 (all terms finite, no implied bounds). Re-check of
    the 23 objective-cut models whose certificates were tolerance-only: simplex 23/23, r²HPDHG
    15/15 still certified (`Certificates.ZeroMeasureFarkasVectorIsRejected`,
    `CertificateVerifier.test_zero_measure_farkas_vector_is_rejected`).

35. **CPU first-order kernels: branch-free projections, parallel checks, nnz-balanced SpMV, fused
    steps** (2026-09-30, profiling on a 4-core x86 VM; docs/AGENT_PROGRESS.md). (a) `clamp` as
    `min(max(v, lo), hi)`: the nested ternary compiled to data-dependent branches and `r2h_dual`
    ran ~3× its branch-free time on real iterates; bit-identical results. (b) `kkt_general` and
    `ray_test` (every termination check) are fixed-chunk parallel reductions
    (`la::parallel_reduce`) with reused buffers; they were serial and allocating (22% of the
    iteration time at 4 threads). Thread-count invariant; Netlib iteration counts identical.
    (c) `Csr::multiply` chunks by nnz + rows instead of by ≥ 4096 rows: wide matrices (osa-*,
    fit2d) never ran A x in parallel. (d) `Backend::r2h_{primal,dual}_fused` compute each row's
    product and use it at once (no `aty` / `ax` round trip; CUDA keeps the default = unfused pair).
    All bit-identical across thread counts (`PdhgThreads.*`, `PdhgBackend.FusedStepsEqualProductThenStep`).

36. **r²HPDHG primal weight: rounding-noise floor on the PID displacements** (`pid_noise_rel`
    = 1e-10, `pdhg::PrimalWeightPid`, rationale and evidence in `pdhg/r2hpdhg.h`). Once one side
    has converged its displacement is proportional to its step, the PID loop is then unstable
    (ω_new ∝ ω^1.99) and ends up chasing rounding noise (sierra: ω 1 → 2.5e-7, 961k iterations).
    Displacements below 1e-10·(1 + ‖iterate‖) now count as degenerate (cuPDLPx: absolute 1e-16),
    i.e. fall back to the best ω — except while an iterate is diverging (norm more than doubles
    between restarts), which is the infeasible/unbounded case where the ω runaway helps the ray.
    Netlib: sierra 961k → 64.7k, perold 241k → 172k iterations, all 85 kept; objective-cut
    infeasible Netlib: 71 certified before and after. The batched engine uses the same controller.

37. **Presolve R5: rows redundant by activity bounds** (`core/presolve.*`). A row whose activity
    range over the model's own column bounds lies inside [rl, ru] — after a worst-case rounding
    margin, so a row that is redundant only in floating point is kept
    (`Presolve.RedundantRowTestIsSafeAgainstRounding`) — is removed with y = 0. Kennington osa-*:
    37 such rows carry ~39% of the nonzeros; osa-07 10,368 → 7,360 and osa-14 18,368 → 12,928
    r²HPDHG iterations (≈2× faster), Kennington total time −13%. Netlib r²HPDHG 85/85 (geomean
    +0.6%), `auto` 93/93, objective-cut infeasible 71 certified before and after (verify.py exact
    rational PASS on the affected ones). Using bounds tightened by R4 instead removed more rows
    and lost agg (and, without the PID floor, bnl2): not done.
    Tried and rejected (same session): R6, rows redundant under bounds implied by single other
    rows (sources kept, so sound). On osa-07 it removes 71 rows / 79,408 nonzeros — the same as
    HiGHS's first presolve pass — but r²HPDHG gets worse where it matters: osa-07 7,360 → 11,392
    iterations (osa-14 −8%, osa-30 −12%), Netlib shifted geomean +7.5% with agg 217k → 10.7M and
    sierra 65k → 351k iterations. Redundant rows are not free to remove for a first-order method:
    their y is 0 but they change the diagonal scaling and the conditioning of the iteration.


38. **Late-convergence "reversal" on cre-b / slow tail on rand-1e6: investigated, no change adopted**
    (2026-09-30, cloud x86 VM, HEAD 9702583; traces with the new `-vvv`, which prints r, r⁰, ω,
    relative residuals and ‖x̂‖, ‖ŷ‖ at every check and changes nothing else — bit-identical).
    *cre-b* (122,368 iterations): relative KKT ≈ 3e-7 at 21–30k with ω ≈ 0.15. At 32,576 a
    necessary-decay restart applies the PID step ω ÷10 (Δy/Δx = 6.7e-3 ≪ ω: the dual has
    converged, the primal moves); later steps ÷4, ÷10, ÷10 down to 3.8e-5. Epochs at ω ≤ 3.8e-3 make
    no progress on the fixed-point residual (r/r⁰ 0.89–0.99 for 19k and 17k iterations, ended only
    by the artificial restart); the primal drifts (‖x̂‖ 11.88 → 15.3, Δx 0.1 → 8.2) and the primal
    residual rises to 1e-3 before the method recovers with ω ≈ 1e-4…2e-3 and converges. The endgame
    needs ω ~100–1000× smaller than the plateau's, so the ÷10 moves point the right way; the cost is
    overshoot plus stalled epochs whose length grows with the total iteration count. Healthy runs
    (rand-1e5, ken-18, pds-20) never exceed 15× their best KKT at a restart (cre-b: 400–1650×) and
    almost never end an epoch with r/r⁰ > 0.8.
    *rand-1e6* (17,408 iterations): no reversal — KKT decreases monotonically and every late epoch
    ends by sufficient decay (r/r⁰ ≈ 0.19) after ~2,900 iterations (~320 at 1e5 rows): the
    size-dependent linear rate of restarted Halpern. ω oscillates 3–7× between restarts; with ω
    frozen at 1 (diagnostic only) 14,592 iterations (−16%).
    Candidates, all tested on cre-b first (baseline 122,368):
    - A stall rejection (epoch ends artificially with r/r⁰ > 0.8 ⇒ revert ω, halve the trust region):
      695,936 — it reverts UP, away from the endgame ω.
    - A′ stall damping (keep the PID direction, halve the trust region): 139,456 (+14%).
    - B KKT rollback (KKT > 100× best ⇒ restart from the best iterate with its ω): time limit at
      806,800 — the best-KKT point lies inside a stalled epoch.
    - C early stall restart (residual fell < δ since half the epoch): time limit at ~800k for
      δ = 0.05 and 0.1 — the slow epochs do necessary work; frequent restarts destabilise ω.
    - Elasticity guard (earlier, #37 notes): time limit at 346k.
    - D reversal damping (a PID step opposite to the previous one is halved): cre-b 118,848 (−3%),
      rand-1e6 16,000 (−8%), Netlib 85/85 but shifted geomean +1.2% with 80bau3b ×1.58, beaconfd ×1.72,
      bore3d ×1.48 (bnl2 ×0.58, pilotnov ×0.45): gains on the hard cases too small for the scatter.
    None adopted (rule: improve the hard cases without materially degrading the rest). Open: a
    primal-weight estimate that is not a displacement ratio (displacements along a degenerate optimal
    face overstate the distance to the solution set), or a restart criterion that notices a stalled
    epoch without restarting healthy slow ones.
