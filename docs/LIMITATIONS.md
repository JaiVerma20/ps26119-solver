# Known limitations

Status words: **VERIFIED** (tested + benchmarked with a committed CSV), **IMPLEMENTED**
(tested, not fully benchmarked), **EXPERIMENTAL**, **PLANNED**. Numbers are in
`docs/EVIDENCE.md` (generated from committed CSVs only).

## GPU
- GPU evidence comes from **one consumer laptop GPU** (RTX 4050 Laptop 6 GB, WSL2, commit
  `183c59c`, `docs/EVIDENCE.md` §3), compared with the same laptop's CPU at 1 thread and 12 threads.
  Speed-ups are quoted against the fastest CPU configuration: refinery year 3.0×, 1e5 rows 3.0×,
  1e6 rows 4.2×; about even at 1.8e4 rows (refinery T=365, 1.1×).
- **The GPU is slower on small models**: 1e4-row random 0.8×, refinery T=12 (588 rows) 0.02×
  (kernel launch and transfer overhead dominates).
- compute-sanitizer (memcheck 0 errors, racecheck 0 hazards) ran on the `Gpu.*` unit tests only,
  under WSL2 with the NVIDIA debugger interface enabled; under the sanitizer's slowdown
  `Gpu.EnginesMatchCpuOnSmallNetlib` stopped at its 120 s test limit on its first model, so its
  remaining Netlib cases were not sanitizer-checked (the test passes without the sanitizer).
- The 1e6-row runs match the known optimum (≤ 1.4e-11) and pass the in-process gate, but
  `verify.py` is skipped at that size.
- No data-centre GPU (A100/H100) result, so no claim about fp64-rate cards. On the laptop card
  mixed precision gives little over fp64 (≤ ~10%, sometimes slower: more iterations).
- Batched scenarios have no GPU SpMM kernel yet.

## LP engines
- **Primal simplex** (VERIFIED on Netlib): full pricing, Devex row recomputed every
  iteration, product-form updates, no hypersparse FTRAN/BTRAN, no Forrest–Tomlin: fine up to
  a few thousand rows, slow beyond ~10⁴ rows (refinery T=365 and rand-10000 in EVIDENCE §2c);
  dfl001 (6,071 rows) is not solved within 60 s. No basis input/output → **no warm start**, no
  ranging. Infeasible / Unbounded verdicts carry a Farkas vector / ray that the gate checks on
  the original model (DECISIONS #31); a verdict whose certificate fails becomes NumericalError.
  Certificates on free columns are often only tolerance-checked (not rounding-proof).
- **r²HPDHG / PDLP** (VERIFIED on CPU): first-order accuracy (relative KKT 1e-8, per-row
  checks), not a vertex; **no crossover** yet. Degenerate / badly scaled Netlib models
  (pilot*, greenbea/b, d2q06c, fit2d) hit the 60 s limit. Infeasibility detection on "barely"
  infeasible LPs is much slower than the simplex (EVIDENCE §4e).
- **Auto** picks by a tuned size rule (rows·nnz ≤ 2·10⁸ → simplex); the full-Netlib `auto`
  result is in-sample for that threshold.
- A verified answer at the default 1e-6 verifier tolerances bounds the objective error only
  to about that level: in the unscaled-simplex ablation, pilot87 passed every check with an
  objective 1.1e-6 (relative) away from HiGHS. Default runs agree with HiGHS far more tightly
  (EVIDENCE §1b).
- Presolve is basic (empty rows, fixed/empty columns, singleton rows, integer bound rounding).
- No interior-point method, no QP, no MIQP/NLP.
- Time limits: engines check the clock every iteration, but first-order setup (scaling; ~5 s
  at 6M nonzeros on the M4) and the final verification are not interruptible, and the
  certified bound's refinement may use up to 1.1 × limit + 1 s. File reading is not counted.
- A first-order run with a tolerance looser than the verifier's (e.g. `--tol 1e-4`) reports
  Optimal at that tolerance with `check FAIL` stated — by request, not verifier-grade.

## MILP (EXPERIMENTAL / prototype)
- Branch-and-bound with cold-started sparse simplex node LPs, certified-bound pruning,
  pseudocost branching (most fractional until there is history), rounding and fractional
  diving heuristics. **No cuts, no strong/reliability branching, no node warm start.** Small MIPLIB 3 models only (EVIDENCE §4d for which are solved).
- Single-threaded tree.
- An **Infeasible** MILP verdict comes from an exhausted tree and carries no certificate
  (LP-style Farkas proofs do not apply); an **Unbounded** MILP verdict carries an integer point
  plus a ray, checked on the original model (DECISIONS: `Mip.UnboundedRelaxationIsDecidedNotAssumed`).
- When a node's certified bound is −∞ (free continuous columns without implied bounds), pruning
  falls back to the node LP objective; such prunes are counted in the message
  ("N prunes by an uncertified LP bound").

## Input / output
- MPS: no gzip, no QUADOBJ/QMATRIX (QP), SOS, semi-continuous bounds, indicator constraints
  (all rejected with an error, never silently ignored). Only the first RHS/RANGES/BOUNDS set is
  used. Fixed-format names limited to 8 characters.
- The negative-`UP` rule (lower → −∞) follows classic MPS practice; highspy (and hence
  `tools/verify.py`) keeps lower = 0 — such files show a fingerprint MISMATCH in the verifier.

## Evidence
- All CPU numbers come from one fanless MacBook Air M4; timings vary run to run (throttling,
  P/E cores); iteration counts are deterministic. In the final suite (`d824f82`) wall times were
  ~2× slower than in the pre-merge run for identical iteration counts, and rand-1e6 (fp64,
  1 thread) hit its 600 s limit — the row is kept.
- Generated refinery LPs have refinery structure but synthetic data (optimum known by
  construction). No plant data. Mittelmann large LPs not yet run.
