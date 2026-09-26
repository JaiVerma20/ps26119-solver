# Known limitations

Status words: **VERIFIED** (tested + benchmarked with a committed CSV), **IMPLEMENTED**
(tested, not fully benchmarked), **EXPERIMENTAL**, **PLANNED**. Numbers are in
`docs/EVIDENCE.md` (generated from committed CSVs only).

## GPU
- GPU evidence comes from **one consumer laptop GPU** (RTX 4050 Laptop 6 GB, WSL2, commit
  `82d376c`, `docs/EVIDENCE.md` §3). Correctness: 40/40 small-Netlib GPU solves verified, fp64 GPU
  iteration counts equal to the CPU's on every scaling model where both finished. The 1e6-row
  runs match the known optimum (≤ 1.4e-11) but `verify.py` is skipped at that size.
- **The CPU baseline in that run is ONE thread.** The multi-core baseline was not measured (the
  harness now runs 1 thread and all cores; next GPU run). Quote ratios only as "vs one CPU thread".
- **The GPU is slower on small models**: refinery T=12 (588 rows) 0.02×, 1e4-row random ≈ 0.9–1.0×;
  it wins from ~1e5 rows (3–4× vs one thread) and on the hourly refinery year (2.6–3.1×).
- **compute-sanitizer has not run**: under WSL2 (WDDM) it cannot attach ("Device not supported").
  The kernels are not memcheck/racecheck-verified; needs native Linux (university server).
- No data-centre GPU (A100/H100) result, so no claim about fp64-rate cards or mixed-precision gain
  there. On the laptop card mixed precision gave no consistent gain over fp64 to 1e-8 (0.75×–1.14×
  of the fp64 time; it sometimes needs more iterations).
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

## MILP (EXPERIMENTAL / prototype)
- Branch-and-bound with cold-started sparse simplex node LPs, certified-bound pruning,
  pseudocost branching (most fractional until there is history), one rounding heuristic.
  **No cuts, no strong/reliability branching, no primal heuristics beyond rounding, no node
  warm start.** Small MIPLIB 3 models only (EVIDENCE §4d for which are solved).
- Single-threaded tree.

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
