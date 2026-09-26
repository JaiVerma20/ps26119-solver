# Known limitations

Status words: **VERIFIED** (tested + benchmarked with a committed CSV), **IMPLEMENTED**
(tested, not fully benchmarked), **EXPERIMENTAL**, **PLANNED**. Numbers are in
`docs/EVIDENCE.md` (generated from committed CSVs only).

## GPU
- The CUDA backend (`src/gpu/cuda_backend.cu`) is **compiled and correctness-tested on NVIDIA
  hardware** (teammate's laptop, WSL2, 2026-09-27: all GPU end-to-end tests pass — GPU answers equal
  CPU answers on 10 Netlib LPs × 2 engines × fp64/mixed). **No GPU benchmark CSV exists yet, so no
  GPU speed claim is made.** `scripts/gpu_check.sh` produces the CSVs; `docs/GPU_VERIFICATION.md`.
- Batched scenarios have no GPU SpMM kernel yet.

## LP engines
- **Primal simplex** (VERIFIED on Netlib): full pricing, Devex row recomputed every
  iteration, product-form updates, no hypersparse FTRAN/BTRAN, no Forrest–Tomlin: fine up to
  a few thousand rows, slow beyond ~10⁴ rows (refinery T=365 and rand-10000 in EVIDENCE §2c);
  dfl001 (6,071 rows) is not solved within 60 s. No basis input/output → **no warm start**, no
  ranging. No infeasibility (Farkas) or unboundedness (ray) certificate is returned; those
  statuses rest on a fresh-factorization re-check.
- **r²HPDHG / PDLP** (VERIFIED on CPU): first-order accuracy (relative KKT 1e-8, per-row
  checks), not a vertex; **no crossover** yet. Degenerate / badly scaled Netlib models
  (pilot*, greenbea/b, d2q06c, fit2d) hit the 60 s limit.
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
  most-fractional branching, one rounding heuristic. **No cuts, no strong/pseudocost
  branching, no primal heuristics beyond rounding, no node warm start.** Small MIPLIB 3
  models only; bell3a/bell5 find no incumbent within the limit.
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
