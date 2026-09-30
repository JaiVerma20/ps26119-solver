# Agent progress log (cloud session 2026-09-30)

Machine for all numbers below: cloud VM, 4 x86-64 cores, 15 GB, no GPU ("cloud-x86-4c").
It is ~4.5x slower per core than the MacBook Air M4 of the committed CSVs, so compare
before/after on THIS machine only.

## Baseline (commit a239f97)
- Netlib 93, r2hpdhg fp64 1e-8, 60 s, 1 thread (4 models in parallel): 85/93 Optimal,
  shifted-geomean iterations 20,666. Unsolved: d2q06c greenbea greenbeb pilot pilot.ja pilot.we pilot4 pilot87.
- Kennington 16, r2hpdhg, 300 s: 16/16 (osa-60 75.8 s, cre-b 107.6 s, ken-18 59.7 s).
- refinery-T8760-s1 (429k x 517k, 1.5M nnz), r2hpdhg: 1 thread 34.4 s, 4 threads 14.3 s (2880 iterations).

## Findings (profiling)
1. clamp() compiled to data-dependent branches: r2h_dual 3x slower than branch-free on real data.
2. KKT check + ray test serial & allocating: 22% of iteration time at 4 threads.
3. Csr::multiply splits by ROW count (>=4096 rows/chunk): wide matrices (osa-*, fit2d) never
   parallelise A x. osa-30 4 threads only 1.44x.

## Changes
| commit | change | result (same machine) |
|---|---|---|
| 43cd3d8 | branch-free projections | refinery 1 thr 34.4 -> 25.8 s, 4 thr 14.3 -> 10.5 s; bit-identical |
| 49716c1 | parallel allocation-free KKT + ray test | refinery 4 thr 10.5 -> 7.7 s; Netlib iterations identical |
| b9205a4 | nnz-balanced SpMV + grain 8192 | osa-30 4 thr 0.85 -> 0.49 ms/it; ken-13 0.33 -> 0.24 ms/it; bit-identical |
| b9ecd66 | fused SpMV + step kernels | bit-identical; ~17% per iteration in isolation, <=5% in full solves (VM noise +-5%) |
| d081992 | backend references scaled matrices, lazy fp32 | refinery peak RSS 397 -> 324 MB |
| 698a72a | PID rounding-noise floor (1e-10) + divergence guard | Netlib 85/85, sierra 961k -> 64.7k, perold 241k -> 172k it; cut set 71/71 certified |
| b19a2c9 | presolve R5 redundant rows (model's own bounds) | osa-07 10368 -> 7360 it, osa-14 18368 -> 12928; Kennington time -13%; Netlib 85/85 (+0.6%); auto 93/93 |

## Experiments (convergence)
- PID primal weight collapses when one side converged (step-limited displacement => unstable
  loop). Rounding-noise floor pid_noise_rel: 1e-10 fixes sierra+perold but alone broke
  objective-cut infeasibility detection (blend, share2b, scsd8, ganges): on infeasible LPs the
  other side DIVERGES and the omega runaway helps. Divergence guard (norm more than doubles
  between restarts => floor off) restores identical detection. Rejected: hold omega below floor,
  bound omega to 10^+-4, best-omega-before-update.
- pilot*, greenbea, d2q06c: long epochs (100k-300k it) with r/r0 ~ 0.5-1 and 10x omega swings;
  not fixed (auto mode sends small models to simplex anyway).
- HiGHS presolve removes 57% of osa-07 nnz; activity-redundant rows alone give 39%.

- R5 with bounds tightened by R4 (chained): Netlib agg lost / bnl2 lost / agg3 x58 -> use the
  model's own bounds only. PID floor "reset once" variant did not fix agg-with-chained-R5 (omega
  oscillates best <-> 10x best) -> reverted.

## Current / next
- Timing benchmark running (scratchpad timing.py): base a239f97 vs new, min of 3, 1 and 4 threads.
- Then: sanitizer builds (asan/ubsan, tsan), check_no_solver_linked, CUDA syntax check, evidence
  CSVs (netlib_full r2hpdhg, scale) at the final commit.
- Open: rand-1e6 needs 17.4k iterations vs 2.5k at 1e5 (same as older M4 runs: not a regression).

## Reproduce
Netlib: `python3 tools/fetch_netlib.py`; Kennington MPS from the mirror in tools/fetch_kennington.py
(netlib.org is blocked in this sandbox); refinery: `python3 bench/generate_refinery_lp.py --periods 8760 --seed 1 --out bench/generated/refinery-T8760-s1.lpm`.
