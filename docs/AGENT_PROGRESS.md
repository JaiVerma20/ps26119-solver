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

## Current / next
- nnz-balanced SpMV partitioning (in progress).
- Then convergence work on r2HPDHG (osa-07 plateau: gap stuck ~1e-5 for 2.5k iterations
  in one artificial-restart epoch; 8 Netlib models unsolved at 60 s).

## Reproduce
Netlib: `python3 tools/fetch_netlib.py`; Kennington MPS from the mirror in tools/fetch_kennington.py
(netlib.org is blocked in this sandbox); refinery: `python3 bench/generate_refinery_lp.py --periods 8760 --seed 1 --out bench/generated/refinery-T8760-s1.lpm`.
