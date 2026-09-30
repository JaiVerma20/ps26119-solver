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

## Final evidence (all CSVs in bench/results/cloud-x86-4c/, machine cloud-x86-4c, HEAD 1dcd40e, base a239f97)
Before/after, `before-after-cloud-x86-4c-1dcd40e.csv` (bench/compare_binaries.py, interleaved, min of 3,
every run Optimal + verify PASS, iteration counts deterministic):

| model | 1 thread base -> new | 4 threads base -> new | iterations |
|---|---|---|---|
| refinery-T8760-s1 (429k x 517k, 1.5M nnz) | 38.9 -> 23.1 s (1.68x) | 13.0 -> 8.7 s (1.49x) | 2880 = |
| rand-100000-s1 | 13.5 -> 10.8 s (1.25x) | 5.8 -> 4.5 s (1.28x) | 2496 -> 2624 |
| sierra (Netlib) | 23.0 -> 1.50 s (15.3x) | 23.5 -> 1.51 s (15.6x) | 961216 -> 64704 |
| osa-07 | 3.72 -> 1.90 s (1.96x) | 3.34 -> 1.22 s (2.74x) | 10368 -> 7360 |
| osa-14 | 15.6 -> 8.6 s (1.81x) | 11.6 -> 3.8 s (3.04x) | 18368 -> 12928 |
| osa-30 | 33.7 -> 25.4 s (1.32x) | 23.7 -> 9.7 s (2.45x) | 20672 -> 17216 |
| osa-60 (1.4M nnz) | 85.1 -> 76.1 s (1.12x) | 52.8 -> 24.6 s (2.14x) | 18176 -> 20160 |
| ken-18 | 59.6 -> 43.7 s (1.36x) | 26.6 -> 17.5 s (1.52x) | 24576 = |
| pds-20 | 30.9 -> 31.1 s (0.99x) | 15.6 -> 14.0 s (1.11x) | 26368 = |
| cre-d | 22.5 -> 21.8 s (1.03x) | 15.7 -> 8.7 s (1.80x) | 27968 = |

Netlib 93, r2hpdhg, official harness (sequential, 60 s, verify.py):
`netlib-full-r2hpdhg-fp64-cloud-x86-4c-{a239f97,1dcd40e}.csv`: 85/93 -> 85/93 (all = HiGHS to 1e-6);
shifted geomean iterations 20666 -> 20004 (-3.1%), time -5.3%, total 226.5 -> 182.4 s on the 85.
Scale (`scale-cloud-x86-4c-1dcd40e.csv`): all Optimal + verify PASS, error vs known optimum <= 3.7e-10;
refinery year 1 thread 25.8 s fp64 / 22.6 s mixed, 4 threads 8.9 s fp64.
Peak RSS refinery year fp64: 397 -> 324 MB (d081992).
Sanitizers (CI flags): asan+ubsan 197/197, tsan 198/198; CUDA front-end check OK; no foreign solver linked.

HiGHS 1.15.1 on the same machine, `compare-highs-cloud-x86-4c-1dcd40e.csv` (bench/compare_highs.py,
Kennington 16 + refinery T12..T8760; run stopped before the two random LPs; "solved" = Optimal +
verify.py PASS + objective within 1e-6 of the reference; r2hpdhg* = 4 threads):

| model | ps r2hpdhg 1 thr | ps r2hpdhg* 4 thr | HiGHS simplex | HiGHS IPM | HiGHS PDLP |
|---|---|---|---|---|---|
| refinery-T365 | 0.55 s | 0.57 s | 8.51 s | 2.66 s | not solved |
| refinery-T2190 | 4.91 s | 2.0 s | 127 s | 70.2 s | not solved |
| refinery-T8760 | 20.8 s | 8.6 s | time limit 300 s | time limit 300 s | not solved |
| osa-14 | 9.0 s | 4.0 s | 1.08 s | 1.13 s | 5.21 s |
| osa-60 | 71.1 s | 25.2 s | 20.2 s | 6.92 s | 53.9 s |
| ken-18 | 43.3 s | 17.3 s | 4.15 s | 7.56 s | not solved |
| cre-b | 104 s | 39.9 s | 1.61 s | 2.85 s | not solved |

Reading: on the refinery family r2HPDHG is far ahead of every HiGHS engine (T8760: HiGHS simplex
and IPM do not finish in 300 s); on Kennington HiGHS's simplex/IPM are 3-25x faster than us, and
our 4-thread r2HPDHG beats HiGHS PDLP where both verify (osa family). No "faster than HiGHS" claim
beyond the refinery family.

## Current / next
- Evidence pipeline complete (HiGHS run stopped before rand-1e4/1e5 at the user's checkpoint request).
- tools/highs_ref.py fixed: O(m^2) solution writing (hours on ken-18) -> O(n) (a725006).
- R6 (implied-bound redundant rows, sources kept) implemented and REJECTED: osa-07 71 rows /
  79,408 nnz removed (= HiGHS first pass) but 7,360 -> 11,392 iterations; Netlib geomean +7.5%
  (agg 217k -> 10.7M, sierra 65k -> 351k). Redundant rows change the scaling/conditioning.
- cre-b (122k it, the slowest Kennington model) reaches rel-KKT ~3e-7 at 23k iterations and then
  goes BACKWARDS (primal residual up to 9e-4 at 69k, plateau until ~100k): omega collapses
  0.149 -> 3.8e-5 in four restarts while the primal displacement GROWS (0.04 -> 8.2), i.e. the
  displacements measure the steps, not the distance to the solution. Analysis: with R = dy/dx and
  elasticity eps = dlogR/dlogomega, the PID update has local multiplier 1 + K_P(eps - 1): stable
  only for eps < 1 (distance-limited eps ~ 0, step-limited eps ~ 2); cre-b's first collapse step
  measures eps ~ 1.35. Tried (knob, reverted): on eps > 1 fall back to the best omega — cre-b
  worse (time limit at 346k it), also when restricted to same-direction runaways: the early
  "best" omega is wrong for the late phase and the collapse restarts from it. The diagnosis
  stands; the remedy is open (candidates: hold omega instead of resetting only while eps > 1;
  estimate eps from a deliberate probe; restart criterion on KKT error as PDLP).
- NEXT ACTION: the 1e-6 -> 1e-8 tail on the largest/hardest LPs (rand-1e6 17.4k it, cre-b 122k):
  restart epochs stop reaching sufficient decay and omega swings ~7x between restarts. Candidates:
  KKT-error-based restart as in PDLP alongside the fixed-point one; omega smoothing in the tail.
  (Global K_P 0.5 was tested and is worse: Netlib +21% geomean, 4 models lost.)
- Open: rand-1e6 needs 17.4k iterations vs 2.6k at 1e5 (same on older M4 runs, not a regression):
  after ~1.9k iterations restarts stop reaching the 0.2 sufficient decay and omega swings 0.5 <-> 3.9.

## Reproduce
Netlib: `python3 tools/fetch_netlib.py`; Kennington MPS from the mirror in tools/fetch_kennington.py
(netlib.org is blocked in this sandbox); refinery: `python3 bench/generate_refinery_lp.py --periods 8760 --seed 1 --out bench/generated/refinery-T8760-s1.lpm`.
