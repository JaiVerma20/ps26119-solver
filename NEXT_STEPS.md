# Next steps before the SIH PPT

Status of `main` and exactly what is left. Every number in the docs comes from a committed,
hash-named CSV (`docs/EVIDENCE.md`); nothing here is a target presented as a result.

## Shivanshu (NVIDIA hardware — nobody else can do these)

Done 2026-09-27 (PR #11, commit `183c59c`): GPU rerun on current `main` with 1- and 12-thread CPU
baselines, 198/198 tests on the CUDA build, compute-sanitizer memcheck + racecheck clean.

1. *(optional, if time)* Re-run the sanitizer on `Gpu.EnginesMatchCpuOnSmallNetlib` with a larger
   test time limit (it stopped at 120 s on its first model under the sanitizer, so its other
   Netlib cases were not sanitizer-checked), e.g. temporarily `o.time_limit = 1200` in
   `tests/unit/test_gpu.cpp`, or filter to one model at a time.
2. **A data-centre GPU run** (university server, A100/H100 class; native Linux) with `scripts/gpu_check.sh`, for
   the fp64-rate comparison (docs/GPU_VERIFICATION.md D5) and the 1e6-row / refinery-year numbers.
3. **Review** (your code, changed during the integration/hardening):
   - `src/simplex/primal_simplex.cpp`: phase 1 tightens its dual tolerance (≤ 4×) until the
     Farkas certificate passes (DECISIONS #31) — 4 NumericalErrors became certified.
   - `src/io/mps_reader.cpp`: an input with no MPS section at all is now a read error (it
     read as a 0×0 model and "solved" to Optimal).
4. **Simplex speed (your area; suggestion from a profile of dfl001):** every iteration
   recomputes all duals from scratch (btran + full pricing) and Devex does a second btran and
   row; updating d_j from the pivot row in phase 2 should save ~25% per iteration. Then
   Forrest–Tomlin + hypersparse FTRAN/BTRAN.

## Ready for the PPT (done, on `main`)

- **All CPU evidence from one commit** (`f440782`, `scripts/reproduce.sh`): `docs/EVIDENCE.md`
  starts with a headline table — copy slide numbers from there, with the source file.
- **Live demo**: `scripts/demo.sh` (~15 s on the M4): verified LP with every engine, a certified
  Infeasible and Unbounded answer re-checked by `tools/verify.py`, a MILP, the hourly refinery
  year (429k rows) solved and verified, and hardened input handling.
- Interfaces and guarantees: `docs/FORMATS.md`; limitations: `docs/LIMITATIONS.md`.

## Jai / integration (no NVIDIA needed)

- Decide the licence (the repository has no LICENSE file; needed before any public release).
- PPT: every slide number from `docs/EVIDENCE.md`. GPU: quote "vs the fastest CPU configuration
  of the same laptop" (refinery year 3.0×, 1e6 rows 4.2×), say the GPU is slower on small
  models, and that it is one consumer laptop GPU (no data-centre GPU yet).
  r²HPDHG alone solves 83/93 Netlib in the f440782 run (85/93 at d824f82 — same iteration
  counts; two models near the 60 s limit timed out on a slower machine state): quote the
  f440782 number, and `auto` 93/93 as the product result.

## Known limitations to state in the PPT (docs/LIMITATIONS.md)

- MILP is a prototype (no cuts, no strong branching, cold node LPs): 12/14 small MIPLIB 3.
- First-order solutions are accurate to the stated tolerance but are not vertices (no crossover).
- r²HPDHG infeasibility detection on "barely" infeasible LPs is slow (70/93 in 60 s vs 91/93
  for the simplex).
- No QP.
