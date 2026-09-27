# Next steps before the SIH PPT

Status of `main` and exactly what is left. Every number in the docs comes from a committed,
hash-named CSV (`docs/EVIDENCE.md`); nothing here is a target presented as a result.

## Shivanshu (NVIDIA hardware — nobody else can do these)

1. **Re-run the GPU validation on the current `main`** (Linux or WSL2):
   ```bash
   git pull && PS26119_MACHINE=rtx4050-laptop scripts/gpu_check.sh
   ```
   What changed since your first run (82d376c):
   - the CPU baseline is now measured at 1 thread AND all cores (`--threads 1,0`), so the
     GPU-vs-CPU table can show a fair "vs best CPU" ratio;
   - `scale.py`'s GPU summary no longer comes out empty;
   - time limits are enforced per iteration and certificates are checked for every
     Infeasible/Unbounded verdict (the GPU path uses the same code);
   - `--gpu` with `--algorithm auto` now always selects r²HPDHG (the GPU engine).
   Open a PR with `bench/results/*.csv` + `bench/results/logs/<machine>-<hash>/` as before.
   `python3 bench/validate_results.py bench/results` must pass (provenance, ctest 100%,
   sanitizer summaries).
2. **compute-sanitizer on native Linux** (memcheck + racecheck of the CUDA kernels). WSL2
   cannot attach ("Failed to initialize WDDM debugger interface" / "Device not supported").
   Options: a university GPU server, a native Linux boot, or NVIDIA's
   `EnableDebuggerInterface.bat` (admin) on Windows. `gpu_check.sh` runs both tools and writes
   `sanitizer-*.log`; the validator reports a tool that could not attach as NOT RUN.
3. **A data-centre GPU run** (university server, A100/H100 class) with the same command, for
   the fp64-rate comparison (docs/GPU_VERIFICATION.md D5) and the 1e6-row / refinery-year numbers.
4. **Review** (your code, changed during the integration/hardening):
   - `src/simplex/primal_simplex.cpp`: phase 1 tightens its dual tolerance (≤ 4×) until the
     Farkas certificate passes (DECISIONS #31) — 4 NumericalErrors became certified.
   - `src/io/mps_reader.cpp`: an input with no MPS section at all is now a read error (it
     read as a 0×0 model and "solved" to Optimal).
5. **Simplex speed (your area; suggestion from a profile of dfl001):** every iteration
   recomputes all duals from scratch (btran + full pricing) and Devex does a second btran and
   row; updating d_j from the pivot row in phase 2 should save ~25% per iteration. Then
   Forrest–Tomlin + hypersparse FTRAN/BTRAN.

## Jai / integration (no NVIDIA needed)

- Merge the GPU PR after checking it (provenance, GPU model/driver/CUDA, ctest, sanitizer
  status, fp64 vs mixed, CPU baseline fairness), then `python3 bench/make_evidence.py`.
- Decide the licence (the repository has no LICENSE file; needed before any public release).
- PPT: every slide number from `docs/EVIDENCE.md`; GPU ratios stated as "vs 1 CPU thread"
  until the all-core rerun lands; say plainly that compute-sanitizer has not run yet.

## Known limitations to state in the PPT (docs/LIMITATIONS.md)

- MILP is a prototype (no cuts, no strong branching, cold node LPs): 12/14 small MIPLIB 3.
- First-order solutions are accurate to the stated tolerance but are not vertices (no crossover).
- r²HPDHG infeasibility detection on "barely" infeasible LPs is slow (70/93 in 60 s vs 91/93
  for the simplex).
- No QP.
