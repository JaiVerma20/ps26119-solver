# Benchmarks — how the evidence is produced

All numbers quoted anywhere come from CSVs in `bench/results/` whose file name carries the
git short hash of the code that produced them (CLAUDE.md §5.1). `docs/EVIDENCE.md` is
generated from the committed CSVs by `bench/make_evidence.py`.

| Script | What it measures | Output |
|---|---|---|
| `bench/netlib_small.py` | 10 small Netlib LPs × {oracle, PDLP-style PDHG, r²HPDHG} × {fp64, mixed}; every solution checked by `tools/verify.py` and against the published optimum | `netlib-small-<hash>.csv` (`netlib-small-gpu-<hash>.csv` with `--gpu`) |
| `bench/scale.py` | generated LPs with known optimum: random sparse (1e4–1e6 rows) and refinery (T = 12, 365, 8760); CPU vs GPU, fp64 vs mixed; iterations / wall time to 1e-4 and 1e-8, ms per iteration, setup time; optional HiGHS reference (`--highs`) | `scale-<machine>-<hash>.csv` + `.png` |
| `scripts/gpu_check.sh` | the one command for GPU machines: build with CUDA, all tests, both benches with `--gpu` | CSVs + `bench/results/logs/<machine>-<hash>/` |
| `scripts/reproduce.sh` | the same on a CPU-only machine | CSVs |

Every CSV row records: git hash, machine label (`PS26119_MACHINE`), CPU, GPU model, driver,
CUDA version, precision, tolerance, date.

## Definitions
- **Relative KKT** (first-order engines): see `src/pdhg/termination.h` — L2 primal residual
  / (1+‖b‖), L2 dual residual / (1+‖c‖), relative gap; both 1e-4 and 1e-8 are reported
  from the same run (the 1e-4 milestone is recorded on the way to 1e-8). A status of
  Optimal at 1e-8 additionally requires the verifier-grade per-row checks (≤ 1e-6).
- **Verify**: `tools/verify.py` re-reads the original model with a different reader and
  recomputes primal/dual feasibility and the gap from x and y only (tolerances in
  `include/ps26119/tolerances.h`).
- **Time**: wall clock of the solve inside the process (excludes file reading); `setup` is the
  part before the first iteration (scaling, ‖A‖₂ power iteration, device upload).
- **Known optimum**: generated instances carry a sidecar `.json` with the optimum from the
  KKT construction (`bench/lpgen.py`); HiGHS reproduces it in `bench/test_generators.py`.

## Caveats
- CPU engines are single-threaded (deterministic); OpenMP is optional and off by default.
- HiGHS is called through highspy in the benchmark process as an external reference; its
  default algorithm returns a vertex at simplex accuracy, not a 1e-8 relative-KKT point.
- Random LPs of this construction favour first-order methods; Mittelmann / Netlib-large
  runs are the next step before any general speed claim.
