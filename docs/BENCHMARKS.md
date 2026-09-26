# Benchmarks — how the evidence is produced

All numbers quoted anywhere come from CSVs in `bench/results/` whose file name carries the
git short hash of the code that produced them (CLAUDE.md §5.1). `docs/EVIDENCE.md` is
generated from the committed CSVs by `bench/make_evidence.py`.

| Script | What it measures | Output |
|---|---|---|
| `bench/netlib_small.py` | 10 small Netlib LPs × {oracle, simplex, PDLP-style PDHG, r²HPDHG} × {fp64, mixed}; every solution checked by `tools/verify.py` and against the published optimum | `netlib-small-<hash>.csv` (`netlib-small-gpu-<hash>.csv` with `--gpu`) |
| `bench/scale.py` | generated LPs with known optimum: random sparse (1e4–1e6 rows) and refinery (T = 12, 365, 8760); CPU vs GPU, fp64 vs mixed; iterations / wall time to 1e-4 and 1e-8, ms per iteration, setup time; optional HiGHS reference (`--highs`) | `scale-<machine>-<hash>.csv` + `.png` |
| `bench/netlib_full.py` | all 93 Netlib LPs (`tools/fetch_netlib.py` first), any engine (`--engine simplex|r2hpdhg|pdlp|auto`), 60 s each, read from MPS by our reader, gated in-process (`check` column) and verified by `tools/verify.py`, vs HiGHS and published optima; `--set k=v --tag t` for ablations | `netlib-full-<engine>-<prec>[-tag]-<machine>-<hash>.csv` |
| `bench/miplib3.py` | small MIPLIB 3 through branch-and-bound; every Optimal verified (feasibility + integrality) and compared with HiGHS | `miplib3-<machine>-<hash>.csv` |
| `bench/lu_netlib.py` | sparse LU on random simplex-style bases of every Netlib matrix (`ps26119 lu-bench`): backward error, fill, PFI update chains | printed table |
| `tools/crosscheck_random_mps.py` | random LPs written as MPS: status + objective vs SciPy/HiGHS (also a CTest) | printed summary |
| `bench/warm_start.py` | refinery what-if re-solves (price / demand / crude): cold vs warm vs warm+ω | `warm-start-<machine>-<hash>.csv` |
| `bench/batch.py` | K price scenarios: K separate solves vs one batched solve (SpMM) | `batch-<machine>-<hash>.csv` |
| `scripts/gpu_check.sh` | the one command for GPU machines: build with CUDA, all tests, both benches with `--gpu` | CSVs + `bench/results/logs/<machine>-<hash>/` |
| `scripts/reproduce.sh` | the same on a CPU-only machine | CSVs |

Every CSV row records: git hash, machine label (`PS26119_MACHINE`), CPU, GPU model, driver,
CUDA version, precision, tolerance, date. The git hash is taken from the BINARY that ran
(`ps26119 --version` prints the commit it was built from, `-dirty` if the tree had
uncommitted changes), never from the working tree at run time. Long runs: use a frozen copy of
the binary and `caffeinate -i` on macOS.

## Methodology (one for every engine)

Same input files (MPS through our reader; generated models as `.lpm`), same machine per CSV,
same time limit per suite, 1 thread unless the `threads` column says otherwise, the same
verification criteria (engine status Optimal → in-process gate → `tools/verify.py` with an
independent reader → agreement with HiGHS/published/known optimum), and the same CSV schema
with provenance columns. A model counts as solved only if all of these hold. Time limits and
failures stay in the tables. Numbers from different machines are never mixed in one claim.

## Definitions
- **Relative KKT** (first-order engines): see `src/pdhg/termination.h` — L2 primal residual
  / (1+‖b‖), L2 dual residual / (1+‖c‖), relative gap; both 1e-4 and 1e-8 are reported
  from the same run (the 1e-4 milestone is recorded on the way to 1e-8). A status of
  Optimal at 1e-8 additionally requires the verifier-grade per-row checks (≤ 1e-6).
- **Simplex**: returns a vertex; its status is final only after a fresh-factorization re-check,
  an unscaled-units check and the in-process gate (primal/dual/gap ≤ 1e-6 relative).
- **Verify**: `tools/verify.py` re-reads the original model with a different reader and
  recomputes primal/dual feasibility and the gap from x and y only (tolerances in
  `include/ps26119/tolerances.h`).
- **Time**: wall clock of the solve inside the process (excludes file reading); `setup` is the
  part before the first iteration (scaling, ‖A‖₂ power iteration, device upload).
- **Known optimum**: generated instances carry a sidecar `.json` with the optimum from the
  KKT construction (`bench/lpgen.py`); HiGHS reproduces it in `bench/test_generators.py`.

## Caveats
- CPU engines default to 1 thread; `--threads N` uses a deterministic pool (bit-identical
  results for any N). Scaling CSVs have a `threads` column.
- HiGHS is called through highspy in the benchmark process as an external reference; its
  default algorithm returns a vertex at simplex accuracy, not a 1e-8 relative-KKT point.
- Random LPs of this construction favour first-order methods; Mittelmann / Netlib-large
  runs are the next step before any general speed claim.
