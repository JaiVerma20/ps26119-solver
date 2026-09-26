# ps26119 — a from-scratch LP / MILP solver (SIH 2026, PS 26119, MRPL)

One library, several engines, one verification system:

- **r²HPDHG** (restarted Halpern PDHG with reflection, the cuPDLPx family) and PDLP-style
  PDHG — first-order engines for large sparse LPs; deterministic multi-threaded CPU backend,
  mixed precision, warm start, batched scenarios, and a CUDA backend behind the same interface.
- **Bounded revised primal simplex** on a Markowitz sparse LU (Devex, Harris ratio test, bound
  flips, perturbation, Bland fallback) — vertex solutions and exact duals for small and medium
  LPs, and the node solver of the branch-and-bound.
- **Branch-and-bound** for MILP (prototype) — pruning by rounding-proof certified bounds.
- **Exact oracles** — a dense double-double simplex and an independent dense tableau simplex.

Nothing from any other solver is inside (`scripts/check_no_solver_linked.sh` runs in CI; see
`docs/PROVENANCE.md`). Every Optimal LP answer is re-checked in-process on the original model
before it is reported, carries a rounding-proof bound on the optimum, and every benchmark
answer is additionally checked by an external verifier with an independent MPS reader.

The code merges two independently written codebases — Jai's ps26119 (first-order engines,
GPU backend, MILP, evidence system) and Shivanshu Vats's gpuopt (MPS reader, sparse LU,
simplex, checker) — with both histories preserved: `docs/FINAL_INTEGRATION_REPORT.md`.

## Results (CPU, MacBook Air M4; every number from `docs/EVIDENCE.md`)

| Benchmark | Result | Source CSV |
|---|---|---|
| Netlib, all 93 LPs, 60 s each, `--algorithm auto` | **93/93** solved, verified by `tools/verify.py`, equal to HiGHS to 1e-6 (auto threshold chosen on this set) | `bench/results/netlib-full-auto-fp64-macbook-air-m4-{d824f82,eb90bbf}.csv` |
| Netlib, simplex alone / r²HPDHG alone | 92/93 / 85/93 | `netlib-full-simplex-…-eb90bbf.csv`, `netlib-full-r2hpdhg-…-d824f82.csv` |
| Small MIPLIB 3 (branch-and-bound, 300 s) | 10/14 proven optimal, verified, equal to HiGHS | `miplib3-macbook-air-m4-eb90bbf.csv` |
| Refinery planning LP, hourly year (429k rows, 1.5M nnz), r²HPDHG to 1e-8 | 19.2 s (1 thread) / 14.7 s (10 threads, mixed); equal to the known optimum, verified | `scale-macbook-air-m4-d824f82.csv` |
| GPU | **not measured yet** — the CUDA backend has not run on NVIDIA hardware | — |

Limitations: `docs/LIMITATIONS.md`.

## Build and test (macOS arm64 / Linux)

```bash
python3 -m pip install --user cmake ninja highspy numpy scipy matplotlib   # tooling only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3)
cmake --build build -j && ctest --test-dir build
```
GPU machines (Linux / WSL2): `PS26119_MACHINE=<name> scripts/gpu_check.sh`. Details: `docs/DEVELOPMENT.md`.

## Use

```bash
build/ps26119 solve data/netlib_small/afiro.mps --out afiro.sol       # auto engine
build/ps26119 solve model.mps --algorithm simplex                     # or r2hpdhg | pdlp | oracle
build/ps26119 info model.mps          # statistics     build/ps26119 print model.mps   # the model as read
python3 tools/verify.py data/netlib_small/afiro.mps afiro.sol --expected -464.75314286
```
Useful options: `--tol 1e-8`, `--time-limit s`, `--threads 0` (all cores, identical results),
`--precision mixed`, `--gpu` (CUDA builds), `--warm prev.sol`, `--no-presolve`.
Many scenarios sharing one matrix: `build/ps26119 batch base.lpm s1.lpm s2.lpm --out-dir out/`.
Models with integer columns go to branch-and-bound; a MILP is never answered by its LP relaxation.
Exit codes: 0 optimal, 1 limit/infeasible/unbounded, 2 usage, 3 read error, 5 numerical.

Python (ctypes over the C API):
```python
import sys; sys.path.insert(0, "python"); import ps26119
r = ps26119.solve_mps("data/netlib_small/afiro.mps", algorithm="simplex", out="afiro.sol")
print(r.status_name, r.objective, r.check)
r = ps26119.solve_lp([3, 5], [[1, 0], [0, 2], [3, 2]], [float("-inf")]*3, [4, 12, 18], sense=-1)
```
C: `include/ps26119/ps26119.h` (`ps26119_solve_lp`, `ps26119_solve_mps`), library `build/libps26119.{dylib,so}`.

## Documentation

`docs/ARCHITECTURE.md` (design) · `docs/EVIDENCE.md` (all numbers) · `docs/BENCHMARKS.md`
(methodology) · `docs/LIMITATIONS.md` · `docs/DECISIONS.md` · `docs/PROVENANCE.md` ·
`docs/GPU_VERIFICATION.md` · `docs/CONTRIBUTING.md` · `docs/DEVELOPMENT.md` ·
`docs/FINAL_INTEGRATION_REPORT.md` · `docs/SIH_STATUS.md` (requirement status matrix) · `docs/audit/` (the independent audit of both codebases).
