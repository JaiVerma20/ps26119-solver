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
`docs/PROVENANCE.md`). Every answer is verified before it is reported:
- **Optimal** LP answers are re-checked in-process on the original model (primal, dual, gap)
  and carry a rounding-proof bound on the optimum;
- **Infeasible / Unbounded** verdicts carry a certificate (Farkas multipliers / a feasible
  point plus a ray) that is checked on the original model — rigorously with directed rounding
  where possible; a verdict whose certificate fails becomes `NumericalError`, never a claim;
- every benchmark answer is checked again by `tools/verify.py`, an external verifier with an
  independent MPS reader (exact rational arithmetic for Farkas certificates where possible).

The code merges two independently written codebases — Jai's ps26119 (first-order engines,
GPU backend, MILP, evidence system) and Shivanshu Vats's gpuopt (MPS reader, sparse LU,
simplex, checker) — with both histories preserved: `docs/FINAL_INTEGRATION_REPORT.md`.

## Results (every number from `docs/EVIDENCE.md`; CPU rows: one run of `scripts/reproduce.sh` at commit `f440782`, MacBook Air M4)

| Benchmark | Result | Source CSV (`bench/results/`) |
|---|---|---|
| Netlib, all 93 LPs, 60 s each, `--algorithm auto` | **93/93** solved, verified by `tools/verify.py`, equal to HiGHS to 1e-6 (the auto threshold was chosen on this set) | `netlib-full-auto-fp64-macbook-air-m4-f440782.csv` |
| Netlib, simplex alone / r²HPDHG alone | 92/93 / 83/93 | `netlib-full-{simplex,r2hpdhg}-fp64-macbook-air-m4-f440782.csv` |
| Certified infeasibility: 93 Netlib LPs + an objective cut 1e-4 below the optimum, 60 s | simplex **91/93**, r²HPDHG **70/93** certified (Farkas certificate passes the in-process gate AND `tools/verify.py`; 127 proved in exact rational arithmetic); the rest are time limits, never a wrong verdict | `infeasible-cut-{simplex,r2hpdhg}-macbook-air-m4-f440782.csv` |
| Small MIPLIB 3 (branch-and-bound, 300 s) | **12/14** proven optimal, verified, equal to HiGHS; pk1 and bell5 at the limit | `miplib3-macbook-air-m4-f440782.csv` |
| Refinery planning LP, hourly year (429k rows, 1.5M nnz), r²HPDHG to 1e-8 | **15.2 s** (1 thread) / **11 s** (all 10 cores, mixed); equal to the known optimum (1.2e-13), verified | `scale-macbook-air-m4-f440782.csv` |
| Random LP, 1M rows, 6M nnz, r²HPDHG to 1e-8 | 570 s (1 thread) / 380 s (all cores); known optimum matched to 1.2e-11 (verify.py skipped at this size) | `scale-macbook-air-m4-f440782.csv` |
| What-if re-solves on the refinery year (warm start) | 0.25–0.69× the cold iterations; the cold "price" solve hit its 600 s limit, the warm ones finished | `warm-start-macbook-air-m4-f440782.csv` |
| GPU (CUDA backend), one RTX 4050 **Laptop** GPU, r²HPDHG to 1e-8 | 40/40 small-Netlib GPU solves verified; vs **one** CPU thread (Core 5 210H, WSL2): refinery year 3.1×, 1e5-row random 4.3×; **GPU slower on small models**. Multi-core CPU baseline and compute-sanitizer pending (`NEXT_STEPS.md`) | `scale-rtx4050-laptop-82d376c.csv`, `netlib-small-gpu-82d376c.csv` |

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
Exit codes: 0 optimal, 1 limit/infeasible/unbounded, 2 usage, 3 read error, 4 cannot write the output, 5 numerical / not solved.

Python (ctypes over the C API):
```python
import sys; sys.path.insert(0, "python"); import ps26119
r = ps26119.solve_mps("data/netlib_small/afiro.mps", algorithm="simplex", out="afiro.sol")
print(r.status_name, r.objective, r.check)
r = ps26119.solve_lp([3, 5], [[1, 0], [0, 2], [3, 2]], [float("-inf")]*3, [4, 12, 18], sense=-1)
```
C: `include/ps26119/ps26119.h` (`ps26119_solve_lp`, `ps26119_solve_lp_ex` with certificates,
`ps26119_solve_mps`; no exceptions cross the API, invalid arguments return
`PS26119_INVALID_ARGUMENT` with a message), library `build/libps26119.{dylib,so}`. `solve()` is
safe to call from several threads at once.

Reproduce all CPU evidence from one commit: `PS26119_MACHINE=<name> scripts/reproduce.sh`
(stages and timings in the script header), then `python3 bench/make_evidence.py`.

## Documentation

`docs/ARCHITECTURE.md` (design) · `docs/FORMATS.md` (statuses, CLI, exit codes, file formats, APIs) · `docs/EVIDENCE.md` (all numbers) · `docs/BENCHMARKS.md`
(methodology) · `docs/LIMITATIONS.md` · `docs/DECISIONS.md` · `docs/PROVENANCE.md` ·
`docs/GPU_VERIFICATION.md` · `docs/CONTRIBUTING.md` · `docs/DEVELOPMENT.md` ·
`docs/FINAL_INTEGRATION_REPORT.md` · `docs/SIH_STATUS.md` (requirement status matrix) · `docs/audit/` (the independent audit of both codebases).
