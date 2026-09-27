# CLAUDE.md — PS26119 GPU optimization solver

Project codename: `ps26119` (final product name TBD — the brand name lives ONLY in
`include/ps26119/version.h`, so a rename is one edit).
Background research: `docs/RESEARCH.md`. Read it before large changes.

## 1. What we are building
A from-scratch LP / MILP / QP solver for Smart India Hackathon 2026, PS 26119
(MRPL). Goal: beat every competing team AND be usable in industry after SIH.
Core differentiator: a GPU-native LP engine based on restarted Halpern PDHG with
reflection (r²HPDHG, the cuPDLPx algorithm family), with an exact simplex beside it
for vertices, duals and warm starts, aimed at large multi-period refinery LPs.
Library first; CLI and Python are thin layers.

## 2. Team split (after the 2026-09-26 integration — one repository)
- Shivanshu Vats wrote the MPS reader, sparse LU, primal simplex, tableau oracle and the
  solution checker (gpuopt, merged here WITH history; `Origin:` lines in the headers). He is
  the primary reviewer for `src/io/mps_*`, `src/la/sparse_lu.*`, `src/simplex/`,
  `src/oracle/dense_tableau.*`, `src/core/solution_checker.*`. Change the Model contract (§6)
  only together.
- Jai: first-order engines, GPU backend, dispatcher/gate, presolve, certified bound, MILP,
  C API/Python, benchmarks and evidence.
- Everyone works in THIS repository on `feature/<person>/<topic>` branches with pull
  requests (docs/CONTRIBUTING.md). `tools/mps_to_lpm.py` remains only to produce `.lpm`
  files for generated models and to cross-check the reader (two independent readers must
  produce the same Model: `Integration.MpsReaderMatchesLpmBridgeOnCommittedData`).

## 3. Machines and build
| machine | role |
|---|---|
| MacBook Air M4 (main dev, where you run) | all CPU code, tests, tooling. **No CUDA here.** |
| teammate's NVIDIA laptop | CUDA build + GPU tests |
| university GPU servers | CUDA build + all published GPU benchmarks |
| GitHub Actions (ubuntu) | CPU CI on every push |

macOS notes (important):
- Apple clang, C++20, CMake ≥ 3.20, Ninja (Homebrew). OpenMP needs `brew install libomp`
  and explicit flags — keep OpenMP optional (`PS26119_ENABLE_OPENMP`).
- **On Apple Silicon `long double` == `double` (64-bit).** Never rely on long double
  for extra precision. Use the in-house double-double type (`src/la/dd.h`) or exact
  rationals where extra precision is needed.
- CUDA code (`src/gpu/*.cu`) is written here but compiled only when
  `-DPS26119_ENABLE_CUDA=ON`. Keep a CPU implementation of every GPU kernel's
  math, so the Mac can unit-test the algorithm; the GPU machines test the kernels.
- `scripts/gpu_check.sh` = the one command the GPU machines run: configure with
  CUDA, build, run all tests, run GPU benchmarks, write CSVs with machine info.

Every benchmark CSV records: git hash, machine, CPU, GPU model, driver, CUDA
version, precision, tolerance, date.

## 4. Source policy (what you may learn from and reuse)
You MAY freely read and learn from any public source: papers, textbooks, and public
repos including competitors (SANKHYA, VYUHA/SOVOPT), cuPDLPx, HiGHS, cuOpt, SCIP.
Use them to understand algorithms, find pitfalls, design tests, and set targets.

Reuse rules — because the PS requires a solver "not built on an existing solver"
and judges/competitors can compare public repos:
1. **Solver core (`src/`)**: write our own implementation. You may follow a public
   implementation's *approach* closely, but not paste it. Cite what informed it
   (paper + "approach informed by <repo>/<file>") in the header comment.
2. **Tooling, tests, benchmarks (`tools/`, `bench/`, `tests/`)**: permissively
   licensed code (MIT / Apache-2.0 / BSD) may be adapted WITH attribution in the file
   header and an entry in `docs/THIRD_PARTY.md` (licence + source URL + what was used).
3. **Data** (MPS instances, published optima, benchmark lists) may be used freely
   with the source recorded in `data/SOURCES.md`.
4. Never copy GPL code anywhere. Never remove a licence header.
5. Nothing from any other solver is ever linked into the binary. CI checks this.

## 5. Hard engineering rules
1. **Evidence rule.** No accuracy or speed claim in README/docs unless it comes from
   a CSV in `bench/results/` whose filename contains the git short hash.
2. **Tests with every change.** Never weaken, skip or delete a test to make it pass.
3. **Tolerances live in one place:** `include/ps26119/tolerances.h`.
4. **Wrong answer is worse than no answer.** If unsure, return `NumericalError` or a
   limit status, never `Optimal`.
5. **Small steps.** One feature per branch/commit; build + tests green before moving on.
6. Before finishing a task: `cmake --build build -j && ctest --test-dir build` passes;
   list the tests you added.
7. `highspy` only in `tools/` and `bench/` as a separate reference, never in `src/`.

## 6. The model contract (shared with the MPS reader author — do not change alone)
    minimise   sense * (cᵀx) + obj_offset      (sense = +1 min, −1 max)
    subject to row_lower ≤ A x ≤ row_upper
               col_lower ≤ x   ≤ col_upper
- A in CSC: `col_start` (n+1), `row_index`, `value`. CSR built on demand.
- Infinity = `std::numeric_limits<double>::infinity()`.
- MPS mapping: first N row → objective; E: lower = upper = rhs; L: (−inf, rhs];
  G: [rhs, +inf). RANGES R: E → [rhs, rhs+|R|] if R>0 else [rhs−|R|, rhs];
  L → [rhs−|R|, rhs]; G → [rhs, rhs+|R|]. BOUNDS UP LO FX FR MI PL BV LI UI.
  Missing RHS = 0. RHS on objective row → obj_offset = −value. MARKER → is_integer.
- Row/column names kept. `Model::fingerprint()` = hash of all numbers.
- Agreed at the integration (DECISIONS #27): crossed bounds (lower > upper) are valid data
  describing an INFEASIBLE model; negative `UP` with no explicit lower bound sets the lower
  bound to −inf with a warning (classic MPS rule; highspy keeps 0 — verify.py then reports a
  fingerprint mismatch, never a silent pass); `nan` tokens and non-finite coefficients are
  read errors; values ≥ 1e30 (and overflow) are infinite; duplicate RHS/RANGES entries warn
  and the last one wins; duplicate matrix entries are summed with a warning.

## 7. Solution contract
Status: Optimal, Infeasible, Unbounded, IterationLimit, TimeLimit, NumericalError,
NotSolved. Fields: x, row_activity, y, z, objective, primal_residual,
dual_residual, gap, iterations, seconds, engine, precision, model fingerprint, check (the
in-process verification gate: every Optimal LP answer is re-checked on the original model
by core/solution_checker; failure → NumericalError), certified bound.
CLI exit codes: 0 optimal, 1 limit/infeasible/unbounded, 2 usage, 3 read error, 4 cannot write output, 5 numerical / not solved.

## 8. Tolerances (defaults)
- First-order engines, relative KKT: 1e-4 "fast" and 1e-8 "high" — report both.
- Oracle: feasibility 1e-9, pivot 1e-11 (double-double arithmetic).
- Verifier: primal 1e-6, dual 1e-6, gap 1e-6 (document absolute vs relative).

## 9. How we beat the field (targets — each needs a CSV before it is claimed)
Gaps competitors state themselves (see docs/RESEARCH.md §2–3):
- GPU numbers only up to ~10k rows / ~1M nnz, one card, synthetic models
- No GPU result on refinery-structured models; hourly refinery year "not proved"
- Mittelmann large LPs mostly unsolved (0–3 of 8)
- PDHG iterate not warm-started across re-solves
- Consumer GPUs weak at fp64
Our unique ideas, in priority order:
1. **r²HPDHG on GPU** (Halpern + reflection + constant step + PID primal weight).
2. **Mixed precision**: fp32 inner iterations + fp64 residuals / restarts / final
   polish, so consumer GPUs (and later Apple GPUs, which have no fp64) run fast.
3. **Warm-started PDHG** across re-solves (SLP loops, rolling horizon, what-if).
4. **Batched scenarios**: solve many LPs sharing one matrix (crude-price or demand
   scenarios, SLP steps) in one GPU pass using SpMM instead of SpMV.
5. **Certified output from a GPU iterate**: safe dual bound (Neumaier-Shcherbina)
   + crossover to a vertex so duals and ranging are exact.
6. **Vendor-neutral GPU layer** (CUDA first; keep kernels behind an interface so
   HIP/ROCm or Metal can follow) — sovereignty also means not depending on one vendor.

## 10. Layout
```
include/ps26119/   model.h solution.h options.h tolerances.h version.h batch.h solve.h ps26119.h (C API)
src/io/            mps_reader.* mps_parser.h (Shivanshu), lpm_reader.*, solution_reader/writer.*
src/core/          solve() dispatcher + gate, presolve, safe_bound, implied_bounds, c_api,
                   solution_checker (Shivanshu), model checks, status
src/la/            CSR (+SpMV/SpMM), CSC + sparse LU (Shivanshu), dd.h, parallel.h
src/simplex/       bounded revised primal simplex + scaling (Shivanshu)
src/oracle/        dense_simplex (double-double, canonical oracle), dense_tableau (Shivanshu, test oracle)
src/pdhg/          PDLP, r2HPDHG, scaling, termination, batch, backend interface
src/gpu/           CUDA backend
src/mip/           branch-and-bound (sparse simplex node LPs, certified-bound pruning)
apps/cli/          ps26119 solve|info|print|lu-bench|batch
tools/             verify.py, mps_to_lpm.py, lpm.py, fetch_*.py, highs_ref.py, crosscheck_random_mps.py
bench/             netlib_small/full, scale, warm_start, batch, miplib3, lu_netlib, make_evidence, results/*.csv
scripts/           gpu_check.sh, reproduce.sh, check_no_solver_linked.sh, setup_github.sh
tests/unit/        GoogleTest;  tests/test_cli.py;  python/test_python.py
data/              small committed instances + SOURCES.md (examples/ from gpuopt)
docs/              ARCHITECTURE BENCHMARKS CONTRIBUTING DECISIONS DEVELOPMENT EVIDENCE
                   FINAL_INTEGRATION_REPORT GPU_VERIFICATION LIMITATIONS PROVENANCE RESEARCH
                   THIRD_PARTY  audit/  history/
```

## 11. Milestones
M0–M1 done (contracts, verifier, oracles, CPU PDLP + r²HPDHG, Netlib evidence).
Integration with gpuopt done 2026-09-26 (reader, sparse LU, simplex, checker gate, MILP node
LPs by simplex) — see docs/FINAL_INTEGRATION_REPORT.md.
M2 (open): CUDA r²HPDHG compiled and validated on NVIDIA hardware (`scripts/gpu_check.sh`).
M3+: simplex warm start / dual simplex (basis I/O), crossover from PDHG to a vertex,
Forrest–Tomlin + hypersparse LU, stronger presolve, MILP cuts/heuristics, QP
(docs/RESEARCH.md §9.2, docs/LIMITATIONS.md).

## 12. Style
No exceptions across the C API. RAII, no raw new/delete. Every algorithm file starts
with purpose, citations, invariants. Deterministic by default on CPU.
