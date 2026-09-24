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

## 2. Team split — what is NOT yours
- **The MPS reader belongs to a teammate** (`src/io/mps_reader.*`). Do NOT write,
  rewrite or "fix" it. Code only against the Model contract in §6.
- Until the reader lands, get real instances through the temporary bridge
  `tools/mps_to_lpm.py` (Prompt 2), which converts MPS → our `.lpm` text format
  using highspy. It is test tooling only and is also used later to cross-check
  the teammate's reader (two independent readers must produce the same Model).

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

## 7. Solution contract
Status: Optimal, Infeasible, Unbounded, IterationLimit, TimeLimit, NumericalError,
NotSolved. Fields: x, row_activity, y, z, objective, primal_residual,
dual_residual, gap, iterations, seconds, engine, precision, model fingerprint.
CLI exit codes: 0 optimal, 1 limit/infeasible/unbounded, 3 read error, 5 numerical.

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
include/ps26119/   model.h solution.h tolerances.h options.h version.h ps26119.h (C API)
src/io/            mps_reader.* (TEAMMATE), lpm_reader.* (ours), solution_writer.*
src/core/          solve() dispatcher, status, model checks, fingerprint
src/la/            CSC/CSR, SpMV/SpMM (CPU), power iteration, dd.h (double-double)
src/oracle/        dense bounded simplex in double-double (TEST ORACLE ONLY)
src/pdhg/          CPU PDLP-style PDHG + CPU r2HPDHG, scaling, termination, precision policy
src/gpu/           CUDA r2HPDHG behind a backend interface
src/simplex/       (later) sparse bounded dual simplex, crossover
src/mip/           (later) branch and bound
apps/cli/          ps26119 solve file.{mps,lpm} [--algorithm ...] [--gpu] [--precision mixed|fp64]
tools/             verify.py, mps_to_lpm.py
bench/             runners, generators, results/*.csv
scripts/           gpu_check.sh, reproduce.sh
tests/unit/        GoogleTest
data/              small committed instances + SOURCES.md
docs/              RESEARCH.md ARCHITECTURE.md BENCHMARKS.md THIRD_PARTY.md
```

## 11. Milestones (the PPT is due in ~4 days; evidence first)
M0 skeleton, contracts, verifier, MPS→lpm bridge, double-double oracle (10 hand LPs + afiro)
M1 CPU PDLP-style PDHG + CPU r²HPDHG, validated on 10 small Netlib LPs, CSV
M2 CUDA r²HPDHG (fp64 + mixed), same answers as CPU; scaling + refinery CSVs from
   the teammate laptop and a university GPU; chart
M3+ after the PPT: warm-started PDHG, batched scenarios, crossover, sparse dual
   simplex, presolve, MILP, QP — see docs/RESEARCH.md §9.2

## 12. Style
No exceptions across the C API. RAII, no raw new/delete. Every algorithm file starts
with purpose, citations, invariants. Deterministic by default on CPU.
