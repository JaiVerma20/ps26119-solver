# gpuopt: an Indigenous Optimization Solver, Built From Scratch

**Smart India Hackathon, problem statement by Mangalore Refinery and Petrochemicals Limited (MRPL)**
*Category: Software · Theme: Smart Automation*

`gpuopt` is a mathematical optimization solver written from first principles in C++20. It reads an
industry-standard MPS model file and returns a **provably optimal** solution. The problem statement
forbids building on any existing solver library (CPLEX, Gurobi, HiGHS, GLPK, CBC, SCIP, OR-Tools,
SciPy), so every numerical component here is our own: the file reader, the sparse LU factorization,
the simplex method and the optimality checker. External solvers appear only as outside comparison
baselines in test scripts. They are never linked into the solver.

> **Status (September 2026):** Layers 1–3 of the 8-layer plan are complete. The solver **solves 48 of
> the 49 Netlib LP benchmark models tested**, and every answer is proven optimal by an independent
> checker. Mixed-integer (MILP), quadratic (QP) and GPU layers are planned and **not built yet**. See
> [Roadmap](#18-roadmap).

---

## Highlights

| What | Result |
|---|---|
| Netlib LP benchmark, solved to the published optimum and independently certified | **48 of 49** models tested (27 to 3,136 rows) |
| Agreement with a dense reference solver on random LPs (differential testing) | **7,400 / 7,400**, 0 mismatches |
| Agreement with SciPy/HiGHS on random MPS models, end to end | **1,000 / 1,000** (dense oracle) and **300 / 300** (simplex) |
| Layer 2 sparse LU on bases from all 49 Netlib models | **49 / 49 PASS**, residuals ≤ 1e-16, fill-in ≤ 1.8× |
| Sparse LU vs textbook dense LU (same basis) | **43×** faster on 25fv47, **~300×** faster on pilot87 |
| Netlib readme discrepancies found and resolved against HiGHS | 4 models (80bau3b, greenbea, pilot, pilot87), plus one objective-constant convention (e226) |

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Project status by layer](#2-project-status-by-layer)
3. [Architecture](#3-architecture)
4. [Repository layout](#4-repository-layout)
5. [Getting started](#5-getting-started)
6. [Using the solver](#6-using-the-solver)
7. [How it works: model representation](#7-how-it-works-model-representation)
8. [Layer 1: MPS reader](#8-layer-1-mps-reader)
9. [The dense reference oracle](#9-the-dense-reference-oracle)
10. [The independent solution checker](#10-the-independent-solution-checker)
11. [Layer 2: sparse LU factorization](#11-layer-2-sparse-lu-factorization)
12. [Layer 3: bounded revised primal simplex](#12-layer-3-bounded-revised-primal-simplex)
13. [Validation methodology](#13-validation-methodology)
14. [Results](#14-results)
15. [Issues found and fixed during development](#15-issues-found-and-fixed-during-development)
16. [Scripts reference](#16-scripts-reference)
17. [Known limitations](#17-known-limitations)
18. [Roadmap](#18-roadmap)
19. [Publishing this repository](#19-publishing-this-repository)
20. [References](#20-references)

---

## 1. Problem statement

A refinery decides daily which crudes to buy, which units process what, and how to blend the outputs.
Every decision must respect pipe capacities, sulphur limits, tank volumes and contracts, and among all
legal plans exactly one maximises margin. An optimization solver finds it.

Today that computation runs on foreign commercial software (CPLEX, Gurobi, FICO Xpress). It carries
licence costs, per-core licensing, and zero visibility into the algorithms. MRPL asks for an **Indian
engine, built from mathematical first principles**, that is transparent and auditable.

The problem statement stresses three requirements:

| Rule | What it means for this project |
|---|---|
| **Built from scratch** | No solver library may be linked or vendored. The sparse LU, ratio test and simplex are our own code. |
| **Engine, not interface** | A command-line binary is the deliverable; no GUI. |
| **Must not lie, must not hang** | Numerical robustness is named three times. A solver that prints a wrong number is worse than one that stops. That is why every answer here is independently certified. |

Problem classes in scope: **LP** (done), **MILP** and **convex QP** (planned), with a path to MIQP/NLP/MINLP.

---

## 2. Project status by layer

The solver is built bottom-up in eight layers. Each layer is tested on its own before the next one
starts.

| Layer | Component | Status |
|---|---|---|
| L1 | MPS reader (front end) | ✅ **Done**: free and fixed format, RANGES, BOUNDS, integer markers, OBJSENSE, objective constant |
| L1 | Presolve / postsolve | ⏳ Planned |
| L2 | Sparse LU, FTRAN/BTRAN, basis updates | ✅ **Done**: Markowitz + threshold pivoting, product-form (PFI) updates, singular-basis repair |
| L2 | Forrest-Tomlin update, hypersparse solves | ⏳ Planned |
| L3 | Bounded revised **primal** simplex | ✅ **Done**: scaling, Devex, Harris ratio test, bound flips, perturbation, Bland fallback |
| L3 | Dual simplex | ⏳ Planned (required for fast MILP warm starts) |
| L4 | Interior point method (Mehrotra) | ⏳ Planned |
| L5 | GPU first-order LP (PDLP, CUDA) | ⏳ Planned |
| L6 | Convex QP | ⏳ Planned |
| L7 | MILP branch-and-cut | ⏳ Planned |
| L8 | Public API (C ABI, Python bindings) | ⏳ Planned (CLI exists) |
| — | Dense reference oracle | ✅ Done |
| — | Independent solution checker | ✅ Done |
| — | Benchmark and validation tooling | ✅ Done |

---

## 3. Architecture

```
                      +------------------------------------------------+
  L8  PUBLIC API      |  CLI (gpuopt)  ·  C ABI / Python      (planned)|
                      +------------------------------------------------+
  L7  MILP            |  branch-and-cut, heuristics           (planned)|
  L6  QP              |  convex QP via interior point         (planned)|
  L5  GPU             |  PDLP first-order LP on CUDA          (planned)|
  L4  IPM             |  Mehrotra predictor-corrector         (planned)|
                      +------------------------------------------------+
  L3  SIMPLEX         |  bounded revised primal simplex          DONE  |
  L2  LINEAR ALGEBRA  |  sparse LU, FTRAN/BTRAN, PFI updates     DONE  |
  L1  FRONT END       |  MPS reader (presolve planned)           DONE  |
                      +------------------------------------------------+

  Verification, alongside every layer:
     dense reference oracle  ·  independent solution checker  ·  benchmark scripts
```

Data flow for one solve:

```
model.mps ──► MPS reader ──► LpProblem ──► primal simplex ──────────────► SolveResult
                                              │    ▲                          │
                                              ▼    │ FTRAN / BTRAN / update   ▼
                                           sparse LU (Layer 2)       solution checker
                                                                     (PASS / FAIL proof)
```

Two data structures are the fixed contracts between layers:

* **`LpProblem`** ([problem.hpp](include/gpuopt/problem.hpp)): the model in compressed sparse column
  (CSC) form. It is produced by the front end and consumed by every engine.
* **`SolveResult`** ([result.hpp](include/gpuopt/result.hpp)): the status, primal values, duals,
  iterations and time. Every engine returns this same structure, so the benchmark tooling scores all
  of them identically.

---

## 4. Repository layout

```
gpu_optimizer/
├── CMakeLists.txt                  build: core library, CLI, 5 test executables
├── README.md
├── include/gpuopt/                 public headers (the contracts between layers)
│   ├── problem.hpp                 LpProblem, SparseMatrixCSC, build_csc
│   ├── result.hpp                  SolveStatus, SolveResult
│   ├── mps_reader.hpp              Layer 1
│   ├── dense_oracle.hpp            reference solver
│   ├── solution_checker.hpp        independent optimality certificate
│   ├── linalg/sparse_lu.hpp        Layer 2
│   └── simplex/
│       ├── scaling.hpp             geometric scaling
│       └── primal_simplex.hpp      Layer 3
├── src/
│   ├── core/                       problem.cpp, result.cpp
│   ├── io/                         mps_reader.cpp
│   ├── linalg/                     sparse_lu.cpp
│   ├── simplex/                    scaling.cpp, primal_simplex.cpp
│   ├── oracle/                     dense_oracle.cpp
│   └── validation/                 solution_checker.cpp
├── apps/
│   ├── gpuopt_cli.cpp              the `gpuopt` command-line program
│   └── lu_bench.{hpp,cpp}          Layer 2 benchmark mode (--lu-bench)
├── tests/
│   ├── test_framework.hpp          minimal dependency-free test harness
│   ├── random_lp.hpp               random LP generator (shared)
│   ├── test_mps_reader.cpp         8 tests
│   ├── test_dense_oracle.cpp       12 tests
│   ├── test_oracle_random.cpp      5,000 random LPs, every optimum certified
│   ├── test_sparse_lu.cpp          9 tests (dense reference, singular, updates, large)
│   └── test_simplex.cpp            6 tests incl. 7,400-LP differential test vs oracle
├── scripts/
│   ├── fetch_netlib.py             download Netlib models
│   ├── bench_netlib.py             Layer 2 LU benchmark over all models
│   ├── solve_netlib.py             Layer 3: solve all models, score vs published optima
│   ├── highs_reference.py          independent HiGHS (SciPy) solve, comparison only
│   └── crosscheck_scipy.py         random MPS models: gpuopt vs SciPy/HiGHS
└── data/
    ├── examples/                   8 hand-made models with known answers
    └── netlib/
        ├── afiro.mps               smallest Netlib model (kept in the repo for the tests)
        ├── optimal_values.csv      published optimal values of all 95 Netlib models
        └── README.md
```

---

## 5. Getting started

### Prerequisites

| Tool | Version | Used for |
|---|---|---|
| C++ compiler | GCC 11+, Clang 14+ or MSVC 2022 (C++20) | building the solver |
| CMake | 3.20+ | build system |
| Ninja | any recent version | build tool (recommended) |
| Python | 3.8+ | benchmark and validation scripts only |
| SciPy + NumPy | recent | only for the comparison scripts (`highs_reference.py`, `crosscheck_scipy.py`, `solve_netlib.py --highs`) |

The solver itself has **no third-party dependencies**.

Development used GCC 15.2 (WinLibs MinGW-w64), CMake 4.1 and Ninja on Windows 11. On MinGW the
executables are linked statically, so `gpuopt.exe` runs outside the toolchain shell.

### Build

```bash
git clone <your-repo-url> gpu_optimizer
cd gpu_optimizer
cmake -S . -B build -G Ninja
cmake --build build
```

Without Ninja, `cmake -S . -B build` uses the platform's default generator (Makefiles or Visual Studio).

### Run the tests

```bash
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 5` (in about 2 seconds).

### First solve

```bash
./build/gpuopt data/netlib/afiro.mps --expect -464.7531428571
```

```
gpuopt 0.2  -  indigenous optimization solver

reading    data/netlib/afiro.mps
           27 rows, 32 columns, 83 nonzeros, free format, 0.001 s

method     revised primal simplex, Devex pricing, sparse LU, scaled, perturbed
status     OPTIMAL
note       phase-1 iters 3, bound flips 0, refactorizations 3, Bland episodes 0, Devex resets 0, ...
iterations 23
time       0.000 s
objective  -4.6475314286e+02
checker    primal viol 1.4e-14  dual viol 5.6e-18  compl 1.3e-14  gap 6.1e-17  => PASS
           primal objective -4.6475314286e+02   dual objective -4.6475314286e+02
expected   -4.6475314286e+02   relative error 9.2e-14  => MATCH
```

(On Windows use `.\build\gpuopt.exe`.)

### Get the Netlib benchmark set

```bash
python scripts/fetch_netlib.py --set all      # 49 models, ~23 MB, into data/netlib/
python scripts/solve_netlib.py --highs        # solve and score all of them
```

---

## 6. Using the solver

```
gpuopt <model.mps> [options]
```

| Option | Meaning |
|---|---|
| *(none)* | read, solve with the revised simplex, verify with the checker |
| `--method simplex\|oracle` | solver: revised simplex (default) or the dense reference oracle |
| `--expect VALUE` | compare the optimum with a known value (relative tolerance 1e-6) |
| `--solution` / `--duals` | print the non-zero primal values / row duals |
| `--info` | read only; print model statistics (row types, bounds, coefficient ranges) |
| `--print-model` | print the model back in algebraic form, exactly as the reader understood it |
| `--format auto\|free\|fixed` | MPS dialect (default `auto`: free, then fixed) |
| `--pricing devex\|dantzig` | simplex pricing rule (default Devex) |
| `--no-scale` / `--no-perturb` | switch off scaling / cost perturbation |
| `--log N` | print a progress line every N simplex iterations |
| `--time-limit S` / `--max-iter N` | limits |
| `--lu-bench` | Layer 2 benchmark: factorize bases built from the model's matrix |
| `--trials N` / `--updates N` | bases per structural fraction / PFI updates in the LU benchmark |

**Exit codes:** `0` success (including a proven INFEASIBLE or UNBOUNDED), `1` usage or read error,
`2` no definitive status (e.g. time limit), `3` verification failed (checker FAIL or `--expect` mismatch).

**Reading the output.** The `checker` line is the proof of optimality (see [§10](#10-the-independent-solution-checker)).
`note` shows solver statistics. `offset` appears when the model has an objective constant.

Example with a model of your own:

```bash
./build/gpuopt data/examples/refinery_blend.mps --print-model --solution --duals
```

```
parsed model
  minimize  60 CRUDE_A + 45 CRUDE_B
  subject to
    PETROL:    0.4 CRUDE_A + 0.3 CRUDE_B >= 12
    DIESEL:    0.5 CRUDE_A + 0.4 CRUDE_B >= 15
    SULPHUR:   -CRUDE_A + CRUDE_B <= 0
    CAPACITY:  CRUDE_A + CRUDE_B <= 50
  bounds
    0 <= CRUDE_A <= 30
    CRUDE_B >= 0
...
status     OPTIMAL
objective  1.8000000000e+03
```

---

## 7. How it works: model representation

Every engine works on one canonical form:

```
minimize / maximize   cᵀx + c₀
subject to            L ≤ A x ≤ U          (row bounds; L = U for equality rows)
                      l ≤ x ≤ u            (column bounds; ±∞ allowed)
                      x_j integer for flagged j   (read now, used by the future MILP layer)
```

`A` is stored in **compressed sparse column (CSC)** form: three arrays (column starts, row indices,
values). This is the layout the sparse LU and the simplex pricing loop need. Two-sided rows are
handled directly, with no conversion to `≤` form.

---

## 8. Layer 1: MPS reader

Source: [mps_reader.cpp](src/io/mps_reader.cpp). MPS is the 1960s IBM text format in which every
public benchmark is distributed:

```
NAME          EXAMPLE
ROWS
 N  COST                     <- objective row
 L  LIM1                     <- a·x <= b
COLUMNS
    X1        COST      1.0   LIM1      1.0
RHS
    RHS       LIM1      4.0
BOUNDS
 UP BND       X1        4.0
ENDATA
```

**How it works.** The file is read line by line. A line starting in column 1 is a section header, an
indented line is data, and `*` starts a comment. Each data line is split into fields and passed to a
handler for its section. Coefficients are collected as (row, column, value) triplets and converted to
CSC at the end, summing any duplicate entries.

**Supported:** `NAME`, `OBJSENSE` (MAX/MIN, on the same line or the next), `OBJNAME`, `ROWS` (N/E/L/G),
`COLUMNS` with integer `MARKER` blocks, `RHS`, `RANGES`, `BOUNDS` (UP LO FX FR MI PL BV LI UI), `ENDATA`,
Fortran-style numbers (`1.5D+03`, `.5`, `-1.`).

**Free vs fixed format.** Free format splits fields on whitespace. Fixed format uses the historical
column positions (2–3, 5–12, 15–22, 25–36, 40–47, 50–61), which lets names contain spaces. `auto`
tries free first and falls back to fixed, and says so in a warning.

**Conventions (the same as the major solvers):**

| Case | Interpretation |
|---|---|
| RHS entry on the objective row | the **negated** objective constant (`c₀ = −value`) |
| `|value| ≥ 1e30` | infinity |
| `UP` with a negative value and no lower bound given | lower bound becomes −∞ (with a warning) |
| `RANGES` on E rows | R > 0: [b, b+R]; R < 0: [b+R, b] |
| `RANGES` on L / G rows | [b−\|R\|, b] / [b, b+\|R\|] |
| Integer columns without bounds | [0, +∞) (the old binary default is available as an option) |
| Extra N rows | dropped, with a warning |
| Several RHS/RANGES/BOUNDS sets | the first set is used; the others are ignored with a warning |

**Refusing to guess.** Unsupported input (gzip, `QUADOBJ`/QP sections, SOS, semi-continuous bounds)
and malformed lines stop with an error naming the **line number**, e.g.
`line 17: unknown row 'PETRL'`. Nothing is silently skipped.

**Verified:** all 49 downloaded Netlib models parse, the largest (dfl001, 6,071 × 12,230) in 24 ms.

---

## 9. The dense reference oracle

Source: [dense_oracle.cpp](src/oracle/dense_oracle.cpp). The blueprint identifies *"a solver that runs,
terminates, prints a number, and is quietly wrong"* as the biggest risk, so the first engine written
was deliberately **slow but obviously correct**. Its job is to be the ground truth that every fast
engine is compared against.

1. **Standard form.** Each row gets an activity variable `r = a·x` with bounds [L, U]. Every bound is
   then removed by a textbook substitution: fixed values become constants, `x = l + z`, `x = u − z`,
   or a free variable is split as `x = z⁺ − z⁻`. The result is `min c'ᵀz, Mz = b, z ≥ 0`.
2. **Phase 1.** Rows are flipped so b ≥ 0, one artificial variable is added per row, and the sum of
   artificials is minimised. A positive optimum proves the model is **infeasible**.
3. **Phase 2.** The real objective is optimised from the phase-1 basis.
4. **Bland's rule** chooses the smallest-index entering and leaving variables. It is slow, but it is
   **guaranteed never to cycle**. `beale_cycling.mps` is the classic model on which the textbook rule
   loops forever, and the oracle solves it.
5. Duals are read off the reduced costs of the artificial columns.

It solved Netlib afiro and adlittle to their published optima, and it has certified 5,000 random LPs.

---

## 10. The independent solution checker

Source: [solution_checker.cpp](src/validation/solution_checker.cpp). The checker never trusts a solver.
Given only the model, a primal vector `x` and row duals `y`, it verifies from first principles (in
minimisation form, with `d = c − Aᵀy`):

| Check | Condition |
|---|---|
| Primal feasibility | `L ≤ Ax ≤ U` and `l ≤ x ≤ u` (scaled violation ≤ 1e-6) |
| Dual feasibility | `yᵢ ≥ 0` if only Lᵢ is finite, `≤ 0` if only Uᵢ is finite, `= 0` for a free row; the same for each `d_j` against its column bounds |
| Complementary slackness | no dual weight on an inactive bound (reported) |
| Zero duality gap | `cᵀx = Σ yᵢ·(Lᵢ or Uᵢ) + Σ d_j·(l_j or u_j)` (relative gap ≤ 1e-6) |

Primal feasibility, dual feasibility and a zero duality gap together are a **mathematical proof of
optimality**, no matter how the answer was found. That is the `=> PASS` in every solve. The checker
catches wrong answers: it flagged a real bug on `pilot` (see [§15](#15-issues-found-and-fixed-during-development)).

---

## 11. Layer 2: sparse LU factorization

Source: [sparse_lu.cpp](src/linalg/sparse_lu.cpp). The simplex never inverts the basis matrix `B`.
It keeps a factorization `B = LU` and answers two questions every iteration:

* **FTRAN:** solve `Bx = b` (how the basic variables move when a column enters)
* **BTRAN:** solve `Bᵀy = d` (the simplex multipliers used for pricing)

**Factorization.** Right-looking Gaussian elimination on the sparse *active submatrix*, stored both by
column (indices and values) and by row (indices), with rows and columns kept in count-bucketed linked
lists:

* **Markowitz pivoting** chooses the pivot that minimises `(rᵢ − 1)(c_j − 1)`, the worst-case fill-in.
  The search visits the sparsest columns and rows first and stops early (Suhl & Suhl).
* **Threshold stability test:** a pivot is accepted only if `|a_ij| ≥ τ · max_k |a_kj|`, with τ = 0.1.
  This trades sparsity against numerical safety, and it prevents pivoting on values like 1e-14.
* **U is stored twice**, by rows for BTRAN and by columns for FTRAN, so both solves skip zeros.

**Singular bases.** If some columns are linearly dependent, they are replaced by unit columns (slacks)
of the uncovered rows, the factorization is completed, and `replaced_columns()` reports each swap.
This is exactly what a simplex crash procedure needs.

**Updates.** After a basis change, the factors are updated in **product form (PFI)**:
`B_new⁻¹ = E⁻¹ B⁻¹`, with one eta vector per update. `needs_refactor()` requests a fresh factorization
after 100 updates or when the eta file outgrows the factors. The simplex also refactorizes when a
residual check detects drift.

**Verified** (see [§14](#14-results)): the sparse LU matches dense Gaussian elimination to 5e-13 on 300
random matrices; PFI chains match fresh factorizations; and bases built from all 49 Netlib models
factorize with residuals ≤ 1e-16.

---

## 12. Layer 3: bounded revised primal simplex

Source: [primal_simplex.cpp](src/simplex/primal_simplex.cpp). This is the engine that solves LPs.

**Internal form.** Every row gets a *logical* (slack) variable, so the model becomes `[A I] z = 0`
with bounds on every variable. The initial basis is the identity (all slacks basic), and nonbasic
structurals start at a bound.

**One iteration:**

```
1. phase    phase 1 if any basic variable violates a bound (cost −1 below / +1 above),
            otherwise phase 2 with the real costs ("composite" simplex)
2. duals    y = B⁻ᵀ c_B  (BTRAN),   d_j = c_j − a_jᵀy  for nonbasic j
3. pricing  entering q = the attractive d_j with the best Devex score d_j² / w_j
4. FTRAN    α = B⁻¹ a_q
5. ratio    Harris two-pass test → pivot row, or a bound flip, or "unbounded"
6. update   move the primal values, swap q into the basis (PFI update), update Devex weights
```

| Technique | Why |
|---|---|
| **Composite phase 1** | Minimises the sum of infeasibilities, then switches to the real objective. Feasibility lost to round-off is repaired the same way. |
| **Devex pricing** (Forrest & Goldfarb) | Approximate steepest edge. Measured against Dantzig pricing it needs about half the iterations on larger models: 25fv47 3,548 vs 7,183; d2q06c 23,347 vs 53,791. (On tiny models there is no gain: afiro 23 vs 20.) Uses the pivot row to update the reference weights. |
| **Harris two-pass ratio test** | Allows bound violations up to the tolerance so it can pivot on a larger, safer `|α|`. |
| **Bound flips** | A boxed entering variable that reaches its other bound just flips, with no basis change. |
| **Cost perturbation** | Small random cost shifts break ties on degenerate models. They are removed before the end, and the solver finishes on the true costs. |
| **Bland fallback** | After a long run of zero-length steps it switches to Bland's rule, which cannot cycle, until progress resumes. |
| **Geometric + cost scaling** | Rows and columns are scaled by powers of two (exact in floating point), and costs are scaled so the largest is about 1. This makes tolerances meaningful on badly scaled models. |
| **Periodic refactorization** | Primal values are recomputed from scratch after every refactorization, which removes drift. |
| **Verified termination** | Any OPTIMAL, INFEASIBLE or UNBOUNDED verdict is re-checked on a fresh factorization before it is believed. |
| **Optimality in original units** | Before declaring OPTIMAL, primal and dual violations are measured in the **unscaled** model; if too large, the tolerances are tightened and iterations continue. |

Default tolerances (scaled model): primal 1e-7, dual 1e-7, pivot 1e-7. Integrality flags are
ignored at this layer (the LP relaxation is solved), and the output says so.

---

## 13. Validation methodology

The success measure for this problem is unusually objective: public benchmarks with published optima.
Evidence is built in independent layers, and no single test is trusted on its own.

| Evidence | What it proves | Where |
|---|---|---|
| Hand-made models with hand-derived optima, confirmed by SciPy | every MPS feature and edge case (ranges, negative bounds, fixed format, cycling, infeasible, unbounded) | `data/examples/`, unit tests |
| 5,000 random LPs, every optimum certified by the checker | the oracle is correct | `test_oracle_random` |
| **Differential testing:** simplex vs oracle on 7,400 random LPs | the fast engine agrees with the slow, obviously correct one (status and objective) | `test_simplex` |
| Random MPS files vs SciPy/HiGHS | the whole pipeline (writer, reader, solver) agrees with an outside solver | `crosscheck_scipy.py` |
| Sparse LU vs dense Gaussian elimination, residual checks | Layer 2 is backward-stable | `test_sparse_lu`, `--lu-bench` |
| Netlib: published optima + checker + HiGHS where they disagree | real-world correctness and robustness | `solve_netlib.py` |

A Netlib model counts as **SOLVED** only when all three hold: the status is OPTIMAL, the independent
checker PASSes, and the objective equals the published value to 1e-6 (or, where the Netlib readme is
known to be off, equals an independent HiGHS solve, and the table says so).

---

## 14. Results

### 14.1 Test suites

```
1/5 Test #1: test_mps_reader ....   Passed      8 tests
2/5 Test #2: test_dense_oracle ..   Passed     12 tests
3/5 Test #3: test_oracle_random .   Passed      5,000 random LPs certified
4/5 Test #4: test_sparse_lu .....   Passed      9 tests
5/5 Test #5: test_simplex .......   Passed      6 tests, 7,400 random LPs vs oracle: 0 mismatches
100% tests passed, 0 tests failed out of 5
```

### 14.2 Layer 2 on Netlib (`python scripts/bench_netlib.py`)

Random simplex-style bases (50%, 80% and 100% structural columns) are built from each model's matrix,
factorized, and verified by residuals. All **49/49 PASS**:

| Measure | Result |
|---|---|
| Worst FTRAN/BTRAN residual (relative) | 7.6e-17 |
| Fill-in `nnz(L+U) / nnz(B)` | 1.00 – 1.80 |
| Slowest factorization | 16 ms (degen3, pilot87) |
| Largest model (dfl001, m = 6,071) | 4.6 ms |
| Condition numbers encountered | up to ≈10²⁴ (pilot87); solves stay backward-stable |
| Dense LU on the same basis | 44.9 ms vs 1.0 ms on 25fv47 (**43×**); 1,725 ms vs 5.7 ms on pilot87 (**~300×**) |

### 14.3 Layer 3 on Netlib (`python scripts/solve_netlib.py --highs`)

**48 of 49 solved.** Times are wall-clock on a Windows laptop, including process start-up. HiGHS
times come from SciPy's HiGHS and are shown for comparison only.

| Model | Rows | Cols | Status | Objective | Iterations | gpuopt (s) | HiGHS (s) | Verdict |
|---|---:|---:|---|---:|---:|---:|---:|---|
| afiro | 27 | 32 | OPTIMAL | -4.6475314286e+02 | 23 | 0.01 | 0.00 | SOLVED |
| kb2 | 43 | 41 | OPTIMAL | -1.7499001299e+03 | 47 | 0.02 | 0.00 | SOLVED |
| sc50a | 50 | 48 | OPTIMAL | -6.4575077059e+01 | 45 | 0.02 | 0.00 | SOLVED |
| sc50b | 50 | 48 | OPTIMAL | -7.0000000000e+01 | 48 | 0.02 | 0.00 | SOLVED |
| adlittle | 56 | 97 | OPTIMAL | 2.2549496316e+05 | 105 | 0.02 | 0.00 | SOLVED |
| blend | 74 | 83 | OPTIMAL | -3.0812149846e+01 | 100 | 0.02 | 0.00 | SOLVED |
| recipe | 91 | 180 | OPTIMAL | -2.6661600000e+02 | 49 | 0.02 | 0.00 | SOLVED |
| share2b | 96 | 79 | OPTIMAL | -4.1573224074e+02 | 109 | 0.03 | 0.00 | SOLVED |
| sc105 | 105 | 103 | OPTIMAL | -5.2202061212e+01 | 102 | 0.02 | 0.00 | SOLVED |
| share1b | 117 | 225 | OPTIMAL | -7.6589318579e+04 | 243 | 0.02 | 0.00 | SOLVED |
| stocfor1 | 117 | 111 | OPTIMAL | -4.1131976219e+04 | 96 | 0.02 | 0.00 | SOLVED |
| scagr7 | 129 | 140 | OPTIMAL | -2.3313898243e+06 | 138 | 0.02 | 0.00 | SOLVED |
| lotfi | 153 | 308 | OPTIMAL | -2.5264706062e+01 | 265 | 0.02 | 0.00 | SOLVED |
| boeing2 | 166 | 143 | OPTIMAL | -3.1501872802e+02 | 187 | 0.02 | 0.00 | SOLVED |
| beaconfd | 173 | 262 | OPTIMAL | 3.3592485807e+04 | 115 | 0.02 | 0.00 | SOLVED |
| israel | 174 | 142 | OPTIMAL | -8.9664482186e+05 | 171 | 0.02 | 0.00 | SOLVED |
| vtpbase | 198 | 203 | OPTIMAL | 1.2983146246e+05 | 193 | 0.02 | 0.00 | SOLVED |
| sc205 | 205 | 203 | OPTIMAL | -5.2202061212e+01 | 220 | 0.03 | 0.00 | SOLVED |
| brandy | 220 | 249 | OPTIMAL | 1.5185098965e+03 | 385 | 0.02 | 0.01 | SOLVED |
| e226 | 223 | 282 | OPTIMAL | -1.1638929066e+01 | 349 | 0.03 | 0.01 | SOLVED ² |
| bore3d | 233 | 315 | OPTIMAL | 1.3730803942e+03 | 179 | 0.02 | 0.00 | SOLVED |
| capri | 271 | 353 | OPTIMAL | 2.6900129138e+03 | 441 | 0.03 | 0.01 | SOLVED |
| scfxm1 | 330 | 457 | OPTIMAL | 1.8416759028e+04 | 443 | 0.04 | 0.01 | SOLVED |
| scorpion | 388 | 358 | OPTIMAL | 1.8781248227e+03 | 385 | 0.02 | 0.01 | SOLVED |
| scsd8 | 397 | 2750 | OPTIMAL | 9.0499999993e+02 | 2085 | 0.13 | 0.04 | SOLVED |
| ship04l | 402 | 2118 | OPTIMAL | 1.7933245380e+06 | 644 | 0.03 | 0.01 | SOLVED |
| ship04s | 402 | 1458 | OPTIMAL | 1.7987147004e+06 | 479 | 0.04 | 0.01 | SOLVED |
| pilot4 | 410 | 1000 | OPTIMAL | -2.5811392589e+03 | 1525 | 0.09 | 0.03 | SOLVED |
| degen2 | 444 | 534 | OPTIMAL | -1.4351780000e+03 | 1128 | 0.08 | 0.02 | SOLVED |
| perold | 625 | 1376 | OPTIMAL | -9.3807552782e+03 | 3771 | 0.30 | 0.04 | SOLVED |
| bnl1 | 643 | 1175 | OPTIMAL | 1.9776295615e+03 | 1343 | 0.08 | 0.02 | SOLVED |
| ship08l | 778 | 4283 | OPTIMAL | 1.9090552114e+06 | 1062 | 0.06 | 0.02 | SOLVED |
| ship08s | 778 | 2387 | OPTIMAL | 1.9200982105e+06 | 663 | 0.05 | 0.01 | SOLVED |
| 25fv47 | 821 | 1571 | OPTIMAL | 5.5018458883e+03 | 3548 | 0.42 | 0.16 | SOLVED |
| czprob | 929 | 3523 | OPTIMAL | 2.1851966989e+06 | 1934 | 0.12 | 0.03 | SOLVED |
| scfxm3 | 990 | 1371 | OPTIMAL | 5.4901254550e+04 | 1471 | 0.09 | 0.03 | SOLVED |
| sctap2 | 1090 | 1880 | OPTIMAL | 1.7248071429e+03 | 1224 | 0.07 | 0.01 | SOLVED |
| woodw | 1098 | 8405 | OPTIMAL | 1.3044763331e+00 | 2272 | 0.32 | 0.07 | SOLVED |
| ship12s | 1151 | 2763 | OPTIMAL | 1.4892361344e+06 | 1219 | 0.07 | 0.02 | SOLVED |
| pilot | 1441 | 3652 | OPTIMAL | -5.5748972927e+02 | 17807 | 5.98 | 0.77 | SOLVED ¹ |
| degen3 | 1503 | 1818 | OPTIMAL | -9.8729400000e+02 | 5128 | 0.90 | 0.14 | SOLVED |
| pilot87 | 2030 | 4883 | OPTIMAL | 3.0171034744e+02 | 24852 | 14.06 | 2.82 | SOLVED ¹ |
| d2q06c | 2171 | 5167 | OPTIMAL | 1.2278421081e+05 | 23347 | 7.55 | 0.76 | SOLVED |
| 80bau3b | 2262 | 9799 | OPTIMAL | 9.8722419241e+05 | 7951 | 1.13 | 0.12 | SOLVED ¹ |
| bnl2 | 2324 | 3489 | OPTIMAL | 1.8112365404e+03 | 5305 | 1.13 | 0.05 | SOLVED |
| greenbea | 2392 | 5405 | OPTIMAL | -7.2555248130e+07 | 16741 | 3.64 | 0.27 | SOLVED ¹ |
| fit2p | 3000 | 13525 | OPTIMAL | 6.8464293294e+04 | 15348 | 20.06 | 0.78 | SOLVED |
| maros-r7 | 3136 | 9408 | OPTIMAL | 1.4971851665e+06 | 4894 | 2.17 | 0.56 | SOLVED |
| dfl001 | 6071 | 12230 | ITERATION_LIMIT | — | 920832 | 900.05 | 4.74 | **not solved** |

¹ Our certified optimum disagrees with the value in the Netlib readme, and an independent HiGHS
solve of the same file agrees with ours to 10–11 significant digits (e.g. 80bau3b: readme
987232.16072; ours and HiGHS 987224.19241).

² The readme value (−18.751929066) leaves out the model's objective constant (+7.113, given as an
RHS entry on the objective row). Without the constant, our optimum is exactly −18.751929066.

**Speed, honestly:** gpuopt is currently **3–25× slower than HiGHS** on the larger models, and it
does not solve dfl001 within 15 minutes. That is expected at this stage. HiGHS has years of
performance engineering (dual simplex, presolve, hypersparsity, Forrest-Tomlin updates) that are on
our roadmap. Correctness came first.

---

## 15. Issues found and fixed during development

Each of these was caught by the validation tooling, which is the point of building it first.

| Symptom | Root cause | Fix |
|---|---|---|
| Random-LP cross-check: 2 "disagreements" with SciPy/HiGHS | HiGHS **presolve** labelled two unbounded models as infeasible. Checked by hand: feasible and unbounded. | The script re-solves with presolve off when they disagree; now 100% agreement. |
| `pilot87`, `maros-r7`: LU update test failed (residual drifted 1e-18 → 1e-8 over 100 updates) | The random starting basis was numerically singular (condition ≥ 10¹⁶); the PFI chain carried its error forward. Fresh factorizations stayed exact. | Residual checked every 10 updates, refactorize on drift, as the blueprint prescribes. Condition estimate added to the report. |
| LU benchmark claimed only 3× over dense LU | The "dense" reference skipped zero multipliers, which made it partly sparse. | Made it a true O(m³) textbook LU. The honest speed-up is 43–300×. |
| LU benchmark counted refactorizations wrongly | Scheduled (eta-size) refactorizations were reported as residual-triggered ones. | Counted separately. |
| **`pilot`: objective off by 1e-4, checker FAIL** | Tiny objective coefficients (2e-3 to 3e-2) made the fixed dual tolerance loose, and column scaling magnified reduced costs up to 64× when mapped back. | **Cost scaling** plus an optimality check in original units. Now matches HiGHS to 3e-11. |
| Netlib scoring counted `MISMATCH` lines as matches | The substring `"MATCH"` is contained in `"MISMATCH"`. | Match on `"=> MATCH"`; results re-run. |
| `e226` differed from the published value by 7.113 | The readme value excludes the objective constant. | The script compares both ways and labels the row. |
| `80bau3b`, `greenbea`, `pilot`, `pilot87` differed from the readme | The readme values are off; confirmed by an independent HiGHS solve. | The script re-solves with HiGHS on any certified disagreement and reports it. |
| Python `urllib` downloads took ~3 minutes per file | Network-specific slowness. | The fetch script uses `curl` (built into Windows 10+) and parallel downloads: 49 files in seconds. |

---

## 16. Scripts reference

| Script | Purpose | Example |
|---|---|---|
| `fetch_netlib.py` | Download Netlib models (COIN-OR Data-Netlib mirror) and decompress them into `data/netlib/`. Sets: `small`, `medium`, `large`, `all`. | `python scripts/fetch_netlib.py --set all` |
| `solve_netlib.py` | Solve every model, score it against `optimal_values.csv`, and optionally time HiGHS; writes a CSV. | `python scripts/solve_netlib.py --highs --csv netlib_results.csv` |
| `bench_netlib.py` | Layer 2 benchmark over every model (fill, condition estimate, residuals, updates). | `python scripts/bench_netlib.py` |
| `highs_reference.py` | Independent HiGHS solve through SciPy, with its own MPS parser. | `python scripts/highs_reference.py data/netlib/pilot.mps` |
| `crosscheck_scipy.py` | Random models written as MPS, solved by gpuopt and by SciPy/HiGHS; compares status and objective. | `python scripts/crosscheck_scipy.py --count 1000` |

**Benchmark data.** Netlib files are downloaded, not committed (see `.gitignore`), except `afiro.mps`,
which the tests use. `data/netlib/optimal_values.csv` holds the published optimal values of all 95
Netlib models, taken from the [Netlib readme](https://www.netlib.org/lp/data/readme).

---

## 17. Known limitations

* **LP only.** Integer markers are read, but MILP is solved as its LP relaxation (the output says so).
  QP sections are rejected with an error.
* **No GPU code yet.** The GPU layer (PDLP) is planned; there are no GPU claims.
* **Primal simplex only.** No dual simplex, presolve or interior point yet.
* **Performance.** PFI updates, no hypersparsity and full pricing on every iteration make large
  models slow: 3–25× slower than HiGHS, and dfl001 (6,071 rows) is not solved within 15 minutes.
  On dfl001 the eta file forces a refactorization every ~23 iterations.
* **Coverage.** 49 of the 98 Netlib models have been tested so far.
* **Input.** No gzip support (decompress first); fixed-format names are limited to the classic 8 characters.

---

## 18. Roadmap

Following the project blueprint's build order:

| Next | Why |
|---|---|
| **Dual simplex** | Warm-starting after bound changes is what branch-and-bound needs; it often beats primal on degenerate models. |
| **Forrest-Tomlin update + hypersparse FTRAN/BTRAN** | Removes the refactorization storm seen on dfl001, and gives 5–10× on large sparse models. |
| **Presolve / postsolve** | Typically removes 30–70% of a model before solving. |
| Full Netlib (98) + Dolan–Moré performance profile vs HiGHS | The standard way to present solver performance. |
| Exact rational certificate on one Netlib instance | A bit-exact proof of optimality ("auditable to the last bit"). |
| Interior point → convex QP | Layers 4 and 6. |
| GPU PDLP (restarts, primal weights, Ruiz/Chambolle–Pock preconditioning) | Layer 5: large LPs where the matrix-vector workload maps onto a GPU. |
| MILP branch-and-cut | Layer 7, on top of the dual simplex. |
| C ABI + Python bindings | Layer 8. |
| Refinery blending MILP case study | Direct relevance to MRPL. |

---

## 19. Publishing this repository

Everything that should be versioned is already covered by `.gitignore` (build outputs, downloaded
benchmark files and result CSVs are excluded).

```bash
git init
git add .
git commit -m "gpuopt: MPS reader, sparse LU, revised simplex, validation tooling"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

After cloning, anyone can reproduce every number in this README with:

```bash
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
python scripts/fetch_netlib.py --set all
python scripts/bench_netlib.py
python scripts/solve_netlib.py --highs
```

*No licence has been chosen yet; add a `LICENSE` file before publishing if others may reuse the code.*

---

## 20. References

* I. Maros, *Computational Techniques of the Simplex Method*, Kluwer, 2003. Tolerances, scaling, degeneracy.
* A. Koberstein, *The Dual Simplex Method, Techniques for a Fast and Stable Implementation*, PhD thesis, Paderborn, 2005.
* J. J. Forrest, D. Goldfarb, "Steepest-edge simplex algorithms for linear programming", *Math. Programming* 57, 1992. Devex pricing.
* P. M. J. Harris, "Pivot selection methods of the Devex LP code", *Math. Programming* 5, 1973. Harris ratio test.
* H. M. Markowitz, "The elimination form of the inverse and its application to linear programming", *Management Science* 3, 1957.
* U. H. Suhl, L. M. Suhl, "Computing sparse LU factorizations for large-scale linear programming bases", *ORSA J. Computing* 2, 1990.
* R. G. Bland, "New finite pivoting rules for the simplex method", *Math. of Operations Research* 2, 1977.
* E. M. L. Beale, "Cycling in the dual simplex algorithm", *Naval Research Logistics Quarterly* 2, 1955.
* Q. Huangfu, J. A. J. Hall, "Parallelizing the dual revised simplex method", *Math. Programming Computation* 10, 2018.
* T. Koch, "The final NETLIB-LP results", *Operations Research Letters* 32, 2004. Exact optimal values of the Netlib set.
* Netlib LP test set: <https://www.netlib.org/lp/data/>; COIN-OR Data-Netlib mirror: <https://github.com/coin-or-tools/Data-Netlib>.
* HiGHS (used only as an external comparison baseline, through SciPy): <https://highs.dev>.
