# Initial integration audit (Phases 0–5)

Date: 2026-09-26. Auditor: integration lead (Claude, for Jai). Machine: MacBook Air M4
(10 cores, 16 GB, macOS 15.5 SDK, Apple clang 17). **No NVIDIA GPU was available for this audit.**

Two codebases:

| | ps26119 (Jai) | gpuopt (Shivanshu Vats) |
|---|---|---|
| Source | this repository, `main` @ `20033bf` (47 commits) | github.com/shivanshu24-code/gpu_optimization @ `c192dd0` (1 commit, 2026-09-25) |
| Size | ~9.6k lines C++/CUDA + ~3.3k lines Python tooling | ~5.1k lines C++ + ~0.6k lines Python |
| Focus | first-order LP (PDHG, r²HPDHG), CPU + CUDA backend, presolve, warm start, batch, certified bound, MILP prototype, C API / Python | MPS reader, sparse LU, bounded revised primal simplex, dense oracle, solution checker |

Nothing was taken on trust from either README. Every number below was re-run on this
machine; the raw outputs are next to this file.

## Phase 0 — protection

- Tag `pre-teammate-integration` → `20033bf` (annotated).
- Full-history bundle `~/Desktop/SIH26119-backups/ps26119-pre-integration-20033bf.bundle` (`git bundle verify`: OK).
- Working tree was clean; no remotes, no stashes. Ignored/generated: `build/`, `bench/generated/`
  (27 M lines of generated LPs, reproducible from `bench/generate_*.py`), `data/netlib/`,
  `data/miplib3/` (downloaded), `__pycache__/`, `.cache/`.
- Teammate repo cloned separately to `~/Desktop/SIH26119-teammate/gpu_optimization`
  (single branch `main`, no tags, one author).

## Phase 1–3 — teammate repository

### Build

| Check | Result |
|---|---|
| Builds on macOS / Apple clang 17 as published | **BROKEN** — `std::from_chars(const char*, const char*, double&)` does not exist in Apple's libc++ (deleted overload), at any deployment target. Only `src/io/mps_reader.cpp:93` is affected. |
| Builds with a 15-line portable replacement of `parse_number` ([auditport.patch](auditport.patch), audit clone only) | yes, `-Wall -Wextra -Wpedantic -Wshadow`: 0 warnings |
| Third-party dependencies / linked libraries | none (CMake links nothing; checked `otool -L`) |
| Test suite on a fresh clone | 4/5 executables pass; `test_simplex` **fails** because it opens `data/netlib/adlittle.mps`, which `.gitignore` excludes. With the file present: 5/5. |

### Component audit (A–L of the brief)

Legend: A exists · B compiles · C linked into the executable · D exercised by tests · E tests genuinely test the claim · F mathematically correct · G scales beyond toys · H depends on an external solver · I independently verified · J quality · K limitations · L worth integrating.

**MPS reader** (`src/io/mps_reader.cpp`, 648 lines) — A yes · B after the portability fix · C yes · D 8 unit tests + every other test · E yes (hand models with hand-derived optima; free/fixed/RANGES/negative UP/OBJSENSE/MARKER) · F **VERIFIED**: the model it produces is *bit-identical* (our 64-bit fingerprint over every number) to the model highspy's reader produces on **all 93 Netlib LPs, all 14 small MIPLIB 3 models, our hand models and 7 of its 8 examples** · G dfl001 (6,071×12,230) parses in milliseconds · H no · I yes (vs highspy) · J production quality: line-numbered errors, refuses QP/SOS/SC/gzip instead of guessing, first RHS/RANGES/BOUNDS set only (warns) · K: see bugs R1–R4 · L **yes — this is the reader the contract (CLAUDE.md §2) reserved for the teammate**.

Malformed-input probe (files in the audit log):

| input | gpuopt reader | highspy |
|---|---|---|
| `4x` as a number | error with line number ✅ | silently reads 4 ❌ |
| lowercase `'marker'` / `'intorg'` | error (safe, but should be accepted) | ignores the marker |
| `nan` matrix coefficient | **accepted; simplex OPTIMAL; checker PASS** ❌ (R1) | silently drops the entry |
| `NaN` RHS | accepted, then `validate()` rejects → NUMERICAL_ERROR | read error |
| duplicate RHS for one row | keeps the **last** value silently (R3) | keeps the first |
| `1e400` | error with the original `from_chars` (out of range) | +inf |
| negative `UP` with default lower | lower → −∞ with a warning (classic MPS rule) | lower stays 0 (column infeasible) |
| duplicate (row, col) entries | summed, with a warning | summed |

**Canonical model** (`LpProblem`) — same information as our `Model` (CSC A, row/col bounds, sense, offset, integrality, names), plus `objective_name`. Differences: `ObjSense` enum vs `int sense`; CSC rows sorted and unique by construction; `validate()` checks sizes and NaN in bounds/objective but **not NaN/Inf in matrix values** (R1). Offset semantics identical to ours (`cᵀx + c₀`, sense chooses min/max). Verdict: **replace by our `Model`** (it is already consumed by 3 engines, presolve, MILP, C API, Python, fingerprints); keep their `SparseMatrixCSC` + `build_csc` as the linear-algebra matrix type (it is a matrix, not a model).

**Sparse LU** (`src/linalg/sparse_lu.cpp`, 420 lines) — right-looking Markowitz with column threshold τ = 0.1, Suhl-style count buckets and early search termination, U stored by row and by column, product-form (PFI) updates, singular bases completed with slack columns (reported through `replaced_columns()`). I checked by hand that the singular completion is exact: `L⁻¹ e_r = e_r` for every uncovered row r, so the completed factors are those of B with the dependent columns replaced. FTRAN/BTRAN and the PFI eta algebra are correct. Not present: hypersparse solves, Forrest–Tomlin, symbolic/numeric split, rook pivoting. Measured: **VERIFIED** — `--lu-bench` on random simplex-style bases of **all 93 Netlib models: 93/93 PASS**, worst normwise backward error ‖Bx−b‖∞/(‖B‖∞‖x‖∞+‖b‖∞) = 1.1e-16, PFI chains of 100 updates agree with fresh factorizations. Fill `nnz(L+U)/nnz(B)` 1.00–3.02 (README: ≤ 1.8 — maros-r7 gives 3.02 here; random bases differ between libstdc++ and libc++). Worth integrating: **yes** (needed for crossover, dual simplex and MILP node LPs).

**Revised primal simplex** (`src/simplex/primal_simplex.cpp`, 598 lines) — composite phase 1, Devex (reference framework, reset at 1e6), Harris two-pass ratio test, bound flips, deterministic cost perturbation removed before the end, Bland fallback after `100+2m` degenerate steps, refactorization every 100 updates or on eta growth, every OPTIMAL/INFEASIBLE/UNBOUNDED re-checked on a fresh factorization, optimality confirmed in unscaled units with up to 4 tolerance tightenings. **VERIFIED on Netlib**: see the claim table. Prototype-level aspects: reduced costs and the Devex pivot row are recomputed from scratch every iteration (O(nnz) per iteration, no partial pricing, no hypersparsity); no basis in/out (no warm start); time limit reported as ITERATION_LIMIT; no infeasibility/unboundedness certificate returned. Bug S1 (below). Worth integrating: **yes** — it is the strongest Netlib engine in either repository.

**Dense oracle** (`src/oracle/dense_oracle.cpp`) — standard-form conversion + two-phase dense tableau + Bland, double precision. Genuinely independent of the simplex (no shared numerical code except `LpProblem` and `SparseMatrixCSC::multiply`). Ours is a bounded dense simplex in double-double. **Keep both**: two independently written oracles agreeing is stronger evidence than either alone; ours stays the canonical oracle (higher precision, used by `--algorithm oracle`), theirs becomes a test-only cross-check.

**Solution checker** (`src/validation/solution_checker.cpp`) — primal feasibility (relative to 1+|bound|), dual sign feasibility of y and d = c − Aᵀy (d recomputed, relative to max(1,‖c‖∞)), Lagrangian dual objective, relative gap, complementarity reported. Same mathematics as our `tools/verify.py`. Weakness: `std::max(v, NaN) == v`, so NaN in x, y or Ax is invisible (R1/C1). Independence: it shares the reader with the solver, so a reader bug cannot be caught by it; `verify.py` uses highspy's reader. **Keep both**: the C++ checker becomes the in-process gate on every `Optimal`, `verify.py` stays the external, independent verifier.

**Scaling** (`src/simplex/scaling.cpp`) — 6 geometric-mean passes, rows rounded to powers of two, then column equilibration to max ≈ 1 (power of two): exact scaling and unscaling. Different purpose from our PDHG scaling (Ruiz + Pock–Chambolle + adaptive geometric mean, tuned for first-order methods). Decision by A/B test in Phase 8.

**Tests** — 5 executables on a 99-line mini harness. The differential test (simplex vs oracle, checker-certified) is real, but the LPs are tiny (≤ 8×10 and ≤ 40×60) with integer coefficients in [−5, 5], and a large fraction are infeasible/unbounded (e.g. medium set: 80 optimal of 400).

**Benchmark tooling** — `solve_netlib.py` counts SOLVED only when OPTIMAL + checker PASS + published/HiGHS match (sound); results are written to an untracked CSV without commit hash or machine info (provenance weaker than ours). HiGHS/SciPy used only in scripts (sound).

**GPU / MILP / QP / IPM / API** — NOT IMPLEMENTED (the README says so honestly).

### Phase 2 — every README claim

| Claim (README) | Re-run on M4 | State |
|---|---|---|
| "48 of 49 Netlib solved and independently certified" | Full 93-model set, 60 s/model, answers checked by **our** `verify.py` (highspy reader): **92/93** OPTIMAL + verify PASS + |obj−HiGHS|/(1+|HiGHS|) ≤ 3.5e-10; dfl001 ITERATION(time)_LIMIT. Total 54 s for the 92. [CSV](teammate-simplex-netlib-full-c192dd0+auditport-macbook-air-m4.csv) | **VERIFIED** (stronger than claimed) |
| "7,400/7,400 random LPs vs dense oracle, 0 mismatches" | 5000 + 400 + 1000 + 1000, 0 mismatches | **VERIFIED** (easy instances; see Tests) |
| "5,000 random LPs certified (oracle)" | reproduced | **VERIFIED** |
| "1,000/1,000 vs SciPy (dense oracle) and 300/300 (simplex)" | `crosscheck_scipy.py --count 1000`: 1000/1000 — but the script runs the **simplex** (default method), not the oracle; there is no `--method` option | **PARTIALLY VERIFIED** (engine label wrong) |
| "Sparse LU 49/49 PASS, residuals ≤ 1e-16, fill ≤ 1.8×" | 93/93 PASS, worst backward error 1.1e-16, fill up to 3.02× | **PARTIALLY VERIFIED** (fill bound not reproduced) |
| "Sparse LU 43× faster than dense on 25fv47, ~300× on pilot87" | 34× and 142× on M4 | **PARTIALLY VERIFIED** (order of magnitude; machine dependent) |
| "Parses all 49 Netlib models; dfl001 in 24 ms" | all 93 parse, bit-identical to highspy | **VERIFIED** |
| "built from scratch; no solver linked" | no link dependencies; SciPy only in scripts | **VERIFIED** |
| "100% tests passed (5/5) in ~2 s" | 4/5 on a fresh clone (missing adlittle.mps) | **CLAIM NOT REPRODUCED** on a fresh clone |
| "No GPU code yet" | correct | NOT APPLICABLE |

### Bugs found in the teammate code

| id | severity | bug | fix plan |
|---|---|---|---|
| R0 | build | Apple libc++ has no floating `from_chars` → reader does not compile on macOS | portable parser (strtod on a validated character set, `inf`/`infinity` accepted) |
| R1 | **wrong answer** | a `nan` token is accepted as a coefficient; `LpProblem::validate` does not check matrix values; the checker ignores NaN → a garbage model is "proven optimal" | reject `nan` tokens (line-numbered error); our `Model::validate` rejects non-finite A; checker fails on non-finite data |
| R2 | minor | `'MARKER'` matched case-sensitively | match case-insensitively |
| R3 | ambiguity | duplicate RHS/RANGES entry for one row silently overwrites | warn (keep the reader's behaviour visible); documented in the §6 contract |
| S1 | **false Optimal** | after 4 tolerance tightenings the simplex returns OPTIMAL even if the unscaled violations still exceed the tolerance | return NumericalError unless the unscaled check passes; plus the in-process checker gate |
| S2 | reporting | time limit reported as ITERATION_LIMIT | map to TimeLimit |
| T1 | test | `test_simplex` depends on an untracked file | use committed data |

## Phase 4 — my repository (audited the same way)

| Check | Result |
|---|---|
| Clean Release build (fresh dir) | 0 warnings |
| Unit tests | 100/100 pass |
| **Python/CLI tests on a fresh configure** | **silently skipped**: CMake picked Python 3.13 (no highspy), so `tools.python`, `cli.contract`, `python.binding`, `bench.generators` disappeared without failing. With `-DPython3_EXECUTABLE=<3.11>`: 104/104. → make this a configure error in CI |
| AddressSanitizer + UBSan | **1 bug**: `Batch.BadScenarioFailsAloneAndMaxSenseWorks` — heap buffer over-read in `solve_batch` (`src/pdhg/batch.cpp:175`): a scenario rejected for wrong vector sizes is still read over all n columns while building the scaled vectors. Result discarded, but undefined behaviour. |
| ThreadSanitizer | clean, 100/100 (incl. `PdhgThreads.BitIdenticalForAnyThreadCount`) |
| Thread sweep 1/2/4/8/10/all × 3 repeats, refinery T=365 | 18/18 solution files bit-identical (1984 iterations each) |
| `Model::validate` | rejects non-finite c, A, offset; duplicates; bad CSC; crossed bounds ✅ |

Component states (my repository):

| Component | State | Evidence |
|---|---|---|
| Model / Solution contracts, fingerprint | VERIFIED | unit tests; fingerprints equal to highspy's reader (Python) on all data |
| `.lpm` bridge + Python reader | VERIFIED | tests; now superseded for MPS by the teammate reader |
| Independent verifier (`tools/verify.py`) | VERIFIED | caught real bugs (egout MILP, per-row violations) |
| Dense double-double oracle | VERIFIED | 15 tests, small Netlib 10/10 |
| PDLP-style PDHG, r²HPDHG (fp64, mixed) | VERIFIED on CPU | small Netlib 50/50, full Netlib 85/93 (60 s), refinery T=8760 verified |
| Ruiz / Pock–Chambolle / adaptive geometric scaling | VERIFIED (A/B CSVs) | DECISIONS #17 |
| Lanczos ‖A‖₂ | VERIFIED | fixed scrs8 divergence (DECISIONS #10) |
| Presolve / postsolve | PARTIAL | basic reductions; 85 vs 83 Netlib; slower on one refinery case |
| Infeasible / unbounded (PDHG) | PARTIAL | checked certificates; not yet run on Netlib infeasible set |
| Certified bound | VERIFIED (limited) | finite on 40/85 Netlib |
| Warm start, batch | VERIFIED (CPU) | CSVs; batch had the ASan bug above |
| Deterministic threading | VERIFIED | TSan + sweep above |
| CUDA backend | **NEEDS GPU** | never compiled by nvcc |
| MILP branch-and-bound | PARTIAL (prototype) | 10/14 small MIPLIB 3, dense dd node LPs, no warm start |
| C API, Python binding, CLI | VERIFIED | tests |
| Evidence generator / provenance | VERIFIED | hash-named CSVs, `--version` from binary |
| CI | PARTIAL | CPU only; no sanitizer job; Python tests skippable |

## Phase 5 — comparison matrix

| Component | Mine | Teammate | Stronger | Why (evidence) | Decision |
|---|---|---|---|---|---|
| MPS reader | none (highspy bridge in tools/) | complete, strict | **teammate** | bit-identical to highspy on 93+14+ models; stricter than highspy on malformed input | **adopt** (+ R0–R3 fixes); `.lpm` kept for generated models |
| Canonical model | `Model` | `LpProblem` | equal content; **mine** more integrated | used by 3 engines, presolve, MILP, C API, Python, fingerprints; stricter `validate` | **keep `Model`**; port teammate code onto it |
| Result contract | `Solution` (residuals, certified bound, provenance) | `SolveResult` | **mine** | superset (fingerprint, engine, precision, bound, timing) | **keep `Solution`** |
| Sparse matrix | CSR (float/double, SpMV/SpMM, threaded) | CSC + multiply | complementary | CSR serves PDHG kernels, CSC serves LU/simplex | keep both (LA layer) |
| Sparse LU, FTRAN/BTRAN | none | Markowitz + PFI | **teammate** | 93/93 bases, backward error 1e-16 | **adopt** |
| Scaling | Ruiz+PC+adaptive geo (PDHG) | geometric+equilibration, powers of 2 (simplex) | purpose-specific | A/B in Phase 8 | keep per engine; A/B decides simplex default |
| Presolve / postsolve | basic | none | mine | — | keep; test with simplex |
| Primal simplex | none | complete | **teammate** | 92/93 Netlib verified | **adopt** as `--algorithm simplex` |
| Dual simplex | none | none | — | — | planned |
| PDHG, r²HPDHG | complete, CPU+CUDA(unrun) | none | **mine** | 85/93 Netlib; 429k-row refinery in 6–10 s | keep |
| IPM, QP | none | none | — | — | planned |
| MILP / B&B | prototype (dense dd node LP) | none (LP relaxation only) | mine | 10/14 MIPLIB 3 | keep; evaluate sparse simplex as node solver |
| Cuts | none | none | — | — | planned |
| Warm start | PDHG (x, y, ω) | none | mine | CSVs | keep; simplex basis warm start later |
| Batch scenarios | CPU SpMM | none | mine | CSVs | keep (fix ASan bug) |
| Certified bound | Neumaier–Shcherbina + implied bounds | none | mine | 40/85 finite | keep; apply to simplex duals too |
| Infeasibility / unboundedness certificate | PDHG rays, checked | status only | mine | — | keep; simplex statuses gated by fresh-factor re-check only |
| Dense oracle | double-double bounded simplex | double Bland tableau | complementary | independent implementations | **keep both** (theirs test-only) |
| Independent verifier | `verify.py` (highspy reader) | C++ checker (same reader) | complementary | independence vs in-process | **keep both**: C++ checker = in-process gate, `verify.py` = external |
| C API / Python / CLI | yes / yes / yes | – / – / rich CLI (`--info`, `--print-model`, `--lu-bench`) | mine + their CLI features | — | keep mine; port `info`, `print-model`, `lu-bench` subcommands |
| CUDA | written, not run | none | mine (only one) | — | keep; validate on teammate GPU (no second implementation) |
| CPU threading | deterministic pool | none | mine | TSan + sweep | keep |
| Netlib | 85/93 (PDHG) | 92/93 (simplex) | **teammate on Netlib** | CSVs | both engines in one benchmark |
| MIPLIB | 10/14 | – | mine | CSV | re-run after integration |
| Refinery | T=8760 verified (PDHG) | – | mine | CSV | re-run incl. simplex |
| Benchmark provenance | hash-named CSVs, machine info, frozen binary | untracked CSV | **mine** | — | keep mine; teammate scripts folded in |
| CI | GitHub Actions CPU | none | mine | — | extend (sanitizers, required Python tests) |
| Documentation | evidence-driven | very thorough README | both | — | unify; teammate README kept in `docs/history/` |

## What happens next

Integration on branch `integration/gpuopt`:
1. Fix the two bugs found in my code (batch over-read; silently skipped Python tests).
2. Import `c192dd0` **with its history and authorship** (subtree merge into `import/gpuopt/`), move files
   into the unified layout in a pure-rename commit, then adapt them to `Model`/`Solution` in separate commits.
3. Fix R0–R3, S1, S2, T1 as separate, tested commits.
4. `Algorithm::Simplex` in the dispatcher; in-process checker gate on every `Optimal`; CLI reads `.mps` directly.
5. Differential tests: dd oracle vs tableau oracle vs simplex vs PDHG vs `verify.py`.
6. Benchmarks with one methodology; MILP node-solver evaluation; docs; GitHub migration.
