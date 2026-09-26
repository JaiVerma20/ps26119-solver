# Final integration report — ps26119 + gpuopt

Date: 2026-09-26. Integration lead: Claude (for Jai), under the brief "correctness >
performance > feature count > code size". Machine for every number below: MacBook Air M4
(10 cores, 16 GB), CPU only. Benchmark CSVs carry the git hash of the binary that produced
them; the final suite ran from a frozen binary built at **`d824f82`**.

## 1. Executive summary

Two independently written solvers were audited claim by claim, then merged into one
repository with one model contract, one result contract, one verification gate and four LP
engines plus a MILP branch-and-bound. Both histories are preserved (the teammate's commit
`c192dd0` is a parent in `main`'s history). Measured at `d824f82`:

- **Netlib, all 93 LPs** (60 s each, read by our own MPS reader, every answer verified by an
  independent checker and compared with HiGHS): unified `auto` **93/93**; simplex alone
  92/93; r²HPDHG alone 85/93 (identical to before the merge — no regression).
- **Small MIPLIB 3**: 10/14 proven optimal and verified (unchanged count), total time on those
  10 down from 383.9 s to 42.5 s after the node LPs moved to the sparse simplex.
- **Bugs found and fixed: 12** — 9 in the teammate's code (reader NaN/non-finite acceptance,
  macOS build, case-sensitive MARKER, silent duplicate RHS, false Optimal above the requested
  tolerance, time limit reported as iteration limit, checker blind to NaN/wrong sizes,
  explicit-zero division in scaling, a test depending on an untracked file) and 3 in mine or
  introduced by the merge (heap over-read in batch, silently skipped Python tests, zero-column
  models reported NumericalError). Each has a regression test (for the false-Optimal,
  scaling and over-read cases it was shown to fail without the fix). Three of them were
  paths to a false "Optimal" inside an engine; the new in-process gate now catches that class.
- **Tests: 165 CTest cases** (C++ unit, differential, CLI, Python, random-MPS cross-checks),
  all passing; ASan/UBSan and TSan clean on the C++ suite.
- **GPU: NOT VERIFIED.** The CUDA backend has still never been compiled by nvcc or run; no GPU
  claim is made. The procedure (`scripts/gpu_check.sh`, WSL2 on the teammate's laptop) is ready.

## 2. What my repository contained (before: tag `pre-teammate-integration` = `20033bf`)

`Model`/`Solution` contracts with fingerprints; `.lpm` bridge (highspy, tooling only);
independent verifier `tools/verify.py`; dense double-double oracle; CPU PDLP-style PDHG and
r²HPDHG (fp64 + mixed precision, Ruiz + Pock–Chambolle + adaptive geometric scaling, Lanczos
‖A‖₂, PID primal weight, ray certificates, warm start); batched scenarios (SpMM); certified
Neumaier–Shcherbina bound with rigorous implied bounds; basic presolve/postsolve;
deterministic thread pool; CUDA backend (written, never run); prototype MILP branch-and-bound
(dense oracle node LPs); C API, Python binding, CLI; hash-named benchmark CSVs and a generated
evidence pack. No MPS reader, no sparse simplex.

## 3. What the teammate's repository contained (`shivanshu24-code/gpu_optimization@c192dd0`)

One commit (Windows/MinGW): strict MPS reader (free/fixed, RANGES, BOUNDS, markers, OBJSENSE,
line-numbered errors); Markowitz/threshold sparse LU with FTRAN/BTRAN, PFI updates and
singular-basis repair; bounded revised primal simplex (composite phase 1, Devex, Harris,
bound flips, perturbation, Bland fallback, fresh-factor re-checks, unscaled optimality check);
dense two-phase Bland tableau oracle; C++ solution checker; CLI with `--info/--print-model/
--lu-bench`; 5 test programs; scripts; an excellent README. No GPU, MILP, QP or API.

## 4. What was independently verified (details: `docs/audit/INITIAL_AUDIT.md`)

| Teammate claim | Re-run | Verdict |
|---|---|---|
| 48/49 Netlib solved and certified | 92/93 on the full set, each checked by our `verify.py` with highspy's reader, ≤ 3.5e-10 from HiGHS | VERIFIED (stronger) |
| 7,400 random LPs vs oracle, 0 mismatches | reproduced | VERIFIED |
| 1,000 vs SciPy (oracle) / 300 (simplex) | 1000/1000 — with the simplex (script had no oracle option) | PARTIALLY VERIFIED |
| Sparse LU 49/49, residual ≤ 1e-16, fill ≤ 1.8× | 93/93, backward error ≤ 1.1e-16, fill up to 3.0× | PARTIALLY VERIFIED |
| 43× / ~300× faster than dense LU | 34× / 142× on M4 | PARTIALLY VERIFIED |
| Reader parses all Netlib | bit-identical Model to highspy on 93 Netlib + 14 MIPLIB 3 | VERIFIED |
| built from scratch | no link dependencies | VERIFIED |
| 5/5 tests pass | 4/5 on a fresh clone (untracked data file) | NOT REPRODUCED (fixed) |

My own system was audited the same way: ASan found a heap over-read in batch solving; a fresh
configure silently skipped all Python/CLI tests; TSan and a 1/2/4/8/10/all-thread sweep were
clean and bit-identical.

## 5. Where the teammate's code was better, and why

- **MPS reader** — mine did not exist; his is complete, strict (rejects `4x`, which highspy
  accepts as 4) and bit-identical to highspy on every benchmark file.
- **Simplex + sparse LU** — the strongest Netlib engine in either codebase (92 vs 85 of 93),
  gives vertices and exact duals, and made the MILP prototype ~9× faster as its node solver.
- **Simplex defaults** (geometric scaling, Devex) — confirmed by A/B on full Netlib (92 vs 91
  without scaling, 92 vs 90 with Dantzig pricing; DECISIONS #30).
- **CLI inspection tools** (`info`, `print`, `lu-bench`) and hand-made example models.

## 6. Where my code was better, and why

- **Large models**: r²HPDHG solves the 429k-row refinery year and 1e6-row random LPs where the
  simplex is ~37× (refinery T=365: 22.8 s vs 0.62 s) to ~1400× (rand-10000: 545 s vs 0.39 s) slower in the same final run (EVIDENCE §2, §2c).
- **Contracts and provenance**: richer `Solution` (fingerprint, certified bound, residuals,
  engine, precision), hash-named CSVs from the binary, machine info, generated evidence.
- **Independent verification**: `verify.py` uses a different reader; the teammate's checker
  shares the solver's reader (both kept, see §7).
- **Everything outside LP-by-simplex**: GPU backend, mixed precision, warm start, batch,
  certified bound, presolve, MILP, C API, Python, CI.

## 7. What was kept from each

| Kept | From |
|---|---|
| `Model`, `Solution`, `Options`, tolerances, C API, Python, evidence system, PDHG family, GPU backend, presolve, certified bound, MILP tree, dd oracle, `verify.py` | mine |
| MPS reader (now the only MPS path in the solver), sparse LU, primal simplex + scaling, tableau oracle (test-only second oracle), solution checker (now the in-process gate), `info`/`print`/`lu-bench`, examples, published Netlib optima, LU bench and random-MPS cross-check scripts | teammate |

## 8. What was discarded, and why

`LpProblem`/`SolveResult` (replaced by `Model`/`Solution` — same content, one contract);
the mini test harness (GoogleTest); their CMake/.gitignore (merged); a duplicate afiro.mps
(same fingerprint); `fetch_netlib.py`, `highs_reference.py`, `solve_netlib.py` (superseded by
`tools/fetch_netlib.py`, `tools/highs_ref.py`, `bench/netlib_full.py`); their CLI binary
(features ported). All remain in history (`c192dd0`).

## 9. What was rewritten, and why

Nothing algorithmic. Changes to teammate code are (a) the mechanical port to `Model`/`Solution`
and namespaces, (b) audit bug fixes, each with a regression test. `git blame` still attributes
417 of 421 lines of the sparse LU and about 90% of the simplex directory to him.

## 10. Adapted rather than copied

Tests ported to GoogleTest with assertions unchanged; `print_model`/`print_statistics` moved
into `apps/cli/model_report.cpp`; scripts retargeted to `ps26119`; the checker's defaults now
come from `tolerances.h`.

## 11–15. Architecture, layout, contracts, engines

See `docs/ARCHITECTURE.md` (pipeline diagram, engines table, verification layers) and
`CLAUDE.md` §10 (layout). Canonical model: `include/ps26119/model.h` (plus the details agreed
at the merge in CLAUDE.md §6: crossed bounds are Infeasible, negative-UP rule, nan rejected).
Canonical result: `include/ps26119/solution.h` (with `check`, the gate result). Engines:
`simplex`, `r2hpdhg`, `pdlp`, `oracle`, `auto` (size rule, DECISIONS #29), MILP
branch-and-bound (sparse simplex nodes, certified-bound pruning, DECISIONS #28).

## 16. GPU status

NOT VERIFIED. Code unchanged by the merge; `scripts/gpu_check.sh` now also runs
compute-sanitizer (memcheck, racecheck) and the simplex on the CPU side; WSL2 instructions in
`docs/GPU_VERIFICATION.md` (nvcc does not support the teammate's MinGW toolchain).

## 17. MILP status

EXPERIMENTAL/prototype, verified answers only. `bench/results/miplib3-macbook-air-m4-d824f82.csv`:
optimal + verified p0033, flugpl, egout, enigma, lseu, mod008, stein27, rgn, misc03, p0201
(10/14); time limit (300 s) pk1, gt2, bell5 (no incumbent), bell3a (incumbent equals the HiGHS
optimum, gap 5.9e-4 not closed). On the 10 solved: 42.5 s vs 383.9 s before; the four smallest
models are slower than with the dense oracle (p0033 0.9 vs 0.2 s).

## 18. Netlib results (`*-d824f82.csv`)

| engine | solved + verified / 93 | not solved (60 s) |
|---|---|---|
| auto | 93 | – |
| simplex | 92 | dfl001 |
| r2hpdhg | 85 | d2q06c, fit2d, greenbea, greenbeb, pilot, pilot.ja, pilot.we, pilot87 |

Small Netlib: 60/60 runs verified (10 models × oracle, simplex, PDLP ×2, r²HPDHG ×2).

## 19–21. MIPLIB, refinery, CPU results

MIPLIB: §17. Refinery and CPU scaling (`scale-macbook-air-m4-d824f82.csv`, r²HPDHG, to 1e-8,
every answer equal to the known optimum and verified where the verifier is run):

| instance | rows | r²HPDHG 1 thread fp64 | 10 threads mixed | simplex (same run) |
|---|---|---|---|---|
| refinery T=12 | 588 | 0.013 s | 0.018 s | 0.040 s |
| refinery T=365 | 17,885 | 0.62 s | 0.52 s | 22.8 s |
| refinery T=8760 (hourly year) | 429,240 | 19.2 s | 14.7 s | not run |
| rand-10000 | 10,000 | 0.39 s | 0.35 s | 545 s |
| rand-100000 | 100,000 | 8.4 s | 4.5 s | not run |
| rand-1000000 | 1,000,000 | **TimeLimit 600 s** (17,280 of the 17,600 iterations it needs) | 425 s | not run |

Honest caveat: this run's wall times are ~2× slower than the pre-merge run `8fd5170` for
identical iteration counts (a fanless laptop after 1.5 h of benchmarking); the rand-1e6
1-thread fp64 limit is that effect, and the row is kept. HiGHS reference on the same
instances (`scale-macbook-air-m4-62a13f2.csv`): TimeLimit (> 600 s) on the hourly year.

## 21b. SIH requirement status

`docs/SIH_STATUS.md` (DONE / PARTIAL / EXPERIMENTAL / NOT STARTED per requirement, with test,
benchmark and evidence columns).

## 22. GPU results

None measured.

## 23. Known failures

Netlib: dfl001 (simplex, 60 s), 8 models for r²HPDHG (above). MIPLIB: pk1, gt2, bell5, bell3a
(300 s). Differential tests: 0 mismatches in 1,800 random LPs × 5 engine configurations.

## 24–25. Known numerical and performance limitations

`docs/LIMITATIONS.md`.

## 26. Remaining roadmap (priority order)

1. GPU validation on NVIDIA hardware (M2) — the single biggest open claim.
2. Simplex basis I/O → warm-started MILP nodes and re-solves; then a dual simplex.
3. Crossover from r²HPDHG to a vertex (uses the integrated LU + simplex).
4. Forrest–Tomlin updates, hypersparse FTRAN/BTRAN, partial pricing (simplex at 10⁵ rows).
5. MILP cuts (Gomory, MIR, cover), pseudocost branching, diving heuristics.
6. Stronger presolve (doubletons, dominated columns); Mittelmann LP runs; QP (IPM or PDHG).

## 27. Git / provenance

- Pre-merge checkpoint: tag `pre-teammate-integration` (`20033bf`) + bundle
  `~/Desktop/SIH26119-backups/ps26119-pre-integration-20033bf.bundle`.
- Audit: `ce09c27`. Import: `56a559d` (merge, second parent `c192dd0`). Renames: `670e9d9`.
  Adaptation: `7fe5eaf`. Final benchmark binary: `d824f82`.
- Branch `integration/gpuopt` merged into `main`; no history rewritten, no force pushes.
- Provenance checks: `scripts/check_no_solver_linked.sh` (binaries, shared library, Python
  package, CMake) — PASS. See `docs/PROVENANCE.md`.

## 28. Contributor / authorship mapping

`docs/PROVENANCE.md` §2 (git-blame table). Commits authored by Jai carry `Co-Authored-By:
Claude` where AI-assisted; the teammate's original commit carries his name, email and date.

## 29. Recommended work split

| Jai | Shivanshu |
|---|---|
| GPU validation support (fix nvcc issues from his logs), crossover, MILP cuts/heuristics, warm-started PDHG in MILP/SLP, evidence + PPT | run `scripts/gpu_check.sh` on WSL2, basis I/O + dual simplex, Forrest–Tomlin/hypersparsity, presolve extensions (his reader/simplex domain) |

## 30. Exact next steps

1. `gh auth login`, then `scripts/setup_github.sh ps26119-solver shivanshu24-code` (creates the
   private repo, pushes `main` + tags, invites the teammate, tries to protect `main`).
2. Teammate: clone the canonical repo, `git switch -c feature/shivanshu/gpu-validation`,
   run `QUICK=1 scripts/gpu_check.sh` under WSL2, open a PR with the logs and CSVs.
3. Jai: fix whatever nvcc reports; merge the GPU CSVs; regenerate `docs/EVIDENCE.md`.
4. Start basis I/O (Shivanshu) and crossover (Jai) on separate feature branches.
