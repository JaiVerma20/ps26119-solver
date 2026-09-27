# Contributing (team guide)

Read `CLAUDE.md` (rules), `docs/ARCHITECTURE.md` (design) and `docs/DEVELOPMENT.md` (build).
There is **one** repository. Nobody develops in a separate repository any more; the former
`shivanshu24-code/gpu_optimization` was merged here with its history (see
`docs/FINAL_INTEGRATION_REPORT.md`).

## Ownership (who reviews what)

| Area | Primary | Notes |
|---|---|---|
| `src/io/mps_reader.*`, `src/io/mps_parser.h` | Shivanshu | the MPS reader; contract `include/ps26119/model.h` (change only together) |
| `src/la/sparse_lu.*`, `src/simplex/`, `src/oracle/dense_tableau.*`, `src/core/solution_checker.*` | Shivanshu | simplex family; dual simplex, Forrest–Tomlin, hypersparsity go here |
| `src/pdhg/`, `src/gpu/`, `src/core/{solve,presolve,safe_bound,implied_bounds,c_api}.*`, `src/mip/`, `python/`, `bench/`, `tools/verify.py` | Jai | first-order engines, GPU backend, dispatcher, MILP, evidence |
| GPU runs (`scripts/gpu_check.sh`) | whoever has the hardware | commit the CSVs + logs it writes |

Either person may change any file; the primary owner reviews.

## Workflow

```
main  (protected: pull request + green CI; nobody pushes to it directly)
  └── feature/<person>/<topic>     e.g. feature/shivanshu/dual-simplex
                                         feature/jai/cuda-validation
```

1. `git switch main && git pull`, then `git switch -c feature/<person>/<topic>`.
2. One feature per branch; tests with every change; never weaken, skip or delete a test.
3. Before pushing: `cmake --build build -j && ctest --test-dir build` (all green, including the
   Python tests — configure with `-DPython3_EXECUTABLE=<python with highspy, numpy, scipy>`).
4. `git push -u origin feature/<person>/<topic>`; open a pull request to `main`.
5. CI must be green (Ubuntu + macOS build/test, ASan/UBSan, TSan, no-foreign-solver check).
   `scripts/ci_local.sh` runs the same jobs locally (before pushing, or when GitHub Actions is
   unavailable — then paste its summary into the PR).
   GPU changes additionally need a `scripts/gpu_check.sh` log from real hardware in the PR.
6. The other teammate reviews; merge with "squash and merge" (or "rebase and merge" to keep
   individual commits). Delete the branch.

## Claims and evidence

- No number in README/docs unless it comes from a committed CSV in `bench/results/` whose
  name carries the git short hash of the binary that produced it (`ps26119 --version`).
- Benchmark from a clean commit with a frozen copy of the binary; never rebuild `build/`
  while a benchmark is running from it; `caffeinate -i` on macOS.
- Never drop failing instances from a table; report time limits as time limits.
- Never claim anything about the GPU until `scripts/gpu_check.sh` has run on NVIDIA hardware
  and its CSVs are committed.

## Commit messages

Imperative one-line summary, blank line, what and why. Bug fixes name the regression test.
Keep authorship honest: do not rewrite someone else's code merely to change its
appearance; if you substantially adapt it, say so in the header comment (`Origin: ...`).
