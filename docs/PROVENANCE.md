# Provenance

Two questions a judge (or a competitor) can ask, and how this repository answers them.

## 1. Is the solver built on an existing solver? — No.

| Check | How | Result |
|---|---|---|
| Link dependencies of the CLI and `libps26119` | `otool -L` / `ldd` (in `scripts/check_no_solver_linked.sh`) | only the C++ runtime and libSystem/libc |
| Foreign solver symbols | `nm` for `Highs_ CPX GRB XPRS SCIP glp_ Clp_ Cbc_` | none |
| Solver headers / imports in `src/ include/ apps/` | grep for highs, cplex, gurobi, xpress, scip, soplex, glpk, clp, cbc, coin, mosek, copt, cuopt, pdlp, ortools, lpsolve | none |
| The Python package `python/ps26119` | no import of highspy/scipy/pulp/cvxpy/pyomo/gurobipy/cplex/ortools/mosek/xpress | none (ctypes over our library) |
| The build | no `find_package`/`find_library` of a solver | none |
| Vendored code | `git ls-files` | none; GoogleTest is fetched for tests only |
| CI | `scripts/check_no_solver_linked.sh` runs on every push | enforced |

External solvers appear only as **references** outside the solver: highspy (HiGHS) in
`tools/` (the verifier's independent MPS reader, the old `.lpm` bridge, reference optima) and
`bench/`; SciPy's HiGHS in `tools/crosscheck_random_mps.py`. See `docs/THIRD_PARTY.md`.

Algorithms follow published papers, cited in each file header. Public implementations that
informed an approach are listed under "Read, not copied" in `docs/THIRD_PARTY.md`.

## 2. Who wrote what?

The repository merges two independently written codebases (2026-09-26):

| Codebase | Author | Imported how |
|---|---|---|
| ps26119 (this repository before the merge, tag `pre-teammate-integration` = `20033bf`) | Jai (AI-assisted: commits carry `Co-Authored-By: Claude`) | native history |
| gpuopt (`github.com/shivanshu24-code/gpu_optimization`, commit `c192dd0`) | Shivanshu Vats | subtree merge `56a559d` whose second parent is `c192dd0` (his authorship and date preserved), then a pure-rename commit `670e9d9`, then adaptation commits |

Every file that came from gpuopt starts with an `Origin: gpuopt …` line naming the original
path. Lines per area attributed by `git blame -C -C -M` (rename and copy detection) at the end
of the integration:

| Area | Shivanshu Vats | Jai |
|---|---:|---:|
| `src/io` (MPS reader; `.lpm` + solution I/O) | 678 | 562 |
| `src/la` (sparse LU, CSC; CSR, double-double, thread pool) | 640 | 593 |
| `src/simplex` (primal simplex, simplex scaling) | 688 | 82 |
| `src/oracle` (tableau oracle; double-double oracle) | 397 | 585 |
| `src/core` (solution checker; dispatcher, gate, presolve, certified bound, C API) | 154 | 1442 |
| `src/pdhg` (PDLP, r²HPDHG, scaling, batch) | — | 1902 |
| `src/gpu` (CUDA backend) | — | 667 |
| `src/mip` (branch-and-bound) | — | 279 |
| `apps/cli` (model report, LU bench; CLI main) | 448 | 425 |
| `tests/unit` | 763 | 2619 |
| `tools`, `bench`, `python` | 208 | 3157 |

The "Jai" column includes the adaptation lines in teammate files (namespace/`Model` port,
bug fixes from the audit); the algorithms in those files are unchanged
(`docs/audit/INITIAL_AUDIT.md`, `docs/FINAL_INTEGRATION_REPORT.md`).

Recompute:
```bash
for f in $(git ls-files src | grep -E '\.(cpp|h|cu)$'); do git blame -C -C -M --line-porcelain "$f" | grep '^author '; done | sort | uniq -c
```
