# Third-party material

Rules: CLAUDE.md §4. Solver core (`src/`) contains only our own code — papers and public
repositories that informed an algorithm are cited in the file header, never pasted.
This table lists third-party code or tools used in `tools/`, `bench/`, `tests/` and the build.

| What | Licence | Source URL | Where used | What was used |
|---|---|---|---|---|
| GoogleTest v1.15.2 | BSD-3-Clause | https://github.com/google/googletest | tests (FetchContent, not in the binary) | test framework |
| highspy (HiGHS Python bindings) | MIT | https://github.com/ERGO-Code/HiGHS | `tools/` and `bench/` only (MPS bridge, verifier's independent reader, reference solves) | Python package, run as a separate reference; never linked into the solver |
| Google OR-Tools (`ortools`) 9.15 | Apache-2.0 | https://github.com/google/or-tools | `tools/ortools_ref.py`, `bench/compare_highs.py --refs …,ortools` | GLOP simplex and PDLP as separate references (own process); never linked |
| PySCIPOpt 6.2 (bundles SCIP 10, Apache-2.0) | MIT | https://github.com/scipopt/PySCIPOpt | `tools/scip_ref.py`, UI live comparison, `--refs …,scip` | SCIP / SoPlex as a separate reference (own process); never linked |
| CyLP 0.94 (bundles COIN-OR CLP and CBC, EPL-2.0) | EPL-2.0 | https://github.com/coin-or/CyLP | `tools/coin_ref.py`, UI live comparison, `--refs …,coin` | CLP dual simplex and CBC branch and cut as separate references (own process); never linked |
| NumPy | BSD-3-Clause | https://numpy.org | `tools/`, `bench/` | array maths in the verifier and generators |
| matplotlib | PSF-based (BSD-compatible) | https://matplotlib.org | `bench/scale.py` charts | plotting only |
| SciPy (`scipy.optimize.linprog`, HiGHS inside) | BSD-3-Clause | https://scipy.org | `tools/crosscheck_random_mps.py` (external reference in the `tools.crosscheck.*` tests) | reference status/objective for random MPS models; never linked |

Team code is not third-party: the MPS reader, sparse LU, primal simplex, tableau oracle and
solution checker were written by team member Shivanshu Vats (gpuopt, merged with history;
see docs/PROVENANCE.md).

## Read, not copied (solver core)

Public implementations we studied for approach and default constants. No code was pasted;
the solver files that were informed by them cite them in their header comment.

| Source | Licence | Informed | Where cited |
|---|---|---|---|
| MIT-Lu-Lab/cuPDLPx (`src/solver.cu`, `src/utils.cu`, `src/preconditioner.cu`) | Apache-2.0 | r²HPDHG defaults (reflection 1.0, restart 0.2/0.5/0.36, PID 0.99/0.01/0, integral decay 0.3), bound/objective rescaling, fused-kernel structure | `src/pdhg/r2hpdhg.h`, `src/pdhg/scaling.h`, `src/gpu/cuda_backend.cu` |
| QD library (Hida, Li, Bailey) — the paper, not the code | BSD-style | double-double algorithms | `src/la/dd.h` |
