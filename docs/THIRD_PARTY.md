# Third-party material

Rules: CLAUDE.md §4. Solver core (`src/`) contains only our own code — papers and public
repositories that informed an algorithm are cited in the file header, never pasted.
This table lists third-party code or tools used in `tools/`, `bench/`, `tests/` and the build.

| What | Licence | Source URL | Where used | What was used |
|---|---|---|---|---|
| GoogleTest v1.15.2 | BSD-3-Clause | https://github.com/google/googletest | tests (FetchContent, not in the binary) | test framework |
| highspy (HiGHS Python bindings) | MIT | https://github.com/ERGO-Code/HiGHS | `tools/` and `bench/` only (MPS bridge, verifier's independent reader, reference solves) | Python package, run as a separate reference; never linked into the solver |
| NumPy | BSD-3-Clause | https://numpy.org | `tools/`, `bench/` | array maths in the verifier and generators |
| matplotlib | PSF-based (BSD-compatible) | https://matplotlib.org | `bench/scale.py` charts | plotting only |

## Read, not copied (solver core)

Public implementations we studied for approach and default constants. No code was pasted;
the solver files that were informed by them cite them in their header comment.

| Source | Licence | Informed | Where cited |
|---|---|---|---|
| MIT-Lu-Lab/cuPDLPx (`src/solver.cu`, `src/utils.cu`, `src/preconditioner.cu`) | Apache-2.0 | r²HPDHG defaults (reflection 1.0, restart 0.2/0.5/0.36, PID 0.99/0.01/0, integral decay 0.3), bound/objective rescaling, fused-kernel structure | `src/pdhg/r2hpdhg.h`, `src/pdhg/scaling.h`, `src/gpu/cuda_backend.cu` |
| QD library (Hida, Li, Bailey) — the paper, not the code | BSD-style | double-double algorithms | `src/la/dd.h` |
