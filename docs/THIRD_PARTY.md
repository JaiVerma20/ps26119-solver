# Third-party material

Rules: CLAUDE.md §4. Solver core (`src/`) contains only our own code — papers and public
repositories that informed an algorithm are cited in the file header, never pasted.
This table lists third-party code or tools used in `tools/`, `bench/`, `tests/` and the build.

| What | Licence | Source URL | Where used | What was used |
|---|---|---|---|---|
| GoogleTest v1.15.2 | BSD-3-Clause | https://github.com/google/googletest | tests (FetchContent, not in the binary) | test framework |
| highspy (HiGHS Python bindings) | MIT | https://github.com/ERGO-Code/HiGHS | `tools/` and `bench/` only (MPS bridge, verifier's independent reader, reference solves) | Python package, run as a separate reference; never linked into the solver |
| NumPy | BSD-3-Clause | https://numpy.org | `tools/`, `bench/` | array maths in the verifier and generators |
