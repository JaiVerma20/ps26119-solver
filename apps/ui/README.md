# PS26119 Optimization Command Center

A local web UI over the real solver. It adds no solver logic: every number on screen comes from
the `ps26119` binary, from `tools/verify.py`, or from the committed CSVs in `bench/results/`.

```bash
cmake --build build -j                 # the UI runs build/ps26119
python3 apps/ui/server.py              # → http://127.0.0.1:8765   (--port N, --open)
```
Python 3.10+, standard library only. The browser page loads nothing from the internet (no CDN,
no fonts), so the demo works offline. `tools/verify.py` needs `highspy` to re-read `.mps` files;
without it, verification of `.mps` models reports an error and the solve itself is unaffected.
Environment: `PS26119_BIN` (solver binary, default `build/ps26119`), `PS26119_PYTHON`
(interpreter for verify.py), `PS26119_UI_LOG=1` (request log).

## Pages
| key | page | what it shows | source |
|---|---|---|---|
| 1 | Dashboard | headline results with their source CSV + commit, one-click scenarios, recent runs | `bench/results/*.csv`, runs |
| 2 | Models | the model library, `ps26119 info` statistics, row/column type split, sparsity picture | `ps26119 info`, the file |
| 3 | Solve | engine / backend / precision / threads / limits, the exact CLI command, live progress, 15 result KPIs, residual and objective charts, solver log, primal and dual values | `ps26119 solve -vv` stderr + the solution file |
| 4 | Verification | the trust chain of the last run: engine verdict → in-process gate or certificate → rounding-proof bound → fingerprint → independent verifier; solver vs verify.py side by side | the solution file + `verify.py --json` |
| 5 | Benchmarks | Netlib per engine, certified infeasibility, GPU vs CPU with sanitizer status, large-LP scaling, MIPLIB | `bench/make_evidence.py` rules over `bench/results/` |

⌘/Ctrl+Enter starts a solve on the Solve page.

## Architecture
```
browser (web/: vanilla ES modules, SVG charts)  ──HTTP/SSE──  server.py (127.0.0.1 only)
                                                               ├─ backend/runner.py   ps26119 solve -vv (subprocess), progress parser, verify.py
                                                               ├─ backend/models.py   catalog, ps26119 info, sparsity grid
                                                               ├─ backend/evidence.py bench/make_evidence helpers over bench/results/*.csv
                                                               ├─ backend/system.py   version, git hash, CPU, CUDA build probe
                                                               └─ backend/paths.py    allowed model folders (no path traversal)
```
- Each run gets a folder in `apps/ui/.runs/<job>/` (git-ignored) holding `solution.sol` and
  `verify.json`; uploads go to `apps/ui/.runs/uploads/`.
- Progress comes from the `-vv` log (r²HPDHG / PDLP checks, simplex every 1000 iterations,
  branch-and-bound nodes) and is streamed as Server-Sent Events. First-order objective values in
  the log are in internal minimisation form; the UI flips the sign for maximisation models.
- Only files under the data folders listed in `backend/paths.py` can be opened or solved.
- JSON never carries NaN or infinity (an Infeasible objective is NaN): they are sent as `null`.

Tests: `apps/ui/test_ui.py` (ctest `ui.backend`) — log parsing, option validation, path safety,
JSON hygiene, a real AFIRO solve + verify.py PASS over HTTP, an infeasible model with its Farkas
certificate, and the evidence endpoint against the CSVs.
