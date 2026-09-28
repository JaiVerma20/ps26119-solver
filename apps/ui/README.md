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
| 1 | Dashboard | headline results with their source CSV + commit, one-click scenarios, recent runs (kept on disk) with their certificates | `bench/results/*.csv`, runs |
| 2 | PS coverage | every requirement of PS 26119: status, implementation, test, evidence, remaining work, and a “show me” link to the screen that demonstrates it | parsed live from `docs/SIH_STATUS.md` |
| 3 | Models | the model library, `ps26119 info` statistics, row/column type split, sparsity picture, test-LP generator | `ps26119 info`, the file |
| 4 | Solve | engine / backend / precision / threads / limits, the exact CLI command, live progress, 15 result KPIs, residual and objective charts, solver log, primal and dual values; after a run, the same model through HiGHS (tools/highs_ref.py) with the same verifier, side by side | `ps26119 solve -vv` stderr + the solution file |
| 5 | Verification | the trust chain of the last run: engine verdict → in-process gate or certificate → rounding-proof bound → fingerprint → independent verifier; solver vs verify.py side by side | the solution file + `verify.py --json` |
| 6 | Certificate | one projection-grade page per run with the final verdict; export as a self-contained HTML report (print it to PDF), "Present" mode | `/api/jobs/<id>/certificate` |
| 7 | Scenarios | what-if planning on the refinery LP: price / crude / demand / CDU / FCC levers; cold vs warm-started re-solve (convergence overlay, Δ profit); sweep of one lever through K values in one `ps26119 batch` pass (optionally also one by one); marginal values of capacity and demand from the duals | `ps26119 solve --warm`, `ps26119 batch`, verify.py on every answer |
| 8 | GPU | every configuration of the committed GPU run (CPU 1 thread / all cores / GPU × fp64 / mixed): time, iterations, objective, error vs known optimum, verify status, speed-ups, sanitizer, GPU-machine facts from its logs | `bench/results/scale-<gpu machine>-<hash>.csv`, `netlib-small-gpu-*.csv`, `bench/results/logs/<machine>-<hash>/` |
| 9 | Benchmarks | Netlib per engine, certified infeasibility, GPU vs CPU with sanitizer status, large-LP scaling, MIPLIB | `bench/make_evidence.py` rules over `bench/results/` |
| 0 | vs HiGHS | ps26119 against HiGHS (dual simplex, interior point, PDLP) on Netlib and the large / refinery models under one rule: performance profile, solved counts, head-to-head scatter, heat-map table, Optimal claims rejected by the verifier, every row | `bench/results/compare-highs-<machine>-<hash>.csv` (`bench/compare_highs.py`) |
| — | System check | demo preflight: binary (and whether it matches the checkout), verifier modules, demo models (generate the missing ones), evidence, disk | checked live |
| — | Jury demo (`#/demo`, top bar) | full-screen, keyboard-driven walk: AFIRO → verification chain → Farkas certificate → MILP → refinery year → CPU vs GPU → final verification with one exported report | live solves + committed GPU evidence; runbook in `apps/ui/DEMO.md` |

Keys 1–9 and 0 open the first ten pages (System check is on the rail); ⌘/Ctrl+Enter starts a solve on the Solve page. Adding a page is one module
with `mount(root)` in `web/js/pages/` plus one line in the registry at the top of `web/js/app.js`.

**Final verdict of a certificate** (`backend/certificate.py`): PASS only if the answer is
definitive (Optimal / Infeasible / Unbounded), the in-process check on the original model passed,
tools/verify.py passed, the verifier read the same model (fingerprints; not recomputed above
verify.py's size limit, which the certificate states), and — for generated models — the objective
is within 1e-6 of the optimum known by construction. Anything else is shown as NOT CERTIFIED with
the failed checks. The certificate also records the SHA-256 of the solution file.

Scenario notes: levers scale every period uniformly; the refinery structure is real and the
prices synthetic. Marginal values are the solver's duals as reported (checked on refinery-T12:
for the CDU and FCC rows y equals Δprofit / Δcapacity by finite differences). On a CPU the
batched sweep is not always faster than one-by-one solves (every scenario stays in the SpMM until
the slowest converges) — the page shows both timings when asked.

## Architecture
```
browser (web/: vanilla ES modules, SVG charts)  ──HTTP/SSE──  server.py (127.0.0.1 only)
                                                               ├─ backend/runner.py   ps26119 solve -vv (subprocess), progress parser, verify.py
                                                               ├─ backend/models.py   catalog, ps26119 info, sparsity grid
                                                               ├─ backend/scenarios.py what-if / sweep jobs on the refinery LP (lever edits, --warm, batch)
                                                               ├─ backend/generate.py bench/generate_*.py on demand, known-optimum sidecars
                                                               ├─ backend/evidence.py bench/make_evidence helpers over bench/results/*.csv (+ GPU detail)
                                                               ├─ backend/certificate.py the verdict of one run, from its own outputs
                                                               ├─ backend/report.py   self-contained HTML report (no scripts, prints to PDF)
                                                               ├─ backend/preflight.py demo-day readiness checks
                                                               ├─ backend/system.py   version, git hash, CPU, CUDA build probe
                                                               └─ backend/paths.py    allowed model folders (no path traversal)
```
- Each run gets a folder in `apps/ui/.runs/jobs/<job>/` (git-ignored) holding `solution.sol`,
  `verify.json` and `job.json` (its events, so certificates and reports survive a restart);
  uploads go to `apps/ui/.runs/uploads/`.
- Progress comes from the `-vv` log (r²HPDHG / PDLP checks, simplex every 1000 iterations,
  branch-and-bound nodes) and is streamed as Server-Sent Events. First-order objective values in
  the log are in internal minimisation form; the UI flips the sign for maximisation models.
- Only files under the data folders listed in `backend/paths.py` can be opened or solved.
- JSON never carries NaN or infinity (an Infeasible objective is NaN): they are sent as `null`.

Tests: `apps/ui/test_ui.py` (ctest `ui.backend`, 18 tests) — log parsing, option validation, path
safety, JSON hygiene, real solves over HTTP (AFIRO + verify.py PASS, Farkas certificate, a limit
that is neither verified nor certified), generated models against their known optimum, scenario
what-if and sweep, certificates (+ persistence across a restart, SHA-256 of the artifact), the HTML
report (content, escaping, no scripts or external files), the GPU detail number-for-number against
its CSV, preflight, the evidence endpoint, and a `node --check` of every browser module.
