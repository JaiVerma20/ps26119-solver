# SIH 2026 PS 26119 (MRPL) — requirement status

Status words: **DONE** (implemented, tested, benchmark CSV committed), **PARTIAL**,
**EXPERIMENTAL**, **NOT STARTED**. Evidence = committed CSV (git hash in the name) or test.
State at the end of the integration (benchmark binary `d824f82`, 2026-09-26).

| Requirement | Implementation | Status | Test | Benchmark | Evidence | Remaining work |
|---|---|---|---|---|---|---|
| Solver not built on an existing solver | all engines, reader, LU written by the team; no solver linked | DONE | `scripts/check_no_solver_linked.sh` (CI) | — | `docs/PROVENANCE.md` | keep the check green |
| Read industry-standard models (MPS) | `src/io/mps_reader.cpp` (free + fixed, RANGES, BOUNDS, markers, OBJSENSE) | DONE | `MpsReader.*`, `Integration.MpsReaderMatchesLpmBridge…` | all Netlib + MIPLIB runs read MPS | reader fingerprint == highspy on 93 Netlib + 14 MIPLIB 3 (`docs/audit`) | gzip, QP sections, SOS |
| LP: correct optimal solutions | simplex, r²HPDHG, PDLP, oracle, `auto` | DONE | unit + differential (1,800 random LPs × 5 configs, 0 mismatches) | `netlib-full-*-d824f82.csv` | Netlib 93/93 (auto), each verified by an independent checker and equal to HiGHS | Mittelmann large LPs |
| LP: duals / reduced costs | all engines return y, z (HiGHS signs) | DONE | gate + `verify.py` dual checks | same | same | ranging / sensitivity report |
| Detect infeasible / unbounded | simplex (fresh-factor re-check), PDHG (checked rays), crossed bounds | DONE (LP) | `Integration.InfeasibleAndUnboundedExamples`, `tools.crosscheck.*` (vs SciPy/HiGHS) | — | random MPS cross-check 200/200 per engine in CTest | Netlib infeasible set; Farkas certificate from the simplex |
| Numerical robustness ("never lie") | gate on every Optimal, certified bound, presolve re-check, NaN rejection | DONE | `Gate.*`, audit regression tests, ASan/UBSan/TSan in CI | — | `docs/audit/INITIAL_AUDIT.md`, DECISIONS #27–#30 | exact rational certificate on one instance |
| Large-scale LP (refinery planning) | r²HPDHG, deterministic CPU threads, mixed precision | DONE (CPU) | `PdhgEngines.*`, thread determinism | `scale-macbook-air-m4-d824f82.csv` | hourly-year refinery LP (429k rows) in 14.7–19.6 s, known optimum reproduced | real plant data |
| GPU acceleration | CUDA backend behind `pdhg/backend.h` | PARTIAL (correct on one laptop GPU; speed vs 1 CPU thread only) | `Gpu.*` pass on the teammate's RTX 4050 laptop (2026-09-27) | `scale-rtx4050-laptop-82d376c.csv`, `netlib-small-gpu-82d376c.csv` | refinery year 3.1× and 1e5 random 4.3× vs ONE CPU thread; GPU slower below ~1e4 rows | multi-core CPU baseline, compute-sanitizer on native Linux, university GPU |
| MILP | branch-and-bound, sparse simplex nodes, certified pruning | EXPERIMENTAL (prototype) | `Mip.*` (enumeration, node-solver agreement) | `miplib3-macbook-air-m4-d824f82.csv` | 10/14 small MIPLIB 3 proven optimal and verified | cuts, heuristics, warm-started nodes, larger MIPLIB |
| QP / MIQP / NLP | — | NOT STARTED | — | — | — | IPM or PDHG for convex QP |
| What-if / scenario analysis | warm start (PDHG), batched scenarios | PARTIAL | `PdhgWarmStart.*`, `Batch.*` | `warm-start-*-8fd5170.csv`, `batch-*-8fd5170.csv` (pre-merge) | same | simplex warm start; GPU SpMM |
| Presolve | safe reductions + postsolve re-check | PARTIAL | `Presolve.*` | Netlib ablations | 85 vs 83 (PDHG, `01f2eec`) | doubletons, dominated columns |
| APIs (library use in industry) | C API, Python (ctypes), CLI | DONE | `CApi.*`, `python.binding`, `cli.contract` | — | — | packaging (pip wheel) |
| Reproducible evidence | hash-named CSVs, frozen binary, `make_evidence.py` | DONE | `bench.generators`, provenance columns | all | `docs/EVIDENCE.md` | GPU CSVs |
| Team workflow | one private repository (github.com/JaiVerma20/ps26119-solver), feature branches, PRs; `main` protected (PR + 4 green CI jobs, no force push) | DONE | CI workflow (Ubuntu, macOS, ASan/UBSan, TSan) | — | PR #1, `docs/CONTRIBUTING.md` | teammate accepts the invitation; optionally enforce the rules for admins too |
