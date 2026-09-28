# Command Center GPU runs — RTX 4050 Laptop (WSL2), commit ac60baa

Runs started from the Command Center (`apps/ui`, Solve page, backend **GPU (CUDA)**) on the
rtx4050-laptop machine: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, driver 616.64), CUDA 13.3,
Intel Core 5 210H (12 threads), WSL2 Ubuntu. Binary `ps26119 0.1.0 (git ac60baa)`, CUDA build.
Every row is a live run of that binary; nothing was re-run or edited for this export.

- `../../ui-gpu-runs-rtx4050-laptop-ac60baa.csv` — one row per run (all 16, including the failed ones)
- `<job>.json` — per run: options, exact command, solver stdout, `tools/verify.py` report, certificate
- `report.html` — the UI's self-contained report of the 13 verified runs (`/api/report`)
- `system.json` — the UI's machine / build probe

## Results (13 of 16 runs Optimal, verify.py PASS, certificate PASS)
| model | rows | precision | iterations | solve s | wall s | verify |
|---|---|---|---|---|---|---|
| afiro (Netlib) | 27 | fp64 / mixed | 448 | 0.5–2.6 | 0.6–2.9 | PASS |
| recipe (Netlib) | 91 | mixed | 1536 | 0.49 | 0.58 | PASS |
| rand-10000-s1 | 1e4 | fp64 | 1536 | 1.1–1.6 | 3.5–3.9 | PASS |
| rand-100000-s1 | 1e5 | fp64 / mixed | 2496 / 2944 | 3.6 / 2.0–2.5 | 28 / 11–12 | PASS |
| rand-1000000-s1 | 1e6 | fp64 | 17600 | 87.4 | 164.8 | PASS |
| refinery-T8760-s1 (hourly year) | 429,240 | fp64 | 2880 | 9.3 | 117.4 | PASS |

Generated models match their optimum known by construction (rel. error in the CSV). The refinery
year's objective is 21,158,525.6251, with a rounding-proof bound of 21,158,525.6284.

## Failed runs (kept, not hidden)
Three rand-1000000-s1 runs (jobs 5add6b72ff2c, b58cd772ee8f, dfd49a08fc9c; time limits of 60 and
80 s) were terminated (exit −15) before the solver finished. Two left a truncated solution file
that verify.py could not read. None of the three is certified. The same model with a 300 s limit
(c7f78606e639) solved and verified.

## Caveat on wall time
"solve s" is the solver's own time. "wall s" also includes reading the model from the Windows drive
(`/mnt/c`, 9P) inside WSL2 and writing the solution back: for the refinery year that is ~108 of the
117 s. Timings from a WSL-native disk will be much closer to "solve s".
