# Jury demo — runbook

The demo is `#/demo` in the Command Center (top-bar button "▶ Jury demo"). Every solve in it is a
live run of the real binary on the presenting laptop, re-checked by `tools/verify.py`; the GPU step
shows the committed GPU run (and runs live if the laptop has a CUDA build). Nothing is recorded or
simulated, so rehearse on the machine you will present with.

## Before going on stage (5 minutes)
1. `git pull` on the branch you present, then `cmake --build build -j` — the binary's git hash must
   match the checkout (a `-dirty` build means uncommitted changes).
2. `python3 apps/ui/server.py --open`
3. Open **System check** (last item on the rail): everything green or "note". If a demo model is missing, click
   **Prepare demo models** (generates the 429k-row refinery year in ~2 s).
4. Close heavy applications; plug the laptop in (the fanless M4 throttles: the refinery year takes
   5–13 s depending on its thermal state — say "about ten seconds", not an exact figure).
5. Open the demo, press **F** for full screen, **N** to check your notes, then **N** again to hide them.

## Keys
`→` / `Space` next · `←` back · `Enter` run the current step live · `N` presenter notes ·
`F` full screen · `Esc` leave (outside full screen). The dots at the top jump to any step and turn
green when a step's run is certified.

## The seven steps (≈ 6 minutes)
| step | what happens | what to point at |
|---|---|---|
| intro | headline evidence (from the committed CSVs) and the agenda | "every number here has a source file" |
| 1 AFIRO | live simplex solve, <1 s | objective −464.7531…, independent verify PASS |
| 2 verification | the four links of the trust chain for that run | fingerprints equal; the verifier uses a different reader |
| 3 infeasible | live Farkas certificate | exact L₀ > 0 — a proof, not a message |
| 4 MILP | live branch-and-bound (gt2) | integer answer re-checked; MIPLIB count in the notes |
| 5 refinery year | live r²HPDHG on 429,240 rows, convergence chart | error vs the known optimum, certified |
| 6 GPU | committed RTX 4050 run vs the fastest CPU of that machine | 3.0× on the refinery year, source line; how each answer was checked |
| 7 final | table of the runs, "ALL N RUNS CERTIFIED", export | hand over the exported HTML report |

## If something goes wrong
- A step shows **NOT CERTIFIED**: say so — the product refuses to certify what it cannot prove. Open
  the Certificate page (link on step 2 / the certificate list) to show which check failed.
- A solve is slow: the refinery step has a 300 s limit; keep talking over the live convergence chart.
- The browser was closed: runs are kept on disk; the Certificate page lists them after a restart.
- No network is needed at any point (no CDN, no fonts, no external calls).

## Questions judges ask (answers backed by the repository)
- *Built on an existing solver?* No: `scripts/check_no_solver_linked.sh` runs in CI; HiGHS appears
  only as the verifier's file reader and as a reference in `bench/`.
- *Is the GPU number real?* `bench/results/scale-rtx4050-laptop-183c59c.csv`, logs and
  compute-sanitizer reports in `bench/results/logs/rtx4050-laptop-183c59c/` (GPU page).
- *How do you know the answer is right?* In-process check on the original model + independent
  verifier + fingerprint + (generated models) known optimum; failure withdraws "Optimal".
