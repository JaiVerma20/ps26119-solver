# Competition — where we stand against SANKHYA (and the field)

Internal working document, refreshed 2026-09-28 from the competitors' public repositories and our
committed evidence. Their numbers are quoted from their own documents (file and commit named); ours
come only from our committed, hash-named CSVs (CLAUDE.md §5.1). Different machines, generators and
rules: a row here is a comparison of *claims*, not a head-to-head measurement, unless it says so.
Never copy their code (CLAUDE.md §4); reading it for ideas and targets is allowed.

Sources: github.com/thegoodengineers/SANKHYA — `README.md`, `docs/BENCHMARKS.md` (Netlib full
`netlib-full-ad57c03.csv`, refinery `scale-refinery-e134aeb.csv`, HiGHS `compare-highs-medium-ad57c03.csv`,
GPU `gpu-l4-fdc89c5.csv`, `gpu-real-{l4-58a8374,a100-fdd1f35}.csv`), last pushed 2026-09-27.
Earlier survey: `docs/RESEARCH.md` §2.

## 1. Head to head on the things both of us measure

| topic | SANKHYA (their docs) | ps26119 (our CSVs) | who leads |
|---|---|---|---|
| Netlib, published / reference optimum to 1e-6 and verified | 81 of 89 (8 disagree with the published optimum) | `auto` 93 of 93 matched to HiGHS 1e-6 and verified (`netlib-full-auto-*-f440782`) | **us** (different reference: they use the Netlib table, we use HiGHS; the Netlib table has known outliers) |
| Infeasibility proofs | Netlib infeasible set, Farkas / ray certificates | 91/93 certified on Netlib + duality cut, 72 in exact rational arithmetic (`infeasible-cut-simplex-*`) | comparable; ours is a harder set (every Netlib LP made infeasible) |
| Refinery year, hourly, CPU | 779,640 rows: PDHG time limit at 120 s (rel. err 1e-8 reached, not proved), simplex / IPM time limit (`scale-refinery-e134aeb`) | 429,240 rows: Optimal, verified, known optimum to ~1e-13, seconds (`scale-*-f440782`) | **us** on "solved and verified"; their model is ~1.8× larger (different generator) |
| GPU vs CPU | laptop 1.37–1.63× at 10k×10k synthetic; L4 3.90× (1e-8) at 10k; A100 4.08×; brazil3 8.1× (L4) / 8.7× (A100) vs host CPU | RTX 4050 laptop: refinery year 3.03× vs the fastest CPU config, 1M-row LP 4.23× (`scale-rtx4050-laptop-183c59c`) | theirs on datacenter cards and brazil3; **ours on a refinery-structured model and at 1M rows on a consumer card** |
| Against HiGHS | 50/50 agree on their medium tier; ~2× slower overall (simplex) | `bench/compare_highs.py` — engine by engine (simplex / IPM / PDLP) on all of Netlib + refinery models, same verifier for both | run pending (see §4) |
| Verification | independent Python verifier, rational oracle | in-process gate on the original model + independent verifier with a different reader + fingerprint + known optimum; certificate page + HTML report in the UI | comparable; our UI makes it visible |

## 2. Where SANKHYA is ahead (honest list)

1. **Breadth**: dual simplex (default), interior point + crossover, convex QP / MIQP, NLP entry point,
   sensitivity ranging, IIS, MILP cuts (Gomory, MIR, cover, clique, flow cover), heuristics (diving,
   feasibility pump, RINS, RENS), parallel tree search, symmetry.
2. **Simplex speed**: close to HiGHS on small Netlib (0.4×–2.9× per instance).
3. **Datacenter GPUs**: L4, A100, V100 runs; multi-GPU row partition; cuDSS IPM.
4. **Benchmark coverage**: Kennington, Mittelmann (2–3 of 8), MIPLIB 2017 easy (≈14 optimal, 9–10 proved),
   Maros–Mészáros, QPLIB, Hock–Schittkowski, MINLPLib, a parametric-LP section, robustness sweep.
5. **Presentation**: public site, "negative results" page, evaluator checklist (PS coverage 29/32).

## 3. Where we are ahead

1. **Solved-and-verified at refinery scale** (hourly year, seconds, known optimum) and **GPU on a
   refinery-structured model** — the PS owner's actual use case (MRPL).
2. **Netlib completeness** (93/93 with `auto`) and certified infeasibility at scale.
3. **Planner workflow** in the product: warm-started what-if re-solves, batched scenario sweeps,
   marginal values of capacity (Command Center → Scenarios).
4. **The Command Center**: a live, verified jury demo, per-run certificates, an exportable report,
   GPU and benchmark evidence with sources — nothing simulated. None of the other teams' videos
   showed a verification chain on screen.

## 4. Plan to close the gaps (priority for the PPT and the finals)

| priority | item | owner | status |
|---|---|---|---|
| 1 | ps26119 vs HiGHS, engine by engine, all of Netlib + refinery models, same verifier | Jai | `bench/compare_highs.py` + UI page "vs HiGHS" |
| 2 | Datacenter-GPU run (A100 / L4 / university server) of `scripts/gpu_check.sh` | Shivanshu | NEXT_STEPS item 2 |
| 3 | Simplex speed: incremental dual update, then Forrest–Tomlin + hypersparse FTRAN/BTRAN | Shivanshu | NEXT_STEPS item 4 |
| 4 | Sensitivity ranging (cost + RHS) from the final basis — planners ask for it | team | not started (new code in `src/`, needs agreement) |
| 5 | Crossover from the PDHG iterate to a vertex (exact duals, ranging for large models) | Jai | M3 (CLAUDE.md §11) |
| 6 | Mittelmann LP subset + Kennington, reported with failures | Jai | not started |
| 7 | MILP: cuts (Gomory / MIR) and a primal heuristic | team | prototype today (12/14 small MIPLIB 3) |
| 8 | QP (the PS lists QP) | team | not started |

What not to do: do not claim "faster than SANKHYA" or "faster than HiGHS" anywhere without a CSV that
measures both on the same machine; do not quote their numbers as ours; do not match features we
cannot verify before the deadline.
