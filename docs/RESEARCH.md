# PS 26119 — Research Dossier (compiled 24 Sep 2026)

Purpose: everything we know about SIH26119 (MRPL, "Indigenous GPU-Accelerated
Optimization Solver — Sovereign Alternative to Xpress / CPLEX"), the competition,
the state of the art, and the plan. Give this file to Claude Code as background.
Every claim about another team comes from their PUBLIC README / site on 24 Sep 2026
and is self-reported by them; re-check before quoting in a PPT.

---

## 0. TL;DR (read this if nothing else)

1. The PS is effectively: build an LP / MILP / QP solver from scratch (no existing
   solver inside), GPU-accelerated, benchmarked on Netlib / MIPLIB / Mittelmann,
   compared to an established solver, robust on degenerate / ill-conditioned models,
   usable through a CLI or API, for refinery-style planning problems.
2. The competition is far ahead in code. One team (SANKHYA) reports 29 of 32
   requirements done, a working CUDA backend, 930 tests, and a public site. Another
   (VYUHA / SOVOPT) has simplex + GPU PDLP + a GPU-batched branch-and-bound idea.
   A third repo (PRAMAAN) uses our name and our "proof" positioning.
3. We cannot win by being a smaller copy of SANKHYA. We need ONE thing we are
   clearly best at, measured, plus a credible general core.
4. Recommended wedge: **GPU-native LP at the 2025 state of the art (restarted
   Halpern PDHG with reflection = the cuPDLPx algorithm family), aimed at large
   multi-period refinery planning LPs**, with an exact simplex beside it for
   vertices, duals and warm starts. The PS title says GPU; the published GPU numbers
   from competitors are modest (1.37x on a laptop, 4.21x on an L4 at 10k rows for
   SANKHYA; 3.1x at ~1M nonzeros for VYUHA). That is the gap.
5. Rename the project (PRAMANA collides with PRAMAAN).
6. CONFIRMED with the college (24 Sep 2026): the same team can submit ideas for two
   problem statements (26173 and 26119), and the PS text, template and deadline are
   all confirmed from the college side.

---

## 1. The problem statement

Official listing: SIH26119, MRPL, Software, title "Indigenous GPU-Accelerated
Optimization Solver (Sovereign Alternative to Express / CEPLEX)" (typos are in the
official title). Listed theme varies by source (Smart Automation / Miscellaneous).
One tracker showed 12 ideas for this PS (early count, will change).

We have NOT seen the official description text directly. The best available
reconstruction is SANKHYA's public coverage tracker, which maps the PS to 32
checkable requirements. Treat this as a checklist, and verify against the portal.

### 1.1 Requirement checklist (reconstructed)

Problem classes
- Linear programming (LP)
- Mixed-integer LP (MILP)
- Quadratic programming (QP), convex; MIQP appears in their tracker
- Modular design that can grow to NLP / MINLP

Algorithms the PS names
- Revised simplex
- First-order methods (PDHG / PDLP)
- Interior-point methods
- Branch-and-bound
- Branch-and-cut / cutting planes
- Presolve
- Heuristics
- Advanced node selection
(SANKHYA also tracks warm start, sensitivity ranging, pricing — possibly their additions)

Implementation requirements
- Sparse matrix techniques
- Efficient numerical linear algebra
- Multi-core parallelisation
- GPU acceleration
- Reproducible runs
- Predictable behaviour under resource limits (time, iterations, nodes)
- NOT built on any existing solver (hard requirement — the "sovereign" part)

Industrial scope (domains to demonstrate)
- Crude blending, refinery scheduling, power system dispatch,
  transportation / supply chain, production planning, logistics, process optimisation

Benchmark bar
- Highly degenerate models
- Ill-conditioned matrices
- Weak LP relaxations
- Difficult MILP formulations
- Thousands to millions of variables

Expected solution
- Basic API OR CLI (a GUI is not required)
- Netlib, MIPLIB, Mittelmann benchmark results
- Comparison against an established solver
- Robustness demonstration
- A transparent, extensible, sovereign foundation

### 1.2 Process and dates (from third-party sources — confirm locally)
- Teams go through a college internal hackathon; the SPOC nominates teams and
  submits ideas on the official portal. One source says SPOC idea submission closes
  30 Sep 2026 and the Grand Finale is in December 2026.
- Shortlisted finalists reportedly get mentors and build a prototype before the
  finale; the software finale is a 36-hour sprint.
- Many colleges require the official 6-slide SIH template, often as PDF.
  Example weightings published by one college portal (NOT official SIH):
  problem understanding & innovation 20%, technical approach & feasibility 25%,
  prototype / implementation plan 25%, impact & scalability 20%, clarity 10%.
- Evaluators reportedly spend only 2-3 minutes per PPT. Clarity beats density.

---

## 2. Competitor landscape (public repos / sites, 24 Sep 2026)

### 2.1 SANKHYA — thegoodengineers (Bengaluru, 6 students) — the benchmark to beat
Repo: github.com/thegoodengineers/SANKHYA (older forks exist under Deekshith2205,
Bhumika-1432006, thegoodengineer, AyushVUpadhye — forks show OLDER state).
Site: sankhya-solver.vercel.app. Apache-2.0. C++20.

What they report as done:
- Primal + dual revised simplex (dual default), Markowitz sparse LU, hyper-sparse
  FTRAN, product-form update, devex default, dual steepest edge opt-in, Harris opt-in
- Presolve (8 reductions + more behind flags) with postsolve that re-measures
- Restarted PDHG (CPU) + CUDA backend (single and multi-GPU), VRAM check, CPU fallback
- Mehrotra interior point with their own sparse LDLᵀ, crossover to a vertex
- Branch-and-bound with reliability branching, warm-started node LPs, conflict
  analysis, cuts (Gomory, cover, MIR, clique, {0,1/2}, flow cover), cut selection,
  heuristics (diving family, feasibility pump, RINS, RENS), parallel tree search
- Convex QP (Condat-Vu), MIQP, convex NLP entry point; non-convex refused with certificate
- Sensitivity ranging (cost + RHS), IIS, Farkas / ray certificates, independent
  Python verifier, deterministic mode, resource limits, warm start + in-place edits
  (rolling-horizon refinery: warm = 0.121x the iterations of cold)
- CLI, C API, Python bindings, SBOM, CI that fails if a solver library is linked
- Netlib full set 80-81/89 matched (they argue the remaining ones are table
  outliers), HiGHS cross-check 50/50 on medium tier, ~2x slower than HiGHS
- MIPLIB 2017 (30 easy): ~14 reach optimum, 9-10 prove it
- Mittelmann (8 smallest): 2/8 on auto, 3/8 across engines
- GPU: RTX 5050 laptop loses below ~10k rows, 1.37x-1.63x at 10k x 10k;
  NVIDIA L4 4.21x at 10k (1e-4), 3.90x (1e-8); 6.47x / 8.30x on Mittelmann brazil3
- Generated refinery year: daily (32,485 rows) solved exactly; hourly (779,640 rows)
  reached by PDHG to ~1e-6 in 120 s, not proved
- Open issue #514 (opened ~24 Sep): global spatial branch-and-bound for pooling,
  estimated 6 days. Issue #516: pooling benchmark set (Haverly, Ben-Tal, Foulds, Adhya).

What they say is still weak:
- GPU claim is one card, synthetic family, crossover at ~10k rows, nothing above
- MIPLIB proofs (9-10 of 30), Mittelmann (2-3 of 8)
- MINLP not attempted; pooling currently refused (sBB in progress)
- No real refinery data

Lessons worth copying (ideas, NOT code):
- Every number tied to a committed CSV + commit hash
- "What this does NOT do" section — honesty scores
- Independent verifier; HiGHS as a separate-process reference only
- One-command reproduce script; evaluator checklist page

### 2.2 VYUHA / SOVOPT — manabsen006-a11y (repo "Sovereign-MOE")
Python + numba + CuPy. MIT licence. From README:
- Bounded primal + dual revised simplex, Harris ratio test, devex, product-form update
- Threshold Markowitz LU (Gilbert-Peierls), hypersparse FTRAN/BTRAN, iterative
  refinement with compensated residuals, Hager condition estimation
- PDLP-class first-order LP on CPU + CUDA (custom kernels)
- MILP B&B with exact dual simplex node LPs OR "Batched Node Relaxation":
  many nodes bounded at once via one sparse-times-dense product on GPU (novel idea),
  made rigorous with Neumaier-Shcherbina safe dual bounds
- GPU vs CPU: 0.14x at 12k nnz, 1.5x at 240k, 3.1x at 960k nnz (laptop RTX 3050)
- Refinery templates, CLI, web UI, verifier
- Not built (per README): ranging, crossover, cuts (a separate description mentions
  Gomory + cover cuts), IPM, QP solve, pooling (a search snippet suggests newer
  pooling / recursion work exists)
- Note from them: consumer GPUs run fp64 at 1/32 rate — a datacentre card changes things

### 2.3 PRAMAAN — Parth-Gochhwal
Skeleton only (3 commits, placeholder CLI). Positioning: "sovereign, structure-aware,
precision-adaptive, proof-carrying LP/MILP/QP solver". SAME NAME + SAME PROOF
positioning as our deck. => We must rename.

### 2.4 From the idea videos (earlier analysis)
- GANIT: GPU-first PDLP plan, no evidence.
- ELYTRA / PIVOT-X: detailed C++20 plan; promised "parallel pricing in simplex on
  GPU" (weak claim).
- SANKHYA and SOVOPT videos = the repos above.

### 2.5 Honest assessment
- A team starting today cannot out-feature SANKHYA before December.
- Idea-stage selection is on the PPT; if selected, finalists have ~2 months.
- Our edge must be depth in one measurable area + honest positioning.

---

## 3. Where the real gaps are (what nobody has nailed)

1. **State-of-the-art GPU LP.** Competitors use classic restarted PDHG / PDLP-class
   methods. The newest published family is restarted Halpern PDHG with reflection
   (r²HPDHG) as used in cuPDLPx (Lu, Peng, Yang 2025), reported 2.5x-5x faster
   than its predecessor on MIPLIB LP relaxations and 3x-6.8x on Mittelmann.
   Nobody in this PS has published results with it.
2. **GPU at refinery scale.** No competitor shows a GPU number on a refinery-
   structured model or above ~10k rows (SANKHYA) / ~1M nnz (VYUHA). The hourly
   refinery year (~780k rows) is "reached, not proved" on CPU.
3. **The planner workflow.** Real refinery planning tools (Aspen PIMS, Honeywell
   RPMS, Haverly GRTMPS) run successive LP / distributive recursion: many LP solves
   in a loop, each warm-started. PIMS historically used CPLEX as its LP engine.
   So the product MRPL actually needs is a fast, warm-startable LP engine that
   survives SLP loops — with duals / ranging the planner reads.
4. **Mixed precision on GPU** (fp32 inner loop + fp64 correction), because consumer
   and many datacentre GPUs are weak at fp64. SANKHYA's video showed a mixed-
   precision idea; no repo shows it measured.

---

## 4. Recommended strategy

Positioning (one line): "A GPU-native LP engine built on the 2025 state of the art
(r²HPDHG), with an exact simplex beside it for vertices, duals and warm starts —
tuned for large multi-period refinery planning models."

Why this wedge:
- It is literally the PS title (GPU-accelerated).
- It is measurable: time-to-1e-4 / 1e-8 on Mittelmann + generated refinery LPs.
- It matches the team's strengths (C++, CUDA, computer architecture interest).
- It extends naturally into MILP (GPU LP for node relaxations / heuristics) later.

What we still must have for the PS (post-submission, before finale):
- Dense exact oracle + sparse revised dual simplex (vertices, duals, warm start)
- Presolve (basic), crossover from PDHG point to basis
- Branch-and-bound MILP (basic), convex QP via PDHG-QP (same family: Lu & Yang
  have a practical optimal first-order method for convex QP)
- Netlib / MIPLIB / Mittelmann harness, HiGHS reference, independent verifier

Unique ideas (full list in CLAUDE.md §9): r²HPDHG on GPU; mixed precision (fp32
iterations + fp64 residuals/polish) for fp64-weak GPUs; warm-started PDHG across
SLP / rolling-horizon re-solves; batched scenario solving (SpMM over many LPs that
share one matrix); certified output from a GPU iterate (safe dual bound + crossover);
a vendor-neutral GPU layer (CUDA first, HIP/Metal later).

Team machines: MacBook Air M4 (CPU dev, no CUDA; note long double = double on Apple
Silicon), teammate's NVIDIA laptop, university GPU servers for published numbers.

What NOT to do:
- A web dashboard (competitors have them; PS does not need a GUI)
- Claims we cannot back with a CSV
- Pasting solver-core code from other solvers or competitors (reading and learning
  from them is encouraged; see the source policy in CLAUDE.md §4)

---

## 5. Algorithms — what to implement and where to read

### 5.1 PDHG for LP (baseline, get this right first)
Standard form: min cᵀx s.t. Ax = b, x ≥ 0. Saddle point with dual y.
    x⁺ = proj_{x≥0}( x − τ (c − Aᵀy) )
    y⁺ = y + σ ( b − A(2x⁺ − x) )
Step sizes: τσ‖A‖₂² < 1. Parameterise τ = η/ω, σ = ηω; η < 1/‖A‖₂ (estimate ‖A‖₂
by power iteration); ω = primal weight.
For our Model's two-sided form (row_lower ≤ Ax ≤ row_upper, col bounds), derive the
dual update carefully (projection of the dual onto the correct sign per bound);
VERIFY SIGNS BY UNIT TEST against the oracle on tiny LPs.

PDLP enhancements (Applegate et al., NeurIPS 2021; journal version 2025):
- Diagonal preconditioning: Ruiz equilibration (~10 passes) + Pock-Chambolle (α=1)
- Adaptive restarts (to average or current iterate) on a merit function
- Primal weight updates
- Feasibility polishing / infeasibility detection
Termination (relative KKT, standard): primal residual ≤ ε(1+‖b‖),
dual residual ≤ ε(1+‖c‖), gap |cᵀx − bᵀy| ≤ ε(1+|cᵀx|+|bᵀy|). ε = 1e-4 and 1e-8.

### 5.2 Restarted Halpern PDHG with reflection (the target)
Papers: Lu & Yang, "Restarted Halpern PDHG for Linear Programming" (arXiv 2407.16144);
Lu, Peng & Yang, "cuPDLPx" (arXiv 2507.14051); overview: Lu & Yang arXiv 2506.02174;
related HPR-LP (Chen, Sun, Yuan, Zhao; relationships paper arXiv 2509.23903).
Key ideas (confirm exact constants in the papers before coding):
- Halpern anchoring: z_{k+1} = (k+1)/(k+2) · T(z_k) + 1/(k+2) · z_0, where T is one
  PDHG step and z_0 is the restart anchor.
- Reflection: use (1+γ)T − γI style reflected operator (r²HPDHG) for extra speed.
- Constant step size (no sequential step-size search — GPU-friendly).
- PID-controlled primal weight update.
- Restart criterion based on the fixed-point residual ‖z − T(z)‖ in the PDHG norm.
- All data resident on GPU; CSR matrix; custom kernels for SpMV / projections /
  reductions; minimise host syncs (a scalar read per iteration kills throughput).

### 5.3 Simplex (vertices, duals, warm start)
- Dense two-phase / bounded simplex in long double with Bland's rule = test oracle.
- Production: bounded dual simplex (Koberstein PhD 2005), Harris two-pass ratio test,
  dual steepest edge, bound flipping, Markowitz LU with threshold pivoting,
  Forrest-Tomlin update, hypersparse FTRAN/BTRAN (Huangfu & Hall 2018).
- Crossover: push a PDHG / IPM point to a vertex (needed for duals, ranging, warm start).

### 5.4 Industry pattern
- NVIDIA cuOpt (open source): concurrent LP = PDLP on GPU + barrier on GPU + dual
  simplex on CPU, first to finish wins. COPT and HiGHS also ship GPU PDLP.
- So "portfolio + concurrent" is the correct architecture story.

### 5.5 Correctness toolkit
- Independent verifier: reads MPS with a DIFFERENT reader (e.g. highspy) and checks
  primal/dual feasibility, complementarity, objective.
- Differential testing: random feasible bounded LPs vs HiGHS (separate process).
- Exact rational re-check of a final basis (Python fractions) for small instances.
- Safe dual bounds for approximate duals: Neumaier-Shcherbina (any y gives a valid
  bound after correction) — useful for PDHG-based bounds.

---

## 6. Refinery domain notes

- Planning LPs: crude selection, CDU cut points (swing cuts), unit modes, yields,
  blending with quality specs, inventory, multi-period, delivery commitments.
- Tools: Aspen PIMS, Honeywell RPMS, Haverly GRTMPS — successive LP (SLP) /
  distributive recursion to handle pooling nonlinearity; P-formulation is standard.
- Pooling problem = bilinear, non-convex. Classic benchmarks: Haverly 1/2/3 with
  global optima 400 / 600 / 750 (profit). Also Ben-Tal 4/5, Foulds 2-5, Adhya 1-4.
  Instance repo: github.com/poolinginstances/poolinginstances. Global method:
  McCormick relaxation + spatial branch-and-bound (a competitor already has an
  issue open for this; VYUHA mentions recursion vs global on Haverly).
- Implication for the solver: warm-start speed and repeated solves matter more
  than a single cold solve. Duals and ranging are what planners act on.

---

## 7. Data and benchmarks

- Netlib LP: uncompressed MPS files are in the coin-or/CyLP repo
  (cylp/input/netlib); originals on netlib.org are in compressed "emps" format.
  Reference optima come from MINOS 5.3 (1988); a few table values are known
  outliers — cross-check disagreements against HiGHS before calling a fail.
  afiro: 28 rows (incl. objective), 32 cols, optimum −4.6475314286E+02.
  Other small ones: adlittle 2.2549496316E+05, blend −3.0812149846E+01,
  sc50a, sc50b, sc105, kb2, share2b, stocfor1, recipe, israel.
- MIPLIB 2017 benchmark / easy set; Mittelmann LP benchmarks (large);
  QPLIB for QP; pooling instances above.
- Generated families with known optimum (build our own): random sparse, staircase
  multi-period, refinery-year (monthly / daily / hourly periods).

---

## 8. GPU development environment

- Free: Google Colab T4 (Turing, compute 7.5). fp64 is slow on T4 — measure fp32
  and fp64 separately; mixed precision is a real design question here.
- Indian GPU clouds (e.g. E2E Networks, used by SANKHYA for an L4 run) for a
  datacentre-card number later.
- CUDA 12.x, nvcc, cuSPARSE allowed? cuSPARSE / cuBLAS are linear-algebra
  libraries, not solvers; still, writing our own SpMV kernel is cleaner for the
  "sovereign" story. Start with our own CSR SpMV; compare against cuSPARSE only
  as a measurement.

---

## 9. Plan

### 9.1 The 4-day PPT sprint (evidence we can honestly show)
Day 1 — repo skeleton, Model contract, CLAUDE.md, verifier, dense oracle
        (teammate: MPS reader)
Day 2 — CPU PDHG baseline (PDLP-style) → CPU r²HPDHG; both validated vs oracle
        on 10 small Netlib LPs; CSV
Day 3 — CUDA r²HPDHG on Colab T4; scaling study on generated LPs (10k → 1M rows)
        and a generated multi-period refinery LP; CPU vs GPU per-iteration and
        time-to-1e-4; CSV + chart
Day 4 — PPT: rename, repositioning, evidence strip, honest limits; submit early

### 9.2 After submission (to December)
M1 sparse bounded dual simplex + presolve/postsolve + crossover
M2 Netlib full harness + HiGHS cross-check + Mittelmann GPU runs
M3 branch-and-bound MILP (dual simplex nodes) + basic cuts + heuristics
M4 convex QP via first-order method; ranging; IIS; warm-start API; SLP loop demo
M5 refinery case study (multi-period planning + SLP pooling), performance profiles

---

## 10. PPT implications
- New name. Positioning = GPU-native, state-of-the-art first-order LP + exact simplex,
  for large refinery planning.
- Remove "no competing student solver offers X" claims.
- Show only measured numbers; label targets as targets.
- Name the algorithm and cite it (r²HPDHG / cuPDLPx family) — judges reward
  literature awareness.
- Keep the "we will not claim GPU simplex" line.

## 11. College-side questions — all confirmed (24 Sep 2026)
- Official PS description text: confirmed by the college.
- Official template, file format and deadline: confirmed by the college.
- Same team submitting ideas for two PS (26173 and 26119): allowed.

## 12. Sources
- SANKHYA: github.com/thegoodengineers/SANKHYA, sankhya-solver.vercel.app,
  docs/PS26119_COVERAGE.md, issues #514 #516
- VYUHA/SOVOPT: github.com/manabsen006-a11y/Sovereign-MOE
- PRAMAAN: github.com/Parth-Gochhwal/PRAMAAN
- PS lists: github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements,
  zaidsayyed.in SIH 2026 PS tool
- cuPDLPx: arXiv 2507.14051, github.com/MIT-Lu-Lab/cuPDLPx
- Restarted Halpern PDHG: arXiv 2407.16144
- cuOpt: docs.nvidia.com/cuopt, developer.nvidia.com blog on GPU barrier
- GAMS blog on cuOpt (COPT and HiGHS GPU PDLP)
- Pooling: arXiv 1803.02955 (Haverly optima table), GAMS haverly model
- Refinery SLP / DR: haverly.com blog on distributive recursion;
  "Distributed Recursion Revisited" arXiv 2411.09554
- Netlib: coin-or/CyLP cylp/input/netlib; netlib lp/data readme
