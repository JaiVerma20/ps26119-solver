# Architecture (one page)

**Goal.** A from-scratch LP solver (MILP/QP later) whose differentiator is a GPU-native
first-order LP engine (restarted Halpern PDHG with reflection, r²HPDHG), with an exact
simplex beside it. Library first; the CLI and Python layers are thin.

```
            ps26119 CLI  (apps/cli)             tools/verify.py  (independent, highspy)
                 │                                     ▲ reads original MPS + our solution file
                 ▼                                     │
   readers:  io/lpm_reader  |  io/mps_reader (TEAMMATE)     io/solution_writer
                 │   Model (CSC, row bounds, col bounds)        ▲
                 ▼                                              │ Solution
          core/solve()  ── validate ── dispatch ──────────────────┘
             │               │                 │
       oracle/dense_simplex  pdhg/ (PDLP, r²HPDHG)   (later) simplex/, mip/
       double-double, tiny    │ engine logic in fp64 on host
                              ▼
                     pdhg/backend.h  ← the only interface engines use for vector math
                      ├── CPU backend  (fp64 or fp32 iterate, fp64 residuals)
                      └── gpu/ CUDA backend (same interface; compiled with PS26119_ENABLE_CUDA)
```

**Contracts.** `include/ps26119/model.h` (shared with the MPS reader author, CLAUDE.md §6)
and `solution.h` (§7). Tolerances live only in `tolerances.h`. Every Solution carries the
model fingerprint, engine, precision, iterations and seconds.

**Layers and rules.**
- `src/la` — sparse CSC/CSR, SpMV, power iteration, `dd.h` double-double (Apple Silicon
  has no 80-bit long double).
- `src/oracle` — a dense bounded two-phase simplex in double-double with Bland's rule.
  Slow on purpose; used only to check other engines on small models.
- `src/pdhg` — engine logic (scaling, restarts, primal weight, termination) runs on the host
  in double; all O(nnz) / O(n) work goes through `Backend`. That is what makes the CUDA
  port a backend swap instead of a rewrite, and keeps the Mac able to unit-test the math.
- `src/gpu` — CUDA implementation of `Backend` (own CSR SpMV kernels, fused
  update+projection kernels, device reductions, host sync only every K iterations).
- Verification is independent: `tools/verify.py` re-reads the original MPS with highspy
  (a different reader) and checks primal/dual feasibility and the gap. highspy is never
  used in `src/`; CI checks that no solver library is linked.

**Evidence.** Benchmarks write `bench/results/<name>-<githash>.csv` with machine info.
README/docs numbers must come from those CSVs.
