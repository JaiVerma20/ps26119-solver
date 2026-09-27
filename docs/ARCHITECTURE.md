# Architecture

One library, one model contract, one result contract, one verification gate, several
engines. The CLI, the C API and the Python binding are thin layers over `ps26119::solve()`.

```
        MPS file                   .lpm file (generated models)      C API / Python arrays
            │ io/mps_reader  (teammate's parser)   │ io/lpm_reader              │ c_api.cpp
            └──────────────────────┬───────────────┴────────────────────────────┘
                                   ▼
                      Model  (include/ps26119/model.h — CSC A, row/col bounds, sense,
                              offset, integrality, names; fingerprint; validate())
                                   │
                                   ▼
   solve()  core/solve.cpp ── validate ── crossed bounds? → Infeasible (certificate)
            │                  │
            │     Auto engine choice (rows·nnz ≤ 2e8 → simplex, else r²HPDHG;
            │                         warm start / first-order knob → r²HPDHG)
            ▼
         presolve (core/presolve: empty rows, fixed/empty cols, singleton rows,
            │       integer bounds rounded inward)
            ▼
   ┌────────────────────┬──────────────────────────┬──────────────────────┬─────────────────┐
   │ integer columns?   │ LP                                                                │
   ▼                    ▼                          ▼                      ▼                 │
 mip/branch_and_bound  simplex/primal_simplex    pdhg/ PDLP, r²HPDHG     oracle/dense_simplex
   node LPs:            (teammate: Devex,         engine logic in fp64;   double-double,
   sparse simplex       Harris, bound flips,      O(nnz) work through     small models,
   (oracle fallback),   perturbation, Bland)      pdhg/backend.h:         test oracle
   pruned by the        │                          ├ CPU backend (fp64 / mixed,
   certified bound      ▼                          │   deterministic thread pool)
                       la/sparse_lu (teammate:     └ gpu/ CUDA backend (same interface)
                       Markowitz + threshold,
                       FTRAN/BTRAN, PFI updates)
   └────────────────────┴──────────────────────────┴──────────────────────┘
                                   │ Solution (include/ps26119/solution.h)
                                   ▼
            postsolve → re-check on the ORIGINAL model (presolve never weakens Optimal)
                                   ▼
   core::gate() — every Optimal LP answer re-checked by core/solution_checker (teammate's
                  checker; verifier tolerances from tolerances.h); failure → NumericalError
                  (first-order runs looser than 1e-6 keep Optimal at their tolerance, check=FAIL);
                  every Infeasible / Unbounded verdict's certificate (Farkas vector / point +
                  ray, core/certificates) checked on the original model; failure → NumericalError
                                   ▼
            certified bound (core/safe_bound, Neumaier–Shcherbina, rounding-proof) from y
                                   ▼
            Solution: status, x, y, z, objective, residuals, check, certified bound,
                      iterations, seconds, engine, precision, model fingerprint, message
                                   ▼
   io/solution_writer ──► tools/verify.py: EXTERNAL verifier with an independent reader
                          (highspy) — primal, dual, gap, integrality, fingerprint
```

## Contracts (one of each)

| Contract | File | Notes |
|---|---|---|
| Model | `include/ps26119/model.h` | Agreed with the reader author (CLAUDE.md §6). Crossed bounds are valid, infeasible data (DECISIONS #27). |
| Solution / Status | `include/ps26119/solution.h` | Dual convention z = c − Aᵀy (HiGHS signs). `check` = result of the gate. |
| Options / Algorithm | `include/ps26119/options.h` | `Auto, Oracle, Pdlp, R2hpdhg, Simplex`; engine knobs by name (`engine_params`). |
| Tolerances | `include/ps26119/tolerances.h` | The single place; `tools/verify.py` parses it. |
| C API | `include/ps26119/ps26119.h` | Exception-free; `ps26119_solve_lp`, `ps26119_solve_mps`. |
| Benchmark CSV | `bench/*.py` + `bench/machine_info.py` | git hash of the binary, machine, CPU, GPU, driver, CUDA, precision, tolerance, threads, date. |

Every engine consumes the same `Model` and returns the same `Solution`. The simplex and LU
code uses `la::SparseMatrixCSC` (a linear-algebra matrix: a basis, a scaled copy of A) and
the first-order code uses `la::Csr<T>`; neither is a second model representation.

## Engines

| Engine | Code | Origin | Best at | Limits |
|---|---|---|---|---|
| `simplex` | `src/simplex/`, `src/la/sparse_lu.*` | gpuopt (Shivanshu Vats) | small/medium LPs, vertex + exact duals, degenerate/badly scaled Netlib models | full pricing, PFI updates, no hypersparsity: slow beyond ~10⁴ rows; no warm start yet |
| `r2hpdhg` | `src/pdhg/r2hpdhg.*`, `engine.*`, `backend.h` | ps26119 (Jai) | large sparse LPs, CPU threads, GPU backend, mixed precision, warm start, batch | first-order accuracy (1e-8 relative KKT), not a vertex; struggles on some degenerate Netlib models |
| `pdlp` | `src/pdhg/pdlp.*` | ps26119 | reference first-order method | slower than r²HPDHG |
| `oracle` | `src/oracle/dense_simplex.*` | ps26119 | exact double-double answers on small models (testing) | dense |
| (test) tableau oracle | `src/oracle/dense_tableau.*` | gpuopt | second, independent oracle for differential tests | dense, test-only |
| MILP | `src/mip/branch_and_bound.*` | ps26119 (node solver: gpuopt simplex) | small MIPs, proven optimal | prototype: no cuts, no heuristics beyond rounding, cold node LPs |

## Verification layers

1. Engine-internal: simplex re-checks every verdict on a fresh factorization and in unscaled
   units; first-order engines terminate on fp64 KKT of the original model plus per-row checks.
2. Presolve safety net: postsolved answers re-checked on the original model; MILP answers
   re-checked for integrality and feasibility.
3. `core::gate()`: in-process checker on every Optimal LP answer (all engines), and on the
   certificate of every Infeasible / Unbounded verdict (DECISIONS #31, #33, #34).
4. Certified bound: rounding-proof bound on the optimum from y.
5. `tools/verify.py`: external, independent reader (highspy), used by every benchmark;
   Farkas certificates in exact rational arithmetic where the model allows.
6. Differential tests: dd oracle vs tableau oracle vs simplex vs r²HPDHG on random LPs;
   random MPS files vs SciPy/HiGHS; reader vs `.lpm` bridge fingerprints.

## GPU

`src/pdhg/backend.h` is the only interface the first-order engines use for O(n)/O(nnz) work;
`src/gpu/cuda_backend.cu` implements it (own CSR SpMV kernels, fused update + projection,
deterministic fixed-grid reductions). The CPU backend implements the same math, so the Mac
unit-tests the algorithm and the GPU machines test the kernels (`docs/GPU_VERIFICATION.md`).
The CUDA backend passes all correctness tests on an NVIDIA laptop GPU (RTX 4050, commit `183c59c`),
follows the CPU's fp64 iteration counts exactly, is compute-sanitizer clean (memcheck, racecheck on
the `Gpu.*` tests) and is 3.0–4.2× faster than the fastest CPU configuration of that laptop on
the refinery year and 1e5–1e6-row LPs, slower on small models (`docs/EVIDENCE.md` §3).

## Threads

`src/la/parallel.h`: a small deterministic pool (fixed chunking, fixed-order reductions) —
bit-identical results for any thread count (TSan-clean; 1/2/4/8/10/all threads verified).
Concurrent `solve()` calls from several threads are safe: pool jobs are serialised
(`Concurrency.ParallelSolvesFromSeveralThreadsMatchSequentialResults`).
The simplex and the MILP tree are single-threaded.
