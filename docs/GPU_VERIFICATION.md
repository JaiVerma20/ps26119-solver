# What must be verified on a GPU (cannot be checked on the MacBook)

The Mac has no CUDA. Everything below was **written and type-checked** on macOS (clang's
CUDA front end with a stub runtime header, host and device passes), but **never compiled
by nvcc and never executed**. Until `scripts/gpu_check.sh` passes on a real GPU, none of
it counts as done, and `docs/EVIDENCE.md` makes no GPU claim.

Command: `PS26119_MACHINE=<name> scripts/gpu_check.sh` (add `QUICK=1` for the first try).

**Where to run it.** Linux, or **WSL2 (Ubuntu) on a Windows laptop** with the NVIDIA Windows
driver and the CUDA toolkit for WSL installed inside Ubuntu. nvcc does not support MinGW;
native Windows needs MSVC + the CUDA toolkit, which this script does not drive. On WSL2:
`sudo apt install build-essential git python3-pip`, install the CUDA toolkit for WSL-Ubuntu
from NVIDIA, `export PATH=/usr/local/cuda/bin:$PATH`, then run the command above from a clone
of the canonical repository (a feature branch such as `feature/shivanshu/gpu-validation`).
The script also runs `compute-sanitizer` (memcheck + racecheck) on the GPU tests when the
toolkit provides it; its logs go to `bench/results/logs/`.

## A. Build (step "configure + build" in gpu_check.sh)
| # | Check | Why it can fail | Pass criterion |
|---|---|---|---|
| A1 | CMake finds CUDA, `enable_language(CUDA)` works, `CMAKE_CUDA_ARCHITECTURES=native` | old CMake (<3.24 → fallback list 75;80;86;89), nvcc/g++ version mismatch | configure succeeds |
| A2 | `src/gpu/cuda_backend.cu` compiles with real nvcc (C++17) | nvcc is stricter than clang in places: `__shfl_down_sync` overloads, `std::` math in device code, lambdas + templates inside `with(...)`, `#pragma omp` in headers | no errors; warnings reported |
| A3 | Library + tests link with `CUDA::cudart` | missing runtime library path | `ps26119` and `ps26119_tests` build |

## B. Correctness of every kernel against the CPU reference (`Gpu.BackendOpsMatchCpu`)
Same sequence of backend calls on the CPU and CUDA backends (afiro, share2b, stocfor1;
fp64 and mixed): 20 reflected-Halpern steps, then primal/dual steps, axpby, dot, norms, KKT.
| # | Kernel | Risk | Tolerance |
|---|---|---|---|
| B1 | `k_spmv_subwarp<TPR>` (4/8/16/32 threads per row) | shuffle width, lane/row mapping, tail rows | fp64 1e-12, fp32 1e-4 (relative) |
| B2 | `k_spmv_block` (rows > 1024 nnz) | Netlib/refinery have no long rows, so `Gpu.LongRowKernelMatchesCpu` builds a model with a 3000-entry row and a dense column | same |
| B3 | `k_primal_step`, `k_dual_step` (projections, ±inf bounds in fp32) | inf handling in float | same |
| B4 | `k_r2h_primal`, `k_r2h_dual` (fused Halpern + reflection) | aliasing of in/out buffers, optional ȳ | same |
| B5 | `k_dot_partial` + `k_reduce_final` (fixed-grid, deterministic) | shared memory reuse between repeated `block_sum` calls | 1e-12 |
| B6 | KKT on device (`k_unscale`, `k_kkt_rows`, `k_kkt_cols`) | max/sum mix-up, partial buffer offsets | ≤1e-9 relative vs CPU |
| B7 | precision switch fp32 ↔ fp64 (`k_convert`) | vector ordering | covered by mixed runs |

## C. End-to-end answers (`Gpu.EnginesMatchCpuOnSmallNetlib`, `netlib_small.py --gpu`)
| # | Check | Pass criterion |
|---|---|---|
| C1 | PDLP and r²HPDHG on GPU, fp64 and mixed, 10 small Netlib LPs | status Optimal, objective within 1e-6 of CPU, verifier-grade KKT ≤ 1e-6 |
| C2 | Independent verifier on the GPU solutions | `netlib-small-gpu-<hash>.csv`: 40/40 PASS |
| C3 | fp64 GPU follows the CPU trajectory | iteration counts within 25% (+256) of CPU (reduction order differs, so not identical) |
| C4 | Determinism: run the GPU bench twice | identical iteration counts and objectives (fixed-grid reductions) — compare two CSVs |

## D. Performance (only meaningful on a GPU; these are the numbers for the PPT)
| # | Measurement | Where |
|---|---|---|
| D1 | ms per iteration, CPU vs GPU, fp64 vs mixed, random LPs 1e4 / 1e5 / 1e6 rows | `scale-<machine>-<hash>.csv` |
| D2 | wall time to 1e-4 and 1e-8, same grid | same |
| D3 | refinery LP T = 12 / 365 / 8760 on GPU (hourly year: 429k rows, 517k cols) | same |
| D4 | the crossover size where the GPU starts to win (expected: GPU LOSES on small models — must be reported) | GPU-vs-CPU summary printed by `scale.py` |
| D5 | mixed precision gain on a consumer card (fp64 is 1/32–1/64 rate on GeForce) vs a datacentre card | laptop vs university CSVs |
| D6 | GPU memory headroom for 1e6 rows (fp64 + fp32 copies of Ã, Ãᵀ, original A) | `nvidia-smi` during the run; OOM → NotSolved "out of memory" (never a crash) |

## E. Known design limits to look at once real numbers exist
- Kernel launch overhead: ~6 launches per iteration, no CUDA Graphs yet (cuPDLPx uses
  them). On small models this dominates; on 1e6 rows it should not.
- fp32 SpMV accumulates in fp32 for short rows (fp64 for long rows); if mixed precision
  stalls earlier on the GPU than on the CPU, this is the first suspect.
- Host syncs happen only in reductions (every K = 64 iterations); K may need to be larger
  on GPU (cuPDLPx uses 200) — tune from D1/D2.
- Our own SpMV vs cuSPARSE was never measured; do it only as a measurement, never as a
  dependency for the "sovereign" story.
