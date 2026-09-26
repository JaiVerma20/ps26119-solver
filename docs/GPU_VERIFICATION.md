# What must be verified on a GPU (cannot be checked on the MacBook)

The Mac has no CUDA. Everything below was written and type-checked on macOS and is
verified on NVIDIA hardware by the GPU machines.

## Status log

| Date | Machine | What ran | Result |
|---|---|---|---|
| 2026-09-27 | teammate's NVIDIA laptop, WSL2 Ubuntu (commit `82d376c`) | full `ctest` of a CUDA build | **A1–A3 pass** (nvcc build, link). 167/168 tests pass: **C1 pass** (`Gpu.EnginesMatchCpuOnSmallNetlib`: PDLP + r²HPDHG × fp64 + mixed × 10 Netlib LPs, all Optimal, equal to CPU within 1e-6, verifier-grade KKT), **B2 pass** (`Gpu.LongRowKernelMatchesCpu`), **B1/B3–B5 vectors pass** (fp64 1e-12, fp32 1e-4). `Gpu.BackendOpsMatchCpu` failed only on its mixed-precision KKT comparisons (≤ 1.6e-6 relative, compared at an fp64 tolerance of 1e-9) — a test bug, fixed by a precision-aware tolerance. |
| pending | same | `scripts/gpu_check.sh` (compute-sanitizer, GPU CSVs: small Netlib, scaling, refinery) | needed before any GPU speed claim (D1–D6), plus C2–C4 |

Command: `PS26119_MACHINE=<name> scripts/gpu_check.sh` (add `QUICK=1` for the first try).

**Where to run it.** Linux, or **WSL2 (Ubuntu) on a Windows laptop** with the NVIDIA Windows
driver and the CUDA toolkit for WSL installed inside Ubuntu. nvcc does not support MinGW;
native Windows needs MSVC + the CUDA toolkit, which this script does not drive.

**Quickstart (WSL2 Ubuntu, first run):**
```bash
sudo apt update && sudo apt install -y build-essential git python3-venv
# CUDA toolkit for WSL-Ubuntu: https://developer.nvidia.com/cuda-downloads (Linux → x86_64 → WSL-Ubuntu)
export PATH=/usr/local/cuda/bin:$PATH
nvidia-smi && nvcc --version                       # both must work before going on
git clone https://github.com/JaiVerma20/ps26119-solver.git && cd ps26119-solver
git switch -c feature/shivanshu/gpu-validation
PS26119_MACHINE=<laptop-gpu-name> QUICK=1 scripts/gpu_check.sh     # build + all tests + small Netlib
PS26119_MACHINE=<laptop-gpu-name> scripts/gpu_check.sh             # + scaling / refinery benchmarks
git add bench/results && git commit -m "bench: <laptop-gpu-name> GPU run" && git push -u origin feature/shivanshu/gpu-validation
```
Then open a pull request. If the build or tests fail, still commit and push
`bench/results/logs/<machine>-<hash>/` (build output, `ctest.log`, sanitizer logs,
`nvidia-smi.txt`) — those logs are exactly what is needed to fix the CUDA code.
The script puts its Python tools (cmake, ninja, highspy, numpy, scipy, matplotlib) in a
private venv `.venv-gpu/` (Ubuntu refuses `pip install --user`), keeps going after test
failures to collect every log, and runs `compute-sanitizer` (memcheck + racecheck) on the GPU
tests when the toolkit provides it; its logs go to `bench/results/logs/`.

Before a GPU run, `tools/check_cuda_syntax.sh` type-checks `src/gpu/*.cu` on any machine with
clang (host and device passes, stub header in `tools/cuda_stub/`; CI runs it). It catches compile
errors early but proves nothing about nvcc or the hardware.

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
| D6 | GPU memory headroom for 1e6 rows (fp64 + fp32 copies of Ã, Ãᵀ, original A) | `nvidia-smi` during the run; OOM → NotSolved "out of memory" (never a crash): `cudaErrorMemoryAllocation` is turned into `std::bad_alloc` (since 2026-09-27); try a model larger than the card's memory and check the status line |

## E. Known design limits to look at once real numbers exist
- Kernel launch overhead: ~6 launches per iteration, no CUDA Graphs yet (cuPDLPx uses
  them). On small models this dominates; on 1e6 rows it should not.
- fp32 SpMV accumulates in fp32 for short rows (fp64 for long rows); if mixed precision
  stalls earlier on the GPU than on the CPU, this is the first suspect.
- Host syncs happen only in reductions (every K = 64 iterations); K may need to be larger
  on GPU (cuPDLPx uses 200) — tune from D1/D2.
- Our own SpMV vs cuSPARSE was never measured; do it only as a measurement, never as a
  dependency for the "sovereign" story.
