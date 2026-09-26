# GPU validation — RTX 4050 Laptop, commit 82d376c

First run of the CUDA backend on NVIDIA hardware. No source code was changed; this is
`scripts/gpu_check.sh` exactly as committed at 82d376c (full run, default sizes).

## Machine
| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, max SM clock 3105 MHz (sm_89) |
| Driver | 616.64 (Windows), CUDA 13.4 driver API (`nvidia-smi` inside WSL: 615.65.07 / KMD 616.64) |
| nvcc | 13.3.73 (`cuda-toolkit` from NVIDIA's wsl-ubuntu apt repo) |
| Host | Intel Core 5 210H, WSL2 Ubuntu 24.04.4, kernel 6.6.87.2-microsoft-standard-WSL2, g++ 13.3.0, 7.7 GB RAM in WSL |
| Details | `nvidia-smi.txt`, `lscpu.txt` |

## A. Build
- `-DPS26119_ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=native`: configure, compile of
  `src/gpu/cuda_backend.cu` and link all succeed. **No nvcc errors or warnings.**
- Only compiler warning in the whole build (host code):
  `src/pdhg/cpu_backend.cpp:292:66: warning: unused parameter 'error'`.

## B. Tests — 167/168 pass (identical in all three runs)
The only failure is **`Gpu.BackendOpsMatchCpu`, mixed precision only** (afiro, share2b, stocfor1):
- fp64: every vector within 1e-12 and every KKT statistic within 1e-9 of the CPU backend — pass.
- mixed: all vector comparisons pass (tol 1e-4). The **6 KKT `EXPECT_NEAR`s fail** because they use
  the fp64 tolerance 1e-9 for both precisions. Observed relative differences are ~2e-7–3e-7
  (fp32 epsilon ≈ 1.2e-7), e.g. afiro mixed: primal_residual 235.49995910 vs 235.49991716
  (4.2e-5 abs), dual_obj −1173.93919 vs −1173.93955. Looks like a test-tolerance issue for mixed
  precision rather than a kernel bug — see `ctest.log` for all 18 assertions.
- `Gpu.EnginesMatchCpuOnSmallNetlib` and `Gpu.LongRowKernelMatchesCpu` pass.
- The 6 Python-dependent tests (tools.python, cli.contract, python.binding, bench.generators,
  tools.crosscheck.*) pass.

## compute-sanitizer — not usable under WSL2
Both memcheck and racecheck stop at startup with
`Failed to initialize WDDM debugger interface` / `Device not supported`, so the "ERRORS" lines
in the summary are this, not kernel findings (`sanitizer-*.log`). Needs native Linux.

## C. Small Netlib (`netlib-small-gpu-82d376c.csv`, `netlib-small-82d376c.csv`)
- GPU: **40/40 Optimal, verified PASS** (pdlp, r2hpdhg × fp64, mixed × 10 LPs).
- CPU: 60/60 verified PASS (oracle, simplex, pdlp, r2hpdhg).
- Max relative error vs published optimum on GPU: pdlp 2.2e-8, r2hpdhg 1.9e-8;
  GPU vs CPU objective within 3.3e-8.
- Iteration counts close to CPU (reduction order differs), e.g. share2b r2hpdhg fp64
  49 344 GPU vs 65 536 CPU-mixed.
- GPU is 50–100× slower on these tiny models (0.3–8 s vs ms) — launch/setup overhead, as expected (D4).

## D. Scaling (`scale-rtx4050-laptop-82d376c.csv` / `.png`), r2hpdhg, tol 1e-8, CPU 1 thread, time limit 600 s
| instance | rows | prec | CPU | GPU | GPU faster to 1e-8 | per iteration | error | verify |
|---|---|---|---|---|---|---|---|---|
| rand-1e4 | 10 000 | fp64 | 0.52 s | 0.57 s | 0.91× | 2.0× | 6.3e-10 | PASS |
| rand-1e4 | 10 000 | mixed | 0.58 s | 0.56 s | 1.03× | 2.7× | 1.8e-10 | PASS |
| rand-1e5 | 100 000 | fp64 | 12.03 s | **2.83 s** | **4.25×** | 5.3× | 2.4e-10 | PASS |
| rand-1e5 | 100 000 | mixed | 11.74 s | **3.77 s** | **3.12×** | 3.9× | 5.1e-11 | PASS |
| rand-1e6 | 1 000 000 | fp64 | TimeLimit 629 s | **Optimal 113.0 s** | CPU did not finish | **12.3×** | 1.4e-11 | skipped (size) |
| rand-1e6 | 1 000 000 | mixed | TimeLimit 628 s | **Optimal 108.6 s** | CPU did not finish | **11.4×** | 1.2e-11 | skipped (size) |
| refinery T=12 | 588 | fp64 | 0.01 s | 0.51 s | 0.02× | 0.1× | 3.2e-11 | PASS |
| refinery T=12 | 588 | mixed | 0.02 s | 0.43 s | 0.04× | 0.1× | 3.0e-10 | PASS |
| refinery T=365 | 17 885 | fp64 | 0.80 s | 0.80 s | 1.00× | 2.0× | 8.2e-12 | PASS |
| refinery T=365 | 17 885 | mixed | 1.18 s | 0.70 s | 1.68× | 2.6× | 8.5e-11 | PASS |
| refinery T=8760 | 429 240 | fp64 | 29.37 s | **9.58 s** | **3.07×** | 3.9× | 1.5e-13 | PASS |
| refinery T=8760 | 429 240 | mixed | 24.45 s | **9.26 s** | **2.64×** | 4.0× | 6.2e-14 | PASS |

- Crossover: GPU wins from ~1e4–2e4 rows; loses clearly below that (setup ≈0.3–0.4 s).
- GPU setup time: 0.3 s (small) → 2.6–2.9 s (refinery 8760) → 9.6–10.9 s (rand-1e6).
- Mixed precision gives no clear GPU gain on this card here (fp32 SpMV not the bottleneck at
  these sizes; mixed sometimes needs more iterations).
- Memory: 980 MiB of 6141 MiB in use during the rand-1e6 GPU fp64 solve (one `nvidia-smi`
  sample mid-solve, run 2; includes ~130 MiB used by Windows). No OOM in any run.
- Note: `scale.py`'s "GPU vs CPU" summary printed an empty table in this run (header only);
  the ratios above were computed from the CSV instead.

## Files
- `gpu_check.log` — run 3 (the complete run these numbers come from); `ctest.log`; `sanitizer-*.log`
- `console-run1-quick.txt` — first `QUICK=1` run (same 167/168, same 40/40)
- `console-run2-full-interrupted.txt` — second full run, stopped during the scaling bench when
  its working directory was deleted (not a solver problem); its 1e6 GPU fp64 result
  (103.7 s Optimal) matches run 3
- `console-run3-full.txt` — console of run 3
- `wsl_runner.sh` — how it was invoked: repo on the Windows drive, build dir and venv in the
  WSL home (`BUILD=`, `VENV=`), `GIT_DIR` set because the checkout is a Windows git worktree
