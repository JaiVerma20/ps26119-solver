# rtx4050-laptop @ 183c59c — GPU validation run 2

- Machine: NVIDIA GeForce RTX 4050 Laptop GPU, driver 616.64, CUDA 13.4 (WSL2 Ubuntu, 12 CPU threads)
- Command: `PS26119_MACHINE=rtx4050-laptop scripts/gpu_check.sh` (full, not QUICK), via `wsl_runner.sh`
- `scripts/gpu_check.sh` and the CUDA code unchanged; only results are committed.
- Full console output: `console-full.txt`

## Correctness
| Check | Result |
|---|---|
| A1–A3 build (CUDA ON) | pass |
| ctest | **198/198 pass** (`ctest.log`) |
| Small Netlib CPU | 60/60 verified PASS |
| Small Netlib GPU (C1/C2) | 40/40 verified PASS |
| C3 fp64 GPU vs CPU trajectory | scaling + refinery: GPU fp64 iteration counts **identical** to CPU fp64 on every instance |
| C4 determinism (two GPU runs) | not run separately in this commit |
| compute-sanitizer memcheck/racecheck | **NOT RUN** — WSL2 cannot attach ("Failed to initialize WDDM debugger interface", "Device not supported"). Kernels are NOT sanitizer-verified. `bench/validate_results.py` flags this folder as FAIL for that reason only. |

## Performance (r²HPDHG, wall time to 1e-8; `scale-rtx4050-laptop-183c59c.csv`)
| Instance | CPU 1 thr | CPU 12 thr | GPU fp64 | GPU mixed | GPU fp64 vs best CPU |
|---|---|---|---|---|---|
| rand 1e4 rows | 0.41 s | 0.71 s | 0.51 s | 0.48 s | 0.80× (GPU slower) |
| rand 1e5 rows | 9.46 s | 5.30 s | 1.79 s | 1.80 s | 2.96× |
| rand 1e6 rows | TimeLimit (600 s) | 489 s | 116 s | 106 s | 4.23× |
| refinery T=12 | 0.008 s | 0.009 s | 0.35 s | 0.38 s | 0.02× (GPU slower) |
| refinery T=365 | 0.65 s | 1.25 s | 0.60 s | 0.58 s | 1.10× |
| refinery T=8760 | 26.5 s | 18.7 s | 6.18 s | 5.78 s | 3.03× |

- D4 crossover: the GPU loses below ~1e4 rows / refinery T=12, is about even at T=365, and wins from 1e5 rows.
- D5 (consumer card): mixed precision gives little over fp64 on this GeForce (≤ ~10%).
- D6: 1e6 rows ran to Optimal on the 6 GB card in both precisions, no OOM.
- 1e6-row solutions were not independently verified (`verify skipped` for that size in `scale.py`).
