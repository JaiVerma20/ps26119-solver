# ps26119 — GPU-native LP solver (SIH 2026, PS 26119)

A from-scratch LP solver (MILP/QP to follow) built around **restarted Halpern PDHG with
reflection (r²HPDHG)** — the algorithm family of cuPDLPx — with a CPU reference
implementation of every GPU kernel, mixed precision (fp32 iterations, fp64 decisions), and
an exact double-double simplex as a test oracle. No existing solver is inside
(`scripts/check_no_solver_linked.sh` runs in CI).

**Evidence:** see [`docs/EVIDENCE.md`](docs/EVIDENCE.md) — generated only from committed CSVs.
**Design:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Benchmarks:** [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)

## Build and test (macOS arm64 / Linux)
```bash
python3 -m pip install --user cmake ninja highspy numpy matplotlib   # tooling only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build
```
CUDA (GPU machines): `PS26119_MACHINE=<name> scripts/gpu_check.sh`.

## Use
```bash
python3 tools/mps_to_lpm.py data/netlib_small/afiro.mps          # until the MPS reader lands
build/ps26119 solve data/netlib_small/afiro.lpm --algorithm r2hpdhg --tol 1e-8 --out afiro.sol
python3 tools/verify.py data/netlib_small/afiro.mps afiro.sol --expected -464.75314286
```
Algorithms: `oracle` (dense double-double simplex, small models), `pdlp` (restarted PDHG),
`r2hpdhg` (default). Options: `--precision fp64|mixed`, `--gpu`, `--time-limit`,
`--iteration-limit`. Exit codes: 0 optimal, 1 limit/infeasible/unbounded, 3 read error,
5 numerical.

## Status
Done: Model/Solution contracts, `.lpm` bridge, independent verifier, double-double oracle,
CPU PDLP-style PDHG and r²HPDHG (fp64 + mixed), CUDA backend (written; awaiting GPU runs),
generators with known optimum (random + multi-period refinery), benchmark harness.
Not yet: see "What we do NOT do yet" in `docs/EVIDENCE.md`.
