# Development

## Requirements

| Tool | Version | Notes |
|---|---|---|
| C++ compiler | Apple clang ≥ 15, GCC ≥ 11, Clang ≥ 14 (C++20) | MSVC is untested |
| CMake / Ninja | ≥ 3.20 | `python3 -m pip install cmake ninja` works everywhere |
| Python | 3.9+ with `highspy numpy scipy matplotlib` | tooling and tests only — never linked |
| CUDA toolkit | ≥ 11.6 (for `compute-sanitizer`) | GPU machines only (Linux or WSL2) |

GoogleTest is fetched by CMake. The solver itself has no third-party dependency.

## Build and test

```bash
python3 -m pip install --user cmake ninja highspy numpy scipy matplotlib
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3)
cmake --build build -j
ctest --test-dir build --output-on-failure
scripts/check_no_solver_linked.sh build
```

If CMake warns "Python tests SKIPPED", point `Python3_EXECUTABLE` at the interpreter that has
highspy/numpy/scipy (on the MacBook: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`).
CI passes `-DPS26119_REQUIRE_PYTHON_TESTS=ON`, which turns that warning into an error.

Sanitizers (what CI runs):

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j && ctest --test-dir build-asan -E "python|tools|cli|bench"
```
(`-fsanitize=thread` the same way for TSan.)

## Machines

| Machine | Role | Notes |
|---|---|---|
| MacBook Air M4 | CPU development, CPU benchmarks | `long double == double`; no floating `std::from_chars` in Apple libc++ (the reader has a portable parser); pip cmake/ninja live in `~/Library/Python/3.11/bin` |
| Teammate laptop (NVIDIA, Windows) | CUDA build + GPU tests | use **WSL2 Ubuntu** + CUDA toolkit for WSL (nvcc does not support MinGW); `scripts/gpu_check.sh` |
| University GPU servers | published GPU benchmarks | `PS26119_MACHINE=<name> scripts/gpu_check.sh` |
| GitHub Actions | CPU CI on every push | Ubuntu + macOS, sanitizers, provenance check |

## Using it

```bash
build/ps26119 info  data/netlib_small/afiro.mps          # statistics, fingerprint
build/ps26119 print data/examples/features.mps           # the model as the reader understood it
build/ps26119 solve data/netlib_small/afiro.mps --out afiro.sol          # auto engine
build/ps26119 solve model.mps --algorithm simplex|r2hpdhg|pdlp|oracle
python3 tools/verify.py data/netlib_small/afiro.mps afiro.sol --expected -464.75314286
build/ps26119 lu-bench data/netlib_small/afiro.mps       # sparse LU on bases built from A
```
Python: `import ps26119; ps26119.solve_mps("model.mps", algorithm="simplex", out="m.sol")`.

## Data

`python3 tools/fetch_netlib.py` (93 Netlib LPs + HiGHS reference optima into `data/netlib/`),
`python3 tools/fetch_miplib3.py` (small MIPLIB 3). Both directories are git-ignored; sources
are recorded in `data/SOURCES.md`.
