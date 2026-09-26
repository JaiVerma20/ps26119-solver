#!/usr/bin/env bash
# check_cuda_syntax.sh — syntax/type-check src/gpu/*.cu WITHOUT the CUDA toolkit (e.g. on macOS),
# using clang's CUDA front end with the minimal stand-in header in tools/cuda_stub/. Catches
# C++/CUDA compile errors before a GPU run; it proves nothing about correctness or about nvcc
# (which is stricter in places) — the real check is scripts/gpu_check.sh on NVIDIA hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX=${CXX:-clang++}
for f in src/gpu/*.cu; do
  for pass in --cuda-host-only --cuda-device-only; do
    "$CXX" -x cuda -std=c++17 -fsyntax-only -nocudainc -nocudalib $pass --cuda-gpu-arch=sm_80 \
      -Itools/cuda_stub -Iinclude -Isrc -DPS26119_HAVE_CUDA=1 "$f"
  done
  echo "ok: $f (host + device passes)"
done
