#!/usr/bin/env bash
# check_cuda_syntax.sh — syntax/type-check src/gpu/*.cu WITHOUT the CUDA toolkit (e.g. on macOS),
# using clang's CUDA front end with the minimal stand-in header in tools/cuda_stub/. Catches
# C++/CUDA compile errors before a GPU run; it proves nothing about correctness or about nvcc
# (which is stricter in places) — the real check is scripts/gpu_check.sh on NVIDIA hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX=${CXX:-clang++}
# -include cstdlib: clang's cuda_wrappers/new (used for <new> in CUDA mode) needs ::malloc/::free
# declared first; clang 18 on Ubuntu does not get them transitively.
# Device pass: -U__FLOAT128__/__SIZEOF_FLOAT128__ — GCC 14's libstdc++ declares
# numeric_limits<__float128> whenever the (host) target advertises __float128, which the sm_80
# device target cannot compile (nvcc handles this itself).
for f in src/gpu/*.cu; do
  for pass in --cuda-host-only --cuda-device-only; do
    extra=()
    [ "$pass" = --cuda-device-only ] && extra=(-U__FLOAT128__ -U__SIZEOF_FLOAT128__)
    "$CXX" -x cuda -std=c++17 -fsyntax-only -nocudainc -nocudalib $pass --cuda-gpu-arch=sm_80 \
      -include cstdlib -Itools/cuda_stub -Iinclude -Isrc -DPS26119_HAVE_CUDA=1 ${extra[@]+"${extra[@]}"} "$f"
  done
  echo "ok: $f (host + device passes)"
done
