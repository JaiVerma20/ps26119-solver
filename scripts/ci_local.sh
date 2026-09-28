#!/usr/bin/env bash
# ci_local.sh — run the jobs of .github/workflows/ci.yml on this machine (for when GitHub Actions
# is unavailable, or before pushing). Separate build dirs: build-ci, build-asan-ubsan, build-tsan.
# Usage: scripts/ci_local.sh [--no-sanitizers]      Exit code: number of failed stages.
# Keep in step with ci.yml (same flags, same ctest filters, same checks).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3)}"
fails=0
stage() { echo; echo "=== $1"; }
run() { if "$@"; then echo "PASS: $*"; else echo "FAIL: $*"; fails=$((fails + 1)); fi; }

stage "build-test (Release, Python tests required)"
run cmake -S . -B build-ci -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PY" -DPS26119_REQUIRE_PYTHON_TESTS=ON
run cmake --build build-ci -j
run ctest --test-dir build-ci --output-on-failure
run scripts/check_no_solver_linked.sh build-ci
run "$PY" bench/validate_results.py --csv-only

if [ "${1:-}" != "--no-sanitizers" ]; then
  for name in asan-ubsan tsan; do
    if [ "$name" = asan-ubsan ]; then flags="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"
    else flags="-fsanitize=thread"; fi
    stage "$name"
    run cmake -S . -B "build-$name" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="$flags" -DCMAKE_C_FLAGS="$flags" -DCMAKE_EXE_LINKER_FLAGS="$flags" \
      -DCMAKE_SHARED_LINKER_FLAGS="$flags" -DPython3_EXECUTABLE="$PY"
    run cmake --build "build-$name" -j
    run ctest --test-dir "build-$name" --output-on-failure -E "python|tools|cli|bench"
  done
  stage "CUDA sources type-check (clang CUDA front end)"
  run tools/check_cuda_syntax.sh
fi

stage "summary"
echo "commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (uncommitted changes)') · $(uname -sm) · failed stages: $fails"
exit "$fails"
