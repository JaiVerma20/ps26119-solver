#!/usr/bin/env bash
# gpu_check.sh — the ONE command the GPU machines run (CLAUDE.md §3).
#
#   PS26119_MACHINE=rtx4060-laptop scripts/gpu_check.sh          # full run
#   PS26119_MACHINE=uni-a100 SCALE_SIZES=1e4,1e5,1e6 scripts/gpu_check.sh
#   QUICK=1 scripts/gpu_check.sh                                 # build + tests + netlib only
#
# Steps: machine info -> configure with CUDA -> build -> ctest (CPU + GPU tests) ->
# no-foreign-solver check -> GPU benches (small Netlib, scaling, refinery) -> CSVs in
# bench/results/ with git hash + machine info, logs in bench/results/logs/.
# Paste the final summary back (or commit bench/results/*.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

HASH=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
export PS26119_MACHINE="${PS26119_MACHINE:-gpu-$(echo "${GPU_NAME:-unknown}" | tr ' A-Z' '-a-z' | tr -cd 'a-z0-9-')}"
BUILD="${BUILD:-build-gpu}"
LOGDIR="bench/results/logs/${PS26119_MACHINE}-${HASH}"
mkdir -p "$LOGDIR"
exec > >(tee "$LOGDIR/gpu_check.log") 2>&1

echo "=== ps26119 gpu_check  commit ${HASH}  machine ${PS26119_MACHINE}  $(date -Iseconds)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "WARNING: working tree has uncommitted changes; CSV names will carry '-dirty'."
fi

echo "=== machine"
uname -a
command -v g++ >/dev/null && g++ --version | head -1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv || {
  echo "ERROR: nvidia-smi not found — is this a CUDA machine?"; exit 1; }
command -v nvcc >/dev/null && nvcc --version | tail -2 || { echo "ERROR: nvcc not found (add /usr/local/cuda/bin to PATH)"; exit 1; }
nvidia-smi -q > "$LOGDIR/nvidia-smi.txt" 2>&1 || true
lscpu > "$LOGDIR/lscpu.txt" 2>/dev/null || true

echo "=== python tools (private venv: cmake, ninja, highspy, numpy, scipy, matplotlib)"
# A venv, not `pip install --user`: Ubuntu 23.04+ / WSL2 Ubuntu 24.04 refuse user installs into
# the system Python (PEP 668, "externally-managed-environment"). Tooling only; never linked.
SYSPY=${PYTHON:-python3}
VENV="${VENV:-.venv-gpu}"
if [ ! -x "$VENV/bin/python" ]; then
  $SYSPY -m venv "$VENV" || { echo "ERROR: '$SYSPY -m venv' failed — run: sudo apt install python3-venv"; exit 1; }
fi
PY="$ROOT/$VENV/bin/python"
$PY -c "import highspy, numpy, scipy, matplotlib, cmake, ninja" 2>/dev/null || \
  $PY -m pip install --quiet --upgrade pip cmake ninja highspy numpy scipy matplotlib
export PATH="$ROOT/$VENV/bin:$PATH"   # the venv's cmake (>= 3.24 for CUDA_ARCHITECTURES=native) and ninja
cmake --version | head -1
GEN="-G Ninja"; command -v ninja >/dev/null || GEN=""

echo "=== configure + build (CUDA ON)"
cmake -S . -B "$BUILD" $GEN -DCMAKE_BUILD_TYPE=Release -DPS26119_ENABLE_CUDA=ON \
      -DPython3_EXECUTABLE="$(command -v $PY)" ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH}
cmake --build "$BUILD" -j

echo "=== tests (CPU + GPU)"
# Keep going on failures: the sanitizer and benchmark logs below are the most useful
# diagnostics on a first GPU run. The summary at the end says whether tests failed.
TESTS_OK=1
ctest --test-dir "$BUILD" --output-on-failure | tee "$LOGDIR/ctest.log" || TESTS_OK=0
grep -q "100% tests passed" "$LOGDIR/ctest.log" || TESTS_OK=0
scripts/check_no_solver_linked.sh "$BUILD"

# Invalid device memory access, races and uninitialised reads in the kernels (Phase 16 of the
# integration plan). compute-sanitizer ships with the CUDA toolkit (>= 11.6).
if command -v compute-sanitizer >/dev/null; then
  echo "=== compute-sanitizer (memcheck, racecheck) on the GPU tests"
  for tool in memcheck racecheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 99 "$BUILD/tests/ps26119_tests" --gtest_filter='Gpu.*' \
      > "$LOGDIR/sanitizer-$tool.log" 2>&1 && echo "  $tool: clean" || echo "  $tool: ERRORS (see $LOGDIR/sanitizer-$tool.log)"
  done
else
  echo "=== compute-sanitizer not found: skipped (report this)"
fi

echo "=== bench: small Netlib (CPU and GPU, fp64 + mixed, verified)"
$PY bench/netlib_small.py --bin "$BUILD/ps26119" --engines oracle,simplex,pdlp,r2hpdhg | tail -2 || true
$PY bench/netlib_small.py --bin "$BUILD/ps26119" --engines pdlp,r2hpdhg --gpu | tail -2 || true

if [ -z "${QUICK:-}" ] && [ -f bench/scale.py ]; then
  echo "=== bench: scaling (generated LPs + refinery), CPU vs GPU, fp64 vs mixed"
  $PY bench/scale.py --bin "$BUILD/ps26119" --gpu ${SCALE_SIZES:+--sizes $SCALE_SIZES} \
      ${REFINERY_T:+--refinery $REFINERY_T} ${SCALE_TIME_LIMIT:+--time-limit $SCALE_TIME_LIMIT}
fi

echo "=== done."
[ "$TESTS_OK" = 1 ] && echo "TESTS: all passed" || echo "TESTS: FAILURES — see $LOGDIR/ctest.log (send the whole $LOGDIR folder)"
echo "New result files:"
ls -1t bench/results/*.csv bench/results/*.png 2>/dev/null | head -10
echo "Commit them:  git add bench/results && git commit -m \"bench: ${PS26119_MACHINE} @ ${HASH}\""
