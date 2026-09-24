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
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv || {
  echo "ERROR: nvidia-smi not found — is this a CUDA machine?"; exit 1; }
command -v nvcc >/dev/null && nvcc --version | tail -2 || { echo "ERROR: nvcc not found (add /usr/local/cuda/bin to PATH)"; exit 1; }
nvidia-smi -q > "$LOGDIR/nvidia-smi.txt" 2>&1 || true
lscpu > "$LOGDIR/lscpu.txt" 2>/dev/null || true

echo "=== python tools (cmake, ninja, highspy, numpy, matplotlib)"
PY=${PYTHON:-python3}
$PY -c "import highspy, numpy, matplotlib" 2>/dev/null || $PY -m pip install --user --quiet cmake ninja highspy numpy matplotlib
export PATH="$($PY -m site --user-base)/bin:$PATH"
GEN="-G Ninja"; command -v ninja >/dev/null || GEN=""

echo "=== configure + build (CUDA ON)"
cmake -S . -B "$BUILD" $GEN -DCMAKE_BUILD_TYPE=Release -DPS26119_ENABLE_CUDA=ON \
      -DPython3_EXECUTABLE="$(command -v $PY)" ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH}
cmake --build "$BUILD" -j

echo "=== tests (CPU + GPU)"
ctest --test-dir "$BUILD" --output-on-failure | tee "$LOGDIR/ctest.log"
scripts/check_no_solver_linked.sh "$BUILD"

echo "=== bench: small Netlib (CPU and GPU, fp64 + mixed, verified)"
$PY bench/netlib_small.py --bin "$BUILD/ps26119" --engines oracle,pdlp,r2hpdhg | tail -2
$PY bench/netlib_small.py --bin "$BUILD/ps26119" --engines pdlp,r2hpdhg --gpu | tail -2

if [ -z "${QUICK:-}" ] && [ -f bench/scale.py ]; then
  echo "=== bench: scaling (generated LPs + refinery), CPU vs GPU, fp64 vs mixed"
  $PY bench/scale.py --bin "$BUILD/ps26119" --gpu ${SCALE_SIZES:+--sizes $SCALE_SIZES} \
      ${REFINERY_T:+--refinery $REFINERY_T} ${SCALE_TIME_LIMIT:+--time-limit $SCALE_TIME_LIMIT}
fi

echo "=== done. New result files:"
ls -1t bench/results/*.csv bench/results/*.png 2>/dev/null | head -10
echo "Commit them:  git add bench/results && git commit -m \"bench: ${PS26119_MACHINE} @ ${HASH}\""
