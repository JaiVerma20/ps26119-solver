#!/usr/bin/env bash
# reproduce.sh — rebuild everything and regenerate the CPU evidence on this machine.
#   PS26119_MACHINE=my-laptop scripts/reproduce.sh            (≈ minutes; SCALE_SIZES to shrink)
# GPU machines: use scripts/gpu_check.sh instead (it also runs the CPU side).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -c "import highspy, numpy, matplotlib" 2>/dev/null || $PY -m pip install --user --quiet cmake ninja highspy numpy matplotlib
export PATH="$($PY -m site --user-base)/bin:$PATH"
GEN="-G Ninja"; command -v ninja >/dev/null || GEN=""
cmake -S . -B build $GEN -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v $PY)"
cmake --build build -j
ctest --test-dir build --output-on-failure
scripts/check_no_solver_linked.sh build
$PY bench/netlib_small.py
$PY bench/scale.py --sizes "${SCALE_SIZES:-1e4,1e5,1e6}" --refinery "${REFINERY_T:-12,365,8760}" \
    --time-limit "${SCALE_TIME_LIMIT:-600}" ${WITH_HIGHS:+--highs}
echo "Commit bench/results/, then: $PY bench/make_evidence.py"
