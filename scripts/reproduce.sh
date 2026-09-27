#!/usr/bin/env bash
# reproduce.sh — rebuild, test, and regenerate the CPU evidence on this machine, all from ONE
# commit (every CSV is named after the binary's git hash; docs/EVIDENCE.md is generated from them).
#
#   PS26119_MACHINE=my-laptop scripts/reproduce.sh               # all stages (~2-3 h on an M4)
#   STAGES="test netlib_small infeasible" scripts/reproduce.sh   # a subset
#
# Stages: test netlib_small netlib_full infeasible miplib scale warm batch
#   test          configure + build (Release) + ctest + no-foreign-solver check
#   netlib_small  10 small Netlib LPs x every engine x fp64/mixed          (~1 min)
#   netlib_full   all 93 Netlib LPs: auto, simplex, r2hpdhg, 60 s each     (~40 min)
#   infeasible    93 Netlib LPs + objective cut: certified Infeasible      (~50 min)
#   miplib        small MIPLIB 3, 300 s each                               (~20 min)
#   scale         generated LPs 1e4-1e6 rows + refinery T=12/365/8760,
#                 1 thread and all cores                                   (~60 min; SCALE_SIZES)
#   warm, batch   refinery what-if re-solves / batched scenarios          (~10 min)
# Data: tools/fetch_netlib.py and tools/fetch_miplib3.py are run only if the data is missing
# (they download public test sets). GPU machines: scripts/gpu_check.sh (it also runs the CPU side).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
STAGES=${STAGES:-"test netlib_small netlib_full infeasible miplib scale warm batch"}
has() { [[ " $STAGES " == *" $1 "* ]]; }
: "${PS26119_MACHINE:?set PS26119_MACHINE=<short machine name> (it goes into every CSV name)}"
export PS26119_MACHINE

$PY -c "import highspy, numpy, scipy, matplotlib" 2>/dev/null ||
  $PY -m pip install --user --quiet cmake ninja highspy numpy scipy matplotlib
export PATH="$($PY -m site --user-base)/bin:$PATH"

if has test; then
  GEN="-G Ninja"; command -v ninja >/dev/null || GEN=""
  cmake -S . -B build $GEN -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v $PY)"
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  scripts/check_no_solver_linked.sh build
fi

# Freeze the binary: every stage uses the same executable, even if the tree is rebuilt meanwhile.
FROZEN=$(mktemp -d)/ps26119
cp build/ps26119 "$FROZEN"
echo "binary: $("$FROZEN" --version)"
case "$("$FROZEN" --version)" in *-dirty*) echo "WARNING: built from a tree with uncommitted changes (CSV names will say -dirty)";; esac

if has netlib_full || has infeasible; then [ -f data/netlib/optima.csv ] || $PY tools/fetch_netlib.py; fi
if has miplib; then [ -f data/miplib3/optima.csv ] || $PY tools/fetch_miplib3.py; fi

has netlib_small && $PY bench/netlib_small.py --bin "$FROZEN"
if has netlib_full; then
  for e in auto simplex r2hpdhg; do $PY bench/netlib_full.py --bin "$FROZEN" --engine "$e" --time-limit 60; done
fi
has infeasible && $PY bench/netlib_infeasible_cut.py --bin "$FROZEN" --engines simplex,r2hpdhg --time-limit 60
has miplib && $PY bench/miplib3.py --bin "$FROZEN" --time-limit 300
has scale && $PY bench/scale.py --bin "$FROZEN" --sizes "${SCALE_SIZES:-1e4,1e5,1e6}" \
  --refinery "${REFINERY_T:-12,365,8760}" --threads 1,0 --time-limit "${SCALE_TIME_LIMIT:-600}" ${WITH_HIGHS:+--highs}
has warm && $PY bench/warm_start.py --bin "$FROZEN"
has batch && $PY bench/batch.py --bin "$FROZEN"

$PY bench/validate_results.py --csv-only
echo "Done. Commit bench/results/*.csv, then: $PY bench/make_evidence.py"
