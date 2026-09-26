#!/usr/bin/env bash
# check_no_solver_linked.sh — CLAUDE.md §4.5: nothing from any other solver is ever linked
# into our binaries, and src/ + include/ never include another solver's headers or highspy.
# Usage: scripts/check_no_solver_linked.sh [build-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
PATTERN='highs|cplex|gurobi|xpress|scip|soplex|glpk|clp|cbc|coin|mosek|copt|cuopt|pdlp|ortools|or-tools|lpsolve'
fail=0

# 1. Source: no foreign solver headers / imports in the solver core.
if grep -rEin "#include *[<\"]($PATTERN)|import +highspy" src include apps 2>/dev/null; then
  echo "FAIL: solver-core source references another solver"; fail=1
fi

# 1b. The Python package is a ctypes binding over OUR library: no solver or modelling
#     package may be imported there (highspy/scipy are allowed only in tools/, bench/, tests).
if grep -rEn "^\s*(import|from) +(highspy|scipy|pulp|cvxpy|pyomo|gurobipy|cplex|ortools|mosek|xpress)" python/ps26119 2>/dev/null; then
  echo "FAIL: python/ps26119 imports a solver/modelling package"; fail=1
fi

# 1c. The build never looks for a solver library.
if grep -rEin "find_(package|library)\s*\(\s*($PATTERN)" CMakeLists.txt cmake tests/CMakeLists.txt 2>/dev/null; then
  echo "FAIL: CMake looks for a solver library"; fail=1
fi

# 2. Binaries: linked libraries and symbols (CLI, static core, shared library for Python/C).
for bin in "$BUILD/ps26119" "$BUILD/libps26119_core.a" "$BUILD/libps26119.dylib" "$BUILD/libps26119.so"; do
  [ -e "$bin" ] || continue
  if [[ "$bin" == *.a ]]; then deps=""; else
    if command -v otool >/dev/null; then deps=$(otool -L "$bin"); else deps=$(ldd "$bin" || true); fi
  fi
  if echo "$deps" | grep -Eiq "lib($PATTERN)"; then
    echo "FAIL: $bin links a solver library:"; echo "$deps"; fail=1
  fi
  # Undefined symbols from a foreign solver API would show up here.
  if nm "$bin" 2>/dev/null | grep -Eq ' U _?(Highs_|CPX|GRB|XPRS|SCIP|glp_|Clp_|Cbc_)'; then
    echo "FAIL: $bin references foreign solver symbols"; fail=1
  fi
done
[ $fail -eq 0 ] && echo "OK: no foreign solver linked or included"
exit $fail
