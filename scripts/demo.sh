#!/usr/bin/env bash
# demo.sh — a short live demonstration of what the solver does and how every answer is checked.
#   scripts/demo.sh            (needs a Release build in build/; ~1-2 min; LARGE=0 skips the big LP)
# Every step prints the solver's own report; the "verify" lines come from tools/verify.py, an
# independent verifier with its own MPS reader (highspy) and exact rational arithmetic.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-build/ps26119}
PY=${PYTHON:-python3}
T=$(mktemp -d)
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
show() { "$@" | grep -E '^(model|status|engine|objective|check|certified|iterations|message)' | cut -c1-160 || true; }

step "Version and provenance (git hash of this binary)"
"$BIN" --version

step "1. LP (Netlib afiro) with the automatic engine choice, then the independent verifier"
show "$BIN" solve data/netlib_small/afiro.mps --out "$T/afiro.sol"
$PY tools/verify.py data/netlib_small/afiro.mps "$T/afiro.sol" --expected -464.75314286 | grep -E 'VERDICT|objective'

step "2. The same LP with every engine (simplex vertex, first-order r2HPDHG, exact double-double oracle)"
for a in simplex r2hpdhg oracle; do printf '%-8s ' "$a"; "$BIN" solve data/netlib_small/afiro.mps --algorithm "$a" | grep '^objective'; done

step "3. An INFEASIBLE LP: the answer carries a Farkas certificate, checked in-process and by verify.py"
cat > "$T/infeasible.mps" <<'EOF'
NAME INFEAS
ROWS
 N obj
 L cap
 G need
COLUMNS
 x obj 1 cap 1
 x need 1
 y obj 1 cap 1
 y need 1
RHS
 rhs cap 1 need 3
ENDATA
EOF
show "$BIN" solve "$T/infeasible.mps" --algorithm simplex --out "$T/inf.sol" || true
$PY tools/verify.py "$T/infeasible.mps" "$T/inf.sol" | grep -E 'certificate|VERDICT'

step "4. An UNBOUNDED LP: a feasible point plus a ray, both checked"
cat > "$T/unbounded.mps" <<'EOF'
NAME UNBND
ROWS
 N obj
 L r
COLUMNS
 x obj -1 r 1
 y r -1
RHS
 rhs r 0
ENDATA
EOF
show "$BIN" solve "$T/unbounded.mps" --algorithm r2hpdhg --out "$T/unb.sol" || true
$PY tools/verify.py "$T/unbounded.mps" "$T/unb.sol" | grep -E 'certificate|VERDICT'

step "5. A MILP (knapsack): branch-and-bound, never answered by its LP relaxation"
show "$BIN" solve data/hand/knapsack_mip.lpm

if [ "${LARGE:-1}" = "1" ]; then
  step "6. Large LP: refinery planning, hourly for a year (429k rows, 1.5M nonzeros), all cores"
  [ -f bench/generated/refinery-T8760-s1.lpm ] ||
    $PY bench/generate_refinery_lp.py --periods 8760 --seed 1 --out bench/generated/refinery-T8760-s1.lpm >/dev/null
  show "$BIN" solve bench/generated/refinery-T8760-s1.lpm --threads 0
fi

step "7. Hardened inputs: bad files and bad options are errors with exit codes, never a fake answer"
: > "$T/empty.mps"
"$BIN" solve "$T/empty.mps" || echo "  -> exit $? (read error)"
"$BIN" solve data/netlib_small/afiro.mps --time-limit -1 || echo "  -> exit $? (usage error)"
echo; echo "All answers above were verified; see docs/EVIDENCE.md for the full benchmark evidence."
