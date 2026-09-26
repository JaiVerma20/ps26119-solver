"""Reference solve with HiGHS (via SciPy) for comparison with gpuopt.

HiGHS is used ONLY as an external baseline, never linked into gpuopt. This
script has its own small MPS parser (free format, the Netlib subset: ROWS,
COLUMNS, RHS, RANGES, BOUNDS UP/LO/FX/FR/MI/PL, objective constant, OBJSENSE),
so the comparison does not depend on gpuopt's reader either.

usage:  python scripts/highs_reference.py data/netlib/80bau3b.mps [more.mps ...]
"""
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, vstack

INF = float("inf")


def read_mps(path):
    rows, row_type, obj_row = {}, [], None
    cols, obj, entries = {}, [], []
    rhs, rng, lo, up = {}, {}, {}, {}
    offset, maximize, section = 0.0, False, None
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip() or line.startswith("*"):
            continue
        t = line.split()
        if not line[0].isspace():
            section = t[0].upper()
            if section == "OBJSENSE" and len(t) > 1:
                maximize = t[1].upper().startswith("MAX")
            continue
        if section == "OBJSENSE":
            maximize = t[0].upper().startswith("MAX")
        elif section == "ROWS":
            if t[0] == "N":
                obj_row = obj_row or t[1]
                if t[1] != obj_row:
                    rows[t[1]] = -2  # extra free row: ignored
            else:
                rows[t[1]] = len(row_type)
                row_type.append(t[0])
        elif section == "COLUMNS":
            if "'MARKER'" in t:
                continue
            j = cols.setdefault(t[0], len(cols))
            if j == len(obj):
                obj.append(0.0)
            for r, v in zip(t[1::2], t[2::2]):
                if r == obj_row:
                    obj[j] += float(v)
                elif rows[r] >= 0:
                    entries.append((rows[r], j, float(v)))
        elif section in ("RHS", "RANGES"):
            pairs = t[1:] if len(t) % 2 == 1 else t
            for r, v in zip(pairs[0::2], pairs[1::2]):
                if r == obj_row:
                    if section == "RHS":
                        offset = -float(v)
                elif rows[r] >= 0:
                    (rhs if section == "RHS" else rng)[rows[r]] = float(v)
        elif section == "BOUNDS":
            kind, j = t[0], cols[t[2] if len(t) >= 3 and t[2] in cols else t[1]]
            v = float(t[-1]) if kind in ("UP", "LO", "FX") else None
            if kind == "UP":
                up[j] = v
                if v < 0 and j not in lo:
                    lo[j] = -INF
            elif kind == "LO":
                lo[j] = v
            elif kind == "FX":
                lo[j] = up[j] = v
            elif kind == "FR":
                lo[j], up[j] = -INF, INF
            elif kind == "MI":
                lo[j] = -INF
            elif kind == "PL":
                up[j] = INF

    m, n = len(row_type), len(cols)
    A = csc_matrix(([e[2] for e in entries], ([e[0] for e in entries], [e[1] for e in entries])), shape=(m, n))
    row_lo, row_up = np.full(m, -INF), np.full(m, INF)
    for i, ty in enumerate(row_type):
        b, R = rhs.get(i, 0.0), rng.get(i)
        if ty == "E":
            row_lo[i] = row_up[i] = b
            if R is not None:
                (row_up if R > 0 else row_lo)[i] = b + R
        elif ty == "L":
            row_up[i] = b
            if R is not None:
                row_lo[i] = b - abs(R)
        else:
            row_lo[i] = b
            if R is not None:
                row_up[i] = b + abs(R)
    bounds = [(lo.get(j, 0.0), up.get(j, INF)) for j in range(n)]
    return np.array(obj), A, row_lo, row_up, bounds, offset, maximize


def solve(path):
    c, A, rl, ru, bounds, offset, maximize = read_mps(path)
    eq = rl == ru
    ub_rows = ~eq & np.isfinite(ru)
    lb_rows = ~eq & np.isfinite(rl)
    A_ub = vstack([A[ub_rows], -A[lb_rows]]).tocsr() if (ub_rows.any() or lb_rows.any()) else None
    b_ub = np.concatenate([ru[ub_rows], -rl[lb_rows]]) if A_ub is not None else None
    cc = -c if maximize else c
    t0 = time.time()
    res = linprog(cc, A_ub=A_ub, b_ub=b_ub, A_eq=A[eq] if eq.any() else None, b_eq=rl[eq] if eq.any() else None,
                  bounds=[(None if l == -INF else l, None if u == INF else u) for l, u in bounds], method="highs")
    seconds = time.time() - t0
    if res.status != 0:
        return res.message, None, seconds
    value = (-res.fun if maximize else res.fun) + offset
    return "OPTIMAL", value, seconds


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for path in sys.argv[1:]:
        status, value, seconds = solve(path)
        print("%-40s %-10s %22s   %.3f s" % (path, status if value is not None else "FAILED",
                                            "%.10e" % value if value is not None else status, seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
