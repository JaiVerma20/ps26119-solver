"""lpgen.py — shared machinery for LP generators whose optimum is known BY CONSTRUCTION.

Construction (KKT certificate first, data second). Given a matrix A (CSC), a point x*,
and for every row / column the side that should be active, we choose duals and derive the
rest so that (x*, y*, z*) satisfies the KKT conditions of

    min c_minᵀx  s.t.  rl ≤ A x ≤ ru,  l ≤ x ≤ u

exactly (up to floating-point rounding):
  * rows:   'eq'  → rl = ru = a_i := (A x*)_i,  y_i any sign
            'ge' active   → rl = a_i, ru = +inf,           y_i > 0
            'le' active   → ru = a_i, rl = -inf,           y_i < 0
            'ge'/'le' inactive → bound = a_i ∓ slack,       y_i = 0
            'range' inactive → [a_i − s1, a_i + s2],        y_i = 0
  * cols:   at lower  (x*_j = l_j)                      → z_j > 0
            at upper  (x*_j = u_j)                      → z_j < 0
            between   (l_j < x*_j < u_j)                → z_j = 0
  * c_min = Aᵀy* + z*.
Then x* is primal feasible, (y*, z*) dual feasible with the right signs, and complementary
slackness holds, so x* is optimal and the optimal value is c_minᵀx* (any other optimal
point has the same value). For a MAX model we emit c = −c_min with sense −1.

Also: a fast .lpm writer (same format as tools/lpm.py) and a sidecar JSON with the known
optimum and the generator parameters.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

INF = math.inf


def csc_from_triplets(m: int, n: int, rows, cols, vals):
    """Sum duplicates, sort by (col,row), drop zeros. Returns col_start, row_index, value."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=float)
    key = cols * m + rows
    order = np.argsort(key, kind="stable")
    key, vals = key[order], vals[order]
    uniq, start = np.unique(key, return_index=True)
    summed = np.add.reduceat(vals, start) if len(vals) else vals
    keep = summed != 0.0
    uniq, summed = uniq[keep], summed[keep]
    c = uniq // m
    r = uniq % m
    col_start = np.zeros(n + 1, dtype=np.int64)
    np.add.at(col_start, c + 1, 1)
    col_start = np.cumsum(col_start)
    return col_start, r.astype(np.int64), summed


def matvec(m, col_start, row_index, value, x):
    cidx = np.repeat(np.arange(len(col_start) - 1), np.diff(col_start))
    return np.bincount(row_index, weights=value * x[cidx], minlength=m)


def rmatvec(n, col_start, row_index, value, y):
    cidx = np.repeat(np.arange(n), np.diff(col_start))
    return np.bincount(cidx, weights=value * y[row_index], minlength=n)


def certify(rng, m, n, col_start, row_index, value, x, col_lower, col_upper, col_at, row_type, row_active,
            dual_scale=(0.5, 5.0), slack_scale=(0.1, 2.0), eq_snap_zero=True):
    """Build bounds, c and duals around x* (see module docstring).

    col_at:     int8 per column: -1 at lower (x = l), +1 at upper (x = u), 0 between.
                col_lower/col_upper must already be consistent with x and col_at.
    row_type:   array of 'eq' | 'ge' | 'le' | 'range'
    row_active: bool per row (ignored for 'eq'; 'range' rows are always inactive here)
    """
    a = matvec(m, col_start, row_index, value, x)
    lo_d, hi_d = dual_scale
    y = np.zeros(m)
    rl = np.full(m, -INF)
    ru = np.full(m, INF)
    s1 = rng.uniform(*slack_scale, m) * (1 + np.abs(a))
    s2 = rng.uniform(*slack_scale, m) * (1 + np.abs(a))
    mag = rng.uniform(lo_d, hi_d, m)

    eq = row_type == "eq"
    a_eq = np.where(eq_snap_zero & (np.abs(a) < 1e-12), 0.0, a)
    rl[eq] = a_eq[eq]
    ru[eq] = a_eq[eq]
    y[eq] = mag[eq] * rng.choice([-1.0, 1.0], eq.sum())

    ge = row_type == "ge"
    rl[ge & row_active] = a[ge & row_active]
    y[ge & row_active] = mag[ge & row_active]
    rl[ge & ~row_active] = (a - s1)[ge & ~row_active]

    le = row_type == "le"
    ru[le & row_active] = a[le & row_active]
    y[le & row_active] = -mag[le & row_active]
    ru[le & ~row_active] = (a + s2)[le & ~row_active]

    rg = row_type == "range"
    rl[rg] = (a - s1)[rg]
    ru[rg] = (a + s2)[rg]

    zmag = rng.uniform(lo_d, hi_d, n)
    z = np.where(col_at < 0, zmag, np.where(col_at > 0, -zmag, 0.0))
    c = rmatvec(n, col_start, row_index, value, y) + z
    opt = math.fsum((c * x).tolist())

    # self-check: dual objective equals primal objective (complementary slackness)
    def bterm(v, lo, up):
        t = np.where(v > 0, np.where(np.isfinite(lo), lo, 0.0) * v, np.where(np.isfinite(up), up, 0.0) * v)
        return math.fsum(t.tolist())

    dual = bterm(y, rl, ru) + bterm(z, col_lower, col_upper)
    gap = abs(opt - dual) / (1 + abs(opt))
    assert gap < 1e-9, f"construction gap {gap}"
    return {"c_min": c, "row_lower": rl, "row_upper": ru, "y": y, "z": z, "optimum_min": opt}


def _write_nums(out, arr, per_line=1):
    lst = arr.tolist()
    if not lst:
        return
    out.write("\n".join(map(repr, lst)))
    out.write("\n")


def write_lpm_fast(path, name, sense, c, col_lower, col_upper, row_lower, row_upper, col_start, row_index, value,
                   row_names=None, col_names=None, source=""):
    m, n = len(row_lower), len(c)
    with open(path, "w") as out:
        out.write("LPM 1\n")
        if source:
            out.write(f"# source {source}\n")
        out.write(f"NAME {name}\nSENSE {'MAX' if sense == -1 else 'MIN'}\n")
        out.write(f"ROWS {m}\nCOLS {n}\nNNZ {len(value)}\nOFFSET 0.0\n")
        for tag, arr in (("OBJ", c), ("COL_LOWER", col_lower), ("COL_UPPER", col_upper),
                         ("ROW_LOWER", row_lower), ("ROW_UPPER", row_upper)):
            out.write(tag + "\n")
            _write_nums(out, np.asarray(arr, dtype=float))
        out.write("COL_START\n")
        _write_nums(out, np.asarray(col_start, dtype=np.int64))
        out.write("ROW_INDEX\n")
        _write_nums(out, np.asarray(row_index, dtype=np.int64))
        out.write("VALUE\n")
        _write_nums(out, np.asarray(value, dtype=float))
        if row_names is not None:
            out.write("ROW_NAMES\n" + "\n".join(row_names) + "\n")
        if col_names is not None:
            out.write("COL_NAMES\n" + "\n".join(col_names) + "\n")
        out.write("END\n")


def write_sidecar(lpm_path, info: dict):
    with open(os.path.splitext(lpm_path)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)


def read_sidecar(lpm_path) -> dict:
    with open(os.path.splitext(lpm_path)[0] + ".json") as f:
        return json.load(f)
