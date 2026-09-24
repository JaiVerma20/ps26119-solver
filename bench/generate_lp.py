#!/usr/bin/env python3
"""generate_lp.py — random sparse LPs whose optimum is known by construction.

usage: generate_lp.py --rows 10000 [--cols-per-row 1.5] [--nnz-per-col 4] [--seed 1]
                      [--out bench/generated/rand-10000.lpm]

Structure: n = round(1.5·m) columns, each with `nnz_per_col` random rows (values
±U[0.5, 2]); every row gets at least one entry. Statuses (fractions of rows / columns):
rows 20% equality, 40% active inequality (half ≥, half ≤), 40% inactive (ranged or one-sided);
columns 40% at lower bound, 10% at upper bound, 50% strictly between bounds (5% free).
The KKT construction is in lpgen.py; the known optimum goes to the sidecar .json.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lpgen  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def generate(m: int, cols_per_row=1.5, nnz_per_col=4, seed=1):
    rng = np.random.default_rng(seed)
    n = int(round(cols_per_row * m))
    k = min(nnz_per_col, m)
    rows = rng.integers(0, m, size=(n, k)).ravel()
    cols = np.repeat(np.arange(n), k)
    # make sure every row has an entry
    missing = np.setdiff1d(np.arange(m), rows)
    rows = np.concatenate([rows, missing])
    cols = np.concatenate([cols, rng.integers(0, n, len(missing))])
    vals = rng.uniform(0.5, 2.0, len(rows)) * rng.choice([-1.0, 1.0], len(rows))
    col_start, row_index, value = lpgen.csc_from_triplets(m, n, rows, cols, vals)

    # columns
    u = rng.random(n)
    col_at = np.where(u < 0.4, -1, np.where(u < 0.5, 1, 0)).astype(np.int8)
    free = (col_at == 0) & (rng.random(n) < 0.1)
    l = np.where(free, -np.inf, 0.0)
    x = np.where(col_at < 0, 0.0, rng.uniform(0.5, 5.0, n))
    up_finite = rng.random(n) < 0.4
    cu = np.where(col_at > 0, x, np.where(up_finite & ~free, x + rng.uniform(0.5, 5.0, n), np.inf))
    cu = np.where(col_at < 0, np.where(up_finite, rng.uniform(0.5, 5.0, n), np.inf), cu)
    x = np.where(free, rng.uniform(-5.0, 5.0, n), x)
    cu = np.where(free, np.inf, cu)

    # rows
    r = rng.random(m)
    row_type = np.where(r < 0.2, "eq", np.where(r < 0.4, "ge", np.where(r < 0.6, "le",
                        np.where(r < 0.8, "range", np.where(r < 0.9, "ge", "le")))))
    row_active = (r >= 0.2) & (r < 0.6)
    cert = lpgen.certify(rng, m, n, col_start, row_index, value, x, l, cu, col_at, row_type, row_active)
    return {
        "m": m, "n": n, "c": cert["c_min"], "col_lower": l, "col_upper": cu, "row_lower": cert["row_lower"],
        "row_upper": cert["row_upper"], "col_start": col_start, "row_index": row_index, "value": value,
        "optimum": cert["optimum_min"], "sense": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=float, required=True)
    ap.add_argument("--cols-per-row", type=float, default=1.5)
    ap.add_argument("--nnz-per-col", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = int(a.rows)
    out = a.out or os.path.join(HERE, "generated", f"rand-{m}-s{a.seed}.lpm")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    g = generate(m, a.cols_per_row, a.nnz_per_col, a.seed)
    name = os.path.splitext(os.path.basename(out))[0]
    lpgen.write_lpm_fast(out, name, g["sense"], g["c"], g["col_lower"], g["col_upper"], g["row_lower"],
                         g["row_upper"], g["col_start"], g["row_index"], g["value"], source="bench/generate_lp.py")
    lpgen.write_sidecar(out, {"name": name, "generator": "generate_lp.py", "rows": g["m"], "cols": g["n"],
                              "nnz": int(len(g["value"])), "seed": a.seed, "sense": g["sense"],
                              "optimum": g["optimum"], "cols_per_row": a.cols_per_row, "nnz_per_col": a.nnz_per_col})
    print(f"{out}: {g['m']} rows, {g['n']} cols, {len(g['value'])} nnz, optimum {g['optimum']!r} "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
