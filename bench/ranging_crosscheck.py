#!/usr/bin/env python3
"""ranging_crosscheck.py — our sensitivity ranging (ps26119 solve --ranging) against HiGHS's own
ranging (Highs.getRanging, a separate reference via highspy; tooling only, never linked).

For each model both solvers solve by simplex; then, per objective coefficient, our [lower, upper]
is compared with HiGHS's [col_cost_dn, col_cost_up], and per binding row our range of the binding
bound with HiGHS's [row_bound_dn, row_bound_up]. Ranges depend on the optimal BASIS: when the two
solvers stop at different vertices (alternative optima) or at the same primal-degenerate vertex
with different bases, ranges can legitimately differ — such models are reported, not hidden, with
the reason (different x, or degenerate basics). Agreement: both infinite with the same sign, or
|a − b| ≤ 1e-6 (1 + |b|).

The ranging itself is checked independently by re-solving in tests/unit/test_ranging.cpp (inside
a range the old x stays optimal / the objective moves by y·Δ); this script adds the comparison
with an established solver.

usage: bench/ranging_crosscheck.py [--bin build/ps26119] [--set netlib_small|netlib] [--only a,b]
Writes bench/results/ranging-crosscheck-<set>-<machine>-<githash>.csv.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
from lpm import read_lpm, read_mps_highspy  # noqa: E402
from machine_info import machine_info  # noqa: E402

FIELDS = ["git_hash", "machine", "cpu", "date", "highs_version", "set", "instance", "rows", "cols", "status",
          "same_x", "degenerate_basics", "costs_compared", "costs_agree", "rows_compared", "rows_agree",
          "max_cost_diff", "max_row_diff", "note"]


def agree(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b) or abs(a) >= 1e30 or abs(b) >= 1e30:
        big_a, big_b = (math.isinf(a) or abs(a) >= 1e30), (math.isinf(b) or abs(b) >= 1e30)
        return big_a and big_b and (a > 0) == (b > 0)
    return abs(a - b) <= 1e-6 * (1 + abs(b))


def diff(a: float, b: float) -> float:
    if agree(a, b):
        return 0.0
    if math.isinf(a) or math.isinf(b) or abs(a) >= 1e30 or abs(b) >= 1e30:
        return math.inf
    return abs(a - b) / (1 + abs(b))


def highs_ranging(m):
    import highspy
    import numpy as np
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "simplex")
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m.num_cols, m.num_rows
    lp.col_cost_ = np.asarray(m.obj, float)
    lp.col_lower_, lp.col_upper_ = np.asarray(m.col_lower, float), np.asarray(m.col_upper, float)
    lp.row_lower_, lp.row_upper_ = np.asarray(m.row_lower, float), np.asarray(m.row_upper, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.asarray(m.col_start, np.int32)
    lp.a_matrix_.index_ = np.asarray(m.row_index, np.int32)
    lp.a_matrix_.value_ = np.asarray(m.value, float)
    lp.offset_ = m.obj_offset
    lp.sense_ = highspy.ObjSense.kMaximize if m.sense == -1 else highspy.ObjSense.kMinimize
    h.passModel(lp)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    status, rg = h.getRanging()
    if status != highspy.HighsStatus.kOk or not rg.valid:
        return None
    x = list(h.getSolution().col_value)
    return x, rg


def ours(bin_path, model_path, tmp):
    sol, rng = os.path.join(tmp, "o.sol"), os.path.join(tmp, "o.ranging.csv")
    for p in (sol, rng):
        if os.path.exists(p):
            os.remove(p)
    p = subprocess.run([bin_path, "solve", model_path, "--algorithm", "simplex", "--out", sol, "--ranging", rng],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not os.path.exists(rng):
        refused = [l.split("refused:", 1)[1].strip() for l in p.stdout.splitlines() if "refused:" in l]
        return None, (refused[0] if refused else f"exit {p.returncode}")
    x = []
    with open(sol) as f:
        cols = False
        for line in f:
            t = line.split()
            if t and t[0] == "COLUMNS":
                cols = True
                continue
            if t and t[0] in ("ROWS", "END"):
                cols = False
            if cols and len(t) >= 2:
                x.append(float(t[1]))
    mdeg = re.search(r"\((\d+) basic variable\(s\) at a bound", p.stdout)
    deg = int(mdeg.group(1)) if mdeg else 0
    costs, rows = [], []
    with open(rng) as f:
        for r in csv.DictReader(f):
            (costs if r["kind"] == "cost" else rows).append(r)
    return (x, costs, rows, deg), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--set", default="netlib_small", choices=["netlib_small", "netlib"])
    ap.add_argument("--only", default="")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import highspy
    info = machine_info(a.bin)
    hv = f"{highspy.Highs().version()}" if hasattr(highspy.Highs(), "version") else ""
    d = os.path.join(ROOT, "data", a.set)
    ext = ".lpm" if a.set == "netlib_small" else ".mps"
    names = sorted(f[:-len(ext)] for f in os.listdir(d) if f.endswith(ext))
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    out = a.out or os.path.join(HERE, "results", f"ranging-crosscheck-{a.set}-{info['machine']}-{info['git_hash']}.csv")
    rows_out = []
    tot = {"c": 0, "ca": 0, "r": 0, "ra": 0, "same": 0, "models": 0}
    with tempfile.TemporaryDirectory(prefix="rngx-") as tmp:
        for name in names:
            path = os.path.join(d, name + ext)
            m = read_lpm(path) if ext == ".lpm" else read_mps_highspy(path)
            row = {k: info.get(k, "") for k in ("git_hash", "machine", "cpu", "date")}
            row.update(highs_version=hv, set=a.set, instance=name, rows=m.num_rows, cols=m.num_cols)
            if m.num_rows > a.max_rows:
                row.update(status="skipped", note=f"more than {a.max_rows} rows")
                rows_out.append(row)
                continue
            got, why = ours(a.bin, path, tmp)
            ref = highs_ranging(m)
            if got is None or ref is None:
                row.update(status="not compared", note=("ours: " + why) if got is None else "HiGHS: no ranging")
                rows_out.append(row)
                print(f"{name:10s} {row['note']}", flush=True)
                continue
            x, costs, rrows, deg = got
            hx, rg = ref
            same_x = len(x) == len(hx) and all(abs(p - q) <= 1e-7 * (1 + abs(q)) for p, q in zip(x, hx))
            ca = cc = ra = rc = 0
            mcd = mrd = 0.0
            for c in costs:
                j = int(c["index"])
                lo, up = float(c["lower"]), float(c["upper"])
                hlo, hup = rg.col_cost_dn.value_[j], rg.col_cost_up.value_[j]
                cc += 1
                ok = agree(lo, hlo) and agree(up, hup)
                ca += ok
                mcd = max(mcd, diff(lo, hlo), diff(up, hup))
            for r in rrows:
                if r["status"] not in ("upper_binding", "lower_binding"):
                    continue
                i = int(r["index"])
                lo, up = float(r["lower"]), float(r["upper"])
                if lo == up:  # one-point range (degenerate equality): HiGHS reports these differently
                    continue
                hlo, hup = rg.row_bound_dn.value_[i], rg.row_bound_up.value_[i]
                rc += 1
                ok = agree(lo, hlo) and agree(up, hup)
                ra += ok
                mrd = max(mrd, diff(lo, hlo), diff(up, hup))
            note = "" if same_x else "different optimal x (alternative optimum): ranges are basis-dependent"
            if same_x and (ca < cc or ra < rc) and deg:
                note = "same x, primal-degenerate vertex: the two bases may differ"
            row.update(status="compared", same_x="yes" if same_x else "no", degenerate_basics=deg, costs_compared=cc,
                       costs_agree=ca, rows_compared=rc, rows_agree=ra, max_cost_diff=f"{mcd:.2e}",
                       max_row_diff=f"{mrd:.2e}", note=note)
            rows_out.append(row)
            tot["models"] += 1
            tot["same"] += same_x
            tot["c"] += cc
            tot["ca"] += ca
            tot["r"] += rc
            tot["ra"] += ra
            print(f"{name:10s} same_x={'yes' if same_x else 'no ':3s} deg={deg:<4d} costs {ca}/{cc}  rows {ra}/{rc}  {note}",
                  flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"{tot['models']} models compared ({tot['same']} at the same x): cost ranges agree {tot['ca']}/{tot['c']}, "
          f"binding-row ranges agree {tot['ra']}/{tot['r']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
