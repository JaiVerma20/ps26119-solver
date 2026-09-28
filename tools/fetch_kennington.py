#!/usr/bin/env python3
"""fetch_kennington.py — the 16 Kennington LPs (Carolan et al., Oper. Res. 38(2), 1990) into
data/kennington/ (git-ignored), checked against the official netlib table, with their published
optimal values as the reference.

Source of the MPS files: github.com/SkyLiu0/netlib (feasible/*.mps, plain MPS expansions of
netlib lp/data/kennington, whose own files are "doubly compressed" EMPS). Integrity: every model is
read with the independent highspy reader and its row / column / nonzero counts must equal the table
in https://www.netlib.org/lp/data/kennington/readme (fetched here); the published optimum (ALPO,
8 significant digits) is written to data/kennington/optima.csv (the table counts rows including
the objective row and columns without slacks, as netlib does).

usage: tools/fetch_kennington.py [--only ken-07,pds-02]
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from lpm import read_mps_highspy  # noqa: E402

MIRROR = "https://raw.githubusercontent.com/SkyLiu0/netlib/main/feasible/"
README = "https://www.netlib.org/lp/data/kennington/readme"
OUT = os.path.join(ROOT, "data", "kennington")


def curl(url: str, dst: str) -> None:
    """curl, not urllib: python.org builds on macOS often lack CA certificates. --compressed lets
    the server gzip the transfer (the MPS files are large and repetitive)."""
    subprocess.run(["curl", "-sfL", "--compressed", "-o", dst, url], check=True)


def official_table(path: str) -> dict:
    """{name: (rows, cols, nnz, optimum)} from the netlib readme table."""
    table = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^([A-Z]+-[A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(\S+)\s*$", line)
            if m:
                table[m[1].lower()] = (int(m[2]), int(m[3]), int(m[4]), float(m[5]))
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    readme = os.path.join(OUT, "readme.netlib.txt")
    curl(README, readme)
    table = official_table(readme)
    if len(table) != 16:
        sys.exit(f"expected 16 models in the netlib table, found {len(table)}")
    names = [n for n in table if not a.only or n in a.only.split(",")]
    rows_out, bad = [], []
    for name in names:
        dst = os.path.join(OUT, name + ".mps")
        if not os.path.exists(dst):
            print(f"downloading {name} …", flush=True)
            curl(MIRROR + name + ".mps", dst)
        m = read_mps_highspy(dst)
        rows, cols, nnz, opt = table[name]
        # netlib counts the objective row among the rows and the objective entries among the nonzeros
        obj_nnz = sum(1 for c in m.obj if c != 0)
        got = (m.num_rows + 1, m.num_cols, m.nnz + obj_nnz)
        ok = got == (rows, cols, nnz)
        print(f"{name:8s} rows {m.num_rows:>7} cols {m.num_cols:>7} nnz {m.nnz:>8}  netlib {rows}/{cols}/{nnz}  "
              f"{'OK' if ok else 'MISMATCH ' + str(got)}  optimum {opt:.8g}", flush=True)
        if not ok:
            bad.append(name)
        rows_out.append({"name": name, "rows": m.num_rows, "cols": m.num_cols, "nnz": m.nnz, "published_optimum": repr(opt),
                         "source": MIRROR + name + ".mps", "size_check": "ok" if ok else "MISMATCH"})
    with open(os.path.join(OUT, "optima.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"-> {OUT}/optima.csv ({len(rows_out)} models, {len(bad)} size mismatches{': ' + ', '.join(bad) if bad else ''})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
