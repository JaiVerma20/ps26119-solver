#!/usr/bin/env python3
"""fetch_netlib_infeasible.py — download the 29 infeasible LPs of the Netlib collection
(https://www.netlib.org/lp/infeas/) as uncompressed MPS files from the MPS conversion in
https://github.com/SkyLiu0/netlib (folder infeasible/) into data/netlib_infeasible/ (git-ignored),
and write data/netlib_infeasible/reference.csv with HiGHS's status for each (reference only).

The files are public test data; record provenance in data/SOURCES.md. The script prints each
file's size and verifies that highspy reads it; nothing here is linked into the solver.

usage: tools/fetch_netlib_infeasible.py [--skip-highs]
Then:  PS26119_MACHINE=<name> python3 bench/netlib_infeasible.py
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

BASE = "https://raw.githubusercontent.com/SkyLiu0/netlib/main/infeasible/"
NAMES = """bgdbg1 bgetam bgindy bgprtr box1 ceria3d chemcom cplex1 cplex2 ex72a ex73a forest6 galenet gosh
gran greenbea itest2 itest6 klein1 klein2 klein3 mondou2 pang pilot4i qual reactor refinery vol1
woodinfe""".split()  # listing of that folder, 2026-09-27 (the Netlib lp/infeas collection)
OUT = os.path.join(ROOT, "data", "netlib_infeasible")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-highs", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name in NAMES:
        dst = os.path.join(OUT, name + ".mps")
        if not os.path.exists(dst):
            subprocess.run(["curl", "-sfL", "-o", dst, BASE + name + ".mps"], check=True)
        size = os.path.getsize(dst)
        status = ""
        if not a.skip_highs:
            import highspy  # tooling only
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(dst)
            h.run()
            status = h.modelStatusToString(h.getModelStatus())
        rows.append({"name": name, "bytes": size, "highs_status": status})
        print(f"{name:10s} {size:9d} bytes  HiGHS: {status}", flush=True)
    with open(os.path.join(OUT, "reference.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "bytes", "highs_status"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} models in {OUT}")


if __name__ == "__main__":
    main()
