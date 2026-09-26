#!/usr/bin/env python3
"""miplib3.py — small MIPLIB 3 instances through the PROTOTYPE branch-and-bound (dense
double-double oracle as node solver, no cuts, no presolve-for-MIP beyond the LP presolve).
Every result is checked by tools/verify.py in MILP mode (feasibility + integrality) and
against the HiGHS optimum. Reports exactly which instances finish within the time limit.

usage: bench/miplib3.py [--bin build/ps26119] [--time-limit 300]   (tools/fetch_miplib3.py first)
Writes bench/results/miplib3-<machine>-<githash>.csv.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
import verify  # noqa: E402
from lpm import read_solution  # noqa: E402
from machine_info import machine_info  # noqa: E402

DATA = os.path.join(ROOT, "data", "miplib3")
FIELDS = ["git_hash", "machine", "cpu", "date", "instance", "rows", "cols", "integers", "status", "objective",
          "highs_objective", "rel_err_highs", "best_bound", "gap", "nodes_message", "seconds", "verify"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(os.path.join(DATA, "optima.csv")) as f:
        models = list(csv.DictReader(f))
    info = machine_info(a.bin)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for mdl in models:
            name = mdl["name"]
            sol = os.path.join(tmp, name + ".sol")
            subprocess.run([a.bin, "solve", os.path.join(DATA, name + ".mps"), "--time-limit", str(a.time_limit),
                            "--out", sol], capture_output=True, text=True)
            r = {**info, "instance": name, "rows": mdl["rows"], "cols": mdl["cols"], "integers": mdl["integers"],
                 "highs_objective": mdl["highs_objective"]}
            if os.path.exists(sol):
                h = read_solution(sol).header
                r.update(status=h.get("status"), objective=h.get("objective"), best_bound=h.get("dual_objective"),
                         gap=h.get("gap"), nodes_message=h.get("message", ""), seconds=h.get("seconds"))
                try:
                    r["rel_err_highs"] = f"{abs(float(r['objective']) - float(mdl['highs_objective'])) / (1 + abs(float(mdl['highs_objective']))):.2e}"
                except ValueError:
                    r["rel_err_highs"] = ""
                exp = float(mdl["highs_objective"]) if mdl["highs_objective"] else None
                r["verify"] = verify.verify(os.path.join(DATA, name + ".mps"), sol, exp)["verdict"] \
                    if r["status"] == "Optimal" else "—"
            else:
                r.update(status="NoOutput", verify="FAIL")
            rows.append(r)
            print(f"{name:9s} {r['status']:14s} obj {str(r.get('objective', '')):>20s}  HiGHS {mdl['highs_objective']:>20s}  "
                  f"t {str(r.get('seconds', '')):>10s}  {r.get('nodes_message', '')}  verify {r['verify']}", flush=True)
    out = a.out or os.path.join(HERE, "results", f"miplib3-{info['machine']}-{info['git_hash']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    solved = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"]
    print(f"\n{len(solved)}/{len(rows)} solved to optimality and verified (limit {a.time_limit:g}s) -> {out}")


if __name__ == "__main__":
    main()
