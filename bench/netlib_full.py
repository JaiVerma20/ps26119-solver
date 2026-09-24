#!/usr/bin/env python3
"""netlib_full.py — the whole Netlib LP set (93 models, data/netlib/, fetched by
tools/fetch_netlib.py) through a first-order engine, every solution checked by
tools/verify.py (independent highspy reader) and compared with the published optimum
(Netlib readme, MINOS 5.3) and with HiGHS.

usage: bench/netlib_full.py [--bin build/ps26119] [--engine r2hpdhg] [--precision fp64]
                            [--time-limit 60] [--gpu] [--only afiro,kb2]
Writes bench/results/netlib-full-<engine>-<precision>[-gpu]-<machine>-<githash>.csv.

Reading the result honestly:
  * "solved" = engine status Optimal (relative KKT 1e-8 AND verifier-grade per-row checks),
    and tools/verify.py PASS;
  * "matches reference" = |obj − ref| / (1 + |ref|) ≤ 1e-6, ref = HiGHS optimum (the Netlib
    table has known outliers, see data/netlib_small/README.netlib.txt);
  * TimeLimit / IterationLimit rows are reported, not dropped. First-order methods are known
    to struggle on some Netlib models (degenerate, badly scaled: pilot*, greenbea, ...).
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

DATA = os.path.join(ROOT, "data", "netlib")
FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "settings", "instance", "rows", "cols", "nnz", "engine",
          "backend", "precision", "tolerance", "time_limit", "status", "iterations", "seconds", "seconds_to_1e-4",
          "objective", "published_optimum", "highs_objective", "rel_err_highs", "rel_err_published", "verify",
          "verify_primal_rel", "verify_dual_rel", "verify_gap_rel", "certified_bound", "certified_gap", "message"]


def rel(a, b):
    try:
        return abs(float(a) - float(b)) / (1 + abs(float(b)))
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--engine", default="r2hpdhg")
    ap.add_argument("--precision", default="fp64")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[], help="engine knob name=value (repeatable)")
    ap.add_argument("--tag", default="", help="suffix for the CSV name, e.g. gm12")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    optima = os.path.join(DATA, "optima.csv")
    if not os.path.exists(optima):
        sys.exit("run tools/fetch_netlib.py first")
    with open(optima) as f:
        models = list(csv.DictReader(f))
    if a.only:
        keep = set(a.only.split(","))
        models = [m for m in models if m["name"] in keep]
    info = machine_info(a.bin)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for mdl in models:
            name = mdl["name"]
            sol = os.path.join(tmp, name + ".sol")
            cmd = [a.bin, "solve", os.path.join(DATA, name + ".lpm"), "--algorithm", a.engine, "--precision",
                   a.precision, "--tol", "1e-8", "--time-limit", str(a.time_limit), "--out", sol]
            if a.gpu:
                cmd.append("--gpu")
            for kv in a.set:
                cmd += ["--set", kv]
            subprocess.run(cmd, capture_output=True, text=True)
            r = {"settings": " ".join(a.set), "instance": name, "rows": mdl["rows"], "cols": mdl["cols"], "nnz": mdl["nnz"], "engine": a.engine,
                 "backend": "gpu" if a.gpu else "cpu", "precision": a.precision, "tolerance": "1e-8",
                 "time_limit": a.time_limit, "published_optimum": mdl["published_optimum"],
                 "highs_objective": mdl["highs_objective"]}
            if not os.path.exists(sol):
                r.update(status="NoOutput", verify="FAIL")
            else:
                h = read_solution(sol).header
                r.update(status=h.get("status"), iterations=h.get("iterations"), seconds=h.get("seconds"),
                         objective=h.get("objective"), message=h.get("message", ""))
                r["seconds_to_1e-4"] = h.get("seconds_to_fast", "")
                cb = h.get("certified_bound", "")
                r["certified_bound"] = cb
                try:
                    cbv = float(cb)
                    r["certified_gap"] = (f"{abs(float(r['objective']) - cbv) / (1 + abs(float(r['objective']))):.2e}"
                                          if abs(cbv) != float("inf") else "inf")
                except ValueError:
                    r["certified_gap"] = ""

                r["rel_err_highs"] = f"{rel(r['objective'], mdl['highs_objective']):.2e}" if mdl["highs_objective"] else ""
                r["rel_err_published"] = (f"{rel(r['objective'], mdl['published_optimum']):.2e}"
                                          if mdl["published_optimum"] else "")
                rep = verify.verify(os.path.join(DATA, name + ".mps"), sol)
                r.update(verify=rep["verdict"], verify_primal_rel=f"{rep.get('primal_rel', float('nan')):.2e}",
                         verify_dual_rel=f"{rep.get('dual_rel', float('nan')):.2e}",
                         verify_gap_rel=f"{rep.get('gap_rel', float('nan')):.2e}")
            r.update(info)
            rows.append(r)
            print(f"{name:10s} {r['status']:15s} it {str(r.get('iterations', '')):>8s} "
                  f"t {str(r.get('seconds', '')):>10s}  obj {str(r.get('objective', '')):>22s}  "
                  f"errHiGHS {r.get('rel_err_highs', ''):>9s}  verify {r['verify']}", flush=True)
    tag = f"{a.engine}-{a.precision}" + ("-gpu" if a.gpu else "") + (f"-{a.tag}" if a.tag else "")
    out = a.out or os.path.join(HERE, "results", f"netlib-full-{tag}-{info['machine']}-{info['git_hash']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    solved = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"]
    match = [r for r in solved if r.get("rel_err_highs") and float(r["rel_err_highs"]) <= 1e-6]
    print(f"\n{len(solved)}/{len(rows)} solved to 1e-8 and verified; {len(match)} of those match HiGHS to 1e-6 "
          f"(time limit {a.time_limit:g}s) -> {out}")


if __name__ == "__main__":
    main()
