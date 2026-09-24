#!/usr/bin/env python3
"""netlib_small.py — run every engine on the small Netlib set, verify each solution with
tools/verify.py (independent reader), compare with the published optimum, and write
bench/results/netlib-small-<githash>.csv.

usage: bench/netlib_small.py [--bin build/ps26119] [--engines oracle,pdlp,r2hpdhg]
                             [--precisions fp64,mixed] [--gpu] [--time-limit 120]
The first-order engines run once at 1e-8; the 1e-4 milestone (iterations, seconds) is
recorded from the same run.
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

DATA = os.path.join(ROOT, "data", "netlib_small")
FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "instance", "rows", "cols", "nnz",
          "engine", "backend", "precision", "tolerance", "status", "objective", "published_optimum",
          "rel_err_published", "iterations", "seconds", "iterations_to_1e-4", "seconds_to_1e-4",
          "verify", "verify_primal_rel", "verify_dual_rel", "verify_gap_rel", "message"]


def optima():
    rows = []
    with open(os.path.join(DATA, "optima.csv")) as f:
        next(f)
        for line in f:
            name, _, _, opt, _ = line.strip().split(",", 4)
            rows.append((name, float(opt)))
    return rows


def run_one(binary, name, opt, engine, precision, gpu, time_limit, tmp):
    sol_path = os.path.join(tmp, f"{name}.{engine}.{precision}.sol")
    cmd = [binary, "solve", os.path.join(DATA, name + ".lpm"), "--algorithm", engine, "--out", sol_path,
           "--time-limit", str(time_limit)]
    if engine != "oracle":
        cmd += ["--precision", precision, "--tol", "1e-8"]
    if gpu:
        cmd.append("--gpu")
    subprocess.run(cmd, capture_output=True, text=True)
    s = read_solution(sol_path)
    h = s.header
    rep = verify.verify(os.path.join(DATA, name + ".mps"), sol_path, opt)
    obj = rep.get("objective", float("nan"))
    return {
        "instance": name, "rows": rep["rows"], "cols": rep["cols"], "nnz": rep["nnz"],
        "engine": engine, "backend": "gpu" if gpu else "cpu",
        "precision": "dd" if engine == "oracle" else precision,
        "tolerance": "exact" if engine == "oracle" else "1e-8",
        "status": h.get("status"), "objective": f"{obj:.12g}", "published_optimum": f"{opt:.11g}",
        "rel_err_published": f"{abs(obj - opt) / (1 + abs(opt)):.2e}",
        "iterations": h.get("iterations"), "seconds": h.get("seconds"),
        "iterations_to_1e-4": h.get("iterations_to_fast", ""), "seconds_to_1e-4": h.get("seconds_to_fast", ""),
        "verify": rep["verdict"], "verify_primal_rel": f"{rep.get('primal_rel', float('nan')):.2e}",
        "verify_dual_rel": f"{rep.get('dual_rel', float('nan')):.2e}",
        "verify_gap_rel": f"{rep.get('gap_rel', float('nan')):.2e}", "message": h.get("message", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--engines", default="oracle,pdlp,r2hpdhg")
    ap.add_argument("--precisions", default="fp64,mixed")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    info = machine_info()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, opt in optima():
            for engine in a.engines.split(","):
                precs = ["dd"] if engine == "oracle" else a.precisions.split(",")
                for p in precs:
                    r = run_one(a.bin, name, opt, engine, p, a.gpu and engine != "oracle", a.time_limit, tmp)
                    r.update(info)
                    rows.append(r)
                    print(f"{name:9s} {engine:8s} {r['precision']:5s} {r['status']:14s} obj {r['objective']:>18s} "
                          f"it {r['iterations']:>7s} {float(r['seconds']):8.3f}s  verify {r['verify']}", flush=True)
    tag = "gpu-" if a.gpu else ""
    out = a.out or os.path.join(HERE, "results", f"netlib-small-{tag}{info['git_hash']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    npass = sum(r["verify"] == "PASS" for r in rows)
    print(f"\n{npass}/{len(rows)} verified PASS -> {out}")
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
