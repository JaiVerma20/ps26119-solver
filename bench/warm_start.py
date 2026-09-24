#!/usr/bin/env python3
"""warm_start.py — what-if re-solves of the refinery planning LP: cold start vs warm start
from the base-case solution (CLAUDE.md §9 idea 3: planners re-solve many slightly changed
LPs — price updates, demand changes, SLP steps — so the re-solve cost is what matters).

Scenarios on refinery-T<T> (bench/generate_refinery_lp.py):
  price    product sale prices (sell columns) scaled by 1 ± 3% (random per product/period)
  demand   demand caps (demand rows) raised by 5%
  crude    crude availability (buy column upper bounds) cut by 5%
Each scenario is solved cold and warm (--warm base.sol) to 1e-8; both solutions are checked
by tools/verify.py and must agree on the objective. Iterations are deterministic; wall
times on a laptop CPU vary run to run.

usage: bench/warm_start.py [--bin build/ps26119] [--periods 365,8760] [--gpu]
Writes bench/results/warm-start-<machine>-<githash>.csv.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
import generate_refinery_lp as gr  # noqa: E402
import lpgen  # noqa: E402
import verify  # noqa: E402
from lpm import read_lpm, read_solution, write_lpm  # noqa: E402
from machine_info import machine_info  # noqa: E402

FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "instance", "rows", "cols", "nnz",
          "scenario", "backend", "precision", "start", "status", "iterations", "seconds", "objective",
          "objective_change_vs_base", "verify", "iteration_ratio_warm_over_cold"]


def perturb(m, scenario, T, seed=7):
    rng = np.random.default_rng(seed)
    L = gr.Layout()
    t = np.arange(T)
    if scenario == "price":
        for p in range(L.P):
            cols = t * L.ncols + L.sell[p]
            f = 1 + rng.uniform(-0.03, 0.03, T)
            for j, fj in zip(cols, f):
                m.obj[j] *= fj
    elif scenario == "demand":
        for p in range(L.P):
            for i in t * L.nrows + L.r_dem[p]:
                if np.isfinite(m.row_upper[i]):
                    m.row_upper[i] *= 1.05
    elif scenario == "crude":
        for k in range(L.K):
            for j in t * L.ncols + L.buy[k]:
                if np.isfinite(m.col_upper[j]):
                    m.col_upper[j] *= 0.95
    return m


def solve(binary, lpm, sol, gpu, warm=None, time_limit=600):
    cmd = [binary, "solve", lpm, "--algorithm", "r2hpdhg", "--tol", "1e-8", "--time-limit", str(time_limit),
           "--out", sol] + (["--gpu"] if gpu else []) + (["--warm", warm] if warm else [])
    subprocess.run(cmd, capture_output=True, text=True)
    return read_solution(sol).header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--periods", default="365,8760")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    info = machine_info()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for T in [int(x) for x in a.periods.split(",")]:
            base = os.path.join(HERE, "generated", f"refinery-T{T}-s1.lpm")
            if not os.path.exists(base):
                subprocess.run([sys.executable, os.path.join(HERE, "generate_refinery_lp.py"), "--periods", str(T),
                                "--out", base], check=True)
            side = lpgen.read_sidecar(base)
            base_sol = os.path.join(tmp, "base.sol")
            hb = solve(a.bin, base, base_sol, a.gpu, time_limit=a.time_limit)
            base_obj = float(hb["objective"])
            print(f"T={T} base: {hb['status']} it {hb['iterations']} t {hb['seconds']}s obj {base_obj:.10g}", flush=True)
            for scen in ("price", "demand", "crude"):
                m = perturb(read_lpm(base), scen, T)
                path = os.path.join(tmp, f"{scen}.lpm")
                write_lpm(m, path)
                res = {}
                for start in ("cold", "warm"):
                    sol = os.path.join(tmp, f"{scen}.{start}.sol")
                    h = solve(a.bin, path, sol, a.gpu, base_sol if start == "warm" else None, a.time_limit)
                    rep = verify.verify(path, sol)
                    res[start] = h
                    rows.append({**info, "instance": side["name"], "rows": side["rows"], "cols": side["cols"],
                                 "nnz": side["nnz"], "scenario": scen, "backend": "gpu" if a.gpu else "cpu",
                                 "precision": "fp64", "start": start, "status": h["status"],
                                 "iterations": h["iterations"], "seconds": h["seconds"], "objective": h["objective"],
                                 "objective_change_vs_base": f"{(float(h['objective']) - base_obj) / abs(base_obj):+.3e}",
                                 "verify": rep["verdict"]})
                ratio = int(res["warm"]["iterations"]) / max(1, int(res["cold"]["iterations"]))
                rows[-1]["iteration_ratio_warm_over_cold"] = f"{ratio:.3f}"
                rows[-2]["iteration_ratio_warm_over_cold"] = f"{ratio:.3f}"
                agree = abs(float(res["warm"]["objective"]) - float(res["cold"]["objective"])) / (
                    1 + abs(float(res["cold"]["objective"])))
                print(f"  {scen:7s} cold it {res['cold']['iterations']:>6s} t {res['cold']['seconds']:>9s}s | "
                      f"warm it {res['warm']['iterations']:>6s} t {res['warm']['seconds']:>9s}s | "
                      f"warm/cold {ratio:.3f} | objectives agree to {agree:.1e} | verify "
                      f"{rows[-2]['verify']}/{rows[-1]['verify']}", flush=True)
    out = a.out or os.path.join(HERE, "results", f"warm-start-{info['machine']}-{info['git_hash']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
