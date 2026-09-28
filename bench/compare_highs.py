#!/usr/bin/env python3
"""compare_highs.py — ps26119 against a real-world solver, HiGHS (separate process-free reference via
highspy, tooling only; never linked), engine by engine, on the same models and the same rules:

  ps26119:  auto, simplex, r2hpdhg (1 thread), r2hpdhg (all cores)          --tol 1e-8
  HiGHS:    simplex (dual), ipm (interior point + crossover), pdlp           pdlp tolerance 1e-8

Models: the Netlib LP set (data/netlib/, tools/fetch_netlib.py) and the generated LPs with an
optimum known by construction (bench/generated: refinery T12 / T365 / T2190 / T8760, random 1e4 /
1e5; --big adds random 1e6).

Rules, identical for both solvers:
  * time = the solve call only: our solution file's `seconds` (reading excluded, presolve included)
    vs the wall time of Highs.run() (the model is passed in memory; presolve included);
  * solved = status Optimal AND tools/verify.py PASS (independent reader, same tolerances for both)
    AND |obj − ref| / (1 + |ref|) ≤ 1e-6, ref = HiGHS reference optimum from data/netlib/optima.csv
    (Netlib) or the known optimum (generated models);
  * a TimeLimit / failure is a row, never dropped.
Honest limits: HiGHS runs with its defaults (serial simplex, serial IPM); its PDLP is the CPU
cuPDLP-C port, whose stopping measure is not identical to ours even at the same 1e-8. One machine.

usage: bench/compare_highs.py [--bin build/ps26119] [--set netlib,scale] [--big] [--only a,b]
                              [--time-limit 60] [--scale-time-limit 300]
Writes bench/results/compare-highs-<machine>-<githash>.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
import highs_ref  # noqa: E402
import verify  # noqa: E402
from lpm import read_solution  # noqa: E402
from machine_info import machine_info  # noqa: E402

FIELDS = ["git_hash", "machine", "cpu", "cpu_cores", "gpu", "driver", "cuda", "date", "set", "instance", "rows", "cols", "nnz",
          "solver", "engine", "threads", "tolerance", "time_limit", "status", "iterations", "seconds", "objective",
          "reference", "reference_kind", "rel_err_ref", "verify", "verify_primal_rel", "verify_dual_rel", "verify_gap_rel",
          "solved", "solver_version", "message"]

OURS = [("auto", 1), ("simplex", 1), ("r2hpdhg", 1), ("r2hpdhg", 0)]
HIGHS = [("simplex", {"solver": "simplex"}), ("ipm", {"solver": "ipm"}),
         ("pdlp", {"solver": "pdlp", "pdlp_optimality_tolerance": 1e-8})]
SCALE = ["refinery-T12-s1", "refinery-T365-s1", "refinery-T2190-s1", "refinery-T8760-s1", "rand-10000-s1", "rand-100000-s1"]


def rel(a, b):
    try:
        return abs(float(a) - float(b)) / (1 + abs(float(b)))
    except (TypeError, ValueError):
        return float("nan")


def instances(sets, only, big):
    out = []
    if "netlib" in sets:
        opt = os.path.join(ROOT, "data", "netlib", "optima.csv")
        if not os.path.exists(opt):
            sys.exit("data/netlib missing: run tools/fetch_netlib.py first")
        with open(opt) as f:
            for m in csv.DictReader(f):
                out.append({"set": "netlib", "name": m["name"], "path": os.path.join(ROOT, "data", "netlib", m["name"] + ".mps"),
                            "rows": m["rows"], "cols": m["cols"], "nnz": m["nnz"], "reference": m["highs_objective"],
                            "reference_kind": "HiGHS optimum (optima.csv)"})
    if "scale" in sets:
        names = SCALE + (["rand-1000000-s1"] if big else [])
        for n in names:
            p = os.path.join(ROOT, "bench", "generated", n + ".lpm")
            if not os.path.exists(p):
                print(f"skip {n}: {p} missing (bench/generate_*.py)", file=sys.stderr)
                continue
            with open(p[:-4] + ".json") as f:
                side = json.load(f)
            out.append({"set": "scale", "name": n, "path": p, "rows": side["rows"], "cols": side["cols"], "nnz": side["nnz"],
                        "reference": repr(side["optimum"]), "reference_kind": "known optimum (by construction)"})
    if only:
        keep = set(only.split(","))
        out = [m for m in out if m["name"] in keep]
    return out


def check(model, sol, r):
    rep = verify.verify(model, sol)
    r.update(verify=rep["verdict"], verify_primal_rel=f"{rep.get('primal_rel', float('nan')):.2e}",
             verify_dual_rel=f"{rep.get('dual_rel', float('nan')):.2e}", verify_gap_rel=f"{rep.get('gap_rel', float('nan')):.2e}")


def finish(r):
    r["rel_err_ref"] = f"{rel(r.get('objective'), r['reference']):.2e}" if r.get("reference") else ""
    try:
        ok = r["status"] == "Optimal" and r.get("verify") == "PASS" and float(r["rel_err_ref"]) <= 1e-6
    except (TypeError, ValueError):
        ok = False
    r["solved"] = "yes" if ok else "no"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--set", default="netlib,scale")
    ap.add_argument("--only", default="")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--scale-time-limit", type=float, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    info = machine_info(a.bin)
    ours_version = subprocess.run([a.bin, "--version"], capture_output=True, text=True).stdout.strip()
    hv = "HiGHS " + highs_ref.highs_version()
    rows = []
    out = a.out or os.path.join(HERE, "results", f"compare-highs-{info['machine']}-{info['git_hash']}.csv")

    def flush():
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})

    with tempfile.TemporaryDirectory() as tmp:
        for m in instances(a.set.split(","), a.only, a.big):
            tl = a.time_limit if m["set"] == "netlib" else a.scale_time_limit
            base = {**info, "set": m["set"], "instance": m["name"], "rows": m["rows"], "cols": m["cols"], "nnz": m["nnz"],
                    "reference": m["reference"], "reference_kind": m["reference_kind"], "time_limit": tl}
            for eng, thr in OURS:
                sol = os.path.join(tmp, f"{m['name']}.{eng}.{thr}.sol")
                subprocess.run([a.bin, "solve", m["path"], "--algorithm", eng, "--tol", "1e-8", "--threads", str(thr),
                                "--time-limit", str(tl), "--out", sol], capture_output=True, text=True)
                r = {**base, "solver": "ps26119", "engine": eng, "threads": thr if thr else info.get("cpu_cores", "all"),
                     "tolerance": "1e-8", "solver_version": ours_version}
                if os.path.exists(sol):
                    h = read_solution(sol).header
                    r.update(status=h.get("status"), iterations=h.get("iterations"), seconds=h.get("seconds"),
                             objective=h.get("objective"), message=h.get("message", "")[:200])
                    check(m["path"], sol, r)
                else:
                    r.update(status="NoOutput", verify="FAIL")
                rows.append(finish(r))
            for eng, opts in HIGHS:
                sol = os.path.join(tmp, f"{m['name']}.highs-{eng}.sol")
                r = {**base, "solver": "highs", "engine": eng, "threads": "default",
                     "tolerance": "1e-8 (pdlp_optimality_tolerance)" if eng == "pdlp" else "HiGHS defaults (1e-7 feasibility)",
                     "solver_version": hv}
                try:
                    res = highs_ref.solve_with_highs(m["path"], sol, time_limit=tl, options=opts)
                    r.update(status=res["status"], iterations=res["iterations"], seconds=f"{res['seconds']:.6f}",
                             objective=repr(res["objective"]), message=f"HiGHS model status {res['highs_status']}")
                    check(m["path"], sol, r)
                except Exception as e:  # noqa: BLE001 — a failed reference run is a row, not a crash
                    r.update(status="Error", verify="FAIL", message=f"{type(e).__name__}: {e}"[:200])
                rows.append(finish(r))
            line = "  ".join(f"{x['solver'][:2]}-{x['engine']}{'' if x['threads'] in (1, 'default') else '*'}:"
                             f"{'✓' if x['solved'] == 'yes' else x['status'][:4]} {float(x.get('seconds') or 'nan'):.3g}s"
                             for x in rows if x["instance"] == m["name"])
            print(f"{m['name']:18s} {line}", flush=True)
            flush()
    flush()
    by = {}
    for r in rows:
        k = f"{r['solver']}-{r['engine']}" + ("-mt" if r["solver"] == "ps26119" and r["threads"] not in (1, "1") else "")
        by.setdefault(k, [0, 0])
        by[k][0] += r["solved"] == "yes"
        by[k][1] += 1
    print("\nsolved (Optimal + verify PASS + within 1e-6 of the reference):")
    for k, (s, t) in by.items():
        print(f"  {k:22s} {s}/{t}")
    print(f"-> {out}  ({time.strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
