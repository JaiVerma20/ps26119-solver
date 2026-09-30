#!/usr/bin/env python3
"""compare_binaries.py — before/after evidence for a change: two ps26119 binaries (e.g. a build of
the previous commit and one of HEAD) on the same machine, same models, same options.

Runs are INTERLEAVED (base, new, base, new, ...) and repeated; the table keeps the fastest run of
each (model, threads, binary) — the usual defence against a noisy (virtualised) machine. The
iteration count is deterministic, so it must be identical across repetitions (checked). Every
solution of the last repetition goes through tools/verify.py (independent reader), and the
objective is compared with the model's known optimum where one exists (.json sidecar of the
generated LPs, data/*/optima.csv otherwise).

usage: bench/compare_binaries.py --base <old ps26119> [--new build/ps26119] [--reps 3]
          [--threads 1,4] [--algorithm r2hpdhg] [--time-limit 1200] model [model ...]
Writes bench/results/compare-binaries-<machine>-<new githash>.csv; the base binary's hash is in
column base_git_hash. Set PS26119_MACHINE to label the machine.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
import verify  # noqa: E402
from machine_info import git_hash, machine_info  # noqa: E402

FIELDS = ["git_hash", "base_git_hash", "machine", "cpu", "cpu_cores", "gpu", "driver", "cuda", "date", "instance", "rows",
          "cols", "nnz", "algorithm", "threads", "binary", "binary_git_hash", "reps", "status", "iterations",
          "seconds_min", "seconds_all", "speedup_vs_base", "objective", "known_optimum", "rel_err_known", "check",
          "verify", "verify_primal_rel", "verify_dual_rel", "verify_gap_rel", "message"]


def known_optimum(path):
    side = os.path.splitext(path)[0] + ".json"
    if os.path.exists(side):
        with open(side) as f:
            v = json.load(f).get("optimum")
            return float(v) if v is not None else None
    optima = os.path.join(os.path.dirname(path), "optima.csv")
    if os.path.exists(optima):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(optima) as f:
            for r in csv.DictReader(f):
                if r.get("name") == name:
                    v = r.get("highs_objective") or r.get("published_optimum")
                    return float(v) if v else None
    return None


def solve(binary, model, algorithm, threads, time_limit, sol):
    if os.path.exists(sol):
        os.remove(sol)
    subprocess.run([binary, "solve", model, "--algorithm", algorithm, "--threads", str(threads), "--tol", "1e-8",
                    "--time-limit", str(time_limit), "--out", sol], capture_output=True, text=True)
    h = {}
    if os.path.exists(sol):
        with open(sol) as f:
            for line in f:
                if line.startswith("COLUMNS"):
                    break
                k, _, v = line.rstrip("\n").partition(" ")
                h[k] = v
    return h


def model_size(binary, model):
    """rows / cols / nnz from `ps26119 info` (lines "rows 27 (...)", "columns 32 (...)", "nonzeros 83 (...)")."""
    out = subprocess.run([binary, "info", model], capture_output=True, text=True).stdout
    size = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            key = {"rows": "rows", "columns": "cols", "nonzeros": "nnz"}.get(parts[0])
            if key:
                size[key] = parts[1]
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--algorithm", default="r2hpdhg")
    ap.add_argument("--time-limit", type=float, default=1200)
    ap.add_argument("--out", default=None)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    info = machine_info(a.new)
    bins = {"base": a.base, "new": a.new}
    hashes = {k: git_hash(v) for k, v in bins.items()}
    threads = [int(t) for t in a.threads.split(",")]
    runs = {}  # (model, threads, which) -> list of headers
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(a.reps):
            for model in a.models:
                for t in threads:
                    for which, b in bins.items():
                        sol = os.path.join(tmp, f"{which}-{t}-{os.path.basename(model)}.sol")
                        h = solve(b, model, a.algorithm, t, a.time_limit, sol)
                        runs.setdefault((model, t, which), []).append(h)
                        print(f"rep {rep} {os.path.basename(model):26s} t{t} {which:4s} {h.get('status', 'NoOutput'):10s} "
                              f"it {h.get('iterations', ''):>8s} {h.get('seconds', ''):>10s}s", flush=True)
                        if rep == a.reps - 1:
                            rep_v = verify.verify(model, sol, known_optimum(model)) if os.path.exists(sol) else {"verdict": "FAIL"}
                            runs[(model, t, which, "verify")] = rep_v
        rows = []
        for model in a.models:
            size = model_size(a.new, model)
            opt = known_optimum(model)
            for t in threads:
                best = {}
                for which in bins:
                    hs = runs[(model, t, which)]
                    secs = [float(h["seconds"]) for h in hs if "seconds" in h]
                    its = {h.get("iterations") for h in hs}
                    if len(its) != 1:
                        sys.exit(f"non-deterministic iteration count for {model} t{t} {which}: {its}")
                    h = hs[-1]
                    v = runs[(model, t, which, "verify")]
                    best[which] = min(secs) if secs else float("nan")
                    r = dict(info)
                    r.update(git_hash=hashes["new"], base_git_hash=hashes["base"], instance=os.path.basename(model),
                             algorithm=a.algorithm, threads=t, binary=which, binary_git_hash=hashes[which], reps=a.reps,
                             status=h.get("status", "NoOutput"), iterations=h.get("iterations", ""),
                             seconds_min=f"{best[which]:.4f}", seconds_all=" ".join(f"{s:.3f}" for s in secs),
                             objective=h.get("objective", ""), check=(h.get("check") or "").split(" ")[0],
                             verify=v.get("verdict", "FAIL"),
                             verify_primal_rel=f"{v.get('primal_rel', float('nan')):.2e}",
                             verify_dual_rel=f"{v.get('dual_rel', float('nan')):.2e}",
                             verify_gap_rel=f"{v.get('gap_rel', float('nan')):.2e}", message=h.get("message", ""), **size)
                    if opt is not None and r["objective"]:
                        r["known_optimum"] = repr(opt)
                        r["rel_err_known"] = f"{abs(float(r['objective']) - opt) / (1 + abs(opt)):.2e}"
                    rows.append(r)
                for r in rows[-2:]:
                    r["speedup_vs_base"] = f"{best['base'] / best['new']:.3f}" if r["binary"] == "new" else "1.000"
    out = a.out or os.path.join(HERE, "results", f"compare-binaries-{info['machine']}-{hashes['new']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\n{'instance':26s} thr {'base s':>9s} {'new s':>9s} speedup  it base -> new")
    for i in range(0, len(rows), 2):
        b, n = rows[i], rows[i + 1]
        print(f"{b['instance']:26s} {b['threads']:>3} {b['seconds_min']:>9s} {n['seconds_min']:>9s} {n['speedup_vs_base']:>7s}"
              f"  {b['iterations']} -> {n['iterations']}  ({b['status']}/{n['status']}, verify {b['verify']}/{n['verify']})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
