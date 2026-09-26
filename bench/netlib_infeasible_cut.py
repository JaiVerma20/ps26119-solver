#!/usr/bin/env python3
"""netlib_infeasible_cut.py — provably infeasible LPs built from the 93 Netlib models, and every
engine's certificate for them.

For each Netlib model with a known optimum f* (HiGHS reference in data/netlib/optima.csv) we add
ONE "objective cut" row
    MIN:  c^T x <= f* - offset - delta        MAX:  c^T x >= f* - offset + delta
with delta = 1e-4 * (1 + |f*|). By LP duality the result is infeasible (no feasible point is
better than the optimum), and a Farkas certificate for it essentially requires the model's optimal
dual solution — a hard, realistic infeasibility proof, not a toy one. The models are written as
.lpm (bench/generated/infeasible_cut/, git-ignored) and solved from those files.

A verdict counts as CERTIFIED only if the engine says Infeasible, the in-process gate passed its
certificate (check=PASS) and tools/verify.py — with its own Python reader of the .lpm file — accepts
the certificate (exact rational Farkas test, else the documented tolerance test). Anything else
(time limit, uncertified, wrong status) is listed, not hidden.

usage: bench/netlib_infeasible_cut.py [--bin build/ps26119] [--engines simplex,r2hpdhg]
                                      [--time-limit 60] [--only afiro,kb2]
Writes bench/results/infeasible-cut-<engine>-<machine>-<githash>.csv per engine.
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
from lpm import read_mps_highspy, read_solution, write_lpm  # noqa: E402
from machine_info import machine_info  # noqa: E402

DATA = os.path.join(ROOT, "data", "netlib")
GEN = os.path.join(ROOT, "bench", "generated", "infeasible_cut")
FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "instance", "rows", "cols", "nnz",
          "highs_optimum", "delta", "engine", "backend", "time_limit", "status", "check", "gate_certificate", "verify",
          "certificate",
          "iterations", "seconds", "message"]


def cut_model(mps_path, fstar, out_path):
    """Write the model plus the objective-cut row as .lpm; returns (rows, cols, nnz, delta)."""
    m = read_mps_highspy(mps_path)
    delta = 1e-4 * (1.0 + abs(fstar))
    n, r = m.num_cols, m.num_rows
    cols = [[] for _ in range(n)]
    for j in range(n):
        for k in range(m.col_start[j], m.col_start[j + 1]):
            cols[j].append((m.row_index[k], m.value[k]))
        if m.obj[j] != 0.0:
            cols[j].append((r, m.obj[j]))  # the new last row
    m.col_start, m.row_index, m.value = [0], [], []
    for j in range(n):
        for i, v in cols[j]:
            m.row_index.append(i)
            m.value.append(v)
        m.col_start.append(len(m.value))
    if m.sense > 0:
        m.row_lower.append(float("-inf"))
        m.row_upper.append(fstar - m.obj_offset - delta)
    else:
        m.row_lower.append(fstar - m.obj_offset + delta)
        m.row_upper.append(float("inf"))
    m.num_rows = r + 1
    if m.row_names:
        m.row_names.append("OBJCUT")
    write_lpm(m, out_path, source=f"{os.path.basename(mps_path)} + objective cut (delta {delta:.3g})")
    return m.num_rows, n, len(m.value), delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--engines", default="simplex,r2hpdhg")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    optima = os.path.join(DATA, "optima.csv")
    if not os.path.exists(optima):
        sys.exit("run tools/fetch_netlib.py first")
    with open(optima) as f:
        models = [r for r in csv.DictReader(f) if r["highs_status"] == "kOptimal"]
    if a.only:
        keep = set(a.only.split(","))
        models = [r for r in models if r["name"] in keep]
    os.makedirs(GEN, exist_ok=True)
    info = machine_info(a.bin)
    cases = []
    for mdl in models:
        lpm = os.path.join(GEN, mdl["name"] + "-cut.lpm")
        fstar = float(mdl["highs_objective"])
        rows, cols, nnz, delta = cut_model(os.path.join(DATA, mdl["name"] + ".mps"), fstar, lpm)
        cases.append((mdl["name"], lpm, rows, cols, nnz, fstar, delta))
    for engine in a.engines.split(","):
        out_rows = []
        with tempfile.TemporaryDirectory() as tmp:
            for name, lpm, rows, cols, nnz, fstar, delta in cases:
                sol = os.path.join(tmp, name + ".sol")
                subprocess.run([a.bin, "solve", lpm, "--algorithm", engine, "--time-limit", str(a.time_limit),
                                "--out", sol], capture_output=True, text=True)
                r = {**info, "instance": name, "rows": rows, "cols": cols, "nnz": nnz, "highs_optimum": fstar,
                     "delta": f"{delta:.3g}", "engine": engine, "backend": "cpu", "time_limit": a.time_limit}
                if not os.path.exists(sol):
                    r.update(status="NoOutput", verify="FAIL")
                else:
                    h = read_solution(sol).header
                    msg = h.get("message", "")
                    # "(rounding-proof)" exactly: the tolerance message says "..., not rounding-proof"
                    gate = ("rounding-proof" if "(rounding-proof)" in msg else
                            "tolerance" if "within tolerance" in msg else "")
                    r.update(status=h.get("status"), iterations=h.get("iterations"), seconds=h.get("seconds"),
                             check=(h.get("check") or "").split(" ")[0], gate_certificate=gate, message=msg[:300])
                    rep = verify.verify(lpm, sol)
                    r.update(verify=rep["verdict"], certificate=rep.get("certificate", ""))
                out_rows.append(r)
                print(f"{engine:8s} {name:10s} {r['status']:15s} check {r.get('check', ''):4s} verify {r['verify']:4s} "
                      f"{r.get('certificate', ''):32s} it {str(r.get('iterations', '')):>8s} t {r.get('seconds', '')}",
                      flush=True)
        out = os.path.join(HERE, "results", f"infeasible-cut-{engine}-{info['machine']}-{info['git_hash']}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        cert = [r for r in out_rows if r["status"] == "Infeasible" and r.get("check") == "PASS" and r["verify"] == "PASS"]
        exact = [r for r in cert if "exact rational" in r.get("certificate", "")]
        print(f"\n{engine}: {len(cert)}/{len(out_rows)} certified infeasible (gate + independent verifier), "
              f"{len(exact)} of them by the exact rational Farkas test -> {out}\n")


if __name__ == "__main__":
    main()
