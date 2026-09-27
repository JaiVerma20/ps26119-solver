#!/usr/bin/env python3
"""netlib_infeasible.py — the Netlib infeasible LP collection (tools/fetch_netlib_infeasible.py)
through every engine; every Infeasible verdict must carry a certificate that passes the in-process
gate AND tools/verify.py (independent highspy reader; exact rational Farkas test, else the
documented tolerance test). HiGHS's status is recorded as a reference only.

A row counts as CERTIFIED only if: status Infeasible, check PASS, verify PASS. Time limits,
uncertified verdicts and any other status are listed, never dropped. A certified Infeasible on a
model HiGHS calls feasible would be a contradiction and is flagged in the summary.

usage: bench/netlib_infeasible.py [--bin build/ps26119] [--engines simplex,r2hpdhg,auto]
                                  [--time-limit 60] [--dir data/netlib_infeasible] [--only a,b]
Writes bench/results/netlib-infeasible-<engine>-<machine>-<githash>.csv per engine.
"""
import argparse
import csv
import glob
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

FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "instance", "rows", "cols", "nnz",
          "highs_status", "engine", "backend", "time_limit", "status", "check", "verify", "certificate",
          "iterations", "seconds", "message"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--engines", default="simplex,r2hpdhg,auto")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "netlib_infeasible"))
    ap.add_argument("--only", default="")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.dir, "*.mps")))
    if not files:
        sys.exit(f"no .mps files in {a.dir}: run tools/fetch_netlib_infeasible.py first")
    ref = {}
    ref_path = os.path.join(a.dir, "reference.csv")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = {r["name"]: r.get("highs_status", "") for r in csv.DictReader(f)}
    if a.only:
        keep = set(a.only.split(","))
        files = [p for p in files if os.path.splitext(os.path.basename(p))[0] in keep]
    info = machine_info(a.bin)
    summary = {}
    for engine in a.engines.split(","):
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            for path in files:
                name = os.path.splitext(os.path.basename(path))[0]
                sol = os.path.join(tmp, name + ".sol")
                subprocess.run([a.bin, "solve", path, "--algorithm", engine, "--time-limit", str(a.time_limit),
                                "--out", sol], capture_output=True, text=True)
                r = {**info, "instance": name, "highs_status": ref.get(name, ""), "engine": engine, "backend": "cpu",
                     "time_limit": a.time_limit}
                if not os.path.exists(sol):
                    r.update(status="NoOutput", verify="FAIL")
                else:
                    h = read_solution(sol).header
                    r.update(status=h.get("status"), iterations=h.get("iterations"), seconds=h.get("seconds"),
                             check=(h.get("check") or "").split(" ")[0], message=h.get("message", "")[:300])
                    rep = verify.verify(path, sol)
                    r.update(verify=rep["verdict"], certificate=rep.get("certificate", ""), rows=rep["rows"],
                             cols=rep["cols"], nnz=rep["nnz"])
                rows.append(r)
                print(f"{engine:8s} {name:10s} {str(r['status']):15s} check {r.get('check', ''):4s} "
                      f"verify {r['verify']:4s} {r.get('certificate', '')}", flush=True)
        out = os.path.join(a.out_dir, f"netlib-infeasible-{engine}-{info['machine']}-{info['git_hash']}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        cert = [r for r in rows if r["status"] == "Infeasible" and r.get("check") == "PASS" and r["verify"] == "PASS"]
        exact = [r for r in cert if "exact rational" in r.get("certificate", "")]
        contra = [r["instance"] for r in cert if r["highs_status"] and "nfeasible" not in r["highs_status"]]
        summary[engine] = (len(cert), len(rows))
        print(f"\n{engine}: {len(cert)}/{len(rows)} certified infeasible ({len(exact)} exact rational)"
              + (f"; CONTRADICTS HiGHS on {contra}" if contra else "") + f" -> {out}\n")
    return summary


if __name__ == "__main__":
    main()
