#!/usr/bin/env python3
"""batch.py — batched scenarios (CLAUDE.md §9 idea 4): K crude/product price scenarios of the
refinery planning LP solved (a) one after another with `ps26119 solve` and (b) together with
`ps26119 batch` (one SpMM per iteration instead of K SpMVs). Every batch answer is checked by
tools/verify.py and must agree with its sequential twin.

usage: bench/batch.py [--bin build/ps26119] [--periods 365] [--k 4,8,16] [--time-limit 1200]
Writes bench/results/batch-<machine>-<githash>.csv.

Honest notes: all scenarios run until the slowest one converges (converged ones are frozen
but still occupy their SpMM column); CPU is single-threaded; "sequential" is the sum of the
in-process solve times of K separate runs (file reading excluded on both sides).
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
import warm_start  # noqa: E402  (scenario perturbations)
from lpm import read_lpm, read_solution, write_lpm  # noqa: E402
from machine_info import machine_info  # noqa: E402

FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "instance", "rows", "cols", "nnz", "K",
          "sequential_seconds", "batch_seconds", "speedup", "sequential_iterations_max", "batch_iterations",
          "all_optimal", "all_verified", "max_objective_disagreement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--periods", default="365")
    ap.add_argument("--k", default="4,8,16")
    ap.add_argument("--time-limit", type=float, default=1200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    info = machine_info()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for T in [int(t) for t in a.periods.split(",")]:
            base = os.path.join(HERE, "generated", f"refinery-T{T}-s1.lpm")
            if not os.path.exists(base):
                subprocess.run([sys.executable, os.path.join(HERE, "generate_refinery_lp.py"), "--periods", str(T),
                                "--out", base], check=True)
            bm = read_lpm(base)
            for K in [int(k) for k in a.k.split(",")]:
                paths = []
                for s in range(K):
                    p = os.path.join(tmp, f"T{T}-price{s}.lpm")
                    if not os.path.exists(p):
                        write_lpm(warm_start.perturb(read_lpm(base), "price", T, seed=100 + s), p)
                    paths.append(p)
                seq_t, seq_it, seq_obj = 0.0, 0, []
                for p in paths:
                    sol = p[:-4] + ".seq.sol"
                    subprocess.run([a.bin, "solve", p, "--tol", "1e-8", "--time-limit", str(a.time_limit), "--out", sol],
                                   capture_output=True)
                    h = read_solution(sol).header
                    seq_t += float(h["seconds"])
                    seq_it = max(seq_it, int(h["iterations"]))
                    seq_obj.append((h["status"], float(h["objective"])))
                outdir = os.path.join(tmp, f"b{T}-{K}")
                os.makedirs(outdir, exist_ok=True)
                subprocess.run([a.bin, "batch", base] + paths + ["--out-dir", outdir, "--tol", "1e-8", "--time-limit",
                                                                 str(a.time_limit)], capture_output=True)
                bt, bit, ok_opt, ok_ver, dis = 0.0, 0, True, True, 0.0
                for p, (sst, sobj) in zip(paths, seq_obj):
                    sol = os.path.join(outdir, os.path.basename(p)[:-4] + ".sol")
                    h = read_solution(sol).header
                    bt = max(bt, float(h["seconds"]))
                    bit = max(bit, int(h["iterations"]))
                    ok_opt &= h["status"] == "Optimal" and sst == "Optimal"
                    ok_ver &= verify.verify(p, sol)["verdict"] == "PASS"
                    dis = max(dis, abs(float(h["objective"]) - sobj) / (1 + abs(sobj)))
                r = {**info, "instance": f"refinery-T{T}-s1", "rows": bm.num_rows, "cols": bm.num_cols, "nnz": bm.nnz,
                     "K": K, "sequential_seconds": f"{seq_t:.3f}", "batch_seconds": f"{bt:.3f}",
                     "speedup": f"{seq_t / bt:.2f}" if bt > 0 else "", "sequential_iterations_max": seq_it,
                     "batch_iterations": bit, "all_optimal": ok_opt, "all_verified": ok_ver,
                     "max_objective_disagreement": f"{dis:.1e}"}
                rows.append(r)
                print(f"T={T} K={K:3d}: sequential {seq_t:8.2f}s  batch {bt:8.2f}s  speed-up {r['speedup']}x  "
                      f"optimal {ok_opt} verified {ok_ver} disagreement {dis:.1e}", flush=True)
    out = a.out or os.path.join(HERE, "results", f"batch-{info['machine']}-{info['git_hash']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
