#!/usr/bin/env python3
"""scale.py — scaling evidence: CPU vs GPU, fp64 vs mixed, on generated LPs whose optimum
is known by construction (random sparse, 1e4…1e6 rows) and on the T-period refinery LP
(T = 12, 365, 8760).

For every instance × backend × precision it runs the engine once to 1e-8 and records the
1e-4 milestone from the same run: iterations and wall time to 1e-4 and 1e-8, setup time,
time per iteration, objective vs the constructed optimum, and the independent verifier's
verdict (tools/verify.py). Writes bench/results/scale-<machine>-<githash>.csv and a PNG.

usage: bench/scale.py [--bin build/ps26119] [--gpu] [--sizes 1e4,1e5,1e6]
                      [--refinery 12,365,8760] [--engines r2hpdhg] [--time-limit 600]
                      [--highs]   (also time HiGHS as the established reference solver)
Honest reporting: rows where the GPU is slower than the CPU are printed as such.
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
import lpgen  # noqa: E402
import verify  # noqa: E402
from lpm import read_solution  # noqa: E402
from machine_info import machine_info  # noqa: E402

GEN = os.path.join(HERE, "generated")
FIELDS = ["git_hash", "machine", "cpu", "gpu", "driver", "cuda", "date", "instance", "family", "rows", "cols", "nnz",
          "engine", "backend", "threads", "precision", "tolerance", "status", "iterations", "seconds", "setup_seconds",
          "ms_per_iteration", "iterations_to_1e-4", "seconds_to_1e-4", "seconds_to_1e-8", "objective",
          "known_optimum", "rel_err_known", "verify", "verify_primal_rel", "verify_dual_rel", "verify_gap_rel",
          "message"]


def ensure_instance(kind, size, seed=1):
    os.makedirs(GEN, exist_ok=True)
    if kind == "rand":
        path = os.path.join(GEN, f"rand-{size}-s{seed}.lpm")
        cmd = [sys.executable, os.path.join(HERE, "generate_lp.py"), "--rows", str(size), "--seed", str(seed), "--out", path]
    else:
        path = os.path.join(GEN, f"refinery-T{size}-s{seed}.lpm")
        cmd = [sys.executable, os.path.join(HERE, "generate_refinery_lp.py"), "--periods", str(size), "--seed", str(seed),
               "--out", path]
    if not (os.path.exists(path) and os.path.exists(os.path.splitext(path)[0] + ".json")):
        subprocess.run(cmd, check=True)
    return path


def run(binary, path, engine, precision, gpu, time_limit, tmp, verify_max_nnz, threads=1):
    info = lpgen.read_sidecar(path)
    sol = os.path.join(tmp, "s.sol")
    if os.path.exists(sol):
        os.remove(sol)
    cmd = [binary, "solve", path, "--algorithm", engine, "--precision", precision, "--tol", "1e-8",
           "--time-limit", str(time_limit), "--out", sol, "--threads", str(threads)] + (["--gpu"] if gpu else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    row = {"instance": info["name"], "family": "refinery" if "refinery" in info["generator"] else "random",
           "rows": info["rows"], "cols": info["cols"], "nnz": info["nnz"], "engine": engine,
           "backend": "gpu" if gpu else "cpu", "threads": threads if not gpu else "", "precision": precision,
           "tolerance": "1e-8",
           "known_optimum": repr(info["optimum"])}
    if not os.path.exists(sol):
        row.update(status="NoOutput", message=(p.stderr or p.stdout).strip()[-200:])
        return row
    s = read_solution(sol)
    h = s.header
    it = int(h.get("iterations", 0))
    secs = float(h.get("seconds", "nan"))
    setup = float(h.get("setup_seconds", 0))
    obj = float(h.get("objective", "nan"))
    row.update(status=h.get("status"), iterations=it, seconds=f"{secs:.4f}", setup_seconds=f"{setup:.4f}",
               ms_per_iteration=f"{1000 * (secs - setup) / it:.4f}" if it else "",
               objective=repr(obj), rel_err_known=f"{abs(obj - info['optimum']) / (1 + abs(info['optimum'])):.2e}",
               message=h.get("message", ""))
    row["iterations_to_1e-4"] = h.get("iterations_to_fast", "")
    row["seconds_to_1e-4"] = h.get("seconds_to_fast", "")
    row["seconds_to_1e-8"] = f"{secs:.4f}" if h.get("status") == "Optimal" else ""
    if info["nnz"] <= verify_max_nnz and s.x:
        rep = verify.verify(path, sol, info["optimum"])
        row.update(verify=rep["verdict"], verify_primal_rel=f"{rep.get('primal_rel', float('nan')):.2e}",
                   verify_dual_rel=f"{rep.get('dual_rel', float('nan')):.2e}",
                   verify_gap_rel=f"{rep.get('gap_rel', float('nan')):.2e}")
    else:
        row["verify"] = "skipped"
    return row


def run_highs(path, time_limit, tmp):
    """HiGHS as a separate process with a HARD wall-clock kill: HiGHS's own time_limit is
    not checked in every phase (seen on the T=8760 refinery LP, which ran far past it)."""
    import json
    import time

    info = lpgen.read_sidecar(path)
    out_json = os.path.join(tmp, "h.json")
    code = ("import json,sys; sys.path.insert(0, %r); from highs_ref import solve_with_highs; "
            "r = solve_with_highs(%r, %r, %r); json.dump(r, open(%r, 'w'))"
            % (os.path.join(ROOT, "tools"), path, os.path.join(tmp, "h.sol"), float(time_limit), out_json))
    t0 = time.perf_counter()
    try:
        subprocess.run([sys.executable, "-c", code], timeout=time_limit + 30, capture_output=True)
        with open(out_json) as f:
            r = json.load(f)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        r = {"status": "TimeLimit", "iterations": "", "seconds": time.perf_counter() - t0, "objective": float("nan")}
    if os.path.exists(out_json):
        os.remove(out_json)
    base = {"instance": info["name"], "family": "refinery" if "refinery" in info["generator"] else "random",
            "rows": info["rows"], "cols": info["cols"], "nnz": info["nnz"], "engine": "highs", "backend": "cpu",
            "precision": "fp64", "tolerance": "HiGHS default", "status": r["status"],
            "iterations": r["iterations"], "seconds": f"{r['seconds']:.4f}", "known_optimum": repr(info["optimum"]),
            "verify": "reference"}
    if r["status"] == "Optimal":
        base.update({"objective": repr(r["objective"]), "seconds_to_1e-8": f"{r['seconds']:.4f}",
                     "rel_err_known": f"{abs(r['objective'] - info['optimum']) / (1 + abs(info['optimum'])):.2e}"})
    return base


def plot(rows, png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: no PNG")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for r in rows:  # a CPU thread count is part of the configuration
        if r.get("threads") not in (None, "", "1") and r["backend"] == "cpu":
            r["backend"] = f"cpu{r['threads']}t"
    configs = sorted({(r["engine"], r["backend"], r["precision"]) for r in rows})
    # one fixed colour per configuration so every panel matches the legend
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = {cfg: palette[i % len(palette)] for i, cfg in enumerate(configs)}
    for fam, marker, ls in (("random", "o", "-"), ("refinery", "s", "--")):
        for cfg in configs:
            pts = [r for r in rows if (r["engine"], r["backend"], r["precision"]) == cfg and r["family"] == fam]
            if not pts:
                continue
            pts.sort(key=lambda r: int(r["nnz"]))
            label = f"{cfg[0]} {cfg[1]} {cfg[2]} ({fam})"
            nnz = [int(r["nnz"]) for r in pts]
            if cfg[0] != "highs":
                mpi = [float(r["ms_per_iteration"]) if r.get("ms_per_iteration") else float("nan") for r in pts]
                axes[0].loglog(nnz, mpi, marker=marker, ls=ls, color=color[cfg], label=label)
                t4 = [float(r["seconds_to_1e-4"]) if r.get("seconds_to_1e-4") else float("nan") for r in pts]
                axes[1].loglog(nnz, t4, marker=marker, ls=ls, color=color[cfg], label=label)
            t8 = [float(r["seconds_to_1e-8"]) if r.get("seconds_to_1e-8") else float("nan") for r in pts]
            axes[2].loglog(nnz, t8, marker=marker, ls=ls, color=color[cfg], label=label)
    axes[0].set_title("time per iteration (ms)")
    axes[1].set_title("wall time to 1e-4 (s)")
    axes[2].set_title("wall time to 1e-8 (s)  [missing = limit hit]")
    for ax in axes:
        ax.set_xlabel("nonzeros")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(f"{rows[0]['machine']} — {rows[0]['cpu']} — GPU: {rows[0]['gpu']} — commit {rows[0]['git_hash']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"chart -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "ps26119"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sizes", default="1e4,1e5,1e6")
    ap.add_argument("--refinery", default="12,365,8760")
    ap.add_argument("--engines", default="r2hpdhg")
    ap.add_argument("--precisions", default="fp64,mixed")
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--verify-max-nnz", type=float, default=3e6)
    ap.add_argument("--highs", action="store_true")
    ap.add_argument("--threads", default="1", help="comma list of CPU thread counts, e.g. 1,4,10 (0 = all cores)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--replot", default=None, help="only redraw the PNG from an existing CSV")
    a = ap.parse_args()
    if a.replot:
        with open(a.replot, newline="") as f:
            plot(list(csv.DictReader(f)), os.path.splitext(a.replot)[0] + ".png")
        return
    info = machine_info()
    instances = [ensure_instance("rand", int(float(s))) for s in a.sizes.split(",") if s] + \
                [ensure_instance("refinery", int(float(t))) for t in a.refinery.split(",") if t]
    backends = [False, True] if a.gpu else [False]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for path in instances:
            for engine in a.engines.split(","):
                for gpu, th in [(g, t) for g in backends for t in (["1"] if g else a.threads.split(","))]:
                    for prec in a.precisions.split(","):
                        r = run(a.bin, path, engine, prec, gpu, a.time_limit, tmp, a.verify_max_nnz, int(th))
                        r.update(info)
                        rows.append(r)
                        print(f"{r['instance']:22s} {engine:8s} {r['backend']:3s}{str(r.get('threads', '')):>3s} {prec:5s} {str(r.get('status')):14s} "
                              f"it {str(r.get('iterations', '')):>8s} t {str(r.get('seconds', '')):>9s}s "
                              f"(1e-4: {str(r.get('seconds_to_1e-4', '')):>9s}s) ms/it {str(r.get('ms_per_iteration', '')):>8s} "
                              f"err {r.get('rel_err_known', '')} verify {r.get('verify', '')}", flush=True)
            if a.highs:
                r = run_highs(path, a.time_limit, tmp)
                r.update(info)
                rows.append(r)
                print(f"{r['instance']:22s} highs    cpu fp64  {r['status']:14s} t {r['seconds']:>9s}s", flush=True)

    tag = info["machine"]
    out = a.out or os.path.join(HERE, "results", f"scale-{tag}-{info['git_hash']}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"csv -> {out}")
    plot(rows, os.path.splitext(out)[0] + ".png")

    if a.gpu:  # honest GPU vs CPU summary (same engine and precision)
        print("\nGPU vs CPU (wall time to 1e-4 / 1e-8; >1 = GPU faster):")
        for r in rows:
            if r["backend"] != "gpu":
                continue
            c = next((x for x in rows if x["backend"] == "cpu" and x["instance"] == r["instance"]
                      and x["engine"] == r["engine"] and x["precision"] == r["precision"]), None)
            if not c:
                continue
            def ratio(k):
                try:
                    return f"{float(c[k]) / float(r[k]):6.2f}x"
                except (ValueError, KeyError, ZeroDivisionError):
                    return "   n/a"
            verdict = []
            for k in ("seconds_to_1e-4", "seconds_to_1e-8"):
                try:
                    if float(r[k]) > float(c[k]):
                        verdict.append(f"GPU LOSES at {k[-4:]}")
                except (ValueError, KeyError):
                    pass
            print(f"  {r['instance']:22s} {r['engine']:8s} {r['precision']:5s} 1e-4 {ratio('seconds_to_1e-4')}  "
                  f"1e-8 {ratio('seconds_to_1e-8')}  {' '.join(verdict)}")


if __name__ == "__main__":
    main()
