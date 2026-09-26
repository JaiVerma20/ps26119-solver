"""Run the Layer 2 LU benchmark on every .mps file in a folder and summarise.

For each model: read it with gpuopt, factorize random simplex-style bases
built from its constraint matrix, and verify FTRAN / BTRAN by residuals.

usage:  python scripts/bench_netlib.py [--dir data/netlib] [--exe build/gpuopt.exe] [--trials 1]
"""
import argparse
import glob
import os
import re
import subprocess
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=os.path.join(root, "data", "netlib"))
    ap.add_argument("--exe", default=os.path.join(root, "build", "gpuopt.exe" if os.name == "nt" else "gpuopt"))
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.mps")))
    if not files:
        print("no .mps files in %s - run scripts/fetch_netlib.py first" % args.dir)
        return 1

    rows = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        proc = subprocess.run([args.exe, path, "--lu-bench", "--trials", str(args.trials)],
                              capture_output=True, text=True)
        out = proc.stdout
        size = re.search(r"(\d+) rows, (\d+) columns, (\d+) nonzeros", out)
        verdict = re.search(r"LU-BENCH (\w+)\s+m=\d+\s+worst residual (\S+)", out)
        if not size or not verdict:
            err = (proc.stderr or out).strip().splitlines()
            rows.append((name, 0, 0, 0, 0.0, 0.0, 0.0, "-", "ERROR: " + (err[-1] if err else "no output")))
            continue
        # Basis table rows: rank, nnz(B), nnz(L+U), fill, cond >=, factor ms, ...
        table = re.findall(r"^\s+\S.*?\s(\d+)/(\d+)\s+\d+\s+\d+\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s", out, re.M)
        max_fill = max((float(t[2]) for t in table), default=0.0)
        max_cond = max((float(t[3]) for t in table), default=0.0)
        max_ms = max((float(t[4]) for t in table), default=0.0)
        upd = re.search(r"refactorizations: (\d+) scheduled .*?, (\d+) triggered", out)
        upd_ok = re.search(r"final residual \S+, difference to fresh factorization \S+\s+(\w+)", out)
        note = "" if not upd_ok else "  (updates %s, refactor: %s sched + %s resid)" % (
            upd_ok.group(1), upd.group(1) if upd else "?", upd.group(2) if upd else "?")
        rows.append((name, int(size.group(1)), int(size.group(2)), int(size.group(3)), max_fill, max_cond,
                     max_ms, verdict.group(2), verdict.group(1) + note))

    rows.sort(key=lambda r: r[1])
    print("%-10s %7s %7s %8s %9s %9s %11s %12s   %s" % ("model", "rows", "cols", "nnz", "max fill",
                                                         "max cond", "max fact ms", "worst resid", "verdict"))
    for r in rows:
        print("%-10s %7d %7d %8d %9.2f %9.0e %11.2f %12s   %s" % r)
    passed = sum(1 for r in rows if r[8].startswith("PASS"))
    print("\n%d of %d models PASS" % (passed, len(rows)))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
