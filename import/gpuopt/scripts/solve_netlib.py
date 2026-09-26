"""Solve every Netlib model in a folder and score it against the published optimum.

For each .mps file with a known optimal value (data/netlib/optimal_values.csv)
this runs `gpuopt <file> --expect <value>` and reports status, objective,
relative error, iterations, time and the independent checker's verdict.

A model counts as SOLVED when the status is OPTIMAL, the objective matches the
published value to 1e-6 (relative) and the checker certifies optimality.

A few published values in the Netlib readme are known to be inaccurate or to
leave out the objective constant. When our certified optimum disagrees with
the readme, the model is re-solved with HiGHS (via SciPy, an external
baseline) and the verdict says which one it agrees with.

usage:  python scripts/solve_netlib.py [--dir data/netlib] [--timeout 600] [--highs] [--csv results.csv] [models...]
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import highs_reference  # noqa: E402


def load_optima(path):
    optima = {}
    with open(path) as f:
        for row in csv.reader(line for line in f if not line.startswith("#")):
            if row and row[0] != "name":
                optima[row[0].lower()] = float(row[1])
    return optima


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", help="model names (default: every .mps in --dir)")
    ap.add_argument("--dir", default=os.path.join(root, "data", "netlib"))
    ap.add_argument("--exe", default=os.path.join(root, "build", "gpuopt.exe" if os.name == "nt" else "gpuopt"))
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per model")
    ap.add_argument("--csv", help="also write the results table to this CSV file")
    ap.add_argument("--extra", default="", help="extra gpuopt options, e.g. \"--pricing dantzig\"")
    ap.add_argument("--highs", action="store_true", help="also time every model with HiGHS for comparison")
    args = ap.parse_args()

    optima = load_optima(os.path.join(args.dir, "optimal_values.csv"))
    if args.models:
        files = [os.path.join(args.dir, m + ".mps") for m in args.models]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.mps")))

    rows = []
    fmt = "%-10s %6s %6s  %-15s %17s %9s %9s %9s %8s  %s"
    print(fmt % ("model", "rows", "cols", "status", "objective", "rel err", "iters", "time s", "HiGHS s",
                 "verdict"))
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name.lower() not in optima:
            continue
        expect = optima[name.lower()]
        cmd = [args.exe, path, "--expect", repr(expect), "--time-limit", str(args.timeout)] + args.extra.split()
        t0 = time.time()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout + 30).stdout
        except subprocess.TimeoutExpired:
            out = ""
        wall = time.time() - t0

        def grab(pattern, default=""):
            mt = re.search(pattern, out, re.M)
            return mt.group(1) if mt else default

        size = re.search(r"(\d+) rows, (\d+) columns", out)
        status = grab(r"^status\s+(\S+)", "TIMEOUT" if not out else "ERROR")
        objective = grab(r"^objective\s+(\S+)")
        rel = grab(r"relative error (\S+)")
        iters = grab(r"^iterations\s+(\d+)")
        checker = "PASS" in grab(r"^checker\s+(.*)$")
        match = "=> MATCH" in out  # (plain "MATCH" would also match "MISMATCH")
        note = ""
        # Netlib's published values leave out the objective constant (RHS entry
        # on the objective row); compare that way too, and say so.
        without_offset = grab(r"without it: (\S+)\)")
        if not match and without_offset:
            if abs(float(without_offset) - expect) <= 1e-6 * (1 + abs(expect)):
                match, note = True, " (published value excludes the objective constant)"
        # DFL001's published value has only 6 significant digits.
        if not match and name.lower() == "dfl001" and objective:
            match = abs(float(objective) - expect) <= 5e-6 * abs(expect)

        # HiGHS: always when timing is requested, otherwise only to settle a disagreement.
        highs_s, highs_value = "", None
        if args.highs or (status == "OPTIMAL" and checker and not match):
            h_status, highs_value, h_sec = highs_reference.solve(path)
            highs_s = "%.2f" % h_sec
        if status == "OPTIMAL" and checker and not match and highs_value is not None:
            if abs(float(objective) - highs_value) <= 1e-7 * (1 + abs(highs_value)):
                match, note = True, " (matches HiGHS %.10e; Netlib readme value differs)" % highs_value

        solved = status == "OPTIMAL" and match and checker
        verdict = ("SOLVED" + note) if solved else ("MISMATCH" if status == "OPTIMAL" else "not solved")
        if status == "OPTIMAL" and match and not checker:
            verdict = "objective ok, checker FAIL"
        row = [name, size.group(1) if size else "", size.group(2) if size else "", status, objective, rel,
               iters, "%.2f" % wall, highs_s, verdict]
        rows.append(row)
        print(fmt % tuple(row), flush=True)

    solved = sum(1 for r in rows if r[9].startswith("SOLVED"))
    via_highs = sum(1 for r in rows if "matches HiGHS" in r[9])
    via_offset = sum(1 for r in rows if "objective constant" in r[9])
    print("\nsolved %d of %d models: status OPTIMAL + independent checker PASS + objective equal to the "
          "published value (1e-6) or, where noted, to HiGHS" % (solved, len(rows)))
    if via_highs or via_offset:
        print("  of which %d match HiGHS where the Netlib readme value differs, and %d differ from the readme only"
              " by the objective constant" % (via_highs, via_offset))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "rows", "cols", "status", "objective", "rel_error", "iterations", "seconds",
                        "highs_seconds", "verdict"])
            w.writerows(rows)
        print("results written to %s" % args.csv)
    return 0 if solved == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
