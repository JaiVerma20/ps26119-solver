"""Differential test: gpuopt vs SciPy/HiGHS on random LPs written as MPS files.

Generates random LPs with every row type (E, L, G, ranged) and bound type
(default, boxed, MI, FR, LO), writes each one as a free-format MPS file, runs
the gpuopt executable on it (exercising the MPS reader end to end), and
compares the status and objective against scipy.optimize.linprog.

SciPy/HiGHS is used ONLY as an external comparison baseline, never linked
into gpuopt.

If the two disagree, HiGHS is re-run with presolve disabled: HiGHS presolve
sometimes labels an unbounded LP as infeasible, and the tie-break shows who
is right.

usage:  python scripts/crosscheck_scipy.py [--exe build/gpuopt.exe] [--count 400] [--seed 7]
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile

from scipy.optimize import linprog

INF = float("inf")
STATUS = {0: "OPTIMAL", 2: "INFEASIBLE", 3: "UNBOUNDED"}


def random_model(rng):
    m, n = rng.randint(1, 7), rng.randint(1, 8)
    A = [[rng.choice([0, 0, rng.randint(-5, 5)]) for _ in range(n)] for _ in range(m)]
    c = [rng.randint(-5, 5) for _ in range(n)]
    maximize = rng.random() < 0.5
    bounds = []
    for _ in range(n):
        b = rng.randint(-4, 4)
        bounds.append(rng.choice([(0, INF), (b, b + rng.randint(1, 6)), (-INF, b), (-INF, INF), (b, INF)]))
    # 70% of models are built around a point x0 inside the bounds, so they are
    # feasible by construction; the rest use arbitrary right-hand sides.
    feasible = rng.random() < 0.7
    x0 = []
    for lo, up in bounds:
        if lo == -INF and up == INF:
            x0.append(rng.randint(-3, 3))
        elif lo == -INF:
            x0.append(up - rng.randint(0, 3))
        elif up == INF:
            x0.append(lo + rng.randint(0, 3))
        else:
            x0.append(rng.randint(lo, up))
    rows = []  # (type, rhs, range)
    for a in A:
        ax0 = sum(v * x for v, x in zip(a, x0))
        slack = rng.randint(0, 3) if feasible else 0
        b = ax0 if feasible else rng.randint(-6, 6)
        rows.append(rng.choice([("E", b, 0), ("L", b + slack, 0), ("G", b - slack, 0),
                                ("R", b + slack, 2 * slack + rng.randint(1, 3))]))
    return A, c, maximize, bounds, rows


def write_mps(path, name, A, c, maximize, bounds, rows):
    m, n = len(A), len(c)
    lines = ["NAME " + name]
    if maximize:
        lines += ["OBJSENSE", "    MAX"]
    lines += ["ROWS", " N obj"]
    lines += [" %s r%d" % ("L" if t == "R" else t, i) for i, (t, _, _) in enumerate(rows)]
    lines += ["COLUMNS"]
    for j in range(n):
        lines.append("    x%d obj %r" % (j, float(c[j])))
        lines += ["    x%d r%d %r" % (j, i, float(A[i][j])) for i in range(m) if A[i][j]]
    lines += ["RHS"] + ["    rhs r%d %r" % (i, float(b)) for i, (_, b, _) in enumerate(rows)]
    lines += ["RANGES"] + ["    rng r%d %r" % (i, float(R)) for i, (t, _, R) in enumerate(rows) if t == "R"]
    lines += ["BOUNDS"]
    for j, (lo, up) in enumerate(bounds):
        if lo == -INF and up == INF:
            lines.append(" FR bnd x%d" % j)
            continue
        if lo == -INF:
            lines.append(" MI bnd x%d" % j)
        elif lo != 0:
            lines.append(" LO bnd x%d %r" % (j, float(lo)))
        if up != INF:
            lines.append(" UP bnd x%d %r" % (j, float(up)))
    lines += ["ENDATA"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def scipy_solve(A, c, maximize, bounds, rows, presolve=True):
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for a, (t, b, R) in zip(A, rows):
        neg = [-v for v in a]
        if t == "E":
            A_eq.append(a); b_eq.append(b)
        elif t == "L":
            A_ub.append(a); b_ub.append(b)
        elif t == "G":
            A_ub.append(neg); b_ub.append(-b)
        else:  # L row with range R: b - R <= a x <= b
            A_ub += [a, neg]; b_ub += [b, -(b - R)]
    r = linprog([-v for v in c] if maximize else c,
                A_ub=A_ub or None, b_ub=b_ub or None, A_eq=A_eq or None, b_eq=b_eq or None,
                bounds=[(None if lo == -INF else lo, None if up == INF else up) for lo, up in bounds],
                options={"presolve": presolve})
    status = STATUS.get(r.status, "OTHER")
    objective = (-r.fun if maximize else r.fun) if status == "OPTIMAL" else None
    return status, objective


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_exe = os.path.join(here, "..", "build", "gpuopt.exe" if os.name == "nt" else "gpuopt")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exe", default=default_exe)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tally = {}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(args.count):
            model = random_model(rng)
            path = os.path.join(tmp, "r%d.mps" % t)
            write_mps(path, "R%d" % t, *model)

            out = subprocess.run([args.exe, path], capture_output=True, text=True).stdout
            ours = re.search(r"^status\s+(\S+)", out, re.M).group(1)
            obj = re.search(r"^objective\s+(\S+)", out, re.M)
            if "checker" in out and "PASS" not in out:
                failures.append((t, "checker FAIL"))

            ref, ref_obj = scipy_solve(*model)
            if ours != ref:
                ref, ref_obj = scipy_solve(*model, presolve=False)
                print("model %d: HiGHS presolve disagreed; without presolve HiGHS says %s, gpuopt says %s"
                      % (t, ref, ours))
            ok = ours == ref
            if ok and ours == "OPTIMAL":
                ok = abs(float(obj.group(1)) - ref_obj) <= 1e-7 * (1 + abs(ref_obj))
            if not ok:
                failures.append((t, ours, ref))
            tally[ours] = tally.get(ours, 0) + 1

    print("statuses: " + ", ".join("%s %d" % kv for kv in sorted(tally.items())))
    print("agreement with SciPy/HiGHS: %d / %d" % (args.count - len(failures), args.count))
    for f in failures[:20]:
        print("  MISMATCH", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
