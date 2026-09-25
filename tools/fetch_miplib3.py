#!/usr/bin/env python3
"""fetch_miplib3.py — download small MIPLIB 3 instances (coin-or-tools/Data-miplib3, MPS.gz)
into data/miplib3/ (git-ignored), convert to .lpm with the highspy bridge and record the
HiGHS optimum (reference) in data/miplib3/optima.csv.

usage: tools/fetch_miplib3.py [names...]   (default: a set of small instances)
"""
import csv
import gzip
import os
import shutil
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from lpm import read_mps_highspy, write_lpm  # noqa: E402

BASE = "https://raw.githubusercontent.com/coin-or-tools/Data-miplib3/master/"
OUT = os.path.join(ROOT, "data", "miplib3")
DEFAULT = "p0033 flugpl egout enigma lseu mod008 stein27 pk1 gt2 rgn bell5 bell3a misc03 p0201".split()



def download(url, dst):
    """curl, not urllib: python.org builds on macOS often lack CA certificates."""
    subprocess.run(["curl", "-sfL", "-o", dst, url], check=True)


def main(argv):
    names = argv or DEFAULT
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name in names:
        mps = os.path.join(OUT, name + ".mps")
        if not os.path.exists(mps):
            gz = mps + ".gz"
            download(BASE + name + ".gz", gz)
            with gzip.open(gz, "rb") as fi, open(mps, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz)
        m = read_mps_highspy(mps)
        write_lpm(m, os.path.join(OUT, name + ".lpm"), source=f"coin-or-tools/Data-miplib3/{name}.gz")
        import highspy

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", 300.0)
        h.readModel(mps)
        h.run()
        st = str(h.getModelStatus()).split(".")[-1]
        obj = h.getInfo().objective_function_value if st == "kOptimal" else ""
        rows.append({"name": name, "rows": m.num_rows, "cols": m.num_cols, "nnz": m.nnz,
                     "integers": sum(1 for v in m.is_integer if v), "highs_status": st, "highs_objective": obj})
        print(rows[-1], flush=True)
    with open(os.path.join(OUT, "optima.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
