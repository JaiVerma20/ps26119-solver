#!/usr/bin/env python3
"""fetch_netlib.py — download the Netlib LP test set (uncompressed MPS copies kept in the
coin-or/CyLP repository) into data/netlib/ (git-ignored), convert each to .lpm with the
highspy bridge, and write data/netlib/optima.csv from the Netlib readme table
(MINOS 5.3 values) plus the HiGHS optimum as a cross-reference.

usage: tools/fetch_netlib.py [--skip-highs]
"""
import argparse
import csv
import os
import re
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from lpm import read_mps_highspy, write_lpm  # noqa: E402

BASE = "https://raw.githubusercontent.com/coin-or/CyLP/master/cylp/input/netlib/"
NAMES = """25fv47 80bau3b adlittle afiro agg agg2 agg3 bandm beaconfd blend bnl1 bnl2 boeing1 boeing2 bore3d brandy capri cycle czprob d2q06c d6cube degen2 degen3 dfl001 e226 etamacro fffff800 finnis fit1d fit1p fit2d fit2p forplan ganges gfrd-pnc greenbea greenbeb grow15 grow22 grow7 israel kb2 lotfi maros-r7 maros modszk1 nesm perold pilot.ja pilot pilot.we pilot4 pilot87 pilotnov recipe sc105 sc205 sc50a sc50b scagr25 scagr7 scfxm1 scfxm2 scfxm3 scorpion scrs8 scsd1 scsd6 scsd8 sctap1 sctap2 sctap3 seba share1b share2b shell ship04l ship04s ship08l ship08s ship12l ship12s sierra stair standata standgub standmps stocfor1 stocfor2 tuff vtp.base wood1p woodw""".split()  # the .mps files in that folder (listing, 2026-09-24)
OUT = os.path.join(ROOT, "data", "netlib")
README = os.path.join(ROOT, "data", "netlib_small", "README.netlib.txt")



def download(url, dst):
    """curl, not urllib: python.org builds on macOS often lack CA certificates."""
    subprocess.run(["curl", "-sfL", "-o", dst, url], check=True)


def published_optima():
    opt = {}
    pat = re.compile(r"^([A-Z0-9.\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(?:\d+|\(see NOTES\))\s+(?:B\s+)?([-+0-9.E]+)")
    with open(README) as f:
        for line in f:
            mt = pat.match(line)
            if mt:
                opt[mt.group(1).lower()] = float(mt.group(5))
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-highs", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = NAMES
    pub = published_optima()
    rows = []
    for name in names:
        mps = os.path.join(OUT, name + ".mps")
        if not os.path.exists(mps):
            download(BASE + name + ".mps", mps)
        try:
            m = read_mps_highspy(mps)
        except Exception as e:  # noqa: BLE001 — record and continue
            print(f"{name}: highspy cannot read ({e})")
            continue
        lpm = os.path.join(OUT, name + ".lpm")
        if not os.path.exists(lpm):
            write_lpm(m, lpm, source=f"coin-or/CyLP netlib/{name}.mps")
        highs_obj, highs_status = "", ""
        if not a.skip_highs:
            import highspy

            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("time_limit", 120.0)
            h.readModel(mps)
            h.run()
            highs_status = str(h.getModelStatus()).split(".")[-1]
            highs_obj = repr(h.getInfo().objective_function_value) if highs_status == "kOptimal" else ""
        rows.append({"name": name, "rows": m.num_rows, "cols": m.num_cols, "nnz": m.nnz,
                     "published_optimum": repr(pub[name]) if name in pub else "",
                     "highs_status": highs_status, "highs_objective": highs_obj})
        print(f"{name:10s} {m.num_rows:6d} {m.num_cols:6d} {m.nnz:7d} pub {rows[-1]['published_optimum']:>22s} "
              f"highs {highs_status} {highs_obj}", flush=True)
    with open(os.path.join(OUT, "optima.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} models -> {OUT}")


if __name__ == "__main__":
    main()
