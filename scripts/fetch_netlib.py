"""Download Netlib LP benchmark instances into data/netlib/ as plain .mps files.

Source: the COIN-OR Data-Netlib mirror (gzip-compressed MPS), decompressed
here with Python's gzip module. Existing files are skipped. Downloads run in
parallel and use curl when it is available (it is on Windows 10+), which is
much faster than urllib on some networks.

usage:
  python scripts/fetch_netlib.py                  # the "small" and "medium" sets
  python scripts/fetch_netlib.py --set large      # one named set: small | medium | large | all
  python scripts/fetch_netlib.py 25fv47 pilot     # specific instances
"""
import argparse
import concurrent.futures
import gzip
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/{}.mps.gz"

SETS = {
    # up to ~500 rows
    "small": ["afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1",
              "scagr7", "israel", "kb2", "recipe", "boeing2", "lotfi", "e226", "brandy",
              "bore3d", "capri", "scorpion", "sc205", "share1b", "vtpbase", "beaconfd"],
    # ~500 - 2,500 rows
    "medium": ["bnl1", "scfxm1", "ship04s", "ship04l", "25fv47", "degen2", "scsd8", "sctap2",
               "ship08s", "ship12s", "scfxm3", "czprob", "ship08l", "bnl2", "pilot4", "perold"],
    # 2,500+ rows: the hard, ill-conditioned classics
    "large": ["degen3", "d2q06c", "greenbea", "pilot", "80bau3b", "woodw", "fit2p", "maros-r7",
              "pilot87", "dfl001"],
}


def download(url):
    curl = shutil.which("curl")
    if curl:
        proc = subprocess.run([curl, "-sSfL", "--max-time", "300", url], capture_output=True)
        if proc.returncode != 0:
            raise OSError(proc.stderr.decode(errors="replace").strip())
        return proc.stdout
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def fetch(name, out_dir):
    target = os.path.join(out_dir, name + ".mps")
    if os.path.exists(target):
        return True, "  %-10s already present" % name
    try:
        data = gzip.decompress(download(BASE_URL.format(name)))
    except (urllib.error.URLError, OSError) as e:
        return False, "  %-10s FAILED (%s)" % (name, e)
    with open(target, "wb") as f:
        f.write(data)
    return True, "  %-10s %8.1f KB" % (name, len(data) / 1024)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="instance names (default: small + medium sets)")
    ap.add_argument("--set", choices=list(SETS) + ["all"])
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "netlib"))
    args = ap.parse_args()

    if args.names:
        names = args.names
    elif args.set == "all":
        names = SETS["small"] + SETS["medium"] + SETS["large"]
    elif args.set:
        names = SETS[args.set]
    else:
        names = SETS["small"] + SETS["medium"]

    os.makedirs(args.out, exist_ok=True)
    print("downloading %d instance(s) into %s" % (len(names), os.path.abspath(args.out)))
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for success, line in pool.map(lambda n: fetch(n, args.out), names):
            print(line, flush=True)
            ok += success
    print("%d of %d available" % (ok, len(names)))
    return 0 if ok == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
