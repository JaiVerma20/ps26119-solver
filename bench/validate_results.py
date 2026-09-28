#!/usr/bin/env python3
"""validate_results.py — provenance and integrity checks for benchmark evidence before it is used
(CLAUDE.md §5.1: no claim unless it comes from a hash-named CSV). Meant especially for results
produced on other machines (the GPU runs of scripts/gpu_check.sh).

For every CSV (default: all of bench/results/*.csv):
  * the git hash in the file name equals the git_hash column of every row ("-dirty" only when the
    file name says so), and that commit exists in this repository's history;
  * provenance columns are present and filled: git_hash, machine, cpu, date;
  * rows with backend=gpu name the GPU, the driver and the CUDA version;
  * no row claims Optimal with verify=FAIL.
For every log folder bench/results/logs/<machine>-<hash>/ (written by gpu_check.sh):
  * ctest.log exists and says "100% tests passed" (otherwise the failing tests are listed);
  * each sanitizer-*.log reports "ERROR SUMMARY: 0 errors";
  * nvidia-smi.txt exists.
Exit status 0 when everything passes, 1 otherwise. Warnings (e.g. a dirty tree) do not fail.

usage: bench/validate_results.py [--csv-only] [files or folders ...]
"""
import csv
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HASH_RE = re.compile(r"-([0-9a-f]{7,12})(-dirty)?\.csv$")


def commit_exists(h):
    return subprocess.run(["git", "cat-file", "-e", h + "^{commit}"], cwd=ROOT, capture_output=True).returncode == 0


def check_csv(path):
    errors, warnings = [], []
    m = HASH_RE.search(os.path.basename(path))
    if not m:
        return [f"file name carries no git hash"], warnings
    fhash, fdirty = m.group(1), bool(m.group(2))
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ["empty CSV"], warnings
    cols = rows[0].keys()
    for c in ("git_hash",):
        if c not in cols:
            errors.append(f"missing column {c}")
    for c in ("machine", "cpu", "date"):
        if c not in cols:
            warnings.append(f"missing provenance column {c} (older CSV format)")
    hashes = {r.get("git_hash", "") for r in rows}
    for h in hashes:
        base = h.replace("-dirty", "")
        if base != fhash:
            errors.append(f"git_hash column {h!r} differs from the file name's {fhash!r}")
        if h.endswith("-dirty") and not fdirty:
            errors.append("rows come from a dirty tree but the file name does not say so")
    if fdirty:
        warnings.append("produced from a tree with uncommitted changes (-dirty)")
    if not commit_exists(fhash):
        errors.append(f"commit {fhash} is not in this repository's history (fetch it first)")
    for i, r in enumerate(rows):
        for c in ("machine", "cpu", "date"):
            if c in cols and not r.get(c):
                errors.append(f"row {i + 2}: empty {c}")
                break
        if r.get("backend") == "gpu":
            if r.get("gpu", "") in ("", "none", "unknown"):
                errors.append(f"row {i + 2}: backend=gpu but no GPU model recorded")
            if not r.get("driver"):
                errors.append(f"row {i + 2}: backend=gpu but no driver version")
            if not r.get("cuda"):
                errors.append(f"row {i + 2}: backend=gpu but no CUDA version")
        # our engines must never report an Optimal that fails verification; rows of a reference
        # solver (compare-highs-*.csv, column solver != ps26119) record its claims as they are
        if r.get("status") == "Optimal" and r.get("verify") == "FAIL" and r.get("solver", "ps26119") == "ps26119":
            errors.append(f"row {i + 2}: status Optimal but verify FAIL ({r.get('instance')})")
    return errors[:20], warnings


def check_logs(folder):
    errors, warnings = [], []
    ct = os.path.join(folder, "ctest.log")
    if not os.path.exists(ct):
        errors.append("no ctest.log")
    else:
        text = open(ct, errors="replace").read()
        if "100% tests passed" not in text:
            failed = re.findall(r"^\s*\d+ - (\S+) \(", text, re.M)
            errors.append("ctest did not pass 100%" + (f": {', '.join(failed[:10])}" if failed else ""))
    sans = sorted(glob.glob(os.path.join(folder, "sanitizer-*.log")))
    if not sans:
        warnings.append("no compute-sanitizer logs")
    for p in sans:
        text = open(p, errors="replace").read()
        name = os.path.basename(p)
        # The tool itself could not attach (e.g. WSL2: "Failed to initialize WDDM debugger interface",
        # "Device not supported"): the kernels were NOT checked — neither clean nor faulty.
        if re.search(r"Failed to initialize .*debugger interface|Device not supported", text):
            errors.append(f"{name}: compute-sanitizer could not run on this device (tool startup error), "
                          "so the kernels were NOT checked")
            continue
        mm = re.search(r"(?:ERROR|RACECHECK) SUMMARY:\s*(\d+)\s+(?:error|hazard)", text)
        if not mm:
            errors.append(f"{name}: no summary line (did the run finish?)")
        elif int(mm.group(1)) != 0:
            errors.append(f"{name}: {mm.group(1)} errors")
    if not os.path.exists(os.path.join(folder, "nvidia-smi.txt")):
        warnings.append("no nvidia-smi.txt")
    return errors, warnings


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    csv_only = "--csv-only" in args  # CI: log folders record historical runs (e.g. a known failure)
    args = [a for a in args if a != "--csv-only"]
    targets = args or [os.path.join(HERE, "results")]
    csvs, logs = [], []
    for t in targets:
        if os.path.isdir(t):
            csvs += sorted(glob.glob(os.path.join(t, "*.csv")))
            if os.path.exists(os.path.join(t, "ctest.log")):
                logs.append(t)
            logs += sorted(d for d in glob.glob(os.path.join(t, "logs", "*")) if os.path.isdir(d))
        elif t.endswith(".csv"):
            csvs.append(t)
    csvs = list(dict.fromkeys(csvs))
    logs = [] if csv_only else list(dict.fromkeys(os.path.normpath(d) for d in logs))
    bad = 0
    for p in csvs:
        e, w = check_csv(p)
        bad += bool(e)
        print(f"{'FAIL' if e else 'ok  '}  {os.path.relpath(p, ROOT)}" + "".join(f"\n        error: {x}" for x in e)
              + "".join(f"\n        warn:  {x}" for x in w))
    for d in logs:
        e, w = check_logs(d)
        bad += bool(e)
        print(f"{'FAIL' if e else 'ok  '}  {os.path.relpath(d, ROOT)}/" + "".join(f"\n        error: {x}" for x in e)
              + "".join(f"\n        warn:  {x}" for x in w))
    print(f"\n{len(csvs)} CSV files, {len(logs)} log folders: {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
