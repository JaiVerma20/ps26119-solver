"""machine_info.py — provenance recorded in every benchmark CSV (CLAUDE.md §3):
git hash, machine, CPU, GPU model, driver, CUDA version, date.

Set PS26119_MACHINE to label the machine (e.g. "macbook-air-m4", "rtx4060-laptop",
"uni-a100"); otherwise OS + architecture is used.
"""
from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_hash(binary: str | None = None) -> str:
    """The commit the BINARY was built from (`ps26119 --version` prints "(git <hash>)").
    Falls back to the working tree only when no binary is given; a binary that does not
    report a hash is labelled "unknown" rather than guessed."""
    if binary:
        out = _run([binary, "--version"])
        if "(git " in out:
            return out.split("(git ", 1)[1].split(")", 1)[0]
        return "unknown"
    h = _run(["git", "rev-parse", "--short", "HEAD"]) or "nogit"
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return h + ("-dirty" if dirty else "")


def cpu_name() -> str:
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_info() -> dict:
    out = {"gpu": "none", "driver": "", "cuda": ""}
    if shutil.which("nvidia-smi"):
        q = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if q:
            first = q.splitlines()[0].split(",")
            out["gpu"] = first[0].strip()
            out["driver"] = first[1].strip() if len(first) > 1 else ""
    if shutil.which("nvcc"):
        v = _run(["nvcc", "--version"])
        for tok in v.replace(",", " ").split():
            if tok.startswith("V") and tok[1:2].isdigit():
                out["cuda"] = tok[1:]
    return out


def machine_info(binary: str | None = None) -> dict:
    g = gpu_info()
    return {
        "git_hash": git_hash(binary),
        "machine": os.environ.get("PS26119_MACHINE", f"{platform.system()}-{platform.machine()}"),
        "cpu": cpu_name(),
        "gpu": g["gpu"],
        "driver": g["driver"],
        "cuda": g["cuda"],
        "date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
