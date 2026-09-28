"""system.py — what this machine and this build are: solver version + git hash, CPU, cores and
whether the binary has a CUDA backend (probed by asking the solver, not assumed)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys

from . import paths

sys.path.insert(0, os.path.join(paths.ROOT, "bench"))
from machine_info import cpu_name, gpu_info  # noqa: E402

_cached: dict | None = None


def probe() -> dict:
    global _cached
    if _cached is not None:
        return _cached
    res: dict = {"binary": paths.rel(paths.BIN), "binary_found": os.path.exists(paths.BIN)}
    if res["binary_found"]:
        v = subprocess.run([paths.BIN, "--version"], capture_output=True, text=True).stdout.strip()
        res["version"] = v
        res["git_hash"] = v.split("(git ", 1)[1].rstrip(")") if "(git " in v else "unknown"
        tiny = os.path.join(paths.ROOT, "data", "examples", "tiny_max.mps")
        p = subprocess.run([paths.BIN, "solve", tiny, "--algorithm", "r2hpdhg", "--gpu"], capture_output=True, text=True)
        res["cuda_build"] = "no CUDA backend" not in p.stdout
        res["cuda_probe"] = next((l.split(None, 1)[1] for l in p.stdout.splitlines() if l.startswith("message")), "")
    from . import pdf
    res["pdf_export"] = pdf.browser() is not None  # a local Chrome / Chromium can print the report to PDF
    res["cpu"] = cpu_name()
    res["cores"] = os.cpu_count()
    res["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    res["python"] = platform.python_version()
    try:
        g = gpu_info()
        res["gpu"] = g.get("gpu") or "none"
        res["gpu_driver"], res["cuda"] = g.get("driver", ""), g.get("cuda", "")
    except Exception:  # noqa: BLE001
        res["gpu"] = "none"
    _cached = res
    return res
