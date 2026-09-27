"""preflight.py — demo-day readiness: every dependency the Command Center and the jury demo need,
checked for real (the binary is run, Python modules are imported by the interpreter verify.py uses,
the demo models are opened, the evidence CSVs are parsed). Each check says how to fix it.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from . import evidence, generate, paths, system

# the models the jury demo walks through (path, what it shows, generator args if generated)
DEMO_MODELS = [
    ("data/netlib_small/afiro.mps", "classic LP (AFIRO)", None),
    ("data/examples/infeasible.mps", "infeasible LP (Farkas certificate)", None),
    ("data/examples/unbounded.mps", "unbounded LP (ray certificate)", None),
    ("data/mip_small/gt2.lpm", "MILP (MIPLIB gt2)", None),
    ("bench/generated/refinery-T8760-s1.lpm", "refinery year, hourly (429k rows)", ("refinery", 8760, 1)),
]


def _item(cid, label, state, detail, fix=""):
    return {"id": cid, "label": label, "state": state, "detail": detail, "fix": fix}


def _git_head() -> str | None:
    if not shutil.which("git"):
        return None
    p = subprocess.run(["git", "-C", paths.ROOT, "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or None if p.returncode == 0 else None


def run() -> dict:
    items = []
    s = system.probe()
    if not s.get("binary_found"):
        items.append(_item("binary", "Solver binary", "fail", f"{paths.rel(paths.BIN)} not found", "cmake --build build -j"))
    else:
        items.append(_item("binary", "Solver binary", "ok", s.get("version", "")))
        head, built = _git_head(), s.get("git_hash", "")
        dirty = built.endswith("-dirty")
        if head and built and not built.startswith(head[:7]):
            items.append(_item("fresh", "Binary matches the checkout", "warn",
                               f"binary built at {built}, checkout is {head}", "cmake --build build -j (then restart the UI)"))
        elif dirty:
            items.append(_item("fresh", "Binary matches the checkout", "warn",
                               f"binary built at {built}: from uncommitted changes", "commit, then cmake --build build -j"))
        elif head:
            items.append(_item("fresh", "Binary matches the checkout", "ok", f"commit {head}"))
        items.append(_item("cuda", "GPU backend", "ok" if s.get("cuda_build") else "info",
                           f"CUDA build · {s.get('gpu')}" if s.get("cuda_build") else
                           "CPU-only build on this machine: GPU results are shown from committed evidence",
                           "" if s.get("cuda_build") else "on an NVIDIA machine: cmake -DPS26119_ENABLE_CUDA=ON"))
    probe = subprocess.run([paths.PYTHON, "-c", "import highspy, numpy; print(highspy.__file__ and 'ok')"],
                           capture_output=True, text=True)
    items.append(_item("verifier", "Independent verifier (tools/verify.py)", "ok" if probe.returncode == 0 else "fail",
                       f"{paths.PYTHON}: highspy + numpy importable" if probe.returncode == 0 else
                       (probe.stderr.strip().splitlines() or ["import failed"])[-1],
                       "" if probe.returncode == 0 else "pip install highspy numpy (or set PS26119_PYTHON)"))
    for rel, what, gen in DEMO_MODELS:
        p = os.path.join(paths.ROOT, rel)
        ok = os.path.exists(p)
        items.append(_item("model:" + rel, f"Demo model: {what}", "ok" if ok else ("fixable" if gen else "fail"),
                           rel if ok else f"{rel} missing", "" if ok else ("click Prepare" if gen else "restore data/ from git")))
    try:
        ev = evidence.collect()
        n = " · ".join(f'{e["engine"]} {e["solved"]}/{e["total"]}' for e in ev["netlib"])
        items.append(_item("evidence", "Benchmark evidence (bench/results)", "ok" if ev["netlib"] else "warn",
                           f"Netlib {n} · GPU machines {len(ev['gpu'])} · MIPLIB {'yes' if ev['miplib'] else 'no'}"))
    except Exception as e:  # noqa: BLE001
        items.append(_item("evidence", "Benchmark evidence (bench/results)", "fail", f"{type(e).__name__}: {e}"))
    try:
        os.makedirs(paths.RUNS, exist_ok=True)
        t = os.path.join(paths.RUNS, ".write-test")
        with open(t, "w") as f:
            f.write("ok")
        os.remove(t)
        free = shutil.disk_usage(paths.RUNS).free
        items.append(_item("disk", "Run folder writable", "ok" if free > 2 << 30 else "warn",
                           f"{paths.rel(paths.RUNS)} · {free / 2**30:.1f} GB free"))
    except OSError as e:
        items.append(_item("disk", "Run folder writable", "fail", str(e)))
    worst = "fail" if any(i["state"] == "fail" for i in items) else \
        "warn" if any(i["state"] in ("warn", "fixable") for i in items) else "ok"
    return {"state": worst, "items": items, "cores": s.get("cores"), "cpu": s.get("cpu")}


def prepare() -> list[dict]:
    """Generate the missing generated demo models (the repository's own generators)."""
    done = []
    for rel, _, gen in DEMO_MODELS:
        if gen and not os.path.exists(os.path.join(paths.ROOT, rel)):
            done.append(generate.generate(*gen))
    return done
