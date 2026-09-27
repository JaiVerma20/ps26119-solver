"""generate.py — build the repository's generated test LPs on demand (bench/generate_*.py), and read
their known-optimum sidecars.

The generators construct a KKT point first, so every generated model has an optimum known exactly
(written to <model>.json by bench/lpgen.write_sidecar). The UI uses it as a check that needs no
other solver: |objective − known| / (1 + |known|). Arguments are whitelisted; the output name is
the one bench/scale.py and scripts/demo.sh use, so the UI and the benchmarks share files.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading

from . import paths

GEN_DIR = os.path.join(paths.ROOT, "bench", "generated")
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()

KINDS = {
    # kind: (script, size argument, size range, output name)
    "refinery": ("generate_refinery_lp.py", "--periods", (1, 8760), "refinery-T{size}-s{seed}.lpm"),
    "random": ("generate_lp.py", "--rows", (100, 1_000_000), "rand-{size}-s{seed}.lpm"),
}


def validate(body: dict):
    kind = body.get("kind")
    if kind not in KINDS:
        return None, f"kind must be one of {sorted(KINDS)}"
    try:
        size, seed = int(body.get("size")), int(body.get("seed", 1))
    except (TypeError, ValueError):
        return None, "size and seed must be integers"
    lo, hi = KINDS[kind][2]
    if not lo <= size <= hi:
        return None, f"{kind}: size must be in [{lo}, {hi}]"
    if not 1 <= seed <= 9999:
        return None, "seed must be in [1, 9999]"
    return (kind, size, seed), None


def generate(kind: str, size: int, seed: int) -> dict:
    """Run the generator unless the model already exists. Returns {path, created, message}."""
    script, arg, _, name = KINDS[kind]
    out = os.path.join(GEN_DIR, name.format(size=size, seed=seed))
    with _guard:
        lock = _locks.setdefault(out, threading.Lock())
    with lock:  # two clicks on the same scenario must not write the same file twice
        if os.path.exists(out) and os.path.exists(os.path.splitext(out)[0] + ".json"):
            return {"path": paths.rel(out), "created": False, "message": "already generated"}
        os.makedirs(GEN_DIR, exist_ok=True)
        cmd = [paths.PYTHON, os.path.join(paths.ROOT, "bench", script), arg, str(size), "--seed", str(seed), "--out", out]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=paths.ROOT)
        if p.returncode != 0 or not os.path.exists(out):
            raise RuntimeError((p.stderr or p.stdout).strip()[-600:] or "generator failed")
        return {"path": paths.rel(out), "created": True, "message": p.stdout.strip().replace(paths.ROOT + os.sep, "")}


def known(model_abs: str) -> dict | None:
    """The known optimum of a generated model (its sidecar), or None. Only sidecars written by the
    repository's generators count (they carry 'generator' and 'optimum')."""
    side = os.path.splitext(model_abs)[0] + ".json"
    if not os.path.isfile(side):
        return None
    try:
        with open(side) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or "optimum" not in d or str(d.get("generator", "")).split(".")[0] not in (
            "generate_lp", "generate_refinery_lp"):
        return None
    # the sidecar's optimum is in the model's own sense (bench/scale.py compares it with the
    # reported objective directly)
    return {"optimum": d["optimum"], "sense": d.get("sense"), "generator": d["generator"],
            "periods": d.get("periods"), "seed": d.get("seed")}
