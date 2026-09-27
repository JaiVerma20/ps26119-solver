"""paths.py — where things live, and which model files the UI may open.

The UI only ever opens model files inside the known data folders (plus its own upload folder),
so the HTTP API cannot be used to read arbitrary files on the machine.
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BIN = os.environ.get("PS26119_BIN", os.path.join(ROOT, "build", "ps26119"))
PYTHON = os.environ.get("PS26119_PYTHON", "python3")
RUNS = os.path.join(ROOT, "apps", "ui", ".runs")          # git-ignored: job outputs, uploads
UPLOADS = os.path.join(RUNS, "uploads")
WEB = os.path.join(ROOT, "apps", "ui", "web")

# (group label, folder relative to ROOT, description)
MODEL_DIRS = [
    ("Examples", "data/examples", "small hand-made models: blending, infeasible, unbounded, MILP"),
    ("Netlib (small)", "data/netlib_small", "10 classic Netlib LPs (committed)"),
    ("Hand LPs", "data/hand", "hand-written LP / MILP test models"),
    ("MILP", "data/mip_small", "committed MIPLIB test copy"),
    ("Generated (large)", "bench/generated", "generated LPs with known optimum: refinery planning, random sparse"),
    ("Netlib (all 93)", "data/netlib", "full Netlib LP set (fetched by tools/fetch_netlib.py)"),
    ("MIPLIB 3", "data/miplib3", "small MIPLIB 3 (fetched by tools/fetch_miplib3.py)"),
    ("Uploaded", "apps/ui/.runs/uploads", "models uploaded through the UI"),
]
MODEL_EXT = (".mps", ".lpm")


def allowed_roots() -> list[str]:
    return [os.path.realpath(os.path.join(ROOT, d)) for _, d, _ in MODEL_DIRS]


def resolve_model(rel: str) -> str | None:
    """Absolute path of a model given relative to ROOT, or None if it is not an allowed model file."""
    if not rel or "\x00" in rel:
        return None
    p = os.path.realpath(os.path.join(ROOT, rel))
    if not p.endswith(MODEL_EXT) or not os.path.isfile(p):
        return None
    if not any(p.startswith(r + os.sep) for r in allowed_roots()):
        return None
    return p


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)
