"""models.py — model catalog, statistics (from `ps26119 info`, the solver's own reader) and a
downsampled sparsity pattern for the Model Explorer.

The sparsity image is a visualization only (nonzeros counted per block of rows x columns); every
number shown next to it comes from the solver's reader.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

from . import generate, paths

sys.path.insert(0, os.path.join(paths.ROOT, "tools"))
from lpm import read_lpm  # noqa: E402

_cache: dict = {}
_lock = threading.Lock()


def catalog() -> list[dict]:
    groups = []
    for label, d, desc in paths.MODEL_DIRS:
        full = os.path.join(paths.ROOT, d)
        if not os.path.isdir(full):
            continue
        files = sorted(f for f in os.listdir(full) if f.endswith(paths.MODEL_EXT))
        # prefer .mps when both formats of the same model exist (the reader everyone uses)
        stems = {}
        for f in files:
            stem, ext = os.path.splitext(f)
            if stem not in stems or ext == ".mps":
                stems[stem] = f
        items = [{"name": s, "path": os.path.join(d, f), "format": f.rsplit(".", 1)[1],
                  "bytes": os.path.getsize(os.path.join(full, f))} for s, f in sorted(stems.items())]
        if items:
            groups.append({"group": label, "folder": d, "description": desc, "models": items})
    return groups


_INFO_KEYS = {
    "name": "name", "sense": "sense", "rows": "rows", "columns": "columns", "column bounds": "column_bounds",
    "nonzeros": "nonzeros", "|matrix coeff|": "matrix_range", "|objective coeff|": "objective_range",
    "|row bounds|": "row_bounds_range", "|column bounds|": "column_bounds_range",
    "matrix dynamism": "dynamism", "fingerprint": "fingerprint", "note": "note",
}


def info(model_abs: str) -> dict:
    """`ps26119 info` parsed into fields (cached by file and mtime)."""
    key = ("info", model_abs, os.path.getmtime(model_abs))
    with _lock:
        if key in _cache:
            return _cache[key]
    p = subprocess.run([paths.BIN, "info", model_abs], capture_output=True, text=True, timeout=600)
    res: dict = {"ok": p.returncode == 0, "raw": p.stdout, "warnings": [l for l in p.stderr.splitlines() if l]}
    if p.returncode != 0:
        res["error"] = (p.stderr or p.stdout).strip()
        return res
    for line in p.stdout.splitlines():
        m = re.match(r"^\s{2}(.+?)\s{2,}(.*)$", line)
        if m and m.group(1).strip() in _INFO_KEYS:
            res[_INFO_KEYS[m.group(1).strip()]] = m.group(2).strip()
    num = lambda s: int(s.split()[0]) if s and s.split()[0].isdigit() else None
    res["n_rows"], res["n_cols"], res["n_nnz"] = num(res.get("rows")), num(res.get("columns")), num(res.get("nonzeros"))
    for field, pat in (("rows", r"E (\d+), L (\d+), G (\d+), ranged (\d+), free (\d+)"),
                       ("columns", r"continuous (\d+), integer (\d+), binary (\d+)")):
        m = re.search(pat, res.get(field, ""))
        if m:
            res[field + "_split"] = [int(g) for g in m.groups()]
    res["known"] = generate.known(model_abs)
    with _lock:
        _cache[key] = res
    return res


def _mps_entries(path: str):
    """(row index, col index) pairs from an MPS COLUMNS section — for the picture only."""
    rows, section, obj = {}, None, None
    cols: dict[str, int] = {}
    with open(path, errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                section = line.split()[0].upper()
                continue
            t = line.split()
            if section == "ROWS" and len(t) >= 2:
                if t[0].upper() == "N":
                    obj = obj or t[1]
                else:
                    rows[t[1]] = len(rows)
            elif section == "COLUMNS" and len(t) >= 3:
                if "'MARKER'" in line:
                    continue
                c = cols.setdefault(t[0], len(cols))
                for k in range(1, len(t) - 1, 2):
                    r = rows.get(t[k])
                    if r is not None:
                        yield r, c
    return


def sparsity(model_abs: str, size: int = 160) -> dict:
    """Nonzero counts on a size x size grid (block counts), cached."""
    key = ("spy", model_abs, os.path.getmtime(model_abs), size)
    with _lock:
        if key in _cache:
            return _cache[key]
    st = info(model_abs)
    m, n = st.get("n_rows") or 0, st.get("n_cols") or 0
    if not m or not n:
        return {"ok": False, "error": "model has no rows or columns"}
    gr, gc = min(size, m), min(size, n)
    grid = [0] * (gr * gc)
    if model_abs.endswith(".lpm"):
        M = read_lpm(model_abs)
        for j in range(M.num_cols):
            cj = j * gc // n
            for k in range(M.col_start[j], M.col_start[j + 1]):
                grid[(M.row_index[k] * gr // m) * gc + cj] += 1
    else:
        for r, c in _mps_entries(model_abs):
            if r < m and c < n:
                grid[(r * gr // m) * gc + c * gc // n] += 1
    res = {"ok": True, "rows": gr, "cols": gc, "model_rows": m, "model_cols": n, "grid": grid, "max": max(grid) or 1}
    res["known"] = generate.known(model_abs)
    with _lock:
        _cache[key] = res
    return res
