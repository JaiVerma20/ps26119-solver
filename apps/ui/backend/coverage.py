"""coverage.py — the PS 26119 requirement matrix for the UI, parsed from docs/SIH_STATUS.md (the
single source: edit the markdown, the page follows). Status words: DONE, PARTIAL, EXPERIMENTAL,
NOT STARTED (a cell may qualify them, e.g. "DONE (LP)")."""
from __future__ import annotations

import os
import re

from . import paths

SOURCE = os.path.join(paths.ROOT, "docs", "SIH_STATUS.md")
WORDS = ("NOT STARTED", "EXPERIMENTAL", "PARTIAL", "DONE")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def status_word(cell: str) -> str:
    up = cell.upper()
    for w in WORDS:  # most cautious first: "DONE on one GPU (PARTIAL: ...)" counts as PARTIAL
        if w in up:
            return w
    return "UNKNOWN"


def parse(path: str = SOURCE) -> dict:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header, rows, updated = None, [], None
    for line in lines:
        m = re.search(r"Updated (\d{4}-\d{2}-\d{2})", line)
        if m and not updated:
            updated = m[1]
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if header is None:
            header = [c.lower() for c in cells]
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        r = dict(zip(header, cells))
        r["status_word"] = status_word(r.get("status", ""))
        rows.append(r)
    counts = {w: sum(r["status_word"] == w for r in rows) for w in WORDS}
    return {"source": os.path.relpath(path, paths.ROOT), "updated": updated, "columns": header or [], "rows": rows,
            "counts": counts}
