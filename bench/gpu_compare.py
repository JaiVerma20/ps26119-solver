"""gpu_compare.py — the ONE place that pairs GPU runs with CPU runs (used by bench/scale.py's
summary and bench/make_evidence.py), so a "speed-up" always means the same thing.

Rules (docs/BENCHMARKS.md, "GPU vs CPU"):
  * pairs only rows from the same CSV (same machine, same binary/commit), same instance, engine
    and precision;
  * two CPU baselines, both labelled: 1 thread, and the FASTEST CPU configuration in that CSV
    (by wall time to 1e-8, with its thread count) — a GPU is never compared only with one core;
  * a ratio is computed only when both runs are Optimal and not verify=FAIL; otherwise the
    statuses are shown instead of a number;
  * ratio = CPU seconds / GPU seconds, so < 1 means the GPU is slower (reported, not hidden).
"""


def _ok(r):
    return r.get("status") == "Optimal" and r.get("verify") != "FAIL"


def _f(r, k):
    try:
        v = float(r.get(k, ""))
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def ratio(cpu, gpu, key):
    """CPU/GPU for `key`, or None when not comparable."""
    if cpu is None or gpu is None or not (_ok(cpu) and _ok(gpu)):
        return None
    c, g = _f(cpu, key), _f(gpu, key)
    if c is None or g is None or g <= 0:
        return None
    return c / g


def pairs(rows):
    """One dict per GPU row: gpu, cpu1 (1-thread row or None), best (fastest verified CPU row or
    None) and the four ratios."""
    out = []
    for g in (r for r in rows if r.get("backend") == "gpu"):
        cpus = [r for r in rows if r.get("backend") == "cpu" and r.get("instance") == g.get("instance")
                and r.get("engine") == g.get("engine") and r.get("precision") == g.get("precision")]
        cpu1 = next((r for r in cpus if str(r.get("threads", "1")) in ("1", "")), None)
        good = [r for r in cpus if _ok(r) and _f(r, "seconds_to_1e-8") is not None]
        best = min(good, key=lambda r: _f(r, "seconds_to_1e-8")) if good else None
        out.append({
            "gpu": g, "cpu1": cpu1, "best": best,
            "vs1_1e-4": ratio(cpu1, g, "seconds_to_1e-4"), "vs1_1e-8": ratio(cpu1, g, "seconds_to_1e-8"),
            "vsbest_1e-4": ratio(best, g, "seconds_to_1e-4"), "vsbest_1e-8": ratio(best, g, "seconds_to_1e-8"),
        })
    return out


def fmt_ratio(v):
    return "–" if v is None else f"{v:.2f}×"


def status_note(p):
    """Why a ratio is missing (statuses), or ''."""
    notes = []
    for label, r in (("GPU", p["gpu"]), ("CPU 1 thr", p["cpu1"]), ("CPU best", p["best"])):
        if r is None:
            notes.append(f"{label}: no run")
        elif not _ok(r):
            notes.append(f"{label}: {r.get('status')}{' verify FAIL' if r.get('verify') == 'FAIL' else ''}")
    return "; ".join(notes)
