"""evidence.py — benchmark evidence for the UI, read from the COMMITTED CSVs in bench/results/
with the same helpers and rules as bench/make_evidence.py (CLAUDE.md §5.1: no number unless it
comes from a hash-named CSV). Every value carries its source file and git hash.
"""
from __future__ import annotations

import os
import sys

from . import paths

sys.path.insert(0, os.path.join(paths.ROOT, "bench"))
import gpu_compare  # noqa: E402
import make_evidence as me  # noqa: E402


def _src(p: str) -> dict:
    r0 = me.load(p)[0]
    return {"file": os.path.basename(p), "git_hash": r0.get("git_hash"), "machine": r0.get("machine"),
            "cpu": r0.get("cpu"), "gpu": r0.get("gpu"), "date": r0.get("date")}


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def collect() -> dict:
    paths_ = me.committed_csvs()
    out: dict = {"netlib": [], "infeasible": [], "miplib": None, "scale_cpu": None, "gpu": [], "small_netlib": None}

    net = me.latest([p for p in paths_ if "gpu" not in os.path.basename(p)], "netlib-small-")
    if net:
        rows = me.load(net)
        out["small_netlib"] = {"source": _src(net), "verified": sum(r["verify"] == "PASS" for r in rows),
                               "total": len(rows)}

    fulls: dict = {}
    for p in paths_:
        b = os.path.basename(p)
        if b.startswith("netlib-full-") and not any(f"-{t}-" in b for t in me.ABLATION_TAGS):
            rows = me.load(p)
            if rows and rows[0].get("backend") != "gpu":
                fulls.setdefault(rows[0]["engine"], []).append(p)
    for eng in ("auto", "simplex", "r2hpdhg"):
        if eng not in fulls:
            continue
        p = me.latest(fulls[eng], "netlib-full-")
        rows = me.load(p)
        ok = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS" and r["rel_err_highs"]
              and float(r["rel_err_highs"]) <= 1e-6]
        out["netlib"].append({
            "engine": eng, "solved": len(ok), "total": len(rows), "source": _src(p),
            "seconds_solved": round(sum(_f(r["seconds"]) or 0 for r in ok), 1),
            "not_solved": sorted(r["instance"] for r in rows if r not in ok),
            "instances": [{"name": r["instance"], "rows": r["rows"], "cols": r["cols"], "nnz": r["nnz"],
                           "status": r["status"], "seconds": _f(r["seconds"]), "iterations": r["iterations"],
                           "verify": r["verify"], "rel_err_highs": _f(r["rel_err_highs"])} for r in rows],
        })

    for eng in ("simplex", "r2hpdhg"):
        c = me.latest([p for p in paths_ if os.path.basename(p).startswith(f"infeasible-cut-{eng}-")], "infeasible-cut-")
        if not c:
            continue
        rows = me.load(c)
        cert = [r for r in rows if r["status"] == "Infeasible" and r["check"] == "PASS" and r["verify"] == "PASS"]
        out["infeasible"].append({
            "engine": eng, "certified": len(cert), "total": len(rows), "source": _src(c),
            "exact_rational": sum("exact rational" in r["certificate"] for r in cert),
            "rounding_proof": sum(r.get("gate_certificate") == "rounding-proof" for r in cert),
        })

    mp = me.latest(paths_, "miplib3-")
    if mp:
        rows = me.load(mp)
        out["miplib"] = {"source": _src(mp), "solved": sum(r["status"] == "Optimal" and r["verify"] == "PASS" for r in rows),
                         "total": len(rows),
                         "instances": [{"name": r["instance"], "status": r["status"], "objective": _f(r["objective"]),
                                        "highs": _f(r["highs_objective"]), "seconds": _f(r["seconds"]),
                                        "gap": _f(r["gap"]), "verify": r["verify"]} for r in rows]}

    scales = [p for p in paths_ if os.path.basename(p).startswith("scale-") and not os.path.basename(p).startswith("scale-simplex-")]
    cpu = [p for p in scales if all(r.get("backend") != "gpu" for r in me.load(p))]
    if cpu:
        p = me.latest(cpu, "scale-")
        out["scale_cpu"] = {"source": _src(p), "rows": [
            {"instance": r["instance"], "rows": int(r["rows"]), "nnz": int(r["nnz"]), "threads": r.get("threads"),
             "precision": r["precision"], "status": r["status"], "iterations": r["iterations"],
             "s_1e4": _f(r.get("seconds_to_1e-4")), "s_1e8": _f(r.get("seconds_to_1e-8")),
             "rel_err_known": _f(r.get("rel_err_known")), "verify": r.get("verify")} for r in me.load(p)]}

    gpu_all = [p for p in scales if p not in cpu]
    by_machine: dict = {}
    for p in gpu_all:
        by_machine.setdefault(me.load(p)[0].get("machine"), []).append(p)
    for machine, ps in sorted(by_machine.items()):
        p = me.latest(ps, "scale-")
        rows = me.load(p)
        pairs = []
        for q in gpu_compare.pairs(rows):
            g, b, c1 = q["gpu"], q["best"], q["cpu1"]
            pairs.append({"instance": g["instance"], "rows": int(g["rows"]), "nnz": int(g["nnz"]), "precision": g["precision"],
                          "gpu_status": g["status"], "gpu_s": _f(g.get("seconds_to_1e-8")),
                          "gpu_iterations": g["iterations"], "cpu1_s": _f(c1.get("seconds_to_1e-8")) if c1 else None,
                          "best_cpu_s": _f(b.get("seconds_to_1e-8")) if b else None,
                          "best_cpu_threads": b.get("threads") if b else None,
                          "ratio_vs_best": q["vsbest_1e-8"], "ratio_vs_1thread": q["vs1_1e-8"],
                          "note": gpu_compare.status_note(q)})
        r0 = rows[0]
        out["gpu"].append({"source": _src(p), "gpu": r0.get("gpu"), "driver": r0.get("driver"), "cuda": r0.get("cuda"),
                           "cpu": r0.get("cpu"), "cpu_cores": r0.get("cpu_cores"),
                           "sanitizer": me.sanitizer_status(p), "pairs": pairs})
    return out
