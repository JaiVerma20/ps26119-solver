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


def _log_facts(machine: str, git_hash: str) -> dict:
    """Facts from the GPU run's committed log folder (bench/results/logs/<machine>-<hash>/)."""
    folder = os.path.join(paths.ROOT, "bench", "results", "logs", f"{machine}-{git_hash}")
    facts: dict = {"folder": os.path.relpath(folder, paths.ROOT) if os.path.isdir(folder) else None}
    if not os.path.isdir(folder):
        return facts
    facts["files"] = sorted(os.listdir(folder))

    def read(name):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            return ""
        with open(p, errors="replace") as f:
            return f.read()
    import re
    # ctest prints "100% tests passed out of 198" or "97% tests passed, 3 tests failed out of 98"
    m = re.search(r"\d+% tests passed(?:, (\d+) tests failed)? out of (\d+)", read("ctest.log"))
    if m:
        facts["ctest"] = {"passed": int(m[2]) - int(m[1] or 0), "total": int(m[2])}
    m = re.search(r"^(NVIDIA[^,\n]+), ([\d.]+), (\d+) MiB, (\d+) MHz", read("gpu_check.log"), re.M)
    if m:
        facts["gpu_memory_mib"], facts["sm_clock_mhz"] = int(m[3]), int(m[4])
    m = re.search(r"^Linux .*?(microsoft|WSL2)", read("gpu_check.log"), re.M | re.I)
    facts["os"] = "Linux (WSL2)" if m else None
    return facts


def gpu_detail() -> dict:
    """Every configuration of the newest GPU scale run per GPU machine (CPU 1 thread, CPU all
    cores, GPU; fp64 and mixed), its pairs (speed-ups as bench/gpu_compare.py defines them), the
    sanitizer status, facts from the committed logs, and the small-Netlib GPU run."""
    paths_ = me.committed_csvs()
    scales = [p for p in paths_ if os.path.basename(p).startswith("scale-") and not os.path.basename(p).startswith("scale-simplex-")]
    gpu_scales = [p for p in scales if any(r.get("backend") == "gpu" for r in me.load(p))]
    by_machine: dict = {}
    for p in gpu_scales:
        by_machine.setdefault(me.load(p)[0].get("machine"), []).append(p)
    machines = []
    base = {g["source"]["file"]: g for g in collect()["gpu"]}
    for machine, ps in sorted(by_machine.items()):
        p = me.latest(ps, "scale-")
        rows = me.load(p)
        r0 = rows[0]
        instances: dict = {}
        for r in rows:
            inst = instances.setdefault(r["instance"], {"instance": r["instance"], "family": r.get("family"),
                                                         "rows": int(r["rows"]), "cols": int(r["cols"]), "nnz": int(r["nnz"]),
                                                         "known_optimum": _f(r.get("known_optimum")), "runs": []})
            inst["runs"].append({
                "backend": r["backend"], "threads": int(r["threads"]) if r.get("threads") else None,
                "precision": r["precision"], "status": r["status"], "iterations": int(r["iterations"]),
                "seconds": _f(r.get("seconds")), "s_1e4": _f(r.get("seconds_to_1e-4")), "s_1e8": _f(r.get("seconds_to_1e-8")),
                "setup_seconds": _f(r.get("setup_seconds")), "ms_per_iteration": _f(r.get("ms_per_iteration")),
                "objective": _f(r.get("objective")), "rel_err_known": _f(r.get("rel_err_known")),
                "verify": r.get("verify"), "verify_primal_rel": _f(r.get("verify_primal_rel")),
                "verify_dual_rel": _f(r.get("verify_dual_rel")), "verify_gap_rel": _f(r.get("verify_gap_rel")),
                "message": r.get("message", "")})
        net = [q for q in paths_ if os.path.basename(q).startswith("netlib-small-gpu-") and me.load(q)[0].get("machine") == machine]
        netlib = None
        if net:
            q = me.latest(net, "netlib-small-gpu-")
            nrows = me.load(q)
            groups: dict = {}
            for r in nrows:
                g = groups.setdefault((r["engine"], r["precision"]), {"engine": r["engine"], "precision": r["precision"],
                                                                      "verified": 0, "total": 0, "seconds": 0.0})
                g["total"] += 1
                g["verified"] += r["status"] == "Optimal" and r["verify"] == "PASS"
                g["seconds"] += _f(r.get("seconds")) or 0.0
            netlib = {"source": _src(q), "groups": list(groups.values()),
                      "instances": sorted({r["instance"] for r in nrows})}
        machines.append({
            "machine": machine, "source": _src(p), "gpu": r0.get("gpu"), "driver": r0.get("driver"), "cuda": r0.get("cuda"),
            "cpu": r0.get("cpu"), "cpu_cores": r0.get("cpu_cores"), "tolerance": r0.get("tolerance"),
            "sanitizer": me.sanitizer_status(p), "logs": _log_facts(machine, r0.get("git_hash")),
            "pairs": base.get(os.path.basename(p), {}).get("pairs", []),
            "instances": sorted(instances.values(), key=lambda i: (i["family"] or "", i["nnz"])), "netlib": netlib})
    return {"machines": machines}


def compare() -> dict:
    """The newest committed ps26119-vs-HiGHS run (bench/compare_highs.py), per machine: every row, as
    written. Aggregates (solved counts, performance profiles, shifted geometric means) are computed
    by the page from these rows, with the rule stated there."""
    paths_ = [p for p in me.committed_csvs() if os.path.basename(p).startswith("compare-highs-")]
    by_machine: dict = {}
    for p in paths_:
        by_machine.setdefault(me.load(p)[0].get("machine"), []).append(p)
    out = []
    for machine, ps in sorted(by_machine.items()):
        p = me.latest(ps, "compare-highs-")
        rows = me.load(p)
        num = ("rows", "cols", "nnz", "iterations", "seconds", "objective", "rel_err_ref", "time_limit")
        out.append({"machine": machine, "source": _src(p), "solver_versions": sorted({r["solver_version"] for r in rows}),
                    "rows": [{**r, **{k: _f(r.get(k)) for k in num}} for r in rows]})
    return {"runs": out}
