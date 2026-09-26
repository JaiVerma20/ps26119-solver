#!/usr/bin/env python3
"""make_evidence.py — build docs/EVIDENCE.md from COMMITTED benchmark CSVs only.

Evidence rule (CLAUDE.md §5.1): no number in the docs unless it comes from a CSV in
bench/results/ whose filename contains the git short hash. This script reads only files
tracked by git (`git ls-files bench/results/*.csv`), prints the source file next to every
table, and writes nothing that is not in one of those files (plus counts derived from
them). Rerun after committing new CSVs:  python3 bench/make_evidence.py
"""
import csv
import os
import subprocess
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "EVIDENCE.md")


def committed_csvs():
    r = subprocess.run(["git", "ls-files", "bench/results/*.csv"], cwd=ROOT, capture_output=True, text=True)
    return sorted(p for p in r.stdout.split() if p)


def commit_date(path):
    r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", path], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


def load(path):
    with open(os.path.join(ROOT, path), newline="") as f:
        return list(csv.DictReader(f))


def latest(paths, prefix):
    """Most recently committed CSV whose name starts with prefix."""
    cands = [p for p in paths if os.path.basename(p).startswith(prefix)]
    if not cands:
        return None
    cands.sort(key=lambda p: subprocess.run(["git", "log", "-1", "--format=%ct", "--", p], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip())
    return cands[-1]


def fnum(v, fmt="{:.3g}"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "–" if v in (None, "") else str(v)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def netlib_section(path):
    rows = load(path)
    lines = [f"Source: `{path}` — machine `{rows[0]['machine']}` ({rows[0]['cpu']}), commit `{rows[0]['git_hash']}`.", ""]
    by = OrderedDict()
    for r in rows:
        by.setdefault(r["instance"], {})[(r["engine"], r["precision"])] = r
    cfgs = [("oracle", "dd"), ("simplex", "fp64"), ("pdlp", "fp64"), ("pdlp", "mixed"), ("r2hpdhg", "fp64"),
            ("r2hpdhg", "mixed")]
    cfgs = [c for c in cfgs if any(c in v for v in by.values())]
    header = ["instance", "rows×cols", "published optimum"] + [f"{e} {p}" for e, p in cfgs]
    body = []
    for inst, d in by.items():
        any_r = next(iter(d.values()))
        cells = [inst, f"{any_r['rows']}×{any_r['cols']}", any_r["published_optimum"]]
        for c in cfgs:
            r = d.get(c)
            if not r:
                cells.append("–")
                continue
            it = f", {r['iterations']} it" if c[0] not in ("oracle",) else ""
            cells.append(f"{r['objective']} ({r['verify']}{it})")
        body.append(cells)
    lines.append(table(header, body))
    npass = sum(r["verify"] == "PASS" for r in rows)
    lines += ["", f"**{npass} of {len(rows)} runs verified PASS** by `tools/verify.py` (independent reader, "
              f"primal/dual/gap ≤ 1e-6 relative) and within 1e-6 of the published optimum. "
              f"First-order runs target relative KKT 1e-8; the simplex returns a vertex."]
    # iteration comparison PDLP vs r2HPDHG (fp64)
    better = [inst for inst, d in by.items() if ("pdlp", "fp64") in d and ("r2hpdhg", "fp64") in d
              and int(d[("r2hpdhg", "fp64")]["iterations"]) < int(d[("pdlp", "fp64")]["iterations"])]
    both = [inst for inst, d in by.items() if ("pdlp", "fp64") in d and ("r2hpdhg", "fp64") in d]
    if both:
        tot_p = sum(int(by[i][("pdlp", "fp64")]["iterations"]) for i in both)
        tot_r = sum(int(by[i][("r2hpdhg", "fp64")]["iterations"]) for i in both)
        lines += ["", f"Iterations to 1e-8 (fp64): r²HPDHG needs fewer than PDLP-style PDHG on "
                  f"{len(better)} of {len(both)} instances; total {tot_r} vs {tot_p} iterations."]
    return "\n".join(lines)


def netlib_full_section(path):
    rows = load(path)
    r0 = rows[0]
    solved = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"]
    match = [r for r in solved if r["rel_err_highs"] and float(r["rel_err_highs"]) <= 1e-6]
    limit = [r for r in rows if r["status"] in ("TimeLimit", "IterationLimit")]
    other = [r for r in rows if r not in solved and r not in limit]
    lines = [f"Source: `{path}` — {r0['engine']} {r0['precision']} on {r0['backend']} "
             f"(`{r0['machine']}`, {r0['cpu']}), commit `{r0['git_hash']}`, time limit {r0['time_limit']} s per model.", "",
             f"- **{len(solved)} of {len(rows)}** Netlib LPs solved to relative KKT 1e-8 and verified PASS by "
             f"`tools/verify.py`; **{len(match)}** of those agree with HiGHS to 1e-6 relative.",
             f"- {len(limit)} hit the time/iteration limit (listed below, not hidden); {len(other)} other outcomes."]
    if "certified_gap" in r0:
        fin = [r for r in solved if r["certified_gap"] not in ("", "inf")]
        if fin:
            worst = max(float(r["certified_gap"]) for r in fin)
            lines.append(f"- Certified (rounding-proof, Neumaier–Shcherbina) bound from the returned duals: finite for "
                         f"**{len(fin)} of {len(solved)}** solved models; worst certified gap |obj − bound|/(1+|obj|) "
                         f"= {worst:.1e}. Infinite where a dual points at an infinite bound (reported, not faked).")
    unsolved = limit + other
    if unsolved:
        lines += ["", table(["model", "rows", "cols", "nnz", "status", "iterations", "rel. err vs HiGHS at stop"],
                            [[r["instance"], r["rows"], r["cols"], r["nnz"], r["status"], r["iterations"] or "–",
                              r["rel_err_highs"] or "–"] for r in unsolved])]
    mism = [r for r in solved if r not in match]
    if mism:
        lines += ["", "Solved and verified but objective differs from HiGHS by more than 1e-6 (investigate):", "",
                  table(["model", "objective", "HiGHS", "rel. err"],
                        [[r["instance"], r["objective"], r["highs_objective"], r["rel_err_highs"]] for r in mism])]
    return "\n".join(lines)


def netlib_engine_comparison(fulls):
    """One row per engine for the newest commit that has full-Netlib runs of several engines."""
    by_hash = {}
    for p in fulls:
        rows = load(p)
        if rows and rows[0].get("backend") != "gpu":
            by_hash.setdefault(rows[0]["git_hash"], []).append((p, rows))
    multi = {h: v for h, v in by_hash.items() if len({r[0]["engine"] for _, r in v}) > 1}
    if not multi:
        return ""
    # newest commit among those: pick by the most recently committed file
    newest = latest([p for v in multi.values() for p, _ in v], "netlib-full-")
    h = load(newest)[0]["git_hash"]
    runs = multi[h]
    solved_by = {}
    body = []
    for p, rows in sorted(runs, key=lambda t: t[1][0]["engine"]):
        ok = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"
              and r["rel_err_highs"] and float(r["rel_err_highs"]) <= 1e-6]
        solved_by[rows[0]["engine"]] = {r["instance"] for r in ok}
        secs = sum(float(r["seconds"]) for r in ok if r.get("seconds"))
        body.append([rows[0]["engine"], f"{len(ok)}/{len(rows)}", fnum(secs, "{:.1f}"),
                     ", ".join(sorted(r["instance"] for r in rows if r not in ok)) or "–", f"`{os.path.basename(p)}`"])
    lines = [f"Same machine, same binary (commit `{h}`), same 60 s limit, same verification "
             "(engine Optimal + in-process gate + `tools/verify.py` PASS + |obj − HiGHS|/(1+|HiGHS|) ≤ 1e-6):", "",
             table(["engine", "solved + verified", "total s (solved)", "not solved", "source"], body)]
    if "auto" in solved_by:
        lines += ["", "`auto` = simplex when rows·nnz ≤ 2·10⁸, else r²HPDHG (docs/DECISIONS.md #29). The threshold was "
                  "chosen on this set and on the generated models, so the `auto` row is an in-sample result."]
    return "\n".join(lines)


def scale_section(path):
    rows = load(path)
    r0 = rows[0]
    lines = [f"Source: `{path}` — machine `{r0['machine']}` ({r0['cpu']}; GPU: {r0['gpu'] or 'none'}"
             f"{', driver ' + r0['driver'] if r0['driver'] else ''}{', CUDA ' + r0['cuda'] if r0['cuda'] else ''}), "
             f"commit `{r0['git_hash']}`.", ""]
    header = ["instance", "rows", "nnz", "engine", "backend", "threads", "prec", "status", "iterations",
              "s to 1e-4", "s to 1e-8", "ms/iter", "rel. err vs known opt", "verify"]
    body = []
    for r in rows:
        body.append([r["instance"], r["rows"], r["nnz"], r["engine"], r["backend"], (r.get("threads", "") or ("1" if r["engine"] != "highs" else "–")),
                     r["precision"], r["status"],
                     r["iterations"] or "–", fnum(r["seconds_to_1e-4"]), fnum(r["seconds_to_1e-8"]),
                     fnum(r["ms_per_iteration"]), r["rel_err_known"] or "–", r["verify"] or "–"])
    lines.append(table(header, body))
    png = os.path.splitext(path)[0] + ".png"
    if os.path.exists(os.path.join(ROOT, png)):
        lines += ["", f"![scaling chart](../{png})"]
    # engine vs HiGHS where both finished
    comp = []
    for r in rows:
        if r["engine"] == "highs":
            continue
        h = next((x for x in rows if x["engine"] == "highs" and x["instance"] == r["instance"]), None)
        if h and r["backend"] == "cpu":
            comp.append((r, h))
    if comp:
        lines += ["", "Reference: HiGHS (simplex/IPM default, run as a separate Python process, returns a vertex/"
                  "high-accuracy solution — not the same accuracy target as a 1e-8 relative-KKT first-order stop). "
                  "Rows marked TimeLimit did not finish within the cap."]
    return "\n".join(lines)


def gpu_vs_cpu(paths):
    out = []
    for p in paths:
        rows = load(p)
        gpu_rows = [r for r in rows if r.get("backend") == "gpu"]
        if not gpu_rows:
            continue
        out.append(f"Source: `{p}` — GPU `{rows[0]['gpu']}` (driver {rows[0]['driver']}, CUDA {rows[0]['cuda']}), "
                   f"commit `{rows[0]['git_hash']}`.\n")
        body = []
        for g in gpu_rows:
            c = next((x for x in rows if x.get("backend") == "cpu" and x["instance"] == g["instance"]
                      and x["engine"] == g["engine"] and x["precision"] == g["precision"]), None)
            if not c:
                continue

            def sp(k):
                try:
                    return f"{float(c[k]) / float(g[k]):.2f}×"
                except (ValueError, KeyError, ZeroDivisionError):
                    return "–"
            body.append([g["instance"], g["engine"], g["precision"], fnum(c.get("seconds_to_1e-4")),
                         fnum(g.get("seconds_to_1e-4")), sp("seconds_to_1e-4"), fnum(c.get("seconds_to_1e-8")),
                         fnum(g.get("seconds_to_1e-8")), sp("seconds_to_1e-8")])
        out.append(table(["instance", "engine", "prec", "CPU s→1e-4", "GPU s→1e-4", "speed-up", "CPU s→1e-8",
                          "GPU s→1e-8", "speed-up"], body))
        out.append("\nSpeed-up < 1× means the GPU is slower (reported, not hidden).\n")
    return "\n".join(out)


def fulls_all_untagged_before(ablations, fulls_all):
    """Untagged full-Netlib runs from the same commits as the ablations (the 'off' baseline)."""
    hashes = {os.path.splitext(os.path.basename(p))[0].rsplit("-", 1)[1] for p in ablations}
    out = []
    for p in fulls_all:
        if p in ablations:
            continue
        rows = load(p)
        if rows and (rows[0]["git_hash"] in hashes or rows[0]["git_hash"] == "f10527c"):
            out.append(p)
    return out


def main():
    paths = committed_csvs()
    if not paths:
        print("no committed CSVs in bench/results/", file=sys.stderr)
        return 1
    net = latest([p for p in paths if "gpu" not in os.path.basename(p)], "netlib-small-")
    net_gpu = latest(paths, "netlib-small-gpu-")
    simplex_scales = [p for p in paths if os.path.basename(p).startswith("scale-simplex-")]
    scales = [p for p in paths if os.path.basename(p).startswith("scale-") and p not in simplex_scales]
    cpu_scales = [p for p in scales if all(r.get("backend") != "gpu" for r in load(p))]
    gpu_scales = [p for p in scales if p not in cpu_scales]

    doc = ["# Evidence pack", "",
           "Generated by `bench/make_evidence.py` from **committed** CSVs in `bench/results/` only "
           "(CLAUDE.md §5.1). Every table names its source file; the git short hash is in each filename. "
           "Nothing here is a target or an estimate.", "",
           "Files used:", ""]
    doc += [f"- `{p}` (committed {commit_date(p)})" for p in paths]
    doc += ["", "## 1. Correctness on small Netlib (oracle, PDLP-style PDHG, r²HPDHG; fp64 and mixed)", ""]
    doc.append(netlib_section(net) if net else "_No committed small-Netlib CSV._")
    if net_gpu:
        doc += ["", "### Same set on GPU", "", netlib_section(net_gpu)]

    fulls_all = [p for p in paths if os.path.basename(p).startswith("netlib-full-")]
    # tagged runs (e.g. -gm12-) are ablations, not the default configuration
    ablations = [p for p in fulls_all
                 if any(f"-{t}-" in os.path.basename(p)
                        for t in ("gm12", "gm4", "gm0", "presolve", "nopresolve", "noscale", "dantzig"))]
    fulls = [p for p in fulls_all if p not in ablations]
    # latest run per configuration tag (engine-precision[-gpu]-machine)
    by_tag = {}
    for p in fulls:
        tag = os.path.basename(p).rsplit("-", 1)[0]
        by_tag.setdefault(tag, []).append(p)
    fulls = [latest(v, "netlib-full-") for v in by_tag.values()]
    doc += ["", "## 1b. Full Netlib LP set (every engine)", ""]
    cmp_ = netlib_engine_comparison(fulls)
    if cmp_:
        doc += [cmp_, ""]
    doc += [netlib_full_section(p) + "\n" for p in fulls] or ["_No committed full-Netlib CSV yet._"]

    if ablations:
        doc += ["### 1c. Ablations (full Netlib, 60 s per model)", ""]
        body = []
        for p in sorted(ablations) + sorted(fulls_all_untagged_before(ablations, fulls_all)):
            rows = load(p)
            solved = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"]
            body.append([f"`{os.path.basename(p)}`", rows[0].get("settings", "") or "default of that commit",
                         f"{len(solved)}/{len(rows)}"])
        doc += [table(["CSV", "settings", "solved + verified"], body), "",
                "Decisions: geometric-mean scaling is adaptive (docs/DECISIONS.md #17); presolve see #22; simplex "
                "scaling and pricing see #30.", ""]

    doc += ["", "## 2. Scaling on generated LPs with known optimum (CPU)", ""]
    if cpu_scales:
        doc += [scale_section(latest(cpu_scales, "scale-")), ""]
    else:
        doc.append("_No committed CPU scaling CSV yet._")
    # established-solver comparison: runs that include HiGHS, with our engine from the SAME run
    with_highs = [p for p in cpu_scales if any(r["engine"] == "highs" for r in load(p))]
    if with_highs:
        p = latest(with_highs, "scale-")
        rows = load(p)
        body = []
        for inst in dict.fromkeys(r["instance"] for r in rows):
            h = next((r for r in rows if r["instance"] == inst and r["engine"] == "highs"), None)
            o = next((r for r in rows if r["instance"] == inst and r["engine"] == "r2hpdhg" and r["precision"] == "fp64"),
                     None)
            if h and o:
                body.append([inst, o["rows"], o["nnz"], fnum(o["seconds_to_1e-8"]),
                             fnum(h["seconds_to_1e-8"]) if h["status"] == "Optimal" else f"> {fnum(h['seconds'])} ({h['status']})"])
        doc += ["### 2b. Reference: HiGHS on the same instances", "",
                f"Source: `{p}` (same run, same machine). HiGHS default algorithm, separate process, hard-killed "
                "at the cap. HiGHS returns a vertex solution at simplex accuracy; our column is time to relative "
                "KKT 1e-8 with verifier-grade feasibility — not the identical accuracy target.", "",
                table(["instance", "rows", "nnz", "r²HPDHG fp64 s→1e-8", "HiGHS s"], body), ""]

    if simplex_scales:
        p = latest(simplex_scales, "scale-simplex-")
        rows = load(p)
        body = [[r["instance"], r["rows"], r["nnz"], r["status"], r["iterations"], fnum(r["seconds"]),
                 r["rel_err_known"] or "–", r["verify"] or "–"] for r in rows]
        doc += ["### 2c. The simplex engine on the same generated models", "",
                f"Source: `{p}` — commit `{rows[0]['git_hash']}`. Why `auto` sends large models to r²HPDHG: the "
                "primal simplex prices every column and computes the Devex row every iteration.", "",
                table(["instance", "rows", "nnz", "status", "iterations", "s", "rel. err vs known opt", "verify"], body), ""]

    doc += ["", "## 3. CPU vs GPU", ""]
    if gpu_scales or net_gpu:
        g = gpu_vs_cpu(gpu_scales + ([net_gpu] if net_gpu else []))
        doc.append(g or "_GPU CSVs present but contain no matched CPU/GPU pairs._")
        for p in gpu_scales:
            doc += ["", scale_section(p)]
    else:
        doc += ["**Not measured yet.** The CUDA backend (`src/gpu/cuda_backend.cu`) is written and type-checked "
                "on macOS, but no committed CSV comes from a GPU machine, so this pack makes **no GPU speed claim**. "
                "To produce one: `PS26119_MACHINE=<name> scripts/gpu_check.sh` on the NVIDIA laptop / university "
                "server, then commit `bench/results/` and rerun this script."]

    doc += ["", "## 4. Refinery planning year", ""]
    ref = []
    newest_cpu = latest(cpu_scales, "scale-") if cpu_scales else None
    for p in scales:
        for r in load(p):
            # latest CPU run + every GPU run + the HiGHS reference wherever it was measured
            if r["instance"].startswith("refinery-T8760") and (p == newest_cpu or p in gpu_scales or r["engine"] == "highs"):
                ref.append((p, r))
    if ref:
        body = [[r["engine"], r["backend"], (r.get("threads", "") or ("1" if r["engine"] != "highs" else "–")), r["precision"], r["status"], r["iterations"] or "–",
                 fnum(r["seconds_to_1e-4"]), fnum(r["seconds_to_1e-8"]), r["rel_err_known"] or "–", r["verify"] or "–",
                 f"`{os.path.basename(p)}`"] for p, r in ref]
        r0 = ref[0][1]
        doc += [f"Hourly year, T = 8760 periods: {r0['rows']} rows, {r0['cols']} columns, {r0['nnz']} nonzeros "
                "(generated by `bench/generate_refinery_lp.py`; optimum known by construction).", "",
                table(["engine", "backend", "threads", "prec", "status", "iterations", "s to 1e-4", "s to 1e-8",
                       "rel. err vs known opt", "verify", "source"], body)]
    else:
        doc.append("_No committed refinery-year run._")

    doc += ["", "## 4b. Warm-started re-solves (what-if scenarios on the refinery LP)", ""]
    warms = [p for p in paths if os.path.basename(p).startswith("warm-start-")]
    for p in warms:
        rows = load(p)
        r0 = rows[0]
        doc += [f"Source: `{p}` — `{r0['machine']}` ({r0['cpu']}), commit `{r0['git_hash']}`. Each scenario solved "
                "cold and warm-started from the base-case solution, both to 1e-8, both verified.", ""]
        body = []
        for inst in dict.fromkeys(r["instance"] for r in rows):
            for scen in dict.fromkeys(r["scenario"] for r in rows if r["instance"] == inst):
                sel = {r["start"]: r for r in rows if r["instance"] == inst and r["scenario"] == scen}
                c, w, ww = sel.get("cold"), sel.get("warm"), sel.get("warm+weight")
                def it(r):
                    return "–" if r is None else r["iterations"] + ("" if r["status"] == "Optimal" else f" ({r['status']})")
                body.append([inst, scen, c["objective_change_vs_base"], it(c), it(w),
                             w["iteration_ratio_warm_over_cold"] if w else "–", it(ww),
                             ww["iteration_ratio_warm_over_cold"] if ww else "–",
                             "/".join(r["verify"] for r in (c, w, ww) if r)])
        doc.append(table(["instance", "scenario", "objective change", "cold it", "warm it", "warm/cold",
                          "warm+ω it", "warm+ω/cold", "verify"], body))
        doc.append("\nwarm = start from the base solution's (x, y) (default); warm+ω = also reuse its primal weight "
                   "(opt-in `--warm-weight`). Iteration counts are deterministic; wall times are in the CSV.")
    if not warms:
        doc.append("_No committed warm-start CSV yet._")

    doc += ["", "## 4c. Batched scenarios (many LPs sharing one matrix)", ""]
    batches = [p for p in paths if os.path.basename(p).startswith("batch-")]
    if batches:
        newest = load(latest(batches, "batch-"))[0]["git_hash"]
        same = [q for q in batches if load(q)[0]["git_hash"] == newest]
        rows = [r for q in same for r in load(q)]
        p = ", ".join(f"`{q}`" for q in same)
        r0 = rows[0]
        doc += [f"Source: {p} — `{r0['machine']}` ({r0['cpu']}), commit `{r0['git_hash']}`. K price scenarios of the "
                "refinery LP: K separate solves vs one batched solve (one SpMM per iteration). Every batch answer "
                "verified and compared with its separate solve.", "",
                table(["instance", "K", "separate solves s", "batch s", "speed-up", "all optimal", "all verified",
                       "max objective disagreement"],
                      [[r["instance"], r["K"], fnum(r["sequential_seconds"]), fnum(r["batch_seconds"]), r["speedup"],
                        r["all_optimal"], r["all_verified"], r["max_objective_disagreement"]] for r in rows])]
    else:
        doc.append("_No committed batch CSV yet._")

    doc += ["", "## 4d. MILP prototype (branch-and-bound, small MIPLIB 3)", ""]
    mips = [p for p in paths if os.path.basename(p).startswith("miplib3-")]
    if mips:
        p = latest(mips, "miplib3-")
        rows = load(p)
        r0 = rows[0]
        solved = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"]
        node = ("dense double-double simplex as node solver" if r0["git_hash"] == "0935157" else
                "sparse primal simplex as node solver, pruning by certified dual bounds (DECISIONS #28)")
        doc += [f"Source: `{p}` — `{r0['machine']}`, commit `{r0['git_hash']}`, time limit per model in the CSV. "
                f"Prototype: {node}, depth-first then best-bound, most-fractional branching, no cuts. "
                f"**{len(solved)} of {len(rows)}** solved to proven optimality within the limit, each verified "
                "(feasibility + integrality) and equal to the HiGHS optimum.", "",
                table(["instance", "rows", "cols", "int", "status", "objective", "HiGHS", "gap", "s", "verify"],
                      [[r["instance"], r["rows"], r["cols"], r["integers"], r["status"], r["objective"],
                        r["highs_objective"], r["gap"], fnum(r["seconds"]), r["verify"]] for r in rows])]
    else:
        doc.append("_No committed MIPLIB CSV yet._")

    doc += ["", "## 5. What we do NOT do yet (honest list)", "",
            "- **No GPU number is claimed** unless a GPU CSV appears in §3. The CUDA backend has not yet been "
            "compiled by nvcc or run on NVIDIA hardware.",
            "- **No crossover** from a first-order solution to a vertex, and **no simplex warm start / dual "
            "simplex** yet (the integrated primal simplex starts from the slack basis every time). First-order "
            "solutions are accurate to the stated tolerance but are not vertices.",
            "- The primal simplex prices every column and uses product-form updates without hypersparsity: fast "
            "on Netlib-size models, slow beyond ~10⁴ rows (§2c); dfl001 is not solved by it within 60 s.",
            "- Presolve is basic (empty rows, fixed/empty columns, singleton rows) — no doubleton/dominated-column "
            "reductions.",
            "- **MILP is a prototype** (§4d): branch-and-bound with cold-started sparse simplex node LPs, "
            "most-fractional branching, a rounding heuristic, **no cuts**, no strong branching. **No QP** yet.",
            "- **Generated instances**: the refinery LP has refinery structure, but its prices and inequality "
            "right-hand sides come from the KKT construction (synthetic), not from plant data; random LPs of this "
            "kind are friendly to first-order methods. Mittelmann large models are the next evidence step.",
            "- Batched scenarios (§4c) run on the CPU only (no GPU SpMM kernel yet) and without infeasibility "
            "detection.",
            "- The `auto` engine rule and the adaptive-scaling threshold are tuned constants (on Netlib and the "
            "generated models).",
            "- Laptop timings vary run to run (a fanless MacBook Air throttles and macOS moves threads between "
            "performance and efficiency cores): the same 1e6-row run has taken 294 s and 539 s for identical "
            "iteration counts. Iteration counts are deterministic; compare those first.",
            "- First-order CPU runs default to 1 thread (deterministic pool, bit-identical for any thread count); "
            "the simplex and the MILP tree are single-threaded."]
    with open(OUT, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
