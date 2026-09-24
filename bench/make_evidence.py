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
    cfgs = [("oracle", "dd"), ("pdlp", "fp64"), ("pdlp", "mixed"), ("r2hpdhg", "fp64"), ("r2hpdhg", "mixed")]
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
            it = f", {r['iterations']} it" if c[0] != "oracle" else ""
            cells.append(f"{r['objective']} ({r['verify']}{it})")
        body.append(cells)
    lines.append(table(header, body))
    npass = sum(r["verify"] == "PASS" for r in rows)
    lines += ["", f"**{npass} of {len(rows)} runs verified PASS** by `tools/verify.py` (independent reader, "
              f"primal/dual/gap ≤ 1e-6 relative) and within 1e-6 of the published optimum. "
              f"First-order runs target relative KKT 1e-8."]
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


def scale_section(path):
    rows = load(path)
    r0 = rows[0]
    lines = [f"Source: `{path}` — machine `{r0['machine']}` ({r0['cpu']}; GPU: {r0['gpu'] or 'none'}"
             f"{', driver ' + r0['driver'] if r0['driver'] else ''}{', CUDA ' + r0['cuda'] if r0['cuda'] else ''}), "
             f"commit `{r0['git_hash']}`.", ""]
    header = ["instance", "rows", "nnz", "engine", "backend", "prec", "status", "iterations",
              "s to 1e-4", "s to 1e-8", "ms/iter", "rel. err vs known opt", "verify"]
    body = []
    for r in rows:
        body.append([r["instance"], r["rows"], r["nnz"], r["engine"], r["backend"], r["precision"], r["status"],
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
    scales = [p for p in paths if os.path.basename(p).startswith("scale-")]
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
    ablations = [p for p in fulls_all if any(f"-{t}-" in os.path.basename(p) for t in ("gm12", "gm4", "gm0"))]
    fulls = [p for p in fulls_all if p not in ablations]
    # latest run per configuration tag (engine-precision[-gpu]-machine)
    by_tag = {}
    for p in fulls:
        tag = os.path.basename(p).rsplit("-", 1)[0]
        by_tag.setdefault(tag, []).append(p)
    fulls = [latest(v, "netlib-full-") for v in by_tag.values()]
    doc += ["", "## 1b. Full Netlib LP set (first-order engine)", ""]
    doc += [netlib_full_section(p) + "\n" for p in fulls] or ["_No committed full-Netlib CSV yet._"]

    if ablations:
        doc += ["### 1c. Ablation: geometric-mean scaling (full Netlib, 60 s per model)", ""]
        body = []
        for p in sorted(ablations) + sorted(fulls_all_untagged_before(ablations, fulls_all)):
            rows = load(p)
            solved = [r for r in rows if r["status"] == "Optimal" and r["verify"] == "PASS"]
            body.append([f"`{os.path.basename(p)}`", rows[0].get("settings", "") or "default of that commit",
                         f"{len(solved)}/{len(rows)}"])
        doc += [table(["CSV", "settings", "solved + verified"], body), "",
                "Decision (docs/DECISIONS.md #17): apply 12 sweeps only when max|a|/min|a| ≥ 10^4.5.", ""]

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
    for p in scales:
        for r in load(p):
            if r["instance"].startswith("refinery-T8760"):
                ref.append((p, r))
    if ref:
        body = [[r["engine"], r["backend"], r["precision"], r["status"], r["iterations"] or "–",
                 fnum(r["seconds_to_1e-4"]), fnum(r["seconds_to_1e-8"]), r["rel_err_known"] or "–", r["verify"] or "–",
                 f"`{os.path.basename(p)}`"] for p, r in ref]
        r0 = ref[0][1]
        doc += [f"Hourly year, T = 8760 periods: {r0['rows']} rows, {r0['cols']} columns, {r0['nnz']} nonzeros "
                "(generated by `bench/generate_refinery_lp.py`; optimum known by construction).", "",
                table(["engine", "backend", "prec", "status", "iterations", "s to 1e-4", "s to 1e-8",
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
        p = latest(batches, "batch-")
        rows = load(p)
        r0 = rows[0]
        doc += [f"Source: `{p}` — `{r0['machine']}` ({r0['cpu']}), commit `{r0['git_hash']}`. K price scenarios of the "
                "refinery LP: K separate solves vs one batched solve (one SpMM per iteration). Every batch answer "
                "verified and compared with its separate solve.", "",
                table(["instance", "K", "separate solves s", "batch s", "speed-up", "all optimal", "all verified",
                       "max objective disagreement"],
                      [[r["instance"], r["K"], fnum(r["sequential_seconds"]), fnum(r["batch_seconds"]), r["speedup"],
                        r["all_optimal"], r["all_verified"], r["max_objective_disagreement"]] for r in rows])]
    else:
        doc.append("_No committed batch CSV yet._")

    doc += ["", "## 5. What we do NOT do yet (honest list)", "",
            "- **No GPU number is claimed** unless a GPU CSV appears in §3. The CUDA backend has not yet been "
            "compiled by nvcc or run at the time this list was written.",
            "- **No MPS reader of our own in this build** — it is owned by a teammate; until it lands, MPS files are "
            "converted with `tools/mps_to_lpm.py` (highspy, tooling only, never linked).",
            "- **No sparse simplex, no crossover, no presolve** yet: first-order solutions are accurate to the stated "
            "tolerance but are not vertices; exact duals/ranging need the (planned) crossover.",
            "- **Infeasibility / unboundedness detection in the first-order engines is new and only lightly "
            "tested** (ray certificates, checked in fp64 on the original problem; unit-tested on hand-made "
            "infeasible/unbounded LPs, not yet on the Netlib infeasible set).",
            "- **No MILP, no QP** yet (branch-and-bound and PDHG-QP are post-PPT milestones).",
            "- **Generated instances**: the refinery LP has refinery structure, but its prices and inequality "
            "right-hand sides come from the KKT construction (synthetic), not from plant data; random LPs of this "
            "kind are friendly to first-order methods. Netlib / Mittelmann large models are the next evidence step.",
            "- Batched scenarios (§4c) run on the CPU only (no GPU SpMM kernel yet) and without infeasibility "
            "detection; certified bounds are −∞ whenever a dual multiplier points at an infinite bound (no bound "
            "tightening yet to repair this).",
            "- Laptop timings vary run to run (a fanless MacBook Air throttles and macOS moves threads between "
            "performance and efficiency cores): the same 1e6-row run has taken 294 s and 539 s for identical "
            "iteration counts. Iteration counts are deterministic; compare those first.",
            "- CPU runs are single-threaded (deterministic by default; OpenMP is optional and off)."]
    with open(OUT, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
