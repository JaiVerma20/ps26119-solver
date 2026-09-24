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

    doc += ["", "## 2. Scaling on generated LPs with known optimum (CPU)", ""]
    if cpu_scales:
        for p in cpu_scales:
            doc += [scale_section(p), ""]
    else:
        doc.append("_No committed CPU scaling CSV yet._")

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

    doc += ["", "## 5. What we do NOT do yet (honest list)", "",
            "- **No GPU number is claimed** unless a GPU CSV appears in §3. The CUDA backend has not yet been "
            "compiled by nvcc or run at the time this list was written.",
            "- **No MPS reader of our own in this build** — it is owned by a teammate; until it lands, MPS files are "
            "converted with `tools/mps_to_lpm.py` (highspy, tooling only, never linked).",
            "- **No sparse simplex, no crossover, no presolve** yet: first-order solutions are accurate to the stated "
            "tolerance but are not vertices; exact duals/ranging need the (planned) crossover.",
            "- **No infeasibility / unboundedness detection in the first-order engines**: they report "
            "IterationLimit/TimeLimit instead (the dense oracle does detect both, on small models only).",
            "- **No MILP, no QP** yet (branch-and-bound and PDHG-QP are post-PPT milestones).",
            "- **Generated instances**: the refinery LP has refinery structure, but its prices and inequality "
            "right-hand sides come from the KKT construction (synthetic), not from plant data; random LPs of this "
            "kind are friendly to first-order methods. Netlib / Mittelmann large models are the next evidence step.",
            "- **Warm start, batched scenarios, certified dual bounds** (CLAUDE.md §9 items 3–5) are not implemented.",
            "- CPU runs are single-threaded (deterministic by default; OpenMP is optional and off)."]
    with open(OUT, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
