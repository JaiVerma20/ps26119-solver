"""report.py — a self-contained HTML report (no scripts, no external files; prints to PDF from any
browser): the verification certificates of chosen runs and, optionally, the benchmark evidence.

Every value is copied from certificate.build() (the run's own outputs) or evidence.collect() /
evidence.gpu_detail() (the committed hash-named CSVs); nothing is computed here except formatting.
"""
from __future__ import annotations

import datetime as _dt
import html
import math

from . import evidence as _evidence
from . import system

CSS = """
:root{--bg:#0b0e12;--panel:#11161d;--line:#223041;--text:#e6edf3;--t2:#9fb0c0;--t3:#6b7c8d;--acc:#5ce1e6;--ok:#3ddc97;--bad:#ff5d6c;--warn:#ffb547}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1060px;margin:0 auto;padding:32px 28px 60px}
.mono,td.num,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:14px;margin:18px 0 8px;color:var(--t2);text-transform:uppercase;letter-spacing:.08em}
.eyebrow{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.16em;color:var(--acc);text-transform:uppercase}
.sub{color:var(--t2)}.dim{color:var(--t3)}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.cert{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:22px 24px;margin:18px 0;break-inside:avoid}
.head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}
.stamp{border:3px solid var(--ok);color:var(--ok);border-radius:8px;padding:8px 16px;font:800 22px ui-monospace,Menlo,monospace;letter-spacing:.12em;white-space:nowrap}
.stamp.no{border-color:var(--bad);color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:16px 0}
.grid div{background:var(--panel);padding:9px 12px;min-width:0}.grid .k{font:600 10px ui-monospace,Menlo,monospace;letter-spacing:.12em;color:var(--t3);text-transform:uppercase}
.grid .v{font:600 15px ui-monospace,Menlo,monospace;overflow-wrap:anywhere}.grid .wide{grid-column:span 2}
table{width:100%;border-collapse:collapse;margin:8px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 10px ui-monospace,Menlo,monospace;letter-spacing:.1em;color:var(--t3);text-transform:uppercase}td.num,th.num{text-align:right}
.src{font:11px ui-monospace,Menlo,monospace;color:var(--t3)}.note{font-size:12px;color:var(--t2);margin:6px 0}
footer{margin-top:40px;border-top:1px solid var(--line);padding-top:14px;font-size:12px;color:var(--t3)}
@media print{:root{--bg:#fff;--panel:#fff;--line:#c9d2dc;--text:#0b0e12;--t2:#33414f;--t3:#5b6b7b;--acc:#006d77;--ok:#08794b;--bad:#b3261e;--warn:#8a5a00}
main{padding:0;max-width:none}.cert{border-color:#9aa8b6}h2{break-after:avoid}}
@media (max-width:760px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.head{flex-direction:column}}
"""


SETS = (("netlib", "Netlib"), ("kennington", "Kennington"), ("scale", "large"))


def e(v) -> str:
    return html.escape("" if v is None else str(v))


def num(v, sig=10) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if isinstance(v, (int, float)) and abs(v) >= 1e-3 and abs(v) < 1e15:
        s = f"{v:,.{max(0, sig - len(str(int(abs(v)))))}f}" if isinstance(v, float) else f"{v:,}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    return f"{v:.3g}" if isinstance(v, (int, float)) else e(v)


def exp(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2e}"


def secs(v) -> str:
    return "—" if v is None else (f"{v * 1000:.1f} ms" if v < 1 else f"{v:.2f} s")


def ratio(v) -> str:
    return "—" if v is None else "%.2f×" % v


def cell(k, v, wide=False) -> str:
    return f'<div class="{"wide" if wide else ""}"><div class="k">{e(k)}</div><div class="v">{v}</div></div>'


def certificate_html(c: dict) -> str:
    f, m, r, run, g, v = c["final"], c["model"], c["result"], c["run"], c["gate"], c["verifier"]
    ok = f["verdict"] == "PASS"
    grid = "".join([
        cell("model", e(m["name"]), True), cell("status", e(r["status"])), cell("objective", num(r["objective"], 12)),
        cell("rows × cols", f'{num(m["rows"])} × {num(m["cols"])}'), cell("nonzeros", num(m["nnz"])),
        cell("engine", e(run["engine"])), cell("backend", e(run["backend"]) + (" · all cores" if run["threads"] == 0 else " · %s thr" % e(run["threads"] or 1))),
        cell("precision", e(run["precision"])), cell("iterations", num(run["iterations"])), cell("solve time", secs(run["solve_seconds"])),
        cell("dual objective", num(r["dual_objective"], 12)),
        cell("primal residual (engine)", exp(r["primal_residual"])), cell("dual residual (engine)", exp(r["dual_residual"])),
        cell("worst-row primal (gate)", exp(g.get("primal"))), cell("worst-row dual (gate)", exp(g.get("dual"))),
        cell("model fingerprint", f'<span class="mono">{e(m["fingerprint"])}</span>', True),
        cell("solver", f'{e(c["solver"]["version"])}', True),
    ])
    rows = "".join(
        f'<tr><td>{"<span class=ok>✓</span>" if ck["ok"] else ("<span class=bad>✕</span>" if ck["required"] else "<span class=dim>·</span>")}</td>'
        f'<td>{e(ck["name"])}{"" if ck["required"] else " <span class=dim>(informative)</span>"}</td><td class="mono">{e(ck["detail"])}</td></tr>'
        for ck in c["checks"])
    ver = (f'<tr><td>verify.py</td><td class="num">{exp(v["primal_rel"])}</td><td class="num">{exp(v["dual_rel"])}</td>'
           f'<td class="num">{exp(v["gap_rel"])}</td><td class="num">{num(v["objective"], 12)}</td></tr>') if v.get("verdict") else ""
    gate = (f'<tr><td>in-process gate</td><td class="num">{exp(g.get("primal"))}</td><td class="num">{exp(g.get("dual"))}</td>'
            f'<td class="num">{exp(g.get("gap"))}</td><td class="num">{num(r["objective"], 12)}</td></tr>')
    art = c["artifact"]
    failed = "" if ok else '<p class="bad">Failed: %s</p>' % e(", ".join(f["reasons"]))
    twochecks = "" if r["status"] != "Optimal" else (
        '<h3>Same numbers, two independent checks (worst row / column, relative)</h3>'
        '<table><thead><tr><th>checked by</th><th class="num">primal</th><th class="num">dual</th><th class="num">gap</th>'
        '<th class="num">objective</th></tr></thead><tbody>' + gate + ver + '</tbody></table>')
    certline = '<p class="note">Certificate: <span class="mono">%s</span></p>' % e(v["certificate"]) if v.get("certificate") else ""
    boundline = '<p class="note">Bound: <span class="mono">%s</span></p>' % e(r["certified_line"]) if r.get("certified_line") else ""
    stamp_cls, stamp = ("", "✓ PASS") if ok else ("no", "NOT CERTIFIED")
    return f"""
<section class="cert">
 <div class="head"><div><div class="eyebrow">verification certificate · run {e(c["id"])}</div>
  <h1>{e(m["name"])}</h1><div class="sub mono">{e(m["path"])} · {e(c["created"])}</div></div>
  <div class="stamp {stamp_cls}">{stamp}</div></div>
 {failed}
 <div class="grid">{grid}</div>
 <h3>Checks</h3><table><tbody>{rows}</tbody></table>
 {twochecks}
 {certline}
 {boundline}
 <p class="note">Command: <code>{e(c["solver"]["command"])}</code></p>
 <p class="note">Solution file <code>{e(art["solution_file"])}</code> · SHA-256 <code>{e(art["sha256"])}</code></p>
 <p class="note">Machine: {e(c["machine"]["cpu"])} · {e(c["machine"]["cores"])} cores · {e(c["machine"]["os"])}</p>
</section>"""


def src(s: dict) -> str:
    return f'<div class="src">source: bench/results/{e(s["file"])} · commit {e(s["git_hash"])} · {e(s["machine"])} · {e(s["date"])}</div>'


def evidence_html() -> str:
    ev, gd = _evidence.collect(), _evidence.gpu_detail()
    out = ['<h2>Benchmark evidence (committed CSVs)</h2>',
           '<p class="note">Read from the hash-named CSVs in bench/results/ with the rules of docs/EVIDENCE.md. '
           'Solved = Optimal + in-process gate + tools/verify.py PASS + |obj − HiGHS| / (1 + |HiGHS|) ≤ 1e-6.</p>']
    if ev["netlib"]:
        out.append("<h3>Netlib LP · all models · 60 s each</h3><table><thead><tr><th>engine</th><th class=num>solved</th>"
                   "<th class=num>seconds (solved)</th><th>not solved</th></tr></thead><tbody>")
        out += [f'<tr><td class=mono>{e(n["engine"])}</td><td class=num>{n["solved"]}/{n["total"]}</td><td class=num>{num(n["seconds_solved"])}</td>'
                f'<td class=dim>{e(", ".join(n["not_solved"]) or "—")}</td></tr>' for n in ev["netlib"]]
        out.append("</tbody></table>" + src(ev["netlib"][0]["source"]))
    if ev["infeasible"]:
        out.append("<h3>Certified infeasibility</h3><table><thead><tr><th>engine</th><th class=num>certified</th><th class=num>exact rational</th></tr></thead><tbody>")
        out += [f'<tr><td class=mono>{e(x["engine"])}</td><td class=num>{x["certified"]}/{x["total"]}</td><td class=num>{x["exact_rational"]}</td></tr>'
                for x in ev["infeasible"]]
        out.append("</tbody></table>" + src(ev["infeasible"][0]["source"]))
    if ev["miplib"]:
        mp = ev["miplib"]
        out.append(f'<h3>MILP · small MIPLIB 3 · {mp["solved"]}/{mp["total"]} proven optimal and verified</h3>' + src(mp["source"]))
    for run in _evidence.compare()["runs"][:1]:
        rows = run["rows"]

        def key(r):
            return f'{r["solver"]} {r["engine"]}' + (" (all cores)" if r["solver"] == "ps26119" and str(r["threads"]).split(".")[0] != "1" else "")
        keys = list(dict.fromkeys(key(r) for r in rows))
        out.append(f'<h3>Against a real-world solver · {e(" vs ".join(run["solver_versions"]))}</h3>'
                   '<table><thead><tr><th>engine</th>' + "".join(f"<th class=num>{lab} solved</th>" for sid, lab in SETS
                                                                if any(r["set"] == sid for r in rows)) +
                   '<th class=num>"Optimal" claims rejected</th></tr></thead><tbody>')
        present = [sid for sid, _ in SETS if any(r["set"] == sid for r in rows)]
        for k in keys:
            cells = []
            for sid in present:
                rs = [r for r in rows if key(r) == k and r["set"] == sid]
                cells.append(f'{sum(r["solved"] == "yes" for r in rs)}/{len(rs)}' if rs else "–")
            claims = [r for r in rows if key(r) == k and r["status"] == "Optimal"]
            rej = sum(r["solved"] != "yes" for r in claims)
            out.append(f'<tr><td class=mono>{e(k)}</td>' + "".join(f"<td class=num>{c}</td>" for c in cells) +
                       f'<td class=num>{rej} of {len(claims)}</td></tr>')
        out.append("</tbody></table>" + src(run["source"]) +
                   '<p class="note">Same models, same machine, one rule for both solvers: Optimal AND tools/verify.py PASS AND '
                   'within 1e-6 of the reference; time = the solve call only. HiGHS runs as a separate reference.</p>')
    for mach in gd["machines"]:
        out.append(f'<h3>GPU vs CPU · {e(mach["gpu"])}</h3>'
                   f'<p class="note">{e(mach["gpu"])} · driver {e(mach["driver"])} · CUDA {e(mach["cuda"])} · CPU baseline {e(mach["cpu"])} '
                   f'({e(mach["cpu_cores"])} threads) · compute-sanitizer memcheck {e(mach["sanitizer"].get("memcheck"))}, '
                   f'racecheck {e(mach["sanitizer"].get("racecheck"))}'
                   + (f' · GPU-machine ctest {mach["logs"]["ctest"]["passed"]}/{mach["logs"]["ctest"]["total"]}' if mach["logs"].get("ctest") else "") + "</p>")
        out.append("<table><thead><tr><th>instance</th><th class=num>rows</th><th>precision</th><th>GPU status</th><th class=num>GPU s</th>"
                   "<th class=num>best CPU s</th><th class=num>vs best CPU</th><th class=num>vs 1 thread</th></tr></thead><tbody>")
        for p in mach["pairs"]:
            thr = " (%s thr)" % e(p["best_cpu_threads"]) if p["best_cpu_threads"] else ""
            out.append(f'<tr><td class=mono>{e(p["instance"])}</td><td class=num>{num(p["rows"])}</td><td class=mono>{e(p["precision"])}</td>'
                       f'<td>{e(p["gpu_status"])}</td><td class=num>{secs(p["gpu_s"])}</td>'
                       f'<td class=num>{secs(p["best_cpu_s"])}{thr}</td>'
                       f'<td class=num>{ratio(p["ratio_vs_best"])}</td><td class=num>{ratio(p["ratio_vs_1thread"])}</td></tr>')
        out.append("</tbody></table>" + src(mach["source"]) +
                   '<p class="note">Ratio = CPU seconds / GPU seconds to relative KKT 1e-8; below 1× the GPU is slower (small models). One consumer laptop GPU.</p>')
    return "\n".join(out)


def render(certs: list[dict], include_evidence: bool = True, title: str = "Verification report") -> str:
    s = system.probe()
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    passed = sum(c["final"]["verdict"] == "PASS" for c in certs)
    summary = ""
    if len(certs) > 1:
        summary = ("<h2>Runs in this report</h2><table><thead><tr><th>run</th><th>model</th><th>status</th><th class=num>objective</th>"
                   "<th>engine</th><th class=num>time</th><th>verdict</th></tr></thead><tbody>" + "".join(
                       f'<tr><td class=mono>{e(c["id"])}</td><td class=mono>{e(c["model"]["name"])}</td><td>{e(c["result"]["status"])}</td>'
                       f'<td class=num>{num(c["result"]["objective"], 12)}</td><td class=mono>{e(c["run"]["engine"])}</td>'
                       f'<td class=num>{secs(c["run"]["solve_seconds"])}</td>'
                       f'<td class="{"ok" if c["final"]["verdict"] == "PASS" else "bad"}">{e(c["final"]["verdict"])}</td></tr>' for c in certs)
                   + f"</tbody></table><p class=note>{passed} of {len(certs)} runs certified.</p>")
    body = "".join(certificate_html(c) for c in certs)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)} · PS26119</title><style>{CSS}</style></head><body><main>
<div class="eyebrow">PS26119 optimization solver · SIH 2026 PS 26119</div>
<h1>{e(title)}</h1>
<div class="sub">Generated {e(now)} by the Command Center · solver {e(s.get("version"))} · {e(s.get("cpu"))}, {e(s.get("cores"))} cores · {e(s.get("os"))}</div>
{summary}{body}{evidence_html() if include_evidence else ""}
<footer>Every value in this report is copied from the solver's own output files, the independent verifier's report
(tools/verify.py: a different model reader and separate checking code) or a committed, hash-named benchmark CSV. Nothing is
simulated or typed in by hand. Reproduce a run with the command shown; reproduce the evidence with scripts/reproduce.sh (CPU)
and scripts/gpu_check.sh (GPU).</footer>
</main></body></html>"""
