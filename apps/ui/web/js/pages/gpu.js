// pages/gpu.js — GPU compute: every configuration of the committed GPU runs (CPU 1 thread, CPU all
// cores, GPU; fp64 and mixed precision) from bench/results/scale-<gpu machine>-<hash>.csv and the
// run's committed logs. Headline figures are picked from the data, never typed in.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, fnum, fexp, fint, fsec, fratio, srcChip, statusBadge, passBadge, toast } from "../util.js";
import { barChart } from "../charts.js";

const cfgLabel = (r, cores) => r.backend === "gpu" ? `GPU · ${r.precision}` : `CPU ${r.threads === 1 ? "1 thread" : `${r.threads || cores} threads`} · ${r.precision}`;
const t8 = (r) => (r.status === "Optimal" ? r.s_1e8 : null);

export function mount(root) {
  let data = null, sel = null, mi = 0;
  const body = h("div.grid", { style: { gap: "14px" } });
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "08 · gpu compute"), h("h1", "GPU compute"),
      h("p", "r²HPDHG on the CUDA backend against the same engine on the CPU (1 thread and all cores), in fp64 and in mixed precision (fp32 iterations, fp64 residuals and restarts). Every figure is read from the committed GPU run and its logs; ratios follow bench/gpu_compare.py (CPU seconds / GPU seconds to relative KKT 1e-8)."))),
    body);

  function live() {
    const s = store.system;
    if (!s) return null;
    return s.cuda_build
      ? h("div.panel.claim-ok", h("div.panel-b", h("b.ok", "● This build has the CUDA backend "), `(${s.gpu}). Run any model live on the GPU from the Solve page (Backend → GPU).`, " ",
          h("a", { href: "#/solve" }, "Open Solve ▸")))
      : h("div.panel", h("div.panel-b.muted", h("b", { style: { color: "var(--text)" } }, "This machine: CPU-only build "), `(${s.cpu}). The GPU figures below are the committed run on the GPU machine named in each table — shown, not re-run here.`));
  }

  function draw() {
    if (!data) { body.replaceChildren(h("div.panel", h("div.empty", "reading GPU evidence…"))); return; }
    if (!data.machines.length) { body.replaceChildren(...[live(), h("div.panel", h("div.empty", "no committed GPU run in bench/results/"))].filter(Boolean)); return; }
    const M = data.machines[mi], cores = M.cpu_cores;
    const insts = M.instances;
    if (!sel || !insts.find((i) => i.instance === sel)) sel = (insts.find((i) => /T8760/.test(i.instance)) || insts.at(-1)).instance;
    const I = insts.find((i) => i.instance === sel);
    const fp = M.pairs.filter((p) => p.precision === "fp64" && p.ratio_vs_best != null);
    const best = fp.reduce((a, p) => (!a || p.ratio_vs_best > a.ratio_vs_best ? p : a), null);
    const ref = fp.find((p) => /T8760/.test(p.instance));
    const allRuns = insts.flatMap((i) => i.runs);
    const verified = allRuns.filter((r) => r.verify === "PASS").length, optimal = allRuns.filter((r) => r.status === "Optimal").length;
    const net = M.netlib;
    const netOk = net ? net.groups.reduce((a, g) => a + g.verified, 0) : 0, netTot = net ? net.groups.reduce((a, g) => a + g.total, 0) : 0;
    const L = M.logs || {};

    const kpis = h("div.kpis",
      best ? h("div.kpi.hl.big.span2", h("div.k", "best GPU speed-up vs fastest CPU"), h("div.v", fratio(best.ratio_vs_best)), h("div.s", `${best.instance} · fp64 · ${fsec(best.gpu_s)} vs ${fsec(best.best_cpu_s)}`)) : null,
      ref ? h("div.kpi.hl", h("div.k", "refinery year · hourly"), h("div.v", fratio(ref.ratio_vs_best)), h("div.s", `GPU ${fsec(ref.gpu_s)} · best CPU ${fsec(ref.best_cpu_s)} (${ref.best_cpu_threads} thr)`)) : null,
      h("div.kpi", h("div.k", "runs Optimal"), h("div.v", `${optimal}/${allRuns.length}`), h("div.s", `${verified} verify.py PASS · ${allRuns.filter((r) => r.verify === "skipped").length} not re-read (above verify.py's size limit)`)),
      net ? h("div.kpi", h("div.k", "small Netlib on GPU"), h("div.v", `${netOk}/${netTot}`), h("div.s", "verified PASS (r²HPDHG + PDLP, fp64 + mixed)")) : null,
      h(`div.kpi.${M.sanitizer.memcheck === "clean" && M.sanitizer.racecheck === "clean" ? "okc" : ""}`, h("div.k", "compute-sanitizer"), h("div.v", M.sanitizer.memcheck === "clean" && M.sanitizer.racecheck === "clean" ? "clean" : "see logs"), h("div.s", `memcheck ${M.sanitizer.memcheck} · racecheck ${M.sanitizer.racecheck}`)),
      L.ctest ? h("div.kpi", h("div.k", "tests on the GPU machine"), h("div.v", `${L.ctest.passed}/${L.ctest.total}`), h("div.s", "ctest, CUDA build")) : null);

    const machine = h("div.panel", h("div.panel-h", "GPU machine", h("span.right", srcChip(M.source))),
      h("dl.kv", { style: { padding: "12px 14px" } },
        h("dt", "GPU"), h("dd", M.gpu + (L.gpu_memory_mib ? ` · ${fint(L.gpu_memory_mib)} MiB` : "") + (L.sm_clock_mhz ? ` · ${fint(L.sm_clock_mhz)} MHz` : "")),
        h("dt", "driver · CUDA"), h("dd", `${M.driver} · ${M.cuda}`),
        h("dt", "OS"), h("dd", L.os || "—"),
        h("dt", "CPU baseline"), h("dd", `${M.cpu} · ${cores} threads (same machine)`),
        h("dt", "tolerance"), h("dd", `relative KKT ${M.tolerance}`),
        h("dt", "commit · date"), h("dd", `${M.source.git_hash} · ${M.source.date}`),
        h("dt", "logs"), h("dd", L.folder ? `${L.folder}/ (${(L.files || []).length} files)` : "—")));

    // speed-up across instances
    const speed = h("div.panel", h("div.panel-h", "Speed-up vs the fastest CPU configuration", h("span.right.dim", "log scale · 1× = same speed")),
      h("div.panel-b", barChart(M.pairs.map((p) => ({ label: `${p.instance} · ${p.precision}`, value: p.ratio_vs_best ?? 0,
        color: (p.ratio_vs_best ?? 0) >= 1 ? "#ffb547" : "#5f6d7c", text: p.ratio_vs_best != null ? fratio(p.ratio_vs_best) : p.note || "—" })),
        { log: true, refLine: 1, refLabel: "1×", labelW: 200 })),
      h("div.note", (() => {
        const w = M.pairs.filter((p) => (p.ratio_vs_best ?? 0) >= 1), l = M.pairs.filter((p) => p.ratio_vs_best != null && p.ratio_vs_best < 1);
        return `In this run the GPU is faster than the fastest CPU configuration in ${w.length} of ${w.length + l.length} comparisons`
          + (l.length ? `; slower on ${[...new Set(l.map((p) => p.instance))].join(", ")} (small models cannot fill the GPU and pay fixed launch / transfer costs).` : ".");
      })()));

    // one instance, every configuration
    const runs = [...I.runs].sort((a, b) => (a.backend === b.backend ? (a.threads || 0) - (b.threads || 0) : a.backend === "gpu" ? 1 : -1) || a.precision.localeCompare(b.precision));
    const instSeg = h("div.seg", { style: { flexWrap: "wrap" } }, insts.map((i) => h("button", { type: "button", class: i.instance === sel ? "on" : null, onclick: () => { sel = i.instance; draw(); } }, i.instance)));
    const detail = h("div.panel", { style: { gridColumn: "1 / -1" } },
      h("div.panel-h", `${I.instance} · ${fint(I.rows)} rows × ${fint(I.cols)} cols · ${fint(I.nnz)} nonzeros`, h("span.right.dim", I.known_optimum != null ? `known optimum ${fnum(I.known_optimum, 12)}` : "")),
      h("div.panel-b", instSeg),
      h("div.grid.g2", { style: { padding: "0 12px 12px", gap: "18px" } },
        h("div", h("div.field", h("span", "time to relative KKT 1e-8 (log scale)")),
          barChart(runs.map((r) => ({ label: cfgLabel(r, cores), value: t8(r) ?? 0, color: r.backend === "gpu" ? "#ffb547" : "#5ce1e6",
            text: t8(r) != null ? fsec(t8(r)) : r.status })), { log: true, labelW: 150 })),
        h("div", h("div.field", h("span", "iterations (same algorithm on every backend)")),
          barChart(runs.map((r) => ({ label: cfgLabel(r, cores), value: r.iterations, color: r.backend === "gpu" ? "#ffb547" : "#5ce1e6", text: fint(r.iterations) })), { labelW: 150 }))),
      h("table.t", h("thead", h("tr", h("th", "configuration"), h("th", "status"), h("th.num", "iterations"), h("th.num", "to 1e-4"), h("th.num", "to 1e-8"),
        h("th.num", "ms / it"), h("th.num", "objective"), h("th.num", "err vs known"), h("th", "verify.py"))),
        h("tbody", runs.map((r) => h("tr", h("td.mono", { style: { color: r.backend === "gpu" ? "var(--warn)" : null } }, cfgLabel(r, cores)), h("td", statusBadge(r.status)),
          h("td.num", fint(r.iterations)), h("td.num", fsec(r.s_1e4)), h("td.num", t8(r) != null ? fsec(t8(r)) : "—"), h("td.num", fnum(r.ms_per_iteration, 4)),
          h("td.num", fnum(r.objective, 12)), h("td.num", fexp(r.rel_err_known)),
          h("td", r.verify === "PASS" ? passBadge("PASS") : h("span.dim.mono", { title: "verify.py skips models above its size limit; checked by the gate and the known optimum" }, r.verify || "—")))))),
      h("div.note", "fp64 on the GPU runs the same iterations as fp64 on the CPU (same arithmetic, same algorithm); mixed precision may take a different number. 'err vs known' is the distance to the optimum known by construction."));

    // fp64 vs mixed on the GPU
    const mixRows = insts.map((i) => {
      const g64 = i.runs.find((r) => r.backend === "gpu" && r.precision === "fp64"), gmx = i.runs.find((r) => r.backend === "gpu" && r.precision === "mixed");
      return g64 && gmx ? { i, g64, gmx } : null;
    }).filter(Boolean);
    const mixed = h("div.panel", h("div.panel-h", "fp64 vs mixed precision on the GPU"),
      h("div", { style: { overflowX: "auto" } }, h("table.t", h("thead", h("tr", h("th", "instance"), h("th.num", "fp64 s"), h("th.num", "mixed s"), h("th.num", "mixed / fp64"), h("th.num", "it fp64"), h("th.num", "it mixed"), h("th.num", "err fp64"), h("th.num", "err mixed"))),
        h("tbody", mixRows.map(({ i, g64, gmx }) => {
          const a = t8(g64), b = t8(gmx), q = a && b ? b / a : null;
          return h("tr", h("td.mono", i.instance), h("td.num", fsec(a)), h("td.num", fsec(b)), h("td.num", { class: q != null && q < 1 ? "ok" : "" }, q != null ? `${q.toFixed(2)}×` : "—"),
            h("td.num", fint(g64.iterations)), h("td.num", fint(gmx.iterations)), h("td.num", fexp(g64.rel_err_known)), h("td.num", fexp(gmx.rel_err_known)));
        })))),
      h("div.note", (() => {
        const q = mixRows.map(({ g64, gmx }) => (t8(g64) && t8(gmx) ? t8(gmx) / t8(g64) : null)).filter((x) => x != null);
        const faster = q.filter((x) => x < 1).length;
        return `Mixed precision keeps the fp64 accuracy target (the same 1e-8 test, fp64 residuals and restarts). In this run it was faster on ${faster} of ${q.length} models`
          + (q.length ? ` (mixed / fp64 from ${Math.min(...q).toFixed(2)}× to ${Math.max(...q).toFixed(2)}×).` : ".")
          + " Its expected benefit is larger on GPUs with weak fp64 throughput; that is not measured here.";
      })()));

    const netP = net ? h("div.panel", h("div.panel-h", "Small Netlib on the GPU", h("span.right", srcChip(net.source))),
      h("table.t", h("thead", h("tr", h("th", "engine"), h("th", "precision"), h("th.num", "verified"), h("th.num", "total s"))),
        h("tbody", net.groups.map((g) => h("tr", h("td.mono", g.engine), h("td.mono", g.precision), h("td.num", `${g.verified}/${g.total}`), h("td.num", fnum(g.seconds, 4)))))),
      h("div.note", `${net.instances.length} models: ${net.instances.join(", ")}.`)) : null;

    body.replaceChildren(...[live(), kpis,
      data.machines.length > 1 ? h("div.seg", data.machines.map((m, k) => h("button", { class: k === mi ? "on" : null, onclick: () => { mi = k; sel = null; draw(); } }, m.machine))) : null,
      h("div.grid.g2", machine, speed), detail, h("div.grid.g2", mixed, netP)].filter(Boolean));
  }

  api.gpu().then((d) => { data = d; draw(); }).catch((e) => toast(e.message));
  const off = store.on((p) => { if ("system" in p) draw(); });
  draw();
  return off;
}
