// pages/bench.js — benchmark evidence, straight from the committed CSVs (bench/results/*.csv).
import { store } from "../store.js";
import { h, fnum, fsec, fratio, fint, fexp, srcChip, statusBadge, passBadge } from "../util.js";
import { barChart } from "../charts.js";

export function mount(root) {
  const body = h("div.grid", { style: { gap: "14px" } });
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "05 · benchmarks"), h("h1", "Benchmark evidence"),
      h("p", "Read live from the committed, hash-named CSVs in bench/results/ with the same rules as docs/EVIDENCE.md. Every table names its source; nothing here is typed in by hand. Reproduce: scripts/reproduce.sh (CPU) and scripts/gpu_check.sh (GPU)."))),
    body);

  function draw() {
    const ev = store.evidence;
    if (!ev) { body.replaceChildren(h("div.panel", h("div.empty", "reading evidence…"))); return; }
    const panels = [];

    // ---- Netlib per engine
    if (ev.netlib.length) {
      panels.push(h("div.panel", h("div.panel-h", "Netlib LP · all 93 models · 60 s each", h("span.right", srcChip(ev.netlib[0].source))),
        h("div.panel-b", barChart(ev.netlib.map((n) => ({ label: n.engine === "r2hpdhg" ? "r²HPDHG" : n.engine, value: n.solved,
          text: `${n.solved}/${n.total}`, color: n.engine === "auto" ? "#3ddc97" : "#5ce1e6" })), { labelW: 90 })),
        h("table.t", h("thead", h("tr", h("th", "engine"), h("th.num", "solved"), h("th.num", "total s (solved)"), h("th", "not solved"))),
          h("tbody", ev.netlib.map((n) => h("tr", h("td.mono", n.engine), h("td.num", `${n.solved}/${n.total}`), h("td.num", fnum(n.seconds_solved)),
            h("td.dim", { style: { whiteSpace: "normal" } }, n.not_solved.join(", ") || "—"))))),
        h("div.note", "Solved = engine Optimal + in-process gate + tools/verify.py PASS + |obj − HiGHS| / (1 + |HiGHS|) ≤ 1e-6. auto = simplex for rows·nnz ≤ 2·10⁸, else r²HPDHG (threshold chosen on this set).")));
    }

    // ---- infeasibility certificates
    if (ev.infeasible.length) {
      panels.push(h("div.panel", h("div.panel-h", "Certified infeasibility · Netlib + objective cut", h("span.right", srcChip(ev.infeasible[0].source))),
        h("table.t", h("thead", h("tr", h("th", "engine"), h("th.num", "certified"), h("th.num", "exact rational"), h("th.num", "rounding-proof (C++)"))),
          h("tbody", ev.infeasible.map((x) => h("tr", h("td.mono", x.engine), h("td.num", `${x.certified}/${x.total}`), h("td.num", x.exact_rational), h("td.num", x.rounding_proof))))),
        h("div.note", "Each Netlib LP gets one extra row cᵀx ≤ f* − δ (δ = 1e-4 (1+|f*|)): infeasible by duality, and its proof needs the optimal dual. Counted only if the Farkas certificate passes the gate AND verify.py. Everything else is a time limit — never a wrong verdict.")));
    }

    // ---- GPU vs CPU
    for (const g of ev.gpu || []) {
      const fp = g.pairs.filter((p) => p.precision === "fp64");
      panels.push(h("div.panel", { style: { gridColumn: "1 / -1" } },
        h("div.panel-h", `GPU vs CPU · ${g.gpu}`, h("span.right", srcChip(g.source))),
        h("div.grid.g2", { style: { padding: "12px", gap: "18px" } },
          h("div", h("div.field", h("span", "speed-up vs the fastest CPU configuration (fp64, log scale)")),
            barChart(fp.map((p) => ({ label: p.instance, value: p.ratio_vs_best ?? 0, color: (p.ratio_vs_best ?? 0) >= 1 ? "#ffb547" : "#5f6d7c",
              text: p.ratio_vs_best != null ? fratio(p.ratio_vs_best) : p.note || "—" })), { log: true, refLine: 1, refLabel: "1× = same speed", labelW: 150 })),
          h("dl.kv",
            h("dt", "GPU"), h("dd", `${g.gpu} · driver ${g.driver} · CUDA ${g.cuda}`),
            h("dt", "CPU baseline"), h("dd", `${g.cpu} · ${g.cpu_cores || "?"} cores (1 thread and all cores measured)`),
            h("dt", "memcheck"), h("dd", h(`span.badge.${g.sanitizer.memcheck === "clean" ? "ok" : "warn"}`, g.sanitizer.memcheck)),
            h("dt", "racecheck"), h("dd", h(`span.badge.${g.sanitizer.racecheck === "clean" ? "ok" : "warn"}`, g.sanitizer.racecheck)),
            h("dt", "reading"), h("dd", { style: { fontFamily: "var(--sans)", color: "var(--text-2)" } },
              "ratio = CPU seconds / GPU seconds to 1e-8; < 1× means the GPU is slower (small models). fp64 GPU iteration counts equal the CPU's."))),
        h("table.t", h("thead", h("tr", h("th", "instance"), h("th.num", "rows"), h("th.num", "nnz"), h("th", "prec"), h("th", "GPU"),
          h("th.num", "GPU s"), h("th.num", "CPU 1 thr s"), h("th.num", "best CPU s"), h("th.num", "vs best CPU"), h("th.num", "vs 1 thread"))),
          h("tbody", g.pairs.map((p) => h("tr", h("td.mono", p.instance), h("td.num", fint(p.rows)), h("td.num", fint(p.nnz)), h("td.mono", p.precision),
            h("td", statusBadge(p.gpu_status)), h("td.num", fsec(p.gpu_s)), h("td.num", p.cpu1_s != null ? fsec(p.cpu1_s) : "limit"),
            h("td.num", p.best_cpu_s != null ? `${fsec(p.best_cpu_s)} (${p.best_cpu_threads} thr)` : "—"),
            h("td.num", { style: { color: (p.ratio_vs_best ?? 0) >= 1 ? "var(--warn)" : "var(--text-3)" } }, fratio(p.ratio_vs_best)), h("td.num.dim", fratio(p.ratio_vs_1thread)))))),
        h("div.note", "One consumer laptop GPU (RTX 4050 6 GB, WSL2). compute-sanitizer ran on the Gpu.* unit tests. 1e6-row rows: verify.py is skipped above 3M nonzeros; they are checked by the gate and against the known optimum.")));
    }

    // ---- CPU scaling
    if (ev.scale_cpu) {
      const rows = ev.scale_cpu.rows;
      panels.push(h("div.panel", { style: { gridColumn: "1 / -1" } }, h("div.panel-h", "Large LPs on the CPU · r²HPDHG to relative KKT 1e-8", h("span.right", srcChip(ev.scale_cpu.source))),
        h("div.scroll", { style: { maxHeight: "420px" } }, h("table.t", h("thead", h("tr", h("th", "instance"), h("th.num", "rows"), h("th.num", "nnz"), h("th.num", "threads"), h("th", "prec"),
          h("th", "status"), h("th.num", "iterations"), h("th.num", "to 1e-4"), h("th.num", "to 1e-8"), h("th.num", "err vs known opt"), h("th", "verify"))),
          h("tbody", rows.map((r) => h("tr", h("td.mono", r.instance), h("td.num", fint(r.rows)), h("td.num", fint(r.nnz)), h("td.num", r.threads), h("td.mono", r.precision),
            h("td", statusBadge(r.status)), h("td.num", fint(+r.iterations)), h("td.num", fsec(r.s_1e4)), h("td.num", fsec(r.s_1e8)), h("td.num", fexp(r.rel_err_known)),
            h("td", r.verify === "PASS" ? passBadge("PASS") : h("span.dim.mono", r.verify || "—"))))))),
        h("div.note", "Generated LPs with a known optimum (refinery structure, synthetic data; random sparse). Wall times on the fanless MacBook Air vary run to run; iteration counts are deterministic.")));
    }

    // ---- MIPLIB
    if (ev.miplib) {
      panels.push(h("div.panel", { style: { gridColumn: "1 / -1" } }, h("div.panel-h", `MILP · small MIPLIB 3 · ${ev.miplib.solved}/${ev.miplib.total} proven optimal`, h("span.right", srcChip(ev.miplib.source))),
        h("table.t", h("thead", h("tr", h("th", "instance"), h("th", "status"), h("th.num", "objective"), h("th.num", "HiGHS"), h("th.num", "gap"), h("th.num", "time"), h("th", "verify"))),
          h("tbody", ev.miplib.instances.map((r) => h("tr", h("td.mono", r.name), h("td", statusBadge(r.status)), h("td.num", fnum(r.objective, 10)),
            h("td.num.dim", fnum(r.highs, 10)), h("td.num", r.gap ? `${(100 * r.gap).toFixed(2)}%` : "0"), h("td.num", fsec(r.seconds)),
            h("td", r.verify === "PASS" ? passBadge("PASS") : h("span.dim.mono", r.verify || "—"))))))));
    }
    body.replaceChildren(h("div.grid.g2", panels));
  }

  const off = store.on((p) => { if ("evidence" in p) draw(); });
  draw();
  return off;
}
