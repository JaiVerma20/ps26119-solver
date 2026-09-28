// pages/scenarios.js — planner what-ifs on the refinery planning LP: change prices, crude
// availability, demand or unit capacities; re-solve warm from the base case; sweep one lever through
// K values in one batched solve. Every number is the solver's output, every answer re-verified.
import { api } from "../api.js";
import { h, fnum, fexp, fint, fsec, statusBadge, passBadge, toast } from "../util.js";
import { lineChart, barChart, heatStrip } from "../charts.js";

// the page keeps its run across navigations (module state)
const S = { meta: null, cfg: null, run: null, listeners: new Set() };
const notify = () => S.listeners.forEach((f) => f());

function defaults(meta) {
  const saved = JSON.parse(sessionStorage.getItem("ps.scen") || "null");
  const base = meta.models.find((m) => m.periods === 365) || meta.models[0];
  return saved && meta.models.some((m) => m.path === saved.base) ? saved : {
    base: base?.path, mode: "whatif", threads: 0, levers: { price: 5, crude: 0, demand: 0, cdu: -10, fcc: 0 },
    sweep: { lever: "price", from: -10, to: 10, steps: 5 }, sequential: false,
  };
}

function start(cfg) {
  const body = { ...cfg, levers: Object.fromEntries(Object.entries(cfg.levers).filter(([k, v]) => !(cfg.mode === "sweep" && k === cfg.sweep.lever) && v)) };
  return fetch("/api/scenario", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then(async (r) => { const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j.job; })
    .then((job) => {
      const run = S.run = { job, cfg: structuredClone(cfg), stage: "base", done: false, commands: [], progress: {}, solved: {}, verified: {},
        points: [], seq: [], base: null, built: null, batch: null, sequential: null, summary: null, error: null };
      notify();
      api.stream(job, (ev) => {
        if (S.run !== run) return;
        if (ev.type === "stage") run.stage = ev.stage;
        else if (ev.type === "command") run.commands.push(ev);
        else if (ev.type === "progress") (run.progress[ev.phase] ||= []).push(ev);
        else if (ev.type === "base") run.base = ev;
        else if (ev.type === "built") run.built = ev;
        else if (ev.type === "solved") run.solved[ev.phase] = ev;
        else if (ev.type === "verified") run.verified[ev.phase] = ev;
        else if (ev.type === "point") run.points[ev.index] = ev;
        else if (ev.type === "sequential_point") run.seq[ev.index] = ev;
        else if (ev.type === "batch") run.batch = ev;
        else if (ev.type === "sequential") run.sequential = ev;
        else if (ev.type === "summary") run.summary = ev;
        else if (ev.type === "error") { run.error = ev.message; toast(ev.message); }
        else if (ev.type === "end" || ev.type === "stream-error") { run.done = true; run.stage = "done"; }
        notify();
      });
    });
}

export function mount(root) {
  const cfgBox = h("div.panel"), out = h("div.grid", { style: { gap: "14px", alignContent: "start" } });
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "07 · scenarios"), h("h1", "What-if planning"),
      h("p", "Change prices, crude availability, demand or unit capacity on the multi-period refinery LP and see the profit impact, the new bottlenecks and what it costs to re-solve. What-if: the scenario is solved cold and warm-started from the base plan. Sweep: K values of one lever are solved together in one batched pass. Each answer goes back through tools/verify.py."))),
    h("div.grid.g-side", cfgBox, out));

  let raf = 0;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; drawOut(); drawButtons(); }); };
  S.listeners.add(schedule);

  // ------------------------------------------------------------------ configuration
  let runBtn, cancelBtn;
  function drawButtons() {
    if (!runBtn) return;
    const busy = S.run && !S.run.done;
    runBtn.style.display = busy ? "none" : ""; cancelBtn.style.display = busy ? "" : "none";
  }
  function drawCfg() {
    const meta = S.meta, c = S.cfg;
    if (!meta) { cfgBox.replaceChildren(h("div.panel-b.muted", "loading…")); return; }
    const save = () => sessionStorage.setItem("ps.scen", JSON.stringify(c));
    if (!meta.models.length) {
      cfgBox.replaceChildren(h("div.panel-h", "Base plan"), h("div.panel-b",
        h("p.muted", "No refinery model yet. They are generated (bench/generate_refinery_lp.py, a few seconds) with an optimum known by construction."),
        h("div", { style: { display: "grid", gap: "8px" } }, [[365, "daily year · 17,885 rows"], [2190, "4-hourly year · 107,310 rows"], [8760, "hourly year · 429,240 rows"]].map(([T, l]) =>
          h("button.btn", { onclick: async (e) => { e.target.disabled = true; e.target.textContent = "generating…";
            try { await api.generate("refinery", T, 1); await load(); } catch (err) { toast(err.message); } } }, `Generate T${T} · ${l}`)))));
      return;
    }
    const seg = (items, get, set) => h("div.seg", items.map(([v, l]) => h("button", { type: "button", class: get() === v ? "on" : null,
      onclick: () => { set(v); save(); drawCfg(); } }, l)));
    const sweepMode = c.mode === "sweep";
    const baseSel = h("select.input", { onchange: (e) => { c.base = e.target.value; save(); drawCfg(); } },
      meta.models.map((m) => h("option", { value: m.path, selected: m.path === c.base || null }, `refinery T${m.periods} · ${m.periods === 8760 ? "hourly" : m.periods === 365 ? "daily" : m.periods === 12 ? "monthly" : `${m.periods} periods`}`)));
    const periods = meta.models.find((m) => m.path === c.base)?.periods;
    const lever = (k) => {
      const L = meta.levers[k], swept = sweepMode && c.sweep.lever === k;
      const val = h("span.mono", { style: { minWidth: "52px", textAlign: "right", color: c.levers[k] ? "var(--accent)" : "var(--text-3)" } }, swept ? "swept" : `${c.levers[k] > 0 ? "+" : ""}${c.levers[k]}%`);
      return h("div.field", { title: L.what }, h("span", L.label),
        h("div", { style: { display: "flex", gap: "10px", alignItems: "center" } },
          h("input", { type: "range", min: L.range[0], max: L.range[1], step: 1, value: c.levers[k], disabled: swept || null, style: { flex: 1, accentColor: "#5ce1e6" },
            oninput: (e) => { c.levers[k] = +e.target.value; val.textContent = `${c.levers[k] > 0 ? "+" : ""}${c.levers[k]}%`; val.style.color = c.levers[k] ? "var(--accent)" : "var(--text-3)"; save(); } }),
          val));
    };
    const numIn = (obj, key, attrs) => h("input.input.mono", { type: "number", value: obj[key], ...attrs, oninput: (e) => { obj[key] = +e.target.value; save(); } });
    runBtn = h("button.btn.primary", { style: { width: "100%", marginTop: "6px" }, onclick: () => start(c).catch((e) => toast(e.message)) }, sweepMode ? "Run sweep ▸" : "Run what-if ▸");
    cancelBtn = h("button.btn.danger", { style: { width: "100%", marginTop: "6px", display: "none" }, onclick: () => S.run && api.cancel(S.run.job) }, "Cancel");
    cfgBox.replaceChildren(h("div.panel-h", "Scenario"), h("div.panel-b",
      h("label.field", h("span", "Base plan"), baseSel),
      h("label.field", h("span", "Mode"), seg([["whatif", "what-if · warm start"], ["sweep", "sweep · batch"]], () => c.mode, (v) => { c.mode = v; })),
      sweepMode && periods > 2190 ? h("div.help.warn", "sweep needs a base with ≤ 2190 periods") : null,
      sweepMode ? h("div", { style: { border: "1px solid var(--line-2)", borderRadius: "6px", padding: "10px 10px 0", marginBottom: "12px" } },
        h("label.field", h("span", "Sweep lever"), h("select.input", { onchange: (e) => { c.sweep.lever = e.target.value; save(); drawCfg(); } },
          Object.entries(meta.levers).map(([k, L]) => h("option", { value: k, selected: k === c.sweep.lever || null }, L.label)))),
        h("div.row", h("label.field", h("span", "from %"), numIn(c.sweep, "from", { min: -50, max: 50 })),
          h("label.field", h("span", "to %"), numIn(c.sweep, "to", { min: -50, max: 50 })),
          h("label.field", h("span", "steps"), numIn(c.sweep, "steps", { min: 2, max: meta.max_sweep })))) : null,
      Object.keys(meta.levers).map(lever),
      h("label.field", h("span", "CPU threads"), seg([[1, "1 thread"], [0, "all cores"]], () => c.threads, (v) => { c.threads = v; })),
      sweepMode ? h("label.check", h("input", { type: "checkbox", checked: c.sequential || null, onchange: (e) => { c.sequential = e.target.checked; save(); } }), "also solve one by one (timing comparison)") : null,
      runBtn, cancelBtn,
      h("div.note", { style: { padding: "12px 0 0" } }, "Levers scale every period uniformly. Structure is the refinery's (crudes, CDU, FCC, blending with quality specs, tanks, demand); the prices are synthetic.")));
    drawButtons();
  }

  // ------------------------------------------------------------------ results
  const kpi = (k, v, s, cls = "") => h(`div.kpi${cls ? "." + cls.split(" ").join(".") : ""}`, h("div.k", k), h("div.v", v), s ? h("div.s", s) : null);
  const verdict = (ph) => { const v = S.run?.verified[ph]; return v ? passBadge(v.verdict) : "—"; };
  const kktSeries = (arr) => (arr || []).filter((p) => p.kind === "pdhg").map((p) => [p.iter, Math.max(p.rp ?? 0, p.rd ?? 0, Math.abs(p.gap ?? 0))]);
  const pipeline = (stages) => h("div.pipeline", { style: { gridTemplateColumns: `repeat(${stages.length}, minmax(0, 1fr))` } }, stages.map(([key, label, sub]) => {
    const r = S.run, order = stages.map((s) => s[0]), cur = order.indexOf(r.stage), i = order.indexOf(key);
    const state = r.error && i === cur ? "bad" : r.done || i < cur ? "ok" : i === cur ? "run" : "";
    const [n, l] = label.split(" · ");
    return h(`div.stage${state ? "." + state : ""}`, h("div.n", `stage ${n}`), h("div.l", l), h("div.s", sub || "—"));
  }));

  function marginalPanel(r) {
    const mb = r.base?.marginal, ms = r.solved.warm?.marginal;
    if (!mb?.cdu) return null;
    const strips = [{ label: "CDU · base", values: mb.cdu }, ...(ms?.cdu ? [{ label: "CDU · scenario", values: ms.cdu }] : []),
      { label: "FCC · base", values: mb.fcc }, ...(ms?.fcc ? [{ label: "FCC · scenario", values: ms.fcc }] : [])];
    const hs = heatStrip(strips, { maxCells: 365 });
    const vmax = +hs.dataset.vmax, bin = +hs.dataset.bin;
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const bound = (a) => a.filter((v) => Math.abs(v) > 1e-9).length;
    const dem = Object.entries(mb.demand).map(([p, a]) => ({ label: p, value: avg(a), text: `${fnum(avg(a), 3)} · ${bound(a)}/${a.length}`, color: "#3ddc97" }));
    return h("div.grid.g2",
      h("div.panel", h("div.panel-h", "Where capacity binds · marginal value per period", h("span.right.dim", "row duals y · profit per extra unit")),
        h("div.panel-b", hs,
          h("div.legend", { style: { padding: "8px 0 0" } }, h("span", h("i", { style: { background: "linear-gradient(90deg,#0b0f14,#5ce1e6)", width: "60px", height: "8px" } }), `0 … ${fnum(vmax, 3)} profit / unit`),
            h("span.dim", `x: period${bin > 1 ? ` · each cell = ${bin} periods (their maximum)` : ""} · dark = spare capacity`))),
        h("div.note", `A lit cell means the unit is a bottleneck in that period: one more unit of capacity adds that much profit (brighter = more). Limiting in (base): CDU ${bound(mb.cdu)}/${mb.cdu.length}, FCC ${bound(mb.fcc)}/${mb.fcc.length} periods. Checked on refinery-T12: for these rows the dual equals the finite-difference profit change Δprofit / Δcapacity.`)),
      h("div.panel", h("div.panel-h", "Marginal value of demand (base plan)", h("span.right.dim", "average · periods limiting")),
        h("div.panel-b", barChart(dem, { labelW: 80 })),
        h("div.note", "Profit from one more unit of demand for each product, averaged over periods; 'limiting' counts the periods where the demand cap has a nonzero value (the plan would sell more if it could).")));
  }

  function drawWhatif(r) {
    const b = r.base, c = r.solved.cold, w = r.solved.warm, sm = r.summary;
    const levers = Object.entries(r.cfg.levers).filter(([, v]) => v).map(([k, v]) => `${S.meta.levers[k].label} ${v > 0 ? "+" : ""}${v}%`).join(" · ") || "no change (base plan)";
    const dCls = sm?.delta == null ? "" : sm.delta >= 0 ? "okc" : "badc";
    const res = [{ name: "cold start", color: "#ffb547", points: kktSeries(r.progress.cold) }, { name: "warm start (from the base plan)", color: "#3ddc97", points: kktSeries(r.progress.warm) }];
    return [
      pipeline([["base", "1 · base plan", b ? (b.cached ? "reused" : fsec(b.seconds)) : ""], ["build", "2 · apply levers", r.built ? fsec(r.built.seconds) : ""],
        ["cold", "3 · cold solve", c ? `${fint(c.iterations)} it` : ""], ["warm", "4 · warm solve", w ? `${fint(w.iterations)} it` : ""],
        ["verify", "5 · verify both", r.verified.warm ? r.verified.warm.verdict : ""]]),
      h("div.panel", h("div.panel-b", { style: { padding: "10px 14px" } }, h("span.dim", "scenario: "), h("span.mono", levers))),
      h("div.kpis",
        kpi("Base profit", fnum(b?.objective, 10), b?.known ? `known optimum ${fnum(b.known.optimum, 10)}` : "", "hl"),
        kpi("Scenario profit", fnum(w?.objective, 10), w ? statusBadge(w.status) : "", "hl"),
        kpi("Δ profit", sm?.delta != null ? `${sm.delta >= 0 ? "+" : ""}${fnum(sm.delta, 6)}` : "—", sm?.delta_pct != null ? `${sm.delta_pct >= 0 ? "+" : ""}${sm.delta_pct.toFixed(3)} %` : "", `big ${dCls}`),
        kpi("Cold re-solve", c ? `${fint(c.iterations)} it` : "—", c ? fsec(c.seconds) : ""),
        kpi("Warm re-solve", w ? `${fint(w.iterations)} it` : "—", w ? fsec(w.seconds) : ""),
        kpi("Warm / cold", sm?.iteration_ratio != null ? `${sm.iteration_ratio.toFixed(2)}×` : "—", sm?.time_ratio != null ? `time ${sm.time_ratio.toFixed(2)}×` : "iterations", sm?.iteration_ratio < 1 ? "okc" : ""),
        kpi("Cold vs warm", sm?.agree != null ? fexp(sm.agree) : "—", "relative objective difference"),
        kpi("Verification", h("span", verdict("cold"), " ", verdict("warm")), "cold · warm (verify.py)", r.verified.warm?.verdict === "PASS" && r.verified.cold?.verdict === "PASS" ? "okc" : "")),
      h("div.panel", h("div.panel-h", "Re-solve convergence: cold vs warm", h("span.right.dim", "relative KKT error, max of primal / dual / gap")),
        h("div.panel-b.tight", lineChart(res, { logY: true, threshold: 1e-8, thresholdLabel: "1e-8 target", xLabel: "iteration", empty: r.done ? "no first-order log" : "waiting for the solver…" })),
        h("div.legend", res.map((s) => h("span", h("i", { style: { background: s.color } }), s.name))),
        h("div.note", "The warm start begins from the base plan's primal and dual solution (--warm); the cold start from zero. Same model, same tolerance; both answers are checked independently and must agree.")),
      marginalPanel(r),
    ];
  }

  function drawSweep(r) {
    const L = S.meta.levers[r.cfg.sweep.lever], b = r.base, pts = r.points.filter(Boolean);
    const allPass = pts.length && pts.every((p, i) => r.verified[`s${i}`]?.verdict === "PASS");
    const seqT = r.sequential?.seconds, batT = r.batch?.seconds;
    const dis = r.seq.length ? Math.max(...r.seq.map((q, i) => (q && pts[i]?.objective != null ? Math.abs(q.objective - pts[i].objective) / (1 + Math.abs(q.objective)) : 0))) : null;
    const running = r.progress.batch?.at(-1);
    return [
      pipeline([["base", "1 · base plan", b ? (b.cached ? "reused" : fsec(b.seconds)) : ""], ["build", "2 · build K models", r.built ? `${r.built.count} · ${fsec(r.built.seconds)}` : ""],
        ["batch", "3 · one batched solve", running && !r.batch ? `${running.running}/${running.total} running` : r.batch ? fsec(batT) : ""], ["verify", "4 · verify each", pts.length ? `${Object.keys(r.verified).length}/${pts.length}` : ""],
        ...(r.cfg.sequential ? [["sequential", "5 · one by one", seqT != null ? fsec(seqT) : r.seq.length ? `${r.seq.filter(Boolean).length}/${pts.length}` : ""]] : [])]),
      h("div.panel", h("div.panel-b", { style: { padding: "10px 14px" } }, h("span.dim", "sweep: "), h("span.mono", `${L.label} ${r.cfg.sweep.from}% → ${r.cfg.sweep.to}% in ${r.cfg.sweep.steps} steps`),
        h("span.dim", " · held fixed: "), h("span.mono", Object.entries(r.cfg.levers).filter(([k, v]) => v && k !== r.cfg.sweep.lever).map(([k, v]) => `${S.meta.levers[k].label} ${v > 0 ? "+" : ""}${v}%`).join(" · ") || "nothing"))),
      h("div.kpis",
        kpi("Base profit", fnum(b?.objective, 10), "all levers at 0", "hl"),
        kpi("Scenarios", `${pts.length || "—"}`, `${L.label} ${r.cfg.sweep.from}% … ${r.cfg.sweep.to}%`),
        kpi("Profit range", pts.length ? `${fnum(Math.min(...pts.map((p) => p.objective)), 8)} … ${fnum(Math.max(...pts.map((p) => p.objective)), 8)}` : "—", "", "span2"),
        kpi("Batched solve", batT != null ? fsec(batT) : "—", r.batch ? `${fint(Math.max(...pts.map((p) => p.iterations)))} it · one SpMM per iteration` : ""),
        kpi("One by one", seqT != null ? fsec(seqT) : r.cfg.sequential ? "…" : "off", seqT != null && batT ? (batT <= seqT ? `batch ${(seqT / batT).toFixed(2)}× faster` : `batch ${(batT / seqT).toFixed(2)}× slower`) : "sum of K separate solves",
          seqT != null && batT ? (batT <= seqT ? "okc" : "") : ""),
        kpi("Batch vs one by one", dis != null && seqT != null ? fexp(dis) : "—", "max objective difference"),
        kpi("Verification", pts.length ? (allPass ? passBadge("PASS") : `${Object.values(r.verified).filter((v) => v.verdict === "PASS").length}/${pts.length}`) : "—", "each scenario (verify.py)", allPass ? "okc" : "")),
      h("div.panel", h("div.panel-h", `Profit vs ${L.label.toLowerCase()}`, h("span.right.dim", "each point: one verified optimal plan")),
        h("div.panel-b.tight", lineChart([{ name: "profit", color: "#5ce1e6", points: pts.map((p) => [p.value, p.objective]) }], { markers: true, xLabel: `${L.label} change, %`, empty: r.done ? "no points" : "solving the batch…" })),
        h("div.note", "The curve is piecewise linear: its slope changes where the optimal plan changes (a new bottleneck or a new crude slate).")),
      h("div.panel", h("div.panel-h", "Scenarios"),
        h("table.t", h("thead", h("tr", h("th.num", "change"), h("th", "status"), h("th.num", "profit"), h("th.num", "Δ vs base"), h("th.num", "iterations"), h("th", "verify"),
          r.cfg.sequential ? [h("th.num", "one-by-one s"), h("th.num", "one-by-one it")] : null)),
          h("tbody", pts.map((p, i) => h("tr", h("td.num", `${p.value > 0 ? "+" : ""}${p.value}%`), h("td", statusBadge(p.status)), h("td.num", fnum(p.objective, 10)),
            h("td.num", { class: b && p.objective >= b.objective ? "ok" : "bad" }, b ? `${p.objective - b.objective >= 0 ? "+" : ""}${fnum(p.objective - b.objective, 8)}` : "—"),
            h("td.num", fint(p.iterations)), h("td", verdict(`s${i}`)),
            r.cfg.sequential ? [h("td.num", r.seq[i] ? fsec(r.seq[i].seconds) : "—"), h("td.num", r.seq[i] ? fint(r.seq[i].iterations) : "—")] : null)))),
        h("div.note", "Batch and one-by-one both run r²HPDHG without presolve (the batch engine has none) at the same thread count and tolerance. The batch keeps every scenario in the SpMM until the slowest one converges, so on a CPU it is not always faster; it is built for the GPU, where one SpMM costs about as much as one SpMV.")),
      marginalPanel(r),
    ];
  }

  function drawOut() {
    const r = S.run;
    if (!r) {
      out.replaceChildren(h("div.panel", h("div.empty", "Set the levers and run. The base plan is solved once and reused; the scenario answer is compared with it.")));
      return;
    }
    const cmds = h("div.console", { style: { height: "auto", maxHeight: "140px", fontSize: "11px" } }, r.commands.map((c) => h("div", h("span.dim", `[${c.phase}] `), `$ ${c.command}`)));
    out.replaceChildren(...[...(r.cfg.mode === "sweep" ? drawSweep(r) : drawWhatif(r)),
      r.error ? h("div.panel", h("div.panel-b.bad", r.error)) : null,
      h("div.panel", h("div.panel-h", "Commands (exactly what ran)"), cmds)].filter(Boolean));
  }

  async function load() {
    S.meta = await api.scenarios();
    S.cfg = S.cfg && S.meta.models.some((m) => m.path === S.cfg.base) ? S.cfg : defaults(S.meta);
    drawCfg(); drawOut();
  }
  load().catch((e) => toast(e.message));
  drawCfg(); drawOut();
  return () => S.listeners.delete(schedule);
}
