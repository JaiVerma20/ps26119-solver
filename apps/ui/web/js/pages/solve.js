// pages/solve.js — configure a run, watch it live, read the verified result.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, fnum, fexp, fint, fsec, fratio, statusBadge, passBadge, toast } from "../util.js";
import { lineChart } from "../charts.js";
import { startSolve, certificateStatus, DEFAULT_OPTS } from "../run.js";

const ENGINES = [["auto", "auto"], ["simplex", "simplex"], ["r2hpdhg", "r²HPDHG"], ["pdlp", "PDLP"], ["oracle", "oracle"]];
const FIRST_ORDER = new Set(["auto", "r2hpdhg", "pdlp"]);
let opts = { ...DEFAULT_OPTS, ...JSON.parse(sessionStorage.getItem("ps.opts") || "{}") };

export function mount(root) {
  const catalog = { groups: [] };
  let info = null;
  const saveOpts = () => sessionStorage.setItem("ps.opts", JSON.stringify(opts));

  // ------------------------------------------------------------------ configuration panel
  const modelSel = h("select.input", { onchange: (e) => { store.set({ selectedModel: e.target.value }); loadInfo(); renderCmd(); } });
  const upload = h("input", { type: "file", accept: ".mps,.lpm", style: { display: "none" }, onchange: async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try { const p = await api.upload(f); await loadCatalog(); store.set({ selectedModel: p }); fillModels(); loadInfo(); renderCmd(); toast(`uploaded ${f.name}`); }
    catch (err) { toast(`upload failed: ${err.message}`); }
  } });
  const seg = (key, items, disabledFn) => {
    const el = h("div.seg");
    const draw = () => el.replaceChildren(...items.map(([v, label]) => h("button", {
      class: opts[key] === v ? "on" : null, disabled: disabledFn?.(v) || null, type: "button",
      onclick: () => { opts[key] = v; saveOpts(); draw(); drawAll(); } }, label)));
    el.draw = draw; draw();
    return el;
  };
  const engineSeg = seg("algorithm", ENGINES);
  const precSeg = seg("precision", [["fp64", "fp64"], ["mixed", "mixed fp32/64"]], () => !FIRST_ORDER.has(opts.algorithm));
  const gpuSeg = seg("gpu", [[false, "CPU"], [true, "GPU (CUDA)"]], (v) => v && !store.system?.cuda_build);
  const threadsSeg = seg("threads", [[1, "1 thread"], [0, "all cores"]]);
  const num = (key, attrs) => h("input.input.mono", { ...attrs, value: opts[key] ?? "", oninput: (e) => { opts[key] = e.target.value; saveOpts(); renderCmd(); } });
  const presolve = h("input", { type: "checkbox", checked: opts.presolve || null, onchange: (e) => { opts.presolve = e.target.checked; saveOpts(); renderCmd(); } });
  const verifyChk = h("input", { type: "checkbox", checked: opts.verify !== false || null, onchange: (e) => { opts.verify = e.target.checked; saveOpts(); } });
  const cmd = h("div.console", { style: { height: "auto", maxHeight: "120px", fontSize: "11px" } });
  const runBtn = h("button.btn.primary", { style: { width: "100%" }, onclick: run }, "Run solver ", h("kbd", "⌘↵"));
  const cancelBtn = h("button.btn.danger", { style: { width: "100%", display: "none" }, onclick: () => store.run && api.cancel(store.run.job) }, "Cancel");
  const gpuNote = h("div.help");

  const config = h("div.panel", h("div.panel-h", "Run configuration"), h("div.panel-b",
    h("label.field", h("span", "Model"), modelSel),
    h("div", { style: { display: "flex", gap: "8px", margin: "-4px 0 12px" } },
      h("a", { href: "#/models" }, "explore models"), h("span.dim", "·"),
      h("a", { href: "#", onclick: (e) => { e.preventDefault(); upload.click(); } }, "upload .mps / .lpm"), upload),
    h("label.field", h("span", "Engine"), engineSeg),
    h("label.field", h("span", "Backend"), gpuSeg), gpuNote,
    h("label.field", h("span", "Precision"), precSeg),
    h("label.field", h("span", "CPU threads"), threadsSeg),
    h("div.row", h("label.field", h("span", "Time limit s"), num("time_limit", { type: "number", min: 1 })),
      h("label.field", h("span", "Tolerance"), num("tol", { placeholder: "1e-8" }))),
    h("label.check", presolve, "presolve + postsolve"),
    h("label.check", verifyChk, "independent verification (tools/verify.py)"),
    runBtn, cancelBtn,
    h("div.field", { style: { marginTop: "14px" } }, h("span", "Command (exactly what runs)"), cmd),
  ));

  // ------------------------------------------------------------------ live area
  const pipeline = h("div.pipeline");
  const kpis = h("div.kpis");
  const chartRes = h("div"), chartObj = h("div");
  const resLegend = h("div.legend"), objLegend = h("div.legend");
  const consoleEl = h("div.console");
  const solTabs = h("div", { style: { display: "flex", gap: "6px" } });
  const solBody = h("div.scroll");
  let solTab = "vars";

  // ------------------------------------------------------------------ reference: the same model through HiGHS
  // (tools/highs_ref.py — a separate reference, never linked) and the same independent verifier
  const refPanel = h("div.panel");
  let ref = null, refSolver = "simplex";
  const REF_LABEL = { simplex: "dual simplex", ipm: "interior point", pdlp: "PDLP" };
  async function runReference() {
    const r = store.run;
    if (!r?.done || !r.result) return;
    ref = { path: r.path, ourJob: r.job, solver: refSolver, events: {}, done: false };
    drawRef();
    try {
      const job = await api.reference(r.path, refSolver, Math.max(60, +r.opts.time_limit || 60));
      api.stream(job, (ev) => {
        if (!ref || ref.ourJob !== r.job) return;
        if (ev.type === "end" || ev.type === "stream-error") ref.done = true; else ref.events[ev.type] = ev;
        drawRef();
      });
    } catch (e) { toast(e.message); ref = null; drawRef(); }
  }
  function drawRef() {
    const r = store.run && store.run.path === store.selectedModel ? store.run : null;
    const mine = r?.result?.solution;
    if (!r?.done || !mine) { refPanel.replaceChildren(); return; }
    const cur = ref && ref.ourJob === r.job ? ref : null;
    const res = cur?.events.result, v = cur?.events.verify, err = cur?.events.error;
    const seg = h("div.seg", { style: { width: "320px" } }, Object.entries(REF_LABEL).map(([k, l]) =>
      h("button", { class: refSolver === k ? "on" : null, disabled: cur && !cur.done || null, onclick: () => { refSolver = k; drawRef(); } }, l)));
    const btn = h("button.btn", { disabled: cur && !cur.done || null, onclick: runReference }, cur && !cur.done ? "running HiGHS…" : "Run HiGHS on this model ▸");
    const rel = res && Number.isFinite(res.objective) && Number.isFinite(mine.objective) ? Math.abs(res.objective - mine.objective) / (1 + Math.abs(mine.objective)) : null;
    const ourVerdict = r.verify?.report?.verdict, refVerdict = v?.report?.verdict;
    const row = (k, a, b, note = "") => h("tr", h("td", k), h("td.num", a), h("td.num", b), h("td.dim", note));
    refPanel.replaceChildren(
      h("div.panel-h", "Real-world reference · the same model through HiGHS", h("span.right.dim", "HiGHS runs as a separate tool; the same independent verifier judges both")),
      h("div.panel-b", { style: { display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" } }, seg, btn,
        res ? h("span.dim.mono", `${res.highs_version ? "HiGHS " + res.highs_version : ""} · ${res.highs_status}`) : null),
      err ? h("div.note.bad", err.message) : null,
      res ? h("table.t", h("thead", h("tr", h("th", ""), h("th.num", "ps26119"), h("th.num", `HiGHS ${REF_LABEL[cur.solver]}`), h("th", ""))),
        h("tbody",
          row("status", statusBadge(mine.status), statusBadge(res.status)),
          row("objective", fnum(mine.objective, 12), fnum(res.objective, 12), rel != null ? `relative difference ${fexp(rel)}` : ""),
          row("solve time", fsec(mine.seconds), fsec(res.seconds), mine.seconds > 0 && res.seconds > 0 ? (mine.seconds <= res.seconds ? `ours ${fratio(res.seconds / mine.seconds)} faster` : `HiGHS ${fratio(mine.seconds / res.seconds)} faster`) : ""),
          row("iterations", fint(mine.iterations), fint(res.iterations), `${mine.engine} vs ${res.engine}`),
          row("independent verify", ourVerdict ? passBadge(ourVerdict) : "—", refVerdict ? passBadge(refVerdict) : v?.skipped ? "n/a" : cur.done ? "—" : h("span.acc", "checking…"),
            refVerdict === "FAIL" ? (v.report.reasons || []).join("; ").slice(0, 120) : ""))) : null,
      h("div.note", "Times are the solve call only on both sides (model reading excluded). One live run on this machine — for a measured comparison over all of Netlib and the refinery models see the ‘vs HiGHS’ page."));
  }

  const area = h("div.grid", { style: { gap: "12px" } },
    pipeline, kpis,
    h("div.grid.g2",
      h("div.panel", h("div.panel-h", h("span#res-title", "Convergence"), h("span.right.dim#res-sub", "")), h("div.panel-b.tight", chartRes), resLegend),
      h("div.panel", h("div.panel-h", h("span#obj-title", "Objective"), h("span.right.dim#obj-sub", "")), h("div.panel-b.tight", chartObj), objLegend)),
    h("div.grid.g2",
      h("div.panel", h("div.panel-h", "Solver log", h("span.right.dim", "stderr, live (-vv)")), consoleEl),
      h("div.panel", h("div.panel-h", "Solution", h("span.right", solTabs)), solBody)),
    refPanel,
  );

  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "03 · solve"), h("h1", "Solve"),
      h("p", "Runs the real ps26119 binary. The run is checked twice: in-process on the original model, then by an independent verifier with its own MPS reader."))),
    h("div.grid.g-side", config, area));

  // ------------------------------------------------------------------ helpers
  async function loadCatalog() { catalog.groups = await api.models(); }
  function fillModels() {
    modelSel.replaceChildren(...catalog.groups.map((g) => h("optgroup", { label: g.group },
      ...g.models.map((m) => h("option", { value: m.path, selected: m.path === store.selectedModel || null }, m.name)))));
    if (!catalog.groups.some((g) => g.models.some((m) => m.path === store.selectedModel)) && catalog.groups[0])
      store.set({ selectedModel: modelSel.value });
  }
  async function loadInfo() {
    info = null; drawKpis();
    try { info = await api.modelInfo(store.selectedModel); } catch { info = null; }
    drawKpis(); drawPipeline();
  }
  function renderCmd() {
    const o = opts, parts = ["ps26119 solve", store.selectedModel, "--algorithm", o.algorithm, "--precision", o.precision,
      "--threads", o.threads, "--time-limit", o.time_limit];
    if (o.tol) parts.push("--tol", o.tol);
    if (!o.presolve) parts.push("--no-presolve");
    if (o.gpu) parts.push("--gpu");
    parts.push("--out solution.sol");
    const r = store.run;
    cmd.replaceChildren(h("span.cmd", "$ "), r && !r.done && r.command ? r.command : parts.join(" "));
  }

  async function run() {
    try {
      if (opts.gpu && !store.system?.cuda_build) throw new Error("this build has no CUDA backend");
      await startSolve(store.selectedModel, opts);
    } catch (e) { toast(e.message); }
  }

  // the first-order engines log objectives in their internal minimisation form: flip for MAX models
  const pdhgSign = () => (info?.sense === "maximize" ? -1 : 1);
  const kpi = (k, v, s, cls = "") => h(`div.kpi${cls ? "." + cls.trim().split(/\s+/).join(".") : ""}`, h("div.k", k), h("div.v", { title: typeof v === "string" ? v : null }, v), s ? h("div.s", s) : null);

  // dual objective, and for generated models the error against the optimum known by construction
  function objSub(sol, res, info) {
    const parts = [];
    if (sol?.dual_objective != null && Number.isFinite(sol.dual_objective)) parts.push(`dual ${fnum(sol.dual_objective, 10)}`);
    const known = res?.known || info?.known;
    if (known?.rel_err != null) parts.push(h("span", { class: known.rel_err <= 1e-6 ? "ok" : "warn", title: `known optimum ${known.optimum} (${known.generator}, by construction)` },
      `vs known optimum ${fexp(known.rel_err)}`));
    else if (known && !sol) parts.push(`known optimum ${fnum(known.optimum, 12)} (by construction)`);
    return parts.length ? parts.flatMap((p, i) => (i ? [" · ", p] : [p])) : "";
  }

  function drawKpis() {
    const r = store.run && store.run.path === store.selectedModel ? store.run : null;
    const res = r?.result, sol = res?.solution, last = r?.progress?.at(-1);
    const sgn = pdhgSign();
    const rows = res?.rows ?? info?.n_rows, cols = res?.cols ?? info?.n_cols, nnz = res?.nnz ?? info?.n_nnz;
    const v = r?.verify?.report, cert = certificateStatus(res, r?.verify);
    const running = r && !r.done;
    const status = sol?.status || (running ? "running" : "—");
    const iters = sol?.iterations ?? (last ? last.iter ?? last.node : null);
    const t = sol?.seconds ?? (last?.t ?? null);
    kpis.replaceChildren(
      kpi("Status", status === "running" ? h("span.acc", "running…") : statusBadge(sol?.status), sol?.message?.includes("auto:") ? sol.message.match(/auto: \w+/)[0] : ""),
      kpi("Objective", fnum(sol?.objective ?? (last?.pobj != null ? sgn * last.pobj : null) ?? last?.obj ?? last?.incumbent, 10), objSub(sol, res, info), "hl big span2"),
      kpi("Rows", fint(rows), info?.rows_split ? `E ${fint(info.rows_split[0])} · L ${fint(info.rows_split[1])} · G ${fint(info.rows_split[2])}` : ""),
      kpi("Columns", fint(cols), info?.columns_split ? `int ${fint(info.columns_split[1] + info.columns_split[2])}` : ""),
      kpi("Nonzeros", fint(nnz), info?.dynamism ? `dynamism ${info.dynamism.split(" ")[0]}` : ""),
      kpi("Engine", sol?.engine || (r ? r.opts.algorithm : opts.algorithm), r?.opts ? `requested ${r.opts.algorithm}` : ""),
      kpi("Backend", (r ? r.opts.gpu : opts.gpu) ? "GPU · CUDA" : "CPU", `${(r ? r.opts.threads : opts.threads) === 0 ? store.system?.cores + " threads" : "1 thread"}`),
      kpi("Precision", sol?.precision || (r ? r.opts.precision : opts.precision), last?.precision ? `now ${last.precision}` : ""),
      kpi("Iterations", fint(iters), sol?.iterations_to_fast >= 0 ? `to 1e-4: ${fint(sol.iterations_to_fast)}` : ""),
      kpi("Solve time", fsec(t), sol?.setup_seconds ? `setup ${fsec(sol.setup_seconds)}` : res ? `wall ${fsec(res.wall_seconds)}` : ""),
      kpi("Primal residual", fexp(sol?.primal_residual ?? last?.rp ?? last?.infeas), "engine · relative L2"),
      kpi("Dual residual", fexp(sol?.dual_residual ?? last?.rd), "engine · relative L2"),
      kpi("Gap", fexp(sol?.gap ?? last?.gap), "relative"),
      kpi("Verification", v ? passBadge(v.verdict) : r?.verify?.skipped ? h("span.dim", "n/a") : r?.stage === "verify" ? h("span.acc", "checking…") : "—",
        v ? `independent · ${fsec(r.verify.seconds)}` : r?.verify?.skipped ? r.verify.reason : "", v ? (v.verdict === "PASS" ? "okc" : "badc") : ""),
      kpi("Certificate", h(`span.badge.${cert.cls}`, cert.label), cert.detail ? cert.detail.slice(0, 60) : ""),
    );
  }

  function drawPipeline() {
    const r = store.run && store.run.path === store.selectedModel ? store.run : null;
    const res = r?.result, sol = res?.solution, v = r?.verify?.report;
    const pre = sol?.message?.match(/presolve removed (\d+) rows, (\d+) cols/);
    const running = r && !r.done && !res;
    const st = (cls, n, l, s) => h(`div.stage${cls ? "." + cls : ""}`, h("div.n", n), h("div.l", l), h("div.s", s || " "));
    const chk = res?.check_line || "";
    const cert = certificateStatus(res, r?.verify);
    pipeline.replaceChildren(
      st(info || res ? "ok" : "", "1 · model", res?.name || info?.name || store.selectedModel.split("/").pop(),
        `${fint(res?.rows ?? info?.n_rows)} × ${fint(res?.cols ?? info?.n_cols)} · ${fint(res?.nnz ?? info?.n_nnz)} nnz`),
      st(!r ? "" : running ? "run" : !r.opts.presolve ? "skip" : "ok", "2 · presolve", !r ? "safe reductions" : !r.opts.presolve ? "off" : pre ? "reduced" : res ? "no reduction" : "…",
        pre ? `−${fint(+pre[1])} rows · −${fint(+pre[2])} cols` : ""),
      st(!r ? "" : running ? "run" : sol?.status === "Optimal" || sol?.status === "Infeasible" || sol?.status === "Unbounded" ? "ok" : res ? "warn" : "bad",
        "3 · engine", sol?.engine || (r ? r.opts.algorithm : opts.algorithm), sol ? `${sol.status} · ${fint(sol.iterations)} it · ${fsec(sol.seconds)}` : running ? `${fint(r.progress.at(-1)?.iter)} it…` : ""),
      st(!res ? "" : chk.startsWith("PASS") ? "ok" : chk.startsWith("FAIL") ? "bad" : "skip", "4 · in-process gate",
        !res ? "original model" : chk.startsWith("PASS") ? "PASS" : chk.startsWith("FAIL") ? "FAIL" : "n/a", chk.replace(/^(PASS|FAIL)\s*/, "").replace(/[()]/g, "").slice(0, 60)),
      st(!res ? "" : cert.cls === "ok" ? "ok" : cert.cls === "warn" ? "warn" : "skip", "5 · certificate / bound", !res ? "rounding-proof" : cert.label, (cert.detail || "").slice(0, 60)),
      st(!r || r.opts.verify === false || r.verify?.skipped ? (r ? "skip" : "") : v ? (v.verdict === "PASS" ? "ok" : "bad") : r.stage === "verify" ? "run" : "",
        "6 · independent verify", v ? `verify.py ${v.verdict}` : r?.verify?.skipped ? "not applicable" : r?.opts?.verify === false ? "off" : "tools/verify.py", v ? `reader ${v.reader} · fingerprint ${v.model_match ? "match" : "MISMATCH"}` : ""),
    );
  }

  function drawCharts() {
    const r = store.run && store.run.path === store.selectedModel ? store.run : null;
    const P = r?.progress || [];
    const pd = P.filter((p) => p.kind === "pdhg"), sx = P.filter((p) => p.kind === "simplex"), bb = P.filter((p) => p.kind === "bb");
    const sgn = pdhgSign();
    const sol = r?.result?.solution;
    let res = [], obj = [], resTitle = "Convergence", objTitle = "Objective", resSub = "", objSub = "", thr = null;
    if (pd.length) {
      res = [{ name: "primal residual", color: "#5ce1e6", points: pd.map((p) => [p.iter, p.rp]) },
             { name: "dual residual", color: "#a78bfa", points: pd.map((p) => [p.iter, p.rd]) },
             { name: "gap", color: "#3ddc97", points: pd.map((p) => [p.iter, p.gap]) }];
      obj = [{ name: "primal objective", color: "#5ce1e6", points: pd.map((p) => [p.iter, sgn * p.pobj]) },
             { name: "dual objective", color: "#a78bfa", dash: "4 3", points: pd.map((p) => [p.iter, sgn * p.dobj]) }];
      resTitle = "Relative KKT residuals (log)"; resSub = `${pd[0].engine} · ${pd.length} checks logged`;
      objTitle = "Primal & dual objective"; objSub = "they meet at the optimum";
      thr = r?.opts?.tol ? +r.opts.tol : 1e-8;
    } else if (sx.length) {
      res = [{ name: "primal infeasibility (phase 1)", color: "#ffb547", points: sx.map((p) => [p.iter, p.infeas]) }];
      obj = [{ name: "objective", color: "#5ce1e6", points: sx.map((p) => [p.iter, p.obj]) }];
      resTitle = "Simplex infeasibility (log)"; resSub = "logged every 1000 iterations";
      objTitle = "Simplex objective";
    } else if (bb.length) {
      obj = [{ name: "incumbent", color: "#3ddc97", points: bb.map((p) => [p.node, p.incumbent]) }];
      objTitle = "Branch-and-bound incumbents"; objSub = "x = node";
      resTitle = "Convergence"; resSub = "not logged for branch-and-bound";
    }
    const emptyMsg = !r ? "run a model to see live convergence" : r.done ? "the engine finished before its first log line (small model)" : "waiting for the first log line…";
    chartRes.replaceChildren(lineChart(res, { logY: true, threshold: thr, thresholdLabel: thr ? `target ${thr}` : "", empty: emptyMsg, xLabel: "iteration" }));
    chartObj.replaceChildren(lineChart(obj, { empty: emptyMsg, xLabel: bb.length ? "node" : "iteration", robust: pd.length > 0 }));
    const leg = (series) => series.map((se) => h("span", h("i", { style: { background: se.color } }), se.name));
    resLegend.replaceChildren(...leg(res)); objLegend.replaceChildren(...leg(obj));
    root.querySelector("#res-title").textContent = resTitle; root.querySelector("#res-sub").textContent = resSub;
    root.querySelector("#obj-title").textContent = objTitle; root.querySelector("#obj-sub").textContent = objSub;
    if (sol && !pd.length && !sx.length && !bb.length && sol.objective != null)
      root.querySelector("#obj-sub").textContent = `final ${fnum(sol.objective, 10)}`;
  }

  function drawConsole() {
    const r = store.run && store.run.path === store.selectedModel ? store.run : null;
    if (!r) { consoleEl.replaceChildren(h("span.ev", "no run yet — choose a model and press Run")); return; }
    const lines = [];
    if (r.command) lines.push(h("span.cmd", `$ ${r.command}\n`));
    const P = r.progress;
    for (const p of P.slice(-60)) {
      if (p.kind === "pdhg") lines.push(`${p.engine} it ${String(p.iter).padStart(7)}  t ${p.t.toFixed(2)}s  pobj ${p.pobj?.toExponential(6)}  dobj ${p.dobj?.toExponential(6)}  rp ${fexp(p.rp)}  rd ${fexp(p.rd)}  gap ${fexp(p.gap)}  ${p.precision}\n`);
      else if (p.kind === "simplex") lines.push(`simplex it ${String(p.iter).padStart(7)}  phase ${p.phase}  obj ${p.obj?.toExponential(8)}  infeas ${fexp(p.infeas)}  ${p.t.toFixed(2)}s\n`);
      else lines.push(`b&b node ${p.node}  incumbent ${p.incumbent}\n`);
    }
    for (const l of r.logs.slice(-20)) lines.push(h("span.ev", l + "\n"));
    if (r.result) lines.push(h("span.ev", "\n" + r.result.stdout + "\n"));
    if (r.verify?.report) lines.push(h("span.cmd", `\n$ python3 tools/verify.py … → ${r.verify.report.verdict} (${fsec(r.verify.seconds)})\n`));
    else if (r.verify?.skipped) lines.push(h("span.cmd", `\nverify.py not run — ${r.verify.reason}\n`));
    if (r.error) lines.push(h("span.bad", r.error));
    consoleEl.replaceChildren(...lines);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  function drawSolution() {
    const r = store.run && store.run.path === store.selectedModel ? store.run : null;
    const sol = r?.result?.solution;
    solTabs.replaceChildren(...[["vars", "Variables"], ["cons", "Constraints"]].map(([k, l]) =>
      h(`button.btn`, { style: { height: "22px", padding: "0 8px", fontSize: "11px", borderColor: solTab === k ? "var(--accent)" : null }, onclick: () => { solTab = k; drawSolution(); } }, l)));
    if (!sol) { solBody.replaceChildren(h("div.empty", "values appear here after a run")); return; }
    if (!sol.columns.length) { solBody.replaceChildren(h("div.empty", sol.status === "Infeasible" ? "no solution: the model is infeasible (see the certificate)" : "no point returned")); return; }
    const rows = solTab === "vars"
      ? [...sol.columns].sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 300).map((c) =>
          h("tr", h("td.mono", c.name), h("td.num", fnum(c.x, 10)), h("td.num.dim", fexp(c.z))))
      : [...sol.rows].sort((a, b) => Math.abs(b.y) - Math.abs(a.y)).slice(0, 300).map((c) =>
          h("tr", h("td.mono", c.name), h("td.num", fnum(c.activity, 10)), h("td.num.dim", fexp(c.y))));
    solBody.replaceChildren(h("table.t", h("thead", h("tr", h("th", solTab === "vars" ? "column" : "row"),
      h("th.num", solTab === "vars" ? "value x" : "activity Ax"), h("th.num", solTab === "vars" ? "reduced cost z" : "dual y"))), h("tbody", rows)),
      h("div.note", `${solTab === "vars" ? sol.n_cols : sol.n_rows} entries, sorted by ${solTab === "vars" ? "|x|" : "|y|"}; first 300 shown${sol.vectors_truncated ? " (large model: the table holds the first 2000 of the solution file)" : ""}. Signs: z = c − Aᵀy.`));
  }

  function drawButtons() {
    const busy = store.run && !store.run.done;
    runBtn.disabled = !!busy;
    runBtn.style.display = busy ? "none" : "";
    cancelBtn.style.display = busy ? "" : "none";
    precSeg.draw(); gpuSeg.draw();
    gpuNote.textContent = store.system && !store.system.cuda_build
      ? "This build has no CUDA backend (Mac). GPU runs happen on the NVIDIA machine: scripts/gpu_check.sh; results are on the Benchmarks page." : "";
  }

  let raf = 0;
  function drawAll() {
    drawRef();
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; drawKpis(); drawPipeline(); drawCharts(); drawConsole(); drawSolution(); drawButtons(); renderCmd(); });
  }

  const off = store.on(drawAll);
  const key = (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); if (!(store.run && !store.run.done)) run(); } };
  window.addEventListener("keydown", key);
  loadCatalog().then(() => { fillModels(); loadInfo(); renderCmd(); drawAll(); });
  drawAll();
  return () => { off(); window.removeEventListener("keydown", key); };
}
