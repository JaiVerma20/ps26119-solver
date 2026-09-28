// pages/demo.js — Jury Demo Mode: a full-screen, keyboard-driven walk through the product.
// Every "run" step solves a real model with the real binary on this machine and is re-checked by the
// independent verifier; the GPU step shows the committed GPU run (or runs live on a CUDA build).
// Nothing is pre-recorded or simulated. The script is the STEPS array below — edit it to change the demo.
//
// Keys: → / Space next · ← back · Enter run this step · N presenter notes · F full screen · Esc leave
import { api } from "../api.js";
import { store } from "../store.js";
import { h, fnum, fexp, fint, fsec, fratio, statusBadge, toast } from "../util.js";
import { lineChart } from "../charts.js";
import { startSolve, whenDone } from "../run.js";

const STEPS = [
  { id: "intro", kind: "intro", title: "A solver that proves its answers",
    lead: "PS26119 — LP and MILP from scratch: an exact simplex, a GPU-native first-order engine (r²HPDHG) for very large models, and a verification chain on every answer.",
    notes: ["Everything you will see runs live on this laptop with the real solver binary.", "Every answer is re-checked by an independent verifier. Nothing on screen is simulated or typed in."] },
  { id: "afiro", kind: "solve", title: "A classic LP, solved exactly", model: "data/netlib_small/afiro.mps", opts: { algorithm: "auto", threads: 0 },
    lead: "AFIRO from the Netlib library — 27 constraints, 32 variables. For small models the dispatcher picks the revised simplex: an exact vertex with exact dual prices.",
    notes: ["Point at: status, objective −464.7531…, engine, time.", "The published optimum of AFIRO is −464.7531428571."] },
  { id: "chain", kind: "chain", of: "afiro", title: "Don't trust the solver — verify it",
    lead: "The same answer, checked twice by different code: inside the solver on the ORIGINAL (un-presolved) model, then by tools/verify.py, which re-reads the file with a different MPS reader.",
    notes: ["Fingerprint equal = both programs read exactly the same numbers.", "A failed check turns 'Optimal' into an error — a wrong answer is worse than no answer."] },
  { id: "infeasible", kind: "solve", title: "Proving that no plan exists", model: "data/examples/infeasible.mps", opts: { algorithm: "simplex" },
    lead: "When constraints contradict each other we do not just say 'infeasible' — we return Farkas multipliers: a certificate anyone can check. The verifier checks it in exact rational arithmetic.",
    notes: (ev) => { const x = ev?.infeasible?.find((i) => i.engine === "simplex");
      return ["L₀(r) > 0 is the proof: a combination of the constraints that reads 0 ≥ positive.",
        x ? `Netlib with a duality cut: ${x.certified} of ${x.total} infeasibility proofs certified (source ${x.source.file}).` : "See Benchmarks for the infeasibility evidence."]; } },
  { id: "milp", kind: "solve", title: "Integer decisions: MILP", model: "data/mip_small/gt2.lpm", opts: { algorithm: "auto" },
    lead: "MIPLIB gt2 — integer variables. Branch-and-bound with simplex node LPs and certified-bound pruning; the integer answer is re-checked for integrality, bounds and every row.",
    notes: (ev) => [ev?.miplib ? `${ev.miplib.solved} of ${ev.miplib.total} small MIPLIB-3 models proven optimal and verified (source ${ev.miplib.source.file}).` : "See Benchmarks for MIPLIB.",
      "A MILP is never answered by its LP relaxation."] },
  { id: "refinery", kind: "solve", title: "Refinery scale: one year, hour by hour", model: "bench/generated/refinery-T8760-s1.lpm", gen: ["refinery", 8760, 1],
    opts: { algorithm: "auto", threads: 0, time_limit: 300 }, chart: true,
    lead: "8,760 hourly periods of crude purchase, distillation, cat cracking, blending with quality specifications, tanks and demand: 429,240 constraints, 1.5 million nonzeros. r²HPDHG on all CPU cores.",
    notes: ["Structure is a real refinery's; prices are synthetic, and the optimum is known by construction — the error against it is shown.",
      "Other teams' own published GPU results stop around 10k rows (our research notes, docs/RESEARCH.md)."] },
  { id: "gpu", kind: "gpu", title: "The same engine on a GPU",
    lead: "r²HPDHG was designed for GPUs: one sparse matrix-vector product per iteration, no factorization. Measured on an NVIDIA laptop GPU against the fastest CPU configuration of the same machine, at the same 1e-8 tolerance; how each answer was checked is shown under it.",
    notes: ["Say: one consumer laptop GPU (RTX 4050); slower than the CPU on small models, faster on large ones.", "Every number here is read from the committed benchmark CSV named on screen."] },
  { id: "compare", kind: "compare", title: "Against a real-world solver",
    lead: "The PS asks for a comparison with real-world solvers. HiGHS — the open-source solver inside SciPy and JuMP — ran the same models, engine by engine, and the same independent verifier judged both. Where HiGHS is faster, it says so.",
    notes: ["Every number here is read from the committed comparison CSV named on screen.", "HiGHS is a mature, heavily optimized solver: say where each wins; our point is verified answers at refinery scale."] },
  { id: "final", kind: "final", title: "Final verification",
    lead: "Every run of this demo, with its independent verdict. The certificates and the benchmark evidence export as one self-contained report.",
    notes: ["Offer the exported report to the jury.", "Reproduce everything from one commit: scripts/reproduce.sh (CPU), scripts/gpu_check.sh (GPU)."] },
];

const compact = (n) => (n == null ? "—" : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e5 ? `${Math.round(n / 1e3)}k` : fint(n));

// kept across navigations while the server runs this tab
const D = { step: 0, notes: false, results: {}, gpu: null, compare: null, busy: false };

export function mount(root) {
  document.body.classList.add("demo-mode");
  // #/demo?step=N opens a given step (rehearsal, or resuming after a browser restart)
  const want = Number(new URLSearchParams(location.hash.split("?")[1] || "").get("step"));
  if (Number.isInteger(want) && want >= 0) D.step = want;
  const stage = h("div.demo");
  root.append(stage);
  let raf = 0;
  const redraw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

  if (!D.gpu) api.gpu().then((g) => { D.gpu = g; redraw(); }).catch(() => {});
  if (!D.compare) api.compare().then((c) => { D.compare = c; redraw(); }).catch(() => {});
  // steps whose evidence is not committed yet are skipped rather than shown empty
  const visible = () => STEPS.filter((x) => x.kind !== "compare" || D.compare?.runs?.length);

  async function runStep(s) {
    if (s.kind !== "solve" && !(s.kind === "gpu" && store.system?.cuda_build)) return;
    if (store.run && !store.run.done) { toast("a solve is already running"); return; }
    const model = s.kind === "gpu" ? "bench/generated/refinery-T8760-s1.lpm" : s.model;
    const opts = s.kind === "gpu" ? { algorithm: "r2hpdhg", gpu: true, threads: 0, time_limit: 300 } : s.opts;
    const key = s.kind === "gpu" ? "gpu-live" : s.id;
    D.busy = true; redraw();
    try {
      if (s.gen || s.kind === "gpu") { const g = await api.generate(...(s.gen || ["refinery", 8760, 1])); if (g.created) toast(g.message); }
      store.set({ selectedModel: model });
      const job = await startSolve(model, opts);
      D.results[key] = { job, running: true };
      redraw();
      const run = await whenDone(job);
      let cert = null;
      try { cert = await api.certificate(job); } catch { /* shown as not certified */ }
      D.results[key] = { job, running: false, run, cert };
    } catch (e) { toast(e.message); } finally { D.busy = false; redraw(); }
  }

  // ------------------------------------------------------------------ step views
  const big = (k, v, s, cls = "") => h(`div.dk${cls ? "." + cls : ""}`, h("div.k", k), h("div.v", v ?? "—"), s ? h("div.s", s) : null);
  const stamp = (ok, text) => h(`div.dstamp${ok ? "" : ".no"}`, text);

  function liveView(s, key = s.id) {
    const R = D.results[key];
    const r = R?.running ? store.run : R?.run;
    if (!R) return h("div.dhint", h("kbd", "Enter"), " run it live on this machine");
    const res = r?.result, sol = res?.solution, last = r?.progress?.at(-1), cert = R.cert;
    const claim = sol?.status === "Infeasible" || sol?.status === "Unbounded";
    const kp = [
      big("status", sol ? statusBadge(sol.status) : h("span.acc", "running…"), sol?.engine || (last ? `${last.kind} · iteration ${fint(last.iter ?? last.node)}` : "starting")),
      big(claim ? "certificate" : "objective", claim ? (sol.status === "Infeasible" ? "Farkas" : "ray") : fnum(sol?.objective ?? null, 11),
        claim ? [(r.verify?.report?.certificate || res?.check_line || ""), r.verify?.report?.farkas_exact_L0 != null ? ` · L₀ = ${r.verify.report.farkas_exact_L0} > 0` : ""].join("") : res?.known?.rel_err != null ? `known optimum: error ${fexp(res.known.rel_err)}` : sol?.dual_objective != null && Number.isFinite(sol.dual_objective) ? `dual ${fnum(sol.dual_objective, 11)}` : "", "wide"),
      big("size", res ? `${compact(res.rows)} × ${compact(res.cols)}` : "—", res ? (res.rows >= 1e5 ? `${fint(res.rows)} × ${fint(res.cols)} · ` : "") + `${fint(res.nnz)} nonzeros` : ""),
      big("solve time", fsec(sol?.seconds ?? last?.t ?? null), sol ? `${fint(sol.iterations)} ${/^branch/.test(sol.engine) ? "nodes / iterations" : "iterations"}` + (r.opts.gpu ? " · GPU" : /^(r2hpdhg|pdlp)/.test(sol.engine || "") ? ` · ${r.opts.threads === 0 ? `${store.system?.cores} CPU cores` : "1 thread"}` : "") : ""),
      big("independent verify", r?.verify?.report ? r.verify.report.verdict : r?.stage === "verify" ? h("span.acc", "checking…") : "—",
        r?.verify?.report ? `reader ${r.verify.report.reader} · ${fsec(r.verify.seconds)}` : "", r?.verify?.report?.verdict === "PASS" ? "okc" : r?.verify?.report ? "badc" : ""),
    ];
    const chart = s.chart || s.kind === "gpu" ? (() => {
      const pd = (r?.progress || []).filter((p) => p.kind === "pdhg");
      return h("div.dchart", lineChart([{ name: "relative KKT error", color: "#5ce1e6", points: pd.map((p) => [p.iter, Math.max(p.rp ?? 0, p.rd ?? 0, Math.abs(p.gap ?? 0))]) }],
        { logY: true, threshold: 1e-8, thresholdLabel: "1e-8 target", xLabel: "iteration", height: 210, empty: "waiting for the first iterations…" }));
    })() : null;
    return h("div", h("div.dkpis", kp), chart,
      cert ? h("div.dcert", stamp(cert.final.verdict === "PASS", cert.final.verdict === "PASS" ? "✓ CERTIFIED" : "NOT CERTIFIED"),
        h("div.dchecks", cert.checks.filter((c) => c.required).map((c) => h(`span.dcheck${c.ok ? ".ok" : ".bad"}`, `${c.ok ? "✓" : "✕"} ${c.name}`)))) : null);
  }

  function chainView(s) {
    const R = D.results[s.of], c = R?.cert;
    if (!c) return h("div.dhint", "run the previous step first (←)");
    const row = (ok, head, body, meta) => h(`div.dlink${ok ? ".is-ok" : ".is-bad"}`, h("div.dot", ok ? "✓" : "✕"), h("div", h("div.h", head), h("div.d", body), meta ? h("div.m.mono", meta) : null));
    const ck = (name) => c.checks.find((k) => k.name.startsWith(name)) || { ok: false, detail: "—" };
    return h("div.dchain",
      row(c.result.status === "Optimal", `1 · The solver: ${c.result.status}`, `${c.run.engine}, ${fint(c.run.iterations)} iterations, ${fsec(c.run.solve_seconds)}`, `objective ${fnum(c.result.objective, 12)}`),
      row(ck("Original-model").ok, "2 · Re-checked on the original model (in-process)", "Primal feasibility, dual feasibility and the duality gap recomputed from the answer on the un-presolved model.", ck("Original-model").detail),
      row(ck("Same model").ok, "3 · Same model? Fingerprint of every number", "The verifier re-reads the MPS file with a different reader and hashes every coefficient.", ck("Same model").detail),
      row(ck("Independent").ok, "4 · Independent verifier: tools/verify.py", "Separate code recomputes feasibility, optimality and the objective from the solution file alone.",
        `primal ${fexp(c.verifier.primal_rel)} · dual ${fexp(c.verifier.dual_rel)} · gap ${fexp(c.verifier.gap_rel)} · objective ${fnum(c.verifier.objective, 12)}`),
      h("div.dchain-foot", stamp(c.final.verdict === "PASS", c.final.verdict === "PASS" ? "✓ CERTIFIED" : "NOT CERTIFIED"),
        h("a.btn", { href: `#/certificate?job=${c.id}` }, "open the certificate ▸")));
  }

  function gpuView(s) {
    const M = D.gpu?.machines?.[0];
    if (!M) return h("div.dhint", D.gpu ? "no committed GPU run found" : "reading the GPU evidence…");
    const fp = M.pairs.filter((p) => p.precision === "fp64" && p.ratio_vs_best != null);
    const big1 = fp.reduce((a, p) => (!a || p.nnz > a.nnz ? p : a), null);
    const ref = fp.find((p) => /T8760/.test(p.instance));
    const bar = (label, secs, max, cls) => h("div.dbar", h("div.l", label), h("div.track", h(`div.fill.${cls}`, { style: { width: `${Math.max(2, 100 * secs / max)}%` } })), h("div.t.mono", fsec(secs)));
    const cmp = (p, title) => p ? h("div.dcmp", h("div.dcmp-h", title, h("span.dim", ` · ${fint(p.rows)} rows · ${fint(p.nnz)} nonzeros`)),
      bar(`CPU · ${p.best_cpu_threads} threads (fastest)`, p.best_cpu_s, Math.max(p.best_cpu_s, p.gpu_s), "cpu"),
      bar(`GPU · ${M.gpu.replace("NVIDIA GeForce ", "")}`, p.gpu_s, Math.max(p.best_cpu_s, p.gpu_s), "gpu"),
      h("div.dcmp-x", fratio(p.ratio_vs_best), h("span", " faster")),
      (() => { const g = M.instances.find((i) => i.instance === p.instance)?.runs.find((r) => r.backend === "gpu" && r.precision === "fp64");
        return g ? h("div.dcmp-v.mono", g.verify === "PASS" ? `GPU answer: ${g.status} · verify.py PASS · error vs known optimum ${fexp(g.rel_err_known)}`
          : `GPU answer: ${g.status} · in-process gate · error vs known optimum ${fexp(g.rel_err_known)} (verify.py: ${g.verify}, above its size limit)`) : null; })()) : null;
    const mine = D.results.refinery?.run?.result?.solution;
    const liveGpu = store.system?.cuda_build;
    return h("div",
      h("div.dcmps", cmp(ref, "Refinery year, hourly"), big1 && big1 !== ref ? cmp(big1, big1.instance) : null),
      h("div.dsrc.mono", `source: bench/results/${M.source.file} · commit ${M.source.git_hash} · ${M.gpu} · driver ${M.driver} · CUDA ${M.cuda} · compute-sanitizer memcheck ${M.sanitizer.memcheck}, racecheck ${M.sanitizer.racecheck}`),
      mine ? h("div.dnote", `On this laptop a moment ago (CPU, ${store.system?.cores} cores): the same refinery model in ${fsec(mine.seconds)} — a different machine from the GPU run above.`) : null,
      liveGpu ? h("div", h("div.dhint", h("kbd", "Enter"), " run the refinery year live on this machine's GPU"), D.results["gpu-live"] ? liveView(s, "gpu-live") : null)
        : h("div.dnote.dim", "This laptop has no CUDA GPU; the figures above are the committed, verified GPU run."));
  }

  function compareView() {
    const run = D.compare?.runs?.[0];
    if (!run) return h("div.dhint", "no committed comparison yet");
    const key = (r) => `${r.solver}·${r.engine}${r.solver === "ps26119" && String(r.threads) !== "1" ? "·mt" : ""}`;
    const NAME = { "ps26119·auto": "ps26119 auto", "highs·simplex": "HiGHS dual simplex", "highs·ipm": "HiGHS interior point", "highs·pdlp": "HiGHS PDLP" };
    const net = run.rows.filter((r) => r.set === "netlib");
    const cnt = (k) => [new Set(net.map((r) => r.instance)).size, net.filter((r) => key(r) === k && r.solved === "yes").length];
    const rej = (k) => run.rows.filter((r) => key(r) === k && r.status === "Optimal" && r.solved !== "yes").length;
    const ref = run.rows.filter((r) => /T8760/.test(r.instance));
    const kp = Object.keys(NAME).map((k) => { const [n, sv] = cnt(k); return big(NAME[k], `${sv}/${n}`, `Netlib solved · ${rej(k)} ‘Optimal’ claims rejected overall`, k.startsWith("ps26119") ? "okc" : ""); });
    const rows = Object.keys(NAME).map((k) => ref.find((r) => key(r) === k)).filter(Boolean);
    const best = Math.min(...rows.filter((r) => r.solved === "yes").map((r) => r.seconds));
    const max = Math.max(...rows.map((r) => (r.solved === "yes" ? r.seconds : r.status === "TimeLimit" ? r.time_limit : r.seconds || 0)), 1e-3);
    return h("div",
      h("div.dkpis", { style: { gridTemplateColumns: `repeat(${kp.length}, minmax(0, 1fr))` } }, kp),
      rows.length ? h("div.dcmp", { style: { marginTop: "16px" } }, h("div.dcmp-h", "Refinery year, hourly", h("span.dim", ` · ${fint(rows[0].rows)} rows · time to a verified optimum`)),
        rows.map((r) => h("div.dbar", h("div.l", NAME[key(r)]), h("div.track", h(`div.fill.${r.solver === "ps26119" ? "cpu" : "gpu"}`, { style: { width: `${Math.max(2, 100 * (r.solved === "yes" ? r.seconds : r.status === "TimeLimit" ? r.time_limit : r.seconds) / max)}%`, opacity: r.solved === "yes" ? 1 : 0.35 } })),
          h("div.t.mono", { style: { minWidth: "150px" } }, r.solved === "yes" ? fsec(r.seconds) + (r.seconds === best ? " ★" : "")
            : r.status === "TimeLimit" ? `limit · ${fsec(r.time_limit)}` : `${r.status === "Optimal" ? "rejected" : "not solved"} · ${fsec(r.seconds)}`)))) : null,
      h("div.dsrc.mono", `source: bench/results/${run.source.file} · commit ${run.source.git_hash} · ${run.solver_versions.join(" · ")} · same machine, same verifier, solve call timed on both sides`));
  }

  function finalView() {
    const done = STEPS.filter((s) => s.kind === "solve").map((s) => ({ s, R: D.results[s.id] })).filter((x) => x.R?.cert);
    if (D.results["gpu-live"]?.cert) done.push({ s: { title: "GPU live run" }, R: D.results["gpu-live"] });
    if (!done.length) return h("div.dhint", "no run yet — go back and run the steps (←)");
    const all = done.every((x) => x.R.cert.final.verdict === "PASS");
    return h("div",
      h("table.dtable", h("thead", h("tr", h("th", "step"), h("th", "model"), h("th", "status"), h("th.num", "objective"), h("th", "engine"), h("th.num", "time"), h("th", "verdict"))),
        h("tbody", done.map(({ s, R }) => { const c = R.cert; return h("tr", h("td", s.title), h("td.mono", c.model.name), h("td", statusBadge(c.result.status)),
          h("td.num", c.result.status === "Optimal" ? fnum(c.result.objective, 11) : "—"), h("td.mono", c.run.engine), h("td.num", fsec(c.run.solve_seconds)),
          h("td", h(`span.${c.final.verdict === "PASS" ? "ok" : "bad"}`, c.final.verdict === "PASS" ? "✓ PASS" : "✕ NOT CERTIFIED"))); }))),
      h("div.dfinal", stamp(all, all ? `✓ ALL ${done.length} RUNS CERTIFIED` : "SOME RUNS NOT CERTIFIED"),
        h("div", { style: { display: "flex", gap: "10px" } },
          h("a.btn.primary", { href: api.reportUrl(done.map((x) => x.R.job), { download: true, title: "Jury demo — verification report" }) }, "Export the report (HTML)"),
          h("a.btn", { href: api.reportUrl(done.map((x) => x.R.job), { title: "Jury demo — verification report" }), target: "_blank" }, "Printable / PDF ↗"))));
  }

  function introView() {
    const ev = store.evidence;
    const n = ev?.netlib?.find((x) => x.engine === "auto"), inf = ev?.infeasible?.find((x) => x.engine === "simplex");
    const M = D.gpu?.machines?.[0], ref = M?.pairs.find((p) => p.precision === "fp64" && /T8760/.test(p.instance) && p.ratio_vs_best != null);
    const cards = [
      n ? big("Netlib LP library", `${n.solved}/${n.total}`, "solved and independently verified") : null,
      inf ? big("infeasibility proofs", `${inf.certified}/${inf.total}`, `certified · ${inf.exact_rational} in exact arithmetic`) : null,
      ev?.miplib ? big("small MIPLIB 3", `${ev.miplib.solved}/${ev.miplib.total}`, "MILP proven optimal and verified") : null,
      ref ? big("GPU · refinery year", fratio(ref.ratio_vs_best), `vs the fastest CPU · ${M.gpu.replace("NVIDIA GeForce ", "")}`) : null,
      big("this machine", store.system ? `${store.system.cores} cores` : "—", store.system ? `${store.system.cpu} · ${store.system.cuda_build ? "CUDA" : "CPU build"}` : ""),
    ].filter(Boolean);
    return h("div",
      h("div.dkpis", { style: { gridTemplateColumns: `repeat(${cards.length}, minmax(0, 1fr))` } }, cards),
      h("div.dsrc.mono", "figures above: committed benchmark CSVs in bench/results/ (see Benchmarks and GPU pages for each source file and commit)"),
      h("div.dagenda", visible().slice(1).map((x, i) => h("button.dag", { onclick: () => { D.step = i + 1; draw(); } },
        h("span.n.mono", String(i + 1).padStart(2, "0")), h("span.t", x.title), h("span.k.mono", x.kind === "solve" ? "live solve" : x.kind === "gpu" || x.kind === "compare" ? "evidence" : x.kind === "chain" ? "verification" : "summary")))));
  }

  function draw() {
    const V = visible();
    D.step = Math.min(D.step, V.length - 1);
    const s = V[D.step];
    const body = s.kind === "intro" ? introView() : s.kind === "solve" ? liveView(s) : s.kind === "chain" ? chainView(s) : s.kind === "gpu" ? gpuView(s) : s.kind === "compare" ? compareView() : finalView();
    stage.replaceChildren(
      h("div.demo-top",
        h("div.demo-brand", h("b", "PS26119"), h("span", "jury demo")),
        h("div.demo-dots", V.map((x, i) => h(`button.ddot${i === D.step ? ".on" : ""}${D.results[x.id]?.cert ? (D.results[x.id].cert.final.verdict === "PASS" ? ".pass" : ".fail") : ""}`,
          { title: x.title, onclick: () => { D.step = i; draw(); } }, i === 0 ? "◆" : String(i)))),
        h("div.demo-actions", h("button.btn", { onclick: () => toggleFull() }, "Full screen (F)"), h("a.btn", { href: "#/dashboard" }, "Exit (Esc)"))),
      h("div.demo-body",
        h("div.demo-step.mono", s.kind === "intro" ? "PS 26119 · SIH 2026 · MRPL" : `step ${D.step} of ${V.length - 1}`),
        h("h1.demo-title", s.title),
        h("p.demo-lead", s.lead),
        h("div.demo-stage", body),
        D.notes ? h("div.demo-notes", h("b", "presenter notes"), h("ul", (typeof s.notes === "function" ? s.notes(store.evidence) : s.notes).map((n) => h("li", n)))) : null),
      h("div.demo-foot.mono",
        h("span", h("kbd", "←"), h("kbd", "→"), " move"), (s.kind === "solve" || (s.kind === "gpu" && store.system?.cuda_build)) ? h("span", h("kbd", "Enter"), D.busy ? " running…" : " run live") : null,
        h("span", h("kbd", "N"), " notes"), h("span", h("kbd", "F"), " full screen"), h("span.grow"),
        h("span.dim", "live runs on this machine · GPU figures from committed evidence · nothing simulated")));
  }

  function toggleFull() {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else document.documentElement.requestFullscreen?.().catch(() => {});
  }
  function key(e) {
    if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
 const V = visible(), s = V[D.step];
    if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") { D.step = Math.min(V.length - 1, D.step + 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { D.step = Math.max(0, D.step - 1); draw(); e.preventDefault(); }
    else if (e.key === "Enter") { runStep(s); e.preventDefault(); }
    else if (e.key === "n" || e.key === "N") { D.notes = !D.notes; draw(); }
    else if (e.key === "f" || e.key === "F") toggleFull();
    else if (e.key === "Escape" && !document.fullscreenElement) location.hash = "#/dashboard";
    e.stopPropagation();
  }
  window.addEventListener("keydown", key, true);
  const off = store.on((p) => { if ("run" in p || "system" in p || "evidence" in p) redraw(); });
  draw();
  return () => { window.removeEventListener("keydown", key, true); off(); document.body.classList.remove("demo-mode"); };
}
