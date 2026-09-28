// pages/compare.js — ps26119 against a real-world solver (HiGHS), engine by engine, from the committed
// bench/results/compare-highs-<machine>-<hash>.csv (bench/compare_highs.py). Every aggregate below
// is computed here from those rows, with its rule stated next to it.
import { api } from "../api.js";
import { h, fnum, fexp, fint, fsec, fratio, srcChip, statusBadge, toast } from "../util.js";
import { barChart, profileChart, scatterChart } from "../charts.js";

const LABEL = {
  "ps26119·auto": "ps26119 · auto", "ps26119·simplex": "ps26119 · simplex", "ps26119·r2hpdhg": "ps26119 · r²HPDHG",
  "ps26119·r2hpdhg·mt": "ps26119 · r²HPDHG all cores", "highs·simplex": "HiGHS · dual simplex", "highs·ipm": "HiGHS · interior point",
  "highs·pdlp": "HiGHS · PDLP",
};
const COLOR = {
  "ps26119·auto": "#5ce1e6", "ps26119·simplex": "#2aa8b0", "ps26119·r2hpdhg": "#3ddc97", "ps26119·r2hpdhg·mt": "#a7f3d0",
  "highs·simplex": "#ffb547", "highs·ipm": "#ff8a5c", "highs·pdlp": "#a78bfa",
};
const DASH = { "highs·simplex": "6 3", "highs·ipm": "6 3", "highs·pdlp": "6 3" };
const keyOf = (r) => `${r.solver}·${r.engine}${r.solver === "ps26119" && r.threads !== 1 && r.threads !== "1" ? "·mt" : ""}`;
const SHIFT = 1; // seconds, for the shifted geometric mean

export function mount(root) {
  let data = null, set = "all";
  const body = h("div.grid", { style: { gap: "14px" } });
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "09 · real-world solvers"), h("h1", "ps26119 vs HiGHS"),
      h("p", "The PS asks for comparison with real-world solvers. HiGHS — the open-source solver inside SciPy and JuMP — is run on the same models, engine by engine: its dual simplex, its interior point and its PDLP (the CPU cuPDLP-C port, the same algorithm family as our r²HPDHG). Both sides are timed on the solve call only and judged by the same rule: Optimal AND the independent verifier passes AND within 1e-6 of the reference optimum."))),
    body);

  function draw() {
    if (!data) { body.replaceChildren(h("div.panel", h("div.empty", "reading the comparison…"))); return; }
    const run = data.runs[0];
    if (!run) {
      body.replaceChildren(h("div.panel", h("div.empty", "no committed comparison yet — run bench/compare_highs.py and commit its CSV")));
      return;
    }
    const all = run.rows.filter((r) => set === "all" || r.set === set);
    const keys = Object.keys(LABEL).filter((k) => all.some((r) => keyOf(r) === k));
    const insts = [...new Set(all.map((r) => r.instance))];
    const byInst = Object.fromEntries(insts.map((i) => [i, Object.fromEntries(all.filter((r) => r.instance === i).map((r) => [keyOf(r), r]))]));
    const solved = (r) => r && r.solved === "yes";
    const count = Object.fromEntries(keys.map((k) => [k, insts.filter((i) => solved(byInst[i][k])).length]));
    const claimsRejected = Object.fromEntries(keys.map((k) => [k, insts.filter((i) => { const r = byInst[i][k]; return r && r.status === "Optimal" && !solved(r); }).length]));
    // shifted geometric mean of time, unsolved counted at the time limit
    const sgm = (k) => {
      const t = insts.map((i) => { const r = byInst[i][k]; return solved(r) ? r.seconds : r?.time_limit ?? null; }).filter((x) => x != null);
      return t.length ? Math.exp(t.reduce((a, x) => a + Math.log(x + SHIFT), 0) / t.length) - SHIFT : null;
    };
    const sg = Object.fromEntries(keys.map((k) => [k, sgm(k)]));
    const bestHighs = keys.filter((k) => k.startsWith("highs")).reduce((a, k) => (!a || count[k] > count[a] ? k : a), null);

    // performance profile over instances some solver solved
    const solvable = insts.filter((i) => keys.some((k) => solved(byInst[i][k])));
    const ratios = Object.fromEntries(keys.map((k) => [k, solvable.map((i) => {
      const best = Math.min(...keys.filter((q) => solved(byInst[i][q])).map((q) => Math.max(byInst[i][q].seconds, 1e-6)));
      const r = byInst[i][k];
      return solved(r) ? Math.max(r.seconds, 1e-6) / best : Infinity;
    })]));
    const within = (k, tau) => ratios[k].filter((x) => x <= tau).length;

    const kpis = h("div.kpis",
      h("div.kpi.hl.big", h("div.k", "ps26119 auto · solved"), h("div.v", `${count["ps26119·auto"] ?? 0}/${insts.length}`), h("div.s", "Optimal + verified + matches the reference")),
      bestHighs ? h("div.kpi.big", h("div.k", `best HiGHS engine · solved`), h("div.v", `${count[bestHighs]}/${insts.length}`), h("div.s", LABEL[bestHighs])) : null,
      h("div.kpi", h("div.k", "fastest within 2× · ours / HiGHS"), h("div.v", `${within("ps26119·auto", 2)} / ${bestHighs ? within(bestHighs, 2) : "—"}`), h("div.s", `of ${solvable.length} solvable models (profile at τ = 2)`)),
      h("div.kpi", h("div.k", "shifted geo-mean time"), h("div.v", `${fsec(sg["ps26119·auto"])} / ${bestHighs ? fsec(sg[bestHighs]) : "—"}`), h("div.s", `ours auto / ${bestHighs ? LABEL[bestHighs] : ""} · shift ${SHIFT} s, unsolved = limit`)),
      h(`div.kpi.${claimsRejected["ps26119·auto"] === 0 ? "okc" : "badc"}`, h("div.k", "‘Optimal’ claims rejected"), h("div.v", `${claimsRejected["ps26119·auto"] ?? 0}`), h("div.s", "ours auto — by the independent verifier or the reference")));

    const setSeg = h("div.seg", { style: { maxWidth: "420px" } }, [["all", "all models"], ["netlib", "Netlib"], ["scale", "large / refinery"]].map(([v, l]) =>
      h("button", { class: set === v ? "on" : null, onclick: () => { set = v; draw(); } }, l)));

    const profile = h("div.panel", { style: { gridColumn: "1 / -1" } },
      h("div.panel-h", "Performance profile (Dolan–Moré)", h("span.right.dim", `${solvable.length} models solved by at least one engine`)),
      h("div.panel-b.tight", profileChart(keys.map((k) => ({ name: LABEL[k], color: COLOR[k], dash: DASH[k], ratios: ratios[k] })), { tauMax: 2 ** 12 })),
      h("div.legend", keys.map((k) => h("span", h("i", { style: { background: COLOR[k], height: "3px" } }), `${LABEL[k]} · ${count[k]}/${insts.length}`))),
      h("div.note", "Read it like this: at τ = 1 the curve shows how often an engine was the fastest; further right, how often it finished within τ × the fastest time; the height at the right edge is the share it solved at all. Higher is better. Unsolved = never reaches the curve."));

    const bars = h("div.panel", h("div.panel-h", "Solved, by engine"),
      h("div.panel-b", barChart(keys.map((k) => ({ label: LABEL[k], value: count[k], color: COLOR[k], text: `${count[k]}/${insts.length}` })), { labelW: 200 })),
      h("div.note", "Solved = status Optimal AND tools/verify.py PASS AND |obj − ref| / (1 + |ref|) ≤ 1e-6. The same rule and the same verifier for both solvers."));

    const rej = h("div.panel", h("div.panel-h", "When a solver says ‘Optimal’ but is not"),
      h("table.t", h("thead", h("tr", h("th", "engine"), h("th.num", "Optimal claims"), h("th.num", "rejected"), h("th", "why it matters"))),
        h("tbody", keys.map((k) => { const claims = insts.filter((i) => byInst[i][k]?.status === "Optimal").length;
          return h("tr", h("td.mono", LABEL[k]), h("td.num", claims), h("td.num", { class: claimsRejected[k] ? "warn" : "ok" }, claimsRejected[k]), h("td.dim", k.startsWith("ps26119") ? "ours: the in-process gate withdraws Optimal when the original model is violated" : "HiGHS's own stopping test; checked here by our verifier")); }))),
      h("div.note", "A rejected claim is an answer its own solver called optimal that violates the original model beyond 1e-6 (worst row, relative) or misses the reference optimum. First-order methods with norm-based stopping tests are the usual source."));

    const pts = insts.map((i) => ({ i, a: byInst[i]["ps26119·auto"], b: byInst[i]["highs·simplex"] })).filter(({ a, b }) => solved(a) && solved(b));
    const scatter = h("div.panel", h("div.panel-h", "Head to head: ps26119 auto vs HiGHS dual simplex", h("span.right.dim", `${pts.length} models both solved`)),
      h("div.panel-b.tight", scatterChart(pts.map(({ i, a, b }) => ({ x: Math.max(b.seconds, 1e-5), y: Math.max(a.seconds, 1e-5), color: a.seconds <= b.seconds ? "#3ddc97" : "#ffb547",
        title: `${i}: ours ${fsec(a.seconds)} · HiGHS ${fsec(b.seconds)}` })), { xLabel: "HiGHS dual simplex, seconds", yLabel: "ps26119 auto, seconds", diagLabel: "same time" })),
      h("div.note", `Below the diagonal (green) ours is faster: ${pts.filter((p) => p.a.seconds <= p.b.seconds).length} of ${pts.length}. HiGHS is a mature, heavily optimized simplex; the honest reading is where each wins.`));

    // large models: heat-map table
    const scaleInsts = [...new Set(run.rows.filter((r) => r.set === "scale").map((r) => r.instance))];
    const heat = scaleInsts.length ? h("div.panel", { style: { gridColumn: "1 / -1" } },
      h("div.panel-h", "Large and refinery models · time to a verified optimum", h("span.right.dim", "green = fastest verified · — = not solved (limit or rejected)")),
      h("div", { style: { overflowX: "auto" } }, h("table.t", h("thead", h("tr", h("th", "model"), h("th.num", "rows"), h("th.num", "nonzeros"), ...Object.keys(LABEL).map((k) => h("th.num", LABEL[k])))),
        h("tbody", scaleInsts.map((i) => {
          const rs = Object.fromEntries(run.rows.filter((r) => r.instance === i).map((r) => [keyOf(r), r]));
          const r0 = Object.values(rs)[0];
          const best = Math.min(...Object.values(rs).filter(solved).map((r) => r.seconds));
          return h("tr", h("td.mono", i), h("td.num", fint(r0.rows)), h("td.num", fint(r0.nnz)), ...Object.keys(LABEL).map((k) => {
            const r = rs[k];
            if (!r) return h("td.num.dim", "—");
            if (!solved(r)) return h("td.num.dim", { title: `${r.status} · verify ${r.verify} · ${r.message || ""}` }, r.status === "Optimal" ? "rejected" : r.status === "TimeLimit" ? "limit" : r.status);
            const q = r.seconds / best;
            return h("td.num", { style: { background: q <= 1.0001 ? "rgba(61,220,151,.18)" : q <= 3 ? "rgba(92,225,230,.08)" : "transparent", color: q <= 1.0001 ? "var(--ok)" : null }, title: `${fratio(q)} of the fastest` }, fsec(r.seconds));
          }));
        })))),
      h("div.note", "Generated models with an optimum known by construction (refinery structure, synthetic prices; random sparse). The reference is that known optimum, so no solver is judged against another.")) : null;

    const table = h("details.panel", { style: { gridColumn: "1 / -1" } }, h("summary.panel-h", `Every run (${all.length} rows)`),
      h("div", { style: { overflowX: "auto", maxHeight: "480px" } }, h("table.t", h("thead", h("tr", ...["model", "engine", "status", "time", "iterations", "objective", "err vs ref", "verify", "solved"].map((x) => h("th", x)))),
        h("tbody", all.map((r) => h("tr", h("td.mono", r.instance), h("td.mono", LABEL[keyOf(r)] || keyOf(r)), h("td", statusBadge(r.status)), h("td.num", fsec(r.seconds)), h("td.num", fint(r.iterations)),
          h("td.num", fnum(r.objective, 11)), h("td.num", fexp(r.rel_err_ref)), h("td.mono", r.verify), h("td", { class: r.solved === "yes" ? "ok" : "dim" }, r.solved))))))));

    body.replaceChildren(...[
      h("div.panel", h("div.panel-b", { style: { display: "flex", gap: "16px", alignItems: "center", flexWrap: "wrap" } }, setSeg,
        h("span.dim", `${run.solver_versions.join(" · ")} · ${run.source.cpu} · ${run.rows[0]?.cpu_cores || "?"} cores`), h("span", { style: { marginLeft: "auto" } }, srcChip(run.source)))),
      kpis, profile, h("div.grid.g2", bars, scatter), rej, heat, table].filter(Boolean));
  }

  api.compare().then((d) => { data = d; draw(); }).catch((e) => toast(e.message));
  draw();
}
