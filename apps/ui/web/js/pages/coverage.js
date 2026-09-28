// pages/coverage.js — the problem statement's requirements, one card each: status, implementation,
// the test that checks it, the evidence, what remains, and where to see it in this app. Parsed live
// from docs/SIH_STATUS.md (edit the markdown, the page follows) — an evaluator's checklist.
import { api } from "../api.js";
import { h, toast } from "../util.js";

const TONE = { DONE: "ok", PARTIAL: "warn", EXPERIMENTAL: "vio", "NOT STARTED": "bad", UNKNOWN: "mut" };
const COLOR = { DONE: "#3ddc97", PARTIAL: "#ffb547", EXPERIMENTAL: "#a78bfa", "NOT STARTED": "#ff5d6c", UNKNOWN: "#5f6d7c" };
// where each requirement can be seen working in the Command Center (matched on the requirement text)
const SHOW = [
  [/industry-standard|MPS/i, "#/models", "model explorer"], [/correct optimal/i, "#/demo?step=1", "live AFIRO + chain"],
  [/duals/i, "#/solve", "solve · solution table"], [/infeasible/i, "#/demo?step=3", "live Farkas certificate"],
  [/robust/i, "#/verify", "verification chain"], [/large-scale/i, "#/demo?step=5", "live refinery year"],
  [/GPU/i, "#/gpu", "GPU evidence"], [/MILP/i, "#/demo?step=4", "live MILP"], [/what-if|scenario/i, "#/scenarios", "what-if planning"],
  [/comparison with real-world/i, "#/compare", "vs HiGHS"], [/user interface/i, "#/demo", "jury demo"],
  [/reproducible/i, "#/bench", "benchmark evidence"], [/not built on/i, "#/preflight", "system check"],
];

// minimal inline markdown: `code` and **bold**, as DOM nodes (no HTML injection)
function md(text) {
  const out = [];
  for (const [i, part] of (text || "").split("`").entries()) {
    if (i % 2) { out.push(h("code", part)); continue; }
    for (const [j, b] of part.split("**").entries()) if (b) out.push(j % 2 ? h("b", b) : b);
  }
  return out;
}

export function mount(root) {
  const body = h("div");
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "02 · ps coverage"), h("h1", "Problem statement coverage"),
      h("p", "Every requirement of SIH 2026 PS 26119, with its status, the test that checks it and the evidence behind it — and where to see it working in this app. Parsed live from docs/SIH_STATUS.md; a status is DONE only when it is implemented, tested and backed by a committed benchmark CSV."))),
    body);

  function draw(d) {
    const total = d.rows.length;
    const order = ["DONE", "PARTIAL", "EXPERIMENTAL", "NOT STARTED"];
    body.replaceChildren(
      h("div.kpis", order.map((w) => h(`div.kpi${w === "DONE" ? ".hl.big" : ""}`, h("div.k", w.toLowerCase()), h("div.v", { style: { color: COLOR[w] } }, String(d.counts[w] || 0)), h("div.s", `of ${total} requirements`))),
        h("div.kpi.span2", h("div.k", "source"), h("div.v", { style: { fontSize: "14px" } }, d.source), h("div.s", `updated ${d.updated || "—"} · status words: DONE, PARTIAL, EXPERIMENTAL, NOT STARTED`))),
      h("div.split", { style: { height: "10px", margin: "12px 0 16px" } }, order.map((w) => h("i", { style: { width: `${(100 * (d.counts[w] || 0)) / total}%`, background: COLOR[w] }, title: `${w} ${d.counts[w] || 0}` }))),
      h("div.cov-grid", d.rows.map((r) => {
        const show = SHOW.find(([re]) => re.test(r.requirement));
        return h(`div.cov.${TONE[r.status_word]}`,
          h("div.cov-h", h("div.cov-t", r.requirement), h(`span.badge.${TONE[r.status_word]}`, r.status)),
          h("dl.cov-kv",
            h("dt", "how"), h("dd", md(r.implementation)),
            r.test && r.test !== "—" ? [h("dt", "test"), h("dd", md(r.test))] : null,
            r.evidence && r.evidence !== "—" ? [h("dt", "evidence"), h("dd", md(r.evidence))] : null,
            r.benchmark && r.benchmark !== "—" ? [h("dt", "csv"), h("dd.mono", md(r.benchmark))] : null,
            r["remaining work"] && r["remaining work"] !== "—" ? [h("dt", "remaining"), h("dd.dim", md(r["remaining work"]))] : null),
          show ? h("a.cov-show", { href: show[1] }, `show me · ${show[2]} ▸`) : null);
      })));
  }
  api.coverage().then(draw).catch((e) => { toast(e.message); body.replaceChildren(h("div.panel", h("div.panel-b.bad", e.message))); });
}
