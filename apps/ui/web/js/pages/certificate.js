// pages/certificate.js — the verification certificate of one run, laid out to be projected to a
// jury, exportable as a self-contained HTML report (print it to PDF from the browser). The content is
// /api/jobs/<id>/certificate: assembled on the server from the run's own outputs and verify.py.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, fnum, fexp, fint, fsec, toast } from "../util.js";

const param = (k) => new URLSearchParams(location.hash.split("?")[1] || "").get(k);

export function certificateCard(c, { compact = false } = {}) {
  const ok = c.final.verdict === "PASS", m = c.model, r = c.result, run = c.run, g = c.gate, v = c.verifier;
  const cell = (k, val, cls = "") => h(`div.cc${cls ? "." + cls : ""}`, h("div.k", k), h("div.v", { title: typeof val === "string" ? val : null }, val ?? "—"));
  const claim = r.status === "Infeasible" || r.status === "Unbounded";
  return h(`div.certv${ok ? ".pass" : ".fail"}${compact ? ".compact" : ""}`,
    h("div.certv-head",
      h("div", h("div.eyebrow", `verification certificate · run ${c.id}`), h("div.certv-title", m.name), h("div.certv-sub.mono", `${m.path} · ${c.created}`)),
      h(`div.stamp${ok ? "" : ".no"}`, h("div.stamp-big", ok ? "✓ PASS" : "NOT CERTIFIED"), h("div.stamp-small", ok ? "every required check passed" : `failed: ${c.final.reasons.join(", ")}`))),
    h("div.certv-grid",
      cell("status", r.status), cell("objective", claim ? "— (no optimum exists)" : fnum(r.objective, 12), "wide"),
      cell("rows × cols", `${fint(m.rows)} × ${fint(m.cols)}`), cell("nonzeros", fint(m.nnz)),
      cell("engine", run.engine), cell("backend", `${run.backend} · ${run.threads === 0 ? "all cores" : `${run.threads || 1} thread`}`),
      cell("precision", run.precision), cell("iterations", fint(run.iterations)), cell("solve time", fsec(run.solve_seconds)),
      cell("primal residual", fexp(r.primal_residual)), cell("dual residual", fexp(r.dual_residual)), cell("duality gap", fexp(r.gap)),
      cell("worst-row primal · gate", fexp(g.primal)), cell("worst-row dual · gate", fexp(g.dual)),
      cell("model fingerprint", m.fingerprint, "wide mono"), cell("verifier reader", m.verifier_reader || "—"),
      cell("rounding-proof bound", r.certified_bound != null ? fnum(r.certified_bound, 12) : "—"),
      cell("independent check", v.verdict ? `${v.verdict} · ${fsec(v.seconds)}` : v.skipped ? "n/a" : "—")),
    h("div.certv-checks", c.checks.map((k) => h(`div.ck${k.ok ? ".is-ok" : k.required ? ".is-bad" : ".is-info"}`,
      h("span.ck-dot", k.ok ? "✓" : k.required ? "✕" : "·"), h("div", h("div.ck-name", k.name, k.required ? "" : h("span.dim", " · informative")), h("div.ck-detail.mono", k.detail))))),
    compact ? null : h("div.certv-foot.mono",
      h("div", `solution file ${c.artifact.solution_file || "—"} · SHA-256 ${c.artifact.sha256 || "—"}`),
      h("div", `${c.solver.version} · ${c.machine.cpu} · ${c.machine.cores} cores · ${c.machine.os}`),
      h("div", `$ ${c.solver.command || ""}`)));
}

export function mount(root) {
  let runs = [], cert = null, jid = param("job");
  const pick = h("select.input", { style: { minWidth: "320px" }, onchange: (e) => { jid = e.target.value; history.replaceState(null, "", `#/certificate?job=${jid}`); load(); } });
  const evChk = h("input", { type: "checkbox", checked: true });
  const body = h("div");
  const exportBtn = h("button.btn.primary", { onclick: () => { if (cert) location.href = api.reportUrl([cert.id], { evidence: evChk.checked, download: true }); } }, "Export HTML report");
  const printBtn = h("button.btn", { onclick: () => { if (cert) window.open(api.reportUrl([cert.id], { evidence: evChk.checked }), "_blank"); } }, "Printable ↗");
  // a real PDF when the server has a local Chrome / Chromium to print it (/api/system pdf_export)
  const pdfBtn = h("button.btn", { style: { display: store.system?.pdf_export ? "" : "none" },
    onclick: () => { if (cert) location.href = api.reportUrl([cert.id], { evidence: evChk.checked, download: true, format: "pdf" }); } }, "Download PDF");
  const presentBtn = h("button.btn", { onclick: () => document.body.classList.toggle("present") }, "Present ⤢");
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "06 · certificate"), h("h1", "Verification certificate"),
      h("p", "One page per run, for the jury: what was solved, how, and every check it passed. Built from the run's own solution file and the independent verifier's report; exported as a self-contained HTML file (open it and print to PDF)."))),
    h("div.toolbar", h("span.dim", "run"), pick, h("label.check", { style: { margin: 0 } }, evChk, "include benchmark evidence"), h("span.grow"), exportBtn, pdfBtn, printBtn, presentBtn),
    body);

  async function load() {
    if (!jid) { body.replaceChildren(h("div.panel", h("div.empty", "no finished run yet — solve a model first"), h("div", { style: { textAlign: "center", paddingBottom: "20px" } }, h("a.btn.primary", { href: "#/solve" }, "Go to Solve ▸")))); return; }
    body.replaceChildren(h("div.panel", h("div.empty", `reading run ${jid}…`)));
    try { cert = await api.certificate(jid); body.replaceChildren(certificateCard(cert)); }
    catch (e) { cert = null; body.replaceChildren(h("div.panel", h("div.panel-b.bad", e.message))); }
  }
  function fill() {
    pick.replaceChildren(...runs.map((r) => h("option", { value: r.id, selected: r.id === jid || null },
      `${new Date(r.created * 1000).toLocaleTimeString()} · ${r.model.split("/").pop()} · ${r.status || "—"} · ${r.verdict || "no verify"}`)));
  }
  const offSys = store.on((p) => { if ("system" in p) pdfBtn.style.display = store.system?.pdf_export ? "" : "none"; });
  api.runs().then((rs) => {
    runs = rs.filter((r) => r.status || r.id === jid);  // a run that produced no answer has no certificate
    if (!jid) jid = (store.run?.done && store.run.result ? store.run.job : null) || runs[0]?.id;
    if (!runs.length) { body.replaceChildren(h("div.panel", h("div.empty", "no finished run yet — solve a model first"))); return; }
    fill(); load();
  }).catch((e) => toast(e.message));
  return () => { offSys(); document.body.classList.remove("present"); };
}
