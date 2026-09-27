// pages/verify.js — the trust chain of the latest run: engine checks, in-process gate,
// certificate / rounding-proof bound, independent verifier (different reader, different code).
import { store } from "../store.js";
import { h, fnum, fexp, fsec, statusBadge, passBadge } from "../util.js";
import { certificateStatus } from "../run.js";

export function mount(root) {
  const body = h("div");
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "04 · verification"), h("h1", "Verification"),
      h("p", "A wrong answer is worse than no answer. Every result passes an in-process check on the ORIGINAL model; Infeasible and Unbounded carry a certificate; then tools/verify.py re-reads the model with a different reader (HiGHS's MPS parser used only as a file reader, or a separate Python reader for .lpm) and re-checks everything from the solution file alone."))),
    body);

  function draw() {
    const r = store.run;
    if (!r || !r.result) {
      body.replaceChildren(h("div.panel", h("div.empty", r ? "the run is still in progress…" : "no run yet — solve a model on the Solve page, then come back here"),
        h("div", { style: { textAlign: "center", paddingBottom: "24px" } }, h("a.btn.primary", { href: "#/solve" }, "Go to Solve ▸"))));
      return;
    }
    const res = r.result, sol = res.solution || {}, v = r.verify?.report, cert = certificateStatus(res, r.verify);
    const chk = res.check_line || "", status = sol.status;
    const claim = status === "Infeasible" || status === "Unbounded";
    const link = (state, head, desc, meta) => h("div.link", h(`div.dot.${state}`, state === "ok" ? "✓" : state === "bad" ? "✕" : state === "warn" ? "!" : "·"),
      h("div", h("div.h", head), h("div.d", desc), meta ? h("div.m", meta) : null));

    const links = [
      link(sol.status ? "ok" : "bad", `Engine verdict: ${sol.status || "none"}`,
        `${sol.engine} (${sol.precision}). The simplex confirms on a fresh factorization in unscaled units; first-order engines stop on the relative KKT test plus per-row checks.`,
        `${fnum(sol.iterations, 10)} it · ${fsec(sol.seconds)}`),
      link(chk.startsWith("PASS") ? "ok" : chk.startsWith("FAIL") ? "bad" : "mut",
        claim ? "In-process certificate check (original model)" : "In-process KKT gate (original model)",
        claim ? (status === "Infeasible" ? "Farkas multipliers r with L₀(r) > 0 prove that no feasible point exists; a noise-level L₀ is refused." : "A feasible point plus a ray d with cᵀd < 0 inside the recession cone.")
          : "Primal feasibility, dual feasibility and the duality gap recomputed from x and y on the unpresolved model (tolerance 1e-6); failure withdraws Optimal.",
        chk || "not applicable"),
      link(cert.cls === "ok" ? "ok" : cert.cls === "warn" ? "warn" : "mut", claim ? `Certificate: ${cert.label}` : "Rounding-proof bound on the optimum",
        claim ? "Evaluated with directed rounding in C++ and in exact rational arithmetic by the verifier where the model allows."
          : "From y with outward rounding (Neumaier–Shcherbina style): the true optimum is guaranteed on this side of the bound.",
        claim ? (v?.certificate || "") : res.certified_line || "—"),
      link(!v ? "mut" : v.fingerprint_model === "skipped" ? "mut" : v.model_match ? "ok" : "bad", "Same model? (fingerprint)", "The verifier re-reads the file with a different reader and hashes every number; the hash must equal the one the solver wrote.",
        !v ? "—" : v.fingerprint_model === "skipped" ? `not recomputed for a model this large (solver: ${v.fingerprint_solution})`
          : `${v.fingerprint_model} ${v.model_match ? "=" : "≠"} ${v.fingerprint_solution}`),
      res.known ? link(res.known.rel_err == null ? "mut" : res.known.rel_err <= 1e-6 ? "ok" : "warn", "Known optimum (by construction)",
        `This model was generated around a KKT point (${res.known.generator}), so its optimum is known without any other solver.`,
        res.known.rel_err == null ? `known ${fnum(res.known.optimum, 14)}` : `|obj − known| / (1 + |known|) = ${fexp(res.known.rel_err)} · known ${fnum(res.known.optimum, 14)}`) : null,
      link(!v ? (r.opts.verify === false || r.verify?.skipped ? "mut" : "warn") : v.verdict === "PASS" ? "ok" : "bad",
        `Independent verifier: ${v ? v.verdict : r.verify?.skipped ? "not applicable" : r.opts.verify === false ? "off" : "pending"}`,
        "tools/verify.py recomputes everything from the solution file alone; it shares no code with the solver's checks.",
        r.verify?.skipped ? r.verify.reason : r.verify ? `${fsec(r.verify.seconds)} · exit ${r.verify.exit_code}` : ""),
    ];

    // the in-process gate's worst-row figures (same definitions as verify.py), from the solution file's check line
    const g = (sol.check || "").match(/primal (\S+) dual (\S+) gap (\S+)/);
    const gate = g ? { p: +g[1], d: +g[2], gap: +g[3] } : {};
    const rowsCmp = claim ? [
      ["verdict", status, v?.status || "—"],
      ["certificate", chk.split("(")[0].trim() || "—", v?.certificate || "—"],
      ["exact L₀ (Farkas)", "—", v?.farkas_exact_L0 != null ? fexp(v.farkas_exact_L0) : "—"],
    ] : [
      ["objective", fnum(sol.objective, 12), fnum(v?.objective, 12)],
      ["dual objective", fnum(sol.dual_objective, 12), fnum(v?.dual_objective, 12)],
      ["worst primal violation (rel)", fexp(gate.p), fexp(v?.primal_rel)],
      ["worst dual violation (rel)", fexp(gate.d), fexp(v?.dual_rel)],
      ["duality gap (rel)", fexp(gate.gap), fexp(v?.gap_rel)],
      ["complementarity (abs)", "—", fexp(v?.complementarity_abs)],
    ];

    body.replaceChildren(
      h("div.kpis", { style: { marginBottom: "14px" } },
        h("div.kpi.hl", h("div.k", "model"), h("div.v", res.name || r.path.split("/").pop()), h("div.s", r.path)),
        h("div.kpi", h("div.k", "status"), h("div.v", statusBadge(status)), h("div.s", sol.engine)),
        h("div.kpi", h("div.k", "objective"), h("div.v", fnum(sol.objective, 10)), h("div.s", claim ? "undefined for this verdict" : "")),
        h("div.kpi", h("div.k", "in-process"), h("div.v", passBadge(chk.startsWith("PASS") ? "PASS" : chk.startsWith("FAIL") ? "FAIL" : null)), h("div.s", "original model")),
        h("div.kpi", h("div.k", "certificate"), h("div.v", h(`span.badge.${cert.cls}`, cert.label)), h("div.s", "")),
        h(`div.kpi.${v ? (v.verdict === "PASS" ? "okc" : "badc") : ""}`, h("div.k", "independent"), h("div.v", v ? v.verdict : "—"), h("div.s", v ? `reader: ${v.reader}` : ""))),
      h("div.grid.g2",
        h("div.panel", h("div.panel-h", "Trust chain"), h("div.chain", links)),
        h("div.panel", h("div.panel-h", "Solver vs independent verifier", h("span.right.dim", "same numbers, different code")),
          h("table.t", h("thead", h("tr", h("th", "quantity"), h("th.num", "solver (in-process)"), h("th.num", "verify.py"))),
            h("tbody", rowsCmp.map(([k, a, b]) => h("tr", h("td", k), h("td.num", a), h("td.num", b))))),
          v?.reasons?.length ? h("div.note.bad", v.reasons.join(" · ")) : null,
          h("div.note", "Both columns use the same definitions (worst row / column, relative); the engine's own stopping measure (relative L2 KKT) is on the Solve page. Tolerances live in one place (include/ps26119/tolerances.h): primal / dual / gap 1e-6 relative; certificates 1e-8 relative to their magnitude."))),
      h("div.panel", { style: { marginTop: "14px" } }, h("div.panel-h", "Raw verifier report (verify.json)"),
        h("div.console", { style: { height: "200px" } }, JSON.stringify(v || {}, null, 1))),
    );
  }

  const off = store.on((p) => { if ("run" in p) draw(); });
  draw();
  return off;
}
