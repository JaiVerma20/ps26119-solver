// pages/dashboard.js — what the solver is, what it has proven (from committed evidence), one-click demos.
import { store } from "../store.js";
import { h, fnum, fsec, fratio, fint, srcChip, statusBadge, passBadge, toast } from "../util.js";
import { startSolve } from "../run.js";
import { api } from "../api.js";

// one-click scenarios: real models in the repository, real solves
const SCENARIOS = [
  { title: "Netlib AFIRO", sub: "classic LP · 27 × 32 · simplex vertex + independent verification",
    path: "data/netlib_small/afiro.mps", opts: { algorithm: "auto" } },
  { title: "Refinery year, hourly", sub: "429,240 rows · 1.5M nonzeros · r²HPDHG on all cores",
    path: "bench/generated/refinery-T8760-s1.lpm", opts: { algorithm: "auto", threads: 0 }, gen: ["refinery", 8760, 1] },
  { title: "Prove infeasibility", sub: "the answer carries a Farkas certificate, checked in exact arithmetic",
    path: "data/examples/infeasible.mps", opts: { algorithm: "simplex" } },
  { title: "Prove unboundedness", sub: "a feasible point plus an improving ray, both checked",
    path: "data/examples/unbounded.mps", opts: { algorithm: "r2hpdhg" } },
  { title: "Refinery blending", sub: "small crude-blending LP with named products",
    path: "data/examples/refinery_blend.mps", opts: { algorithm: "auto" } },
  { title: "MILP · MIPLIB gt2", sub: "branch-and-bound, pseudocosts + diving; integer answer checked",
    path: "data/mip_small/gt2.lpm", opts: { algorithm: "auto" } },
];

export function mount(root) {
  const claims = h("div.grid.g4");
  const sysCard = h("div.hero-card");
  const recent = h("div.panel-b.tight");

  root.append(
    h("div.hero",
      h("div.hero-card",
        h("div.eyebrow", "SIH 2026 · PS 26119 · MRPL"),
        h("h2", "Indigenous LP / MILP solver with verified answers"),
        h("p", "Written from scratch in C++20: a GPU-native r²HPDHG engine for large refinery LPs, a sparse revised simplex for exact vertices, branch-and-bound for MILP. Every answer is checked on the original model before it is shown — Optimal by the KKT gate, Infeasible and Unbounded by a mathematical certificate — and again by an independent verifier."),
        h("div.pills",
          h("span.badge.acc", "no third-party solver inside"), h("span.badge.ok", "certified Infeasible / Unbounded"),
          h("span.badge.acc", "CUDA r²HPDHG · mixed precision"), h("span.badge.mut", "deterministic · reproducible")),
        h("svg.deco", { width: 260, height: 150, viewBox: "0 0 260 150", html:
          '<path d="M10 140 L70 60 L120 110 L180 30 L250 70" stroke="#5ce1e6" stroke-opacity=".35" stroke-width="2" fill="none"/><path d="M10 140 L60 100 L120 125 L170 70 L250 95" stroke="#a78bfa" stroke-opacity=".25" stroke-width="2" fill="none"/>' })),
      sysCard),
    h("div.page-head", { style: { marginTop: "4px" } }, h("div", h("div.eyebrow", "measured"), h("h1", "What it has proven"),
      h("p", "Recomputed on every load from the committed benchmark CSVs (the same rules as docs/EVIDENCE.md). Hover a source tag for commit, machine and date.")),
      h("div.actions", h("a.btn", { href: "#/bench" }, "all benchmarks →"))),
    claims,
    h("div.page-head", { style: { marginTop: "22px" } }, h("div", h("div.eyebrow", "live"), h("h1", "Run a scenario"),
      h("p", "Real models from the repository, solved by the real binary on this machine, then independently verified."))),
    h("div.grid.g3", SCENARIOS.map((sc) => h("div.claim", { style: { cursor: "pointer" }, onclick: () => launch(sc) },
      h("div.t", sc.title), h("div.d", sc.sub),
      h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px" } },
        h("span.src-chip", sc.path), h("span.acc.mono", "run ▸"))))),
    h("div.panel", { style: { marginTop: "22px" } }, h("div.panel-h", "Recent runs in this session", h("span.right.dim", "kept in this browser tab")), recent),
  );

  async function launch(sc) {
    if (sc.gen) {  // generated models are git-ignored: build it (bench/generate_*.py, ~2 s) on first use
      try {
        toast(`preparing ${sc.path.split("/").pop()} …`);
        const g = await api.generate(...sc.gen);
        if (g.created) toast(g.message);
      } catch (e) { toast(e.message); return; }
    }
    store.set({ selectedModel: sc.path });
    sessionStorage.setItem("ps.opts", JSON.stringify({ ...JSON.parse(sessionStorage.getItem("ps.opts") || "{}"), ...sc.opts, gpu: false }));
    location.hash = "#/solve";
    try { await startSolve(sc.path, { ...JSON.parse(sessionStorage.getItem("ps.opts")) }); } catch (e) { toast(e.message); }
  }

  function drawSystem() {
    const s = store.system;
    if (!s) { sysCard.replaceChildren(h("div.muted", "reading system…")); return; }
    sysCard.replaceChildren(
      h("div.eyebrow", "this machine · this build"),
      h("dl.kv", { style: { marginTop: "10px" } },
        h("dt", "solver"), h("dd", s.version),
        h("dt", "binary"), h("dd", s.binary),
        h("dt", "cpu"), h("dd", `${s.cpu} · ${s.cores} cores`),
        h("dt", "gpu backend"), h("dd", s.cuda_build ? h("span.ok", `CUDA · ${s.gpu}`) : h("span.muted", "not in this build (CPU-only Mac build)")),
        h("dt", "os"), h("dd", s.os),
        h("dt", "evidence"), h("dd", store.evidence?.netlib?.[0] ? `CPU @ ${store.evidence.netlib[0].source.git_hash}` + (store.evidence.gpu?.[0] ? ` · GPU @ ${store.evidence.gpu[0].source.git_hash}` : "") : "…")),
      h("div", { style: { marginTop: "12px", display: "flex", gap: "8px" } }, h("a.btn.primary", { href: "#/solve" }, "Open solver ▸"), h("a.btn", { href: "#/verify" }, "Verification")));
  }

  function drawClaims() {
    const ev = store.evidence;
    if (!ev) { claims.replaceChildren(h("div.muted", "reading committed evidence…")); return; }
    const out = [];
    const auto = ev.netlib.find((n) => n.engine === "auto");
    if (auto) out.push(h("div.claim.okb", h("div.big", auto.solved, h("small", ` / ${auto.total}`)), h("div.t", "Netlib LPs solved & verified"),
      h("div.d", `all 93 classic LPs, auto engine, 60 s each; equal to HiGHS to 1e-6`), h("div.src", srcChip(auto.source))));
    const sx = ev.infeasible.find((x) => x.engine === "simplex"), pd = ev.infeasible.find((x) => x.engine === "r2hpdhg");
    if (sx) out.push(h("div.claim.vio", h("div.big", sx.certified, h("small", ` / ${sx.total}`)), h("div.t", "Infeasible LPs proven, not just detected"),
      h("div.d", `Farkas certificates checked in-process and by the verifier; ${sx.exact_rational + (pd?.exact_rational || 0)} in exact rational arithmetic (simplex ${sx.certified}, r²HPDHG ${pd?.certified ?? "—"})`), h("div.src", srcChip(sx.source))));
    const cpu = ev.scale_cpu;
    const ref = cpu?.rows.filter((r) => r.instance === "refinery-T8760-s1" && r.status === "Optimal" && r.s_1e8);
    if (ref?.length) {
      const best = ref.reduce((a, b) => (b.s_1e8 < a.s_1e8 ? b : a));
      out.push(h("div.claim", h("div.big", fsec(best.s_1e8)), h("div.t", "Refinery planning year, hourly"),
        h("div.d", `${fint(best.rows)} rows · ${fint(best.nnz)} nnz to 1e-8 on the CPU (${best.threads} threads, ${best.precision}); known optimum matched to ${best.rel_err_known?.toExponential(1)}`), h("div.src", srcChip(cpu.source))));
    }
    const g = ev.gpu?.[0];
    const gp = g?.pairs.find((p) => p.instance === "refinery-T8760-s1" && p.precision === "fp64");
    if (gp) out.push(h("div.claim.warnb", h("div.big", fratio(gp.ratio_vs_best)), h("div.t", "GPU vs the fastest CPU configuration"),
      h("div.d", `refinery year on ${g.gpu}: ${fsec(gp.gpu_s)} vs ${fsec(gp.best_cpu_s)} (${gp.best_cpu_threads} CPU threads); sanitizer ${g.sanitizer.memcheck}/${g.sanitizer.racecheck}`), h("div.src", srcChip(g.source))));
    if (ev.miplib) out.push(h("div.claim.okb", h("div.big", ev.miplib.solved, h("small", ` / ${ev.miplib.total}`)), h("div.t", "MILP: small MIPLIB 3 proven optimal"),
      h("div.d", "branch-and-bound with certified node bounds, pseudocosts and diving; 300 s each"), h("div.src", srcChip(ev.miplib.source))));
    if (ev.small_netlib) out.push(h("div.claim.okb", h("div.big", ev.small_netlib.verified, h("small", ` / ${ev.small_netlib.total}`)), h("div.t", "Every engine × precision verified"),
      h("div.d", "10 Netlib LPs × oracle, simplex, PDLP, r²HPDHG × fp64 / mixed, independent verifier"), h("div.src", srcChip(ev.small_netlib.source))));
    const big = ev.gpu?.[0]?.pairs.find((p) => p.instance === "rand-1000000-s1" && p.precision === "fp64");
    if (big) out.push(h("div.claim.warnb", h("div.big", fratio(big.ratio_vs_best)), h("div.t", "1M-row LP on the GPU"),
      h("div.d", `${fint(big.rows)} rows · ${fint(big.nnz)} nnz: GPU ${fsec(big.gpu_s)} vs ${fsec(big.best_cpu_s)} on ${big.best_cpu_threads} CPU threads`), h("div.src", srcChip(g.source))));
    const simplexNet = ev.netlib.find((n) => n.engine === "simplex");
    if (simplexNet) out.push(h("div.claim", h("div.big", simplexNet.solved, h("small", ` / ${simplexNet.total}`)), h("div.t", "Netlib with the simplex alone"),
      h("div.d", `sparse revised primal simplex; not solved: ${simplexNet.not_solved.join(", ") || "—"}`), h("div.src", srcChip(simplexNet.source))));
    claims.replaceChildren(...out);
    drawSystem();
  }

  function drawRecent() {
    if (!store.history.length) { recent.replaceChildren(h("div.empty", "no runs yet — try a scenario above")); return; }
    recent.replaceChildren(h("table.t", h("thead", h("tr", h("th", "model"), h("th", "status"), h("th", "engine"),
      h("th.num", "rows × cols"), h("th.num", "objective"), h("th.num", "time"), h("th", "independent verify"))),
      h("tbody", store.history.map((r) => h("tr", h("td.mono", r.path.split("/").pop()), h("td", statusBadge(r.status)), h("td.mono", r.engine || "—"),
        h("td.num", `${fint(r.rows)} × ${fint(r.cols)}`), h("td.num", fnum(r.objective, 10)), h("td.num", fsec(r.seconds)), h("td", passBadge(r.verdict)))))));
  }

  const off = store.on((p) => { if ("evidence" in p) drawClaims(); if ("system" in p) drawSystem(); if ("run" in p) drawRecent(); });
  drawSystem(); drawClaims(); drawRecent();
  return off;
}
