// app.js — shell: router, top-bar system status, status bar.
import { api } from "./api.js";
import { store } from "./store.js";
import { h, $, fsec, fnum } from "./util.js";
import * as dashboard from "./pages/dashboard.js";
import * as coverage from "./pages/coverage.js";
import * as models from "./pages/models.js";
import * as solve from "./pages/solve.js";
import * as verify from "./pages/verify.js";
import * as certificate from "./pages/certificate.js";
import * as scenarios from "./pages/scenarios.js";
import * as gpu from "./pages/gpu.js";
import * as bench from "./pages/bench.js";
import * as compare from "./pages/compare.js";
import * as preflight from "./pages/preflight.js";
import * as demo from "./pages/demo.js";

// The page registry: the rail, the keyboard shortcuts (1–9 and 0, in this order) and the router all come
// from this list. A new page = one module with mount(root) → optional cleanup, plus one line here.
const ICON = {
  dashboard: "M3 13h8V3H3zm10 8h8V11h-8zM3 21h8v-6H3zm10-18v6h8V3z",
  coverage: "M4 5h2v2H4zm4 0h12v2H8zM4 11h2v2H4zm4 0h12v2H8zm-4 6h2v2H4zm4 0h12v2H8z",
  models: "M4 4h4v4H4zm6 0h4v4h-4zm6 0h4v4h-4zM4 10h4v4H4zm6 0h4v4h-4zm6 0h4v4h-4zM4 16h4v4H4zm6 0h4v4h-4zm6 0h4v4h-4z",
  solve: "M8 5v14l11-7z",
  verify: "M12 2 4 5v6c0 5 3.4 9.7 8 11 4.6-1.3 8-6 8-11V5zm-1.2 14.2-3.5-3.5 1.4-1.4 2.1 2.1 4.8-4.8 1.4 1.4z",
  certificate: "M6 2h9l5 5v15H6zm8 1.5V8h4.5zM8 12h8v1.6H8zm0 3.4h8V17H8zm0 3.3h5v1.6H8z",
  scenarios: "M3 17h4l4-10 4 6 3-3h3v2h-2l-4 4-3-4-4 9H3z",
  gpu: "M4 6h16v12H4zm2 2v8h12V8zm2 2h3v4H8zm5 0h3v4h-3zM2 9h2v2H2zm0 4h2v2H2zm18-4h2v2h-2zm0 4h2v2h-2z",
  bench: "M4 20V10h3v10zm6.5 0V4h3v16zM17 20v-7h3v7z",
  compare: "M3 20h4V9H3zm7 0h4V4h-4zm7 0h4v-7h-4zM2 22h20v-1.5H2z",
  preflight: "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z",
};
const PAGES = [
  ["dashboard", "Dashboard", dashboard], ["coverage", "PS coverage", coverage], ["models", "Models", models], ["solve", "Solve", solve],
  ["verify", "Verification", verify], ["certificate", "Certificate", certificate], ["scenarios", "Scenarios", scenarios],
  ["gpu", "GPU", gpu], ["bench", "Benchmarks", bench], ["compare", "vs HiGHS", compare], ["preflight", "System check", preflight],
];
const HIDDEN = { demo };  // routable, not in the rail (full-screen)
const BY_NAME = Object.fromEntries([...PAGES.map(([k, , m]) => [k, m]), ...Object.entries(HIDDEN)]);
let current = null, unmount = null;

function renderRail() {
  $(".rail").replaceChildren(...PAGES.map(([k, label], i) => h("a", { href: `#/${k}`, "data-page": k, title: `${label} (${(i + 1) % 10})` },
    h("span.ico", { html: `<svg viewBox="0 0 24 24"><path d="${ICON[k]}"/></svg>` }), h("span", label))));
}

function route() {
  const name = (location.hash.replace(/^#\//, "").split("?")[0]) || "dashboard";
  const page = BY_NAME[name] || dashboard;
  document.querySelectorAll(".rail a").forEach((a) => a.classList.toggle("active", a.dataset.page === name));
  if (unmount) unmount();
  const main = $("#main");
  main.replaceChildren();
  current = page;
  unmount = page.mount(main) || null;
  main.scrollTop = 0;
}

function renderStatus() {
  const s = store.system;
  const box = $("#sys-status");
  if (!s) return;
  const run = store.run;
  const busy = run && !run.done;
  box.replaceChildren(
    h("a.btn.primary.demo-btn", { href: "#/demo", title: "Jury demo mode (full screen, keyboard driven)" }, "▶ Jury demo"),
    h("span.chip", { title: s.version }, h("span.led.on"), "solver ", h("b", { style: { color: "var(--text)" } }, s.git_hash)),
    h("span.chip", { title: s.os }, h("span.led.on"), `${s.cpu} · ${s.cores} cores`),
    h("span.chip", { title: s.cuda_build ? "this build has the CUDA backend" : s.cuda_probe || "no CUDA backend in this build" },
      h(`span.led.${s.cuda_build ? "on" : "off"}`), s.cuda_build ? `CUDA · ${s.gpu}` : "CPU build (no CUDA)"),
    h("span.chip", h(`span.led.${busy ? "busy" : "off"}`), busy ? `solving · ${run.stage}` : "idle"),
  );
  $("#sb-build").textContent = `${s.version} · ${s.binary}`;
}

function renderStatusbar() {
  const r = store.run, el = $("#sb-run");
  if (!r) { el.textContent = "no run yet"; return; }
  const sol = r.result?.solution;
  const name = r.path.split("/").pop();
  if (!r.done && !sol) { el.replaceChildren(h("span.acc", "● "), `solving ${name} …`); return; }
  const v = r.verify?.report?.verdict;
  el.replaceChildren(
    `last run: ${name} · `, h(sol?.status === "Optimal" ? "span.ok" : "span.warn", sol?.status || "no result"),
    sol ? ` · ${sol.engine} · obj ${fnum(sol.objective)} · ${fsec(sol.seconds)}` : "",
    v ? h(v === "PASS" ? "span.ok" : "span.bad", ` · verify ${v}`) : "",
  );
}

store.on((patch) => {
  if ("system" in patch || "run" in patch) renderStatus();
  if ("run" in patch) renderStatusbar();
});

window.addEventListener("hashchange", route);
window.addEventListener("keydown", (e) => {
  if (e.target?.matches?.("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  const n = e.key === "0" ? 10 : Number(e.key);  // 1–9, 0 = tenth page
  if (n >= 1 && n <= PAGES.length) location.hash = `#/${PAGES[n - 1][0]}`;
});

api.system().then((s) => store.set({ system: s })).catch((e) => {
  $("#sys-status").replaceChildren(h("span.chip", h("span.led.bad"), `backend unreachable: ${e.message}`));
});
api.evidence().then((ev) => store.set({ evidence: ev })).catch(() => {});
renderRail();
route();
