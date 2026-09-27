// app.js — shell: router, top-bar system status, status bar.
import { api } from "./api.js";
import { store } from "./store.js";
import { h, $, fsec, fnum } from "./util.js";
import * as dashboard from "./pages/dashboard.js";
import * as models from "./pages/models.js";
import * as solve from "./pages/solve.js";
import * as verify from "./pages/verify.js";
import * as bench from "./pages/bench.js";
import * as scenarios from "./pages/scenarios.js";

const PAGES = { dashboard, models, solve, verify, bench, scenarios };
let current = null, unmount = null;

function route() {
  const name = (location.hash.replace(/^#\//, "").split("?")[0]) || "dashboard";
  const page = PAGES[name] || dashboard;
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
  if (e.target.matches("input, select, textarea")) return;
  const map = { 1: "dashboard", 2: "models", 3: "solve", 4: "verify", 5: "bench", 6: "scenarios" };
  if (map[e.key] && !e.metaKey && !e.ctrlKey) location.hash = `#/${map[e.key]}`;
});

api.system().then((s) => store.set({ system: s })).catch((e) => {
  $("#sys-status").replaceChildren(h("span.chip", h("span.led.bad"), `backend unreachable: ${e.message}`));
});
api.evidence().then((ev) => store.set({ evidence: ev })).catch(() => {});
route();
