// pages/preflight.js — demo-day readiness: every dependency checked for real, with the fix.
import { api } from "../api.js";
import { h, toast } from "../util.js";

const LED = { ok: "on", info: "off", warn: "amber", fixable: "amber", fail: "bad" };
const WORD = { ok: "ready", info: "note", warn: "check", fixable: "fixable", fail: "blocked" };

export function mount(root) {
  const body = h("div");
  const recheck = h("button.btn", { onclick: () => load() }, "Re-check");
  const prep = h("button.btn.primary", { onclick: async () => {
    prep.disabled = true; prep.textContent = "preparing…";
    try { const r = await api.prepare(); draw(r); toast(r.generated.length ? r.generated.map((g) => g.message).join(" · ") : "nothing to prepare"); }
    catch (e) { toast(e.message); } finally { prep.disabled = false; prep.textContent = "Prepare demo models"; }
  } }, "Prepare demo models");
  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "10 · system check"), h("h1", "Demo preflight"),
      h("p", "Run this before going on stage. Each line is checked for real: the solver binary is executed, the verifier's Python modules are imported, the demo models are opened, the evidence CSVs are parsed.")),
      h("div.actions", recheck, prep)),
    body);

  function draw(r) {
    const banner = { ok: ["ok", "Ready for the demo"], warn: ["warn", "Ready, with notes below"], fail: ["bad", "Not ready — fix the blocked items"] }[r.state];
    body.replaceChildren(
      h(`div.panel.preflight-banner.${banner[0]}`, h("div.panel-b", h(`span.led.${LED[r.state === "fail" ? "fail" : r.state === "warn" ? "warn" : "ok"]}`), h("b", banner[1]),
        h("span.dim", ` · ${r.cpu} · ${r.cores} cores`))),
      h("div.panel", h("table.t", h("thead", h("tr", h("th", ""), h("th", "check"), h("th", "detail"), h("th", "fix"))),
        h("tbody", r.items.map((i) => h("tr", h("td", h(`span.led.${LED[i.state]}`), h("span.mono", { class: i.state === "fail" ? "bad" : i.state === "ok" ? "ok" : "warn" }, ` ${WORD[i.state]}`)),
          h("td", i.label), h("td.mono.dim", { style: { whiteSpace: "normal" } }, i.detail), h("td.mono", i.fix || "")))))),
      h("div.note", "Start the UI with: python3 apps/ui/server.py (then open http://127.0.0.1:8765). The jury demo is under Demo, keys ← → to move, Enter to run a step, F for full screen."));
  }
  async function load() {
    body.replaceChildren(h("div.panel", h("div.empty", "checking…")));
    try { draw(await api.preflight()); } catch (e) { body.replaceChildren(h("div.panel", h("div.panel-b.bad", e.message))); }
  }
  load();
}
