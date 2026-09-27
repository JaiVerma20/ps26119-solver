// store.js — shared state: system info, evidence, the current / last solve (kept in this tab).
const listeners = new Set();
export const store = {
  system: null,
  evidence: null,
  selectedModel: sessionStorage.getItem("ps.model") || "data/netlib_small/afiro.mps",
  run: null,          // { job, opts, events[], progress[], result, verify, stage, done }
  history: JSON.parse(sessionStorage.getItem("ps.history") || "[]"),
  set(patch) {
    Object.assign(store, patch);
    if ("selectedModel" in patch) sessionStorage.setItem("ps.model", patch.selectedModel);
    for (const f of listeners) f(patch);
  },
  on(f) { listeners.add(f); return () => listeners.delete(f); },
  remember(entry) {
    store.history = [entry, ...store.history].slice(0, 25);
    sessionStorage.setItem("ps.history", JSON.stringify(store.history));
  },
};
