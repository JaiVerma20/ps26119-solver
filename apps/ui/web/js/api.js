// api.js — the only place that talks to the backend (apps/ui/server.py).
async function get(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

export const api = {
  system: () => get("/api/system"),
  models: () => get("/api/models"),
  modelInfo: (path) => get(`/api/model/info?path=${encodeURIComponent(path)}`),
  sparsity: (path) => get(`/api/model/sparsity?path=${encodeURIComponent(path)}`),
  evidence: () => get("/api/evidence"),
  scenarios: () => get("/api/scenarios"),
  gpu: () => get("/api/gpu"),
  compare: () => get("/api/compare"),
  preflight: () => get("/api/preflight"),
  async prepare() {
    const r = await fetch("/api/preflight/prepare", { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  },
  runs: () => get("/api/runs"),
  certificate: (job) => get(`/api/jobs/${job}/certificate`),
  // the self-contained HTML report (certificates of these runs + optionally the evidence)
  reportUrl: (jobs, { evidence = true, download = false, title = "" } = {}) =>
    `/api/report?jobs=${jobs.join(",")}&evidence=${evidence ? 1 : 0}${download ? "&download=1" : ""}${title ? `&title=${encodeURIComponent(title)}` : ""}`,
  async solve(opts) {
    const r = await fetch("/api/solve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j.job;
  },
  async generate(kind, size, seed = 1) {
    const r = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, size, seed }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  },
  cancel: (job) => fetch(`/api/jobs/${job}/cancel`, { method: "POST" }),
  // live events of a job (server-sent events); returns a close() function
  stream(job, onEvent) {
    const es = new EventSource(`/api/jobs/${job}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "end") es.close();
      onEvent(ev);
    };
    es.onerror = () => { es.close(); onEvent({ type: "stream-error" }); };
    return () => es.close();
  },
  async upload(file) {
    const r = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j.path;
  },
};
