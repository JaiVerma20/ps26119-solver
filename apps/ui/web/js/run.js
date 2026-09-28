// run.js — start a solve and collect its live events into store.run (shared by every page).
import { api } from "./api.js";
import { store } from "./store.js";

export const DEFAULT_OPTS = { algorithm: "auto", precision: "fp64", threads: 0, time_limit: 60, tol: "", presolve: true, gpu: false };

export async function startSolve(path, opts) {
  if (store.run && !store.run.done) throw new Error("a solve is already running");
  const full = { ...DEFAULT_OPTS, ...opts, path };
  const job = await api.solve(full);
  const run = { job, path, opts: full, events: [], progress: [], logs: [], result: null, verify: null, stage: "solve",
    done: false, started: Date.now(), command: "" };
  store.set({ run });
  api.stream(job, (ev) => {
    const r = store.run;
    if (!r || r.job !== job) return;
    r.events.push(ev);
    if (ev.type === "started") r.command = ev.command;
    else if (ev.type === "progress") r.progress.push(ev);
    else if (ev.type === "log") r.logs.push(ev.line);
    else if (ev.type === "stage") r.stage = ev.stage;
    else if (ev.type === "result") { r.result = ev; r.stage = r.opts.verify === false ? "done" : "verify"; }
    else if (ev.type === "verify") { r.verify = ev; }
    else if (ev.type === "error") { r.error = ev.message; }
    if (ev.type === "end" || ev.type === "stream-error") {
      r.done = true;
      r.stage = "done";
      if (r.result) {
        const sol = r.result.solution || {};
        store.remember({ job, path, when: Date.now(), status: sol.status, objective: sol.objective, engine: sol.engine,
          seconds: sol.seconds, verdict: r.verify?.report?.verdict, rows: r.result.rows, cols: r.result.cols, nnz: r.result.nnz });
      }
    }
    store.set({ run: r });
  });
  return job;
}

// how the result should be described: Optimal answers are checked by the gate, Infeasible and
// Unbounded answers by their certificate
export function certificateStatus(result, verify) {
  const sol = result?.solution;
  if (!sol) return { label: "—", cls: "mut" };
  if (sol.status === "Infeasible" || sol.status === "Unbounded") {
    const cert = verify?.report?.certificate || "";
    const inproc = (result.check_line || "").startsWith("PASS");
    if (!inproc) return { label: "not certified", cls: "warn", detail: result.check_line };
    const kind = /exact rational/.test(cert) ? "exact rational" : /tolerance/.test(cert) ? "tolerance" : "checked";
    return { label: `${sol.status === "Infeasible" ? "Farkas" : "ray"} · ${kind}`, cls: "ok", detail: cert };
  }
  if (sol.status === "Optimal") {
    const b = sol.certified_bound;
    return Number.isFinite(b) ? { label: "rounding-proof bound", cls: "ok", detail: result.certified_line }
      : { label: "no finite bound", cls: "mut", detail: result.certified_line };
  }
  return { label: "—", cls: "mut" };
}
