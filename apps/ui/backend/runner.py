"""runner.py — solve jobs: run the real `ps26119` CLI, stream its log, read its solution file and
run the independent verifier (tools/verify.py). Nothing here computes or invents a result:
every number shown in the UI comes from the solver's own output or the verifier's report.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

from . import generate, paths

sys.path.insert(0, os.path.join(paths.ROOT, "tools"))
from lpm import read_solution  # noqa: E402  (pure Python, no highspy)

ALGORITHMS = ("auto", "simplex", "r2hpdhg", "pdlp", "oracle")
PRECISIONS = ("fp64", "mixed")

# -vv progress lines of each engine (stderr)
RE_PDHG = re.compile(r"^(r2hpdhg|pdlp)\s+it\s+(\d+)\s+t\s+([\d.]+)s\s+pobj\s+(\S+)\s+dobj\s+(\S+)\s+rp\s+(\S+)\s+rd\s+(\S+)"
                     r"\s+gap\s+(\S+)\s+(\S+)")
RE_SIMPLEX = re.compile(r"^\s*iter\s+(\d+)\s+phase\s+(\d)\s+objective\s+(\S+)\s+infeasibility\s+(\S+)\s+([\d.]+)s")
RE_BB = re.compile(r"^bb node (\d+) incumbent (\S+)")
RE_MODEL = re.compile(r"^model\s+(.*?)\s+rows (\d+)\s+cols (\d+)\s+nnz (\d+)\s+fingerprint (\S+)")


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_progress(line: str) -> dict | None:
    m = RE_PDHG.match(line)
    if m:
        return {"kind": "pdhg", "engine": m[1], "iter": int(m[2]), "t": float(m[3]), "pobj": _f(m[4]),
                "dobj": _f(m[5]), "rp": _f(m[6]), "rd": _f(m[7]), "gap": _f(m[8]), "precision": m[9]}
    m = RE_SIMPLEX.match(line)
    if m:
        return {"kind": "simplex", "iter": int(m[1]), "phase": int(m[2]), "obj": _f(m[3]), "infeas": _f(m[4]),
                "t": float(m[5])}
    m = RE_BB.match(line)
    if m:
        return {"kind": "bb", "node": int(m[1]), "incumbent": _f(m[2])}
    return None


def parse_stdout(text: str) -> dict:
    """The CLI's key/value report (model size line, check line, certified line)."""
    out: dict = {}
    for line in text.splitlines():
        m = RE_MODEL.match(line)
        if m:
            out.update(name=m[1], rows=int(m[2]), cols=int(m[3]), nnz=int(m[4]), fingerprint=m[5])
        elif line.startswith("check "):
            out["check_line"] = line[len("check"):].strip()
        elif line.startswith("certified "):
            out["certified_line"] = line[len("certified"):].strip()
        elif line.startswith("engine "):
            out["engine_line"] = line[len("engine"):].strip()
    return out


def solution_summary(path: str, max_vector: int = 2000) -> dict:
    """Header of a solution file plus (at most max_vector) primal/dual entries for the tables."""
    s = read_solution(path)
    h = dict(s.header)
    num = lambda k: _f(h.get(k))
    res = {
        "status": h.get("status"), "engine": h.get("engine"), "precision": h.get("precision"),
        "objective": num("objective"), "dual_objective": num("dual_objective"),
        "certified_bound": num("certified_bound"), "primal_residual": num("primal_residual"),
        "dual_residual": num("dual_residual"), "gap": num("gap"), "iterations": num("iterations"),
        "seconds": num("seconds"), "setup_seconds": num("setup_seconds"),
        "iterations_to_fast": num("iterations_to_fast"), "seconds_to_fast": num("seconds_to_fast"),
        "check": h.get("check", ""), "message": h.get("message", ""), "fingerprint": h.get("model"),
        "n_cols": len(s.x), "n_rows": len(s.y),
        "has_dual_ray": bool(s.dual_ray), "has_primal_ray": bool(s.primal_ray),
    }
    names_c = getattr(s, "col_names", None) or []
    names_r = getattr(s, "row_names", None) or []
    res["columns"] = [{"i": j, "name": names_c[j] if j < len(names_c) else f"x{j}", "x": s.x[j], "z": s.z[j]}
                      for j in range(min(len(s.x), max_vector))]
    res["rows"] = [{"i": i, "name": names_r[i] if i < len(names_r) else f"r{i}", "activity": s.row_activity[i],
                    "y": s.y[i]} for i in range(min(len(s.y), max_vector))]
    res["vectors_truncated"] = len(s.x) > max_vector or len(s.y) > max_vector
    return res


class Job:
    def __init__(self, model_rel: str, opts: dict):
        self.id = uuid.uuid4().hex[:12]
        self.model_rel = model_rel
        self.opts = opts
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()
        self.dir = os.path.join(paths.RUNS, "jobs", self.id)
        self.proc: subprocess.Popen | None = None
        self.created = time.time()

    def emit(self, event_type: str, /, **data):
        with self.cond:
            self.events.append({**data, "type": event_type, "at": round(time.time() - self.created, 3)})
            self.cond.notify_all()

    def command(self, model_abs: str, sol: str) -> list[str]:
        o = self.opts
        cmd = [paths.BIN, "solve", model_abs, "--algorithm", o["algorithm"], "--precision", o["precision"],
               "--threads", str(o["threads"]), "--time-limit", str(o["time_limit"]), "--out", sol, "-vv"]
        if o.get("tol"):
            cmd += ["--tol", str(o["tol"])]
        if not o.get("presolve", True):
            cmd.append("--no-presolve")
        if o.get("gpu"):
            cmd.append("--gpu")
        return cmd

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 — a job must always end, with the reason visible
            self.emit("error", message=f"internal error: {type(e).__name__}: {e}")
        finally:
            self.finish()

    def _run(self):
        model_abs = paths.resolve_model(self.model_rel)
        os.makedirs(self.dir, exist_ok=True)
        sol = os.path.join(self.dir, "solution.sol")
        cmd = self.command(model_abs, sol)
        shown = ["ps26119"] + [paths.rel(c) if os.path.isabs(c) and c.startswith(paths.ROOT) else c for c in cmd[1:]]
        self.emit("started", command=" ".join(shown), stage="solve")
        t0 = time.time()
        stdout_lines: list[str] = []
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

            def pump_stdout():
                for line in self.proc.stdout:
                    stdout_lines.append(line.rstrip("\n"))

            th = threading.Thread(target=pump_stdout, daemon=True)
            th.start()
            last = 0.0
            for line in self.proc.stderr:
                line = line.rstrip("\n")
                p = parse_progress(line)
                if p:
                    # throttle very chatty logs (the simplex prints every 1000 iterations anyway)
                    now = time.time()
                    if p["kind"] != "pdhg" or now - last > 0.004:
                        self.emit("progress", **p)
                        last = now
                else:
                    self.emit("log", line=line)
            code = self.proc.wait()
            th.join(timeout=5)
            self.proc.stdout.close()
            self.proc.stderr.close()
        except OSError as e:
            self.emit("error", message=f"cannot run the solver: {e}")
            return
        wall = time.time() - t0
        stdout = "\n".join(stdout_lines)
        info = parse_stdout(stdout)
        result = {"exit_code": code, "wall_seconds": round(wall, 3), "stdout": stdout, **info,
                  "backend": "gpu" if self.opts.get("gpu") else "cpu", "threads": self.opts["threads"]}
        if os.path.exists(sol):
            try:
                result["solution"] = solution_summary(sol)
            except Exception as e:  # noqa: BLE001 — report, never crash the server
                result["solution_error"] = str(e)
        known = generate.known(model_abs)
        if known:
            obj = (result.get("solution") or {}).get("objective")
            if isinstance(obj, float) and obj == obj and abs(obj) != float("inf"):
                known["rel_err"] = abs(obj - known["optimum"]) / (1 + abs(known["optimum"]))
            result["known"] = known
        self.emit("result", **result)
        status = (result.get("solution") or {}).get("status")
        if status and status not in ("Optimal", "Infeasible", "Unbounded"):
            # a limit or an error is not an answer: there is nothing to verify (verify.py would say FAIL)
            self.emit("verify", skipped=True, reason=f"{status}: no answer to verify")
        elif os.path.exists(sol) and self.opts.get("verify", True):
            self.emit("stage", stage="verify")
            self.emit("verify", **verify(model_abs, sol, self.dir))

    def finish(self):
        with self.cond:
            self.done = True
            self.cond.notify_all()


def verify(model_abs: str, sol: str, workdir: str) -> dict:
    """tools/verify.py: independent reader (highspy) + independent checks; its JSON report."""
    out = os.path.join(workdir, "verify.json")
    t0 = time.time()
    p = subprocess.run([paths.PYTHON, os.path.join(paths.ROOT, "tools", "verify.py"), model_abs, sol, "--json", out,
                        "--quiet"], capture_output=True, text=True, timeout=1800)
    rep = {}
    if os.path.exists(out):
        with open(out) as f:
            rep = json.load(f)
    return {"exit_code": p.returncode, "seconds": round(time.time() - t0, 3), "report": rep,
            "stderr": p.stderr[-2000:]}


class Jobs:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def start(self, model_rel: str, opts: dict) -> Job:
        return self.add(Job(model_rel, opts))

    def add(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        return job

    def get(self, jid: str) -> Job | None:
        with self.lock:
            return self.jobs.get(jid)


def validate_options(body: dict) -> tuple[dict | None, str | None]:
    """Options from the UI, checked the same way the CLI checks them."""
    o = {
        "algorithm": body.get("algorithm", "auto"), "precision": body.get("precision", "fp64"),
        "threads": body.get("threads", 0), "time_limit": body.get("time_limit", 60),
        "tol": body.get("tol"), "presolve": bool(body.get("presolve", True)), "gpu": bool(body.get("gpu", False)),
        "verify": bool(body.get("verify", True)),
    }
    if o["algorithm"] not in ALGORITHMS:
        return None, f"unknown algorithm {o['algorithm']!r}"
    if o["precision"] not in PRECISIONS:
        return None, f"unknown precision {o['precision']!r}"
    try:
        o["threads"] = int(o["threads"])
        o["time_limit"] = float(o["time_limit"])
        o["tol"] = float(o["tol"]) if o["tol"] not in (None, "") else None
    except (TypeError, ValueError):
        return None, "threads, time_limit and tol must be numbers"
    if not (0 <= o["threads"] <= 4096) or not (0 < o["time_limit"] <= 86400) or (o["tol"] is not None and not o["tol"] > 0):
        return None, "threads 0-4096, time_limit (0, 86400], tol > 0"
    return o, None
