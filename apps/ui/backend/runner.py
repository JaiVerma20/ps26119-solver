"""runner.py — solve jobs: run the real `ps26119` CLI, stream its log, read its solution file and
run the independent verifier (tools/verify.py). Nothing here computes or invents a result:
every number shown in the UI comes from the solver's own output or the verifier's report.
"""
from __future__ import annotations

import hashlib
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
    # generated models carry no names (empty strings in the file): show x<j> / r<i> instead
    res["columns"] = [{"i": j, "name": (names_c[j] if j < len(names_c) else "") or f"x{j}", "x": s.x[j], "z": s.z[j]}
                      for j in range(min(len(s.x), max_vector))]
    res["rows"] = [{"i": i, "name": (names_r[i] if i < len(names_r) else "") or f"r{i}", "activity": s.row_activity[i],
                    "y": s.y[i]} for i in range(min(len(s.y), max_vector))]
    res["vectors_truncated"] = len(s.x) > max_vector or len(s.y) > max_vector
    return res


def ranging_summary(path: str, stdout: str, max_rows: int = 2000) -> dict:
    """The CLI's --ranging CSV (cost and right-hand-side ranging at the optimal vertex), or why it
    was refused (the CLI prints 'ranging    refused: <reason>')."""
    import csv
    refused = next((ln.split("refused:", 1)[1].strip() for ln in stdout.splitlines() if ln.startswith("ranging") and "refused:" in ln), None)
    if refused or not os.path.exists(path):
        return {"ok": False, "message": refused or "no ranging file was written"}
    num = lambda v: _f(v)
    cost, rhs = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            item = {"i": int(r["index"]), "name": r["name"] or (("x" if r["kind"] == "cost" else "r") + r["index"]),
                    "value": num(r["value"]), "status": r["status"], "lower": num(r["lower"]), "upper": num(r["upper"]),
                    "dual": num(r["dual_or_reduced_cost"])}
            (cost if r["kind"] == "cost" else rhs).append(item)
    # what a planner reads first, before the lists are cut: binding rows by |dual|, basic columns by |cost|
    rhs.sort(key=lambda r: (r["status"] == "not_binding", -abs(r["dual"] or 0.0)))
    cost.sort(key=lambda c: (c["status"] != "basic", -abs(c["value"] or 0.0)))
    note = next((ln.split("(", 1)[1].rstrip(")") for ln in stdout.splitlines() if ln.startswith("ranging") and "(" in ln), "")
    return {"ok": True, "message": note, "costs": cost[:max_rows], "rows": rhs[:max_rows],
            "truncated": len(cost) > max_rows or len(rhs) > max_rows, "n_costs": len(cost), "n_rows": len(rhs)}


_supports_cache: dict = {}


def binary_supports(flag: str) -> bool:
    """Whether the solver binary's usage text lists `flag` (cached per binary mtime): an older build
    — e.g. one compiled before --ranging existed — must still run, without the option."""
    try:
        key = (paths.BIN, os.path.getmtime(paths.BIN), flag)
    except OSError:
        return False
    if key not in _supports_cache:
        try:
            p = subprocess.run([paths.BIN, "--help"], capture_output=True, text=True, timeout=30)
            _supports_cache[key] = flag in (p.stdout + p.stderr)
        except (OSError, subprocess.TimeoutExpired):
            _supports_cache[key] = False
    return _supports_cache[key]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Job:
    kind = "solve"

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
        if o.get("ranging") and binary_supports("--ranging"):
            cmd += ["--ranging", sol[:-4] + ".ranging.csv"]
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
            stderr_tail: list[str] = []
            for line in self.proc.stderr:
                line = line.rstrip("\n")
                stderr_tail = (stderr_tail + [line])[-20:]
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
            result["solution_file"] = paths.rel(sol)
            result["solution_sha256"] = sha256_file(sol)  # identifies the exact artifact the verifier checked
            try:
                result["solution"] = solution_summary(sol)
                if self.opts.get("ranging"):
                    result["ranging"] = ranging_summary(sol[:-4] + ".ranging.csv", stdout) if binary_supports("--ranging") else \
                        {"ok": False, "message": "this solver binary predates --ranging; rebuild it (cmake --build build)"}
            except Exception as e:  # noqa: BLE001 — report, never crash the server
                result["solution_error"] = str(e)
        known = generate.known(model_abs)
        if known:
            obj = (result.get("solution") or {}).get("objective")
            if isinstance(obj, float) and obj == obj and abs(obj) != float("inf"):
                known["rel_err"] = abs(obj - known["optimum"]) / (1 + abs(known["optimum"]))
            result["known"] = known
        self.emit("result", **result)
        if not os.path.exists(sol) and code in (2, 3, 4, 5):
            # no answer at all (usage / read / output error): say why instead of a silent empty page
            why = {2: "usage error", 3: "the model could not be read", 4: "the output could not be written", 5: "numerical error"}[code]
            detail = next((ln.strip() for ln in reversed(stderr_tail) if ln.strip() and not ln.startswith("  ")), "")
            self.emit("error", message=f"ps26119 exited with code {code} ({why}){': ' + detail if detail else ''}")
            return
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
        self.persist()

    def persist(self):
        """Keep the run on disk (apps/ui/.runs/jobs/<id>/job.json) so its certificate and report
        survive a server restart. Progress is thinned to the last 2000 points."""
        try:
            os.makedirs(self.dir, exist_ok=True)
            prog = [e for e in self.events if e["type"] == "progress"]
            keep = [e for e in self.events if e["type"] != "progress"] + prog[-2000:]
            keep.sort(key=lambda e: e.get("at", 0))
            with open(os.path.join(self.dir, "job.json.tmp"), "w") as f:
                json.dump({"id": self.id, "kind": self.kind, "model": self.model_rel, "created": self.created,
                           "opts": {k: v for k, v in self.opts.items() if k != "base"}, "events": keep}, f,
                          default=lambda v: None)
            os.replace(os.path.join(self.dir, "job.json.tmp"), os.path.join(self.dir, "job.json"))
        except (OSError, TypeError, ValueError):
            pass  # persistence is a convenience; the live run is unaffected


class StoredJob(Job):
    """A finished run read back from disk."""

    def __init__(self, d: dict):
        super().__init__(d.get("model", ""), d.get("opts", {}))
        self.id, self.kind, self.created = d["id"], d.get("kind", "solve"), d.get("created", 0)
        self.dir = os.path.join(paths.RUNS, "jobs", self.id)
        self.events, self.done = d.get("events", []), True

    def persist(self):
        pass


def load_job(jid: str) -> Job | None:
    p = os.path.join(paths.RUNS, "jobs", jid, "job.json")
    try:
        with open(p) as f:
            return StoredJob(json.load(f))
    except (OSError, ValueError, KeyError):
        return None


def run_summary(job: Job) -> dict:
    ev = {}
    for e in job.events:
        ev[e["type"]] = e
    sol = (ev.get("result") or {}).get("solution") or {}
    ver = (ev.get("verify") or {}).get("report") or {}
    return {"id": job.id, "kind": job.kind, "model": job.model_rel, "created": job.created, "done": job.done,
            "status": sol.get("status"), "objective": sol.get("objective"), "engine": sol.get("engine"),
            "seconds": sol.get("seconds"), "verdict": ver.get("verdict"), "algorithm": job.opts.get("algorithm"),
            "gpu": bool(job.opts.get("gpu")), "rows": (ev.get("result") or {}).get("rows"),
            # a run without an answer says why (older runs: the solver's exit code)
            "error": (ev.get("error") or {}).get("message") or (None if sol or not ev.get("result") else
                                                              f"ps26119 exited with code {ev['result'].get('exit_code')}")}


def list_runs(limit: int = 60) -> list[dict]:
    """Solve runs on disk, newest first (live runs are listed by Jobs.runs())."""
    root = os.path.join(paths.RUNS, "jobs")
    if not os.path.isdir(root):
        return []
    items = []
    for jid in os.listdir(root):
        p = os.path.join(root, jid, "job.json")
        if re.fullmatch(r"[0-9a-f]{12}", jid) and os.path.exists(p):
            items.append((os.path.getmtime(p), jid))
    out = []
    for _, jid in sorted(items, reverse=True)[:limit]:
        j = load_job(jid)
        if j and j.kind == "solve":
            out.append(run_summary(j))
    return out


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
            job = self.jobs.get(jid)
        return job or (load_job(jid) if re.fullmatch(r"[0-9a-f]{12}", jid) else None)


def validate_options(body: dict) -> tuple[dict | None, str | None]:
    """Options from the UI, checked the same way the CLI checks them."""
    o = {
        "algorithm": body.get("algorithm", "auto"), "precision": body.get("precision", "fp64"),
        "threads": body.get("threads", 0), "time_limit": body.get("time_limit", 60),
        "tol": body.get("tol"), "presolve": bool(body.get("presolve", True)), "gpu": bool(body.get("gpu", False)),
        "verify": bool(body.get("verify", True)), "ranging": bool(body.get("ranging", False)),
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


# Real-world reference solvers for the live comparison on the Solve page. Each runs as a separate
# tool process (tools/*_ref.py — tooling only, never linked into ps26119), writes our solution-file
# format, and its Optimal answer is judged by the same independent verifier as ours.
#   key: (label, tool, tool arguments, Python module it needs, solves LP, solves MILP, note)
REFERENCES = {
    "highs-simplex": ("HiGHS · dual simplex", "highs_ref.py", ["--solver", "simplex"], "highspy", True, False, ""),
    "highs-ipm": ("HiGHS · interior point", "highs_ref.py", ["--solver", "ipm"], "highspy", True, False, ""),
    "highs-pdlp": ("HiGHS · PDLP", "highs_ref.py", ["--solver", "pdlp"], "highspy", True, False, "CPU cuPDLP-C port, tolerance 1e-8"),
    "highs-mip": ("HiGHS · branch and cut", "highs_ref.py", ["--solver", "choose"], "highspy", False, True, ""),
    "ortools-glop": ("OR-Tools · GLOP simplex", "ortools_ref.py", ["--solver", "glop"], "ortools", True, False, ""),
    "ortools-pdlp": ("OR-Tools · PDLP", "ortools_ref.py", ["--solver", "pdlp"], "ortools", True, False, "Google's PDLP, tolerance 1e-8"),
    "scip": ("SCIP · SoPlex LP / branch and cut", "scip_ref.py", [], "pyscipopt", True, True,
             "LP with presolve off so that its duals can be verified (not SCIP's fastest LP setting)"),
    "coin-clp": ("COIN-OR CLP · dual simplex", "coin_ref.py", ["--solver", "clp"], "cylp", True, False, ""),
    "coin-cbc": ("COIN-OR CBC · branch and cut", "coin_ref.py", ["--solver", "cbc"], "cylp", False, True, ""),
}
# the first live reference (PR #18) took HiGHS engine names
REFERENCE_ALIASES = {"simplex": "highs-simplex", "ipm": "highs-ipm", "pdlp": "highs-pdlp", "choose": "highs-mip"}
_available_cache: dict | None = None


def available_references() -> dict:
    """{module: installed?} for the tool interpreter (checked once, in a subprocess: the tools may
    run under a different Python than the server)."""
    global _available_cache
    if _available_cache is None:
        mods = sorted({r[3] for r in REFERENCES.values()})
        code = "import importlib.util as u, json; print(json.dumps({m: u.find_spec(m) is not None for m in %r}))" % mods
        try:
            p = subprocess.run([paths.PYTHON, "-c", code], capture_output=True, text=True, timeout=60)
            _available_cache = json.loads(p.stdout.strip() or "{}")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            _available_cache = {m: False for m in mods}
    return _available_cache


def reference_list() -> list:
    av = available_references()
    return [{"key": k, "label": r[0], "tool": r[1], "module": r[3], "lp": r[4], "mip": r[5], "note": r[6],
             "available": bool(av.get(r[3]))} for k, r in REFERENCES.items()]


class ReferenceJob(Job):
    """The same model through a real-world solver (REFERENCES) and then, for an Optimal answer,
    through the same independent verifier as our runs — the live comparison on the Solve page."""
    kind = "reference"

    def _run(self):
        model_abs = paths.resolve_model(self.model_rel)
        os.makedirs(self.dir, exist_ok=True)
        sol = os.path.join(self.dir, "reference.sol")
        o = self.opts
        label, tool, args, _mod, _lp, _mip, note = REFERENCES[o["solver"]]
        cmd = [paths.PYTHON, os.path.join(paths.ROOT, "tools", tool), model_abs, sol, *args,
               "--time-limit", str(o["time_limit"]), "--json"]
        self.emit("started", command=f"python3 tools/{tool} {self.model_rel} {' '.join(args)}".rstrip(), stage="reference",
                  solver=o["solver"], label=label, note=note)
        t0 = time.time()
        try:  # a solver without an internal time limit (CLP) is stopped here
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=o["time_limit"] + 30)
        except subprocess.TimeoutExpired:
            self.emit("result", status="TimeLimit", objective=None, seconds=o["time_limit"], engine=o["solver"],
                      solver=o["solver"], label=label, wall_seconds=round(time.time() - t0, 3))
            self.emit("verify", skipped=True, reason="TimeLimit: no answer to verify")
            return
        try:
            res = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            self.emit("error", message=(p.stderr.strip().splitlines() or ["the reference run failed"])[-1][:300])
            return
        res.setdefault("version", res.get("highs_version") and f"HiGHS {res['highs_version']}" or
                       res.get("ortools_version") and f"OR-Tools {res['ortools_version']}" or "")
        self.emit("result", **res, solver=o["solver"], label=label, wall_seconds=round(time.time() - t0, 3))
        if os.path.exists(sol) and res.get("status") == "Optimal":
            self.emit("stage", stage="verify")
            self.emit("verify", **verify(model_abs, sol, self.dir))
        elif res.get("status") in ("Infeasible", "Unbounded"):
            self.emit("verify", skipped=True, reason=f"{res['status']}: this solver gives no certificate to check")
        else:
            self.emit("verify", skipped=True, reason=f"{res.get('status')}: no answer to verify")


def validate_reference(body: dict):
    solver = REFERENCE_ALIASES.get(body.get("solver", "highs-simplex"), body.get("solver", "highs-simplex"))
    if solver not in REFERENCES:
        return None, f"solver must be one of {', '.join(REFERENCES)}"
    if not available_references().get(REFERENCES[solver][3]):
        return None, f"{REFERENCES[solver][0]} is not installed (Python module {REFERENCES[solver][3]})"
    try:
        tl = float(body.get("time_limit", 120))
    except (TypeError, ValueError):
        return None, "time_limit must be a number"
    if not 0 < tl <= 3600:
        return None, "time_limit (0, 3600]"
    return {"solver": solver, "time_limit": tl}, None
