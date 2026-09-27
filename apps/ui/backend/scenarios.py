"""scenarios.py — planner what-ifs on the refinery planning LP (bench/generate_refinery_lp.py):
lever changes (prices, crude availability, demand, unit capacities) written as a new model, then

  what-if : the scenario solved cold and warm-started from the base solution (--warm), both
            checked by tools/verify.py — the re-solve cost planners actually pay (CLAUDE.md §9.3);
  sweep   : K values of one lever solved together by `ps26119 batch` (one SpMM per iteration,
            CLAUDE.md §9.4), optionally also one by one for an honest timing comparison.

Marginal values are the solver's row duals y as reported: verified on refinery-T12 that
y_i = d(objective)/d(row bound i) in the model's own (MAX) sense for both the simplex and r2HPDHG,
so no sign is changed here. Prices are synthetic (see the generator); the structure is real.
Lever application follows the scenario idea of bench/warm_start.py but is deterministic (a
planner's "+5%" means +5% everywhere, not a random draw).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

from . import generate, paths, runner

sys.path.insert(0, os.path.join(paths.ROOT, "bench"))
import generate_refinery_lp as gr  # noqa: E402  (row/column layout of the refinery model)
import lpgen  # noqa: E402  (fast .lpm writer)
from lpm import read_lpm, read_solution  # noqa: E402

LAYOUT = gr.Layout()
PRODUCTS = list(gr.PRODS)

# lever: (label, what it scales, allowed % range)
LEVERS = {
    "price": ("Product prices", "sale price of every product (objective of the sell columns)", (-50, 50)),
    "crude": ("Crude availability", "upper bound of every crude purchase", (-50, 50)),
    "demand": ("Product demand", "demand cap of every product", (-50, 50)),
    "cdu": ("CDU capacity", "crude distillation capacity, every period", (-50, 50)),
    "fcc": ("FCC capacity", "cat-cracker capacity, every period", (-50, 50)),
}
MAX_SWEEP = 16
RE_BATCH = re.compile(r"^batch it (\d+)\s+t\s+([\d.]+)s\s+running (\d+)/(\d+)")


def refinery_models() -> list[dict]:
    """Generated refinery models that can be used as a base (they carry the generator's sidecar)."""
    out = []
    if os.path.isdir(generate.GEN_DIR):
        for f in sorted(os.listdir(generate.GEN_DIR)):
            p = os.path.join(generate.GEN_DIR, f)
            k = generate.known(p) if f.startswith("refinery-") and f.endswith(".lpm") else None
            if k and k.get("periods"):
                out.append({"path": paths.rel(p), "periods": k["periods"], "known_optimum": k["optimum"]})
    return sorted(out, key=lambda d: d["periods"])


def validate(body: dict):
    base = paths.resolve_model(body.get("base", ""))
    k = generate.known(base) if base else None
    if not k or not k.get("periods") or not os.path.basename(base).startswith("refinery-"):
        return None, "base must be a generated refinery model (bench/generated/refinery-T*.lpm)"
    mode = body.get("mode", "whatif")
    if mode not in ("whatif", "sweep"):
        return None, "mode must be whatif or sweep"
    levers = {}
    for name, v in (body.get("levers") or {}).items():
        if name not in LEVERS:
            return None, f"unknown lever {name!r}"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None, f"{name}: not a number"
        lo, hi = LEVERS[name][2]
        if not lo <= v <= hi:
            return None, f"{name}: must be in [{lo}, {hi}] %"
        if v:
            levers[name] = v
    o = {"base": base, "periods": int(k["periods"]), "mode": mode, "levers": levers}
    try:
        o["threads"] = int(body.get("threads", 0))
        o["time_limit"] = float(body.get("time_limit", 300))
    except (TypeError, ValueError):
        return None, "threads and time_limit must be numbers"
    if not (0 <= o["threads"] <= 4096 and 0 < o["time_limit"] <= 3600):
        return None, "threads 0-4096, time_limit (0, 3600]"
    if mode == "sweep":
        s = body.get("sweep") or {}
        if s.get("lever") not in LEVERS:
            return None, "sweep.lever must be one of " + ", ".join(LEVERS)
        try:
            lo, hi, n = float(s.get("from")), float(s.get("to")), int(s.get("steps"))
        except (TypeError, ValueError):
            return None, "sweep.from, sweep.to, sweep.steps must be numbers"
        a, b = LEVERS[s["lever"]][2]
        if not (a <= lo <= b and a <= hi <= b and lo < hi and 2 <= n <= MAX_SWEEP):
            return None, f"sweep: {a} ≤ from < to ≤ {b}, 2 ≤ steps ≤ {MAX_SWEEP}"
        if o["periods"] > 2190:
            return None, "sweep: use a base with at most 2190 periods (K copies are solved together)"
        o["sweep"] = {"lever": s["lever"], "values": [round(lo + (hi - lo) * i / (n - 1), 6) for i in range(n)]}
        o["sequential"] = bool(body.get("sequential", False))
    return o, None


# ------------------------------------------------------------------------ model edits
_model_cache: dict = {}
_model_lock = threading.Lock()


def _base_arrays(base: str) -> dict:
    """The base model as numpy arrays (cached by path + mtime; reading T8760 takes ~2 s)."""
    key = (base, os.path.getmtime(base))
    with _model_lock:
        if key not in _model_cache:
            m = read_lpm(base)
            _model_cache.clear()
            _model_cache[key] = {"name": m.name, "sense": m.sense, "obj": np.array(m.obj), "col_lower": np.array(m.col_lower),
                                 "col_upper": np.array(m.col_upper), "row_lower": np.array(m.row_lower),
                                 "row_upper": np.array(m.row_upper), "col_start": np.array(m.col_start, dtype=np.int64),
                                 "row_index": np.array(m.row_index, dtype=np.int64), "value": np.array(m.value)}
        return _model_cache[key]


def lever_indices(T: int) -> dict:
    L, t = LAYOUT, np.arange(T)
    per = lambda locs, size: (t[:, None] * size + np.atleast_1d(locs)[None, :]).ravel()
    return {"price": ("obj", per(L.sell, L.ncols)), "crude": ("col_upper", per(L.buy, L.ncols)),
            "demand": ("row_upper", per(L.r_dem, L.nrows)), "cdu": ("row_upper", per(L.r_cdu, L.nrows)),
            "fcc": ("row_upper", per(L.r_fcccap, L.nrows))}


def write_scenario(base: str, T: int, levers: dict, out: str, name: str) -> None:
    b = _base_arrays(base)
    arr = {k: b[k].copy() for k in ("obj", "col_upper", "row_upper")}
    idx = lever_indices(T)
    for lever, pct in levers.items():
        field, ii = idx[lever]
        v = arr[field][ii]
        arr[field][ii] = np.where(np.isfinite(v), v * (1 + pct / 100.0), v)
    lpgen.write_lpm_fast(out, name, b["sense"], arr["obj"], b["col_lower"], arr["col_upper"], b["row_lower"],
                         arr["row_upper"], b["col_start"], b["row_index"], b["value"],
                         source=f"what-if of {os.path.basename(base)}: " + ", ".join(f"{k} {v:+g}%" for k, v in levers.items()))


def marginal_values(T: int, y: list) -> dict:
    """Row duals of the capacity and demand rows per period: the change in profit per extra unit."""
    L, y = LAYOUT, np.asarray(y, dtype=float)
    if len(y) != T * L.nrows:
        return {}
    Y = y.reshape(T, L.nrows)
    return {"cdu": Y[:, L.r_cdu].tolist(), "fcc": Y[:, L.r_fcccap].tolist(),
            "demand": {p: Y[:, L.r_dem[i]].tolist() for i, p in enumerate(PRODUCTS)}}


# ------------------------------------------------------------------------ the job
class ScenarioJob(runner.Job):
    def __init__(self, o: dict):
        super().__init__(paths.rel(o["base"]), o)
        self.cancelled = False

    def cli(self, cmd: list[str], phase: str) -> int:
        """Run the CLI, streaming -vv progress tagged with the phase (base / cold / warm / batch)."""
        if self.cancelled:
            return -1
        shown = ["ps26119"] + [paths.rel(c) if os.path.isabs(c) and c.startswith(paths.ROOT) else c for c in cmd[1:]]
        self.emit("command", phase=phase, command=" ".join(shown))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
        last = 0.0
        for line in self.proc.stderr:
            p = runner.parse_progress(line.rstrip("\n"))
            m = None if p else RE_BATCH.match(line)
            if m:
                p = {"kind": "batch", "iter": int(m[1]), "t": float(m[2]), "running": int(m[3]), "total": int(m[4])}
            if p and (time.time() - last > 0.02):
                self.emit("progress", phase=phase, **p)
                last = time.time()
        code = self.proc.wait()
        self.proc.stderr.close()
        return code

    def solve(self, model: str, sol: str, phase: str, warm: str | None = None) -> dict | None:
        o = self.opts
        cmd = [paths.BIN, "solve", model, "--algorithm", "r2hpdhg", "--tol", "1e-8", "--threads", str(o["threads"]),
               "--time-limit", str(o["time_limit"]), "--out", sol, "-vv"] + (["--warm", warm] if warm else [])
        self.cli(cmd, phase)
        if not os.path.exists(sol):
            self.emit("error", message=f"{phase}: the solver wrote no solution")
            return None
        s = read_solution(sol)
        h = s.header
        num = lambda k: runner._f(h.get(k))
        return {"status": h.get("status"), "objective": num("objective"), "iterations": int(float(h.get("iterations", 0))),
                "seconds": num("seconds"), "engine": h.get("engine"), "check": h.get("check", ""), "y": s.y}

    def check(self, model: str, sol: str, phase: str) -> None:
        v = runner.verify(model, sol, os.path.dirname(sol))
        rep = v.get("report") or {}
        self.emit("verified", phase=phase, verdict=rep.get("verdict"), seconds=v["seconds"],
                  reasons=rep.get("reasons", []), objective=rep.get("objective"))

    def base_solution(self) -> tuple[str, dict] | tuple[None, None]:
        """The base case, solved once per base model and thread setting and reused (warm starts need it)."""
        o = self.opts
        cache = os.path.join(paths.RUNS, "scenarios")
        os.makedirs(cache, exist_ok=True)
        stem = os.path.splitext(os.path.basename(o["base"]))[0]
        sol = os.path.join(cache, f"{stem}-{int(os.path.getmtime(o['base']))}.base.sol")
        if os.path.exists(sol):
            s = read_solution(sol)
            h = s.header
            res = {"status": h.get("status"), "objective": runner._f(h.get("objective")),
                   "iterations": int(float(h.get("iterations", 0))), "seconds": runner._f(h.get("seconds")),
                   "y": s.y, "cached": True}
        else:
            self.emit("stage", stage="base")
            res = self.solve(o["base"], sol + ".tmp", "base")
            if not res:
                return None, None
            if res["status"] != "Optimal":
                self.emit("error", message=f"base case: {res['status']} (needs Optimal to compare against)")
                return None, None
            os.replace(sol + ".tmp", sol)
            res["cached"] = False
        return sol, res

    def _run(self):
        o = self.opts
        os.makedirs(self.dir, exist_ok=True)
        T = o["periods"]
        self.emit("started", mode=o["mode"], base=paths.rel(o["base"]), periods=T, levers=o["levers"],
                  sweep=o.get("sweep"))
        base_sol, base = self.base_solution()
        if not base_sol:
            return
        mv = marginal_values(T, base.pop("y"))
        self.emit("base", **base, known=generate.known(o["base"]), marginal=mv)
        if o["mode"] == "whatif":
            self.whatif(base_sol, base, T)
        else:
            self.sweep(base_sol, base, T)

    def whatif(self, base_sol, base, T):
        o = self.opts
        self.emit("stage", stage="build")
        t0 = time.time()
        scen = os.path.join(self.dir, "scenario.lpm")
        write_scenario(o["base"], T, o["levers"], scen, f"whatif-T{T}")
        self.emit("built", seconds=round(time.time() - t0, 3), path=paths.rel(scen))
        runs = {}
        for phase, warm in (("cold", None), ("warm", base_sol)):
            self.emit("stage", stage=phase)
            r = self.solve(scen, os.path.join(self.dir, f"{phase}.sol"), phase, warm)
            if not r or self.cancelled:
                return
            y = r.pop("y")
            if phase == "warm":
                r["marginal"] = marginal_values(T, y)
            runs[phase] = r
            self.emit("solved", phase=phase, **r)
        self.emit("stage", stage="verify")
        for phase in runs:
            if runs[phase]["status"] in ("Optimal", "Infeasible", "Unbounded"):
                self.check(scen, os.path.join(self.dir, f"{phase}.sol"), phase)
        c, w = runs["cold"], runs["warm"]
        summary = {"delta": None, "delta_pct": None, "agree": None,
                   "iteration_ratio": (w["iterations"] / c["iterations"]) if c["iterations"] else None,
                   "time_ratio": (w["seconds"] / c["seconds"]) if c["seconds"] else None}
        if c["status"] == w["status"] == "Optimal":
            summary["delta"] = w["objective"] - base["objective"]
            summary["delta_pct"] = 100 * summary["delta"] / abs(base["objective"]) if base["objective"] else None
            summary["agree"] = abs(c["objective"] - w["objective"]) / (1 + abs(c["objective"]))
        self.emit("summary", **summary)

    def sweep(self, base_sol, base, T):
        o = self.opts
        sw = o["sweep"]
        self.emit("stage", stage="build")
        t0 = time.time()
        files = []
        for i, v in enumerate(sw["values"]):
            lev = {**o["levers"], sw["lever"]: v} if v else {k: x for k, x in o["levers"].items() if k != sw["lever"]}
            f = os.path.join(self.dir, f"s{i:02d}.lpm")
            write_scenario(o["base"], T, lev, f, f"sweep-{sw['lever']}-{i}")
            files.append(f)
        self.emit("built", seconds=round(time.time() - t0, 3), count=len(files))
        self.emit("stage", stage="batch")
        out = os.path.join(self.dir, "batch")
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        code = self.cli([paths.BIN, "batch", o["base"], *files, "--out-dir", out, "--tol", "1e-8", "--threads",
                         str(o["threads"]), "--time-limit", str(o["time_limit"]), "-vv"], "batch")
        batch_wall = time.time() - t0
        if self.cancelled:
            return
        points = []
        for i, (v, f) in enumerate(zip(sw["values"], files)):
            sol = os.path.join(out, os.path.basename(f)[:-4] + ".sol")
            if not os.path.exists(sol):
                points.append({"value": v, "status": "no solution"})
                continue
            h = read_solution(sol).header
            points.append({"value": v, "status": h.get("status"), "objective": runner._f(h.get("objective")),
                           "iterations": int(float(h.get("iterations", 0))), "seconds": runner._f(h.get("seconds"))})
            self.emit("point", index=i, **points[-1])
        self.emit("batch", exit_code=code, wall_seconds=round(batch_wall, 3),
                  seconds=max((p.get("seconds") or 0) for p in points) if points else None)
        self.emit("stage", stage="verify")
        for i, f in enumerate(files):
            sol = os.path.join(out, os.path.basename(f)[:-4] + ".sol")
            if os.path.exists(sol) and points[i].get("status") in ("Optimal", "Infeasible", "Unbounded"):
                self.check(f, sol, f"s{i}")
            if self.cancelled:
                return
        if o.get("sequential"):
            self.emit("stage", stage="sequential")
            total = 0.0
            for i, f in enumerate(files):
                sol = os.path.join(self.dir, f"seq{i:02d}.sol")
                # --no-presolve: the batch engine has no presolve, so both sides run the same algorithm
                cmd = [paths.BIN, "solve", f, "--algorithm", "r2hpdhg", "--tol", "1e-8", "--no-presolve", "--threads",
                       str(o["threads"]), "--time-limit", str(o["time_limit"]), "--out", sol, "-vv"]
                self.cli(cmd, f"seq{i}")
                if self.cancelled or not os.path.exists(sol):
                    return
                h = read_solution(sol).header
                total += runner._f(h.get("seconds")) or 0.0
                self.emit("sequential_point", index=i, status=h.get("status"), objective=runner._f(h.get("objective")),
                          seconds=runner._f(h.get("seconds")), iterations=int(float(h.get("iterations", 0))))
            self.emit("sequential", seconds=round(total, 3))
        self.emit("summary", points=len(points))
