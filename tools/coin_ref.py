#!/usr/bin/env python3
"""coin_ref.py — solve a model with COIN-OR CLP (LP: dual simplex, as the clp program does by
default) or CBC (MILP: branch and cut) through cylp (EPL-2.0 COIN-OR solvers, native build) as a
separate reference (tooling only; never linked into ps26119, CLAUDE.md §4.5), and write the result
in OUR solution-file format, so tools/verify.py and the benches judge it like any other engine.

The model is read by CLP's own MPS reader (a .lpm file is first written as MPS by HiGHS); integer
markers are copied in for CBC. 'seconds' is the wall time of the solve call only. CLP has no time
limit in this interface: the caller's process timeout is the limit (the UI reports TimeLimit).

usage: coin_ref.py MODEL.{mps,lpm} OUT.sol [--solver clp|clp-primal|cbc] [--time-limit S] [--json]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpm import read_lpm, read_mps_highspy  # noqa: E402
from refsol import activity, mps_for, write_solution  # noqa: E402

ENGINES = ("clp", "clp-primal", "cbc")


def coin_version() -> str:
    try:
        from importlib.metadata import version
        return f"cylp {version('cylp')}"
    except Exception:  # noqa: BLE001
        return "cylp"


class _QuietStdout:
    """COIN-OR prints from C to file descriptor 1 (e.g. 'MAX found after OBJSENSE - Coin ignores');
    send it to stderr so that stdout carries only this tool's own output (the --json line)."""

    def __enter__(self):
        sys.stdout.flush()
        self.saved = os.dup(1)
        os.dup2(2, 1)

    def __exit__(self, *exc):
        try:  # C stdio buffers: flush them while fd 1 still points at stderr
            import ctypes
            ctypes.CDLL(None).fflush(None)
        except (OSError, AttributeError):
            pass
        os.dup2(self.saved, 1)
        os.close(self.saved)


def solve_with_coin(model_path: str, out_path: str, solver: str = "clp", time_limit: float = 0.0) -> dict:
    import numpy as np
    from cylp.cy import CyClpSimplex
    if solver not in ENGINES:
        raise ValueError(f"solver must be one of {ENGINES}")
    m = read_lpm(model_path) if model_path.lower().endswith(".lpm") else read_mps_highspy(model_path)
    mip = any(m.is_integer)
    if mip and solver != "cbc":
        solver = "cbc"  # CLP alone would solve the LP relaxation — never report that as the MILP answer
    if not mip and solver == "cbc":
        solver = "clp"  # an LP goes to CBC's LP solver, CLP, directly
    s = CyClpSimplex()
    s.logLevel = 0
    with tempfile.TemporaryDirectory(prefix="coin-ref-") as tmp:
        if s.readMps(mps_for(model_path, tmp, m)) != 0:
            raise RuntimeError(f"CLP could not read {model_path}")
    if s.nConstraints != m.num_rows or s.nVariables != m.num_cols:
        raise RuntimeError(f"size mismatch: CLP read {s.nConstraints}x{s.nVariables}, expected {m.num_rows}x{m.num_cols}")
    if m.sense == -1:
        s.optimizationDirection = "max"  # CLP's MPS reader ignores OBJSENSE MAX
    kw, obj, iters = {}, float("nan"), 0
    if solver == "cbc":
        if mip:
            s.copyInIntegerInformation(np.asarray([1 if v else 0 for v in m.is_integer], dtype=np.uint8))
        cbc = s.getCbcModel()
        cbc.logLevel = 0
        if time_limit > 0:
            cbc.maximumSeconds = float(time_limit)
        t0 = time.perf_counter()
        cbc.solve()
        secs = time.perf_counter() - t0
        raw = str(cbc.status)
        # 'solution' is cylp's name for CbcModel::isProvenOptimal()
        if raw == "solution" and cbc.solutionCount > 0:
            st = "Optimal"
        elif raw in ("relaxation infeasible", "problem proven infeasible"):
            st = "Infeasible"
        elif raw in ("stopped on time", "stopped on nodes", "stopped on solutions", "stopped on gap"):
            st = "TimeLimit"
        else:
            st = "NotSolved"
        if st == "Optimal":
            x = np.asarray(cbc.primalVariableSolution, float).tolist()
            obj = m.obj_offset + float(np.dot(m.obj, x))  # the model's objective at CBC's point
            kw = {"x": x, "z": [0.0] * m.num_cols, "act": activity(m, x), "y": [0.0] * m.num_rows}
        iters = int(cbc.nodeCount)
        engine = "coin-cbc"
    else:
        t0 = time.perf_counter()
        try:
            (s.dual if solver == "clp" else s.primal)()
            raw = s.getStatusString()
        except Exception as e:  # noqa: BLE001 — a failure inside the binding is a NotSolved row, not a crash
            raw = f"error: {type(e).__name__}"
        secs = time.perf_counter() - t0
        st = {"optimal": "Optimal", "primal infeasible": "Infeasible", "dual infeasible": "Unbounded"}.get(raw, "NotSolved")
        if st == "Optimal":
            x = np.asarray(s.primalVariableSolution, float).tolist()
            obj = m.obj_offset + float(np.dot(m.obj, x))
            kw = {"x": x, "z": np.asarray(s.dualVariableSolution, float).tolist(),
                  "act": np.asarray(s.primalConstraintSolution, float).tolist(),
                  "y": np.asarray(s.dualConstraintSolution, float).tolist()}
        iters = int(getattr(s, "iteration", 0) or 0)
        engine = f"coin-{solver}"
    write_solution(out_path, m, st, engine, obj, secs, iters, **kw)
    return {"status": st, "objective": obj, "seconds": secs, "iterations": iters, "engine": engine,
            "version": coin_version(), "raw_status": raw}


def _solve_quietly(*args) -> dict:
    with _QuietStdout():
        return solve_with_coin(*args)


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="COIN-OR CLP / CBC as a separate reference (tooling only)")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--solver", choices=ENGINES, default="clp")
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = _solve_quietly(a.model, a.out, a.solver, a.time_limit)
    print(json.dumps(res) if a.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
