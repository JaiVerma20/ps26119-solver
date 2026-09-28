#!/usr/bin/env python3
"""scip_ref.py — solve a model with SCIP (via PySCIPOpt; SCIP and PySCIPOpt are Apache-2.0) as a
separate reference (tooling only; never linked into ps26119, CLAUDE.md §4.5) and write the result
in OUR solution-file format, so tools/verify.py and the benches judge it like any other engine.

LP: SCIP solves the LP with its LP solver (SoPlex) — presolve, propagation, separation and
heuristics are switched OFF so that the row duals (getDualsolLinear) and reduced costs
(getVarRedcost) belong to the original model and can be verified; this is SCIP as an LP solver,
not tuned for LP speed (noted wherever its times are shown). MILP: SCIP with its defaults (branch
and cut); no duals. The model is read by SCIP's own MPS reader (a .lpm file is first written as MPS
by HiGHS). 'seconds' is the wall time of optimize() only.

usage: scip_ref.py MODEL.{mps,lpm} OUT.sol [--time-limit S] [--json]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpm import read_lpm, read_mps_highspy  # noqa: E402
from refsol import activity, mps_for, names_ok, write_solution  # noqa: E402


def scip_version() -> str:
    from pyscipopt import Model
    return str(Model().version())


def singleton_rows_to_duals(m, x, y, z, tol: float = 1e-9) -> None:
    """SCIP turns a one-entry row (l ≤ a·x_j ≤ u) into a bound on x_j even with presolve off, so that
    row's multiplier arrives inside the reduced cost z_j. Where z_j has a sign x_j's OWN bounds do
    not allow (our convention, with s = +1 min / −1 max: s·z > 0 needs x_j at its lower bound,
    s·z < 0 at its upper, else z = 0), move it back to such a row that is active on the matching
    side: y_i += z_j / a, z_j = 0 — the same dual solution, written for the original rows."""
    sgn = -1.0 if m.sense == -1 else 1.0
    count = [0] * m.num_rows
    for p in range(len(m.row_index)):
        count[m.row_index[p]] += 1
    at = lambda v, b: abs(b) < float("inf") and abs(v - b) <= 1e-9 * (1 + abs(b))  # noqa: E731

    def allowed(val, side_lo, side_up):  # a multiplier's sign against the active side(s)
        return (sgn * val <= tol or side_lo) and (sgn * val >= -tol or side_up)

    for j in range(m.num_cols):
        if abs(z[j]) <= tol or allowed(z[j], at(x[j], m.col_lower[j]), at(x[j], m.col_upper[j])):
            continue
        for p in range(m.col_start[j], m.col_start[j + 1]):
            i, a = m.row_index[p], m.value[p]
            if count[i] != 1 or a == 0:
                continue
            yi = y[i] + z[j] / a
            if allowed(yi, at(a * x[j], m.row_lower[i]), at(a * x[j], m.row_upper[i])):
                y[i], z[j] = yi, 0.0
                break


def solve_with_scip(model_path: str, out_path: str, time_limit: float = 0.0) -> dict:
    from pyscipopt import SCIP_PARAMSETTING, Model
    m = read_lpm(model_path) if model_path.lower().endswith(".lpm") else read_mps_highspy(model_path)
    mip = any(m.is_integer)
    M = Model()
    M.hideOutput()
    with tempfile.TemporaryDirectory(prefix="scip-ref-") as tmp:
        M.readProblem(mps_for(model_path, tmp, m))
    variables, conss = M.getVars(), M.getConss()
    if len(variables) != m.num_cols or len(conss) != m.num_rows:
        raise RuntimeError(f"size mismatch: SCIP read {len(conss)}x{len(variables)}, expected {m.num_rows}x{m.num_cols}")
    if time_limit > 0:
        M.setParam("limits/time", float(time_limit))
    if not mip:  # duals of the ORIGINAL model (see the header)
        M.setPresolve(SCIP_PARAMSETTING.OFF)
        M.setHeuristics(SCIP_PARAMSETTING.OFF)
        M.setSeparating(SCIP_PARAMSETTING.OFF)
        M.disablePropagation()
    t0 = time.perf_counter()
    M.optimize()
    secs = time.perf_counter() - t0
    raw = M.getStatus()
    st = {"optimal": "Optimal", "infeasible": "Infeasible", "unbounded": "Unbounded", "timelimit": "TimeLimit",
          "inforunbd": "Infeasible"}.get(raw, "NotSolved")
    kw = {}
    obj = float("nan")
    if st == "Optimal":  # SCIP keeps the original order of variables and constraints (sizes checked above)
        obj = float(M.getObjVal())
        sol = M.getBestSol()
        # by name when the names survive the MPS round trip (SCIP orders MILP variables by type),
        # else by position (HiGHS wrote the file in model order)
        col = {v.name: j for j, v in enumerate(variables)}
        row = {c.name: i for i, c in enumerate(conss)}
        cperm = [col[nm] for nm in m.col_names] if names_ok(m.col_names, m.num_cols) and set(m.col_names) == set(col) \
            else list(range(m.num_cols))
        rperm = [row[nm] for nm in m.row_names] if names_ok(m.row_names, m.num_rows) and set(m.row_names) == set(row) \
            else list(range(m.num_rows))
        if mip and [variables[k].name for k in cperm] != [v.name for v in variables] and not names_ok(m.col_names, m.num_cols):
            raise RuntimeError("SCIP reorders MILP variables and this model's column names cannot map them back")
        x = [float(M.getSolVal(sol, variables[k])) for k in cperm]
        y, z = [0.0] * m.num_rows, [0.0] * m.num_cols
        if not mip:
            # getDualsolLinear answers for SCIP's internal minimisation: our (HiGHS) sign convention,
            # c = Aᵀy + z in the model's own sense, needs y negated for a maximisation (z already is)
            ysign = -1.0 if m.sense == -1 else 1.0
            y = [ysign * float(M.getDualsolLinear(conss[k])) for k in rperm]
            z = [float(M.getVarRedcost(variables[k])) for k in cperm]
            singleton_rows_to_duals(m, x, y, z)
        kw = {"x": x, "z": z, "act": activity(m, x), "y": y}
    iters = int(M.getNLPIterations())
    engine = "scip-mip" if mip else "scip-lp"
    write_solution(out_path, m, st, engine, obj, secs, iters, **kw)
    return {"status": st, "objective": obj, "seconds": secs, "iterations": iters, "engine": engine,
            "version": f"SCIP {scip_version()}", "raw_status": raw}


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="SCIP as a separate reference (tooling only)")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = solve_with_scip(a.model, a.out, a.time_limit)
    print(json.dumps(res) if a.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
