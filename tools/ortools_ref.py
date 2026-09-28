#!/usr/bin/env python3
"""ortools_ref.py — solve a model with Google OR-Tools (separate reference, tooling only; never
linked into ps26119, CLAUDE.md §4.5) and write the result in OUR solution-file format, so
tools/verify.py and the benches treat it like any other engine.

Engines: glop (Google's primal/dual simplex) and pdlp (Google's primal-dual hybrid gradient —
the reference implementation of the PDLP family our r2HPDHG belongs to), through OR-Tools'
model_builder API. The model is read by OR-Tools' own MPS reader (.mps); a .lpm file is first
written as MPS by HiGHS (tools/lpm.py reader -> highspy writer). 'seconds' is the wall time of the
solve call only (model reading excluded, like tools/highs_ref.py).

usage: ortools_ref.py MODEL.{mps,lpm} OUT.sol [--solver glop|pdlp] [--time-limit S] [--tol 1e-8] [--json]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpm import fingerprint, read_lpm, read_mps_highspy  # noqa: E402

ENGINES = ("glop", "pdlp")


def _mps_for(model_path: str, tmp: str, m) -> str:
    if model_path.lower().endswith(".mps"):
        return model_path
    import highspy
    import numpy as np
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m.num_cols, m.num_rows
    lp.col_cost_ = np.asarray(m.obj, float)
    lp.col_lower_, lp.col_upper_ = np.asarray(m.col_lower, float), np.asarray(m.col_upper, float)
    lp.row_lower_, lp.row_upper_ = np.asarray(m.row_lower, float), np.asarray(m.row_upper, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.asarray(m.col_start, np.int32)
    lp.a_matrix_.index_ = np.asarray(m.row_index, np.int32)
    lp.a_matrix_.value_ = np.asarray(m.value, float)
    lp.offset_ = m.obj_offset
    lp.sense_ = highspy.ObjSense.kMaximize if m.sense == -1 else highspy.ObjSense.kMinimize
    h.passModel(lp)
    out = os.path.join(tmp, "model.mps")
    h.writeModel(out)
    return out


def ortools_version() -> str:
    import ortools
    return ortools.__version__


def solve_with_ortools(model_path: str, out_path: str, solver: str = "glop", time_limit: float = 0.0,
                       tol: float = 1e-8) -> dict:
    from ortools.linear_solver.python import model_builder as mb
    if solver not in ENGINES:
        raise ValueError(f"solver must be one of {ENGINES}")
    m = read_lpm(model_path) if model_path.lower().endswith(".lpm") else read_mps_highspy(model_path)
    with tempfile.TemporaryDirectory(prefix="ortools-ref-") as tmp:
        model = mb.Model()
        if not model.import_from_mps_file(_mps_for(model_path, tmp, m)):
            raise RuntimeError(f"OR-Tools could not read {model_path}")
    if model.num_variables != m.num_cols or model.num_constraints != m.num_rows:
        raise RuntimeError(f"size mismatch: OR-Tools read {model.num_constraints}x{model.num_variables}, "
                           f"expected {m.num_rows}x{m.num_cols}")
    s = mb.Solver(solver)
    if time_limit > 0:
        s.set_time_limit_in_seconds(float(time_limit))
    if solver == "pdlp":  # the same relative / absolute optimality target as our first-order engines
        s.set_solver_specific_parameters(
            f"termination_criteria {{ simple_optimality_criteria {{ eps_optimal_relative: {tol} eps_optimal_absolute: {tol} }} }}")
    t0 = time.perf_counter()
    status = s.solve(model)
    secs = time.perf_counter() - t0
    raw = str(status).split(".")[-1]
    st = {"OPTIMAL": "Optimal", "INFEASIBLE": "Infeasible", "UNBOUNDED": "Unbounded"}.get(raw, "NotSolved")
    if raw in ("FEASIBLE", "ABNORMAL", "NOT_SOLVED", "MODEL_INVALID") and s.status_string and "time" in s.status_string.lower():
        st = "TimeLimit"
    variables, constraints = model.get_variables(), model.get_linear_constraints()
    obj = float(s.objective_value) if st == "Optimal" else float("nan")
    with open(out_path, "w") as f:
        f.write("PS26119-SOLUTION 1\n")
        f.write(f"model {fingerprint(m)}\nname {m.name}\nstatus {st}\nengine ortools-{solver}\nprecision fp64\n")
        f.write(f"objective {obj!r}\nseconds {secs:.6f}\n")
        if st == "Optimal":
            import numpy as np
            x, rc = list(s.values(variables)), list(s.reduced_costs(variables))
            y = list(s.dual_values(constraints))
            cols = np.repeat(np.arange(m.num_cols), np.diff(np.asarray(m.col_start, dtype=np.int64)))
            act = np.bincount(np.asarray(m.row_index, dtype=np.int64), weights=np.asarray(m.value) * np.asarray(x)[cols],
                              minlength=m.num_rows).tolist()
            f.write(f"COLUMNS {m.num_cols}\n")
            for j in range(m.num_cols):
                f.write(f"{j} {x[j]!r} {rc[j]!r} {m.col_names[j] if m.col_names else ''}\n")
            f.write(f"ROWS {m.num_rows}\n")
            for i in range(m.num_rows):
                f.write(f"{i} {act[i]!r} {y[i]!r} {m.row_names[i] if m.row_names else ''}\n")
        f.write("END\n")
    return {"status": st, "objective": obj, "seconds": secs, "engine": f"ortools-{solver}",
            "ortools_version": ortools_version(), "ortools_status": raw}


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="OR-Tools as a separate reference (tooling only)")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--solver", choices=ENGINES, default="glop")
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = solve_with_ortools(a.model, a.out, a.solver, a.time_limit, a.tol)
    print(json.dumps(res) if a.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
