#!/usr/bin/env python3
"""highs_ref.py — solve a model with HiGHS (separate reference, tooling only) and write the
result in OUR solution-file format, so tools/verify.py and the benches can treat HiGHS
like any other engine. Never used by the solver (CLAUDE.md §5.7).

usage: highs_ref.py MODEL.{mps,lpm} OUT.sol [--solver simplex|ipm|pdlp|choose] [--time-limit S] [--json]
       (--json prints the result as one JSON line; pdlp runs to pdlp_optimality_tolerance 1e-8)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpm import fingerprint, read_lpm, read_mps_highspy  # noqa: E402


def solve_with_highs(model_path: str, out_path: str, time_limit: float = 0.0, options: dict | None = None) -> dict:
    """options: HiGHS option name -> value (e.g. {"solver": "pdlp", "pdlp_optimality_tolerance": 1e-8}).
    'seconds' is the wall time of Highs.run() (model reading and passing excluded)."""
    import highspy
    import numpy as np

    m = read_lpm(model_path) if model_path.endswith(".lpm") else read_mps_highspy(model_path)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit > 0:
        h.setOptionValue("time_limit", float(time_limit))
    for k, v in (options or {}).items():
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected option {k}={v!r}")
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
    if any(m.is_integer):  # a MILP stays a MILP (HiGHS branch and cut), never its LP relaxation
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in m.is_integer]
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    secs = time.perf_counter() - t0  # wall time of the solve call only (presolve included, like ours)
    status = h.getModelStatus()
    st = {
        highspy.HighsModelStatus.kOptimal: "Optimal",
        highspy.HighsModelStatus.kInfeasible: "Infeasible",
        highspy.HighsModelStatus.kUnbounded: "Unbounded",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "Infeasible",
        highspy.HighsModelStatus.kTimeLimit: "TimeLimit",
        highspy.HighsModelStatus.kIterationLimit: "IterationLimit",
    }.get(status, "NotSolved")
    raw_status = str(status).split(".")[-1]  # HiGHS's own status name, e.g. kUnknown
    sol = h.getSolution()
    info = h.getInfo()
    # iterations of the engine that ran (simplex, interior point, or PDLP)
    iters = max(int(getattr(info, k, 0) or 0) for k in ("simplex_iteration_count", "ipm_iteration_count", "pdlp_iteration_count"))
    engine = "highs-" + str((options or {}).get("solver", "choose"))
    with open(out_path, "w") as f:
        f.write("PS26119-SOLUTION 1\n")
        f.write(f"model {fingerprint(m)}\nname {m.name}\nstatus {st}\nengine {engine}\nprecision fp64\n")
        f.write(f"objective {info.objective_function_value!r}\n")
        f.write(f"iterations {iters}\nseconds {secs:.6f}\n")
        if st == "Optimal":
            # One copy of each vector: in recent highspy every attribute access (sol.row_value, …)
            # converts the WHOLE C++ vector to a new Python list, so indexing it inside the loop made
            # writing the file O(m²) — hours for Kennington ken-18 (105k rows).
            col_value, col_dual = list(sol.col_value), list(sol.col_dual)
            row_value, row_dual = list(sol.row_value), list(sol.row_dual)
            col_names, row_names = list(m.col_names or []), list(m.row_names or [])
            f.write(f"COLUMNS {m.num_cols}\n")
            for j in range(m.num_cols):
                name = col_names[j] if col_names else ""
                f.write(f"{j} {col_value[j]!r} {col_dual[j]!r} {name}\n")
            f.write(f"ROWS {m.num_rows}\n")
            for i in range(m.num_rows):
                name = row_names[i] if row_names else ""
                f.write(f"{i} {row_value[i]!r} {row_dual[i]!r} {name}\n")
        f.write("END\n")
    return {"status": st, "objective": info.objective_function_value, "seconds": secs, "iterations": iters,
            "engine": engine, "highs_version": highs_version(h), "highs_status": raw_status}


def highs_version(h=None) -> str:
    import highspy
    h = h or highspy.Highs()
    for f in ("version", "versionMajor"):
        if hasattr(h, f) and f == "version":
            try:
                return str(h.version())
            except Exception:  # noqa: BLE001
                pass
    try:
        return f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}"
    except Exception:  # noqa: BLE001
        return "unknown"


ENGINES = {"choose": {}, "simplex": {"solver": "simplex"}, "ipm": {"solver": "ipm"},
           "pdlp": {"solver": "pdlp", "pdlp_optimality_tolerance": 1e-8}}


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="HiGHS as a separate reference (tooling only)")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--solver", choices=sorted(ENGINES), default="choose")
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = solve_with_highs(a.model, a.out, time_limit=a.time_limit, options=ENGINES[a.solver])
    print(json.dumps(res) if a.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
