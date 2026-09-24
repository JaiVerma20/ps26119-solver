#!/usr/bin/env python3
"""highs_ref.py — solve a model with HiGHS (separate reference, tooling only) and write the
result in OUR solution-file format, so tools/verify.py and the benches can treat HiGHS
like any other engine. Never used by the solver (CLAUDE.md §5.7).

usage: highs_ref.py MODEL.{mps,lpm} OUT.sol
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpm import fingerprint, read_lpm, read_mps_highspy  # noqa: E402


def solve_with_highs(model_path: str, out_path: str) -> dict:
    import highspy
    import numpy as np

    m = read_lpm(model_path) if model_path.endswith(".lpm") else read_mps_highspy(model_path)
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
    t0 = time.perf_counter()
    h.run()
    secs = time.perf_counter() - t0
    status = h.getModelStatus()
    st = {
        highspy.HighsModelStatus.kOptimal: "Optimal",
        highspy.HighsModelStatus.kInfeasible: "Infeasible",
        highspy.HighsModelStatus.kUnbounded: "Unbounded",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "Infeasible",
    }.get(status, "NotSolved")
    sol = h.getSolution()
    info = h.getInfo()
    with open(out_path, "w") as f:
        f.write("PS26119-SOLUTION 1\n")
        f.write(f"model {fingerprint(m)}\nname {m.name}\nstatus {st}\nengine highs\nprecision fp64\n")
        f.write(f"objective {info.objective_function_value!r}\n")
        f.write(f"iterations {info.simplex_iteration_count}\nseconds {secs:.6f}\n")
        if st == "Optimal":
            f.write(f"COLUMNS {m.num_cols}\n")
            for j in range(m.num_cols):
                name = m.col_names[j] if m.col_names else ""
                f.write(f"{j} {sol.col_value[j]!r} {sol.col_dual[j]!r} {name}\n")
            f.write(f"ROWS {m.num_rows}\n")
            for i in range(m.num_rows):
                name = m.row_names[i] if m.row_names else ""
                f.write(f"{i} {sol.row_value[i]!r} {sol.row_dual[i]!r} {name}\n")
        f.write("END\n")
    return {"status": st, "objective": info.objective_function_value, "seconds": secs,
            "iterations": info.simplex_iteration_count}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    print(solve_with_highs(sys.argv[1], sys.argv[2]))
