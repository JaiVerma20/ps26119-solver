"""refsol.py — shared helpers of the reference-solver tools (tools/*_ref.py, tooling only): write a
reference answer in OUR solution-file format (docs/FORMATS.md) so tools/verify.py judges it exactly
like a ps26119 answer, and hand a model to a solver's own MPS reader."""
from __future__ import annotations

import os

from lpm import fingerprint


def activity(m, x):
    """A·x for a PyModel in CSC form (row activity of a reference answer)."""
    import numpy as np
    cols = np.repeat(np.arange(m.num_cols), np.diff(np.asarray(m.col_start, dtype=np.int64)))
    return np.bincount(np.asarray(m.row_index, dtype=np.int64), weights=np.asarray(m.value, float) * np.asarray(x, float)[cols],
                       minlength=m.num_rows).tolist()


def names_ok(names, n: int) -> bool:
    """Names an MPS file can carry and a solver can map back: n unique tokens without blanks."""
    return len(names) == n and len(set(names)) == n and all(nm and not any(ch.isspace() for ch in nm) for nm in names)


def mps_for(model_path: str, tmp: str, m) -> str:
    """The model as an MPS file for a solver's own reader: .mps as is; .lpm written by HiGHS
    (integrality included, so MILPs stay MILPs)."""
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
    if names_ok(m.col_names, m.num_cols) and names_ok(m.row_names, m.num_rows):
        lp.col_names_, lp.row_names_ = list(m.col_names), list(m.row_names)
    if any(m.is_integer):
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in m.is_integer]
    h.passModel(lp)
    out = os.path.join(tmp, "model.mps")
    h.writeModel(out)
    return out


def write_solution(path: str, m, status: str, engine: str, objective: float, seconds: float, iterations: int = 0,
                   x=None, z=None, act=None, y=None) -> None:
    with open(path, "w") as f:
        f.write("PS26119-SOLUTION 1\n")
        f.write(f"model {fingerprint(m)}\nname {m.name}\nstatus {status}\nengine {engine}\nprecision fp64\n")
        f.write(f"objective {objective!r}\niterations {int(iterations)}\nseconds {seconds:.6f}\n")
        if status == "Optimal" and x is not None:
            z = z if z is not None else [0.0] * m.num_cols
            f.write(f"COLUMNS {m.num_cols}\n")
            for j in range(m.num_cols):
                f.write(f"{j} {float(x[j])!r} {float(z[j])!r} {m.col_names[j] if m.col_names else ''}\n")
            if act is not None:
                y = y if y is not None else [0.0] * m.num_rows
                f.write(f"ROWS {m.num_rows}\n")
                for i in range(m.num_rows):
                    f.write(f"{i} {float(act[i])!r} {float(y[i])!r} {m.row_names[i] if m.row_names else ''}\n")
        f.write("END\n")
