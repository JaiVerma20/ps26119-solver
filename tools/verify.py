#!/usr/bin/env python3
"""verify.py — independent checker for ps26119 solution files.

Reads the ORIGINAL model with a reader that is not ours (highspy for .mps; the Python
.lpm reader for generated instances), reads our solution file, and recomputes everything
from x and y alone — it never trusts objective / residual values printed by the solver:

  primal   max bound violation, max row violation (A x recomputed here)
  dual     sign feasibility of y and of z = c − Aᵀy (z recomputed here)
  gap      primal objective vs the Lagrangian dual objective built from y and z
  compl.   complementary slackness products (reported, not a PASS criterion)
  model    fingerprint in the solution file vs fingerprint of the model read here

Tolerances are parsed from include/ps26119/tolerances.h (the single source of truth):
  primal  max_i viol_i / (1 + |violated bound_i|)          <= kVerifyPrimal
  dual    max sign violation / (1 + ||c||_inf)             <= kVerifyDual
  gap     |p − d| / (1 + |p| + |d|)                        <= kVerifyGap
  ref     |obj − expected| / (1 + |expected|)              <= kVerifyReference (if given)
Absolute values are printed next to the relative ones.

Sign convention (include/ps26119/solution.h): z = c − Aᵀy in the ORIGINAL objective sense.
For MIN, y_i ≥ 0 at a row's lower bound and ≤ 0 at its upper bound; same for z_j and
column bounds. For MAX all signs flip. We multiply by `sense` to work in min form.

usage: verify.py MODEL.{mps,lpm} SOLUTION [--expected OBJ] [--json OUT.json] [--quiet]
exit:  0 PASS, 1 FAIL, 2 usage / read error
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lpm import fingerprint, read_lpm, read_mps_highspy, read_solution  # noqa: E402

FINGERPRINT_MAX_NNZ = 300_000
TOL_HEADER = os.path.join(HERE, "..", "include", "ps26119", "tolerances.h")


def load_tolerances(path: str = TOL_HEADER) -> dict:
    tol = {}
    with open(path) as f:
        for name, val in re.findall(r"inline constexpr double (k\w+)\s*=\s*([0-9.eE+-]+);", f.read()):
            tol[name] = float(val)
    for need in ("kVerifyPrimal", "kVerifyDual", "kVerifyGap", "kVerifyReference", "kMipIntegrality"):
        if need not in tol:
            raise RuntimeError(f"{need} missing from {path}")
    return tol


def load_model(path: str):
    if path.lower().endswith(".lpm"):
        return read_lpm(path), "lpm.py"
    return read_mps_highspy(path), "highspy"


def verify(model_path: str, solution_path: str, expected: float | None = None) -> dict:
    tol = load_tolerances()
    m, reader = load_model(model_path)
    s = read_solution(solution_path)
    n, mrows = m.num_cols, m.num_rows
    rep: dict = {
        "model": os.path.basename(model_path),
        "reader": reader,
        "rows": mrows,
        "cols": n,
        "nnz": m.nnz,
        "status": s.header.get("status", "?"),
        "engine": s.header.get("engine", "?"),
        "precision": s.header.get("precision", "?"),
        # pure-Python FNV is slow (~1 s per 100k nnz); skip it for big models
        "fingerprint_model": fingerprint(m) if m.nnz <= FINGERPRINT_MAX_NNZ else "skipped",
        "fingerprint_solution": s.header.get("model", ""),
    }
    rep["model_match"] = rep["fingerprint_model"] in ("skipped", rep["fingerprint_solution"])
    reasons = []
    if rep["status"] != "Optimal":
        reasons.append(f"status is {rep['status']}, not Optimal")
    if len(s.x) != n or (len(s.y) != mrows and not any(m.is_integer)):
        reasons.append("solution has no primal/dual vectors of the right size")
        rep.update(verdict="FAIL", reasons=reasons)
        return rep

    is_mip = any(m.is_integer)
    rep["mip"] = is_mip
    sense = m.sense
    c = np.asarray(m.obj, dtype=float)
    cl, cu = np.asarray(m.col_lower, float), np.asarray(m.col_upper, float)
    rl, ru = np.asarray(m.row_lower, float), np.asarray(m.row_upper, float)
    val = np.asarray(m.value, float)
    ridx = np.asarray(m.row_index, dtype=np.int64)
    cidx = np.repeat(np.arange(n), np.diff(np.asarray(m.col_start, dtype=np.int64)))
    x = np.asarray(s.x, float)
    y = np.asarray(s.y, float) if len(s.y) == mrows else np.zeros(mrows)

    # ---------------- primal
    ax = np.bincount(ridx, weights=val * x[cidx], minlength=mrows) if mrows else np.zeros(0)
    col_viol = np.maximum(np.maximum(cl - x, x - cu), 0.0)
    col_den = 1.0 + np.where(cl - x > x - cu, np.abs(np.where(np.isfinite(cl), cl, 0)), np.abs(np.where(np.isfinite(cu), cu, 0)))
    row_viol = np.maximum(np.maximum(rl - ax, ax - ru), 0.0)
    row_den = 1.0 + np.where(rl - ax > ax - ru, np.abs(np.where(np.isfinite(rl), rl, 0)), np.abs(np.where(np.isfinite(ru), ru, 0)))
    rep["max_col_violation_abs"] = float(col_viol.max(initial=0.0))
    rep["max_row_violation_abs"] = float(row_viol.max(initial=0.0))
    rep["primal_rel"] = float(max((col_viol / col_den).max(initial=0.0), (row_viol / row_den).max(initial=0.0)))
    if len(s.row_activity) == mrows and mrows:
        rep["activity_mismatch_abs"] = float(np.max(np.abs(np.asarray(s.row_activity) - ax)))

    if is_mip:
        # MILP: primal feasibility + integrality + objective (no duals exist for a MILP optimum)
        ints = np.asarray([bool(v) for v in m.is_integer])
        frac = np.abs(x[ints] - np.round(x[ints])) if ints.any() else np.zeros(0)
        rep["max_fractionality"] = float(frac.max(initial=0.0))
        rep["objective"] = float(c @ x) + m.obj_offset
        if rep["primal_rel"] > tol["kVerifyPrimal"]:
            reasons.append(f"primal violation {rep['primal_rel']:.3e} > {tol['kVerifyPrimal']:.0e}")
        if rep["max_fractionality"] > tol["kMipIntegrality"]:
            reasons.append(f"integer column off by {rep['max_fractionality']:.2e}")
        if expected is not None:
            rep["expected_objective"] = expected
            rep["reference_rel"] = abs(rep["objective"] - expected) / (1.0 + abs(expected))
            if rep["reference_rel"] > tol["kVerifyReference"]:
                reasons.append(f"objective {rep['objective']:.10g} differs from expected {expected:.10g}")
        rep["verdict"] = "FAIL" if reasons else "PASS"
        rep["reasons"] = reasons
        return rep

    # ---------------- dual (min form)
    aty = np.bincount(cidx, weights=val * y[ridx], minlength=n) if n else np.zeros(0)
    z = c - aty
    if len(s.z) == n and n:
        rep["reduced_cost_mismatch_abs"] = float(np.max(np.abs(np.asarray(s.z) - z)))
    ym, zm = sense * y, sense * z
    y_viol = np.where(ym > 0, np.where(np.isfinite(rl), 0.0, ym), np.where(np.isfinite(ru), 0.0, -ym))
    z_viol = np.where(zm > 0, np.where(np.isfinite(cl), 0.0, zm), np.where(np.isfinite(cu), 0.0, -zm))
    dual_abs = float(max(y_viol.max(initial=0.0), z_viol.max(initial=0.0)))
    cnorm = float(np.abs(c).max(initial=0.0))
    rep["dual_violation_abs"] = dual_abs
    rep["dual_rel"] = dual_abs / (1.0 + cnorm)

    # ---------------- objectives and gap (min form, offset excluded)
    p_min = sense * float(c @ x)
    def bound_term(v, lo, up):
        t = np.where(v > 0, np.where(np.isfinite(lo), lo, 0.0) * v, np.where(np.isfinite(up), up, 0.0) * v)
        return float(np.sum(t))
    d_min = bound_term(ym, rl, ru) + bound_term(zm, cl, cu)
    rep["objective"] = sense * p_min + m.obj_offset
    rep["dual_objective"] = sense * d_min + m.obj_offset
    rep["gap_abs"] = abs(p_min - d_min)
    rep["gap_rel"] = abs(p_min - d_min) / (1.0 + abs(p_min) + abs(d_min))
    try:
        rep["objective_reported"] = float(s.header.get("objective", "nan"))
    except ValueError:
        rep["objective_reported"] = float("nan")

    # ---------------- complementary slackness (report only)
    dl = np.where(np.isfinite(cl), x - cl, 0.0)
    du = np.where(np.isfinite(cu), cu - x, 0.0)
    comp_col = np.where(zm > 0, zm * np.abs(dl), -zm * np.abs(du))
    sl = np.where(np.isfinite(rl), ax - rl, 0.0)
    su = np.where(np.isfinite(ru), ru - ax, 0.0)
    comp_row = np.where(ym > 0, ym * np.abs(sl), -ym * np.abs(su))
    rep["complementarity_abs"] = float(max(comp_col.max(initial=0.0), comp_row.max(initial=0.0)))

    # ---------------- verdict
    if rep["primal_rel"] > tol["kVerifyPrimal"]:
        reasons.append(f"primal violation {rep['primal_rel']:.3e} > {tol['kVerifyPrimal']:.0e}")
    if rep["dual_rel"] > tol["kVerifyDual"]:
        reasons.append(f"dual violation {rep['dual_rel']:.3e} > {tol['kVerifyDual']:.0e}")
    if rep["gap_rel"] > tol["kVerifyGap"]:
        reasons.append(f"relative gap {rep['gap_rel']:.3e} > {tol['kVerifyGap']:.0e}")
    if expected is not None:
        rep["expected_objective"] = expected
        rep["reference_rel"] = abs(rep["objective"] - expected) / (1.0 + abs(expected))
        if rep["reference_rel"] > tol["kVerifyReference"]:
            reasons.append(f"objective {rep['objective']:.10g} differs from expected {expected:.10g} "
                           f"(rel {rep['reference_rel']:.2e})")
    rep["verdict"] = "FAIL" if reasons else "PASS"
    rep["reasons"] = reasons
    return rep


def print_report(rep: dict) -> None:
    def g(k, fmt="{:.3e}"):
        v = rep.get(k)
        return "-" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))
    print(f"model      {rep['model']}  ({rep['rows']} rows, {rep['cols']} cols, {rep['nnz']} nnz, read by {rep['reader']})")
    print(f"solution   status={rep['status']} engine={rep['engine']} precision={rep['precision']}")
    print(f"fingerprint model={rep['fingerprint_model']} solution={rep['fingerprint_solution']} "
          f"{'(not computed: large model)' if rep['fingerprint_model'] == 'skipped' else 'match' if rep['model_match'] else 'MISMATCH (different reader or model)'}")
    if "objective" in rep:
        print(f"objective  primal={g('objective', '{:.12g}')}  dual={g('dual_objective', '{:.12g}')}  "
              f"reported={g('objective_reported', '{:.12g}')}")
        print(f"primal     rel={g('primal_rel')}  abs: col={g('max_col_violation_abs')} row={g('max_row_violation_abs')}")
        print(f"dual       rel={g('dual_rel')}  abs={g('dual_violation_abs')}")
        print(f"gap        rel={g('gap_rel')}  abs={g('gap_abs')}")
        print(f"compl.     abs={g('complementarity_abs')}  (report only)")
        if "expected_objective" in rep:
            print(f"reference  expected={rep['expected_objective']:.12g}  rel diff={g('reference_rel')}")
    print(f"VERDICT    {rep['verdict']}" + ("" if not rep["reasons"] else "  — " + "; ".join(rep["reasons"])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("solution")
    ap.add_argument("--expected", type=float, default=None, help="published optimal objective")
    ap.add_argument("--json", default=None, help="write the report as JSON here")
    ap.add_argument("--quiet", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    # allow `--expected -464.75` (argparse would read the negative number as an option)
    for k in range(len(argv) - 1):
        if argv[k] == "--expected":
            argv[k : k + 2] = [f"--expected={argv[k + 1]}"]
            break
    a = ap.parse_args(argv)
    try:
        rep = verify(a.model, a.solution, a.expected)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"verify.py: {e}", file=sys.stderr)
        return 2
    if not a.quiet:
        print_report(rep)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v)
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
