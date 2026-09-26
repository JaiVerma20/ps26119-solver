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
    for need in ("kVerifyPrimal", "kVerifyDual", "kVerifyGap", "kVerifyReference", "kMipIntegrality", "kVerifyRay"):
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
    if rep["status"] in ("Infeasible", "Unbounded") and not any(m.is_integer):
        return verify_certificate(m, s, tol, rep)
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


EXACT_MAX_NNZ = 300_000


def _round_out(q, up: bool):
    """The double nearest to the rational q, moved one ulp outward if needed so that it is >= q
    (up=True) or <= q (up=False), as an exact Fraction; None if q is outside the double range."""
    from fractions import Fraction
    try:
        f = float(q)
    except OverflowError:
        return None
    if not math.isfinite(f):
        return None
    if up and Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    elif not up and Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    return Fraction(f) if math.isfinite(f) else None


def _implied_bounds(m, passes: int = 20):
    """Column bounds implied by the rows, by iterated one-row propagation, EXACTLY (Fraction).
    Each pass uses only bounds already proven (original bounds, then earlier implied ones), so
    every derived bound holds for every feasible x — usable in a proof that no feasible x exists.
    For a row a_i x <= U_i (or -a_i x <= -L_i) and a coefficient c = a_ij > 0:
        x_j <= (U_i - sum_{k != j} min_{x_k in [lo_k, up_k]} a_ik x_k) / a_ij   (symmetric for c < 0).
    Each derived bound is computed exactly, then rounded OUTWARD to a double (upper bounds up, lower
    bounds down) before it is stored: the stored bound is still valid (only looser, by <= 1 ulp),
    and every stored value stays a dyadic rational. Without the rounding, the divisions make the
    denominators grow with every pass (wood1p + objective cut: > 10 minutes in gcd); with it, the
    cost per pass is bounded. Independent of the C++ implementation (src/core/implied_bounds.cpp).
    Returns (lo, up) lists with None for "no finite bound"."""
    from fractions import Fraction
    cache = getattr(m, "_implied_cache", None)
    if cache is not None:
        return cache
    lo = [Fraction(v) if math.isfinite(v) else None for v in m.col_lower]
    up = [Fraction(v) if math.isfinite(v) else None for v in m.col_upper]
    rows = [[] for _ in range(m.num_rows)]
    for jj in range(m.num_cols):
        for k in range(m.col_start[jj], m.col_start[jj + 1]):
            rows[m.row_index[k]].append((jj, Fraction(m.value[k]), m.value[k]))
    # Speed without giving up rigour: (1) worklist — a row is re-examined only when one of its
    # columns' bounds changed; (2) a floating-point estimate first, the exact computation only when
    # the estimate promises an improvement; (3) an improvement must exceed 1e-9 (1 + |old|), which
    # stops endless micro-tightening. (2) and (3) can only make a bound looser, never invalid:
    # every stored bound is still the exact, outward-rounded result of one propagation step.
    flo = [float(v) if v is not None else None for v in lo]
    fup = [float(v) if v is not None else None for v in up]
    dirty = set(range(m.num_rows))
    for _ in range(passes):
        if not dirty:
            break
        changed_cols = set()
        for i in sorted(dirty):
            for sgn, rhs in ((1, m.row_upper[i]), (-1, m.row_lower[i])):
                if not math.isfinite(rhs):
                    continue
                unb, unb_col, ftotal = 0, -1, 0.0
                for kk, v, fv in rows[i]:
                    b = flo[kk] if sgn * fv > 0 else fup[kk]
                    if b is None:
                        unb += 1
                        unb_col = kk
                        if unb > 1:
                            break
                    else:
                        ftotal += sgn * fv * b
                if unb > 1:
                    continue
                total = None  # exact minimum of sgn*a_i x over the box, computed on first need
                R = Fraction(sgn) * Fraction(rhs)
                for kk, v, fv in rows[i]:
                    if unb == 1 and kk != unb_col:
                        continue
                    fc = sgn * fv
                    fown = flo[kk] if fc > 0 else fup[kk]
                    old = fup[kk] if fc > 0 else flo[kk]
                    est = (sgn * rhs - (ftotal if fown is None else ftotal - fc * fown)) / fc
                    thr = 1e-9 * (1.0 + abs(old)) if old is not None else 0.0
                    if old is not None and math.isfinite(est) and (est >= old - thr if fc > 0 else est <= old + thr):
                        continue
                    if total is None:
                        total = Fraction(0)
                        for k2, v2, _ in rows[i]:
                            c2 = sgn * v2
                            b2 = lo[k2] if c2 > 0 else up[k2]
                            if b2 is not None:
                                total += c2 * b2
                    c = sgn * v
                    own = lo[kk] if c > 0 else up[kk]
                    rest = total if own is None else total - c * own
                    bound = _round_out((R - rest) / c, up=c > 0)
                    if bound is None:
                        continue
                    fb = float(bound)
                    if c > 0 and (up[kk] is None or fb < fup[kk] - thr):
                        up[kk], fup[kk] = bound, fb
                        changed_cols.add(kk)
                    elif c < 0 and (lo[kk] is None or fb > flo[kk] + thr):
                        lo[kk], flo[kk] = bound, fb
                        changed_cols.add(kk)
                    # 'total' keeps kk's previous bound for the rest of this row: still valid, only looser
        dirty = {m.row_index[k] for jj in changed_cols for k in range(m.col_start[jj], m.col_start[jj + 1])}
    m._implied_cache = (lo, up)
    return lo, up


def verify_certificate(m, s, tol: dict, rep: dict) -> dict:
    """Independent check of an Infeasible (Farkas dual ray) or Unbounded (feasible point + primal
    ray) claim, with THIS module's reader. Infeasible: L0(r) = sum_i min_{s in [rl_i, ru_i]} r_i s
    + sum_j min_{x in [l_j, u_j]} (-A^T r)_j x > 0 proves that no feasible x exists; evaluated
    EXACTLY in rational arithmetic (every double is a rational) when the model is small enough,
    else / if the exact test cannot decide, the tolerance test of include/ps26119/tolerances.h
    (violation <= kVerifyRay * L0 after normalising). Unbounded: x primal-feasible within
    kVerifyPrimal and d in the recession cone with (sense c)^T d < 0 (kVerifyRay)."""
    from fractions import Fraction
    reasons = []
    inf = float("inf")
    n, mr = m.num_cols, m.num_rows
    rep["certificate"] = "none"
    if rep["status"] == "Infeasible":
        r = list(s.dual_ray)
        if len(r) != mr or mr == 0:
            rep.update(verdict="FAIL", reasons=["Infeasible without a dual ray (Farkas certificate)"])
            return rep
        exact_ok = None
        if m.nnz <= EXACT_MAX_NNZ and all(math.isfinite(v) for v in r):
            # Multipliers pointing at an infinite row bound are set to 0 first: L0 is a valid
            # bound for ANY r, so this only removes -inf terms (rounding-sized wrong signs on
            # inactive rows); lam = -A^T r is computed from the modified r.
            R = [Fraction(0) if (v > 0 and not math.isfinite(m.row_lower[i])) or
                 (v < 0 and not math.isfinite(m.row_upper[i])) else Fraction(v) for i, v in enumerate(r)]
            L0 = Fraction(0)
            bad = False
            for i in range(mr):
                if R[i] > 0:
                    if math.isfinite(m.row_lower[i]): L0 += R[i] * Fraction(m.row_lower[i])
                    else: bad = True
                elif R[i] < 0:
                    if math.isfinite(m.row_upper[i]): L0 += R[i] * Fraction(m.row_upper[i])
                    else: bad = True
            lam = [Fraction(0)] * n
            for j in range(n):
                acc = Fraction(0)
                for k in range(m.col_start[j], m.col_start[j + 1]):
                    acc += Fraction(m.value[k]) * R[m.row_index[k]]
                lam[j] = -acc
            implied = 0
            for j in range(n):
                if lam[j] > 0:
                    if math.isfinite(m.col_lower[j]): L0 += lam[j] * Fraction(m.col_lower[j])
                    else:
                        b = _implied_bounds(m)[0][j]
                        if b is None: bad = True
                        else: L0 += lam[j] * b; implied += 1
                elif lam[j] < 0:
                    if math.isfinite(m.col_upper[j]): L0 += lam[j] * Fraction(m.col_upper[j])
                    else:
                        b = _implied_bounds(m)[1][j]
                        if b is None: bad = True
                        else: L0 += lam[j] * b; implied += 1
            exact_ok = (not bad) and L0 > 0
            rep["farkas_exact_L0"] = None if bad else float(L0)
            rep["farkas_implied_bounds"] = implied
        if exact_ok:
            rep["certificate"] = "Farkas, exact rational check" + (
                f" (with {rep['farkas_implied_bounds']} implied column bounds)" if rep.get("farkas_implied_bounds") else "")
        else:
            rr = np.asarray(r, float)
            rmax = float(np.abs(rr).max(initial=0.0))
            obj, viol = 0.0, 0.0
            for i in range(mr):
                if rr[i] > 0:
                    if math.isfinite(m.row_lower[i]): obj += rr[i] * m.row_lower[i]
                    else: viol = max(viol, rr[i])
                elif rr[i] < 0:
                    if math.isfinite(m.row_upper[i]): obj += rr[i] * m.row_upper[i]
                    else: viol = max(viol, -rr[i])
            val = np.asarray(m.value, float)
            ridx = np.asarray(m.row_index, dtype=np.int64)
            cidx = np.repeat(np.arange(n), np.diff(np.asarray(m.col_start, dtype=np.int64)))
            lam = -np.bincount(cidx, weights=val * rr[ridx], minlength=n) if n else np.zeros(0)
            for j in range(n):
                if lam[j] > 0:
                    if math.isfinite(m.col_lower[j]): obj += lam[j] * m.col_lower[j]
                    else: viol = max(viol, lam[j])
                elif lam[j] < 0:
                    if math.isfinite(m.col_upper[j]): obj += lam[j] * m.col_upper[j]
                    else: viol = max(viol, -lam[j])
            scale = max(rmax, viol)
            L = obj / scale if scale > 0 else 0.0
            v = viol / scale if scale > 0 else 0.0
            rep["farkas_tolerance_L0"] = L
            rep["farkas_tolerance_violation"] = v
            if math.isfinite(L) and L > 0 and v <= tol["kVerifyRay"] * L:
                rep["certificate"] = "Farkas, tolerance check (not exact)"
            else:
                reasons.append(f"Farkas certificate fails: L0/s = {L:.3e}, violation/s = {v:.3e}")
    else:  # Unbounded
        d = list(s.primal_ray)
        if len(d) != n or len(s.x) != n or n == 0:
            rep.update(verdict="FAIL", reasons=["Unbounded without a primal ray and point"])
            return rep
        x = np.asarray(s.x, float)
        dd = np.asarray(d, float)
        val = np.asarray(m.value, float)
        ridx = np.asarray(m.row_index, dtype=np.int64)
        cidx = np.repeat(np.arange(n), np.diff(np.asarray(m.col_start, dtype=np.int64)))
        cl, cu = np.asarray(m.col_lower, float), np.asarray(m.col_upper, float)
        rl, ru = np.asarray(m.row_lower, float), np.asarray(m.row_upper, float)
        ax = np.bincount(ridx, weights=val * x[cidx], minlength=mr) if mr else np.zeros(0)
        def rel_viol(v, lo, up):
            out = 0.0
            for a, l, u in zip(v, lo, up):
                if math.isfinite(l) and a < l: out = max(out, (l - a) / (1 + abs(l)))
                if math.isfinite(u) and a > u: out = max(out, (a - u) / (1 + abs(u)))
            return out
        prim = max(rel_viol(x, cl, cu), rel_viol(ax, rl, ru))
        dmax = float(np.abs(dd).max(initial=0.0))
        if not math.isfinite(dmax) or dmax == 0:
            reasons.append("primal ray is zero or not finite")
        else:
            dd = dd / dmax
            ad = np.bincount(ridx, weights=val * dd[cidx], minlength=mr) if mr else np.zeros(0)
            cd = float(m.sense * np.dot(np.asarray(m.obj, float), dd))
            viol = 0.0
            viol = max(viol, float(np.max(np.where(np.isfinite(cl), -dd, 0.0), initial=0.0)))
            viol = max(viol, float(np.max(np.where(np.isfinite(cu), dd, 0.0), initial=0.0)))
            if mr:
                viol = max(viol, float(np.max(np.where(np.isfinite(rl), -ad, 0.0), initial=0.0)))
                viol = max(viol, float(np.max(np.where(np.isfinite(ru), ad, 0.0), initial=0.0)))
            cmax = max(1.0, float(np.abs(np.asarray(m.obj, float)).max(initial=0.0)))
            rep.update(ray_decrease=-cd, ray_cone_violation=viol, point_primal_rel=prim)
            if prim > tol["kVerifyPrimal"]:
                reasons.append(f"point not feasible: primal violation {prim:.3e} > {tol['kVerifyPrimal']:.0e}")
            if -cd < tol["kVerifyRay"] * cmax:
                reasons.append(f"ray does not decrease the objective: -(sense c)'d = {-cd:.3e}")
            if viol > tol["kVerifyRay"] * max(-cd, 0.0):
                reasons.append(f"ray leaves the recession cone: violation {viol:.3e}")
            if not reasons:
                rep["certificate"] = "ray + feasible point, tolerance check"
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
    if "certificate" in rep:
        print(f"certificate {rep['certificate']}")
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
