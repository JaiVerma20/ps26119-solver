#!/usr/bin/env python3
"""generate_refinery_lp.py — T-period refinery planning LP, optimum known by construction.

usage: generate_refinery_lp.py --periods 365 [--seed 1] [--out bench/generated/refinery-T365.lpm]
       (T = 12 monthly, 365 daily, 8760 hourly)

Model per period t (all flows ≥ 0; MAX profit):
  crudes k (6):   buy[k,t], cdu[k,t], cinv[k,t]
     crude balance    cinv[k,t] − cinv[k,t−1] − buy[k,t] + cdu[k,t] = 0     (cinv[k,−1] = data)
  CDU:            Σ_k cdu[k,t] ≤ CDU capacity
     yields           prod[j,t] − Σ_k Y[k,j]·cdu[k,t] = 0     cuts j: LPG NAP KERO DSL VGO RES
  FCC:            fcc[t] ≤ FCC capacity;  fout[o,t] − FY[o]·fcc[t] = 0   o: LPG GAS LCO SLURRY
  components c (10) with balances  supply_c − Σ_{arcs c→p} blend[c,p,t] = 0
     (VGO supply is prod[VGO] − fcc; LPG = prod[LPG] + fout[LPG]; ...)
  products p (5):  out[p,t] − Σ_{arcs c→p} blend[c,p,t] = 0      LPG GASOLINE JET DIESEL FUELOIL
  quality specs (linear blending, P-formulation):
     Σ_{c→p} (q_c − s_p)·blend[c,p,t] ≤ 0 (max specs: sulphur, viscosity index)
                                       ≥ 0 (min specs: octane)
  product tanks:  pinv[p,t] − pinv[p,t−1] − out[p,t] + sell[p,t] = 0,  0 ≤ pinv ≤ tank
  demand:         sell[p,t] ≤ demand[p,t]

Construction (lpgen.py): a physically consistent flow pattern x* is simulated period by
period (all balances hold); capacity, demand and spec rows are active in a random subset of
periods (bottlenecks, zero quality giveaway) and slack elsewhere; some crude purchases sit
at their availability limit and some tanks at capacity. Balance rows keep rhs 0 (or the
initial inventory in period 0). Right-hand sides of inequality rows and the prices c come
out of the KKT construction, so the optimum is known exactly; the matrix structure is the
refinery's. Honest caveat: prices are synthetic (derived from chosen shadow prices).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lpgen  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CUTS = ["LPG", "NAP", "KERO", "DSL", "VGO", "RES"]
FCC_OUT = ["LPG", "GAS", "LCO", "SLURRY"]
COMPS = ["lpg", "nap", "kero", "dsl", "vgo", "res", "fcc_lpg", "fcc_gas", "lco", "slurry"]
PRODS = ["LPG", "GASOLINE", "JET", "DIESEL", "FUELOIL"]
ARCS = [("lpg", "LPG"), ("fcc_lpg", "LPG"), ("nap", "GASOLINE"), ("fcc_gas", "GASOLINE"), ("lpg", "GASOLINE"),
        ("kero", "JET"), ("kero", "DIESEL"), ("dsl", "DIESEL"), ("lco", "DIESEL"), ("vgo", "FUELOIL"),
        ("res", "FUELOIL"), ("slurry", "FUELOIL"), ("lco", "FUELOIL"), ("dsl", "FUELOIL"), ("nap", "FUELOIL")]
# specs: (product, quality index, direction) — quality 0 = sulphur-like (max), 1 = octane-like (min)
SPECS = [("GASOLINE", 0, "le"), ("GASOLINE", 1, "ge"), ("JET", 0, "le"), ("DIESEL", 0, "le"),
         ("FUELOIL", 0, "le"), ("FUELOIL", 1, "ge")]


class Layout:
    """Per-period variable/row layout; global index = t·size + local."""

    def __init__(self):
        K, J, O, P, A = 6, len(CUTS), len(FCC_OUT), len(PRODS), len(ARCS)
        self.K, self.J, self.O, self.P, self.A = K, J, O, P, A
        c = 0
        self.buy = np.arange(c, c + K); c += K
        self.cdu = np.arange(c, c + K); c += K
        self.cinv = np.arange(c, c + K); c += K
        self.prod = np.arange(c, c + J); c += J
        self.fcc = c; c += 1
        self.fout = np.arange(c, c + O); c += O
        self.blend = np.arange(c, c + A); c += A
        self.out = np.arange(c, c + P); c += P
        self.pinv = np.arange(c, c + P); c += P
        self.sell = np.arange(c, c + P); c += P
        self.ncols = c
        r = 0
        self.r_cbal = np.arange(r, r + K); r += K
        self.r_cdu = r; r += 1
        self.r_yield = np.arange(r, r + J); r += J
        self.r_fcccap = r; r += 1
        self.r_fy = np.arange(r, r + O); r += O
        self.r_comp = np.arange(r, r + len(COMPS)); r += len(COMPS)
        self.r_prod = np.arange(r, r + P); r += P
        self.r_spec = np.arange(r, r + len(SPECS)); r += len(SPECS)
        self.r_pbal = np.arange(r, r + P); r += P
        self.r_dem = np.arange(r, r + P); r += P
        self.nrows = r


def build(T: int, seed: int):
    rng = np.random.default_rng(seed)
    L = Layout()
    K, J, O, P = L.K, L.J, L.O, L.P
    comp_i = {c: i for i, c in enumerate(COMPS)}
    prod_i = {p: i for i, p in enumerate(PRODS)}

    # ---------------- data
    Y = rng.dirichlet(np.ones(J) * 3, K)                # crude-specific yields, rows sum to 1
    FY = np.array([0.15, 0.50, 0.25, 0.10])             # FCC yields
    q = np.stack([rng.uniform(0.1, 3.0, len(COMPS)),     # sulphur-like
                  rng.uniform(60, 100, len(COMPS))], 1)  # octane-like
    cdu_cap = 1000.0

    # ---------------- per-period matrix template: (row, col, val, col_period_shift)
    R, C, V, S = [], [], [], []

    def add(r, c, v, shift=0):
        R.append(r), C.append(c), V.append(v), S.append(shift)

    for k in range(K):
        add(L.r_cbal[k], L.cinv[k], 1.0)
        add(L.r_cbal[k], L.cinv[k], -1.0, -1)
        add(L.r_cbal[k], L.buy[k], -1.0)
        add(L.r_cbal[k], L.cdu[k], 1.0)
        add(L.r_cdu, L.cdu[k], 1.0)
    for j in range(J):
        add(L.r_yield[j], L.prod[j], 1.0)
        for k in range(K):
            add(L.r_yield[j], L.cdu[k], -Y[k, j])
    add(L.r_fcccap, L.fcc, 1.0)
    for o in range(O):
        add(L.r_fy[o], L.fout[o], 1.0)
        add(L.r_fy[o], L.fcc, -FY[o])
    supply = {  # component -> list of (col, coef)
        "lpg": [(L.prod[0], 1.0)], "nap": [(L.prod[1], 1.0)], "kero": [(L.prod[2], 1.0)], "dsl": [(L.prod[3], 1.0)],
        "vgo": [(L.prod[4], 1.0), (L.fcc, -1.0)], "res": [(L.prod[5], 1.0)], "fcc_lpg": [(L.fout[0], 1.0)],
        "fcc_gas": [(L.fout[1], 1.0)], "lco": [(L.fout[2], 1.0)], "slurry": [(L.fout[3], 1.0)],
    }
    for c, terms in supply.items():
        for col, coef in terms:
            add(L.r_comp[comp_i[c]], col, coef)
    for a, (c, p) in enumerate(ARCS):
        add(L.r_comp[comp_i[c]], L.blend[a], -1.0)
        add(L.r_prod[prod_i[p]], L.blend[a], -1.0)
    for p in range(P):
        add(L.r_prod[p], L.out[p], 1.0)
    # spec limits s_p: set from a typical blend so both signs of (q_c − s) occur
    spec_limit = []
    for s, (p, qi, d) in enumerate(SPECS):
        comps = [comp_i[c] for c, pp in ARCS if pp == p]
        lim = float(np.mean(q[comps, qi]))
        spec_limit.append(lim)
        for a, (c, pp) in enumerate(ARCS):
            if pp == p:
                add(L.r_spec[s], L.blend[a], q[comp_i[c], qi] - lim)
    for p in range(P):
        add(L.r_pbal[p], L.pinv[p], 1.0)
        add(L.r_pbal[p], L.pinv[p], -1.0, -1)
        add(L.r_pbal[p], L.out[p], -1.0)
        add(L.r_pbal[p], L.sell[p], 1.0)
        add(L.r_dem[p], L.sell[p], 1.0)
    R, C, V, S = map(np.asarray, (R, C, V, S))

    # tile over periods
    t = np.repeat(np.arange(T), len(R))
    rows = np.tile(R, T) + t * L.nrows
    tc = t + np.tile(S, T)
    keep = tc >= 0
    cols = (np.tile(C, T) + tc * L.ncols)[keep]
    rows, vals = rows[keep], np.tile(V, T)[keep]
    m, n = T * L.nrows, T * L.ncols
    col_start, row_index, value = lpgen.csc_from_triplets(m, n, rows, cols, vals)

    # ---------------- consistent flows x*, period by period (vectorised over periods where possible)
    x = np.zeros((T, L.ncols))
    share = rng.dirichlet(np.ones(K), T)
    load = rng.uniform(0.6, 1.0, T)
    cdu_active = rng.random(T) < 0.3                     # CDU is the bottleneck in 30% of periods
    load[cdu_active] = 1.0
    x[:, L.cdu] = share * (cdu_cap * load)[:, None]
    x[:, L.prod] = x[:, L.cdu] @ Y
    fcc_frac = rng.uniform(0.5, 0.9, T)
    x[:, L.fcc] = fcc_frac * x[:, L.prod[4]]
    fcc_cap_active = rng.random(T) < 0.3
    x[:, L.fout] = x[:, [L.fcc]] * FY[None, :]
    sup = np.zeros((T, len(COMPS)))
    for c, terms in supply.items():
        for col, coef in terms:
            sup[:, comp_i[c]] += coef * x[:, col]
    for c in COMPS:  # split each component over its arcs
        arcs = [a for a, (cc, _) in enumerate(ARCS) if cc == c]
        fr = rng.dirichlet(np.ones(len(arcs)), T)
        parts = fr * sup[:, [comp_i[c]]]
        parts[:, -1] = sup[:, comp_i[c]] - parts[:, :-1].sum(1)
        x[:, L.blend[arcs]] = parts
    for p, pn in enumerate(PRODS):
        arcs = [a for a, (_, pp) in enumerate(ARCS) if pp == pn]
        x[:, L.out[p]] = x[:, L.blend[arcs]].sum(1)
    # crude: buy around consumption, inventory stays within [0, tank]
    crude_tank, prod_tank = 3000.0, 2000.0
    cinv0 = np.full(K, 500.0)
    pinv0 = np.full(P, 300.0)
    ci, pi = cinv0.copy(), pinv0.copy()
    for tt in range(T):
        buy = np.clip(x[tt, L.cdu] * rng.uniform(0.7, 1.3, K), 0, None)
        buy = np.minimum(buy, crude_tank - ci + x[tt, L.cdu])  # keep crude inventory ≤ tank
        ci = ci + buy - x[tt, L.cdu]
        low = ci < 0
        buy[low] -= ci[low]
        ci[low] = 0.0
        x[tt, L.buy], x[tt, L.cinv] = buy, ci
        sell = x[tt, L.out] * rng.uniform(0.7, 1.3, P)
        pi_new = pi + x[tt, L.out] - sell
        neg = pi_new < 0
        sell[neg] += pi_new[neg]
        pi_new[neg] = 0.0
        over = pi_new > prod_tank
        sell[over] += pi_new[over] - prod_tank
        pi_new[over] = prod_tank
        pi = pi_new
        x[tt, L.sell], x[tt, L.pinv] = sell, pi
    x = np.maximum(x, 0.0)
    xf = x.ravel()

    # ---------------- statuses and bounds
    col_lower = np.zeros(n)
    col_upper = np.full(n, np.inf)
    col_at = np.zeros(n, dtype=np.int8)
    col_at[xf == 0.0] = -1                                  # unused flows sit at lower bound 0
    tidx = np.arange(T)[:, None] * L.ncols
    buy_cols = (tidx + L.buy[None, :]).ravel()
    cap_buy = rng.random(buy_cols.size) < 0.2                # crude availability binding
    cap_buy &= xf[buy_cols] > 0
    col_upper[buy_cols] = np.where(cap_buy, xf[buy_cols], xf[buy_cols] * 1.5 + 50.0)
    col_at[buy_cols[cap_buy]] = 1
    cinv_cols = (tidx + L.cinv[None, :]).ravel()
    col_upper[cinv_cols] = crude_tank
    pinv_cols = (tidx + L.pinv[None, :]).ravel()
    col_upper[pinv_cols] = prod_tank
    col_at[pinv_cols[xf[pinv_cols] >= prod_tank]] = 1
    col_at[cinv_cols[xf[cinv_cols] >= crude_tank]] = 1

    row_type = np.empty(m, dtype=object)
    row_active = np.zeros(m, dtype=bool)
    tr = np.arange(T)[:, None] * L.nrows
    eq_rows = np.concatenate([(tr + L.r_cbal).ravel(), (tr + L.r_yield).ravel(), (tr + L.r_fy).ravel(),
                              (tr + L.r_comp).ravel(), (tr + L.r_prod).ravel(), (tr + L.r_pbal).ravel()])
    row_type[eq_rows] = "eq"
    cdu_rows = np.arange(T) * L.nrows + L.r_cdu
    row_type[cdu_rows] = "le"
    row_active[cdu_rows] = cdu_active
    fcc_rows = np.arange(T) * L.nrows + L.r_fcccap
    row_type[fcc_rows] = "le"
    row_active[fcc_rows] = fcc_cap_active
    dem_rows = (tr + L.r_dem).ravel()
    row_type[dem_rows] = "le"
    row_active[dem_rows] = rng.random(dem_rows.size) < 0.6
    spec_rows = (tr + L.r_spec).ravel()
    spec_dir = np.tile([d for _, _, d in SPECS], T)
    row_type[spec_rows] = spec_dir
    row_active[spec_rows] = rng.random(spec_rows.size) < 0.5
    row_type = row_type.astype(str)

    # Spec rows: the chosen x* may violate the nominal limit; the construction sets each
    # spec row's rhs to (activity) if active or (activity ± slack) if not, so x* is feasible.
    cert = lpgen.certify(rng, m, n, col_start, row_index, value, xf, col_lower, col_upper, col_at, row_type,
                         row_active, dual_scale=(0.5, 5.0), slack_scale=(0.05, 0.5))
    return {
        "m": m, "n": n, "c": -cert["c_min"], "sense": -1, "optimum": -cert["optimum_min"],
        "col_lower": col_lower, "col_upper": col_upper, "row_lower": cert["row_lower"],
        "row_upper": cert["row_upper"], "col_start": col_start, "row_index": row_index, "value": value,
        "rows_per_period": L.nrows, "cols_per_period": L.ncols,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "generated", f"refinery-T{a.periods}-s{a.seed}.lpm")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    g = build(a.periods, a.seed)
    name = os.path.splitext(os.path.basename(out))[0]
    lpgen.write_lpm_fast(out, name, g["sense"], g["c"], g["col_lower"], g["col_upper"], g["row_lower"],
                         g["row_upper"], g["col_start"], g["row_index"], g["value"],
                         source="bench/generate_refinery_lp.py")
    lpgen.write_sidecar(out, {"name": name, "generator": "generate_refinery_lp.py", "periods": a.periods,
                              "rows": g["m"], "cols": g["n"], "nnz": int(len(g["value"])), "seed": a.seed,
                              "sense": g["sense"], "optimum": g["optimum"],
                              "rows_per_period": g["rows_per_period"], "cols_per_period": g["cols_per_period"]})
    print(f"{out}: T={a.periods}  {g['m']} rows, {g['n']} cols, {len(g['value'])} nnz, "
          f"optimum {g['optimum']!r} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
