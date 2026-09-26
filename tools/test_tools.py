#!/usr/bin/env python3
"""test_tools.py — tests for the tooling: MPS→lpm bridge, fingerprint, verifier.

Run: python3 tools/test_tools.py   (CI runs it; needs highspy + numpy)
"""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import lpm  # noqa: E402
import verify  # noqa: E402
from highs_ref import solve_with_highs  # noqa: E402

NETLIB = os.path.join(ROOT, "data", "netlib_small")
HAND = os.path.join(ROOT, "data", "hand")


def optima():
    out = {}
    with open(os.path.join(NETLIB, "optima.csv")) as f:
        next(f)
        for line in f:
            name, _, _, opt, _ = line.strip().split(",", 4)
            out[name] = float(opt)
    return out


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_afiro_sizes(self):
        m = lpm.read_mps_highspy(os.path.join(NETLIB, "afiro.mps"))
        self.assertEqual((m.num_rows, m.num_cols, m.nnz), (27, 32, 83))

    def test_round_trip_all_small_netlib(self):
        for name in optima():
            src = lpm.read_mps_highspy(os.path.join(NETLIB, name + ".mps"))
            out = os.path.join(self.tmp, name + ".lpm")
            lpm.write_lpm(src, out)
            back = lpm.read_lpm(out)
            self.assertEqual(lpm.fingerprint(src), lpm.fingerprint(back), name)
            self.assertEqual(src.row_names, back.row_names)
            self.assertEqual(src.col_names, back.col_names)
            # committed .lpm must match a fresh conversion
            committed = lpm.read_lpm(os.path.join(NETLIB, name + ".lpm"))
            self.assertEqual(lpm.fingerprint(committed), lpm.fingerprint(src), name)

    def test_mps_mapping_ranges_offset_sense(self):
        m = lpm.read_mps_highspy(os.path.join(HAND, "max_ranged.mps"))
        self.assertEqual(m.sense, -1)
        self.assertEqual(m.obj_offset, 7.5)  # RHS on objective row -> offset = -value
        # E row with R>0: [rhs, rhs+|R|]
        self.assertEqual((m.row_lower[1], m.row_upper[1]), (-1.0, 2.0))
        self.assertEqual((m.row_lower[0], m.row_upper[0]), (-lpm.INF, 4.0))
        self.assertEqual((m.row_lower[2], m.row_upper[2]), (1.0, lpm.INF))
        self.assertEqual((m.col_lower[1], m.col_upper[1]), (-lpm.INF, lpm.INF))

    def test_fingerprint_ignores_in_column_order(self):
        m = lpm.read_mps_highspy(os.path.join(NETLIB, "afiro.mps"))
        fp = lpm.fingerprint(m)
        j = next(j for j in range(m.num_cols) if m.col_start[j + 1] - m.col_start[j] >= 2)
        a, b = m.col_start[j], m.col_start[j + 1]
        m.row_index[a:b] = m.row_index[a:b][::-1]
        m.value[a:b] = m.value[a:b][::-1]
        self.assertEqual(fp, lpm.fingerprint(m))
        m.value[a] += 1e-12
        self.assertNotEqual(fp, lpm.fingerprint(m))


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _highs(self, path):
        out = os.path.join(self.tmp, os.path.basename(path) + ".sol")
        solve_with_highs(path, out)
        return out

    def test_highs_solutions_pass_with_published_optima(self):
        for name, opt in optima().items():
            path = os.path.join(NETLIB, name + ".mps")
            rep = verify.verify(path, self._highs(path), opt)
            self.assertEqual(rep["verdict"], "PASS", (name, rep["reasons"]))
            self.assertTrue(rep["model_match"])

    def test_max_problem_sign_convention(self):
        path = os.path.join(HAND, "max_ranged.mps")
        rep = verify.verify(path, self._highs(path))
        self.assertEqual(rep["verdict"], "PASS", rep["reasons"])

    def _rewrite(self, sol_path, fn):
        s = lpm.read_solution(sol_path)
        fn(s)
        out = sol_path + ".mod"
        with open(out, "w") as f:
            f.write("PS26119-SOLUTION 1\n")
            for k, v in s.header.items():
                f.write(f"{k} {v}\n")
            f.write(f"COLUMNS {len(s.x)}\n")
            for j, (x, z) in enumerate(zip(s.x, s.z)):
                f.write(f"{j} {x!r} {z!r}\n")
            f.write(f"ROWS {len(s.y)}\n")
            for i, (a, y) in enumerate(zip(s.row_activity, s.y)):
                f.write(f"{i} {a!r} {y!r}\n")
            f.write("END\n")
        return out

    def test_detects_primal_violation(self):
        path = os.path.join(NETLIB, "afiro.mps")
        bad = self._rewrite(self._highs(path), lambda s: s.x.__setitem__(1, s.x[1] + 1.0))
        rep = verify.verify(path, bad)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertGreater(rep["primal_rel"], 1e-6)

    def test_detects_wrong_dual_sign(self):
        path = os.path.join(NETLIB, "afiro.mps")
        def flip(s):
            s.y[:] = [-v for v in s.y]
        rep = verify.verify(path, self._rewrite(self._highs(path), flip))
        self.assertEqual(rep["verdict"], "FAIL")

    def test_detects_wrong_expected_objective(self):
        path = os.path.join(NETLIB, "afiro.mps")
        rep = verify.verify(path, self._highs(path), -464.0)
        self.assertEqual(rep["verdict"], "FAIL")

    def test_non_optimal_status_fails(self):
        path = os.path.join(NETLIB, "afiro.mps")
        def st(s):
            s.header["status"] = "IterationLimit"
        rep = verify.verify(path, self._rewrite(self._highs(path), st))
        self.assertEqual(rep["verdict"], "FAIL")


class CertificateVerifier(unittest.TestCase):
    """Infeasible / Unbounded certificates (verify.verify_certificate): exact proofs pass,
    and — soundness — no certificate is ever accepted for a model that is actually feasible."""

    def _model(self, c, rows, rl, ru, cl, cu, sense=1):
        m = lpm.PyModel()
        m.num_rows, m.num_cols, m.sense = len(rows), len(c), sense
        m.obj, m.col_lower, m.col_upper = list(map(float, c)), list(map(float, cl)), list(map(float, cu))
        m.row_lower, m.row_upper = list(map(float, rl)), list(map(float, ru))
        m.col_start, m.row_index, m.value = [0], [], []
        for j in range(len(c)):
            for i, row in enumerate(rows):
                if row[j] != 0:
                    m.row_index.append(i)
                    m.value.append(float(row[j]))
            m.col_start.append(len(m.value))
        return m

    def _check(self, m, status, dual_ray=None, x=None, ray=None):
        s = lpm.PySolution()
        s.header["status"] = status
        s.dual_ray = dual_ray or []
        s.x = x or []
        s.primal_ray = ray or []
        rep = {"status": status}
        return verify.verify_certificate(m, s, verify.load_tolerances(), rep)

    def test_exact_farkas_needs_implied_bounds(self):
        # x + y <= 1 and x + y >= 3 with x, y >= 0 (no upper bounds): exactly r = (-1, 1).
        inf = float("inf")
        m = self._model([0, 0], [[1, 1], [1, 1]], [-inf, 3], [1, inf], [0, 0], [inf, inf])
        rep = self._check(m, "Infeasible", dual_ray=[-1.0, 1.0])
        self.assertEqual(rep["verdict"], "PASS")
        self.assertIn("exact rational", rep["certificate"])
        # r = (-1, 1 + 1e-16): -A^T r = -1e-16 on both columns points at their infinite upper
        # bounds; the exact test must use the implied bounds x, y <= 1 (from row 0) to prove it.
        rep = self._check(m, "Infeasible", dual_ray=[-1.0, 1.0 + 2.220446049250313e-16])
        self.assertEqual(rep["verdict"], "PASS")
        self.assertIn("implied column bounds", rep["certificate"])
        self.assertEqual(self._check(m, "Infeasible", dual_ray=[1.0, -1.0])["verdict"], "FAIL")
        self.assertEqual(self._check(m, "Infeasible")["verdict"], "FAIL")  # no certificate

    def test_implied_bounds_valid_dyadic_and_fast(self):
        # Chain 3 x_{k+1} - x_k <= 0, x_0 in [0, 1], x_k >= 0: the tight bounds are 3^-k. Exact
        # propagation without rounding grows denominators 3^k per pass; the stored bounds must be
        # doubles (power-of-two denominators), valid (>= 3^-k) and at most slightly looser.
        from fractions import Fraction
        import time
        inf, n = float("inf"), 40
        rows = [[(3 if j == k + 1 else -1 if j == k else 0) for j in range(n)] for k in range(n - 1)]
        m = self._model([0] * n, rows, [-inf] * (n - 1), [0] * (n - 1), [0] * n, [1] + [inf] * (n - 1))
        t = time.time()
        lo, up = verify._implied_bounds(m)
        self.assertLess(time.time() - t, 5.0)
        for k in range(1, 21):  # 20 passes reach at least x_20
            self.assertIsNotNone(up[k])
            self.assertGreaterEqual(up[k], Fraction(1, 3 ** k))
            self.assertLessEqual(float(up[k]), (1 + 1e-6) / 3 ** k)
            d = up[k].denominator
            self.assertEqual(d & (d - 1), 0, "bound is not a double")
        # validity on random feasible models: a known feasible x0 satisfies every implied bound
        import random
        rng = random.Random(11)
        for _ in range(200):
            mr, nc = rng.randint(1, 5), rng.randint(1, 5)
            x0 = [rng.uniform(-3, 3) for _ in range(nc)]
            rws = [[rng.choice([0, rng.uniform(-4, 4)]) for _ in range(nc)] for _ in range(mr)]
            ax = [sum(r[j] * x0[j] for j in range(nc)) for r in rws]
            rl = [a - rng.choice([1e-9, 1, inf]) for a in ax]
            ru = [a + rng.choice([1e-9, 1, inf]) for a in ax]
            cl = [x - rng.choice([0.5, inf]) for x in x0]
            cu = [x + rng.choice([0.5, inf]) for x in x0]
            mm = self._model([0] * nc, rws, rl, ru, cl, cu)
            lo2, up2 = verify._implied_bounds(mm)
            for j in range(nc):
                if lo2[j] is not None:
                    self.assertLessEqual(lo2[j], Fraction(x0[j]))
                if up2[j] is not None:
                    self.assertGreaterEqual(up2[j], Fraction(x0[j]))

    def test_soundness_no_certificate_for_feasible_models(self):
        import random
        rng = random.Random(3)
        inf = float("inf")
        accepted = 0
        for _ in range(300):
            mr, n = rng.randint(1, 4), rng.randint(1, 4)
            x0 = [rng.uniform(-2, 2) for _ in range(n)]
            rows = [[rng.choice([0, rng.randint(-3, 3)]) for _ in range(n)] for _ in range(mr)]
            ax = [sum(r[j] * x0[j] for j in range(n)) for r in rows]
            rl = [a - rng.choice([0, 1, inf]) for a in ax]
            ru = [a + rng.choice([0, 1, inf]) for a in ax]
            cl = [v - rng.choice([0.5, inf]) for v in x0]
            cu = [v + rng.choice([0.5, inf]) for v in x0]
            m = self._model([rng.randint(-2, 2) for _ in range(n)], rows, rl, ru, cl, cu)
            r = [rng.uniform(-5, 5) for _ in range(mr)]
            if self._check(m, "Infeasible", dual_ray=r)["verdict"] == "PASS":
                accepted += 1
        self.assertEqual(accepted, 0)  # a feasible model has no Farkas certificate

    def test_unbounded_ray(self):
        inf = float("inf")
        m = self._model([-1, 0], [[1, -1]], [-inf], [1], [0, 0], [inf, inf])
        self.assertEqual(self._check(m, "Unbounded", x=[0, 0], ray=[1, 1])["verdict"], "PASS")
        self.assertEqual(self._check(m, "Unbounded", x=[2, 0], ray=[1, 1])["verdict"], "FAIL")  # infeasible point
        self.assertEqual(self._check(m, "Unbounded", x=[0, 0], ray=[1, 0])["verdict"], "FAIL")  # leaves the cone
        self.assertEqual(self._check(m, "Unbounded", x=[0, 0])["verdict"], "FAIL")


class MilpVerifier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, x):
        p = os.path.join(self.tmp, "k.sol")
        with open(p, "w") as f:
            f.write("PS26119-SOLUTION 1\nstatus Optimal\nCOLUMNS 5\n")
            for j, v in enumerate(x):
                f.write(f"{j} {v!r} 0\n")
            f.write("ROWS 1\n0 0 0\nEND\n")
        return p

    def test_integral_optimum_passes_fractional_fails(self):
        mps = os.path.join(HAND, "knapsack_mip.mps")
        good = verify.verify(mps, self._write([1, 1, 1, 0, 0]), 28.0)  # 5+3+4 = 12, value 28
        self.assertTrue(good["mip"])
        self.assertEqual(good["verdict"], "PASS", good["reasons"])
        frac = verify.verify(mps, self._write([1, 1, 0.5, 0, 0.3]), None)
        self.assertEqual(frac["verdict"], "FAIL")
        over = verify.verify(mps, self._write([1, 1, 1, 1, 0]), None)  # capacity 14 > 12
        self.assertEqual(over["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=1)
