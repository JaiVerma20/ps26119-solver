#!/usr/bin/env python3
"""Tests for the Python binding (run by ctest when the shared library is built)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps26119  # noqa: E402

INF = np.inf


class Binding(unittest.TestCase):
    def test_wyndor_all_algorithms(self):
        A = [[1, 0], [0, 2], [3, 2]]
        for alg in ("oracle", "pdlp", "r2hpdhg", "simplex"):
            r = ps26119.solve_lp([3, 5], A, [-INF] * 3, [4, 12, 18], sense=-1, algorithm=alg)
            self.assertEqual(r.status_name, "Optimal", r.message)
            self.assertEqual(r.check, "PASS", alg)
            self.assertAlmostEqual(r.objective, 36, delta=1e-6)
            np.testing.assert_allclose(r.x, [2, 6], atol=1e-5)
            np.testing.assert_allclose(r.y, [0, 1.5, 1], atol=1e-5)
            self.assertGreaterEqual(r.certified_bound, 36 - 1e-9)  # MAX: certified upper bound
            self.assertLess(r.certified_bound, 36 + 1e-4)

    def test_csc_input_and_warm_start(self):
        cs, ri, v = [0, 2, 4], [0, 2, 1, 2], [1.0, 3.0, 2.0, 2.0]
        r = ps26119.solve_lp([3, 5], (cs, ri, v), [-INF] * 3, [4, 12, 18], sense=-1)
        w = ps26119.solve_lp([3, 5], (cs, ri, v), [-INF] * 3, [4, 12, 18], sense=-1, warm_x=r.x, warm_y=r.y)
        self.assertEqual(w.status, ps26119.OPTIMAL)
        self.assertLessEqual(w.iterations, r.iterations)

    def test_infeasible_and_invalid(self):
        r = ps26119.solve_lp([1, 1], [[1, 1], [1, 1]], [-INF, 3], [1, INF], algorithm="oracle")
        self.assertEqual(r.status, ps26119.INFEASIBLE)
        self.assertIsNone(r.x)
        bad = ps26119.solve_lp([1], [[1]], [0], [1], col_lower=[INF], col_upper=[INF])
        self.assertEqual(bad.status, ps26119.INVALID_ARGUMENT)
        self.assertIn("invalid model", bad.message)
        # crossed bounds: valid data, infeasible model (docs/DECISIONS.md #27)
        crossed = ps26119.solve_lp([1], [[1]], [0], [1], col_lower=[5], col_upper=[1])
        self.assertEqual(crossed.status, ps26119.INFEASIBLE)

    def test_solve_mps(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        afiro = os.path.join(root, "data", "netlib_small", "afiro.mps")
        out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ps26119_py_afiro.sol")
        for alg in ("simplex", "r2hpdhg"):
            r = ps26119.solve_mps(afiro, algorithm=alg, out=out)
            self.assertEqual(r.status, ps26119.OPTIMAL, r.message)
            self.assertAlmostEqual(r.objective, -464.75314286, delta=1e-5)
            self.assertEqual(r.check, "PASS")
            self.assertEqual(r.engine, alg)
            with open(out) as f:
                self.assertIn("status Optimal", f.read())
        bad = ps26119.solve_mps(os.path.join(root, "data", "no_such_file.mps"))
        self.assertEqual(bad.status, ps26119.INVALID_ARGUMENT)
        self.assertIn("read error", bad.message)

    def test_version(self):
        self.assertTrue(ps26119.version())


if __name__ == "__main__":
    unittest.main(verbosity=1)
