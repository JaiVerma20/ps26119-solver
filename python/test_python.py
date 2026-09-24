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
        for alg in ("oracle", "pdlp", "r2hpdhg"):
            r = ps26119.solve_lp([3, 5], A, [-INF] * 3, [4, 12, 18], sense=-1, algorithm=alg)
            self.assertEqual(r.status_name, "Optimal", r.message)
            self.assertAlmostEqual(r.objective, 36, delta=1e-6)
            np.testing.assert_allclose(r.x, [2, 6], atol=1e-5)
            np.testing.assert_allclose(r.y, [0, 1.5, 1], atol=1e-5)

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
        bad = ps26119.solve_lp([1], [[1]], [0], [1], col_lower=[5], col_upper=[1])
        self.assertEqual(bad.status, ps26119.INVALID_ARGUMENT)
        self.assertIn("invalid model", bad.message)

    def test_version(self):
        self.assertTrue(ps26119.version())


if __name__ == "__main__":
    unittest.main(verbosity=1)
