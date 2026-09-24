#!/usr/bin/env python3
"""test_generators.py — generated LPs really have the optimum the sidecar claims.

For small instances of both generators: HiGHS (independent) reaches the constructed
optimum, and the .lpm round-trips through tools/lpm.py. Run by ctest when highspy exists.
"""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import generate_lp  # noqa: E402
import generate_refinery_lp  # noqa: E402
import lpgen  # noqa: E402
from highs_ref import solve_with_highs  # noqa: E402
from lpm import read_lpm  # noqa: E402


class Generators(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, g, name):
        path = os.path.join(self.tmp, name + ".lpm")
        lpgen.write_lpm_fast(path, name, g["sense"], g["c"], g["col_lower"], g["col_upper"], g["row_lower"],
                             g["row_upper"], g["col_start"], g["row_index"], g["value"])
        return path

    def _check(self, g, name):
        path = self._write(g, name)
        m = read_lpm(path)
        self.assertEqual((m.num_rows, m.num_cols, m.nnz), (g["m"], g["n"], len(g["value"])))
        r = solve_with_highs(path, os.path.join(self.tmp, name + ".sol"))
        self.assertEqual(r["status"], "Optimal")
        self.assertAlmostEqual(r["objective"], g["optimum"], delta=1e-7 * (1 + abs(g["optimum"])))

    def test_random_lps(self):
        for seed in (1, 2, 3):
            for m in (50, 400):
                self._check(generate_lp.generate(m, seed=seed), f"rand{m}_{seed}")

    def test_refinery(self):
        for T in (1, 2, 12, 30):
            g = generate_refinery_lp.build(T, seed=T)
            self.assertEqual(g["sense"], -1)
            self._check(g, f"ref{T}")

    def test_refinery_sizes(self):
        g = generate_refinery_lp.build(24, 1)
        self.assertEqual(g["m"], 24 * g["rows_per_period"])
        self.assertEqual(g["n"], 24 * g["cols_per_period"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
