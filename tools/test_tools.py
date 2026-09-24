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


if __name__ == "__main__":
    unittest.main(verbosity=1)
