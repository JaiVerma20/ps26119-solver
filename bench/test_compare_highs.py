#!/usr/bin/env python3
"""test_compare_highs.py — the rules of bench/compare_highs.py (the same "solved" definition for
both solvers) and the HiGHS reference runner it relies on (tools/highs_ref.py)."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import compare_highs as C  # noqa: E402
import highs_ref  # noqa: E402
import verify  # noqa: E402

AFIRO = os.path.join(ROOT, "data", "netlib_small", "afiro.mps")


class Rules(unittest.TestCase):
    def row(self, **kw):
        return C.finish({"reference": "-464.75314285714285", **kw})

    def test_solved_needs_optimal_verified_and_the_reference(self):
        self.assertEqual(self.row(status="Optimal", verify="PASS", objective="-464.7531428571")["solved"], "yes")
        self.assertEqual(self.row(status="Optimal", verify="FAIL", objective="-464.7531428571")["solved"], "no")
        self.assertEqual(self.row(status="TimeLimit", verify="PASS", objective="-464.7531428571")["solved"], "no")
        self.assertEqual(self.row(status="Optimal", verify="PASS", objective="-464.70")["solved"], "no")  # 1e-4 off
        self.assertEqual(self.row(status="Optimal", verify="PASS", objective="nan")["solved"], "no")
        self.assertEqual(self.row(status="Error", verify="FAIL")["solved"], "no")


class HighsReference(unittest.TestCase):
    def test_each_engine_writes_a_verifiable_solution(self):
        with tempfile.TemporaryDirectory() as tmp:
            for eng, opts in C.HIGHS:
                sol = os.path.join(tmp, eng + ".sol")
                res = highs_ref.solve_with_highs(AFIRO, sol, options=opts)
                self.assertEqual(res["engine"], "highs-" + eng)
                self.assertGreater(res["iterations"], 0)
                self.assertTrue(res["highs_status"].startswith("k"))
                if eng != "pdlp":  # PDLP's own stopping test is looser; compare_highs records that as a row
                    self.assertEqual(verify.verify(AFIRO, sol)["verdict"], "PASS", eng)

    def test_unknown_option_is_an_error_not_a_silent_default(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            highs_ref.solve_with_highs(AFIRO, os.path.join(tmp, "x.sol"), options={"no_such_option": 1})


class OrToolsReference(unittest.TestCase):
    """OR-Tools runs in its own process (it bundles a HiGHS that clashes with highspy's)."""

    def test_glop_and_pdlp_write_verifiable_solutions(self):
        import json
        import subprocess
        if subprocess.run([sys.executable, "-c", "import ortools"], capture_output=True).returncode != 0:
            self.skipTest("OR-Tools not installed (optional reference)")
        with tempfile.TemporaryDirectory() as tmp:
            for eng in ("glop", "pdlp"):
                sol = os.path.join(tmp, eng + ".sol")
                p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ortools_ref.py"), AFIRO, sol, "--solver", eng,
                                    "--json"], capture_output=True, text=True, timeout=300)
                res = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1])
                self.assertEqual((res["status"], res["engine"]), ("Optimal", "ortools-" + eng))
                self.assertAlmostEqual(res["objective"], -464.75314286, places=5)
                rep = verify.verify(AFIRO, sol)
                self.assertTrue(rep["model_match"])
                if eng == "glop":  # a vertex at simplex accuracy; PDLP's own test is looser (recorded as a row)
                    self.assertEqual(rep["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=1)
