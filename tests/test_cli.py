#!/usr/bin/env python3
"""test_cli.py — the command line contract (CLAUDE.md §7): exit codes, solution files,
--warm, batch, --version. Run by ctest with the path of the built binary as argv[1]."""
import os
import subprocess
import sys
import tempfile
import unittest

BIN = sys.argv.pop(1) if len(sys.argv) > 1 else "build/ps26119"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

INFEASIBLE = """LPM 1
NAME inf
SENSE MIN
ROWS 2
COLS 2
NNZ 4
OFFSET 0
OBJ
1 1
COL_LOWER
0 0
COL_UPPER
inf inf
ROW_LOWER
-inf 3
ROW_UPPER
1 inf
COL_START
0 2 4
ROW_INDEX
0 1 0 1
VALUE
1 1 1 1
END
"""


def run(*args):
    p = subprocess.run([BIN, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_version_has_git_hash(self):
        code, out = run("--version")
        self.assertEqual(code, 0)
        self.assertIn("(git ", out)

    def test_optimal_exit_0_and_solution_file(self):
        sol = os.path.join(self.tmp, "a.sol")
        code, out = run("solve", os.path.join(DATA, "netlib_small", "afiro.lpm"), "--out", sol)
        self.assertEqual(code, 0, out)
        self.assertIn("Optimal", out)
        self.assertIn("certified", out)
        text = open(sol).read()
        self.assertTrue(text.startswith("PS26119-SOLUTION 1"))
        self.assertIn("certified_bound", text)

    def test_infeasible_exit_1(self):
        p = os.path.join(self.tmp, "inf.lpm")
        open(p, "w").write(INFEASIBLE)
        for alg in ("oracle", "r2hpdhg"):
            code, out = run("solve", p, "--algorithm", alg)
            self.assertEqual(code, 1, out)
            self.assertIn("Infeasible", out)

    def test_iteration_limit_exit_1(self):
        code, out = run("solve", os.path.join(DATA, "netlib_small", "share2b.lpm"), "--iteration-limit", "64")
        self.assertEqual(code, 1, out)
        self.assertIn("IterationLimit", out)

    def test_read_errors_exit_3(self):
        code, _ = run("solve", os.path.join(self.tmp, "missing.lpm"))
        self.assertEqual(code, 3)
        bad = os.path.join(self.tmp, "bad.lpm")
        open(bad, "w").write("LPM 1\nROWS x\n")
        code, _ = run("solve", bad)
        self.assertEqual(code, 3)

    def test_usage_errors_exit_2(self):
        self.assertEqual(run("solve", os.path.join(DATA, "netlib_small", "afiro.lpm"), "--tol", "-1")[0], 2)
        self.assertEqual(run("solve", os.path.join(DATA, "netlib_small", "afiro.lpm"), "--bogus")[0], 2)
        self.assertEqual(run("frobnicate")[0], 2)

    def test_warm_start_from_own_solution(self):
        lp = os.path.join(DATA, "netlib_small", "stocfor1.lpm")
        sol = os.path.join(self.tmp, "s.sol")
        self.assertEqual(run("solve", lp, "--out", sol)[0], 0)
        code, out = run("solve", lp, "--warm", sol)
        self.assertEqual(code, 0, out)
        self.assertIn("warm start", out)

    def test_batch_two_scenarios(self):
        lp = os.path.join(DATA, "netlib_small", "afiro.lpm")
        code, out = run("batch", lp, lp, lp, "--out-dir", self.tmp)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("Optimal"), 2)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "afiro.sol")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
