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
        for alg in ("oracle", "r2hpdhg", "simplex"):
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
        self.assertEqual(run("solve", lp, "--algorithm", "r2hpdhg", "--out", sol)[0], 0)
        code, out = run("solve", lp, "--warm", sol)  # auto honours a warm start: r2hpdhg
        self.assertEqual(code, 0, out)
        self.assertIn("warm start", out)
        self.assertIn("engine     r2hpdhg", out)
        code, out = run("solve", lp, "--algorithm", "simplex", "--warm", sol)
        self.assertEqual(code, 0, out)
        self.assertIn("warm start ignored", out)

    def test_mps_input_every_engine_and_check_line(self):
        mps = os.path.join(DATA, "netlib_small", "afiro.mps")
        for alg in ("simplex", "r2hpdhg", "oracle"):
            sol = os.path.join(self.tmp, alg + ".sol")
            code, out = run("solve", mps, "--algorithm", alg, "--out", sol)
            self.assertEqual(code, 0, out)
            self.assertIn("check      PASS", out)
            self.assertIn("check PASS", open(sol).read())
        # the verifier accepts the simplex solution with its independent (highspy) reader
        v = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "verify.py"), mps,
                            os.path.join(self.tmp, "simplex.sol"), "--expected", "-464.75314286"],
                           capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_mps_read_error_has_line_number_exit_3(self):
        bad = os.path.join(self.tmp, "bad.mps")
        open(bad, "w").write("NAME B\nROWS\n N obj\n L c1\nCOLUMNS\n    x obj 1 c1 nan\nENDATA\n")
        code, out = run("solve", bad)
        self.assertEqual(code, 3, out)
        self.assertIn("line 6", out)

    def test_info_print_and_lu_bench(self):
        mps = os.path.join(DATA, "examples", "features.mps")
        code, out = run("info", mps)
        self.assertEqual(code, 0, out)
        self.assertIn("rows               5", out)
        self.assertIn("fingerprint", out)
        self.assertIn("negative upper bound", out)  # reader warning on stderr
        code, out = run("print", mps)
        self.assertEqual(code, 0, out)
        self.assertIn("2 <= X3 + X4 <= 5", out)
        code, out = run("lu-bench", os.path.join(DATA, "netlib_small", "afiro.mps"), "--trials", "1")
        self.assertEqual(code, 0, out)
        self.assertIn("LU-BENCH PASS", out)

    def test_certificates_verified_independently_and_tampering_detected(self):
        verify = os.path.join(ROOT, "tools", "verify.py")
        def ver(model, sol):
            return subprocess.run([sys.executable, verify, model, sol], capture_output=True, text=True)
        for name in ("infeasible", "unbounded"):
            mps = os.path.join(DATA, "examples", name + ".mps")
            for alg in ("simplex", "r2hpdhg"):
                sol = os.path.join(self.tmp, f"{name}-{alg}.sol")
                code, out = run("solve", mps, "--algorithm", alg, "--out", sol)
                self.assertEqual(code, 1, out)  # Infeasible / Unbounded exit code
                self.assertIn("check      PASS", out)
                v = ver(mps, sol)
                self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
                self.assertIn("certificate", v.stdout)
                # tamper: flip the sign of every certificate entry -> the verifier must reject it
                lines = open(sol).read().splitlines()
                out_lines, in_ray = [], False
                for ln in lines:
                    if ln.startswith(("DUAL_RAY", "PRIMAL_RAY")):
                        in_ray = True
                    elif in_ray and ln == "END":
                        in_ray = False
                    elif in_ray:
                        k, val = ln.split()
                        ln = f"{k} {-float(val)!r}"
                    out_lines.append(ln)
                bad = os.path.join(self.tmp, f"{name}-{alg}-bad.sol")
                open(bad, "w").write("\n".join(out_lines) + "\n")
                self.assertEqual(ver(mps, bad).returncode, 1, f"tampered {name}/{alg} certificate accepted")
                # and a claim without any certificate is not accepted either
                bare = os.path.join(self.tmp, f"{name}-{alg}-bare.sol")
                keep, skip = [], False
                for ln in lines:
                    if ln.startswith(("DUAL_RAY", "PRIMAL_RAY")):
                        skip = True
                        continue
                    if skip and ln == "END":
                        skip = False
                    if not skip:
                        keep.append(ln)
                open(bare, "w").write("\n".join(keep) + "\n")
                self.assertEqual(ver(mps, bare).returncode, 1)

    def test_batch_two_scenarios(self):
        lp = os.path.join(DATA, "netlib_small", "afiro.lpm")
        code, out = run("batch", lp, lp, lp, "--out-dir", self.tmp)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("Optimal"), 2)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "afiro.sol")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
