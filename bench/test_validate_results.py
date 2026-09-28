#!/usr/bin/env python3
"""Tests for bench/validate_results.py on SYNTHETIC files (no benchmark numbers)."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_results as vr  # noqa: E402

HEAD = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=os.path.dirname(HERE),
                      capture_output=True, text=True).stdout.strip()
COLS = "git_hash,machine,cpu,gpu,driver,cuda,date,instance,backend,status,verify\n"


class Validate(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def write(self, name, body):
        p = os.path.join(self.d, name)
        with open(p, "w") as f:
            f.write(body)
        return p

    def test_good_cpu_and_gpu_rows(self):
        p = self.write(f"scale-lap-{HEAD}.csv", COLS +
                       f"{HEAD},lap,Intel,none,,,2026-09-27,m,cpu,Optimal,PASS\n"
                       f"{HEAD},lap,Intel,RTX X,550.1,12.4,2026-09-27,m,gpu,Optimal,PASS\n")
        self.assertEqual(vr.check_csv(p)[0], [])

    def test_rejections(self):
        cases = {
            "hash mismatch": (f"x-{HEAD}.csv", COLS + f"0000000,lap,c,none,,,d,m,cpu,Optimal,PASS\n"),
            "gpu without model": (f"x-{HEAD}.csv", COLS + f"{HEAD},lap,c,none,,,d,m,gpu,Optimal,PASS\n"),
            "gpu without cuda": (f"x-{HEAD}.csv", COLS + f"{HEAD},lap,c,RTX,550,,d,m,gpu,Optimal,PASS\n"),
            "optimal but verify fail": (f"x-{HEAD}.csv", COLS + f"{HEAD},lap,c,none,,,d,m,cpu,Optimal,FAIL\n"),
            "unknown commit": ("x-abcdef1.csv", COLS + "abcdef1,lap,c,none,,,d,m,cpu,Optimal,PASS\n"),
            "no hash in name": ("results.csv", COLS + f"{HEAD},lap,c,none,,,d,m,cpu,Optimal,PASS\n"),
            "dirty rows, clean name": (f"x-{HEAD}.csv", COLS + f"{HEAD}-dirty,lap,c,none,,,d,m,cpu,Optimal,PASS\n"),
        }
        for why, (name, body) in cases.items():
            errors, _ = vr.check_csv(self.write(name, body))
            self.assertTrue(errors, why)

    def test_reference_solver_rows_may_record_rejected_claims(self):
        # compare-highs-*.csv: a HiGHS row may say Optimal while verify.py rejects it (that is the
        # finding); a ps26119 row may not, with or without the solver column
        cols = "git_hash,machine,cpu,gpu,driver,cuda,date,instance,backend,status,verify,solver\n"
        ok = self.write(f"compare-highs-lap-{HEAD}.csv", cols + f"{HEAD},lap,c,none,,,d,m,,Optimal,FAIL,highs\n")
        self.assertEqual(vr.check_csv(ok)[0], [])
        bad = self.write(f"compare-highs-lap-{HEAD}.csv", cols + f"{HEAD},lap,c,none,,,d,m,,Optimal,FAIL,ps26119\n")
        self.assertTrue(vr.check_csv(bad)[0])

    def test_logs(self):
        good = os.path.join(self.d, "logs", "lap-" + HEAD)
        os.makedirs(good)
        self.write(os.path.join("logs", "lap-" + HEAD, "ctest.log"), "100% tests passed, 0 tests failed out of 168\n")
        self.write(os.path.join("logs", "lap-" + HEAD, "sanitizer-memcheck.log"), "========= ERROR SUMMARY: 0 errors\n")
        self.write(os.path.join("logs", "lap-" + HEAD, "nvidia-smi.txt"), "x\n")
        self.assertEqual(vr.check_logs(good)[0], [])
        bad = os.path.join(self.d, "logs", "bad")
        os.makedirs(bad)
        self.write(os.path.join("logs", "bad", "ctest.log"),
                   "99% tests passed, 1 tests failed out of 168\nThe following tests FAILED:\n\t131 - Gpu.X (Failed)\n")
        self.write(os.path.join("logs", "bad", "sanitizer-racecheck.log"), "========= ERROR SUMMARY: 3 errors\n")
        errors, _ = vr.check_logs(bad)
        self.assertTrue(any("Gpu.X" in e for e in errors))
        self.assertTrue(any("3 errors" in e for e in errors))
        # WSL2: the sanitizer cannot attach — must be reported as NOT RUN, not as clean or as kernel errors
        wsl = os.path.join(self.d, "logs", "wsl")
        os.makedirs(wsl)
        self.write(os.path.join("logs", "wsl", "ctest.log"), "100% tests passed\n")
        self.write(os.path.join("logs", "wsl", "sanitizer-memcheck.log"),
                   "========= Error: Failed to initialize WDDM debugger interface.\n"
                   "========= Error: Device not supported.\n========= ERROR SUMMARY: 2 errors\n")
        self.write(os.path.join("logs", "wsl", "sanitizer-racecheck.log"),
                   "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n")
        errors, _ = vr.check_logs(wsl)
        self.assertEqual(len(errors), 1)
        self.assertIn("could not run", errors[0])


if __name__ == "__main__":
    unittest.main(verbosity=1)
