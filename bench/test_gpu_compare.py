#!/usr/bin/env python3
"""Tests for bench/gpu_compare.py — the GPU/CPU pairing rules. The rows below are SYNTHETIC test
inputs (no benchmark numbers): they only exercise the pairing and ratio logic."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_compare  # noqa: E402


def row(backend, threads, status="Optimal", t4="1", t8="10", verify="PASS", prec="fp64", inst="m"):
    return {"backend": backend, "threads": threads, "status": status, "seconds_to_1e-4": t4,
            "seconds_to_1e-8": t8, "verify": verify, "precision": prec, "instance": inst, "engine": "r2hpdhg"}


class Pairing(unittest.TestCase):
    def test_best_cpu_baseline_and_one_thread(self):
        rows = [row("cpu", "1", t8="40"), row("cpu", "10", t8="8"), row("gpu", "", t4="0.5", t8="4")]
        (p,) = gpu_compare.pairs(rows)
        self.assertEqual(p["best"]["threads"], "10")
        self.assertAlmostEqual(p["vs1_1e-8"], 10.0)
        self.assertAlmostEqual(p["vsbest_1e-8"], 2.0)  # the headline compares with the FASTEST CPU run
        self.assertEqual(gpu_compare.status_note(p), "")

    def test_gpu_slower_is_a_ratio_below_one(self):
        (p,) = gpu_compare.pairs([row("cpu", "1", t8="2"), row("gpu", "", t8="4")])
        self.assertAlmostEqual(p["vsbest_1e-8"], 0.5)
        self.assertEqual(gpu_compare.fmt_ratio(p["vsbest_1e-8"]), "0.50×")

    def test_no_ratio_for_failed_or_unverified_runs(self):
        for bad in (row("gpu", "", status="TimeLimit"), row("gpu", "", verify="FAIL"), row("gpu", "", t8="")):
            (p,) = gpu_compare.pairs([row("cpu", "1"), bad])
            self.assertIsNone(p["vsbest_1e-8"])
            self.assertEqual(gpu_compare.fmt_ratio(p["vsbest_1e-8"]), "–")
        (p,) = gpu_compare.pairs([row("cpu", "1", status="NumericalError"), row("gpu", "")])
        self.assertIsNone(p["vs1_1e-8"])
        self.assertIsNone(p["best"])  # a failed CPU run is never a baseline
        self.assertIn("CPU 1 thr: NumericalError", gpu_compare.status_note(p))

    def test_pairs_only_same_precision_and_instance(self):
        rows = [row("cpu", "1", prec="mixed"), row("cpu", "1", inst="other"), row("gpu", "")]
        (p,) = gpu_compare.pairs(rows)
        self.assertIsNone(p["cpu1"])
        self.assertIsNone(p["best"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
