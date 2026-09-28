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

    def test_hardened_inputs(self):
        afiro = os.path.join(DATA, "netlib_small", "afiro.lpm")
        # empty / comment-only MPS is a read error, not a 0x0 "Optimal" model
        for text in ("", "* only a comment\n\n"):
            empty = os.path.join(self.tmp, "empty.mps")
            open(empty, "w").write(text)
            code, out = run("solve", empty)
            self.assertEqual(code, 3, out)
            self.assertIn("no MPS section", out)
        # numeric flags are validated completely (solve and batch)
        for bad in (["--threads", "-2"], ["--threads", "1.5"], ["--time-limit", "inf"], ["--tol", "nan"],
                    ["--set", "=3"], ["--set", "x"], ["--iteration-limit", "abc"]):
            code, out = run("solve", afiro, *bad)
            self.assertEqual(code, 2, (bad, out))
        self.assertEqual(run("batch", afiro, afiro, "--tol", "abc")[0], 2)
        self.assertEqual(run("batch", afiro, afiro, "--set", "noequals")[0], 2)
        # an unwritable --out fails BEFORE solving, with exit 4
        code, out = run("solve", afiro, "--out", os.path.join(self.tmp, "no", "such", "dir.sol"))
        self.assertEqual(code, 4, out)
        self.assertNotIn("status", out)
        # the writability probe leaves no file behind when the solve then fails to read the model
        probe = os.path.join(self.tmp, "probe.sol")
        self.assertEqual(run("solve", os.path.join(self.tmp, "missing.lpm"), "--out", probe)[0], 3)
        self.assertFalse(os.path.exists(probe))

    def test_reader_fuzz_never_crashes(self):
        # Mutated afiro (.lpm and .mps): truncations, corrupted bytes and tokens, deleted lines.
        # Every run must end with a contract exit code; in the ASan/UBSan CI job a memory error
        # would also show up as a sanitizer report on stderr.
        import random
        rng = random.Random(7)
        for fmt in ("lpm", "mps"):
            src = open(os.path.join(DATA, "netlib_small", "afiro." + fmt)).read()
            path = os.path.join(self.tmp, "fuzz." + fmt)
            for _ in range(60):
                s, k = src, rng.randint(0, 3)
                if k == 0:
                    s = s[:rng.randint(0, len(s))]
                elif k == 1:
                    i = rng.randrange(len(s))
                    s = s[:i] + rng.choice(["-", "1e999", "nan", "\x00", "x", "99999999999", " ", "\n", ""]) + s[i + 1:]
                elif k == 2:
                    toks = s.split(" ")
                    toks[rng.randrange(len(toks))] = rng.choice(["-5", "1e308", "inf", "-inf", "abc", "0", "2147483648"])
                    s = " ".join(toks)
                else:
                    lines = s.split("\n")
                    del lines[rng.randrange(len(lines))]
                    s = "\n".join(lines)
                open(path, "w", errors="replace").write(s)
                for args in (["info", path], ["solve", path, "--time-limit", "5"]):
                    code, out = run(*args)
                    self.assertIn(code, (0, 1, 3, 5), (args, out[-300:]))
                    self.assertNotIn("Sanitizer", out, out[-600:])
                    self.assertNotIn("runtime error", out, out[-600:])

    def test_malformed_warm_start_files(self):
        lp = os.path.join(DATA, "netlib_small", "afiro.lpm")
        good = os.path.join(self.tmp, "good.sol")
        self.assertEqual(run("solve", lp, "--algorithm", "r2hpdhg", "--out", good)[0], 0)
        text = open(good).read()
        cols = text.index("COLUMNS")
        first = text[cols:].split("\n")[1]  # "0 <x0> <z0>"
        fields = first.split()
        idx, x0, z0 = fields[0], fields[1], fields[2]
        cases = {
            "overflow": text.replace(first, " ".join([idx, "1e999"] + fields[2:]), 1),  # inf value
            "garbage": text.replace(first, " ".join([idx, x0 + "xyz"] + fields[2:]), 1),  # not a number
            "negcount": text.replace(text[cols:].split("\n")[0], "COLUMNS -5", 1),
            "hugecount": text.replace(text[cols:].split("\n")[0], "COLUMNS 99999999999", 1),
            "bigcount": text.replace(text[cols:].split("\n")[0], "COLUMNS 999999999", 1),  # no huge allocation
        }
        for name, body in cases.items():
            bad = os.path.join(self.tmp, name + ".sol")
            open(bad, "w").write(body)
            code, out = run("solve", lp, "--algorithm", "r2hpdhg", "--warm", bad)
            self.assertEqual(code, 3, (name, out[-300:]))
            self.assertIn("read error (warm start)", out, name)

    def test_bogus_lpm_sizes_fail_fast_without_huge_allocations(self):
        # "NNZ 99999999999" used to allocate gigabytes and get the process killed (exit 137)
        src = open(os.path.join(DATA, "netlib_small", "afiro.lpm")).read()
        for old, new in (("NNZ 83", "NNZ 99999999999"), ("NNZ 83", "NNZ 2000000000"), ("ROWS 27", "ROWS 2000000000"),
                         ("COLS 32", "COLS 2000000000"), ("ROWS 27", "ROWS -3")):
            bad = os.path.join(self.tmp, "bad.lpm")
            open(bad, "w").write(src.replace(old + "\n", new + "\n", 1))
            code, out = run("info", bad)
            self.assertEqual(code, 3, (new, out[-200:]))

    def test_gpu_request_is_never_served_silently_by_the_cpu(self):
        # auto + --gpu must pick r2hpdhg (the only GPU engine). On a build without CUDA this is
        # a clear NotSolved (exit 5), never a CPU simplex answer presented as a GPU run.
        code, out = run("solve", os.path.join(DATA, "netlib_small", "afiro.lpm"), "--gpu")
        self.assertIn("auto: r2hpdhg", out)
        if "no CUDA backend" in out:
            self.assertEqual(code, 5, out)
            self.assertIn("NotSolved", out)
        else:  # CUDA build
            self.assertEqual(code, 0, out)

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

    def test_batch_solution_files_name_their_own_scenario_model(self):
        # regression: every scenario .sol used to carry the BASE model's fingerprint and name
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from lpm import read_lpm, read_solution, write_lpm
        base = os.path.join(DATA, "netlib_small", "afiro.lpm")
        m = read_lpm(base)
        m.obj[0] += 1.0
        m.name = "afiro_obj"
        s1 = os.path.join(self.tmp, "s_obj.lpm")
        write_lpm(m, s1)
        m = read_lpm(base)
        m.col_upper[1] = 50.0
        m.name = "afiro_ub"
        s2 = os.path.join(self.tmp, "s_ub.lpm")
        write_lpm(m, s2)
        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        code, out = run("batch", base, base, s1, s2, "--out-dir", out_dir)
        self.assertEqual(code, 0, out)

        def info_fp(path):
            c, o = run("info", path)
            self.assertEqual(c, 0, o)
            return next(l.split()[-1] for l in o.splitlines() if l.strip().startswith("fingerprint"))

        fps = {}
        for model, stem, name in ((base, "afiro", "afiro"), (s1, "s_obj", "afiro_obj"), (s2, "s_ub", "afiro_ub")):
            h = read_solution(os.path.join(out_dir, stem + ".sol")).header
            self.assertEqual(h["model"], info_fp(model), stem)
            self.assertEqual(h["name"], name, stem)
            fps[stem] = h["model"]
        self.assertEqual(len(set(fps.values())), 3)
        # and the independent verifier agrees that each file belongs to its scenario
        ver = os.path.join(ROOT, "tools", "verify.py")
        for model, stem in ((s1, "s_obj"), (s2, "s_ub")):
            p = subprocess.run([sys.executable, ver, model, os.path.join(out_dir, stem + ".sol"), "--json",
                                os.path.join(self.tmp, stem + ".json"), "--quiet"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            import json
            with open(os.path.join(self.tmp, stem + ".json")) as f:
                rep = json.load(f)
            self.assertTrue(rep["model_match"], stem)
            self.assertEqual(rep["fingerprint_model"], rep["fingerprint_solution"], stem)


if __name__ == "__main__":
    unittest.main(verbosity=1)
