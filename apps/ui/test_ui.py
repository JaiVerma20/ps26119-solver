#!/usr/bin/env python3
"""test_ui.py — the Command Center backend: log parsing, option validation, path safety, JSON
hygiene, a real end-to-end solve + independent verification over HTTP, and the evidence endpoint.
Run by ctest (ui.backend) with the built binary as argv[1]."""
import json
import os
import sys
import threading
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    os.environ["PS26119_BIN"] = os.path.abspath(sys.argv.pop(1))
os.environ.setdefault("PS26119_PYTHON", sys.executable)
sys.path.insert(0, HERE)

import server  # noqa: E402
from backend import evidence, generate, models, paths, runner, scenarios  # noqa: E402
from lpm import read_lpm  # noqa: E402


class Parsing(unittest.TestCase):
    def test_progress_lines_of_every_engine(self):
        p = runner.parse_progress("r2hpdhg it      128  t    0.13s  pobj -8.4215992353e+05  dobj -8.7666891275e+05  "
                                  "rp 2.29e-03  rd 2.74e-03  gap 2.01e-02  fp64")
        self.assertEqual((p["kind"], p["iter"], p["precision"]), ("pdhg", 128, "fp64"))
        self.assertAlmostEqual(p["rp"], 2.29e-3)
        s = runner.parse_progress("  iter      1000  phase 1  objective  1.6331455186e+04  infeasibility 7.66e+02  0.11s")
        self.assertEqual((s["kind"], s["iter"], s["phase"]), ("simplex", 1000, 1))
        b = runner.parse_progress("bb node 2046 incumbent 24266")
        self.assertEqual((b["kind"], b["node"], b["incumbent"]), ("bb", 2046, 24266.0))
        self.assertIsNone(runner.parse_progress("  restart at 64 (inner 64): r/r0 0.038"))

    def test_emit_accepts_fields_named_like_its_own_arguments(self):
        # progress dicts carry "kind" and "t": this killed the job thread once (a TypeError)
        job = runner.Job("data/netlib_small/afiro.mps", runner.validate_options({})[0])
        job.emit("progress", kind="pdhg", t=1.0, type="x")
        self.assertEqual(job.events[-1]["type"], "progress")
        self.assertEqual(job.events[-1]["kind"], "pdhg")

    def test_options_are_validated_like_the_cli(self):
        ok, err = runner.validate_options({"algorithm": "simplex", "threads": 0, "time_limit": 5})
        self.assertIsNone(err)
        for bad in ({"algorithm": "cplex"}, {"precision": "fp16"}, {"threads": -1}, {"time_limit": 0},
                    {"time_limit": "abc"}, {"tol": -1}):
            self.assertIsNotNone(runner.validate_options(bad)[1], bad)

    def test_json_never_contains_nan_or_infinity(self):
        # an Infeasible objective is NaN; JSON.parse in the browser rejects NaN and drops the event
        s = server.dumps({"a": float("nan"), "b": [float("inf"), -float("inf"), 1.5]})
        self.assertEqual(json.loads(s), {"a": None, "b": [None, None, 1.5]})
        self.assertNotIn("NaN", s)


class PathSafety(unittest.TestCase):
    def test_only_model_files_in_the_data_folders(self):
        self.assertIsNotNone(paths.resolve_model("data/netlib_small/afiro.mps"))
        for bad in ("../etc/passwd", "/etc/passwd", "data/netlib_small/../../CMakeLists.txt", "README.md",
                    "data/netlib_small/afiro.mps\x00", "tools/verify.py", ""):
            self.assertIsNone(paths.resolve_model(bad), bad)


class Endpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=120) as r:
            return json.loads(r.read())

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_static_shell_and_traversal(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            self.assertIn(b"PS26119", r.read())
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(self.base + "/../server.py", timeout=10)

    def test_system_and_model_info_come_from_the_solver(self):
        s = self.get("/api/system")
        self.assertTrue(s["binary_found"])
        self.assertIn("(git ", s["version"])
        i = self.get("/api/model/info?path=data/netlib_small/afiro.mps")
        self.assertEqual((i["n_rows"], i["n_cols"], i["n_nnz"]), (27, 32, 83))
        sp = self.get("/api/model/sparsity?path=data/netlib_small/afiro.mps")
        self.assertEqual(sum(sp["grid"]), 83)  # afiro is small enough for an exact picture
        code, _ = self.post("/api/solve", {"path": "../../etc/passwd"})
        self.assertEqual(code, 404)

    def test_afiro_end_to_end_solve_and_independent_verification(self):
        code, j = self.post("/api/solve", {"path": "data/netlib_small/afiro.mps", "algorithm": "auto"})
        self.assertEqual(code, 200)
        for _ in range(600):
            d = self.get(f"/api/jobs/{j['job']}")
            if d["done"]:
                break
            time.sleep(0.1)
        ev = {e["type"]: e for e in d["events"]}
        self.assertNotIn("error", ev, ev.get("error"))
        sol = ev["result"]["solution"]
        self.assertEqual(sol["status"], "Optimal")
        self.assertAlmostEqual(sol["objective"], -464.75314286, places=6)
        self.assertEqual((ev["result"]["rows"], ev["result"]["cols"], ev["result"]["nnz"]), (27, 32, 83))
        self.assertTrue(ev["result"]["check_line"].startswith("PASS"))
        self.assertEqual(ev["verify"]["report"]["verdict"], "PASS")

    def test_infeasible_result_survives_json_and_carries_its_certificate(self):
        code, j = self.post("/api/solve", {"path": "data/examples/infeasible.mps", "algorithm": "simplex"})
        for _ in range(600):
            d = self.get(f"/api/jobs/{j['job']}")
            if d["done"]:
                break
            time.sleep(0.1)
        ev = {e["type"]: e for e in d["events"]}
        self.assertEqual(ev["result"]["solution"]["status"], "Infeasible")
        self.assertIsNone(ev["result"]["solution"]["objective"])  # NaN -> null
        self.assertIn("Farkas", ev["result"]["check_line"])
        self.assertIn("Farkas", ev["verify"]["report"]["certificate"])

    def wait(self, job):
        for _ in range(1200):
            d = self.get(f"/api/jobs/{job}")
            if d["done"]:
                return {e["type"]: e for e in d["events"]}
            time.sleep(0.1)
        self.fail("job did not finish")

    def test_generated_model_is_checked_against_its_known_optimum(self):
        self.assertEqual(generate.validate({"kind": "refinery", "size": 12}), (("refinery", 12, 1), None))
        for bad in ({"kind": "rm -rf"}, {"kind": "refinery", "size": 0}, {"kind": "refinery", "size": 9000},
                    {"kind": "random", "size": "x"}, {"kind": "random", "size": 1000, "seed": 0}):
            self.assertIsNotNone(generate.validate(bad)[1], bad)
        out = os.path.join(generate.GEN_DIR, "refinery-T12-s4242")
        try:
            code, g = self.post("/api/generate", {"kind": "refinery", "size": 12, "seed": 4242})
            self.assertEqual((code, g["path"], g["created"]), (200, "bench/generated/refinery-T12-s4242.lpm", True))
            self.assertFalse(self.post("/api/generate", {"kind": "refinery", "size": 12, "seed": 4242})[1]["created"])
            info = self.get("/api/model/info?path=" + g["path"])
            self.assertEqual(info["known"]["generator"], "generate_refinery_lp.py")
            code, j = self.post("/api/solve", {"path": g["path"], "algorithm": "simplex"})
            ev = self.wait(j["job"])
            self.assertEqual(ev["result"]["solution"]["status"], "Optimal")
            self.assertLess(ev["result"]["known"]["rel_err"], 1e-6)
        finally:
            for ext in (".lpm", ".json"):
                if os.path.exists(out + ext):
                    os.remove(out + ext)

    def test_a_limit_is_not_sent_to_the_verifier(self):
        code, j = self.post("/api/solve", {"path": "data/netlib_small/afiro.mps", "algorithm": "simplex", "time_limit": 1e-9})
        ev = self.wait(j["job"])
        status = ev["result"]["solution"]["status"]
        if status == "Optimal":
            self.skipTest("AFIRO solved inside the time limit on this machine")
        self.assertEqual(status, "TimeLimit")
        self.assertTrue(ev["verify"]["skipped"])
        self.assertNotIn("report", ev["verify"])

    def test_scenarios_whatif_and_sweep_on_a_refinery_model(self):
        stem = os.path.join(generate.GEN_DIR, "refinery-T12-s4243")
        try:
            g = generate.generate("refinery", 12, 4243)
            meta = self.get("/api/scenarios")
            self.assertIn(g["path"], [m["path"] for m in meta["models"]])
            for bad in ({"base": "data/netlib_small/afiro.mps"}, {"base": g["path"], "levers": {"price": 80}},
                        {"base": g["path"], "levers": {"tax": 5}}, {"base": g["path"], "mode": "sweep", "sweep": {"lever": "price", "from": 5, "to": 1, "steps": 3}}):
                self.assertEqual(self.post("/api/scenario", bad)[0], 400, bad)

            # levers touch exactly the rows / columns they name
            out = stem + "-cdu.lpm"
            scenarios.write_scenario(stem + ".lpm", 12, {"cdu": -10}, out, "t")
            a, b = read_lpm(stem + ".lpm"), read_lpm(out)
            changed = [i for i, (x, y) in enumerate(zip(a.row_upper, b.row_upper)) if x != y]
            L = scenarios.LAYOUT
            self.assertEqual(changed, [t * L.nrows + L.r_cdu for t in range(12)])
            self.assertAlmostEqual(b.row_upper[L.r_cdu], 0.9 * a.row_upper[L.r_cdu])
            self.assertEqual((a.obj, a.col_upper, a.value), (b.obj, b.col_upper, b.value))
            os.remove(out)

            code, j = self.post("/api/scenario", {"base": g["path"], "mode": "whatif", "levers": {"price": 5, "cdu": -10}})
            self.assertEqual(code, 200)
            ev = self.wait(j["job"])
            self.assertNotIn("error", ev, ev.get("error"))
            base, sm = ev["base"], ev["summary"]
            self.assertLess(abs(base["objective"] - base["known"]["optimum"]) / (1 + abs(base["known"]["optimum"])), 1e-6)
            self.assertEqual(len(base["marginal"]["cdu"]), 12)
            self.assertLess(sm["agree"], 1e-6)  # cold and warm answers agree
            self.assertAlmostEqual(sm["delta"], ev["solved"]["objective"] - base["objective"], places=6)
            verdicts = [e["verdict"] for e in self.events(j["job"]) if e["type"] == "verified"]
            self.assertEqual(verdicts, ["PASS", "PASS"])

            code, j = self.post("/api/scenario", {"base": g["path"], "mode": "sweep", "sweep": {"lever": "price", "from": -5, "to": 5, "steps": 3},
                                                  "sequential": True})
            evs = self.events(j["job"], wait=True)
            pts = [e for e in evs if e["type"] == "point"]
            seq = [e for e in evs if e["type"] == "sequential_point"]
            self.assertEqual([p["value"] for p in pts], [-5.0, 0.0, 5.0])
            self.assertTrue(all(p["status"] == "Optimal" for p in pts))
            self.assertAlmostEqual(pts[1]["objective"], base["objective"], delta=1e-6 * (1 + abs(base["objective"])))
            self.assertLess(pts[0]["objective"], pts[1]["objective"])  # lower prices, lower profit
            self.assertLess(pts[1]["objective"], pts[2]["objective"])
            for p, q in zip(pts, seq):  # batched and one-by-one answers agree
                self.assertLess(abs(p["objective"] - q["objective"]) / (1 + abs(q["objective"])), 1e-6)
            self.assertEqual([e["verdict"] for e in evs if e["type"] == "verified"], ["PASS"] * 3)
        finally:
            for f in (stem + ".lpm", stem + ".json"):
                if os.path.exists(f):
                    os.remove(f)

    def events(self, job, wait=True):
        if wait:
            self.wait(job)
        return self.get(f"/api/jobs/{job}")["events"]

    def test_evidence_matches_the_committed_csvs(self):
        e = self.get("/api/evidence")
        direct = evidence.collect()
        self.assertEqual([(n["engine"], n["solved"], n["total"]) for n in e["netlib"]],
                         [(n["engine"], n["solved"], n["total"]) for n in direct["netlib"]])
        for n in e["netlib"]:
            self.assertRegex(n["source"]["file"], r"-[0-9a-f]{7}\.csv$")


if __name__ == "__main__":
    unittest.main(verbosity=1)
