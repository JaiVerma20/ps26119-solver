#!/usr/bin/env python3
"""test_ui.py — the Command Center backend: log parsing, option validation, path safety, JSON
hygiene, a real end-to-end solve + independent verification over HTTP, and the evidence endpoint.
Run by ctest (ui.backend) with the built binary as argv[1]."""
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    os.environ["PS26119_BIN"] = os.path.abspath(sys.argv.pop(1))
os.environ.setdefault("PS26119_PYTHON", sys.executable)
# keep test runs out of the user's run history (apps/ui/.runs/jobs)
os.environ["PS26119_UI_RUNS"] = tempfile.mkdtemp(prefix="ps26119-ui-test-")
import atexit  # noqa: E402
import shutil  # noqa: E402
atexit.register(shutil.rmtree, os.environ["PS26119_UI_RUNS"], True)
sys.path.insert(0, HERE)

import server  # noqa: E402
from backend import certificate, evidence, generate, models, paths, preflight, report, runner, scenarios  # noqa: E402
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

    def test_typed_models_become_the_model_contract(self):
        from backend import lptext
        m = lptext.parse(lptext.EXAMPLES["wyndor"], "wyndor")
        self.assertEqual((m.sense, m.num_rows, m.num_cols, m.obj), (-1, 3, 2, [3.0, 5.0]))
        self.assertEqual((m.row_lower, m.row_upper), ([-math.inf] * 3, [4.0, 12.0, 18.0]))
        self.assertEqual((m.col_start, m.row_index, m.value), ([0, 2, 4], [0, 2, 1, 2], [1.0, 3.0, 2.0, 2.0]))
        self.assertEqual((m.row_names, m.col_names), (["plant1", "plant2", "plant3"], ["doors", "windows"]))
        # both sides, ranged rows, constants, bounds, integers, lp_solve style
        m = lptext.parse("min: 3*x + 2.5e1 y - 7;\n -5 <= x - y <= 5;\n x + 2 <= 3 y;\n int y;\n", "t")
        self.assertEqual((m.obj, m.obj_offset), ([3.0, 25.0], -7.0))
        self.assertEqual((m.row_lower, m.row_upper), ([-5.0, -math.inf], [5.0, -2.0]))
        self.assertEqual(m.value, [1.0, 1.0, -1.0, -3.0])
        self.assertEqual(m.is_integer, [0, 1])
        m = lptext.parse("max a\nst\n 10 >= a + b\nbounds\n a <= 4; -inf <= b <= 2; c free\nbinary d\nend", "t")
        self.assertEqual((m.col_lower, m.col_upper), ([0.0, -math.inf, -math.inf, 0.0], [4.0, 2.0, math.inf, 1.0]))
        self.assertEqual(m.is_integer, [0, 0, 0, 1])
        for bad, line in (("st x <= 1", 1), ("max x\nst\n x <= y <= 3", 3), ("max x\nst\n x 3 y <= 3", 3),
                          ("max x\nst\n c: x <= 1\n c: x <= 2", 4), ("max x\nst\n 5 <= x <= 3", 3), ("max x", 1)):
            with self.assertRaises(lptext.LpTextError, msg=bad) as cm:
                lptext.parse(bad)
            self.assertEqual(cm.exception.line, line, bad)

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
        # let every job finish (a job persists its job.json when it ends) before the temp run
        # folder is removed at exit
        deadline = time.time() + 120
        while time.time() < deadline and any(not j.done for j in list(server.JOBS.jobs.values())):
            time.sleep(0.2)
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

    def test_server_reports_when_its_code_changed_on_disk(self):
        self.assertFalse(self.get("/api/system")["server_stale"])
        saved = server.STARTED_CODE
        try:  # as if a backend file had changed after start-up (git pull, branch switch)
            server.STARTED_CODE = saved[:-1] + ((saved[-1][0], saved[-1][1] - 1),)
            self.assertTrue(self.get("/api/system")["server_stale"])
        finally:
            server.STARTED_CODE = saved

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

    def test_sensitivity_ranging_reaches_the_page(self):
        code, j = self.post("/api/solve", {"path": "data/netlib_small/afiro.mps", "algorithm": "simplex", "ranging": True})
        self.assertEqual(code, 200)
        for _ in range(600):
            d = self.get(f"/api/jobs/{j['job']}")
            if d["done"]:
                break
            time.sleep(0.1)
        ev = {e["type"]: e for e in d["events"]}
        self.assertNotIn("error", ev, ev.get("error"))
        rg = ev["result"]["ranging"]
        self.assertTrue(rg["ok"], rg)
        self.assertEqual((rg["n_costs"], rg["n_rows"]), (32, 27))
        # binding rows first, by |dual|; basic columns first
        st = [r["status"] != "not_binding" for r in rg["rows"]]
        self.assertEqual(st, sorted(st, reverse=True))
        duals = [abs(r["dual"]) for r in rg["rows"] if r["status"] != "not_binding"]
        self.assertEqual(duals, sorted(duals, reverse=True))
        self.assertEqual(rg["costs"][0]["status"], "basic")
        # a refusal reaches the page with the CLI's reason
        refused = runner.ranging_summary(os.path.join(os.environ["PS26119_UI_RUNS"], "none.csv"), "ranging    refused: ranging needs an Optimal solution\n")
        self.assertEqual(refused, {"ok": False, "message": "ranging needs an Optimal solution"})

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

    def solve_and_wait(self, body):
        code, j = self.post("/api/solve", body)
        self.assertEqual(code, 200, j)
        self.wait(j["job"])
        return j["job"]

    def test_certificate_report_and_persistence(self):
        jid = self.solve_and_wait({"path": "data/netlib_small/afiro.mps", "algorithm": "simplex"})
        c = self.get(f"/api/jobs/{jid}/certificate")
        self.assertEqual(c["final"], {"verdict": "PASS", "reasons": []})
        names = [k["name"] for k in c["checks"] if k["required"]]
        self.assertEqual(names, ["Definitive answer", "Original-model check (in-process)", "Independent verification",
                                 "Same model (fingerprint)"])
        self.assertEqual(c["model"]["fingerprint"], c["model"]["verifier_fingerprint"])
        self.assertAlmostEqual(c["result"]["objective"], -464.75314286, places=6)
        # the recorded SHA-256 is that of the solution file the verifier read
        self.assertEqual(c["artifact"]["sha256"], runner.sha256_file(os.path.join(paths.ROOT, c["artifact"]["solution_file"])))
        # the run survives a server restart (read back from job.json)
        stored = runner.load_job(jid)
        self.assertIsNotNone(stored)
        self.assertEqual(certificate.build(stored)["final"]["verdict"], "PASS")
        self.assertIn(jid, [r["id"] for r in self.get("/api/runs")])
        # the self-contained report: the certificate, the evidence with sources, no scripts, no external files
        with urllib.request.urlopen(f"{self.base}/api/report?jobs={jid}&download=1", timeout=60) as r:
            html_ = r.read().decode()
            self.assertIn("attachment", r.headers.get("Content-Disposition", ""))
        self.assertIn("✓ PASS", html_)
        self.assertIn(c["artifact"]["sha256"], html_)
        self.assertIn("bench/results/", html_)
        for banned in ("<script", "src=\"http", "href=\"http", "@import"):
            self.assertNotIn(banned, html_)
        self.assertEqual(report.e("<b>&"), "&lt;b&gt;&amp;")
        for bad in ("jobs=../../x", "jobs=" + ",".join(["0" * 12] * 21), "jobs=ffffffffffff"):
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(f"{self.base}/api/report?{bad}", timeout=10)

    def test_report_as_pdf_when_a_browser_is_available(self):
        from backend import pdf
        if not pdf.browser():
            self.skipTest("no Chrome / Chromium on this machine (the HTML report is the fallback)")
        self.assertTrue(self.get("/api/system")["pdf_export"])
        jid = self.solve_and_wait({"path": "data/netlib_small/afiro.mps", "algorithm": "simplex"})
        with urllib.request.urlopen(f"{self.base}/api/report?jobs={jid}&format=pdf&download=1&evidence=0", timeout=120) as r:
            body = r.read()
            self.assertEqual(r.headers.get("Content-Type"), "application/pdf")
            self.assertIn(".pdf", r.headers.get("Content-Disposition", ""))
        self.assertTrue(body.startswith(b"%PDF"))
        self.assertGreater(len(body), 5000)

    def test_a_limit_or_claim_is_certified_only_as_what_it_is(self):
        jid = self.solve_and_wait({"path": "data/netlib_small/afiro.mps", "algorithm": "simplex", "time_limit": 1e-9})
        c = self.get(f"/api/jobs/{jid}/certificate")
        if c["result"]["status"] == "Optimal":
            self.skipTest("AFIRO solved inside the time limit on this machine")
        self.assertEqual(c["final"]["verdict"], "NOT CERTIFIED")
        self.assertIn("Definitive answer", c["final"]["reasons"])
        jid = self.solve_and_wait({"path": "data/examples/infeasible.mps", "algorithm": "simplex"})
        c = self.get(f"/api/jobs/{jid}/certificate")
        self.assertEqual((c["result"]["status"], c["final"]["verdict"]), ("Infeasible", "PASS"))
        cert = next(k for k in c["checks"] if k["name"] == "Certificate")
        self.assertIn("Farkas", cert["detail"])

    def test_gpu_detail_is_the_committed_csv(self):
        import csv
        g = self.get("/api/gpu")
        for m in g["machines"]:
            path = os.path.join(paths.ROOT, "bench", "results", m["source"]["file"])
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(sum(len(i["runs"]) for i in m["instances"]), len(rows))
            for r in rows:  # every configuration, number for number
                inst = next(i for i in m["instances"] if i["instance"] == r["instance"])
                run = next(x for x in inst["runs"] if x["backend"] == r["backend"] and x["precision"] == r["precision"]
                           and (x["threads"] or None) == (int(r["threads"]) if r["threads"] else None))
                self.assertEqual(run["status"], r["status"])
                self.assertEqual(run["iterations"], int(r["iterations"]))
                if r["seconds_to_1e-8"]:
                    self.assertEqual(run["s_1e8"], float(r["seconds_to_1e-8"]))
            for p in m["pairs"]:  # ratios are CPU seconds / GPU seconds, as bench/gpu_compare.py defines them
                if p["ratio_vs_best"] is not None:
                    self.assertAlmostEqual(p["ratio_vs_best"], p["best_cpu_s"] / p["gpu_s"], places=6)

    def test_preflight_reports_every_dependency(self):
        r = self.get("/api/preflight")
        self.assertIn(r["state"], ("ok", "warn", "fail"))
        ids = {i["id"]: i for i in r["items"]}
        self.assertEqual(ids["binary"]["state"], "ok")
        self.assertEqual(ids["verifier"]["state"], "ok")  # from tools/verify.py --self-check
        self.assertIn(".lpm via lpm.py", ids["verifier"]["detail"])
        self.assertTrue(all(i["state"] in ("ok", "info", "warn", "fixable", "fail") for i in r["items"]))
        self.assertTrue(any(k.startswith("model:") for k in ids))

    def test_reference_run_through_highs_is_verified_like_ours(self):
        code, j = self.post("/api/reference", {"path": "data/netlib_small/afiro.mps", "solver": "ipm"})
        self.assertEqual(code, 200, j)
        ev = self.wait(j["job"])
        self.assertEqual(ev["result"]["status"], "Optimal")
        self.assertEqual(ev["result"]["engine"], "highs-ipm")
        self.assertAlmostEqual(ev["result"]["objective"], -464.75314286, places=6)
        self.assertEqual(ev["verify"]["report"]["verdict"], "PASS")
        for bad in ({"path": "data/netlib_small/afiro.mps", "solver": "gurobi"}, {"path": "../x.mps"},
                    {"path": "data/netlib_small/afiro.mps", "time_limit": 0}):
            self.assertIn(self.post("/api/reference", bad)[0], (400, 404), bad)
        # reference runs are not ps26119 runs: no certificate, not in the run list
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get(f"/api/jobs/{j['job']}/certificate")
        self.assertEqual(cm.exception.code, 409)
        self.assertNotIn(j["job"], [r["id"] for r in self.get("/api/runs")])

    def test_a_typed_model_is_saved_solved_and_verified(self):
        for ext in (".lpm", ".lp.txt"):  # the uploads folder is the user's: leave nothing behind
            self.addCleanup(lambda p=os.path.join(paths.UPLOADS, "ui_test_typed" + ext): os.path.exists(p) and os.remove(p))
        code, r = self.post("/api/model/text", {"name": "ui_test_typed", "text": self.get("/api/model/examples")["examples"]["blend"]})
        self.assertEqual(code, 200, r)
        self.assertEqual((r["rows"], r["cols"], r["sense"]), (4, 4, "minimize"))
        self.assertTrue(r["path"].endswith("ui_test_typed.lpm"))
        code, j = self.post("/api/solve", {"path": r["path"], "algorithm": "simplex"})
        self.assertEqual(code, 200, j)
        ev = self.wait(j["job"])
        self.assertEqual(ev["result"]["solution"]["status"], "Optimal")
        self.assertEqual(ev["verify"]["report"]["verdict"], "PASS")
        code, e = self.post("/api/model/text", {"name": "ui_test_typed", "text": "max x\nst\n x 3 y <= 3\n"})
        self.assertEqual((code, e["line"]), (400, 3))
        self.assertEqual(self.post("/api/model/text", {"name": "../evil", "text": "max x\nst\n x <= 1"})[0], 400)

    def test_every_installed_reference_solver_is_verified_like_ours(self):
        refs = {r["key"]: r for r in self.get("/api/references")["references"]}
        self.assertTrue({"highs-simplex", "ortools-glop", "scip", "coin-clp", "coin-cbc"} <= set(refs))
        self.assertTrue(refs["highs-simplex"]["available"])  # highspy is a hard dependency of the tools
        for key, r in refs.items():
            if not r["available"]:
                self.assertEqual(self.post("/api/reference", {"path": "data/netlib_small/afiro.mps", "solver": key})[0], 400)
                continue
            model, want = ("data/netlib_small/afiro.mps", -464.75314286) if r["lp"] else ("data/mip_small/gt2.lpm", 21166.0)
            code, j = self.post("/api/reference", {"path": model, "solver": key, "time_limit": 60})
            self.assertEqual(code, 200, (key, j))
            ev = self.wait(j["job"])
            self.assertEqual(ev["result"]["status"], "Optimal", key)
            # every Optimal claim is judged; first-order PDLP answers are only ~1e-8 relative-KKT accurate
            # and may honestly FAIL the verifier's 1e-6 worst-row test
            pdlp = key.endswith("pdlp")
            self.assertLess(abs(ev["result"]["objective"] - want) / (1 + abs(want)), 1e-4 if pdlp else 1e-8, key)
            self.assertIn(ev["verify"]["report"]["verdict"], ("PASS", "FAIL") if pdlp else ("PASS",), key)
        # an LP-only solver is refused on a MILP (it would answer the LP relaxation)
        code, j = self.post("/api/reference", {"path": "data/mip_small/gt2.lpm", "solver": "highs-simplex"})
        self.assertEqual(code, 400)
        self.assertIn("LPs only", j["error"])

    def test_coverage_is_parsed_from_the_status_matrix(self):
        c = self.get("/api/coverage")
        self.assertEqual(c["source"], "docs/SIH_STATUS.md")
        self.assertGreaterEqual(len(c["rows"]), 15)
        self.assertEqual(sum(c["counts"].values()), len(c["rows"]))  # every row has a known status word
        self.assertTrue(all(r["status_word"] != "UNKNOWN" for r in c["rows"]))
        self.assertTrue(any(r["requirement"].startswith("Comparison with real-world solvers") for r in c["rows"]))
        from backend import coverage as cov  # a qualified DONE with a PARTIAL caveat counts as PARTIAL
        self.assertEqual(cov.status_word("DONE on one laptop GPU (PARTIAL: no data-centre GPU yet)"), "PARTIAL")
        self.assertEqual(cov.status_word("DONE (LP)"), "DONE")

    def test_compare_endpoint_serves_the_committed_csv(self):
        c = self.get("/api/compare")
        self.assertIn("runs", c)
        for run in c["runs"]:  # zero runs is valid until a comparison CSV is committed
            self.assertRegex(run["source"]["file"], r"^compare-(highs|ortools|scip|coin)(-(ortools|scip|coin))*-.+-[0-9a-f]{7}\.csv$")
            self.assertTrue({r["solver"] for r in run["rows"]} <= {"ps26119", "highs", "ortools", "scip", "coin"})
            for r in run["rows"]:
                self.assertIn(r["solved"], ("yes", "no"))
                if r["solved"] == "yes":  # the rule the page states: Optimal + verified + within 1e-6
                    self.assertEqual((r["status"], r["verify"]), ("Optimal", "PASS"))
                    self.assertLessEqual(r["rel_err_ref"], 1e-6)

    def test_web_modules_parse(self):
        import shutil
        import subprocess
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed (syntax check of the browser modules)")
        web = os.path.join(HERE, "web", "js")
        files = [os.path.join(d, f) for d, _, fs in os.walk(web) for f in fs if f.endswith(".js")]
        self.assertGreater(len(files), 10)
        # a .js file may be parsed as CommonJS or ES module by `node --check` (Node's syntax
        # detection); a copy named .mjs forces strict ES-module parsing — how the browser loads them
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                m = os.path.join(tmp, os.path.basename(f)[:-3] + ".mjs")
                shutil.copyfile(f, m)
                p = subprocess.run([node, "--check", m], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, f + ": " + p.stderr)

    def test_evidence_matches_the_committed_csvs(self):
        e = self.get("/api/evidence")
        direct = evidence.collect()
        self.assertEqual([(n["engine"], n["solved"], n["total"]) for n in e["netlib"]],
                         [(n["engine"], n["solved"], n["total"]) for n in direct["netlib"]])
        for n in e["netlib"]:
            self.assertRegex(n["source"]["file"], r"-[0-9a-f]{7}\.csv$")


if __name__ == "__main__":
    unittest.main(verbosity=1)
