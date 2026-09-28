#!/usr/bin/env python3
"""server.py — the PS26119 Optimization Command Center (local web UI).

    python3 apps/ui/server.py            then open http://127.0.0.1:8765
    python3 apps/ui/server.py --port 9000 --open

A thin layer: it runs the real `ps26119` CLI (build/ps26119), reads its solution files, runs the
independent verifier tools/verify.py and serves benchmark evidence from the committed CSVs. It
never computes or invents a solver result. Standard library only; listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import certificate, evidence, generate, models, paths, preflight, report, runner, scenarios, system  # noqa: E402

JOBS = runner.Jobs()


def clean(o):
    """JSON has no NaN / Infinity: non-finite numbers become null (an Infeasible objective is NaN,
    an unavailable certified bound is ±inf). Without this the browser's JSON.parse rejects the event."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def dumps(o) -> str:
    return json.dumps(clean(o), allow_nan=False, default=lambda v: None)


MAX_UPLOAD = 256 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "ps26119-ui"

    def log_message(self, fmt, *args):  # quieter console
        if os.environ.get("PS26119_UI_LOG"):
            super().log_message(fmt, *args)

    # ---------------------------------------------------------------- helpers
    def send_json(self, obj, code=200):
        body = dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, code, message):
        self.send_json({"error": message}, code)

    def query(self):
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def model_arg(self):
        p = paths.resolve_model(self.query().get("path", ""))
        if not p:
            self.error(404, "unknown model (only files in the data folders can be opened)")
        return p

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        try:
            if route == "/api/system":
                return self.send_json(system.probe())
            if route == "/api/models":
                return self.send_json(models.catalog())
            if route == "/api/model/info":
                p = self.model_arg()
                return p and self.send_json({"path": paths.rel(p), **models.info(p)})
            if route == "/api/model/sparsity":
                p = self.model_arg()
                return p and self.send_json(models.sparsity(p))
            if route == "/api/scenarios":
                return self.send_json({"levers": {k: {"label": v[0], "what": v[1], "range": v[2]} for k, v in scenarios.LEVERS.items()},
                                       "models": scenarios.refinery_models(), "products": scenarios.PRODUCTS,
                                       "max_sweep": scenarios.MAX_SWEEP})
            if route == "/api/evidence":
                return self.send_json(evidence.collect())
            if route == "/api/gpu":
                return self.send_json(evidence.gpu_detail())
            if route == "/api/preflight":
                return self.send_json(preflight.run())
            if route == "/api/runs":
                return self.send_json(runner.list_runs())
            if route == "/api/report":
                return self.report()
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(/events|/certificate)?", route)
            if m:
                job = JOBS.get(m[1])
                if not job:
                    return self.error(404, "unknown job")
                if m[2] == "/certificate":
                    c = certificate.build(job)
                    return self.send_json(c) if c else self.error(409, "no certificate: not a finished solve run")
                return self.stream(job) if m[2] else self.send_json({"events": job.events, "done": job.done})
            return self.static(route)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 — the UI must get an error, never a dead socket
            self.error(500, f"{type(e).__name__}: {e}")

    def report(self):
        """/api/report?jobs=a,b&evidence=1&download=1 — the self-contained HTML report."""
        q = self.query()
        ids = [j for j in q.get("jobs", "").split(",") if j]
        if len(ids) > 20 or not all(re.fullmatch(r"[0-9a-f]{12}", j) for j in ids):
            return self.error(400, "jobs: up to 20 run ids")
        certs = []
        for jid in ids:
            job = JOBS.get(jid)
            c = certificate.build(job) if job else None
            if not c:
                return self.error(404, f"no certificate for run {jid}")
            certs.append(c)
        title = q.get("title") or ("Verification certificate" if len(certs) == 1 else "Verification report")
        body = report.render(certs, include_evidence=q.get("evidence", "1") != "0", title=title[:120]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if q.get("download"):
            name = "ps26119-" + ("-".join(c["model"]["name"] for c in certs[:3]) or "evidence") + ".html"
            self.send_header("Content-Disposition", f'attachment; filename="{re.sub(r"[^A-Za-z0-9._-]", "_", name)}"')
        self.end_headers()
        self.wfile.write(body)

    def static(self, route):
        rel = "index.html" if route in ("", "/") else route.lstrip("/")
        p = os.path.realpath(os.path.join(paths.WEB, rel))
        if not p.startswith(os.path.realpath(paths.WEB) + os.sep) or not os.path.isfile(p):
            return self.error(404, "not found")
        with open(p, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(p)[0] or "application/octet-stream") +
                         ("; charset=utf-8" if p.endswith((".html", ".js", ".css")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def stream(self, job):
        """Server-sent events: every job event, as it happens."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        sent = 0
        while True:
            with job.cond:
                while sent >= len(job.events) and not job.done:
                    job.cond.wait(timeout=15)
                    if sent >= len(job.events) and not job.done:
                        break  # keep-alive below
                batch = job.events[sent:]
                done = job.done and sent + len(batch) >= len(job.events)
            if not batch and not done:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                continue
            for ev in batch:
                self.wfile.write(f"data: {dumps(ev)}\n\n".encode())
            sent += len(batch)
            self.wfile.flush()
            if done:
                self.wfile.write(b"data: {\"type\": \"end\"}\n\n")
                self.wfile.flush()
                return

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        try:
            if route == "/api/solve":
                body = self.body_json()
                model = paths.resolve_model(body.get("path", ""))
                if not model:
                    return self.error(404, "unknown model")
                opts, err = runner.validate_options(body)
                if err:
                    return self.error(400, err)
                job = JOBS.start(paths.rel(model), opts)
                return self.send_json({"job": job.id})
            if route == "/api/scenario":
                o, err = scenarios.validate(self.body_json())
                if err:
                    return self.error(400, err)
                return self.send_json({"job": JOBS.add(scenarios.ScenarioJob(o)).id})
            if route == "/api/preflight/prepare":
                return self.send_json({"generated": preflight.prepare(), **preflight.run()})
            if route == "/api/generate":
                args, err = generate.validate(self.body_json())
                if err:
                    return self.error(400, err)
                return self.send_json(generate.generate(*args))
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/cancel", route)
            if m:
                job = JOBS.get(m[1])
                if job and hasattr(job, "cancelled"):
                    job.cancelled = True
                if job and job.proc and job.proc.poll() is None:
                    job.proc.terminate()
                    job.emit("log", line="cancelled by the user")
                return self.send_json({"ok": True})
            if route == "/api/upload":
                name = os.path.basename(self.query().get("name", ""))
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", name) or not name.endswith(paths.MODEL_EXT):
                    return self.error(400, "file name must be [A-Za-z0-9._-] and end in .mps or .lpm")
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= MAX_UPLOAD:
                    return self.error(400, "empty or too large (max 256 MB)")
                os.makedirs(paths.UPLOADS, exist_ok=True)
                dest = os.path.join(paths.UPLOADS, name)
                with open(dest, "wb") as f:
                    remaining = n
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                return self.send_json({"path": paths.rel(dest)})
            return self.error(404, "not found")
        except (ValueError, json.JSONDecodeError) as e:
            self.error(400, str(e))
        except Exception as e:  # noqa: BLE001
            self.error(500, f"{type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PS26119 Optimization Command Center")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args(argv)
    if not os.path.exists(paths.BIN):
        print(f"solver binary not found: {paths.BIN} (build it first: cmake --build build)", file=sys.stderr)
        return 3
    os.makedirs(paths.RUNS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{a.port}"
    print(f"PS26119 Command Center on {url}  (solver {paths.rel(paths.BIN)}; Ctrl+C to stop)")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
