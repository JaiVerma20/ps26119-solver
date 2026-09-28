"""certificate.py — the verification certificate of one solve run, assembled ONLY from what the run
produced: the solver's stdout and solution file, the in-process gate line, and tools/verify.py's
JSON report. It decides nothing numerically; it states which checks passed.

Final verdict (all must hold, otherwise "NOT CERTIFIED" with the reasons):
  1. the answer is definitive: Optimal, Infeasible or Unbounded (a limit is not an answer);
  2. the in-process gate passed on the ORIGINAL model (KKT for Optimal, Farkas / ray certificate
     for Infeasible / Unbounded);
  3. the independent verifier (different reader, different code) says PASS;
  4. the verifier read the same model (fingerprints equal; not recomputed above verify.py's size
     limit, which is stated);
  5. for generated models with an optimum known by construction: |obj - known| / (1 + |known|) <= 1e-6.
"""
from __future__ import annotations

import datetime as _dt
import re

from . import system

DEFINITIVE = ("Optimal", "Infeasible", "Unbounded")
KNOWN_TOL = 1e-6  # the verifier's relative objective tolerance (include/ps26119/tolerances.h)


def _events(job) -> dict:
    ev: dict = {}
    for e in job.events:
        ev[e["type"]] = e
    return ev


def _gate_numbers(check: str) -> dict:
    m = re.search(r"primal (\S+) dual (\S+) gap (\S+)", check or "")
    if not m:
        return {}
    out = {}
    for k, v in zip(("primal", "dual", "gap"), m.groups()):
        try:
            x = float(v)
            out[k] = x if x == x else None
        except ValueError:
            out[k] = None
    return out


def build(job) -> dict | None:
    ev = _events(job)
    res = ev.get("result")
    if job.kind != "solve" or not res:
        return None
    sol = res.get("solution") or {}
    vev = ev.get("verify") or {}
    ver = vev.get("report") or {}
    status = sol.get("status")
    claim = status in ("Infeasible", "Unbounded")
    gate_line = res.get("check_line") or ""
    sysinfo = system.probe()
    known = res.get("known")

    checks = []

    def check(name, ok, detail, required=True):
        checks.append({"name": name, "ok": ok, "detail": detail, "required": required})

    check("Definitive answer", status in DEFINITIVE, f"status {status or 'none'}"
          + ("" if status in DEFINITIVE else " — a limit or an error is not an answer"))
    check("Original-model check (in-process)", gate_line.startswith("PASS"), gate_line or "no gate line")
    if vev.get("skipped"):
        check("Independent verification", False, vev.get("reason", "skipped"))
    elif ver:
        check("Independent verification", ver.get("verdict") == "PASS",
              f"tools/verify.py {ver.get('verdict')} · reader {ver.get('reader')}"
              + (f" · {'; '.join(ver.get('reasons', []))}" if ver.get("reasons") else ""))
    else:
        check("Independent verification", False, "not run" if job.opts.get("verify") is False else "no report")
    if ver:
        skipped = ver.get("fingerprint_model") == "skipped"
        check("Same model (fingerprint)", bool(ver.get("model_match")),
              f"solver {ver.get('fingerprint_solution')}" + (" · not recomputed by the verifier at this size" if skipped
                                                             else f" = verifier {ver.get('fingerprint_model')}"
                                                             if ver.get("model_match") else f" ≠ verifier {ver.get('fingerprint_model')}"))
    if known and known.get("rel_err") is not None:
        check("Known optimum (by construction)", known["rel_err"] <= KNOWN_TOL,
              f"|obj − known| / (1 + |known|) = {known['rel_err']:.2e} (known {known['optimum']!r})")
    if status == "Optimal":
        b = sol.get("certified_bound")
        check("Rounding-proof bound", b is not None, res.get("certified_line") or "no finite bound", required=False)
    elif claim:
        cert = ver.get("certificate") or ""
        check("Certificate", gate_line.startswith("PASS") and bool(cert), cert or gate_line, required=False)

    failed = [c for c in checks if c["required"] and not c["ok"]]
    return {
        "id": job.id,
        "created": _dt.datetime.fromtimestamp(job.created).astimezone().isoformat(timespec="seconds"),
        "final": {"verdict": "PASS" if not failed else "NOT CERTIFIED", "reasons": [c["name"] for c in failed]},
        "checks": checks,
        "model": {"name": res.get("name") or job.model_rel.rsplit("/", 1)[-1], "path": job.model_rel,
                  "rows": res.get("rows"), "cols": res.get("cols"), "nnz": res.get("nnz"),
                  "fingerprint": sol.get("fingerprint") or res.get("fingerprint"),
                  "verifier_fingerprint": ver.get("fingerprint_model"), "verifier_reader": ver.get("reader")},
        "solver": {"version": sysinfo.get("version"), "git_hash": sysinfo.get("git_hash"), "binary": sysinfo.get("binary"),
                   "command": (ev.get("started") or {}).get("command")},
        "machine": {"cpu": sysinfo.get("cpu"), "cores": sysinfo.get("cores"), "os": sysinfo.get("os"),
                    "gpu": sysinfo.get("gpu"), "cuda_build": sysinfo.get("cuda_build")},
        "run": {"engine": sol.get("engine"), "requested": job.opts.get("algorithm"),
                "backend": "GPU (CUDA)" if job.opts.get("gpu") else "CPU", "threads": job.opts.get("threads"),
                "precision": sol.get("precision"), "presolve": job.opts.get("presolve", True),
                "tolerance": job.opts.get("tol"), "iterations": sol.get("iterations"), "solve_seconds": sol.get("seconds"),
                "wall_seconds": res.get("wall_seconds")},
        "result": {"status": status, "objective": sol.get("objective"), "dual_objective": sol.get("dual_objective"),
                   "certified_bound": sol.get("certified_bound"), "certified_line": res.get("certified_line"),
                   "primal_residual": sol.get("primal_residual"), "dual_residual": sol.get("dual_residual"),
                   "gap": sol.get("gap"), "message": sol.get("message")},
        "gate": {"line": gate_line, **_gate_numbers(sol.get("check", ""))},
        "verifier": {"verdict": ver.get("verdict"), "reader": ver.get("reader"), "seconds": vev.get("seconds"),
                     "objective": ver.get("objective"), "primal_rel": ver.get("primal_rel"), "dual_rel": ver.get("dual_rel"),
                     "gap_rel": ver.get("gap_rel"), "complementarity_abs": ver.get("complementarity_abs"),
                     "certificate": ver.get("certificate"), "farkas_exact_L0": ver.get("farkas_exact_L0"),
                     "reasons": ver.get("reasons", []), "skipped": vev.get("reason") if vev.get("skipped") else None},
        "known": known,
        "artifact": {"solution_file": res.get("solution_file"), "sha256": res.get("solution_sha256")},
    }
