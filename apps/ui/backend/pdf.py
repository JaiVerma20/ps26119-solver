"""pdf.py — the verification report as a PDF, printed by a locally installed Chrome / Chromium in
headless mode (--print-to-pdf) from the same self-contained HTML the report endpoint serves. Optional:
without a browser binary the UI offers the HTML report (which prints to PDF from any browser).
A throwaway profile and a mock keychain are used, so no user data is touched and no prompt appears."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
]


def browser() -> str | None:
    env = os.environ.get("PS26119_PDF_BROWSER")
    for c in ([env] if env else []) + CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.exists(p):
            return p
    return None


def html_to_pdf(html: str, timeout: float = 90) -> bytes:
    exe = browser()
    if not exe:
        raise RuntimeError("no Chrome / Chromium found for PDF export (use the HTML report and print it)")
    with tempfile.TemporaryDirectory(prefix="ps26119-pdf-") as tmp:
        src, out = os.path.join(tmp, "report.html"), os.path.join(tmp, "report.pdf")
        with open(src, "w", encoding="utf-8") as f:
            f.write(html)
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run", "--use-mock-keychain", "--password-store=basic",
               "--disable-extensions", f"--user-data-dir={os.path.join(tmp, 'profile')}", "--no-pdf-header-footer",
               f"--print-to-pdf={out}", "file://" + src]
        # headless Chrome on macOS may keep running after it has written the PDF: wait for the file
        # to appear and stop growing, then end the browser ourselves
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline, last = time.time() + timeout, -1
            while time.time() < deadline:
                size = os.path.getsize(out) if os.path.exists(out) else -1
                if size > 1000 and size == last:
                    break
                if proc.poll() is not None and size <= 0:
                    break
                last = size
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        if not os.path.exists(out) or os.path.getsize(out) < 1000:
            raise RuntimeError("the browser did not produce a PDF")
        with open(out, "rb") as f:
            return f.read()
