#!/usr/bin/env python3
"""AgentDS dev launcher.

Starts the FastAPI backend (:8000) and the Next.js frontend (:3000) together,
waits for the frontend to come up, then prints the link. Works on Windows,
WSL, macOS and Linux. Standard library only — no extra install.

    python run.py                 # both services
    python run.py --backend-only
    python run.py --frontend-only
    python run.py --no-open       # don't try to open a browser

Ctrl+C stops both. If either process exits on its own, the other is stopped too.
Override URLs/ports with AGENTDS_FRONTEND_URL / AGENTDS_BACKEND_URL /
BACKEND_HOST / BACKEND_PORT.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
IS_WIN = os.name == "nt"


def say(*a: object) -> None:
    print(*a, flush=True)

BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = os.environ.get("BACKEND_PORT", "8000")
FRONTEND_URL = os.environ.get("AGENTDS_FRONTEND_URL", "http://localhost:3000")
BACKEND_URL = os.environ.get(
    "AGENTDS_BACKEND_URL", f"http://localhost:{BACKEND_PORT}"
)


def backend_python() -> str:
    """Prefer a virtualenv beside the backend; fall back to this interpreter."""
    for rel in (
        "venv/Scripts/python.exe",
        "venv/bin/python",
        ".venv/Scripts/python.exe",
        ".venv/bin/python",
    ):
        p = BACKEND / rel
        if p.exists():
            return str(p)
    return sys.executable


def npm_cmd() -> str:
    return "npm.cmd" if IS_WIN else "npm"


def open_browser(url: str) -> None:
    # On WSL a Linux webbrowser call usually fails silently; try wslview first.
    if shutil.which("wslview"):
        try:
            subprocess.Popen(["wslview", url])
            return
        except Exception:
            pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


def wait_for(url: str, timeout: float = 120.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True  # a 404/500 still means the server is listening
        except Exception:
            time.sleep(0.6)
    return False


def spawn(cmd: list[str], cwd: Path) -> subprocess.Popen:
    """Start a child in its own process group so we can signal the whole tree
    (npm -> next, uvicorn reloader -> worker) on shutdown, not just the parent."""
    kwargs: dict = {}
    if IS_WIN:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, cwd=str(cwd), **kwargs)


def stop(p: subprocess.Popen, hard: bool = False) -> None:
    if p.poll() is not None:
        return
    try:
        if IS_WIN:
            p.kill() if hard else p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            sig = signal.SIGKILL if hard else signal.SIGTERM
            os.killpg(os.getpgid(p.pid), sig)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def ensure_backend_env() -> None:
    env, example = BACKEND / ".env", BACKEND / ".env.example"
    if not env.exists() and example.exists():
        env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        say("  (created backend/.env from .env.example)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the AgentDS backend + frontend.")
    ap.add_argument("--backend-only", action="store_true")
    ap.add_argument("--frontend-only", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args()

    run_backend = not args.frontend_only
    run_frontend = not args.backend_only
    procs: list[tuple[str, subprocess.Popen]] = []

    if run_backend:
        if not BACKEND.is_dir():
            print(f"! backend dir not found: {BACKEND}", file=sys.stderr)
            return 1
        ensure_backend_env()
        py = backend_python()
        say(f"-> backend   {py} -m uvicorn app.main:app  (cwd: backend)")
        procs.append(
            (
                "backend",
                spawn(
                    [
                        py, "-m", "uvicorn", "app.main:app", "--reload",
                        "--host", BACKEND_HOST, "--port", str(BACKEND_PORT),
                    ],
                    BACKEND,
                ),
            )
        )

    if run_frontend:
        if not FRONTEND.is_dir():
            print(f"! frontend dir not found: {FRONTEND}", file=sys.stderr)
            return 1
        if shutil.which("npm") is None and shutil.which(npm_cmd()) is None:
            print("! npm not found on PATH - install Node 20+", file=sys.stderr)
            return 1
        say(f"-> frontend  {npm_cmd()} run dev  (cwd: frontend)")
        procs.append(
            ("frontend", spawn([npm_cmd(), "run", "dev"], FRONTEND))
        )

    if not procs:
        print("nothing to run", file=sys.stderr)
        return 1

    def announce() -> None:
        if run_frontend and wait_for(FRONTEND_URL):
            line = "-" * 52
            say(f"\n{line}")
            say(f"  Frontend : {FRONTEND_URL}")
            if run_backend:
                say(f"  API docs : {BACKEND_URL}/docs")
            say(f"{line}\n  Ctrl+C to stop.\n")
            if not args.no_open:
                open_browser(FRONTEND_URL)
        elif run_frontend:
            say(f"! frontend did not respond at {FRONTEND_URL} in time")
        elif run_backend:
            say(f"\n  API docs : {BACKEND_URL}/docs\n  Ctrl+C to stop.\n")

    threading.Thread(target=announce, daemon=True).start()

    try:
        while True:
            for name, p in procs:
                rc = p.poll()
                if rc is not None:
                    say(f"\n! {name} exited (code {rc}) - stopping the rest.")
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        say("\nstopping...")
    finally:
        for _name, p in procs:
            stop(p)
        deadline = time.time() + 8
        for _name, p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                stop(p, hard=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
