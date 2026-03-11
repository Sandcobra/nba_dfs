"""NBA DFS Pro — launch the web UI.

Usage:
    python run_ui.py
    python run_ui.py --port 8080
    python run_ui.py --host 0.0.0.0 --port 8080

Then open http://localhost:8080 in your browser.
"""
import subprocess
import sys
import argparse

# ── Auto-install missing packages ─────────────────────────────────────────
REQUIRED = ["fastapi", "uvicorn", "python-multipart"]
missing  = []
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Installing: {', '.join(missing)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q"])

# ── Parse args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NBA DFS Pro UI")
parser.add_argument("--host",   default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
parser.add_argument("--port",   default=8080, type=int, help="Port (default: 8080)")
parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev)")
args = parser.parse_args()

# ── Launch ─────────────────────────────────────────────────────────────────
import uvicorn  # noqa: E402  (imported after auto-install)

url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
print(f"\n  NBA DFS Pro  |  {url}\n")

uvicorn.run(
    "nba_dfs.ui.app:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
    log_level="warning",
)
