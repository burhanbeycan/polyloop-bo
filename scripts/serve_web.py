#!/usr/bin/env python3
"""Serve the repository with a local static HTTP server.

Usage:
    python scripts/serve_web.py --port 8000

Then open http://127.0.0.1:8000 to view the interactive site.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the PolyLoop‑BO interactive website locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))

    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Serving {root} at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
        server.serve_forever()


if __name__ == "__main__":
    main()
