"""Command-line launcher for the AIMS4PT web interface."""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Sequence

from aims4pt_web.config import settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8003
PORT_SCAN_LIMIT = 100
STARTUP_TIMEOUT_SECONDS = 180
READINESS_INTERVAL_SECONDS = 0.5


def _find_available_port(host: str, preferred_port: int) -> int:
    """Return the first available TCP port at or above the preferred port."""
    for port in range(preferred_port, preferred_port + PORT_SCAN_LIMIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No available port found from {preferred_port} to "
        f"{preferred_port + PORT_SCAN_LIMIT - 1}."
    )


def _start_server(host: str, port: int) -> subprocess.Popen[bytes]:
    """Start uvicorn in a background subprocess."""
    command: Sequence[str] = (
        sys.executable,
        "-m",
        "uvicorn",
        "aims4pt_web.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(settings.web_workers_recommended),
    )
    return subprocess.Popen(command)


def _wait_until_ready(url: str, process: subprocess.Popen[bytes]) -> None:
    """Block until the HTTP server responds or startup fails."""
    deadline = time.monotonic() + int(
        os.getenv("AIMS4PT_WEB_STARTUP_TIMEOUT_SECONDS", STARTUP_TIMEOUT_SECONDS)
    )
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Web server exited before it was ready with code {process.returncode}."
            ) from last_error
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(READINESS_INTERVAL_SECONDS)

    raise TimeoutError(f"Timed out waiting for the web server at {url}.") from last_error


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    """Terminate the background server process if it is still running."""
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _handle_termination_signal(
    signum: int,
    frame: object,
) -> None:
    """Convert an application termination request into a clean shutdown."""
    del signum, frame
    raise KeyboardInterrupt


def main() -> None:
    """Launch the local web server and open it in the default browser."""
    host = os.getenv("AIMS4PT_WEB_HOST", DEFAULT_HOST)
    preferred_port = int(os.getenv("AIMS4PT_WEB_PORT", str(DEFAULT_PORT)))
    port = _find_available_port(host, preferred_port)
    url = f"http://{host}:{port}"

    print(f"Starting AIMS4PT web server at {url}")
    process = _start_server(host, port)
    atexit.register(_stop_server, process)
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    try:
        _wait_until_ready(url, process)
        print(f"Opening {url}")
        webbrowser.open(url)
        print("Press Ctrl+C to stop the web server.")
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping AIMS4PT web server.")
    finally:
        _stop_server(process)


if __name__ == "__main__":
    main()
