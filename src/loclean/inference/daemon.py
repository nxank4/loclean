"""Ollama daemon lifecycle management.

Provides helpers to detect, start, and wait for the local Ollama daemon
so that ``OllamaEngine`` can self-bootstrap without manual user steps.
"""

import logging
import os
import shutil
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)

_DAEMON_TIMEOUT = 30.0
_POLL_INTERVAL = 0.5
_PING_TIMEOUT = 3.0
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_local_host(host: str) -> bool:
    """Determine whether *host* points to the local machine.

    Args:
        host: Ollama server URL (e.g. ``http://localhost:11434``).

    Returns:
        ``True`` when the hostname resolves to a loopback address.
    """
    parsed = urlparse(host)
    hostname = (parsed.hostname or "").lower()
    return hostname in _LOCAL_HOSTNAMES


def _ping(host: str, timeout: float = _PING_TIMEOUT) -> bool:
    """Send a lightweight HTTP GET to the Ollama version endpoint.

    Args:
        host: Ollama server URL.
        timeout: Socket timeout in seconds.

    Returns:
        ``True`` if the server responded with a 2xx status.
    """
    url = f"{host.rstrip('/')}/api/version"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            status: int = resp.status
            return 200 <= status < 300
    except (URLError, OSError, ValueError):
        return False


def _find_ollama_binary() -> str:
    """Locate the ``ollama`` executable on ``$PATH``.

    Returns:
        Absolute path to the binary.

    Raises:
        FileNotFoundError: If the binary is not installed.
    """
    binary = shutil.which("ollama")
    if binary is None:
        raise FileNotFoundError(
            "The 'ollama' executable was not found on your system PATH.\n"
            "Please install Ollama before using loclean:\n"
            "  • Linux / WSL:  curl -fsSL https://ollama.com/install.sh | sh\n"
            "  • macOS:        brew install ollama\n"
            "  • All platforms: https://ollama.com/download"
        )
    return binary


def _start_daemon(binary: str, host: str) -> subprocess.Popen[bytes]:
    """Launch the Ollama server as a detached background process.

    Sets ``OLLAMA_HOST`` so the daemon binds to the configured address.

    Args:
        binary: Path to the ``ollama`` executable.
        host: Ollama server URL (used to derive the bind address).

    Returns:
        The ``Popen`` handle (caller does **not** need to manage it).
    """
    env = os.environ.copy()
    parsed = urlparse(host)
    bind_addr = f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 11434}"
    env["OLLAMA_HOST"] = bind_addr

    logger.info(
        f"[yellow]⏳[/yellow] Starting Ollama daemon ([bold]{bind_addr}[/bold]) …"
    )

    return subprocess.Popen(
        [binary, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def ensure_daemon(
    host: str,
    timeout: float = _DAEMON_TIMEOUT,
    poll_interval: float = _POLL_INTERVAL,
) -> None:
    """Guarantee that an Ollama daemon is reachable at *host*.

    For **local** hosts the function will:
    1. Ping the server.
    2. If unreachable, locate and start the ``ollama serve`` binary.
    3. Poll until the socket opens or *timeout* expires.

    For **remote** hosts only step 1 is performed; failure raises
    immediately.

    Args:
        host: Ollama server URL.
        timeout: Maximum seconds to wait for the daemon to start.
        poll_interval: Seconds between ping attempts.

    Raises:
        FileNotFoundError: If the ``ollama`` binary is missing (local only).
        ConnectionError: If the daemon did not become reachable in time.
    """
    if _ping(host):
        return

    if not _is_local_host(host):
        raise ConnectionError(
            f"Could not reach Ollama at {host}. "
            "The host appears to be remote so automatic startup is "
            "not available. Please ensure the Ollama server is running "
            "on the target machine."
        )

    binary = _find_ollama_binary()
    _start_daemon(binary, host)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        if _ping(host):
            logger.info("[green]✓[/green] Ollama daemon is now running.")
            return

    raise ConnectionError(
        f"Ollama daemon did not become reachable at {host} within "
        f"{timeout:.0f} seconds. Please check the installation and try "
        "starting it manually with: ollama serve"
    )
