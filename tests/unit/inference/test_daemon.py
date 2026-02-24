"""Tests for loclean.inference.daemon module."""

from unittest.mock import MagicMock, patch

import pytest

from loclean.inference.daemon import (
    _find_ollama_binary,
    _is_local_host,
    _ping,
    _start_daemon,
    ensure_daemon,
)


class TestIsLocalHost:
    """Tests for _is_local_host."""

    @pytest.mark.parametrize(
        "host",
        [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://0.0.0.0:11434",
            "http://[::1]:11434",
        ],
    )
    def test_local_hosts(self, host: str) -> None:
        assert _is_local_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "http://gpu-server:11434",
            "http://192.168.1.100:11434",
            "http://ollama.example.com:11434",
        ],
    )
    def test_remote_hosts(self, host: str) -> None:
        assert _is_local_host(host) is False

    def test_empty_host_defaults_to_non_local(self) -> None:
        assert _is_local_host("") is False


class TestPing:
    """Tests for _ping."""

    @patch("loclean.inference.daemon.urlopen")
    def test_successful_ping(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        assert _ping("http://localhost:11434") is True

    @patch("loclean.inference.daemon.urlopen")
    def test_failed_ping_connection_error(self, mock_urlopen: MagicMock) -> None:
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")
        assert _ping("http://localhost:11434") is False

    @patch("loclean.inference.daemon.urlopen")
    def test_failed_ping_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = OSError("timeout")
        assert _ping("http://localhost:11434") is False


class TestFindOllamaBinary:
    """Tests for _find_ollama_binary."""

    @patch("loclean.inference.daemon.shutil.which", return_value="/usr/bin/ollama")
    def test_binary_found(self, _mock_which: MagicMock) -> None:
        assert _find_ollama_binary() == "/usr/bin/ollama"

    @patch("loclean.inference.daemon.shutil.which", return_value=None)
    def test_binary_not_found_raises(self, _mock_which: MagicMock) -> None:
        with pytest.raises(FileNotFoundError, match="ollama.*not found"):
            _find_ollama_binary()


class TestStartDaemon:
    """Tests for _start_daemon."""

    @patch("loclean.inference.daemon.subprocess.Popen")
    def test_popen_called_with_serve(self, mock_popen: MagicMock) -> None:
        _start_daemon("/usr/bin/ollama", "http://localhost:11434")

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert args[0] == ["/usr/bin/ollama", "serve"]
        assert kwargs["start_new_session"] is True
        assert "OLLAMA_HOST" in kwargs["env"]

    @patch("loclean.inference.daemon.subprocess.Popen")
    def test_env_contains_correct_bind_address(self, mock_popen: MagicMock) -> None:
        _start_daemon("/usr/bin/ollama", "http://localhost:8080")

        _, kwargs = mock_popen.call_args
        assert kwargs["env"]["OLLAMA_HOST"] == "localhost:8080"


class TestEnsureDaemon:
    """Tests for ensure_daemon."""

    @patch("loclean.inference.daemon._ping", return_value=True)
    def test_already_running_returns_immediately(self, mock_ping: MagicMock) -> None:
        ensure_daemon("http://localhost:11434")
        mock_ping.assert_called_once_with("http://localhost:11434")

    @patch("loclean.inference.daemon._ping", return_value=False)
    def test_remote_host_raises_connection_error(self, _mock_ping: MagicMock) -> None:
        with pytest.raises(ConnectionError, match="remote"):
            ensure_daemon("http://gpu-server:11434")

    @patch("loclean.inference.daemon._ping", side_effect=[False, True])
    @patch("loclean.inference.daemon._start_daemon")
    @patch(
        "loclean.inference.daemon._find_ollama_binary",
        return_value="/usr/bin/ollama",
    )
    @patch("loclean.inference.daemon.time.sleep")
    def test_auto_starts_local_daemon(
        self,
        _mock_sleep: MagicMock,
        mock_find: MagicMock,
        mock_start: MagicMock,
        _mock_ping: MagicMock,
    ) -> None:
        ensure_daemon("http://localhost:11434", timeout=5, poll_interval=0.1)

        mock_find.assert_called_once()
        mock_start.assert_called_once_with("/usr/bin/ollama", "http://localhost:11434")

    @patch("loclean.inference.daemon._ping", return_value=False)
    @patch("loclean.inference.daemon._start_daemon")
    @patch(
        "loclean.inference.daemon._find_ollama_binary",
        return_value="/usr/bin/ollama",
    )
    @patch("loclean.inference.daemon.time.sleep")
    @patch("loclean.inference.daemon.time.monotonic", side_effect=[0, 0.5, 100])
    def test_timeout_raises_connection_error(
        self,
        _mock_monotonic: MagicMock,
        _mock_sleep: MagicMock,
        _mock_find: MagicMock,
        _mock_start: MagicMock,
        _mock_ping: MagicMock,
    ) -> None:
        with pytest.raises(ConnectionError, match="did not become reachable"):
            ensure_daemon("http://localhost:11434", timeout=1)

    @patch("loclean.inference.daemon._ping", return_value=False)
    @patch(
        "loclean.inference.daemon._find_ollama_binary",
        side_effect=FileNotFoundError("not found"),
    )
    def test_missing_binary_raises_file_not_found(
        self, _mock_find: MagicMock, _mock_ping: MagicMock
    ) -> None:
        with pytest.raises(FileNotFoundError):
            ensure_daemon("http://localhost:11434")
