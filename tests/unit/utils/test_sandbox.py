"""Tests for the sandboxed execution utilities."""

from __future__ import annotations

import pytest

from loclean.utils.sandbox import compile_sandboxed, run_with_timeout

# ------------------------------------------------------------------
# compile_sandboxed
# ------------------------------------------------------------------


class TestCompileSandboxed:
    def test_safe_builtins_present(self) -> None:
        source = "def f():\n    return len([1, 2, 3]), int('5'), str(10), dict(a=1)\n"
        fn = compile_sandboxed(source, "f")
        assert fn() == (3, 5, "10", {"a": 1})

    def test_range_and_list_work(self) -> None:
        source = "def f():\n    return list(range(5))\n"
        fn = compile_sandboxed(source, "f")
        assert fn() == [0, 1, 2, 3, 4]

    def test_open_blocked(self) -> None:
        source = "def f():\n    return open('/etc/passwd')\n"
        fn = compile_sandboxed(source, "f")
        with pytest.raises(NameError):
            fn()

    def test_exec_blocked(self) -> None:
        source = "def f():\n    exec('x = 1')\n"
        fn = compile_sandboxed(source, "f")
        with pytest.raises(NameError):
            fn()

    def test_eval_blocked(self) -> None:
        source = "def f():\n    return eval('1 + 1')\n"
        fn = compile_sandboxed(source, "f")
        with pytest.raises(NameError):
            fn()

    def test_import_blocked(self) -> None:
        source = "def f():\n    return __import__('os')\n"
        fn = compile_sandboxed(source, "f")
        with pytest.raises(NameError):
            fn()

    def test_import_statement_blocked(self) -> None:
        source = "import os\ndef f():\n    return os.listdir('.')\n"
        with pytest.raises(ValueError, match="Compilation failed"):
            compile_sandboxed(source, "f")

    def test_allowed_modules_injected(self) -> None:
        source = "def f(x):\n    return math.log(x)\n"
        fn = compile_sandboxed(source, "f", ["math"])
        assert fn(1.0) == pytest.approx(0.0)

    def test_missing_function_raises(self) -> None:
        source = "def wrong_name():\n    return 1\n"
        with pytest.raises(ValueError, match="does not define 'target'"):
            compile_sandboxed(source, "target")

    def test_syntax_error_raises(self) -> None:
        source = "def f(\n"
        with pytest.raises(ValueError, match="Compilation failed"):
            compile_sandboxed(source, "f")

    def test_exception_types_available(self) -> None:
        source = (
            "def f(x):\n"
            "    if x < 0:\n"
            "        raise ValueError('negative')\n"
            "    return x\n"
        )
        fn = compile_sandboxed(source, "f")
        assert fn(5) == 5
        with pytest.raises(ValueError, match="negative"):
            fn(-1)


# ------------------------------------------------------------------
# run_with_timeout
# ------------------------------------------------------------------


class TestRunWithTimeout:
    def test_success_returns_result(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        result, error = run_with_timeout(add, (2, 3), timeout_s=1.0)
        assert result == 5
        assert error == ""

    def test_timeout_returns_error(self) -> None:
        import threading

        gate = threading.Event()

        def blocks() -> None:
            gate.wait(timeout=10)

        result, error = run_with_timeout(blocks, (), timeout_s=0.1)
        assert result is None
        assert "timed out" in error
        gate.set()

    def test_exception_returns_error(self) -> None:
        def bad() -> None:
            raise RuntimeError("boom")

        result, error = run_with_timeout(bad, (), timeout_s=1.0)
        assert result is None
        assert "boom" in error

    def test_result_none_is_valid(self) -> None:
        def returns_none() -> None:
            return None

        result, error = run_with_timeout(returns_none, (), timeout_s=1.0)
        assert result is None
        assert error == ""
