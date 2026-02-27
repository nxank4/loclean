"""Sandboxed execution utilities for LLM-generated code.

Provides restricted ``exec`` compilation and wall-clock timeout
enforcement via ``concurrent.futures.ThreadPoolExecutor``.
"""

from __future__ import annotations

import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, TypeVar

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)

T = TypeVar("T")

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ArithmeticError": ArithmeticError,
    "OverflowError": OverflowError,
}


class SandboxTimeoutError(RuntimeError):
    """Raised when sandboxed execution exceeds the time limit."""


def compile_sandboxed(
    source: str,
    fn_name: str,
    allowed_modules: list[str] | None = None,
) -> Callable[..., Any]:
    """Compile *source* in a restricted namespace and return *fn_name*.

    The execution environment has:

    * ``__builtins__`` replaced by a curated safe subset (no ``open``,
      ``exec``, ``eval``, ``compile``, ``exit``, ``quit``, ``input``,
      ``breakpoint``, ``globals``, ``locals``, ``vars``, ``dir``).
    * A restricted ``__import__`` that only permits explicitly listed
      modules — LLM-generated ``import`` statements work for allowed
      modules but raise ``ImportError`` for anything else.
    * Only explicitly listed standard-library modules injected.

    Args:
        source: Python source code string.
        fn_name: Name of the function to extract from the namespace.
        allowed_modules: Standard-library module names to inject
            (e.g. ``["math", "re"]``).

    Returns:
        The compiled callable.

    Raises:
        ValueError: If compilation fails or *fn_name* is not defined.
    """
    allowed = set(allowed_modules or [])
    preloaded: dict[str, Any] = {}

    for mod_name in allowed:
        try:
            preloaded[mod_name] = importlib.import_module(mod_name)
        except ImportError:
            logger.warning(
                f"[yellow]⚠[/yellow] Module '{mod_name}' not available, skipping"
            )

    def _restricted_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in the sandbox. "
                f"Permitted modules: {sorted(allowed)}"
            )
        if root in preloaded:
            return preloaded[root]
        return importlib.import_module(name)

    builtins = _SAFE_BUILTINS.copy()
    builtins["__import__"] = _restricted_import

    safe_globals: dict[str, Any] = {"__builtins__": builtins}
    safe_globals.update(preloaded)

    try:
        exec(source, safe_globals)  # noqa: S102
    except Exception as exc:
        raise ValueError(f"Compilation failed: {exc}") from exc

    fn = safe_globals.get(fn_name)
    if fn is None or not callable(fn):
        raise ValueError(f"Source does not define '{fn_name}'")
    return fn  # type: ignore[no-any-return]


def run_with_timeout(
    fn: Callable[..., T],
    args: tuple[Any, ...],
    timeout_s: float = 2.0,
) -> tuple[T | None, str]:
    """Execute *fn* with a wall-clock time limit.

    Uses a daemon-threaded pool so the interpreter can exit even
    when a timed-out function is still running.

    Args:
        fn: Callable to execute.
        args: Positional arguments forwarded to *fn*.
        timeout_s: Maximum seconds to wait.

    Returns:
        ``(result, "")`` on success, or ``(None, error_message)`` on
        timeout or exception.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args)
    try:
        result = future.result(timeout=timeout_s)
        return result, ""
    except FuturesTimeoutError:
        msg = f"Execution timed out after {timeout_s}s"
        logger.warning(f"[yellow]⚠[/yellow] {msg}")
        return None, msg
    except Exception as exc:
        return None, str(exc)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
