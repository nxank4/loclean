"""Loclean experiment tracking integration.

Provides a pluggable tracker interface for logging evaluation runs,
steps, and scores to external platforms such as Langfuse.
"""

from loclean.tracking.base import BaseTracker

__all__ = ["BaseTracker"]

try:
    from loclean.tracking.langfuse_tracker import LangfuseTracker  # noqa: F401

    __all__.append("LangfuseTracker")
except ImportError:
    pass
