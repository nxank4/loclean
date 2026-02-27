"""Extraction module for structured data extraction using Pydantic schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .extract_dataframe import extract_dataframe_compiled
from .extractor import Extractor

if TYPE_CHECKING:
    from .feature_discovery import FeatureDiscovery
    from .leakage_auditor import TargetLeakageAuditor
    from .missingness_recognizer import MissingnessRecognizer
    from .optimizer import InstructionOptimizer
    from .oversampler import SemanticOversampler
    from .resolver import EntityResolver
    from .shredder import RelationalShredder
    from .trap_pruner import TrapPruner

__all__ = [
    "EntityResolver",
    "Extractor",
    "FeatureDiscovery",
    "InstructionOptimizer",
    "MissingnessRecognizer",
    "RelationalShredder",
    "SemanticOversampler",
    "TargetLeakageAuditor",
    "TrapPruner",
    "extract_dataframe_compiled",
]

_LAZY_IMPORTS: dict[str, str] = {
    "EntityResolver": ".resolver",
    "FeatureDiscovery": ".feature_discovery",
    "InstructionOptimizer": ".optimizer",
    "MissingnessRecognizer": ".missingness_recognizer",
    "RelationalShredder": ".shredder",
    "SemanticOversampler": ".oversampler",
    "TargetLeakageAuditor": ".leakage_auditor",
    "TrapPruner": ".trap_pruner",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is not None:
        import importlib

        module = importlib.import_module(module_path, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
