"""Orchestration integration for workflow automation tools.

Provides a lightweight JSON-in / JSON-out runner that can be invoked
as a subprocess by orchestration tools (n8n, Apache Airflow, etc.).
"""

from loclean.orchestration.runner import (
    EXIT_INVALID_PAYLOAD,
    EXIT_OK,
    EXIT_PIPELINE_ERROR,
    EXIT_SERIALIZATION_ERROR,
    PipelinePayload,
    main,
    run_pipeline,
)

__all__ = [
    "EXIT_INVALID_PAYLOAD",
    "EXIT_OK",
    "EXIT_PIPELINE_ERROR",
    "EXIT_SERIALIZATION_ERROR",
    "PipelinePayload",
    "main",
    "run_pipeline",
]
