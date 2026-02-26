"""Deterministic cache-key generation for compiled extraction functions."""

from __future__ import annotations

import hashlib


def compute_code_key(
    *,
    columns: list[str],
    dtypes: list[str],
    target_col: str,
    module_prefix: str,
) -> str:
    """Build a SHA256 key from structural metadata.

    Implements ``Key = H(module_prefix + target_col +
    sorted(columns) + sorted(dtypes))``.

    Args:
        columns: DataFrame column names.
        dtypes: Corresponding dtype strings.
        target_col: Name of the target / log column.
        module_prefix: Module-level discriminator (e.g. ``"feature_discovery"``
            or ``"shredder"``).

    Returns:
        Hex-encoded SHA256 digest.
    """
    parts = [
        module_prefix,
        target_col,
        ",".join(sorted(columns)),
        ",".join(sorted(dtypes)),
    ]
    payload = "::".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
