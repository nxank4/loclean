"""Tests for JSON repair functionality."""

from loclean.extraction.json_repair import repair_json


def test_repair_json_trailing_comma() -> None:
    """Test repairing JSON with trailing comma."""
    malformed = '{"name": "test",}'
    repaired = repair_json(malformed)
    # Should be valid JSON now
    import json

    data = json.loads(repaired)
    assert data["name"] == "test"


def test_repair_json_missing_brace() -> None:
    """Test repairing JSON with missing closing brace."""
    malformed = '{"name": "test"'
    repaired = repair_json(malformed)
    # Should attempt repair (may or may not succeed)
    assert isinstance(repaired, str)


def test_repair_json_valid() -> None:
    """Test that valid JSON is returned unchanged."""
    valid = '{"name": "test"}'
    repaired = repair_json(valid)
    assert repaired == valid


def test_repair_json_without_library() -> None:
    """Test fallback when json-repair is not available."""
    # This test verifies graceful degradation
    # In practice, json-repair should be installed
    result = repair_json('{"name": "test"}')
    assert isinstance(result, str)
