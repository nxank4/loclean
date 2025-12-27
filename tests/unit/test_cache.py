from typing import Any

from loclean.cache import LocleanCache


def test_hashing_determinism(temp_cache_db: Any) -> None:
    """
    Verify that _hash produces consistent output for the same input.
    """
    cache = LocleanCache(cache_dir=temp_cache_db.parent)
    # The cache class usually takes a dir, and creates cache.db inside.
    # Our fixture gives a full file path, but the class expects a dir.
    # We'll pass the parent of the tmp_path.

    # Actually, looking at LocleanCache implementation:
    # __init__(self, cache_dir: Optional[Path] = None)
    # self.db_path = self.cache_dir / "cache.db"

    # So we should pass a directory.
    cache_dir = temp_cache_db.parent
    cache = LocleanCache(cache_dir=cache_dir)

    h1 = cache._hash("10kg", "Convert to kg")
    h2 = cache._hash("10kg", "Convert to kg")
    h3 = cache._hash("10kg", "Different instruction")

    assert h1 == h2
    assert h1 != h3


def test_set_and_get_batch(temp_cache_db: Any) -> None:
    """
    Verify that we can save to and retrieve from the cache.
    """
    cache_dir = temp_cache_db.parent
    cache = LocleanCache(cache_dir=cache_dir)

    items = ["10kg", "500g"]
    instruction = "Extract weight"

    results = {"10kg": {"value": 10, "unit": "kg"}, "500g": {"value": 500, "unit": "g"}}

    # Set
    cache.set_batch(items, instruction, results)

    # Get
    cached_items = cache.get_batch(items, instruction)

    assert len(cached_items) == 2
    assert cached_items["10kg"] == results["10kg"]
    assert cached_items["500g"] == results["500g"]


def test_cache_miss(temp_cache_db: Any) -> None:
    """
    Verify that get_batch returns empty dict for unknown keys.
    """
    cache_dir = temp_cache_db.parent
    cache = LocleanCache(cache_dir=cache_dir)

    items = ["unknown_item"]
    instruction = "Extract weight"

    cached_items = cache.get_batch(items, instruction)

    assert cached_items == {}
