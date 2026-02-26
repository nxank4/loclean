import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from loclean.utils.logging import configure_module_logger

logger = configure_module_logger(__name__, level=logging.INFO)


class LocleanCache:
    """
    Persistent caching layer using SQLite3.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the cache.

        Args:
            cache_dir: Optional directory for the cache database.
                       Defaults to ~/.cache/loclean.
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "loclean"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """
        Initialize the database schema and enable WAL mode.

        WAL (Write-Ahead Logging) mode allows concurrent reads and writes,
        improving performance for cache operations that may be called from
        multiple threads or processes.
        """
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_cache (
                hash_key TEXT PRIMARY KEY,
                json_response TEXT,
                last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_cache (
                hash_key TEXT PRIMARY KEY,
                source_code TEXT NOT NULL,
                last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def _hash(self, text: str, instruction: str) -> str:
        """
        Generate a SHA256 hash specific to the input text and instruction.

        The "v3::" prefix allows cache invalidation when the extraction logic
        changes, ensuring users get updated results after schema changes.
        """
        uniq_str = f"v3::{instruction}::{text}"
        return hashlib.sha256(uniq_str.encode("utf-8")).hexdigest()

    def get_batch(self, items: List[str], instruction: str) -> Dict[str, Any]:
        """
        Retrieve cached results for a batch of items.

        Args:
            items: List of raw input strings.
            instruction: The instruction used.

        Returns:
            Dictionary mapping original_text -> parsed_json_dict.
            Only contains items found in the cache.
        """
        item_map = {self._hash(item, instruction): item for item in items}
        hashes = list(item_map.keys())

        if not hashes:
            return {}

        placeholders = ",".join(["?"] * len(hashes))
        query = (
            f"SELECT hash_key, json_response FROM inference_cache "
            f"WHERE hash_key IN ({placeholders})"
        )

        cursor = self.conn.cursor()
        results = {}

        try:
            cursor.execute(query, hashes)
            rows = cursor.fetchall()

            for hash_key, json_str in rows:
                original_text = item_map.get(hash_key)
                if original_text:
                    try:
                        results[original_text] = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Corrupt cache entry: skip it and treat as cache miss.
                        # This allows the system to recover by recomputing the result.
                        logger.warning(f"Corrupt JSON in cache for hash {hash_key}")
                        continue
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return {}

        return results

    def set_batch(
        self, items: List[str], instruction: str, results: Dict[str, Any]
    ) -> None:
        """
        Save results to the cache.

        Args:
            items: List of raw input strings that correspond to the results.
            instruction: The instruction used.
            results: Dictionary mapping original_text -> result_dict to save.
        """
        if not items or not results:
            return

        data_to_insert = []
        for item in items:
            if item in results:
                result_data = results[item]
                # Only cache valid results to avoid storing None values that would
                # prevent retry attempts on failed extractions.
                if result_data is not None:
                    hash_key = self._hash(item, instruction)
                    json_str = json.dumps(result_data)
                    data_to_insert.append((hash_key, json_str))

        if not data_to_insert:
            return

        cursor = self.conn.cursor()
        try:
            insert_query = (
                "INSERT OR IGNORE INTO inference_cache "
                "(hash_key, json_response) VALUES (?, ?)"
            )
            cursor.executemany(insert_query, data_to_insert)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def get_code(self, key: str) -> Optional[str]:
        """Retrieve cached source code by hash key.

        Args:
            key: SHA256 hash key.

        Returns:
            Source code string if found, ``None`` on miss.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT source_code FROM code_cache WHERE hash_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute(
                "UPDATE code_cache SET last_access = CURRENT_TIMESTAMP "
                "WHERE hash_key = ?",
                (key,),
            )
            return row[0]  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error reading code cache: {e}")
            return None

    def set_code(self, key: str, source: str) -> None:
        """Store source code in the cache.

        Args:
            key: SHA256 hash key.
            source: Python source code string.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO code_cache "
                "(hash_key, source_code) VALUES (?, ?)",
                (key, source),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error writing code cache: {e}")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "LocleanCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()
