"""
Document Cache

SQLite-based caching system for LLM responses and fetched documents.
Reduces API costs and improves performance.
"""

import logging
import sqlite3
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class DocumentCache:
    """
    SQLite-based cache for LLM responses and documents.

    Features:
    - TTL-based expiration (default 7 days)
    - SHA256-based cache keys
    - Automatic cleanup of expired entries
    - Thread-safe operations
    """

    def __init__(self, config):
        """
        Initialize document cache.

        Args:
            config: LLMEnhancementConfig instance
        """
        self.config = config
        self.enabled = config.cache_enabled
        self.ttl_days = config.cache_ttl_days
        self.cache_dir = Path(config.cache_dir)

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

        # Initialize database
        if self.enabled:
            self._init_database()
        else:
            logger.info("[LLM] Cache is disabled")

    def _init_database(self):
        """Initialize SQLite database."""
        try:
            # Create cache directory if needed
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Database path
            self.db_path = self.cache_dir / "llm_cache.db"

            # Create table
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                """)

                # Create index on expires_at for efficient cleanup
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at
                    ON cache(expires_at)
                """)

                conn.commit()

            logger.info(f"[LLM] Cache initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"[LLM] Failed to initialize cache: {e}")
            self.enabled = False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache.

        Args:
            key: Cache key (will be hashed)

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        try:
            with self._lock:
                cache_key = self._hash_key(key)

                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT value, expires_at FROM cache WHERE key = ?",
                        (cache_key,)
                    )
                    row = cursor.fetchone()

                    if not row:
                        logger.debug(f"[LLM] Cache miss: {key[:50]}...")
                        return None

                    value_json, expires_at = row

                    # Check expiration
                    if datetime.fromisoformat(expires_at) < datetime.now():
                        logger.debug(f"[LLM] Cache expired: {key[:50]}...")
                        # Delete expired entry
                        conn.execute("DELETE FROM cache WHERE key = ?", (cache_key,))
                        conn.commit()
                        return None

                    logger.debug(f"[LLM] Cache hit: {key[:50]}...")
                    return json.loads(value_json)

        except Exception as e:
            logger.error(f"[LLM] Error reading from cache: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any], ttl_days: Optional[int] = None):
        """
        Set value in cache.

        Args:
            key: Cache key (will be hashed)
            value: Value to cache (must be JSON-serializable)
            ttl_days: Time-to-live in days (defaults to config.cache_ttl_days)
        """
        if not self.enabled:
            return

        try:
            with self._lock:
                cache_key = self._hash_key(key)
                value_json = json.dumps(value)

                ttl = ttl_days or self.ttl_days
                timestamp = datetime.now()
                expires_at = timestamp + timedelta(days=ttl)

                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache (key, value, timestamp, expires_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (cache_key, value_json, timestamp.isoformat(), expires_at.isoformat())
                    )
                    conn.commit()

                logger.debug(f"[LLM] Cached: {key[:50]}... (TTL: {ttl} days)")

        except Exception as e:
            logger.error(f"[LLM] Error writing to cache: {e}")

    def delete(self, key: str):
        """
        Delete value from cache.

        Args:
            key: Cache key (will be hashed)
        """
        if not self.enabled:
            return

        try:
            with self._lock:
                cache_key = self._hash_key(key)

                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache WHERE key = ?", (cache_key,))
                    conn.commit()

                logger.debug(f"[LLM] Deleted from cache: {key[:50]}...")

        except Exception as e:
            logger.error(f"[LLM] Error deleting from cache: {e}")

    def cleanup_expired(self):
        """Remove all expired entries from cache."""
        if not self.enabled:
            return

        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache WHERE expires_at < ?",
                        (datetime.now().isoformat(),)
                    )
                    deleted = cursor.rowcount
                    conn.commit()

                if deleted > 0:
                    logger.info(f"[LLM] Cleaned up {deleted} expired cache entries")

        except Exception as e:
            logger.error(f"[LLM] Error cleaning up cache: {e}")

    def clear(self):
        """Clear all entries from cache."""
        if not self.enabled:
            return

        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache")
                    conn.commit()

                logger.info("[LLM] Cache cleared")

        except Exception as e:
            logger.error(f"[LLM] Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            dict: {
                "total_entries": int,
                "expired_entries": int,
                "valid_entries": int,
                "cache_size_mb": float
            }
        """
        if not self.enabled:
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "valid_entries": 0,
                "cache_size_mb": 0.0
            }

        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path)) as conn:
                    # Total entries
                    cursor = conn.execute("SELECT COUNT(*) FROM cache")
                    total = cursor.fetchone()[0]

                    # Expired entries
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
                        (datetime.now().isoformat(),)
                    )
                    expired = cursor.fetchone()[0]

                    valid = total - expired

                    # Cache size
                    cache_size_mb = self.db_path.stat().st_size / (1024 * 1024)

                    return {
                        "total_entries": total,
                        "expired_entries": expired,
                        "valid_entries": valid,
                        "cache_size_mb": round(cache_size_mb, 2)
                    }

        except Exception as e:
            logger.error(f"[LLM] Error getting cache stats: {e}")
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "valid_entries": 0,
                "cache_size_mb": 0.0
            }

    def _hash_key(self, key: str) -> str:
        """
        Hash cache key using SHA256.

        Args:
            key: Original key

        Returns:
            Hashed key (hex digest)
        """
        return hashlib.sha256(key.encode()).hexdigest()
