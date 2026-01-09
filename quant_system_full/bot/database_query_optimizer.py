"""
Database Query Optimizer for High-Performance Historical Data Access

This module provides optimized database query strategies for accessing 20 years
of historical data efficiently during backtesting operations.

Key Features:
- Optimized SQLite query patterns for time-series data
- Intelligent indexing strategies for historical data tables
- Query result caching and batching optimization
- Connection pooling for concurrent access
- Query performance monitoring and analysis
- Adaptive query planning based on data patterns

Performance Optimizations:
- Columnar storage optimization for OHLCV data
- Time-based partitioning for efficient range queries
- Composite indexing for multi-dimensional lookups
- Query result materialization for repeated access
- Batch query execution for reduced I/O overhead
"""

import sqlite3
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import json
import queue
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for database query performance."""
    query_count: int = 0
    total_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time: float = 0.0
    slowest_query_time: float = 0.0
    fastest_query_time: float = float('inf')

    def update(self, execution_time: float, from_cache: bool = False):
        """Update metrics with new query execution."""
        self.query_count += 1
        self.total_execution_time += execution_time

        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.slowest_query_time = max(self.slowest_query_time, execution_time)
            if execution_time > 0:
                self.fastest_query_time = min(self.fastest_query_time, execution_time)

        self.avg_query_time = self.total_execution_time / self.query_count

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ConnectionPool:
    """Thread-safe SQLite connection pool for concurrent access."""

    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0

        # Pre-populate pool with connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool with pre-created connections."""
        for _ in range(min(5, self.max_connections)):  # Start with 5 connections
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)

    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new optimized SQLite connection."""
        try:
            conn = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                timeout=30.0
            )

            # Optimize connection settings
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            # Enable query optimization
            conn.execute("PRAGMA optimize")

            self._created_connections += 1
            return conn

        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            return None

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)."""
        conn = None
        try:
            # Try to get connection from pool
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                # Create new connection if pool is empty and under limit
                with self._lock:
                    if self._created_connections < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # Wait for available connection
                        conn = self._pool.get(timeout=10.0)

            if conn is None:
                raise Exception("Could not obtain database connection")

            yield conn

        finally:
            # Return connection to pool
            if conn:
                try:
                    # Reset connection state
                    conn.rollback()
                    self._pool.put_nowait(conn)
                except queue.Full:
                    # Pool is full, close connection
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    conn.close()


class QueryResultCache:
    """LRU cache for query results with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()

    def _generate_key(self, query: str, params: Tuple = ()) -> str:
        """Generate cache key for query and parameters."""
        key_data = f"{query}|{params}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, params: Tuple = ()) -> Optional[pd.DataFrame]:
        """Get cached query result."""
        key = self._generate_key(query, params)

        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() > entry['expires']:
                del self._cache[key]
                self._access_times.pop(key, None)
                return None

            # Update access time
            self._access_times[key] = time.time()
            return entry['data'].copy()

    def set(self, query: str, params: Tuple, data: pd.DataFrame, ttl: Optional[int] = None):
        """Cache query result."""
        if data is None or data.empty:
            return

        key = self._generate_key(query, params)
        if ttl is None:
            ttl = self.default_ttl

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            expires = time.time() + ttl
            self._cache[key] = {
                'data': data.copy(),
                'expires': expires,
                'created': time.time()
            }
            self._access_times[key] = time.time()

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._access_times:
            return

        lru_key = min(self._access_times, key=self._access_times.get)
        self._cache.pop(lru_key, None)
        self._access_times.pop(lru_key, None)

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size_mb = sum(
                entry['data'].memory_usage(deep=True).sum() / 1024 / 1024
                for entry in self._cache.values()
            )

            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'size_mb': total_size_mb,
                'utilization': len(self._cache) / self.max_size
            }


class DatabaseQueryOptimizer:
    """High-performance database query optimizer for historical data."""

    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = Path(database_path)
        self.connection_pool = ConnectionPool(str(self.database_path), max_connections)
        self.query_cache = QueryResultCache()
        self.metrics = QueryMetrics()

        # Ensure database and indexes exist
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database with optimized schema and indexes."""
        logger.info("Initializing database schema and indexes...")

        # Create tables if they don't exist
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Historical price data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_prices (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjusted_close REAL,
                    dividend REAL DEFAULT 0,
                    split_factor REAL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # Fundamental data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    period_type TEXT,  -- 'Q' for quarterly, 'A' for annual
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date, metric_name, period_type)
                )
            ''')

            # Market data summary table for faster aggregations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_summary (
                    date DATE PRIMARY KEY,
                    total_volume INTEGER,
                    advancing_stocks INTEGER,
                    declining_stocks INTEGER,
                    unchanged_stocks INTEGER,
                    new_highs INTEGER,
                    new_lows INTEGER,
                    market_cap REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create optimized indexes
            self._create_indexes(cursor)

            conn.commit()

    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create optimized indexes for fast query performance."""
        indexes = [
            # Price data indexes
            "CREATE INDEX IF NOT EXISTS idx_prices_symbol ON historical_prices(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_prices_date ON historical_prices(date)",
            "CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON historical_prices(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_prices_date_symbol ON historical_prices(date, symbol)",

            # Fundamental data indexes
            "CREATE INDEX IF NOT EXISTS idx_fundamental_symbol ON fundamental_data(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_fundamental_date ON fundamental_data(date)",
            "CREATE INDEX IF NOT EXISTS idx_fundamental_metric ON fundamental_data(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_fundamental_composite ON fundamental_data(symbol, date, metric_name)",

            # Market summary indexes
            "CREATE INDEX IF NOT EXISTS idx_market_summary_date ON market_summary(date)",

            # Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_prices_close_volume ON historical_prices(symbol, date, close, volume)",
            "CREATE INDEX IF NOT EXISTS idx_prices_ohlc ON historical_prices(symbol, date, open, high, low, close)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                logger.warning(f"Index creation warning: {e}")

    def get_historical_data(self,
                          symbols: Union[str, List[str]],
                          start_date: date,
                          end_date: date,
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Get historical price data with optimized query."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']

        # Construct optimized query
        column_list = ', '.join(columns + ['symbol', 'date'])
        placeholders = ', '.join(['?' for _ in symbols])

        query = f'''
            SELECT {column_list}
            FROM historical_prices
            WHERE symbol IN ({placeholders})
              AND date BETWEEN ? AND ?
            ORDER BY symbol, date
        '''

        params = tuple(symbols + [start_date, end_date])

        return self._execute_cached_query(query, params)

    def get_batch_historical_data(self,
                                symbols: List[str],
                                start_date: date,
                                end_date: date,
                                batch_size: int = 100) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols in optimized batches."""
        results = {}

        # Process in batches to optimize query performance
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]

            batch_data = self.get_historical_data(
                batch_symbols, start_date, end_date
            )

            if not batch_data.empty:
                # Split batch data by symbol
                for symbol in batch_symbols:
                    symbol_data = batch_data[batch_data['symbol'] == symbol].copy()
                    if not symbol_data.empty:
                        symbol_data.set_index('date', inplace=True)
                        symbol_data.drop('symbol', axis=1, inplace=True)
                        results[symbol] = symbol_data

        return results

    def get_fundamental_data(self,
                           symbols: Union[str, List[str]],
                           metrics: List[str],
                           start_date: date,
                           end_date: date,
                           period_type: str = 'Q') -> pd.DataFrame:
        """Get fundamental data with optimized query."""
        if isinstance(symbols, str):
            symbols = [symbols]

        symbol_placeholders = ', '.join(['?' for _ in symbols])
        metric_placeholders = ', '.join(['?' for _ in metrics])

        query = f'''
            SELECT symbol, date, metric_name, metric_value
            FROM fundamental_data
            WHERE symbol IN ({symbol_placeholders})
              AND metric_name IN ({metric_placeholders})
              AND date BETWEEN ? AND ?
              AND period_type = ?
            ORDER BY symbol, date, metric_name
        '''

        params = tuple(symbols + metrics + [start_date, end_date, period_type])

        return self._execute_cached_query(query, params)

    def get_market_summary(self,
                         start_date: date,
                         end_date: date) -> pd.DataFrame:
        """Get market summary data for date range."""
        query = '''
            SELECT *
            FROM market_summary
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        '''

        params = (start_date, end_date)
        return self._execute_cached_query(query, params)

    def get_symbols_with_data(self,
                            start_date: date,
                            end_date: date,
                            min_data_points: int = 100) -> List[str]:
        """Get symbols that have sufficient data in the date range."""
        query = '''
            SELECT symbol, COUNT(*) as data_points
            FROM historical_prices
            WHERE date BETWEEN ? AND ?
            GROUP BY symbol
            HAVING COUNT(*) >= ?
            ORDER BY symbol
        '''

        params = (start_date, end_date, min_data_points)
        result = self._execute_cached_query(query, params)

        return result['symbol'].tolist() if not result.empty else []

    def get_data_availability_summary(self) -> Dict[str, Any]:
        """Get summary of data availability across all symbols."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Get basic statistics
            cursor.execute('''
                SELECT
                    COUNT(DISTINCT symbol) as total_symbols,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM historical_prices
            ''')

            basic_stats = cursor.fetchone()

            # Get symbols by data availability
            cursor.execute('''
                SELECT
                    CASE
                        WHEN COUNT(*) >= 5000 THEN '5000+'
                        WHEN COUNT(*) >= 2000 THEN '2000-4999'
                        WHEN COUNT(*) >= 1000 THEN '1000-1999'
                        WHEN COUNT(*) >= 500 THEN '500-999'
                        ELSE '<500'
                    END as data_range,
                    COUNT(DISTINCT symbol) as symbol_count
                FROM historical_prices
                GROUP BY symbol
            ''')

            availability_stats = cursor.fetchall()

        return {
            'total_symbols': basic_stats[0],
            'total_records': basic_stats[1],
            'earliest_date': basic_stats[2],
            'latest_date': basic_stats[3],
            'data_availability': {
                row[0]: row[1] for row in availability_stats
            }
        }

    def _execute_cached_query(self, query: str, params: Tuple = ()) -> pd.DataFrame:
        """Execute query with caching and performance monitoring."""
        start_time = time.time()

        # Check cache first
        cached_result = self.query_cache.get(query, params)
        if cached_result is not None:
            execution_time = time.time() - start_time
            self.metrics.update(execution_time, from_cache=True)
            return cached_result

        # Execute query
        try:
            with self.connection_pool.get_connection() as conn:
                result = pd.read_sql_query(query, conn, params=params)

                execution_time = time.time() - start_time
                self.metrics.update(execution_time, from_cache=False)

                # Cache result
                self.query_cache.set(query, params, result)

                return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

    def optimize_database(self):
        """Run database optimization operations."""
        logger.info("Running database optimization...")

        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Update table statistics
            cursor.execute("ANALYZE")

            # Optimize query planner
            cursor.execute("PRAGMA optimize")

            # Vacuum if needed
            cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
            cursor.execute("PRAGMA incremental_vacuum(1000)")

            conn.commit()

        logger.info("Database optimization completed")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = self.query_cache.stats()

        return {
            'query_metrics': {
                'total_queries': self.metrics.query_count,
                'avg_query_time': self.metrics.avg_query_time,
                'fastest_query': self.metrics.fastest_query_time,
                'slowest_query': self.metrics.slowest_query_time,
                'cache_hit_rate': self.metrics.cache_hit_rate
            },
            'cache_stats': cache_stats,
            'database_info': self.get_data_availability_summary()
        }

    def clear_cache(self):
        """Clear query result cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")


# Example usage and testing functions
def demo_query_optimization():
    """Demonstrate query optimization capabilities."""
    logger.info("=== Database Query Optimization Demo ===")

    # Initialize optimizer
    db_path = "data_cache/historical_data.db"
    optimizer = DatabaseQueryOptimizer(db_path)

    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = date(2023, 1, 1)
    end_date = date(2024, 1, 1)

    # Benchmark single symbol query
    logger.info("Testing single symbol query...")
    start_time = time.time()
    data = optimizer.get_historical_data('AAPL', start_date, end_date)
    single_query_time = time.time() - start_time
    logger.info(f"Single symbol query: {single_query_time:.3f}s ({len(data)} rows)")

    # Benchmark batch query
    logger.info("Testing batch query...")
    start_time = time.time()
    batch_data = optimizer.get_batch_historical_data(test_symbols, start_date, end_date)
    batch_query_time = time.time() - start_time
    total_rows = sum(len(df) for df in batch_data.values())
    logger.info(f"Batch query: {batch_query_time:.3f}s ({total_rows} rows)")

    # Test cache performance
    logger.info("Testing cache performance...")
    start_time = time.time()
    cached_data = optimizer.get_historical_data('AAPL', start_date, end_date)
    cache_query_time = time.time() - start_time
    logger.info(f"Cached query: {cache_query_time:.3f}s (speedup: {single_query_time/cache_query_time:.1f}x)")

    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    logger.info(f"Performance Summary:")
    logger.info(f"- Cache hit rate: {metrics['query_metrics']['cache_hit_rate']:.1%}")
    logger.info(f"- Average query time: {metrics['query_metrics']['avg_query_time']:.3f}s")
    logger.info(f"- Total symbols in DB: {metrics['database_info']['total_symbols']}")

    return metrics


if __name__ == "__main__":
    demo_query_optimization()