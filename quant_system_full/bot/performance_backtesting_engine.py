"""
High-Performance Backtesting Engine for Three-Phase Validation System

This module provides optimized performance infrastructure for handling 20 years
of historical data across 4000+ stocks with efficient parallel processing,
intelligent caching, and memory management.

Key Features:
- Multi-threaded parallel processing for stock universe backtesting
- Intelligent data caching with Parquet-based storage optimization
- Memory-efficient chunked processing for large datasets
- Query optimization for historical data retrieval
- Progress monitoring and performance benchmarking
- Resource utilization tracking and bottleneck detection
- Scalable processing pipeline for walk-forward validation

Performance Targets:
- Process 4000 stocks over 20 years in under 2 hours
- Memory usage under 16GB for large-scale operations
- Cache hit rates above 85% for repeated calculations
- Parallel efficiency above 70% with multi-core utilization
"""

import os
import sys
import time
import json
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import hashlib
import pickle
from functools import wraps, lru_cache
import gc
import warnings

# Data processing
import pandas as pd
import numpy as np
import sqlite3
from scipy import stats

# Performance monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not available - limited performance monitoring")

# High-performance storage
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    warnings.warn("pyarrow not available - using pandas for I/O")

# Import existing system components
from bot.performance_optimizer import PerformanceOptimizer, get_optimizer
from bot.config import SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting performance parameters."""
    # Processing configuration
    max_workers: int = min(mp.cpu_count(), 16)
    chunk_size: int = 100  # Stocks per chunk
    batch_size: int = 50   # Batch size for data loading
    memory_limit_gb: float = 12.0

    # Cache configuration
    enable_data_cache: bool = True
    cache_dir: str = "data_cache"
    cache_compression: str = "snappy"  # or "gzip", "lz4"
    max_cache_size_gb: float = 5.0

    # Performance thresholds
    target_cache_hit_rate: float = 0.85
    max_memory_usage_percent: float = 80.0
    parallel_efficiency_threshold: float = 0.70

    # Progress monitoring
    enable_progress_monitoring: bool = True
    checkpoint_interval: int = 300  # seconds
    log_level: str = "INFO"


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting operations."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Processing metrics
    total_stocks_processed: int = 0
    total_time_periods: int = 0
    processing_rate_stocks_per_second: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_efficiency: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Parallel processing metrics
    parallel_efficiency: float = 0.0
    thread_utilization: float = 0.0

    # I/O metrics
    data_loading_time: float = 0.0
    data_processing_time: float = 0.0
    io_efficiency: float = 0.0

    def calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        if self.end_time and self.start_time:
            total_time = (self.end_time - self.start_time).total_seconds()
            if total_time > 0:
                self.processing_rate_stocks_per_second = self.total_stocks_processed / total_time

        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations > 0:
            self.cache_hit_rate = self.cache_hits / total_cache_operations

        if self.data_loading_time + self.data_processing_time > 0:
            self.io_efficiency = self.data_processing_time / (
                self.data_loading_time + self.data_processing_time
            )


class HighPerformanceDataCache:
    """High-performance data cache optimized for backtesting workloads."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self._memory_cache = {}
        self._cache_metadata = {}
        self._lock = threading.RLock()
        self._optimizer = get_optimizer()

        # Initialize cache statistics
        self.hits = 0
        self.misses = 0
        self._last_cleanup = time.time()

        logger.info(f"Initialized data cache at {self.cache_dir}")

    def _generate_cache_key(self, symbol: str, start_date: date, end_date: date,
                          data_type: str = "price") -> str:
        """Generate deterministic cache key for data request."""
        key_parts = [symbol, start_date.isoformat(), end_date.isoformat(), data_type]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def get(self, symbol: str, start_date: date, end_date: date,
            data_type: str = "price") -> Optional[pd.DataFrame]:
        """Retrieve data from cache."""
        cache_key = self._generate_cache_key(symbol, start_date, end_date, data_type)

        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                self.hits += 1
                return self._memory_cache[cache_key].copy()

            # Check disk cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    if HAS_ARROW:
                        data = pq.read_table(cache_path).to_pandas()
                    else:
                        data = pd.read_parquet(cache_path)

                    # Store in memory cache for future access
                    self._memory_cache[cache_key] = data
                    self._cache_metadata[cache_key] = {
                        'last_access': time.time(),
                        'size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                    }

                    self.hits += 1
                    return data.copy()
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_key}: {e}")
                    cache_path.unlink(missing_ok=True)

        self.misses += 1
        return None

    def set(self, symbol: str, start_date: date, end_date: date,
            data: pd.DataFrame, data_type: str = "price") -> None:
        """Store data in cache."""
        if data is None or data.empty:
            return

        cache_key = self._generate_cache_key(symbol, start_date, end_date, data_type)

        with self._lock:
            try:
                # Store in memory cache
                self._memory_cache[cache_key] = data.copy()
                self._cache_metadata[cache_key] = {
                    'last_access': time.time(),
                    'size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                }

                # Store in disk cache
                cache_path = self._get_cache_path(cache_key)
                if HAS_ARROW:
                    table = pa.Table.from_pandas(data)
                    pq.write_table(
                        table, cache_path,
                        compression=self.config.cache_compression
                    )
                else:
                    data.to_parquet(
                        cache_path,
                        compression=self.config.cache_compression
                    )

                # Cleanup if needed
                self._cleanup_if_needed()

            except Exception as e:
                logger.error(f"Failed to cache data for {cache_key}: {e}")

    def _cleanup_if_needed(self):
        """Cleanup cache if size or age limits exceeded."""
        current_time = time.time()

        # Cleanup every 5 minutes
        if current_time - self._last_cleanup < 300:
            return

        self._last_cleanup = current_time

        try:
            # Calculate total cache size
            total_size_mb = sum(
                meta['size_mb'] for meta in self._cache_metadata.values()
            )

            # Remove old items if cache is too large
            if total_size_mb > self.config.max_cache_size_gb * 1024:
                # Sort by last access time
                items_by_age = sorted(
                    self._cache_metadata.items(),
                    key=lambda x: x[1]['last_access']
                )

                # Remove oldest 25% of items
                items_to_remove = items_by_age[:len(items_by_age) // 4]

                for cache_key, _ in items_to_remove:
                    self._memory_cache.pop(cache_key, None)
                    self._cache_metadata.pop(cache_key, None)

                    cache_path = self._get_cache_path(cache_key)
                    cache_path.unlink(missing_ok=True)

                logger.info(f"Cleaned up {len(items_to_remove)} cache entries")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_operations = self.hits + self.misses
        hit_rate = self.hits / total_operations if total_operations > 0 else 0.0

        total_size_mb = sum(
            meta['size_mb'] for meta in self._cache_metadata.values()
        )

        return {
            'hit_rate': hit_rate,
            'total_hits': self.hits,
            'total_misses': self.misses,
            'cache_size_mb': total_size_mb,
            'cache_entries': len(self._memory_cache),
            'cache_efficiency': hit_rate >= self.config.target_cache_hit_rate
        }


class ParallelBacktestExecutor:
    """High-performance parallel execution engine for backtesting."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cache = HighPerformanceDataCache(config)
        self.metrics = PerformanceMetrics()
        self._stop_monitoring = threading.Event()

        # Start performance monitoring
        if config.enable_progress_monitoring:
            self._start_monitoring()

    def _start_monitoring(self):
        """Start background performance monitoring."""
        def monitor():
            while not self._stop_monitoring.wait(10):  # Check every 10 seconds
                if HAS_PSUTIL:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024

                    self.metrics.peak_memory_mb = max(
                        self.metrics.peak_memory_mb, memory_mb
                    )

                    # Update average memory (simple moving average)
                    if self.metrics.avg_memory_mb == 0:
                        self.metrics.avg_memory_mb = memory_mb
                    else:
                        self.metrics.avg_memory_mb = (
                            0.9 * self.metrics.avg_memory_mb + 0.1 * memory_mb
                        )

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def process_stock_chunk(self,
                          stock_symbols: List[str],
                          backtest_func: Callable,
                          start_date: date,
                          end_date: date,
                          **kwargs) -> List[Dict[str, Any]]:
        """Process a chunk of stocks in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks for each stock
            future_to_symbol = {}

            for symbol in stock_symbols:
                future = executor.submit(
                    self._process_single_stock,
                    symbol, backtest_func, start_date, end_date, **kwargs
                )
                future_to_symbol[future] = symbol

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        self.metrics.total_stocks_processed += 1
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        return results

    def _process_single_stock(self,
                            symbol: str,
                            backtest_func: Callable,
                            start_date: date,
                            end_date: date,
                            **kwargs) -> Optional[Dict[str, Any]]:
        """Process a single stock with caching and error handling."""
        try:
            # Check cache first
            data = self.cache.get(symbol, start_date, end_date)

            if data is None:
                # Load data and cache it
                data = self._load_stock_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    self.cache.set(symbol, start_date, end_date, data)
                else:
                    return None

            # Run backtest function
            result = backtest_func(symbol, data, **kwargs)

            if result is not None:
                result['symbol'] = symbol
                result['data_points'] = len(data)

            return result

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            return None

    def _load_stock_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Load stock data from data source."""
        try:
            # Import here to avoid circular imports
            from bot.data import fetch_history

            data = fetch_history(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if data is not None and not data.empty:
                # Ensure datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)

                return data

        except Exception as e:
            logger.warning(f"Failed to load data for {symbol}: {e}")

        return None

    def execute_parallel_backtest(self,
                                stock_universe: List[str],
                                backtest_func: Callable,
                                start_date: date,
                                end_date: date,
                                **kwargs) -> Dict[str, Any]:
        """Execute parallel backtesting across stock universe."""
        logger.info(f"Starting parallel backtest for {len(stock_universe)} stocks")

        self.metrics.start_time = datetime.now()
        all_results = []

        try:
            # Split stocks into chunks
            chunks = [
                stock_universe[i:i + self.config.chunk_size]
                for i in range(0, len(stock_universe), self.config.chunk_size)
            ]

            logger.info(f"Processing {len(chunks)} chunks of ~{self.config.chunk_size} stocks each")

            # Process chunks
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} stocks)")

                chunk_start_time = time.time()
                chunk_results = self.process_stock_chunk(
                    chunk, backtest_func, start_date, end_date, **kwargs
                )
                chunk_time = time.time() - chunk_start_time

                all_results.extend(chunk_results)

                # Log progress - Fix: Calculate processed stocks correctly
                processed_stocks = sum(len(chunk) for chunk in chunks[:i+1])
                total_stocks = len(stock_universe)
                progress = processed_stocks / total_stocks * 100

                logger.info(
                    f"Chunk {i+1} completed in {chunk_time:.1f}s "
                    f"({len(chunk_results)}/{len(chunk)} successful). "
                    f"Overall progress: {progress:.1f}%"
                )

                # Garbage collection after each chunk
                gc.collect()

        except Exception as e:
            logger.error(f"Parallel backtest failed: {e}")
            raise

        finally:
            self.metrics.end_time = datetime.now()
            self.metrics.calculate_derived_metrics()
            self._stop_monitoring.set()

        return {
            'results': all_results,
            'metrics': asdict(self.metrics),
            'cache_stats': self.cache.get_statistics(),
            'success_rate': len(all_results) / len(stock_universe) if stock_universe else 0
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_stats = self.cache.get_statistics()

        # Calculate efficiency metrics
        memory_efficiency = 1.0
        if HAS_PSUTIL and self.metrics.peak_memory_mb > 0:
            available_memory_mb = psutil.virtual_memory().total / 1024 / 1024
            memory_efficiency = min(1.0, available_memory_mb / self.metrics.peak_memory_mb)

        return {
            'performance_metrics': asdict(self.metrics),
            'cache_performance': cache_stats,
            'resource_efficiency': {
                'memory_efficiency': memory_efficiency,
                'cache_efficiency': cache_stats['cache_efficiency'],
                'parallel_efficiency': self.metrics.parallel_efficiency
            },
            'recommendations': self._generate_recommendations(cache_stats)
        }

    def _generate_recommendations(self, cache_stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if cache_stats['hit_rate'] < self.config.target_cache_hit_rate:
            recommendations.append(
                f"Cache hit rate ({cache_stats['hit_rate']:.1%}) below target "
                f"({self.config.target_cache_hit_rate:.1%}). Consider increasing cache size."
            )

        if self.metrics.peak_memory_mb > self.config.memory_limit_gb * 1024:
            recommendations.append(
                f"Peak memory usage ({self.metrics.peak_memory_mb:.1f}MB) exceeded limit "
                f"({self.config.memory_limit_gb * 1024:.1f}MB). Consider reducing chunk size."
            )

        if self.metrics.parallel_efficiency < self.config.parallel_efficiency_threshold:
            recommendations.append(
                f"Parallel efficiency ({self.metrics.parallel_efficiency:.1%}) below threshold. "
                "Consider optimizing I/O operations or reducing thread contention."
            )

        return recommendations


class BacktestBenchmarkSuite:
    """Benchmarking suite for backtesting performance validation."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = {}

    def benchmark_data_loading(self,
                             symbols: List[str],
                             date_range: Tuple[date, date]) -> Dict[str, float]:
        """Benchmark data loading performance."""
        executor = ParallelBacktestExecutor(self.config)
        start_date, end_date = date_range

        start_time = time.time()

        # Test sequential loading
        sequential_times = []
        for symbol in symbols[:10]:  # Test first 10 symbols
            symbol_start = time.time()
            data = executor._load_stock_data(symbol, start_date, end_date)
            sequential_times.append(time.time() - symbol_start)

        avg_sequential_time = np.mean(sequential_times)

        # Test parallel loading
        parallel_start = time.time()
        results = executor.process_stock_chunk(
            symbols[:10],
            lambda s, d: {'success': True},
            start_date,
            end_date
        )
        parallel_time = time.time() - parallel_start

        return {
            'sequential_avg_per_stock': avg_sequential_time,
            'parallel_total_10_stocks': parallel_time,
            'parallel_speedup': (avg_sequential_time * 10) / parallel_time if parallel_time > 0 else 0,
            'cache_hit_rate': executor.cache.get_statistics()['hit_rate']
        }

    def benchmark_memory_usage(self,
                             data_size_mb: int = 100) -> Dict[str, float]:
        """Benchmark memory usage patterns."""
        if not HAS_PSUTIL:
            return {'error': 'psutil not available for memory benchmarking'}

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create test data
        rows = int(data_size_mb * 1024 * 1024 / (8 * 6))  # Rough estimate for DataFrame
        test_data = pd.DataFrame({
            'open': np.random.randn(rows),
            'high': np.random.randn(rows),
            'low': np.random.randn(rows),
            'close': np.random.randn(rows),
            'volume': np.random.randint(1000, 10000, rows)
        })

        peak_memory = process.memory_info().rss / 1024 / 1024

        # Test cache storage
        cache = HighPerformanceDataCache(self.config)
        cache.set('TEST', date.today(), date.today(), test_data)

        cache_memory = process.memory_info().rss / 1024 / 1024

        # Cleanup
        del test_data
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024

        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'cache_memory_mb': cache_memory,
            'final_memory_mb': final_memory,
            'memory_efficiency': (peak_memory - initial_memory) / data_size_mb if data_size_mb > 0 else 0
        }

    def run_comprehensive_benchmark(self,
                                  test_symbols: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        if test_symbols is None:
            test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 10  # 50 symbols

        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'system_info': {}
        }

        # System information
        if HAS_PSUTIL:
            benchmark_results['system_info'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'unknown'
            }

        # Data loading benchmark
        logger.info("Running data loading benchmark...")
        date_range = (date(2023, 1, 1), date(2024, 1, 1))
        benchmark_results['data_loading'] = self.benchmark_data_loading(
            test_symbols, date_range
        )

        # Memory usage benchmark
        logger.info("Running memory usage benchmark...")
        benchmark_results['memory_usage'] = self.benchmark_memory_usage(100)

        # Cache performance benchmark
        logger.info("Running cache performance benchmark...")
        executor = ParallelBacktestExecutor(self.config)

        # Warm up cache
        for symbol in test_symbols[:5]:
            executor._load_stock_data(symbol, date_range[0], date_range[1])

        benchmark_results['cache_performance'] = executor.cache.get_statistics()

        return benchmark_results


def create_optimized_config(target_memory_gb: float = 12.0,
                          target_parallel_workers: int = None) -> BacktestConfig:
    """Create optimized configuration based on system resources."""
    if target_parallel_workers is None:
        target_parallel_workers = min(mp.cpu_count(), 16)

    # Adjust chunk size based on available memory
    base_chunk_size = 100
    if target_memory_gb >= 16:
        chunk_size = 200
    elif target_memory_gb >= 8:
        chunk_size = 100
    else:
        chunk_size = 50

    return BacktestConfig(
        max_workers=target_parallel_workers,
        chunk_size=chunk_size,
        memory_limit_gb=target_memory_gb,
        max_cache_size_gb=min(target_memory_gb * 0.3, 5.0),
        enable_progress_monitoring=True,
        cache_compression="snappy" if HAS_ARROW else "gzip"
    )


# Demo and testing functions
def demo_performance_optimization():
    """Demonstrate performance optimization capabilities."""
    logger.info("=== Performance Optimization Demo ===")

    # Create optimized configuration
    config = create_optimized_config(target_memory_gb=8.0)
    logger.info(f"Configuration: {asdict(config)}")

    # Initialize executor
    executor = ParallelBacktestExecutor(config)

    # Demo backtest function
    def simple_backtest(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple backtest for demonstration."""
        if data.empty:
            return None

        # Calculate simple returns
        returns = data['close'].pct_change().dropna()

        return {
            'total_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1),
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min()
        }

    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    # Run parallel backtest
    start_date = date(2023, 1, 1)
    end_date = date(2024, 1, 1)

    logger.info(f"Running backtest for {len(test_symbols)} symbols...")

    results = executor.execute_parallel_backtest(
        test_symbols, simple_backtest, start_date, end_date
    )

    # Display results
    logger.info(f"Backtest completed:")
    logger.info(f"- Processed: {len(results['results'])}/{len(test_symbols)} stocks")
    logger.info(f"- Success rate: {results['success_rate']:.1%}")
    logger.info(f"- Processing rate: {results['metrics']['processing_rate_stocks_per_second']:.2f} stocks/sec")
    logger.info(f"- Cache hit rate: {results['cache_stats']['hit_rate']:.1%}")
    logger.info(f"- Peak memory: {results['metrics']['peak_memory_mb']:.1f} MB")

    return results


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    logger.info("=== Performance Benchmark Suite ===")

    config = create_optimized_config()
    benchmark_suite = BacktestBenchmarkSuite(config)

    results = benchmark_suite.run_comprehensive_benchmark()

    logger.info("Benchmark Results:")
    logger.info(f"- Data loading speedup: {results['data_loading']['parallel_speedup']:.2f}x")
    logger.info(f"- Memory efficiency: {results['memory_usage']['memory_efficiency']:.2f}")
    logger.info(f"- Cache hit rate: {results['cache_performance']['hit_rate']:.1%}")

    return results


if __name__ == "__main__":
    # Run demonstration
    demo_results = demo_performance_optimization()

    print("\n" + "="*50)

    # Run benchmark
    benchmark_results = run_performance_benchmark()

    # Save results
    output_dir = Path("reports/performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"demo_results_{timestamp}.json", 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)

    with open(output_dir / f"benchmark_results_{timestamp}.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}")