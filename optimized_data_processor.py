#!/usr/bin/env python3
"""
Optimized Data Processing Module - High Performance Data Pipeline
高性能数据处理模块

Performance Optimizations:
- Async I/O for concurrent data fetching
- Batch processing with intelligent chunking
- Memory-efficient streaming for large datasets
- Connection pooling and rate limiting
- Intelligent caching with TTL
- Vectorized data transformations

Target: 60% I/O wait time reduction, 300+ symbols/second processing
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import json
from datetime import datetime, timedelta
import sqlite3
import threading
from pathlib import Path

# Performance enhancement imports
try:
    from performance_optimization_engine import PerformanceCache, MemoryOptimizer
    OPTIMIZATION_ENGINE_AVAILABLE = True
except ImportError:
    OPTIMIZATION_ENGINE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataProcessingConfig:
    """Configuration for optimized data processing"""

    # Concurrency settings
    max_concurrent_requests: int = 20
    max_workers: int = 16
    batch_size: int = 50

    # Connection settings
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Caching settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_size_mb: int = 500

    # Memory optimization
    enable_memory_optimization: bool = True
    enable_streaming: bool = True
    chunk_size: int = 1000

    # Rate limiting
    requests_per_second: float = 100.0
    burst_limit: int = 200

@dataclass
class ProcessingResult:
    """Result of data processing operation"""

    symbol: str
    success: bool
    data: Optional[pd.DataFrame] = None
    error_message: str = ""
    processing_time: float = 0.0
    cache_hit: bool = False
    source: str = "unknown"

class RateLimiter:
    """Async rate limiter for API requests"""

    def __init__(self, rate: float, burst: int):
        self.rate = rate  # requests per second
        self.burst = burst  # max burst size
        self.tokens = burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                # Wait until we can get a token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                return True

class ConnectionPool:
    """Optimized connection pool for HTTP requests"""

    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = None
        self.lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create session"""
        if self.session is None or self.session.closed:
            async with self.lock:
                if self.session is None or self.session.closed:
                    timeout = aiohttp.ClientTimeout(total=30, connect=10)
                    self.session = aiohttp.ClientSession(
                        connector=self.connector,
                        timeout=timeout
                    )
        return self.session

    async def close(self):
        """Close connection pool"""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.connector.close()

class StreamingDataProcessor:
    """Memory-efficient streaming data processor"""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.memory_optimizer = MemoryOptimizer() if OPTIMIZATION_ENGINE_AVAILABLE else None

    async def stream_process_symbols(self,
                                   symbols: List[str],
                                   processor_func: Callable,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Stream process large symbol lists in chunks"""

        results = {}
        total_symbols = len(symbols)
        processed_count = 0

        # Process in chunks to manage memory
        for i in range(0, total_symbols, self.chunk_size):
            chunk = symbols[i:i + self.chunk_size]

            logger.info(f"Processing chunk {i//self.chunk_size + 1}: {len(chunk)} symbols")

            # Process chunk
            chunk_results = await processor_func(chunk)
            results.update(chunk_results)

            processed_count += len(chunk)

            # Progress callback
            if progress_callback:
                progress = processed_count / total_symbols
                await progress_callback(progress, processed_count, total_symbols)

            # Memory optimization
            if self.memory_optimizer and processed_count % (self.chunk_size * 5) == 0:
                import gc
                gc.collect()
                logger.info(f"Memory cleanup: {processed_count} symbols processed")

        return results

class OptimizedDataProcessor:
    """High-performance data processor with advanced optimizations"""

    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()

        # Performance components
        self.cache = PerformanceCache(self.config.max_cache_size_mb) if OPTIMIZATION_ENGINE_AVAILABLE else None
        self.memory_optimizer = MemoryOptimizer() if OPTIMIZATION_ENGINE_AVAILABLE else None
        self.rate_limiter = RateLimiter(self.config.requests_per_second, self.config.burst_limit)
        self.connection_pool = ConnectionPool(self.config.max_concurrent_requests)
        self.streaming_processor = StreamingDataProcessor(self.config.chunk_size)

        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'data_points_processed': 0
        }

        logger.info("Optimized data processor initialized")
        logger.info(f"Max concurrent requests: {self.config.max_concurrent_requests}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Rate limit: {self.config.requests_per_second} req/sec")

    async def fetch_batch_data(self,
                              symbols: List[str],
                              data_source: str = "yahoo",
                              period: str = "1d",
                              enable_cache: bool = True) -> Dict[str, ProcessingResult]:
        """
        Fetch data for multiple symbols with maximum optimization

        Args:
            symbols: List of stock symbols
            data_source: Data source ("yahoo", "tiger", etc.)
            period: Time period for data
            enable_cache: Whether to use caching

        Returns:
            Dictionary of symbol -> ProcessingResult
        """
        start_time = time.time()
        logger.info(f"Starting optimized batch data fetch for {len(symbols)} symbols")

        # Use streaming processing for large datasets
        if len(symbols) > self.config.chunk_size:
            return await self._stream_fetch_data(symbols, data_source, period, enable_cache)

        # Regular batch processing for smaller datasets
        return await self._batch_fetch_data(symbols, data_source, period, enable_cache)

    async def _stream_fetch_data(self,
                               symbols: List[str],
                               data_source: str,
                               period: str,
                               enable_cache: bool) -> Dict[str, ProcessingResult]:
        """Stream processing for large symbol lists"""

        async def process_chunk(chunk_symbols: List[str]) -> Dict[str, ProcessingResult]:
            return await self._batch_fetch_data(chunk_symbols, data_source, period, enable_cache)

        async def progress_callback(progress: float, current: int, total: int):
            logger.info(f"Streaming progress: {progress:.1%} ({current}/{total})")

        results = await self.streaming_processor.stream_process_symbols(
            symbols, process_chunk, progress_callback
        )

        return results

    async def _batch_fetch_data(self,
                              symbols: List[str],
                              data_source: str,
                              period: str,
                              enable_cache: bool) -> Dict[str, ProcessingResult]:
        """Optimized batch data fetching"""

        results = {}

        # Step 1: Check cache for existing data
        if enable_cache and self.cache:
            cached_results, remaining_symbols = await self._check_cache_batch(symbols, data_source, period)
            results.update(cached_results)
            symbols = remaining_symbols

        if not symbols:
            logger.info("All data retrieved from cache")
            return results

        # Step 2: Split into batches for parallel processing
        batches = [symbols[i:i + self.config.batch_size]
                  for i in range(0, len(symbols), self.config.batch_size)]

        # Step 3: Process batches concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def process_batch(batch: List[str]) -> Dict[str, ProcessingResult]:
            async with semaphore:
                return await self._fetch_batch_symbols(batch, data_source, period)

        # Execute all batches concurrently
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Step 4: Combine results
        for batch_result in batch_results:
            if isinstance(batch_result, dict):
                results.update(batch_result)

                # Cache successful results
                if enable_cache and self.cache:
                    await self._cache_batch_results(batch_result, data_source, period)
            else:
                logger.error(f"Batch processing error: {batch_result}")

        # Step 5: Update statistics
        self._update_batch_stats(results)

        processing_time = time.time() - start_time
        throughput = len(results) / processing_time if processing_time > 0 else 0

        logger.info(f"Batch processing completed: {throughput:.1f} symbols/second")
        logger.info(f"Success rate: {self._calculate_success_rate(results):.1%}")

        return results

    async def _fetch_batch_symbols(self,
                                 symbols: List[str],
                                 data_source: str,
                                 period: str) -> Dict[str, ProcessingResult]:
        """Fetch data for a batch of symbols"""

        results = {}

        # Create tasks for parallel symbol processing
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self._fetch_single_symbol(symbol, data_source, period)
            )
            tasks.append((symbol, task))

        # Wait for all tasks with timeout
        for symbol, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=self.config.read_timeout)
                results[symbol] = result
            except asyncio.TimeoutError:
                results[symbol] = ProcessingResult(
                    symbol=symbol,
                    success=False,
                    error_message="Request timeout",
                    source=data_source
                )
            except Exception as e:
                results[symbol] = ProcessingResult(
                    symbol=symbol,
                    success=False,
                    error_message=str(e),
                    source=data_source
                )

        return results

    async def _fetch_single_symbol(self,
                                 symbol: str,
                                 data_source: str,
                                 period: str) -> ProcessingResult:
        """Fetch data for a single symbol with optimizations"""

        start_time = time.time()

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            # Simulate data fetching (replace with actual implementation)
            if data_source == "yahoo":
                data = await self._fetch_yahoo_data(symbol, period)
            elif data_source == "tiger":
                data = await self._fetch_tiger_data(symbol, period)
            else:
                data = await self._fetch_mock_data(symbol, period)

            # Memory optimization
            if data is not None and self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe(data)

            processing_time = time.time() - start_time

            return ProcessingResult(
                symbol=symbol,
                success=True,
                data=data,
                processing_time=processing_time,
                source=data_source
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error fetching {symbol}: {e}")

            return ProcessingResult(
                symbol=symbol,
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                source=data_source
            )

    async def _fetch_yahoo_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance (optimized implementation)"""
        # This would be replaced with actual Yahoo Finance API calls
        # For now, return mock data
        return await self._fetch_mock_data(symbol, period)

    async def _fetch_tiger_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from Tiger API (optimized implementation)"""
        # This would be replaced with actual Tiger API calls
        # For now, return mock data
        return await self._fetch_mock_data(symbol, period)

    async def _fetch_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate mock data for testing"""
        # Simulate network delay
        await asyncio.sleep(0.01)  # 10ms simulated delay

        # Generate realistic OHLCV data
        np.random.seed(hash(symbol) % 2**31)
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')

        base_price = 50 + (hash(symbol) % 1000) / 10
        price_changes = np.random.randn(len(dates)) * 0.02
        prices = base_price * np.exp(np.cumsum(price_changes))

        return pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
            'high': prices * (1 + abs(np.random.randn(len(dates))) * 0.005),
            'low': prices * (1 - abs(np.random.randn(len(dates))) * 0.005),
            'close': prices,
            'volume': np.random.randint(10000, 1000000, len(dates))
        })

    async def _check_cache_batch(self,
                               symbols: List[str],
                               data_source: str,
                               period: str) -> Tuple[Dict[str, ProcessingResult], List[str]]:
        """Check cache for batch of symbols"""

        cached_results = {}
        remaining_symbols = []

        if not self.cache:
            return cached_results, symbols

        for symbol in symbols:
            cache_key = f"{data_source}_{symbol}_{period}"
            cached_data = self.cache.get(cache_key)

            if cached_data:
                cached_results[symbol] = ProcessingResult(
                    symbol=symbol,
                    success=True,
                    data=cached_data,
                    cache_hit=True,
                    source=f"{data_source}_cache"
                )
                self.stats['cache_hits'] += 1
            else:
                remaining_symbols.append(symbol)

        if cached_results:
            logger.info(f"Cache hits: {len(cached_results)}/{len(symbols)} symbols")

        return cached_results, remaining_symbols

    async def _cache_batch_results(self,
                                 results: Dict[str, ProcessingResult],
                                 data_source: str,
                                 period: str):
        """Cache successful batch results"""

        if not self.cache:
            return

        cached_count = 0
        for symbol, result in results.items():
            if result.success and result.data is not None:
                cache_key = f"{data_source}_{symbol}_{period}"
                self.cache.set(cache_key, result.data)
                cached_count += 1

        if cached_count > 0:
            logger.debug(f"Cached {cached_count} successful results")

    def _update_batch_stats(self, results: Dict[str, ProcessingResult]):
        """Update processing statistics"""

        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        total_time = sum(r.processing_time for r in results.values())

        self.stats['total_requests'] += len(results)
        self.stats['successful_requests'] += successful
        self.stats['failed_requests'] += failed
        self.stats['total_processing_time'] += total_time

        # Count data points
        for result in results.values():
            if result.success and result.data is not None:
                self.stats['data_points_processed'] += len(result.data)

    def _calculate_success_rate(self, results: Dict[str, ProcessingResult]) -> float:
        """Calculate success rate for batch"""
        if not results:
            return 0.0

        successful = sum(1 for r in results.values() if r.success)
        return successful / len(results)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""

        avg_processing_time = (self.stats['total_processing_time'] /
                             self.stats['total_requests']
                             if self.stats['total_requests'] > 0 else 0)

        cache_hit_rate = (self.stats['cache_hits'] /
                         self.stats['total_requests']
                         if self.stats['total_requests'] > 0 else 0)

        success_rate = (self.stats['successful_requests'] /
                       self.stats['total_requests']
                       if self.stats['total_requests'] > 0 else 0)

        summary = {
            'statistics': self.stats.copy(),
            'performance_metrics': {
                'average_processing_time_seconds': avg_processing_time,
                'cache_hit_rate': cache_hit_rate,
                'success_rate': success_rate,
                'throughput_symbols_per_second': (
                    self.stats['successful_requests'] /
                    self.stats['total_processing_time']
                    if self.stats['total_processing_time'] > 0 else 0
                )
            },
            'configuration': {
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'batch_size': self.config.batch_size,
                'rate_limit_rps': self.config.requests_per_second,
                'cache_enabled': self.config.enable_caching,
                'streaming_enabled': self.config.enable_streaming
            }
        }

        if self.cache:
            summary['cache_statistics'] = self.cache.get_stats()

        if self.memory_optimizer:
            summary['memory_report'] = self.memory_optimizer.get_memory_report()

        return summary

    async def close(self):
        """Clean up resources"""
        await self.connection_pool.close()
        logger.info("Optimized data processor closed")

# Example usage and testing
async def main():
    """Demonstration of optimized data processing"""

    # Initialize processor
    config = DataProcessingConfig(
        max_concurrent_requests=20,
        batch_size=50,
        requests_per_second=100.0,
        enable_caching=True,
        enable_streaming=True
    )

    processor = OptimizedDataProcessor(config)

    try:
        # Test with different dataset sizes
        test_sizes = [100, 500, 1000, 2000]

        for size in test_sizes:
            symbols = [f"STOCK_{i:04d}" for i in range(size)]

            print(f"\nTesting with {size} symbols...")

            start_time = time.time()
            results = await processor.fetch_batch_data(
                symbols=symbols,
                data_source="yahoo",
                period="1d",
                enable_cache=True
            )
            end_time = time.time()

            processing_time = end_time - start_time
            throughput = len(results) / processing_time
            success_rate = sum(1 for r in results.values() if r.success) / len(results)

            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Throughput: {throughput:.1f} symbols/second")
            print(f"Success rate: {success_rate:.1%}")

            # Test cache performance on second run
            if size <= 500:  # Only test cache for smaller datasets
                print("Testing cache performance (second run)...")
                cache_start = time.time()
                cached_results = await processor.fetch_batch_data(
                    symbols=symbols,
                    data_source="yahoo",
                    period="1d",
                    enable_cache=True
                )
                cache_end = time.time()

                cache_time = cache_end - cache_start
                cache_speedup = processing_time / cache_time if cache_time > 0 else float('inf')
                print(f"Cached processing time: {cache_time:.2f} seconds")
                print(f"Cache speedup: {cache_speedup:.1f}x")

        # Performance summary
        summary = processor.get_performance_summary()
        print(f"\nOverall Performance Summary:")
        print(f"Total requests: {summary['statistics']['total_requests']}")
        print(f"Success rate: {summary['performance_metrics']['success_rate']:.1%}")
        print(f"Cache hit rate: {summary['performance_metrics']['cache_hit_rate']:.1%}")
        print(f"Average throughput: {summary['performance_metrics']['throughput_symbols_per_second']:.1f} symbols/sec")

    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())