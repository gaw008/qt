"""
Parallel Data Fetcher for High-Performance Stock Data Acquisition

This module implements a high-performance parallel data fetching system optimized
for multi-core CPUs and large memory systems. It provides:

- Multi-process parallel data fetching with configurable worker pools
- Intelligent rate limiting and API management
- Progress monitoring with detailed statistics
- Error handling and retry mechanisms
- Cache integration for optimal performance
- Memory-efficient batch processing

Features:
- Supports 8-16 parallel workers (configurable)
- Batch processing for efficient API usage
- Real-time progress tracking
- Comprehensive error handling
- Performance metrics and statistics
- Integration with existing caching system
"""

import multiprocessing as mp
import concurrent.futures
import time
import logging
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a single symbol fetch operation"""
    symbol: str
    data: Optional[pd.DataFrame]
    success: bool
    error: Optional[str] = None
    fetch_time: float = 0.0
    source: str = "api"  # "api" or "cache"
    retry_count: int = 0


@dataclass
class BatchResult:
    """Result of a batch fetch operation"""
    total_symbols: int
    successful_fetches: int
    failed_fetches: int
    cache_hits: int
    api_calls: int
    total_time: float
    avg_time_per_symbol: float
    results: Dict[str, FetchResult]


@dataclass
class ParallelFetchConfig:
    """Configuration for parallel data fetching"""
    max_workers: int = 12
    batch_size: int = 50
    api_delay: float = 0.1
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_per_symbol: float = 10.0
    rate_limit_per_minute: int = 500
    enable_progress_bar: bool = True
    log_level: str = "INFO"


class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_calls_per_minute: int = 500):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = queue.deque()
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Wait if we're hitting rate limits"""
        with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            
            # Check if we need to wait
            if len(self.calls) >= self.max_calls_per_minute:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = 60 - (now - oldest_call)
                if wait_time > 0:
                    logger.info(f"[RATE_LIMITER] Waiting {wait_time:.1f}s for rate limit")
                    time.sleep(wait_time)
                    # Clean up old calls after waiting
                    now = time.time()
                    while self.calls and now - self.calls[0] > 60:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(now)


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.symbol_times = {}
        self.errors = []
        self.cache_hits = 0
        self.api_calls = 0
        self.lock = threading.Lock()
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        
    def stop(self):
        """Stop monitoring"""
        self.end_time = time.time()
        
    def record_symbol_fetch(self, symbol: str, fetch_time: float, from_cache: bool = False):
        """Record a symbol fetch time"""
        with self.lock:
            self.symbol_times[symbol] = fetch_time
            if from_cache:
                self.cache_hits += 1
            else:
                self.api_calls += 1
                
    def record_error(self, symbol: str, error: str):
        """Record an error"""
        with self.lock:
            self.errors.append({"symbol": symbol, "error": error, "time": time.time()})
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_time = (self.end_time or time.time()) - (self.start_time or time.time())
        total_symbols = len(self.symbol_times)
        
        return {
            "total_time": total_time,
            "total_symbols": total_symbols,
            "successful_symbols": total_symbols - len(self.errors),
            "failed_symbols": len(self.errors),
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "cache_hit_rate": self.cache_hits / max(1, total_symbols) * 100,
            "avg_time_per_symbol": total_time / max(1, total_symbols),
            "symbols_per_minute": total_symbols / max(0.017, total_time / 60),  # Avoid division by zero
            "error_rate": len(self.errors) / max(1, total_symbols) * 100
        }


def fetch_single_symbol_worker(args: Tuple) -> FetchResult:
    """
    Worker function for fetching a single symbol.
    This runs in a separate process.
    """
    symbol, period, limit, config, use_cache = args
    
    start_time = time.time()
    result = FetchResult(symbol=symbol, data=None, success=False)
    
    try:
        # Import required modules (necessary in worker process)
        import sys
        import os
        from pathlib import Path
        
        # Add bot path for imports
        current_dir = Path(__file__).parent
        bot_path = current_dir.parent / 'bot'
        if bot_path.exists():
            sys.path.insert(0, str(bot_path))
        
        # Import Yahoo data functions
        try:
            from yahoo_data import fetch_yahoo_price_history
        except ImportError:
            # Fallback import path
            sys.path.insert(0, str(current_dir.parent.parent / 'bot'))
            from yahoo_data import fetch_yahoo_price_history
            
        # Attempt to fetch data
        for attempt in range(config.max_retries):
            try:
                data = fetch_yahoo_price_history(
                    symbol=symbol,
                    period=period, 
                    limit=limit,
                    max_retries=1,  # Handle retries at this level
                    retry_delay=config.retry_delay,
                    use_cache=use_cache
                )
                
                if data is not None and not data.empty:
                    result.data = data
                    result.success = True
                    result.source = "cache" if use_cache else "api"
                    break
                    
            except Exception as e:
                result.error = str(e)
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay * (attempt + 1))
                    result.retry_count += 1
                    
        result.fetch_time = time.time() - start_time
        
    except Exception as e:
        result.error = f"Worker error: {str(e)}"
        result.fetch_time = time.time() - start_time
        
    return result


class ParallelDataFetcher:
    """
    High-performance parallel data fetcher optimized for multi-core systems.
    """
    
    def __init__(self, config: Optional[ParallelFetchConfig] = None):
        """Initialize the parallel data fetcher"""
        self.config = config or ParallelFetchConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        self.monitor = PerformanceMonitor()
        
        # Optimize worker count based on system resources
        self._optimize_worker_count()
        
        logger.info(f"[PARALLEL_FETCHER] Initialized with {self.config.max_workers} workers, "
                   f"batch_size={self.config.batch_size}, rate_limit={self.config.rate_limit_per_minute}/min")
    
    def _optimize_worker_count(self):
        """Optimize worker count based on system resources"""
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Optimal worker count based on system resources
        if memory_gb >= 32 and cpu_count >= 8:
            # High-end system
            optimal_workers = min(16, cpu_count)
        elif memory_gb >= 16 and cpu_count >= 4:
            # Mid-range system
            optimal_workers = min(12, cpu_count)
        else:
            # Lower-end system
            optimal_workers = min(8, cpu_count)
        
        # Use configured max_workers as upper bound
        self.config.max_workers = min(self.config.max_workers, optimal_workers)
        
        logger.info(f"[PARALLEL_FETCHER] System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM, "
                   f"optimized to {self.config.max_workers} workers")
    
    def fetch_symbols_parallel(
        self, 
        symbols: List[str], 
        period: str = 'day', 
        limit: int = 300,
        use_cache: bool = True
    ) -> BatchResult:
        """
        Fetch data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols to fetch
            period: Time period for data
            limit: Number of data points
            use_cache: Whether to use caching
            
        Returns:
            BatchResult with comprehensive results and statistics
        """
        logger.info(f"[PARALLEL_FETCHER] Starting parallel fetch of {len(symbols)} symbols "
                   f"with {self.config.max_workers} workers")
        
        self.monitor.start()
        results = {}
        
        # Prepare arguments for worker processes
        worker_args = [
            (symbol, period, limit, self.config, use_cache) 
            for symbol in symbols
        ]
        
        # Progress tracking
        progress_bar = None
        if self.config.enable_progress_bar:
            progress_bar = tqdm(
                total=len(symbols), 
                desc="Fetching stock data", 
                unit="stocks",
                ncols=100
            )
        
        successful_fetches = 0
        failed_fetches = 0
        cache_hits = 0
        api_calls = 0
        
        try:
            # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(fetch_single_symbol_worker, args): args[0] 
                    for args in worker_args
                }
                
                # Process completed futures
                for future in concurrent.futures.as_completed(
                    future_to_symbol, 
                    timeout=self.config.timeout_per_symbol * len(symbols)
                ):
                    symbol = future_to_symbol[future]
                    
                    try:
                        result = future.result()
                        results[symbol] = result
                        
                        # Update statistics
                        if result.success:
                            successful_fetches += 1
                            if result.source == "cache":
                                cache_hits += 1
                            else:
                                api_calls += 1
                        else:
                            failed_fetches += 1
                        
                        # Update progress
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix({
                                'success': successful_fetches,
                                'failed': failed_fetches,
                                'cache%': f"{cache_hits/(successful_fetches+failed_fetches)*100:.1f}" if (successful_fetches+failed_fetches) > 0 else "0.0"
                            })
                        
                        # Rate limiting for API calls
                        if result.success and result.source == "api":
                            self.rate_limiter.wait_if_needed()
                    
                    except concurrent.futures.TimeoutError:
                        logger.error(f"[PARALLEL_FETCHER] Timeout fetching {symbol}")
                        results[symbol] = FetchResult(
                            symbol=symbol, 
                            data=None, 
                            success=False, 
                            error="Timeout"
                        )
                        failed_fetches += 1
                        if progress_bar:
                            progress_bar.update(1)
                    
                    except Exception as e:
                        logger.error(f"[PARALLEL_FETCHER] Error fetching {symbol}: {e}")
                        results[symbol] = FetchResult(
                            symbol=symbol, 
                            data=None, 
                            success=False, 
                            error=str(e)
                        )
                        failed_fetches += 1
                        if progress_bar:
                            progress_bar.update(1)
        
        except Exception as e:
            logger.error(f"[PARALLEL_FETCHER] Critical error in parallel execution: {e}")
            # Ensure all symbols have results even if there was a critical error
            for symbol in symbols:
                if symbol not in results:
                    results[symbol] = FetchResult(
                        symbol=symbol,
                        data=None,
                        success=False,
                        error=f"Critical error: {str(e)}"
                    )
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        self.monitor.stop()
        
        # Create batch result
        total_time = self.monitor.end_time - self.monitor.start_time
        batch_result = BatchResult(
            total_symbols=len(symbols),
            successful_fetches=successful_fetches,
            failed_fetches=failed_fetches,
            cache_hits=cache_hits,
            api_calls=api_calls,
            total_time=total_time,
            avg_time_per_symbol=total_time / len(symbols),
            results=results
        )
        
        # Log summary
        stats = self.monitor.get_statistics()
        logger.info(f"[PARALLEL_FETCHER] Completed: {successful_fetches}/{len(symbols)} successful "
                   f"({cache_hits} cache hits, {api_calls} API calls) in {total_time:.1f}s "
                   f"({stats['symbols_per_minute']:.1f} symbols/min)")
        
        if failed_fetches > 0:
            logger.warning(f"[PARALLEL_FETCHER] {failed_fetches} symbols failed")
        
        return batch_result
    
    def fetch_with_progress_callback(
        self,
        symbols: List[str],
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
        period: str = 'day',
        limit: int = 300,
        use_cache: bool = True
    ) -> BatchResult:
        """
        Fetch symbols with custom progress callback.
        
        Args:
            symbols: List of symbols to fetch
            progress_callback: Callback function(completed, total, stats)
            period: Time period
            limit: Data limit
            use_cache: Use cache flag
            
        Returns:
            BatchResult
        """
        if progress_callback is None:
            return self.fetch_symbols_parallel(symbols, period, limit, use_cache)
        
        # Disable built-in progress bar when using callback
        original_progress_setting = self.config.enable_progress_bar
        self.config.enable_progress_bar = False
        
        try:
            logger.info(f"[PARALLEL_FETCHER] Starting parallel fetch with callback")
            
            # For callback version, we'll process in smaller batches to provide regular updates
            batch_size = 20
            all_results = {}
            completed = 0
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i+batch_size]
                batch_result = self.fetch_symbols_parallel(batch_symbols, period, limit, use_cache)
                
                # Merge results
                all_results.update(batch_result.results)
                completed += len(batch_symbols)
                
                # Call progress callback
                stats = {
                    'successful': sum(1 for r in all_results.values() if r.success),
                    'failed': sum(1 for r in all_results.values() if not r.success),
                    'cache_hits': sum(1 for r in all_results.values() if r.success and r.source == "cache")
                }
                
                progress_callback(completed, len(symbols), stats)
            
            # Create final batch result
            successful = sum(1 for r in all_results.values() if r.success)
            failed = len(all_results) - successful
            cache_hits = sum(1 for r in all_results.values() if r.success and r.source == "cache")
            
            return BatchResult(
                total_symbols=len(symbols),
                successful_fetches=successful,
                failed_fetches=failed,
                cache_hits=cache_hits,
                api_calls=successful - cache_hits,
                total_time=0.0,  # Not tracked in callback mode
                avg_time_per_symbol=0.0,
                results=all_results
            )
            
        finally:
            self.config.enable_progress_bar = original_progress_setting
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return self.monitor.get_statistics()
    
    def estimate_fetch_time(self, symbol_count: int, cache_hit_rate: float = 0.5) -> float:
        """
        Estimate time to fetch given number of symbols.
        
        Args:
            symbol_count: Number of symbols to fetch
            cache_hit_rate: Expected cache hit rate (0.0 to 1.0)
            
        Returns:
            Estimated time in seconds
        """
        api_calls = symbol_count * (1 - cache_hit_rate)
        cache_hits = symbol_count * cache_hit_rate
        
        # Estimate based on parallel processing
        time_per_api_call = 2.0  # seconds (including network latency)
        time_per_cache_hit = 0.1  # seconds
        
        # With parallel processing
        parallel_api_time = (api_calls * time_per_api_call) / self.config.max_workers
        parallel_cache_time = (cache_hits * time_per_cache_hit) / self.config.max_workers
        
        # Add overhead for process management
        overhead = 5.0  # seconds
        
        total_time = parallel_api_time + parallel_cache_time + overhead
        
        return total_time


# Convenience functions for easy integration
def create_high_performance_fetcher() -> ParallelDataFetcher:
    """Create a high-performance fetcher optimized for the current system"""
    config = ParallelFetchConfig(
        max_workers=16,  # Will be optimized based on system resources
        batch_size=50,
        api_delay=0.05,  # Aggressive for high-performance systems
        max_retries=2,
        retry_delay=0.5,
        timeout_per_symbol=15.0,
        rate_limit_per_minute=600,
        enable_progress_bar=True
    )
    return ParallelDataFetcher(config)


def create_conservative_fetcher() -> ParallelDataFetcher:
    """Create a conservative fetcher for systems with API rate limits"""
    config = ParallelFetchConfig(
        max_workers=8,
        batch_size=25,
        api_delay=0.2,
        max_retries=3,
        retry_delay=1.0,
        timeout_per_symbol=20.0,
        rate_limit_per_minute=300,
        enable_progress_bar=True
    )
    return ParallelDataFetcher(config)


def quick_parallel_fetch(
    symbols: List[str], 
    period: str = 'day', 
    limit: int = 300,
    use_cache: bool = True,
    high_performance: bool = True
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Quick utility function for parallel fetching.
    
    Args:
        symbols: List of symbols to fetch
        period: Time period
        limit: Data limit
        use_cache: Use cache flag
        high_performance: Use high-performance settings
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if high_performance:
        fetcher = create_high_performance_fetcher()
    else:
        fetcher = create_conservative_fetcher()
    
    batch_result = fetcher.fetch_symbols_parallel(symbols, period, limit, use_cache)
    
    # Convert to simple dictionary format
    return {
        symbol: result.data if result.success else None
        for symbol, result in batch_result.results.items()
    }


if __name__ == "__main__":
    # Test the parallel fetcher
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    print("Testing Parallel Data Fetcher")
    print("="*50)
    
    # Test high-performance fetcher
    fetcher = create_high_performance_fetcher()
    
    # Estimate time
    estimated_time = fetcher.estimate_fetch_time(len(test_symbols), cache_hit_rate=0.3)
    print(f"Estimated time for {len(test_symbols)} symbols: {estimated_time:.1f} seconds")
    
    # Perform fetch
    start_time = time.time()
    batch_result = fetcher.fetch_symbols_parallel(test_symbols, 'day', 50, use_cache=True)
    actual_time = time.time() - start_time
    
    print(f"\nActual fetch time: {actual_time:.1f} seconds")
    print(f"Successful: {batch_result.successful_fetches}/{batch_result.total_symbols}")
    print(f"Cache hits: {batch_result.cache_hits}")
    print(f"API calls: {batch_result.api_calls}")
    print(f"Performance: {len(test_symbols)/actual_time:.1f} symbols/second")
    
    # Show performance stats
    stats = fetcher.get_performance_stats()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"Symbols per minute: {stats['symbols_per_minute']:.1f}")