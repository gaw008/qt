"""
Performance Optimization System for Quantitative Trading System

This module provides comprehensive performance optimization capabilities:
- Data caching mechanisms for frequently accessed market data
- Concurrent processing optimization for multi-stock operations
- Memory usage optimization for continuous operation
- API call rate limiting and intelligent batching
- Performance monitoring and profiling
"""

import os
import time
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
import weakref
import gc

# Optional imports for enhanced performance
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    max_memory_mb: int = 100
    default_ttl: int = 300  # 5 minutes
    redis_url: Optional[str] = None
    enable_redis: bool = False
    cache_hit_threshold: float = 0.8  # Target cache hit rate


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    api_errors: int = 0
    avg_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_threads: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            
            # Check TTL
            if time.time() > entry['expires']:
                self._remove(key)
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
                
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            expires = time.time() + ttl
            self._cache[key] = {
                'value': value,
                'expires': expires,
                'created': time.time()
            }
            self._access_times[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """Remove key from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
            
        lru_key = min(self._access_times, key=self._access_times.get)
        self._remove(lru_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            expired_count = sum(1 for entry in self._cache.values() 
                              if now > entry['expires'])
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_entries': expired_count,
                'hit_rate': getattr(self, '_hit_rate', 0.0)
            }


class RedisCache:
    """Redis-based cache with fallback to memory cache."""
    
    def __init__(self, redis_url: str, fallback_cache: MemoryCache):
        self.redis_url = redis_url
        self.fallback_cache = fallback_cache
        self._redis = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        if not HAS_REDIS:
            logger.warning("Redis not available, using memory cache only")
            return
            
        try:
            self._redis = redis.from_url(self.redis_url)
            self._redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache")
            self._redis = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache with fallback."""
        if self._redis:
            try:
                value = self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        return self.fallback_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache with fallback."""
        if self._redis:
            try:
                serialized = json.dumps(value, default=str)
                if ttl:
                    self._redis.setex(key, ttl, serialized)
                else:
                    self._redis.set(key, serialized)
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        self.fallback_cache.set(key, value, ttl)


class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.metrics = PerformanceMetrics()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._rate_limiters: Dict[str, 'RateLimiter'] = {}
        
        # Initialize caching
        self.memory_cache = MemoryCache(
            max_size=1000,
            default_ttl=self.config.default_ttl
        )
        
        if self.config.enable_redis and self.config.redis_url and HAS_REDIS:
            self.cache = RedisCache(self.config.redis_url, self.memory_cache)
        else:
            self.cache = self.memory_cache
        
        # Performance monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start performance monitoring thread."""
        def monitor():
            while True:
                try:
                    self._update_metrics()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _update_metrics(self):
        """Update performance metrics."""
        if HAS_PSUTIL:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
            self.metrics.active_threads = process.num_threads()
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.metrics.cache_hits += 1
                    return cached_result
                
                # Execute function and cache result
                self.metrics.cache_misses += 1
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key."""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def batch_execute(self, 
                     func: Callable, 
                     items: List[Any], 
                     batch_size: int = 10,
                     max_workers: int = 5) -> List[Any]:
        """Execute function on items in parallel batches."""
        results = []
        
        def process_batch(batch):
            batch_results = []
            for item in batch:
                try:
                    result = func(item)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    batch_results.append(None)
            return batch_results
        
        # Split items into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch 
                              for batch in batches}
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        return results
    
    def rate_limit(self, calls_per_second: float = 10.0, key: str = "default"):
        """Decorator for rate limiting function calls."""
        if key not in self._rate_limiters:
            self._rate_limiters[key] = RateLimiter(calls_per_second)
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self._rate_limiters[key].acquire()
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    self.metrics.api_calls += 1
                    
                    # Update response time
                    response_time = time.time() - start_time
                    if self.metrics.avg_response_time == 0:
                        self.metrics.avg_response_time = response_time
                    else:
                        # Exponential moving average
                        alpha = 0.1
                        self.metrics.avg_response_time = (
                            alpha * response_time + 
                            (1 - alpha) * self.metrics.avg_response_time
                        )
                    
                    return result
                except Exception as e:
                    self.metrics.api_errors += 1
                    raise e
            return wrapper
        return decorator
    
    def memory_optimize(self):
        """Perform memory optimization."""
        # Clear expired cache entries
        if hasattr(self.cache, 'clear_expired'):
            self.cache.clear_expired()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory optimization completed")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_stats = self.cache.stats() if hasattr(self.cache, 'stats') else {}
        
        return {
            'metrics': {
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'api_calls': self.metrics.api_calls,
                'api_errors': self.metrics.api_errors,
                'avg_response_time': self.metrics.avg_response_time,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'active_threads': self.metrics.active_threads
            },
            'cache_stats': cache_stats,
            'timestamp': datetime.now().isoformat()
        }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.tokens = calls_per_second
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self):
        """Acquire a token (blocks if necessary)."""
        with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.calls_per_second, 
                            self.tokens + elapsed * self.calls_per_second)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
            else:
                # Wait for next token
                sleep_time = (1 - self.tokens) / self.calls_per_second
                time.sleep(sleep_time)
                self.tokens = 0


# Thread-safe global performance optimizer instance
import threading
_global_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance with thread safety."""
    global _global_optimizer
    if _global_optimizer is None:
        with _optimizer_lock:
            if _global_optimizer is None:
                _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Global caching decorator."""
    return get_optimizer().cached(ttl=ttl, key_func=key_func)


def rate_limit(calls_per_second: float = 10.0, key: str = "default"):
    """Global rate limiting decorator."""
    return get_optimizer().rate_limit(calls_per_second=calls_per_second, key=key)


def batch_process(func: Callable, 
                 items: List[Any], 
                 batch_size: int = 10,
                 max_workers: int = 5) -> List[Any]:
    """Global batch processing function."""
    return get_optimizer().batch_execute(func, items, batch_size, max_workers)


# Example usage functions
def demo_caching():
    """Demonstrate caching capabilities."""
    optimizer = get_optimizer()
    
    @optimizer.cached(ttl=60)
    def expensive_calculation(n: int) -> int:
        """Simulate expensive calculation."""
        time.sleep(0.1)  # Simulate work
        return n ** 2
    
    # First calls (cache miss)
    print("First calls (cache miss):")
    start = time.time()
    results = [expensive_calculation(i) for i in range(5)]
    print(f"Results: {results}, Time: {time.time() - start:.2f}s")
    
    # Second calls (cache hit)
    print("\nSecond calls (cache hit):")
    start = time.time()
    results = [expensive_calculation(i) for i in range(5)]
    print(f"Results: {results}, Time: {time.time() - start:.2f}s")
    
    print(f"\nCache hit rate: {optimizer.metrics.cache_hit_rate:.2%}")


def demo_rate_limiting():
    """Demonstrate rate limiting."""
    optimizer = get_optimizer()
    
    @optimizer.rate_limit(calls_per_second=2.0, key="api_calls")
    def api_call(data: str) -> str:
        """Simulate API call."""
        return f"Processed: {data}"
    
    print("Rate limiting demo (2 calls/second):")
    start = time.time()
    
    for i in range(5):
        result = api_call(f"data_{i}")
        elapsed = time.time() - start
        print(f"{elapsed:.1f}s: {result}")


def demo_batch_processing():
    """Demonstrate batch processing."""
    def process_item(item: int) -> int:
        """Simulate processing an item."""
        time.sleep(0.1)  # Simulate work
        return item * 2
    
    items = list(range(20))
    
    print("Sequential processing:")
    start = time.time()
    sequential_results = [process_item(item) for item in items]
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s")
    
    print("\nBatch processing:")
    start = time.time()
    batch_results = batch_process(process_item, items, batch_size=5, max_workers=4)
    batch_time = time.time() - start
    print(f"Time: {batch_time:.2f}s")
    print(f"Speedup: {sequential_time / batch_time:.1f}x")


if __name__ == "__main__":
    print("Performance Optimization Demo")
    print("=" * 40)
    
    demo_caching()
    print("\n" + "=" * 40)
    
    demo_rate_limiting() 
    print("\n" + "=" * 40)
    
    demo_batch_processing()
    print("\n" + "=" * 40)
    
    # Show performance summary
    optimizer = get_optimizer()
    summary = optimizer.get_performance_summary()
    print("Performance Summary:")
    print(json.dumps(summary, indent=2, default=str))