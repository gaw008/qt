"""
Smart Data Caching System for Quantitative Trading

This module provides an intelligent caching layer to dramatically reduce API calls
by implementing memory-based caching with LRU eviction, time-based validation,
and multi-process support.

Features:
- LRU (Least Recently Used) cache eviction strategy
- Time-based cache invalidation with configurable TTL
- Multi-process safe using multiprocessing.Manager
- Cache hit rate monitoring and statistics
- Automatic memory management for 64GB RAM optimization
- Thread-safe operations
- Different TTL for different time periods
"""

import threading
import time
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import hashlib
import json
import pickle
from multiprocessing import Manager, Lock, RLock
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata"""
    data: pd.DataFrame
    timestamp: datetime
    symbol: str
    period: str
    limit: int
    size_bytes: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    current_entries: int = 0
    total_memory_mb: float = 0.0
    hit_rate: float = 0.0


class SmartDataCache:
    """
    Intelligent data cache for stock market data with LRU eviction,
    time-based invalidation, and multi-process support.
    """
    
    def __init__(self, 
                 max_memory_mb: int = 8192,  # 8GB default for cache
                 default_ttl_minutes: int = 30,
                 enable_multiprocess: bool = True):
        """
        Initialize the smart data cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB (8GB default)
            default_ttl_minutes: Default TTL for cached data in minutes
            enable_multiprocess: Enable multiprocess-safe operations
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.enable_multiprocess = enable_multiprocess
        
        # TTL configuration for different periods
        self.period_ttl = {
            '1min': timedelta(minutes=2),      # Very short for minute data
            '5min': timedelta(minutes=5),      # Short for 5-minute data  
            '15min': timedelta(minutes=15),    # Medium for 15-minute data
            '30min': timedelta(minutes=30),    # Medium for 30-minute data
            '1h': timedelta(hours=1),          # Longer for hourly data
            'day': timedelta(hours=6),         # Long for daily data (6 hours)
            '1wk': timedelta(hours=24),        # Very long for weekly data
            '1mo': timedelta(hours=48),        # Very long for monthly data
        }
        
        if enable_multiprocess:
            # Initialize multiprocess-safe structures
            manager = Manager()
            self._cache = manager.dict()
            self._access_order = manager.list()
            self._cache_lock = manager.RLock()
            self._stats_dict = manager.dict()
            
            # Initialize stats
            stats = CacheStats()
            for key, value in asdict(stats).items():
                self._stats_dict[key] = value
                
        else:
            # Single-process structures
            self._cache = {}
            self._access_order = []
            self._cache_lock = RLock()
            self._stats_dict = asdict(CacheStats())
        
        # Thread-local storage for local statistics
        self._local_stats = threading.local()
        
        logger.info(f"[CACHE] Initialized SmartDataCache: max_memory={max_memory_mb}MB, "
                   f"default_ttl={default_ttl_minutes}min, multiprocess={enable_multiprocess}")

    def _get_cache_key(self, symbol: str, period: str, limit: int) -> str:
        """Generate a unique cache key for the given parameters"""
        key_data = f"{symbol.upper()}_{period}_{limit}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_data_size(self, df: pd.DataFrame) -> int:
        """Estimate DataFrame memory usage in bytes"""
        try:
            # More accurate memory estimation
            return df.memory_usage(deep=True).sum()
        except:
            # Fallback estimation
            return len(df) * len(df.columns) * 8  # Assume 8 bytes per cell

    def _get_ttl_for_period(self, period: str) -> timedelta:
        """Get appropriate TTL based on data period"""
        period_lower = period.lower()
        for key, ttl in self.period_ttl.items():
            if key in period_lower:
                return ttl
        return self.default_ttl

    def _is_entry_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired"""
        ttl = self._get_ttl_for_period(entry.period)
        return datetime.now() - entry.timestamp > ttl

    def _update_access_order(self, cache_key: str):
        """Update LRU access order"""
        try:
            # Remove from current position
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            # Add to end (most recently used)
            self._access_order.append(cache_key)
        except (ValueError, AttributeError):
            # Handle multiprocess list edge cases
            if cache_key not in self._access_order:
                self._access_order.append(cache_key)

    def _get_current_memory_usage(self) -> int:
        """Calculate current total memory usage"""
        total_bytes = 0
        for cache_key in list(self._cache.keys()):
            try:
                entry_data = self._cache[cache_key]
                if isinstance(entry_data, dict):
                    total_bytes += entry_data.get('size_bytes', 0)
                else:
                    total_bytes += entry_data.size_bytes
            except (KeyError, AttributeError):
                continue
        return total_bytes

    def _evict_lru_entries(self, target_bytes: int):
        """Evict least recently used entries to free up memory"""
        evicted_count = 0
        current_memory = self._get_current_memory_usage()
        
        # Create a copy of access order to safely iterate
        access_order_copy = list(self._access_order)
        
        for cache_key in access_order_copy:
            if current_memory <= target_bytes:
                break
                
            try:
                if cache_key in self._cache:
                    entry_data = self._cache[cache_key]
                    entry_size = entry_data.get('size_bytes', 0) if isinstance(entry_data, dict) else entry_data.size_bytes
                    
                    # Remove from cache and access order
                    del self._cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
                    
                    current_memory -= entry_size
                    evicted_count += 1
                    
                    logger.debug(f"[CACHE] Evicted entry {cache_key}, freed {entry_size} bytes")
                    
            except (KeyError, ValueError, AttributeError):
                # Handle multiprocess edge cases
                continue
        
        # Update statistics
        self._stats_dict['evictions'] = self._stats_dict.get('evictions', 0) + evicted_count
        
        if evicted_count > 0:
            logger.info(f"[CACHE] Evicted {evicted_count} entries, freed memory")

    def get(self, symbol: str, period: str = 'day', limit: int = 300) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            symbol: Stock symbol
            period: Time period
            limit: Number of data points
            
        Returns:
            DataFrame if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(symbol, period, limit)
        
        with self._cache_lock:
            self._stats_dict['total_requests'] = self._stats_dict.get('total_requests', 0) + 1
            
            if cache_key not in self._cache:
                self._stats_dict['cache_misses'] = self._stats_dict.get('cache_misses', 0) + 1
                logger.debug(f"[CACHE] MISS: {symbol} {period} {limit}")
                return None
            
            try:
                entry_data = self._cache[cache_key]
                
                # Handle both dict and CacheEntry formats
                if isinstance(entry_data, dict):
                    # Reconstruct CacheEntry from dict
                    entry = CacheEntry(
                        data=pickle.loads(entry_data['data_pickle']) if 'data_pickle' in entry_data else entry_data['data'],
                        timestamp=entry_data['timestamp'],
                        symbol=entry_data['symbol'],
                        period=entry_data['period'],
                        limit=entry_data['limit'],
                        size_bytes=entry_data['size_bytes'],
                        access_count=entry_data.get('access_count', 0),
                        last_accessed=entry_data.get('last_accessed')
                    )
                else:
                    entry = entry_data
                
                # Check if expired
                if self._is_entry_expired(entry):
                    del self._cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
                    self._stats_dict['cache_misses'] = self._stats_dict.get('cache_misses', 0) + 1
                    logger.debug(f"[CACHE] EXPIRED: {symbol} {period} {limit}")
                    return None
                
                # Update access information
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self._update_access_order(cache_key)
                
                # Store updated entry
                if isinstance(entry_data, dict):
                    entry_dict = asdict(entry)
                    entry_dict['data_pickle'] = pickle.dumps(entry.data)
                    entry_dict.pop('data', None)  # Remove original data to save space
                    self._cache[cache_key] = entry_dict
                else:
                    self._cache[cache_key] = entry
                
                # Update statistics
                self._stats_dict['cache_hits'] = self._stats_dict.get('cache_hits', 0) + 1
                total_requests = self._stats_dict.get('total_requests', 1)
                self._stats_dict['hit_rate'] = self._stats_dict.get('cache_hits', 0) / total_requests * 100
                
                logger.info(f"[CACHE] HIT: {symbol} {period} {limit} (accessed {entry.access_count} times)")
                return entry.data.copy()  # Return a copy to prevent external modification
                
            except Exception as e:
                logger.error(f"[CACHE] Error retrieving {cache_key}: {e}")
                # Clean up corrupted entry
                if cache_key in self._cache:
                    del self._cache[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._stats_dict['cache_misses'] = self._stats_dict.get('cache_misses', 0) + 1
                return None

    def put(self, symbol: str, data: pd.DataFrame, period: str = 'day', limit: int = 300):
        """
        Store data in cache with automatic memory management.
        
        Args:
            symbol: Stock symbol
            data: DataFrame to cache
            period: Time period
            limit: Number of data points
        """
        if data is None or data.empty:
            return
        
        cache_key = self._get_cache_key(symbol, period, limit)
        data_size = self._get_data_size(data)
        
        # Skip caching if data is too large
        if data_size > self.max_memory_bytes * 0.1:  # Don't cache entries > 10% of total memory
            logger.warning(f"[CACHE] Skipping large entry: {symbol} {period} {limit} ({data_size} bytes)")
            return
        
        with self._cache_lock:
            # Check if we need to evict entries
            current_memory = self._get_current_memory_usage()
            if current_memory + data_size > self.max_memory_bytes:
                target_memory = self.max_memory_bytes * 0.8  # Keep memory at 80%
                self._evict_lru_entries(target_memory - data_size)
            
            # Create cache entry
            entry = CacheEntry(
                data=data.copy(),  # Store a copy to prevent external modification
                timestamp=datetime.now(),
                symbol=symbol.upper(),
                period=period,
                limit=limit,
                size_bytes=data_size,
                access_count=1,
                last_accessed=datetime.now()
            )
            
            try:
                # Store in multiprocess-safe format
                entry_dict = asdict(entry)
                entry_dict['data_pickle'] = pickle.dumps(entry.data)
                entry_dict.pop('data', None)  # Remove original data to save space
                self._cache[cache_key] = entry_dict
                
                self._update_access_order(cache_key)
                
                # Update statistics
                self._stats_dict['current_entries'] = len(self._cache)
                self._stats_dict['total_memory_mb'] = self._get_current_memory_usage() / (1024 * 1024)
                
                logger.info(f"[CACHE] STORED: {symbol} {period} {limit} "
                           f"({data_size} bytes, {len(self._cache)} total entries)")
                
            except Exception as e:
                logger.error(f"[CACHE] Error storing {cache_key}: {e}")

    def invalidate(self, symbol: str = None, period: str = None):
        """
        Invalidate cache entries based on criteria.
        
        Args:
            symbol: Specific symbol to invalidate (None for all)
            period: Specific period to invalidate (None for all)
        """
        with self._cache_lock:
            keys_to_remove = []
            
            for cache_key in list(self._cache.keys()):
                try:
                    entry_data = self._cache[cache_key]
                    entry_symbol = entry_data.get('symbol', '')
                    entry_period = entry_data.get('period', '')
                    
                    should_remove = True
                    if symbol is not None and entry_symbol.upper() != symbol.upper():
                        should_remove = False
                    if period is not None and entry_period != period:
                        should_remove = False
                    
                    if should_remove:
                        keys_to_remove.append(cache_key)
                        
                except (KeyError, AttributeError):
                    keys_to_remove.append(cache_key)  # Remove corrupted entries
            
            # Remove identified entries
            for cache_key in keys_to_remove:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
            
            # Update statistics
            self._stats_dict['current_entries'] = len(self._cache)
            self._stats_dict['total_memory_mb'] = self._get_current_memory_usage() / (1024 * 1024)
            
            if keys_to_remove:
                logger.info(f"[CACHE] Invalidated {len(keys_to_remove)} entries")

    def clear(self):
        """Clear all cache entries"""
        with self._cache_lock:
            self._cache.clear()
            if hasattr(self._access_order, 'clear'):
                self._access_order.clear()
            else:
                del self._access_order[:]
            
            # Reset statistics
            stats = CacheStats()
            for key, value in asdict(stats).items():
                self._stats_dict[key] = value
            
            logger.info("[CACHE] Cleared all entries")

    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self._cache_lock:
            # Update current statistics
            self._stats_dict['current_entries'] = len(self._cache)
            self._stats_dict['total_memory_mb'] = self._get_current_memory_usage() / (1024 * 1024)
            
            total_requests = self._stats_dict.get('total_requests', 0)
            cache_hits = self._stats_dict.get('cache_hits', 0)
            self._stats_dict['hit_rate'] = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return CacheStats(**dict(self._stats_dict))

    def cleanup_expired(self):
        """Remove all expired entries"""
        with self._cache_lock:
            keys_to_remove = []
            
            for cache_key in list(self._cache.keys()):
                try:
                    entry_data = self._cache[cache_key]
                    
                    # Reconstruct minimal entry for expiry check
                    timestamp = entry_data.get('timestamp')
                    period = entry_data.get('period', '')
                    
                    if timestamp and period:
                        ttl = self._get_ttl_for_period(period)
                        if datetime.now() - timestamp > ttl:
                            keys_to_remove.append(cache_key)
                    else:
                        keys_to_remove.append(cache_key)  # Remove malformed entries
                        
                except (KeyError, AttributeError, TypeError):
                    keys_to_remove.append(cache_key)  # Remove corrupted entries
            
            # Remove expired entries
            for cache_key in keys_to_remove:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
            
            # Update statistics
            self._stats_dict['current_entries'] = len(self._cache)
            self._stats_dict['total_memory_mb'] = self._get_current_memory_usage() / (1024 * 1024)
            
            if keys_to_remove:
                logger.info(f"[CACHE] Cleaned up {len(keys_to_remove)} expired entries")

    def print_stats(self):
        """Print detailed cache statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("SMART DATA CACHE STATISTICS")
        print("="*50)
        print(f"Total Requests: {stats.total_requests}")
        print(f"Cache Hits: {stats.cache_hits}")
        print(f"Cache Misses: {stats.cache_misses}")
        print(f"Hit Rate: {stats.hit_rate:.2f}%")
        print(f"Current Entries: {stats.current_entries}")
        print(f"Total Evictions: {stats.evictions}")
        print(f"Memory Usage: {stats.total_memory_mb:.2f} MB")
        print(f"Memory Limit: {self.max_memory_bytes / (1024*1024):.2f} MB")
        print("="*50)


# Global cache instance
_global_cache = None
_cache_lock = threading.Lock()


def get_cache() -> SmartDataCache:
    """Get the global cache instance (singleton pattern)"""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = SmartDataCache(
                    max_memory_mb=8192,  # 8GB for cache
                    default_ttl_minutes=30,
                    enable_multiprocess=True
                )
    return _global_cache


def set_extended_cache_ttl(ttl_seconds: int):
    """
    Set extended cache TTL for large-scale operations like 5000+ stock processing.
    
    Args:
        ttl_seconds: TTL in seconds (e.g., 5400 = 90 minutes)
    """
    cache = get_cache()
    cache.default_ttl = timedelta(seconds=ttl_seconds)
    logger.info(f"[CACHE] Extended cache TTL set to {ttl_seconds} seconds ({ttl_seconds/60:.1f} minutes)")


def reset_cache_ttl():
    """Reset cache TTL to default (30 minutes)"""
    cache = get_cache()
    cache.default_ttl = timedelta(minutes=30)
    logger.info("[CACHE] Cache TTL reset to default 30 minutes")


def cached_fetch_history(fetch_func, symbol: str, period: str = 'day', 
                        limit: int = 300, *args, **kwargs) -> Optional[pd.DataFrame]:
    """
    Cached wrapper for data fetching functions.
    
    Args:
        fetch_func: Original data fetching function
        symbol: Stock symbol
        period: Time period
        limit: Number of data points
        *args, **kwargs: Additional arguments for fetch_func
        
    Returns:
        DataFrame from cache or fresh from API
    """
    cache = get_cache()
    
    # Try cache first
    cached_data = cache.get(symbol, period, limit)
    if cached_data is not None:
        return cached_data
    
    # Fetch fresh data
    try:
        fresh_data = fetch_func(symbol, period, limit, *args, **kwargs)
        if fresh_data is not None and not fresh_data.empty:
            cache.put(symbol, fresh_data, period, limit)
            return fresh_data
    except Exception as e:
        logger.error(f"[CACHE] Fetch function failed for {symbol}: {e}")
    
    return None


def cached_batch_fetch(fetch_func, symbols: List[str], period: str = 'day', 
                      limit: int = 300, *args, **kwargs) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Cached wrapper for batch data fetching functions.
    
    Args:
        fetch_func: Original batch data fetching function
        symbols: List of stock symbols
        period: Time period
        limit: Number of data points
        *args, **kwargs: Additional arguments for fetch_func
        
    Returns:
        Dictionary mapping symbols to their DataFrames
    """
    cache = get_cache()
    result = {}
    symbols_to_fetch = []
    
    # Check cache for each symbol
    for symbol in symbols:
        cached_data = cache.get(symbol, period, limit)
        if cached_data is not None:
            result[symbol] = cached_data
        else:
            symbols_to_fetch.append(symbol)
    
    # Fetch missing data
    if symbols_to_fetch:
        logger.info(f"[CACHE] Fetching {len(symbols_to_fetch)} symbols, "
                   f"{len(result)} from cache")
        
        try:
            fresh_data = fetch_func(symbols_to_fetch, period, limit, *args, **kwargs)
            
            # Cache the fresh data
            for symbol, df in fresh_data.items():
                if df is not None and not df.empty:
                    cache.put(symbol, df, period, limit)
                result[symbol] = df
                
        except Exception as e:
            logger.error(f"[CACHE] Batch fetch function failed: {e}")
            # Return None for symbols we couldn't fetch
            for symbol in symbols_to_fetch:
                result[symbol] = None
    
    return result


if __name__ == "__main__":
    # Test the cache system
    cache = SmartDataCache(max_memory_mb=100, default_ttl_minutes=5)
    
    # Create sample data
    import numpy as np
    sample_data = pd.DataFrame({
        'time': pd.date_range('2023-01-01', periods=100, freq='D'),
        'open': np.random.randn(100) * 10 + 100,
        'high': np.random.randn(100) * 10 + 110,
        'low': np.random.randn(100) * 10 + 90,
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Test cache operations
    print("Testing cache operations...")
    
    # Store data
    cache.put("AAPL", sample_data, "day", 100)
    cache.put("GOOGL", sample_data, "day", 100)
    
    # Retrieve data
    retrieved = cache.get("AAPL", "day", 100)
    print(f"Retrieved data shape: {retrieved.shape if retrieved is not None else 'None'}")
    
    # Print statistics
    cache.print_stats()