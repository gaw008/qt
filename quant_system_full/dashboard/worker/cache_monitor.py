"""
Cache Performance Monitor

This script monitors the performance of the smart data cache system
and provides real-time statistics about cache efficiency.
"""

import time
import threading
import logging
from datetime import datetime
from typing import List
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_cache import get_cache
try:
    from ..bot.yahoo_data import print_cache_stats, get_cache_stats
    from ..bot.data import print_data_cache_stats
except ImportError:
    # Fallback imports
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot'))
    try:
        from yahoo_data import print_cache_stats, get_cache_stats
        from data import print_data_cache_stats
    except ImportError as e:
        print(f"Warning: Could not import cache stats functions: {e}")
        print_cache_stats = None
        get_cache_stats = None
        print_data_cache_stats = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheMonitor:
    """Real-time cache performance monitor"""
    
    def __init__(self, update_interval: int = 30):
        """
        Initialize cache monitor
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None
        self.last_stats = None
        
    def start(self):
        """Start monitoring cache performance"""
        if self.running:
            logger.warning("Cache monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Cache monitor started with {self.update_interval}s update interval")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Cache monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._update_stats()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in cache monitor loop: {e}")
                time.sleep(5)  # Wait a bit before retrying
                
    def _update_stats(self):
        """Update and display cache statistics"""
        try:
            cache = get_cache()
            current_stats = cache.get_stats()
            
            # Calculate deltas if we have previous stats
            if self.last_stats:
                delta_requests = current_stats.total_requests - self.last_stats.total_requests
                delta_hits = current_stats.cache_hits - self.last_stats.cache_hits
                delta_misses = current_stats.cache_misses - self.last_stats.cache_misses
                delta_evictions = current_stats.evictions - self.last_stats.evictions
                
                if delta_requests > 0:
                    recent_hit_rate = (delta_hits / delta_requests) * 100
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] CACHE PERFORMANCE UPDATE")
                    print(f"Recent Activity: {delta_requests} requests, {delta_hits} hits, {delta_misses} misses")
                    print(f"Recent Hit Rate: {recent_hit_rate:.1f}%")
                    if delta_evictions > 0:
                        print(f"Evictions: {delta_evictions}")
            
            # Always show overall stats
            print(f"\n{'='*60}")
            print(f"CACHE STATISTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Total Requests: {current_stats.total_requests}")
            print(f"Cache Hits: {current_stats.cache_hits}")
            print(f"Cache Misses: {current_stats.cache_misses}")
            print(f"Overall Hit Rate: {current_stats.hit_rate:.2f}%")
            print(f"Current Entries: {current_stats.current_entries}")
            print(f"Total Evictions: {current_stats.evictions}")
            print(f"Memory Usage: {current_stats.total_memory_mb:.1f} MB")
            
            # Performance assessment
            if current_stats.hit_rate >= 80:
                performance = "EXCELLENT"
            elif current_stats.hit_rate >= 60:
                performance = "GOOD"
            elif current_stats.hit_rate >= 40:
                performance = "FAIR"
            else:
                performance = "POOR"
            
            print(f"Performance: {performance}")
            print(f"{'='*60}")
            
            self.last_stats = current_stats
            
        except Exception as e:
            logger.error(f"Error updating cache stats: {e}")
            
    def print_detailed_report(self):
        """Print a detailed performance report"""
        try:
            cache = get_cache()
            stats = cache.get_stats()
            
            print(f"\n{'='*80}")
            print("DETAILED CACHE PERFORMANCE REPORT")
            print(f"{'='*80}")
            print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Request statistics
            print("REQUEST STATISTICS:")
            print(f"  Total Requests: {stats.total_requests:,}")
            print(f"  Cache Hits: {stats.cache_hits:,}")
            print(f"  Cache Misses: {stats.cache_misses:,}")
            print(f"  Hit Rate: {stats.hit_rate:.2f}%")
            print()
            
            # Memory statistics
            print("MEMORY STATISTICS:")
            print(f"  Current Entries: {stats.current_entries:,}")
            print(f"  Memory Usage: {stats.total_memory_mb:.2f} MB")
            max_memory_mb = cache.max_memory_bytes / (1024 * 1024)
            usage_percent = (stats.total_memory_mb / max_memory_mb) * 100
            print(f"  Memory Limit: {max_memory_mb:.2f} MB")
            print(f"  Memory Usage: {usage_percent:.1f}%")
            print()
            
            # Eviction statistics
            print("EVICTION STATISTICS:")
            print(f"  Total Evictions: {stats.evictions:,}")
            if stats.total_requests > 0:
                eviction_rate = (stats.evictions / stats.total_requests) * 100
                print(f"  Eviction Rate: {eviction_rate:.2f}%")
            print()
            
            # Performance assessment
            print("PERFORMANCE ASSESSMENT:")
            if stats.hit_rate >= 80:
                assessment = "EXCELLENT - Cache is highly effective"
            elif stats.hit_rate >= 60:
                assessment = "GOOD - Cache is performing well"
            elif stats.hit_rate >= 40:
                assessment = "FAIR - Some room for improvement"
            elif stats.hit_rate >= 20:
                assessment = "POOR - Cache effectiveness is limited"
            else:
                assessment = "VERY POOR - Cache may need tuning"
            
            print(f"  Overall Rating: {assessment}")
            
            # Recommendations
            print("\nRECOMMENDations:")
            if stats.hit_rate < 50:
                print("  - Consider increasing cache TTL for certain data types")
                print("  - Review data access patterns for optimization opportunities")
            
            if usage_percent > 90:
                print("  - Consider increasing cache memory limit")
                print("  - Monitor for excessive evictions")
            elif usage_percent < 30 and stats.current_entries > 100:
                print("  - Current memory usage is low, cache size appears appropriate")
            
            if stats.evictions > stats.cache_hits:
                print("  - High eviction rate detected, consider increasing cache size")
            
            print(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")


def test_cache_performance(symbols: List[str] = None, iterations: int = 3):
    """
    Test cache performance with sample data fetches
    
    Args:
        symbols: List of symbols to test with
        iterations: Number of test iterations
    """
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']
    
    print(f"\n{'='*60}")
    print("CACHE PERFORMANCE TEST")
    print(f"{'='*60}")
    print(f"Testing with {len(symbols)} symbols, {iterations} iterations")
    print(f"Symbols: {', '.join(symbols)}")
    print()
    
    try:
        # Import data fetching function
        try:
            from ..bot.yahoo_data import fetch_yahoo_price_history_with_stats
        except ImportError:
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot'))
            from yahoo_data import fetch_yahoo_price_history_with_stats
        
        cache = get_cache()
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}:")
            start_time = time.time()
            
            for symbol in symbols:
                data = fetch_yahoo_price_history_with_stats(
                    symbol, 'day', 100, log_stats=True
                )
                if data is not None:
                    print(f"  {symbol}: {len(data)} rows fetched")
                else:
                    print(f"  {symbol}: Failed to fetch data")
            
            elapsed = time.time() - start_time
            print(f"  Iteration completed in {elapsed:.2f} seconds")
            
            # Print stats after each iteration
            stats = cache.get_stats()
            print(f"  Current hit rate: {stats.hit_rate:.1f}%")
            
            if iteration < iterations - 1:
                print("  Waiting 2 seconds before next iteration...")
                time.sleep(2)
        
        print(f"\n{'='*60}")
        print("FINAL PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        # Print final detailed report
        monitor = CacheMonitor()
        monitor.print_detailed_report()
        
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure the bot module is properly configured")
    except Exception as e:
        print(f"Error during cache performance test: {e}")


if __name__ == "__main__":
    # Run cache performance test
    print("Starting Cache Performance Monitor and Test")
    
    # Test cache performance first
    test_cache_performance()
    
    # Start monitoring
    print("\nStarting continuous cache monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    monitor = CacheMonitor(update_interval=60)  # Update every 60 seconds
    
    try:
        monitor.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping cache monitor...")
        monitor.stop()
        print("Cache monitor stopped")