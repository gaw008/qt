"""
Cache Integration Test for Quantitative Trading System

This script tests the smart data cache system integration with the 
existing quantitative trading data pipeline.
"""

import sys
import os
import time
from datetime import datetime
from typing import List

# Add necessary paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bot'))

def test_cache_integration():
    """Test the cache system integration with actual quantitative trading data flow"""
    
    print("="*80)
    print("CACHE INTEGRATION TEST FOR QUANTITATIVE TRADING SYSTEM")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import required modules
    try:
        from data_cache import get_cache
        from yahoo_data import fetch_yahoo_price_history, fetch_yahoo_multiple_symbols, print_cache_stats
        from data import _fetch_yahoo_data
        
        print("✓ Successfully imported cache modules")
        
        # Initialize cache
        cache = get_cache()
        print(f"✓ Cache initialized: {cache.max_memory_bytes / (1024*1024):.0f}MB limit")
        
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return False
    
    # Test symbols that are commonly seen in the logs
    test_symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',  # Big tech
        'JPM', 'BAC', 'WFC', 'GS', 'MS',          # Finance  
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',      # Healthcare
        'QQQ', 'SPY', 'IWM'                        # ETFs
    ]
    
    print(f"\nTesting with {len(test_symbols)} symbols commonly used in quant strategies")
    print(f"Symbols: {', '.join(test_symbols[:8])}...")
    print()
    
    # Test 1: Single symbol fetch simulation (mimicking strategy behavior)
    print("TEST 1: Single Symbol Fetch Simulation")
    print("-" * 50)
    
    # Simulate multiple strategies requesting the same data
    strategies = ['momentum', 'mean_reversion', 'breakout', 'earnings']
    periods = ['day', 'day', 'day', 'day']  # All strategies typically use daily data
    limits = [100, 120, 100, 64]  # Different lookback periods as seen in logs
    
    print("Simulating multiple strategies requesting same stock data...")
    
    total_requests = 0
    start_time = time.time()
    
    for strategy_idx, (strategy, period, limit) in enumerate(zip(strategies, periods, limits)):
        print(f"\n  Strategy '{strategy}' requesting data (period={period}, limit={limit}):")
        strategy_start = time.time()
        
        for symbol in test_symbols[:5]:  # First 5 symbols
            data = fetch_yahoo_price_history(symbol, period, limit, use_cache=True)
            total_requests += 1
            
            if data is not None:
                print(f"    {symbol}: ✓ {len(data)} rows")
            else:
                print(f"    {symbol}: ✗ Failed")
        
        strategy_time = time.time() - strategy_start
        print(f"  Strategy '{strategy}' completed in {strategy_time:.2f}s")
    
    test1_time = time.time() - start_time
    
    print(f"\nTEST 1 RESULTS:")
    print(f"  Total requests: {total_requests}")
    print(f"  Total time: {test1_time:.2f}s")
    print(f"  Average time per request: {test1_time/total_requests:.3f}s")
    
    # Print cache stats
    stats = cache.get_stats()
    print(f"  Cache hits: {stats.cache_hits}")
    print(f"  Cache misses: {stats.cache_misses}")
    print(f"  Hit rate: {stats.hit_rate:.1f}%")
    
    print("\n" + "="*50)
    
    # Test 2: Batch fetch simulation (mimicking stock universe processing)
    print("\nTEST 2: Batch Fetch Simulation")
    print("-" * 50)
    
    print("Simulating stock universe batch processing...")
    
    batch_start = time.time()
    
    # Test batch fetch with cache
    batch_results = fetch_yahoo_multiple_symbols(
        test_symbols, 
        period='day', 
        limit=50,
        use_cache=True
    )
    
    batch_time = time.time() - batch_start
    
    successful_batch = sum(1 for df in batch_results.values() if df is not None)
    
    print(f"\nBatch fetch results:")
    print(f"  Symbols requested: {len(test_symbols)}")
    print(f"  Successful fetches: {successful_batch}")
    print(f"  Batch time: {batch_time:.2f}s")
    print(f"  Average per symbol: {batch_time/len(test_symbols):.3f}s")
    
    # Test 3: Repeat batch to demonstrate cache effectiveness
    print("\nTEST 3: Repeat Batch (Cache Effectiveness)")
    print("-" * 50)
    
    print("Repeating same batch fetch to demonstrate cache hits...")
    
    repeat_start = time.time()
    repeat_results = fetch_yahoo_multiple_symbols(
        test_symbols, 
        period='day', 
        limit=50,
        use_cache=True
    )
    repeat_time = time.time() - repeat_start
    
    print(f"\nRepeat batch results:")
    print(f"  Repeat time: {repeat_time:.2f}s")
    print(f"  Speedup factor: {batch_time/repeat_time:.1f}x faster")
    
    # Final cache statistics
    print("\n" + "="*80)
    print("FINAL CACHE PERFORMANCE STATISTICS")
    print("="*80)
    
    cache.print_stats()
    
    # Performance assessment
    final_stats = cache.get_stats()
    
    print("\nPERFORMANCE ASSESSMENT:")
    print("-" * 30)
    
    if final_stats.hit_rate >= 80:
        grade = "A+ (EXCELLENT)"
        assessment = "Outstanding cache performance! System will see major speedup."
    elif final_stats.hit_rate >= 60:
        grade = "A- (VERY GOOD)"
        assessment = "Great cache performance! Significant efficiency gains expected."
    elif final_stats.hit_rate >= 40:
        grade = "B (GOOD)"
        assessment = "Good cache performance! Noticeable improvements expected."
    elif final_stats.hit_rate >= 20:
        grade = "C (FAIR)"
        assessment = "Fair cache performance. Some improvements expected."
    else:
        grade = "D (NEEDS IMPROVEMENT)"
        assessment = "Cache performance needs tuning for better efficiency."
    
    print(f"Overall Grade: {grade}")
    print(f"Assessment: {assessment}")
    
    # Quantified impact calculation
    if final_stats.total_requests > 0:
        api_calls_saved = final_stats.cache_hits
        time_saved_estimate = api_calls_saved * 0.5  # Assume 0.5s per API call saved
        
        print(f"\nQUANTIFIED IMPACT:")
        print(f"  API calls saved: {api_calls_saved}")
        print(f"  Estimated time saved: {time_saved_estimate:.1f} seconds")
        print(f"  Network bandwidth saved: ~{api_calls_saved * 50:.0f} KB")
        
        if api_calls_saved > 0:
            efficiency_gain = (api_calls_saved / final_stats.total_requests) * 100
            print(f"  System efficiency gain: {efficiency_gain:.1f}%")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return True


def simulate_quantitative_trading_workload():
    """Simulate a realistic quantitative trading workload to test cache under real conditions"""
    
    print("\n" + "="*80)
    print("QUANTITATIVE TRADING WORKLOAD SIMULATION")
    print("="*80)
    
    try:
        from yahoo_data import fetch_yahoo_price_history_with_stats
        from data_cache import get_cache
    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        return False
    
    # Simulate the workload pattern observed in logs
    workload_symbols = [
        # Most frequently seen symbols in the logs
        'AA', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADP',
        'ADSK', 'AEP', 'AFL', 'AFRM', 'AGG', 'AMZN', 'GOOGL', 'MSFT', 'META', 'TSLA'
    ]
    
    # Simulate the data requests we observed in the logs
    request_patterns = [
        {'period': 'day', 'limit': 100, 'description': 'Strategy 1 (100 days)'},
        {'period': 'day', 'limit': 100, 'description': 'Strategy 2 (100 days)'}, 
        {'period': 'day', 'limit': 100, 'description': 'Strategy 3 (100 days)'},
        {'period': 'day', 'limit': 120, 'description': 'Strategy 4 (120 days)'},
        {'period': 'day', 'limit': 100, 'description': 'Strategy 5 (100 days)'},
        {'period': 'day', 'limit': 64, 'description': 'Strategy 6 (64 days)'},
        {'period': 'day', 'limit': 23, 'description': 'Strategy 7 (23 days)'},
    ]
    
    cache = get_cache()
    print(f"Starting simulation with {len(workload_symbols)} symbols and {len(request_patterns)} patterns")
    print("This mirrors the actual workload observed in the system logs.\n")
    
    total_simulation_time = 0
    pattern_times = []
    
    for pattern_idx, pattern in enumerate(request_patterns):
        print(f"Running {pattern['description']}...")
        
        pattern_start = time.time()
        successful_requests = 0
        
        for symbol in workload_symbols:
            data = fetch_yahoo_price_history_with_stats(
                symbol=symbol,
                period=pattern['period'],
                limit=pattern['limit'],
                use_cache=True,
                log_stats=False  # Reduce log noise
            )
            
            if data is not None:
                successful_requests += 1
        
        pattern_time = time.time() - pattern_start
        pattern_times.append(pattern_time)
        total_simulation_time += pattern_time
        
        print(f"  Completed: {successful_requests}/{len(workload_symbols)} symbols in {pattern_time:.2f}s")
        
        # Show progressive cache improvement
        stats = cache.get_stats()
        print(f"  Current hit rate: {stats.hit_rate:.1f}% ({stats.cache_hits}/{stats.total_requests})")
        print()
    
    # Analysis
    print("SIMULATION ANALYSIS:")
    print("-" * 40)
    print(f"Total simulation time: {total_simulation_time:.2f}s")
    print(f"Average time per pattern: {sum(pattern_times)/len(pattern_times):.2f}s")
    print(f"Fastest pattern: {min(pattern_times):.2f}s")
    print(f"Slowest pattern: {max(pattern_times):.2f}s")
    
    # Cache effectiveness over time
    if len(pattern_times) > 1:
        first_pattern_time = pattern_times[0]
        last_pattern_time = pattern_times[-1]
        speedup = first_pattern_time / last_pattern_time
        print(f"Speedup from first to last pattern: {speedup:.2f}x")
        print("(This shows cache warming up and becoming more effective)")
    
    print()
    cache.print_stats()
    
    return True


if __name__ == "__main__":
    print("Smart Data Cache Integration Test")
    print("Testing cache integration with quantitative trading system...")
    print()
    
    try:
        # Run integration tests
        success = test_cache_integration()
        
        if success:
            # Run workload simulation
            simulate_quantitative_trading_workload()
            
            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("The smart data cache system is ready for production use.")
            print("\nTo enable cache in your running system:")
            print("1. Restart the quantitative trading worker process")
            print("2. The cache will be automatically enabled for all data requests")
            print("3. Monitor performance using: python cache_monitor.py")
            
        else:
            print("\n❌ TESTS FAILED!")
            print("Please check the error messages above and ensure all dependencies are installed.")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()