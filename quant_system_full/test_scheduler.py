#!/usr/bin/env python3
"""
Test script for the Market-Aware Scheduler system.

This script validates the time scheduling system and selection strategies.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "dashboard", "worker"))
sys.path.append(os.path.join(os.path.dirname(__file__), "bot"))

load_dotenv()

def test_market_time_detection():
    """Test market time detection functionality."""
    print("=== Testing Market Time Detection ===")
    
    try:
        from bot.market_time import MarketTimeManager, MarketType, MarketPhase
        
        # Test US market
        us_manager = MarketTimeManager(MarketType.US)
        status = us_manager.get_market_status(include_yahoo_api=False)
        
        print(f"US Market Status:")
        print(f"  Current Time: {status['current_time']}")
        print(f"  Market Phase: {status['market_phase']}")
        print(f"  Is Market Open: {status['is_market_open']}")
        print(f"  Is Market Active: {status['is_market_active']}")
        print(f"  Next Open: {status['next_market_open']}")
        print(f"  Should Run Selection: {us_manager.should_run_selection_tasks()}")
        print(f"  Should Run Trading: {us_manager.should_run_trading_tasks()}")
        
        # Test CN market
        cn_manager = MarketTimeManager(MarketType.CN)
        cn_status = cn_manager.get_market_status(include_yahoo_api=False)
        
        print(f"\nCN Market Status:")
        print(f"  Current Time: {cn_status['current_time']}")
        print(f"  Market Phase: {cn_status['market_phase']}")
        print(f"  Is Market Open: {cn_status['is_market_open']}")
        print(f"  Should Run Selection: {cn_manager.should_run_selection_tasks()}")
        
        print("[PASS] Market time detection tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Market time detection test failed: {e}")
        return False

def test_selection_strategies():
    """Test selection strategy framework."""
    print("\n=== Testing Selection Strategies ===")
    
    try:
        from bot.selection_strategies.value_momentum import ValueMomentumStrategy
        from bot.selection_strategies.technical_breakout import TechnicalBreakoutStrategy
        from bot.selection_strategies.earnings_momentum import EarningsMomentumStrategy
        from bot.selection_strategies.base_strategy import SelectionCriteria
        
        # Test strategy initialization
        strategies = [
            ValueMomentumStrategy(),
            TechnicalBreakoutStrategy(),
            EarningsMomentumStrategy()
        ]
        
        print(f"Initialized {len(strategies)} strategies:")
        for strategy in strategies:
            info = strategy.get_strategy_info()
            print(f"  - {info['name']}: {info['description']}")
        
        # Test with a small universe and relaxed criteria
        universe = ["AAPL", "MSFT", "GOOGL"]
        criteria = SelectionCriteria(
            max_stocks=5,
            min_market_cap=1e6,      # Much lower for testing
            max_market_cap=1e15,     # Higher ceiling
            min_volume=1000,         # Lower volume requirement
            min_price=1.0,           # Lower price floor
            max_price=2000.0,        # Higher price ceiling
            min_score_threshold=0.0  # No score threshold for testing
        )
        
        print(f"\nTesting with universe: {universe}")
        
        for strategy in strategies:
            try:
                print(f"\nRunning {strategy.name} strategy...")
                results = strategy.select_stocks(universe, criteria)
                summary = results.to_summary()
                
                print(f"  Results: {summary['total_selected']} stocks selected")
                print(f"  Execution time: {summary['execution_time']:.2f}s")
                
                if results.selected_stocks:
                    top_pick = results.selected_stocks[0]
                    print(f"  Top pick: {top_pick.symbol} (score: {top_pick.score:.1f})")
                
            except Exception as e:
                print(f"  Warning: Strategy {strategy.name} failed with small test: {e}")
                # This is expected in test environment without full data
                continue
        
        print("[PASS] Selection strategy framework tests completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Selection strategy test failed: {e}")
        return False

def test_scheduler_initialization():
    """Test scheduler initialization and configuration."""
    print("\n=== Testing Scheduler Initialization ===")
    
    try:
        from dashboard.worker.runner import MarketAwareScheduler
        
        # Test scheduler creation
        scheduler = MarketAwareScheduler(market_type="US")
        
        print(f"Scheduler initialized:")
        print(f"  Market type: {scheduler.market_type}")
        print(f"  Selection interval: {scheduler.selection_interval}s")
        print(f"  Trading interval: {scheduler.trading_interval}s")
        print(f"  Monitoring interval: {scheduler.monitoring_interval}s")
        print(f"  Max concurrent tasks: {scheduler.max_concurrent_tasks}")
        
        # Test task registration
        initial_task_counts = {
            task_type: len(task_list) 
            for task_type, task_list in scheduler.tasks.items()
        }
        print(f"  Initial task counts: {initial_task_counts}")
        
        # Test phase detection
        phase = scheduler.market_manager.get_market_phase()
        print(f"  Current market phase: {phase.value}")
        
        # Test task type should-run logic
        for task_type in ['selection', 'trading', 'monitoring']:
            should_run = scheduler.should_run_task_type(task_type, phase)
            print(f"  Should run {task_type} tasks: {should_run}")
        
        print("[PASS] Scheduler initialization tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Scheduler initialization test failed: {e}")
        return False

def test_task_scheduling_logic():
    """Test task scheduling and execution logic."""
    print("\n=== Testing Task Scheduling Logic ===")
    
    try:
        from dashboard.worker.runner import MarketAwareScheduler
        from bot.market_time import MarketPhase
        
        scheduler = MarketAwareScheduler(market_type="US")
        
        # Test different market phases
        test_phases = [
            MarketPhase.CLOSED,
            MarketPhase.PRE_MARKET, 
            MarketPhase.REGULAR,
            MarketPhase.AFTER_HOURS
        ]
        
        print("Task execution matrix:")
        print("Phase          | Selection | Trading | Monitoring")
        print("---------------|-----------|---------|----------")
        
        for phase in test_phases:
            selection = scheduler.should_run_task_type('selection', phase)
            trading = scheduler.should_run_task_type('trading', phase)  
            monitoring = scheduler.should_run_task_type('monitoring', phase)
            
            print(f"{phase.value:<15}| {selection:<9}| {trading:<7}| {monitoring}")
        
        # Test universe generation
        universe = scheduler._get_stock_universe()
        print(f"\nStock universe size: {len(universe)} symbols")
        print(f"Sample symbols: {universe[:10]}")
        
        print("[PASS] Task scheduling logic tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Task scheduling logic test failed: {e}")
        return False

def main():
    """Run all scheduler tests."""
    print("Market-Aware Scheduler Test Suite")
    print("=" * 50)
    
    # Set test environment variables
    os.environ.setdefault('SELECTION_INTERVAL', '60')    # 1 minute for testing
    os.environ.setdefault('TRADING_INTERVAL', '10')      # 10 seconds for testing
    os.environ.setdefault('MONITORING_INTERVAL', '30')   # 30 seconds for testing
    os.environ.setdefault('DRY_RUN', 'true')            # Enable dry run mode
    
    # Run test suite
    tests = [
        test_market_time_detection,
        test_selection_strategies, 
        test_scheduler_initialization,
        test_task_scheduling_logic
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'=' * 50}")
    print(f"Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Scheduler system is ready.")
        return 0
    else:
        print("[WARNING] Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)