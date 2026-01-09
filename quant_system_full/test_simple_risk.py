"""
Simple Test for Risk Control and Backtesting Systems

A simplified test to verify the core functionality works correctly.
"""

import sys
import os
from pathlib import Path

# Add bot directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from bot.risk_filters import RiskFilterEngine, RiskLimits
        print("+ risk_filters imported successfully")
    except ImportError as e:
        print(f"- Failed to import risk_filters: {e}")
        return False
    
    try:
        from bot.risk_integrated_selection import RiskIntegratedSelector
        print("+ risk_integrated_selection imported successfully")
    except ImportError as e:
        print(f"- Failed to import risk_integrated_selection: {e}")
        return False
    
    try:
        from backtest import PortfolioBacktester
        print("+ backtest imported successfully")
    except ImportError as e:
        print(f"- Failed to import backtest: {e}")
        return False
    
    return True

def test_risk_filter_creation():
    """Test basic risk filter engine creation."""
    print("\nTesting risk filter engine creation...")
    
    try:
        from bot.risk_filters import RiskFilterEngine, RiskLimits
        
        # Test default creation
        engine = RiskFilterEngine()
        print("+ Default risk engine created successfully")
        
        # Test custom limits
        custom_limits = RiskLimits(max_volatility=0.5, min_avg_volume=1000000)
        engine = RiskFilterEngine(custom_limits)
        print("+ Custom risk engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"- Risk filter engine creation failed: {e}")
        return False

def test_backtester_creation():
    """Test portfolio backtester creation."""
    print("\nTesting portfolio backtester creation...")
    
    try:
        from backtest import PortfolioBacktester
        
        backtester = PortfolioBacktester(
            start_date="2022-01-01",
            end_date="2023-01-01",
            initial_capital=1000000.0
        )
        print("+ Portfolio backtester created successfully")
        
        return True
        
    except Exception as e:
        print(f"- Portfolio backtester creation failed: {e}")
        return False

def test_integrated_selector_creation():
    """Test integrated selector creation."""
    print("\nTesting integrated selector creation...")
    
    try:
        from bot.risk_integrated_selection import RiskIntegratedSelector, SelectionConfig
        
        # Test default creation
        selector = RiskIntegratedSelector()
        print("+ Default selector created successfully")
        
        # Test custom config
        config = SelectionConfig(max_positions=15, target_volatility=0.12)
        selector = RiskIntegratedSelector(config)
        print("+ Custom selector created successfully")
        
        return True
        
    except Exception as e:
        print(f"- Integrated selector creation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from bot.risk_filters import RiskFilterEngine
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create simple test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate simple test data for AAPL
        np.random.seed(42)  # For reproducible results
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_days)))
        volumes = np.random.randint(1000000, 5000000, n_days)
        
        test_df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            'close': prices,
            'volume': volumes
        })
        
        market_data = {'AAPL': test_df}
        symbols = ['AAPL']
        
        # Test risk engine
        engine = RiskFilterEngine()
        risk_metrics = engine._calculate_risk_metrics(symbols, market_data, None, None)
        
        if 'AAPL' in risk_metrics:
            metrics = risk_metrics['AAPL']
            print(f"+ Risk metrics calculated: volatility={metrics.volatility:.3f}, liquidity_score={metrics.liquidity_score:.3f}")
        else:
            print("- Risk metrics calculation failed")
            return False
        
        # Test basic filtering
        filtered_symbols, _ = engine.apply_risk_filters(symbols, market_data)
        print(f"+ Risk filtering completed: {len(filtered_symbols)} symbols passed")
        
        return True
        
    except Exception as e:
        print(f"- Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_tests():
    """Run all simple tests."""
    print("=" * 60)
    print("SIMPLE RISK CONTROL AND BACKTESTING TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_risk_filter_creation,
        test_backtester_creation,
        test_integrated_selector_creation,
        test_basic_functionality
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"- {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"TESTS PASSED: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED! Basic functionality is working.")
    else:
        print("WARNING: SOME TESTS FAILED! Please check the implementation.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)