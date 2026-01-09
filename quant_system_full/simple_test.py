#!/usr/bin/env python3
"""Simple validation test for portfolio management integration."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

def test_portfolio():
    """Test portfolio management functionality."""
    print('Testing portfolio management...')
    try:
        from bot.portfolio import MultiStockPortfolio, AllocationMethod, PositionType
        print('[OK] Portfolio module imported successfully')
        
        # Test basic portfolio creation
        portfolio = MultiStockPortfolio(initial_capital=10000.0)
        print(f'[OK] Portfolio created with ${portfolio.get_total_value():,.2f}')
        
        # Test position addition
        success = portfolio.add_position('AAPL', 10, 150.0, score=85.0)
        if success:
            print('[OK] Position added successfully')
        else:
            print('[ERROR] Position addition failed')
        
        print(f'[OK] Portfolio now has {portfolio.get_position_count()} positions')
        return True
        
    except Exception as e:
        print(f'[ERROR] Portfolio test failed: {e}')
        return False

def test_execution():
    """Test execution engine functionality."""
    print('\nTesting execution engine...')
    try:
        from bot.execution import ExecutionEngine, OrderSide, OrderType
        print('[OK] Execution engine imported successfully')
        
        # Test engine creation
        engine = ExecutionEngine()
        print('[OK] Execution engine created')
        return True
        
    except Exception as e:
        print(f'[ERROR] Execution engine test failed: {e}')
        return False

def test_monitor():
    """Test real-time monitor functionality."""
    print('\nTesting real-time monitor...')
    try:
        from bot.realtime_monitor import RealTimeMonitor, AlertType
        print('[OK] Real-time monitor imported successfully')
        
        # Test monitor creation
        monitor = RealTimeMonitor()
        print('[OK] Real-time monitor created')
        return True
        
    except Exception as e:
        print(f'[ERROR] Real-time monitor test failed: {e}')
        return False

def main():
    """Run all tests."""
    print("Portfolio Management Integration Validation")
    print("=" * 50)
    
    results = []
    results.append(test_portfolio())
    results.append(test_execution())
    results.append(test_monitor())
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All integration components validated successfully!")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)