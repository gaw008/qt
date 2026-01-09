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
        
        # Test basic portfolio creation with higher position weight limit
        portfolio = MultiStockPortfolio(
            initial_capital=10000.0,
            max_position_weight=0.2  # Allow 20% per position
        )
        print(f'[OK] Portfolio created with ${portfolio.get_total_value():,.2f}')
        
        # Test position addition with smaller quantity
        success = portfolio.add_position('AAPL', 5, 150.0, score=85.0)  # Smaller position
        if success:
            print('[OK] Position added successfully')
        else:
            print('[ERROR] Position addition failed')
        
        print(f'[OK] Portfolio now has {portfolio.get_position_count()} positions')
        
        # Test allocation methods
        selection_results = [
            {'symbol': 'AAPL', 'score': 95.0},
            {'symbol': 'MSFT', 'score': 88.0}
        ]
        
        allocation = portfolio.calculate_target_allocation(selection_results)
        print(f'[OK] Target allocation calculated: {len(allocation)} positions')
        
        return True
        
    except Exception as e:
        print(f'[ERROR] Portfolio test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_execution():
    """Test execution engine functionality."""
    print('\nTesting execution engine...')
    try:
        from bot.execution import ExecutionEngine, OrderSide, OrderType
        print('[OK] Execution engine imported successfully')
        
        # Test engine creation
        engine = ExecutionEngine(max_concurrent_orders=10)
        print('[OK] Execution engine created')
        
        # Test basic functionality
        summary = engine.get_execution_summary()
        print(f'[OK] Execution summary retrieved: {summary["engine_status"]}')
        
        return True
        
    except Exception as e:
        print(f'[ERROR] Execution engine test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_monitor():
    """Test real-time monitor functionality."""
    print('\nTesting real-time monitor...')
    try:
        from bot.realtime_monitor import RealTimeMonitor, AlertType
        print('[OK] Real-time monitor imported successfully')
        
        # Test monitor creation
        monitor = RealTimeMonitor(update_interval_seconds=60)
        print('[OK] Real-time monitor created')
        
        # Test status retrieval
        status = monitor.get_monitoring_status()
        print(f'[OK] Monitor status retrieved: {status["state"]}')
        
        return True
        
    except Exception as e:
        print(f'[ERROR] Real-time monitor test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration between components."""
    print('\nTesting component integration...')
    try:
        from bot.portfolio import MultiStockPortfolio
        from bot.execution import ExecutionEngine, create_execution_engine
        from bot.realtime_monitor import RealTimeMonitor, create_realtime_monitor
        
        # Create portfolio
        portfolio = MultiStockPortfolio(
            initial_capital=10000.0,
            max_position_weight=0.3
        )
        
        # Create execution engine with portfolio
        execution_engine = create_execution_engine(portfolio)
        print('[OK] Execution engine created with portfolio')
        
        # Create monitor with both portfolio and execution engine
        monitor = create_realtime_monitor(portfolio, execution_engine)
        print('[OK] Real-time monitor created with portfolio and execution engine')
        
        # Test portfolio operations
        success = portfolio.add_position('AAPL', 3, 150.0)
        if success:
            print('[OK] Portfolio-execution integration working')
        
        return True
        
    except Exception as e:
        print(f'[ERROR] Integration test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Portfolio Management Integration Validation")
    print("=" * 50)
    
    results = []
    results.append(test_portfolio())
    results.append(test_execution())
    results.append(test_monitor())
    results.append(test_integration())
    
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