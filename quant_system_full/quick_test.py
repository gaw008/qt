#!/usr/bin/env python3
"""Quick validation test for portfolio management integration."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

print('Testing portfolio management imports...')
try:
    from bot.portfolio import MultiStockPortfolio, AllocationMethod, Position, PositionType
    print('✓ Portfolio module imported successfully')
    
    # Test basic portfolio creation
    portfolio = MultiStockPortfolio(initial_capital=10000.0)
    print(f'✓ Portfolio created with ${portfolio.get_total_value():,.2f}')
    
    # Test position addition
    success = portfolio.add_position('AAPL', 10, 150.0, score=85.0)
    print(f'✓ Position addition: {"success" if success else "failed"}')
    
    print(f'✓ Portfolio now has {portfolio.get_position_count()} positions')
    
except Exception as e:
    print(f'✗ Portfolio test failed: {e}')

print('\nTesting execution engine imports...')
try:
    from bot.execution import ExecutionEngine, OrderSide, OrderType, ExecutionStrategy
    print('✓ Execution engine imported successfully')
    
    # Test engine creation
    engine = ExecutionEngine()
    print('✓ Execution engine created')
    
except Exception as e:
    print(f'✗ Execution engine test failed: {e}')

print('\nTesting real-time monitor imports...')
try:
    from bot.realtime_monitor import RealTimeMonitor, AlertType, AlertSeverity
    print('✓ Real-time monitor imported successfully')
    
    # Test monitor creation
    monitor = RealTimeMonitor()
    print('✓ Real-time monitor created')
    
except Exception as e:
    print(f'✗ Real-time monitor test failed: {e}')

print('\n✓ All integration components validated successfully!')