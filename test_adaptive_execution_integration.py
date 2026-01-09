#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Adaptive Execution Engine Integration
Tests the integration of adaptive execution into auto trading engine
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
bot_path = str((Path(__file__).parent / "quant_system_full" / "bot").resolve())
worker_path = str((Path(__file__).parent / "quant_system_full" / "dashboard" / "worker").resolve())
backend_path = str((Path(__file__).parent / "quant_system_full" / "dashboard" / "backend").resolve())
sys.path.insert(0, bot_path)
sys.path.insert(0, worker_path)
sys.path.insert(0, backend_path)

def test_adaptive_execution_integration():
    """Test adaptive execution integration with auto trading engine"""

    print("=" * 80)
    print("ADAPTIVE EXECUTION ENGINE INTEGRATION TEST")
    print("=" * 80)

    # Test 1: Import checks
    print("\n[TEST 1] Import Checks")
    print("-" * 80)

    try:
        from adaptive_execution_engine import AdaptiveExecutionEngine, ExecutionOrder, ExecutionUrgency
        print("[PASS] Adaptive Execution Engine imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import Adaptive Execution Engine: {e}")
        return False

    try:
        from auto_trading_engine import AutoTradingEngine
        print("[PASS] Auto Trading Engine imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import Auto Trading Engine: {e}")
        return False

    # Test 2: Initialize Auto Trading Engine in dry run mode
    print("\n[TEST 2] Auto Trading Engine Initialization (DRY RUN)")
    print("-" * 80)

    try:
        engine = AutoTradingEngine(dry_run=True, max_daily_trades=10)
        print(f"[PASS] Auto Trading Engine initialized")
        print(f"  - DRY_RUN: {engine.dry_run}")
        print(f"  - Adaptive Execution Available: {engine.use_adaptive_execution}")
        print(f"  - Max Daily Trades: {engine.max_daily_trades}")
    except Exception as e:
        print(f"[FAIL] Failed to initialize Auto Trading Engine: {e}")
        return False

    # Test 3: Check trading summary
    print("\n[TEST 3] Trading Summary")
    print("-" * 80)

    try:
        summary = engine.get_trading_summary()
        print(f"[PASS] Trading summary retrieved:")
        print(f"  - Daily Trade Count: {summary['daily_trade_count']}")
        print(f"  - Max Daily Trades: {summary['max_daily_trades']}")
        print(f"  - Total Trades: {summary['total_trades']}")
        print(f"  - DRY_RUN: {summary['dry_run']}")
    except Exception as e:
        print(f"[FAIL] Failed to get trading summary: {e}")
        return False

    # Test 4: Simulate trading signals
    print("\n[TEST 4] Simulate Trading Signal Analysis")
    print("-" * 80)

    try:
        # Current positions (empty)
        current_positions = []

        # Recommended positions
        recommended_positions = [
            {
                'symbol': 'AAPL',
                'action': 'buy',
                'score': 85.5,
                'price': 150.0
            },
            {
                'symbol': 'MSFT',
                'action': 'buy',
                'score': 82.3,
                'price': 350.0
            }
        ]

        signals = engine.analyze_trading_opportunities(current_positions, recommended_positions)

        print(f"[PASS] Trading signals analyzed:")
        print(f"  - Buy Signals: {len(signals['buy'])}")
        print(f"  - Sell Signals: {len(signals['sell'])}")
        print(f"  - Hold Signals: {len(signals['hold'])}")

        if signals['buy']:
            print("\n  Buy Signals:")
            for signal in signals['buy']:
                print(f"    - {signal['symbol']}: {signal['qty']} shares @ ${signal['price']:.2f}")
                print(f"      Estimated Value: ${signal['estimated_value']:,.2f}")
                print(f"      Reason: {signal['reason']}")
    except Exception as e:
        print(f"[FAIL] Failed to analyze trading opportunities: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Simulate order execution (dry run)
    print("\n[TEST 5] Simulate Order Execution (DRY RUN)")
    print("-" * 80)

    try:
        if signals['buy']:
            # Execute first buy signal
            test_signal = signals['buy'][0]
            result = engine._execute_order(test_signal)

            print(f"[PASS] Order execution simulated:")
            print(f"  - Success: {result['success']}")
            print(f"  - Symbol: {result['symbol']}")
            print(f"  - Action: {result['action']}")
            print(f"  - Quantity: {result['qty']}")
            print(f"  - Price: ${result['price']:.2f}")
            print(f"  - Order ID: {result['order_id']}")
            print(f"  - Message: {result['message']}")
    except Exception as e:
        print(f"[FAIL] Failed to execute order: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Fund management status
    print("\n[TEST 6] Fund Management Status")
    print("-" * 80)

    try:
        fund_status = engine.get_fund_management_status()
        print(f"[PASS] Fund management status retrieved:")
        print(f"  - Raw Buying Power: ${fund_status['raw_buying_power']:,.2f}")
        print(f"  - Safe Buying Power: ${fund_status['safe_buying_power']:,.2f}")
        print(f"  - Cash Reserve: ${fund_status['cash_reserve']:,.2f}")
        print(f"  - Safety Factor: {fund_status['safety_factor']:.1%}")
        print(f"  - Max Position Value: ${fund_status['max_position_value']:,.2f}")
        print(f"  - Max Position Percent: {fund_status['max_position_percent']:.1%}")
    except Exception as e:
        print(f"[FAIL] Failed to get fund management status: {e}")
        return False

    # Test 7: Execution quality report
    print("\n[TEST 7] Execution Quality Report")
    print("-" * 80)

    try:
        quality_report = engine.get_execution_quality_report()
        print(f"[PASS] Execution quality report retrieved:")
        print(f"  - Adaptive Execution Enabled: {quality_report.get('adaptive_execution_enabled', False)}")

        if quality_report.get('adaptive_execution_enabled'):
            print(f"  - Total Orders Executed: {quality_report.get('total_orders_executed', 0)}")
            print(f"  - Avg Implementation Shortfall: {quality_report.get('average_implementation_shortfall_bps', 0):.2f} bps")
            print(f"  - Avg Market Impact: {quality_report.get('average_market_impact_bps', 0):.2f} bps")
            print(f"  - Fill Rate: {quality_report.get('fill_rate', 0):.1%}")
            print(f"  - Cost Savings vs Naive: {quality_report.get('cost_savings_vs_naive_bps', 0):.2f} bps")
        else:
            print(f"  - Message: {quality_report.get('message', 'N/A')}")
    except Exception as e:
        print(f"[FAIL] Failed to get execution quality report: {e}")
        return False

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_adaptive_execution_integration()
    sys.exit(0 if success else 1)