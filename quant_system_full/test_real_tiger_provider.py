#!/usr/bin/env python3
"""
Test Real Tiger Data Provider
Verify that the real Tiger data provider works correctly
"""

import os
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'dashboard' / 'backend'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

async def test_real_tiger_provider():
    """Test the real Tiger data provider"""
    print("Testing Real Tiger Data Provider")
    print("=" * 50)

    try:
        from tiger_data_provider_real import real_tiger_provider

        # Initialize the provider
        print("\n1. Initializing Tiger provider...")
        success = await real_tiger_provider.initialize()
        print(f"   Initialization: {'SUCCESS' if success else 'FAILED'}")

        if not success:
            print("   Cannot continue without successful initialization")
            return False

        # Test positions
        print("\n2. Testing get_positions()...")
        positions = await real_tiger_provider.get_positions()
        print(f"   Found {len(positions)} positions")
        for pos in positions:
            print(f"   - {pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f}")
            print(f"     Market Value: ${pos['market_value']:,.2f}, P&L: ${pos['unrealized_pnl']:+,.2f}")

        # Test portfolio summary
        print("\n3. Testing get_portfolio_summary()...")
        portfolio = await real_tiger_provider.get_portfolio_summary()
        print(f"   Total Value: ${portfolio['total_value']:,.2f}")
        print(f"   Cash Balance: ${portfolio['cash_balance']:,.2f}")
        print(f"   Buying Power: ${portfolio['buying_power']:,.2f}")
        print(f"   Total P&L: ${portfolio['total_pnl']:+,.2f} ({portfolio['total_pnl_percent']:+.2f}%)")
        print(f"   Positions Count: {portfolio['positions_count']}")

        # Test orders
        print("\n4. Testing get_orders()...")
        orders = await real_tiger_provider.get_orders(limit=5)
        print(f"   Found {len(orders)} recent orders")
        for order in orders:
            print(f"   - {order['symbol']} {order['side'].upper()} {order['quantity']} @ {order['status']}")
            if order['avg_fill_price']:
                print(f"     Fill Price: ${order['avg_fill_price']:.2f}")

        print("\n" + "=" * 50)
        print("Real Tiger Data Provider test completed successfully!")
        return True

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_real_tiger_provider())