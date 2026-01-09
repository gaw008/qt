#!/usr/bin/env python3
"""
Test Backend App Directly
Test the actual backend app's Tiger provider to see what's happening
"""

import os
import sys
import asyncio
from pathlib import Path

# Setup paths exactly like backend does
sys.path.append(str(Path(__file__).parent / 'dashboard' / 'backend'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

async def test_backend_app_tiger():
    """Test the backend app's Tiger provider directly"""

    print("Testing Backend App Tiger Provider")
    print("=" * 40)

    try:
        # Import exactly like the backend app.py does
        print("\n1. Testing backend imports...")

        try:
            from state_manager import read_status, write_status, set_kill, is_killed, read_log_tail, write_daily_report
            from tiger_data_provider_real import real_tiger_provider as tiger_provider
            STATE_MANAGER_AVAILABLE = True
            print("   Backend imports: SUCCESS")
            print(f"   STATE_MANAGER_AVAILABLE: {STATE_MANAGER_AVAILABLE}")
        except ImportError as e:
            print(f"   Backend imports: FAILED - {e}")
            STATE_MANAGER_AVAILABLE = False

            # Try fallback
            try:
                print("   Trying fallback import...")
                from tiger_data_provider_real import real_tiger_provider as tiger_provider
                print("   Fallback import: SUCCESS")
            except ImportError as e2:
                print(f"   Fallback import: FAILED - {e2}")
                return False

        # Test 2: Check if Tiger provider is the real one
        print(f"\n2. Tiger provider type: {type(tiger_provider)}")
        print(f"   Tiger provider available: {tiger_provider.is_available()}")

        # Test 3: Initialize if needed
        if not tiger_provider.is_available():
            print("\n3. Initializing Tiger provider...")
            success = await tiger_provider.initialize()
            print(f"   Initialization: {'SUCCESS' if success else 'FAILED'}")
            print(f"   Now available: {tiger_provider.is_available()}")

        # Test 4: Get some data
        if tiger_provider.is_available():
            print("\n4. Testing data retrieval...")

            positions = await tiger_provider.get_positions()
            print(f"   Positions: {len(positions)} found")
            for pos in positions:
                print(f"   - {pos['symbol']}: {pos['quantity']} @ ${pos['current_price']:.2f}")

            portfolio = await tiger_provider.get_portfolio_summary()
            print(f"   Portfolio value: ${portfolio['total_value']:,.2f}")

            orders = await tiger_provider.get_orders(limit=2)
            print(f"   Orders: {len(orders)} found")
            for order in orders:
                print(f"   - {order['symbol']} {order['side'].upper()} {order['quantity']} @ {order['status']}")

        # Test 5: Check if this matches what API should return
        print("\n5. Summary:")
        if tiger_provider.is_available():
            positions = await tiger_provider.get_positions()
            if any(pos['symbol'] in ['C', 'CAT'] for pos in positions):
                print("   SUCCESS: Getting real Tiger data (C, CAT positions)")
                return True
            else:
                print("   WARNING: Getting different data than expected")
                return False
        else:
            print("   FAILED: Tiger provider not available")
            return False

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_backend_app_tiger())

    if success:
        print("\nThe backend should be returning real Tiger data!")
        print("If API is still returning mock data, there might be caching or server restart needed.")
    else:
        print("\nThere's an issue with the backend Tiger integration.")