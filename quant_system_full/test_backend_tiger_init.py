#!/usr/bin/env python3
"""
Test Backend Tiger Initialization
Debug why Tiger provider initialization might be failing in backend context
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

async def test_backend_tiger_init():
    """Test Tiger initialization in backend context"""

    print("Testing Backend Tiger Initialization")
    print("=" * 45)

    try:
        # Test 1: Import exactly like backend app.py does
        print("\n1. Importing Tiger provider like backend...")
        try:
            from tiger_data_provider_real import real_tiger_provider as tiger_provider
            print("   Import successful")
        except ImportError as e:
            print(f"   Import failed: {e}")
            return False

        # Test 2: Check initial state
        print(f"\n2. Initial availability: {tiger_provider.is_available()}")

        # Test 3: Initialize like backend startup does
        print("\n3. Initializing Tiger provider...")
        success = await tiger_provider.initialize()
        print(f"   Initialization result: {success}")
        print(f"   Final availability: {tiger_provider.is_available()}")

        if success:
            # Test 4: Try to get data
            print("\n4. Testing data retrieval...")
            try:
                positions = await tiger_provider.get_positions()
                print(f"   Positions count: {len(positions)}")

                portfolio = await tiger_provider.get_portfolio_summary()
                print(f"   Portfolio value: ${portfolio['total_value']:,.2f}")

                orders = await tiger_provider.get_orders(limit=3)
                print(f"   Orders count: {len(orders)}")

                print("   SUCCESS: All data retrieval working!")
                return True
            except Exception as e:
                print(f"   Data retrieval failed: {e}")
                return False
        else:
            print("   Tiger provider initialization failed")

            # Try to understand why
            print("\n5. Debugging initialization failure...")
            print(f"   TIGER_ID: {os.getenv('TIGER_ID', 'NOT SET')}")
            print(f"   ACCOUNT: {os.getenv('ACCOUNT', 'NOT SET')}")
            private_key_path = os.getenv('PRIVATE_KEY_PATH', '')
            print(f"   PRIVATE_KEY_PATH: {private_key_path}")
            if private_key_path:
                print(f"   Private key exists: {os.path.exists(private_key_path)}")

            return False

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_backend_tiger_init())

    if success:
        print("\nTiger integration should be working in backend!")
    else:
        print("\nTiger integration has issues that need to be resolved.")