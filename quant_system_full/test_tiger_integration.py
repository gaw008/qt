#!/usr/bin/env python3
"""
Test Tiger Integration in Backend
Debug why Tiger provider might not be working in backend
"""

import os
import sys
import asyncio
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'dashboard' / 'backend'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

async def debug_tiger_integration():
    """Debug the Tiger integration"""

    print("Debugging Tiger Integration")
    print("=" * 40)

    try:
        # Test 1: Direct Tiger SDK import
        print("\n1. Testing Tiger SDK import...")
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.common.util.signature_utils import read_private_key
            from tigeropen.trade.trade_client import TradeClient
            print("   Tiger SDK import: SUCCESS")
        except ImportError as e:
            print(f"   Tiger SDK import: FAILED - {e}")
            return False

        # Test 2: Environment variables
        print("\n2. Checking environment variables...")
        tiger_id = os.getenv("TIGER_ID", "")
        account = os.getenv("ACCOUNT", "")
        private_key_path = os.getenv("PRIVATE_KEY_PATH", "")

        print(f"   TIGER_ID: {'SET' if tiger_id else 'NOT SET'}")
        print(f"   ACCOUNT: {'SET' if account else 'NOT SET'}")
        print(f"   PRIVATE_KEY_PATH: {'SET' if private_key_path else 'NOT SET'}")

        if private_key_path:
            print(f"   Private key file exists: {os.path.exists(private_key_path)}")

        # Test 3: Real Tiger provider import
        print("\n3. Testing real Tiger provider import...")
        try:
            from tiger_data_provider_real import real_tiger_provider
            print("   Real Tiger provider import: SUCCESS")
        except ImportError as e:
            print(f"   Real Tiger provider import: FAILED - {e}")
            return False

        # Test 4: Initialize provider
        print("\n4. Testing provider initialization...")
        success = await real_tiger_provider.initialize()
        print(f"   Provider initialization: {'SUCCESS' if success else 'FAILED'}")
        print(f"   Provider available: {real_tiger_provider.is_available()}")

        if success:
            # Test 5: Get real data
            print("\n5. Testing real data retrieval...")
            positions = await real_tiger_provider.get_positions()
            print(f"   Real positions count: {len(positions)}")

            portfolio = await real_tiger_provider.get_portfolio_summary()
            print(f"   Real portfolio value: ${portfolio['total_value']:,.2f}")

        # Test 6: Check backend app integration
        print("\n6. Testing backend app integration...")
        try:
            # Import the way backend app does it
            from tiger_data_provider_real import real_tiger_provider as tiger_provider
            print("   Backend import: SUCCESS")
            print(f"   Backend provider available: {tiger_provider.is_available()}")
        except ImportError as e:
            print(f"   Backend import: FAILED - {e}")

        return True

    except Exception as e:
        print(f"Debug failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(debug_tiger_integration())