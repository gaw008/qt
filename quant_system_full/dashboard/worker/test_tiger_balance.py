#!/usr/bin/env python3
"""
Test Tiger API Balance Retrieval

This script tests fetching the real account balance from Tiger API.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
bot_path = Path(__file__).parent.parent.parent / 'bot'
sys.path.append(str(bot_path))

try:
    from tradeup_client import build_clients
    from execution_tiger import create_tiger_execution_engine
    from account_balance_manager import get_balance_manager
    
    print("Testing Tiger API balance retrieval...")
    print("=" * 50)
    
    # Build Tiger clients
    print("1. Building Tiger API clients...")
    quote_client, trade_client = build_clients()
    
    if not quote_client or not trade_client:
        print("FAILED: Could not build Tiger API clients")
        sys.exit(1)
    
    print("SUCCESS: Tiger API clients built successfully")
    
    # Create execution engine
    print("2. Creating execution engine...")
    execution_engine = create_tiger_execution_engine(quote_client, trade_client)
    
    if not execution_engine:
        print("FAILED: Could not create execution engine")
        sys.exit(1)
    
    print("SUCCESS: Execution engine created successfully")
    
    # Get account assets
    print("3. Getting account assets from Tiger API...")
    try:
        assets = execution_engine.get_account_assets()
        print(f"Assets response: {assets}")
        
        if assets:
            if hasattr(assets, 'summary'):
                cash_balance = getattr(assets.summary, 'cash_balance', None)
                net_liquidation = getattr(assets.summary, 'net_liquidation', None)
                
                print(f"Cash Balance: {cash_balance}")
                print(f"Net Liquidation: {net_liquidation}")
            else:
                print(f"Assets structure: {dir(assets)}")
        else:
            print("FAILED: No assets returned from Tiger API")
            
    except Exception as e:
        print(f"ERROR getting assets: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    # Test balance manager sync
    print("4. Testing balance manager sync...")
    try:
        balance_manager = get_balance_manager()
        
        print(f"Current balance in manager: ${balance_manager.get_available_balance():.2f}")
        
        success = balance_manager.update_balance_from_tiger_api(execution_engine)
        
        if success:
            new_balance = balance_manager.get_available_balance()
            print(f"SUCCESS: Balance sync successful: ${new_balance:.2f}")
        else:
            print("FAILED: Balance sync failed")
            
    except Exception as e:
        print(f"ERROR in balance manager sync: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    print("=" * 50)
    print("Tiger API balance test completed")
    
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Make sure all required modules are available")
except Exception as e:
    print(f"UNEXPECTED ERROR: {e}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")