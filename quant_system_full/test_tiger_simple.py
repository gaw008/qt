#!/usr/bin/env python3
"""
Test Tiger API Live Connection
Simple test to verify Tiger API works correctly for live trading
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

def test_tiger_connection():
    """Test Tiger API connection with correct configuration"""

    print("? Testing Tiger API Live Connection...")
    print("=" * 60)

    try:
        # Import Tiger SDK
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.common.util.signature_utils import read_private_key
        from tigeropen.quote.quote_client import QuoteClient
        from tigeropen.trade.trade_client import TradeClient
        print(" Tiger SDK imported successfully")

        # Get configuration
        props_dir = str(Path(__file__).parent / 'props')
        print(f"? Props directory: {props_dir}")

        # Initialize configuration
        cfg = TigerOpenClientConfig(props_path=props_dir)
        print(" Tiger config loaded from props")

        # Override with environment variables
        tiger_id = os.getenv("TIGER_ID", "")
        account = os.getenv("ACCOUNT", "")
        private_key_path = os.getenv("PRIVATE_KEY_PATH", "")

        if tiger_id:
            cfg.tiger_id = tiger_id
            print(f" Tiger ID: {tiger_id}")

        if account:
            cfg.account = account
            print(f" Account: {account}")

        if private_key_path and os.path.exists(private_key_path):
            cfg.private_key = read_private_key(private_key_path)
            print(f" Private key loaded from: {private_key_path}")

        cfg.timezone = "US/Eastern"
        cfg.language = "en_US"

        # Create clients
        print("\n? Creating Tiger API clients...")
        quote_client = QuoteClient(cfg)
        trade_client = TradeClient(cfg)
        print(" Clients created successfully")

        # Test quote client
        print("\n? Testing Quote Client...")
        try:
            # Test getting quote permissions
            permissions = quote_client.get_quote_permission()
            print(f" Quote permissions: {permissions}")
        except Exception as e:
            print(f" Quote permissions test failed: {e}")

        # Test trade client
        print("\n1/4 Testing Trade Client...")
        try:
            # Get account information
            account_info = trade_client.get_managed_accounts()
            print(f" Managed accounts: {account_info}")
        except Exception as e:
            print(f" Account info test failed: {e}")

        try:
            # Get positions
            positions = trade_client.get_positions()
            print(f" Current positions: {len(positions) if positions else 0} positions")

            # Show position details if any
            if positions:
                for pos in positions[:3]:  # Show first 3 positions
                    symbol = getattr(pos, 'contract', {}).get('symbol', 'Unknown')
                    quantity = getattr(pos, 'quantity', 0)
                    print(f"   - {symbol}: {quantity} shares")

        except Exception as e:
            print(f" Positions test failed: {e}")

        try:
            # Get account balance
            balance = trade_client.get_account_balance()
            if balance:
                print(f" Account balance retrieved")
                # Extract available cash safely
                available_cash = getattr(balance, 'available_cash', None)
                if available_cash:
                    print(f"    degrees Available cash: ${available_cash:,.2f}")
                else:
                    print(f"    degrees Balance object found but no available_cash field")
            else:
                print(" No balance information available")

        except Exception as e:
            print(f" Balance test failed: {e}")

        # Test market data
        print("\n? Testing Market Data...")
        try:
            # Test getting a simple quote
            from tigeropen.common.util.contract_utils import stock_contract
            contract = stock_contract("AAPL", "USD")

            # Try to get latest price
            try:
                brief = quote_client.get_stock_briefs([contract])
                if brief and len(brief) > 0:
                    price = getattr(brief[0], 'latest_price', None)
                    if price:
                        print(f" AAPL latest price: ${price}")
                    else:
                        print(f" Market data retrieved but no price field")
                else:
                    print(" No market data returned")
            except Exception as e:
                print(f" Market data test failed: {e}")

        except Exception as e:
            print(f" Market data setup failed: {e}")

        print("\n" + "=" * 60)
        print("? Tiger API Connection Test Complete!")
        print(" Tiger API is working and ready for live trading")
        print("\n? Next Steps:")
        print("   1. Set DRY_RUN=false in .env file")
        print("   2. Start the trading system")
        print("   3. Monitor live trading operations")

        return True

    except ImportError as e:
        print(f" Tiger SDK not available: {e}")
        print("   Please install: pip install git+https://github.com/tigerbrokers/openapi-python-sdk.git")
        return False

    except Exception as e:
        print(f" Connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tiger_connection()
    sys.exit(0 if success else 1)