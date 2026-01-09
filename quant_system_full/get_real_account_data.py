#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

def get_real_tiger_account_data():
    """Get real account data from Tiger API"""

    print("=== Tiger API Real Account Data ===")
    print()

    try:
        # Import Tiger modules
        from tigeropen.common.consts import Language
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient

        # Configuration
        config = TigerOpenClientConfig()
        config.private_key = os.getenv('PRIVATE_KEY_PATH')
        config.tiger_id = os.getenv('TIGER_ID')
        config.account = os.getenv('ACCOUNT')
        config.language = Language.en_US
        config.timezone = 'US/Eastern'

        # Create client
        trade_client = TradeClient(config)
        print("Tiger client initialized successfully")

        # Get account assets
        print("\nFetching real account data...")
        assets = trade_client.get_assets()

        if assets and len(assets) > 0:
            asset = assets[0]

            print("\n*** REAL TIGER ACCOUNT DATA ***")
            print(f"Currency: {asset.currency}")
            print(f"Available Cash: ${asset.cash:,.2f}")
            print(f"Buying Power: ${asset.buying_power:,.2f}")
            print(f"Net Liquidation: ${asset.net_liquidation:,.2f}")
            print(f"Equity with Loan: ${asset.equity_with_loan:,.2f}")

            # Calculate real available capital
            buying_power = asset.buying_power
            safety_factor = float(os.getenv('TRADING_SAFETY_FACTOR', '0.8'))
            min_reserve = float(os.getenv('MIN_CASH_RESERVE', '5000'))

            safe_capital = max(0, (buying_power - min_reserve) * safety_factor)

            print(f"\n*** AVAILABLE CAPITAL CALCULATION ***")
            print(f"Tiger Buying Power: ${buying_power:,.2f}")
            print(f"Minus Safety Reserve: ${min_reserve:,.2f}")
            print(f"Apply Safety Factor ({safety_factor:.1%}): ${safe_capital:,.2f}")
            print(f"\n*** REAL AVAILABLE CAPITAL: ${safe_capital:,.2f} ***")

            return {
                'cash': asset.cash,
                'buying_power': buying_power,
                'net_liquidation': asset.net_liquidation,
                'available_capital': safe_capital
            }
        else:
            print("No asset data available")
            return None

    except Exception as e:
        print(f"Tiger API Error: {e}")
        print("\nNote: This may be due to:")
        print("1. Private key configuration issues")
        print("2. API rate limits or permissions")
        print("3. Market hours restrictions")
        print("4. Network connectivity")
        return None

if __name__ == "__main__":
    get_real_tiger_account_data()