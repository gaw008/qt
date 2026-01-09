#!/usr/bin/env python3
"""
Test Tiger Account Data
Test real Tiger account connection for backend integration
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

def test_tiger_connection():
    """Test Tiger API connection for backend integration"""

    print("Tiger Account Data Test")
    print("=" * 60)

    try:
        # Import Tiger SDK
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.common.util.signature_utils import read_private_key
        from tigeropen.trade.trade_client import TradeClient

        # Configuration
        props_dir = str(Path(__file__).parent / 'props')
        cfg = TigerOpenClientConfig(props_path=props_dir)

        # Environment variables
        tiger_id = os.getenv("TIGER_ID", "")
        account = os.getenv("ACCOUNT", "")
        private_key_path = os.getenv("PRIVATE_KEY_PATH", "")

        if tiger_id:
            cfg.tiger_id = tiger_id
        if account:
            cfg.account = account
        if private_key_path and os.path.exists(private_key_path):
            cfg.private_key = read_private_key(private_key_path)

        cfg.timezone = "US/Eastern"
        cfg.language = "en_US"

        # Create client
        trade_client = TradeClient(cfg)

        # 1. Account basic info
        print(f"\nAccount Info:")
        print(f"   Tiger ID: {tiger_id}")
        print(f"   Account Number: {account}")

        # 2. Account assets
        print(f"\nAccount Assets:")
        try:
            assets = trade_client.get_assets(account=account)
            if assets and len(assets) > 0:
                asset = assets[0]
                summary = asset.summary

                print(f"   Net Asset: ${summary.net_liquidation:,.2f}")
                print(f"   Cash Balance: ${summary.cash:,.2f}")
                print(f"   Buying Power: ${summary.buying_power:,.2f}")
                print(f"   Realized P&L: ${summary.realized_pnl:,.2f}")
                print(f"   Unrealized P&L: ${summary.unrealized_pnl:,.2f}")

                # Calculate total P&L percentage
                if summary.net_liquidation > 0:
                    total_pnl = summary.realized_pnl + summary.unrealized_pnl
                    total_pnl_percent = (total_pnl / summary.net_liquidation) * 100
                    print(f"   Total P&L Percentage: {total_pnl_percent:+.2f}%")
            else:
                print("   No asset information available")
        except Exception as e:
            print(f"   Asset info error: {str(e)}")

        # 3. Current positions
        print(f"\nCurrent Positions:")
        try:
            positions = trade_client.get_positions(account=account)
            if positions and len(positions) > 0:
                total_market_value = 0
                total_unrealized_pnl = 0

                for i, pos in enumerate(positions, 1):
                    # Get stock symbol
                    symbol = pos.contract.symbol if hasattr(pos.contract, 'symbol') else str(pos.contract)

                    # Calculate market value
                    market_value = pos.market_value
                    total_market_value += market_value
                    total_unrealized_pnl += pos.unrealized_pnl

                    print(f"\n   {i}. {symbol}")
                    print(f"      Quantity: {pos.quantity:,}")
                    print(f"      Average Cost: ${pos.average_cost:.4f}")
                    print(f"      Current Price: ${pos.market_price:.2f}")
                    print(f"      Market Value: ${market_value:,.2f}")
                    print(f"      Unrealized P&L: ${pos.unrealized_pnl:+,.2f}")
                    print(f"      P&L Percentage: {pos.unrealized_pnl_percent:+.2%}")

                print(f"\n   Position Summary:")
                print(f"      Total Positions: {len(positions)}")
                print(f"      Total Market Value: ${total_market_value:,.2f}")
                print(f"      Total Unrealized P&L: ${total_unrealized_pnl:+,.2f}")
            else:
                print("   No current positions")
        except Exception as e:
            print(f"   Position info error: {str(e)}")

        # 4. Recent orders (latest 3)
        print(f"\nRecent Orders (Latest 3):")
        try:
            orders = trade_client.get_orders(account=account)
            if orders and len(orders) > 0:
                recent_orders = orders[:3]

                for i, order in enumerate(recent_orders, 1):
                    symbol = order.contract.symbol if hasattr(order.contract, 'symbol') else str(order.contract)
                    action_text = "Buy" if order.action == "BUY" else "Sell"
                    order_time = datetime.fromtimestamp(order.order_time / 1000).strftime("%Y-%m-%d %H:%M:%S")

                    print(f"\n   {i}. Order #{order.id}")
                    print(f"      Symbol: {symbol}")
                    print(f"      Action: {action_text}")
                    print(f"      Quantity: {order.quantity:,}")
                    print(f"      Filled: {order.filled:,}")
                    print(f"      Status: {order.status}")
                    print(f"      Time: {order_time}")
            else:
                print("   No order history")
        except Exception as e:
            print(f"   Order info error: {str(e)}")

        print("\n" + "=" * 60)
        print("Tiger API connection test completed successfully")
        print(f"Query time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True

    except Exception as e:
        print(f"System error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tiger_connection()