#!/usr/bin/env python3
"""
Real Tiger Account Data
显示真实的Tiger账户持仓和余额信息
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

def get_real_tiger_data():
    """获取并显示真实的Tiger账户数据"""

    print("💰 您的真实Tiger账户信息")
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

        # 1. 账户基本信息
        print(f"\n📊 账户基本信息:")
        print(f"   Tiger ID: {tiger_id}")
        print(f"   账户号码: {account}")

        # 2. 账户资产信息
        print(f"\n💰 账户资产:")
        try:
            assets = trade_client.get_assets(account=account)
            if assets and len(assets) > 0:
                asset = assets[0]
                summary = asset.summary

                print(f"   💵 净资产: ${summary.net_liquidation:,.2f}")
                print(f"   💸 现金余额: ${summary.cash:,.2f}")
                print(f"   💳 购买力: ${summary.buying_power:,.2f}")

                # 从segments获取更详细信息
                if 'S' in asset.segments:
                    seg = asset.segments['S']
                    print(f"   📈 可用资金: ${seg.available_funds:,.2f}")
                    print(f"   📊 股票市值: ${seg.gross_position_value:,.2f}")
                    print(f"   💎 超额流动性: ${seg.excess_liquidity:,.2f}")

                print(f"   📈 已实现盈亏: ${summary.realized_pnl:,.2f}")
                print(f"   📊 未实现盈亏: ${summary.unrealized_pnl:,.2f}")

                # 计算总盈亏比例
                if summary.net_liquidation > 0:
                    total_pnl = summary.realized_pnl + summary.unrealized_pnl
                    total_pnl_percent = (total_pnl / summary.net_liquidation) * 100
                    print(f"   📊 总盈亏比例: {total_pnl_percent:+.2f}%")
            else:
                print("   ❌ 无法获取资产信息")
        except Exception as e:
            print(f"   ❌ 资产信息错误: {str(e)}")

        # 3. 持仓信息
        print(f"\n📈 当前持仓:")
        try:
            positions = trade_client.get_positions(account=account)
            if positions and len(positions) > 0:
                total_market_value = 0
                total_unrealized_pnl = 0

                for i, pos in enumerate(positions, 1):
                    # 获取股票代码
                    symbol = pos.contract.symbol if hasattr(pos.contract, 'symbol') else str(pos.contract)

                    # 计算市值
                    market_value = pos.market_value
                    total_market_value += market_value
                    total_unrealized_pnl += pos.unrealized_pnl

                    print(f"\n   📍 {i}. {symbol}")
                    print(f"      数量: {pos.quantity:,}")
                    print(f"      平均成本: ${pos.average_cost:.4f}")
                    print(f"      当前价格: ${pos.market_price:.2f}")
                    print(f"      市值: ${market_value:,.2f}")
                    print(f"      未实现盈亏: ${pos.unrealized_pnl:+,.2f}")
                    print(f"      盈亏比例: {pos.unrealized_pnl_percent:+.2%}")
                    print(f"      今日盈亏: ${pos.today_pnl:+,.2f}")

                print(f"\n   📊 持仓汇总:")
                print(f"      总持仓数: {len(positions)}")
                print(f"      总市值: ${total_market_value:,.2f}")
                print(f"      总未实现盈亏: ${total_unrealized_pnl:+,.2f}")
            else:
                print("   📭 当前无持仓")
        except Exception as e:
            print(f"   ❌ 持仓信息错误: {str(e)}")

        # 4. 最近订单
        print(f"\n📋 最近订单 (最新5笔):")
        try:
            orders = trade_client.get_orders(account=account)
            if orders and len(orders) > 0:
                recent_orders = orders[:5]

                for i, order in enumerate(recent_orders, 1):
                    symbol = order.contract.symbol if hasattr(order.contract, 'symbol') else str(order.contract)
                    action_cn = "买入" if order.action == "BUY" else "卖出"
                    order_time = datetime.fromtimestamp(order.order_time / 1000).strftime("%Y-%m-%d %H:%M:%S")

                    print(f"\n   📄 {i}. 订单 #{order.id}")
                    print(f"      股票: {symbol}")
                    print(f"      方向: {action_cn}")
                    print(f"      数量: {order.quantity:,}")
                    print(f"      已成交: {order.filled:,}")
                    print(f"      剩余: {order.remaining:,}")
                    print(f"      平均成交价: ${order.avg_fill_price:.4f}" if order.avg_fill_price else "      平均成交价: N/A")
                    print(f"      订单类型: {order.order_type}")
                    print(f"      状态: {order.status}")
                    print(f"      时间: {order_time}")
            else:
                print("   📭 无订单历史")
        except Exception as e:
            print(f"   ❌ 订单信息错误: {str(e)}")

        print("\n" + "=" * 60)
        print("✅ 真实Tiger账户数据获取完成")
        print(f"📅 查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_real_tiger_data()