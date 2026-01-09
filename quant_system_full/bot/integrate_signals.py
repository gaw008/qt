#!/usr/bin/env python3
"""
集成交易信号到系统交易引擎
"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.append('.')
from quick_data_fix import process_stock_selection
from position_manager import PositionManager

def integrate_trading_signals():
    """将生成的交易信号集成到系统中"""
    
    # 初始化仓位管理器
    position_manager = PositionManager()
    
    # 显示账户摘要
    summary = position_manager.get_portfolio_summary()
    print(f"=== Account Summary ===")
    print(f"Available Cash: ${summary['account_balance']:,.2f}")
    print(f"Current Exposure: {summary['current_exposure_pct']:.1f}%")
    print(f"Available for Trading: ${summary['available_for_trading']:,.2f}")
    
    # 读取选股结果
    selection_file = Path('../dashboard/state/selection_results.json')
    with open(selection_file, 'r') as f:
        selection_data = json.load(f)

    selected_stocks = [stock['symbol'] for stock in selection_data['selected_stocks']]
    print(f"\nProcessing {len(selected_stocks)} selected stocks...")

    # 生成交易信号（处理前5只避免超时）
    signals = process_stock_selection(selected_stocks[:5])

    # 准备交易记录
    trades = []
    new_orders = []

    for signal in signals:
        if signal['signal'] in ['BUY', 'SELL']:
            # 计算智能仓位大小
            position_info = position_manager.calculate_position_size(
                signal['symbol'],
                signal['current_price'], 
                signal['signal'],
                signal['score']
            )
            
            # 如果计算出的数量为0，跳过此订单
            if position_info['quantity'] == 0:
                print(f"SKIP {signal['symbol']}: {position_info['reason']}")
                continue
            
            # 创建交易记录
            trade = {
                'symbol': signal['symbol'],
                'signal': signal['signal'],
                'score': signal['score'],
                'price': signal['current_price'],
                'quantity': position_info['quantity'],
                'trade_value': position_info['trade_value'],
                'allocation_pct': position_info['allocation_pct'],
                'timestamp': datetime.now().isoformat(),
                'reason': signal['reason'],
                'position_reason': position_info['reason'],
                'status': 'SIGNAL_GENERATED'
            }
            trades.append(trade)
            
            # 创建订单记录
            order_id = f"smart_{signal['symbol']}_{int(datetime.now().timestamp())}"
            order = {
                'order_id': order_id,
                'symbol': signal['symbol'],
                'order_type': 'MARKET',
                'side': signal['signal'],
                'quantity': position_info['quantity'],
                'price': signal['current_price'],
                'trade_value': position_info['trade_value'],
                'allocation_pct': position_info['allocation_pct'],
                'status': 'SIGNAL_READY',
                'created_time': datetime.now().isoformat(),
                'source': 'smart_position_manager'
            }
            new_orders.append(order)

    print(f"\n=== Generated {len(trades)} Trading Signals ===")
    for trade in trades:
        print(f"{trade['symbol']}: {trade['signal']} {trade['quantity']} shares @ ${trade['price']:.2f}")
        print(f"  Score: {trade['score']:.1f} | Value: ${trade['trade_value']:.2f} | Allocation: {trade['allocation_pct']:.1f}%")

    # 写入交易文件
    trades_file = Path('../dashboard/state/trades.json')
    with open(trades_file, 'w') as f:
        json.dump({'trades': trades}, f, indent=2)
    
    # 读取现有订单并添加新订单
    orders_file = Path('../dashboard/state/orders.json')
    try:
        with open(orders_file, 'r') as f:
            orders_data = json.load(f)
        existing_orders = orders_data.get('orders', [])
    except:
        existing_orders = []
    
    # 添加新订单
    all_orders = existing_orders + new_orders
    with open(orders_file, 'w') as f:
        json.dump({'orders': all_orders}, f, indent=2)

    # 更新系统状态
    status_file = Path('../dashboard/state/status.json')
    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
    except:
        status_data = {}
    
    # 更新关键状态
    if trades:
        status_data['last_signal'] = trades[0]['signal']
        status_data['reason'] = f"Generated {len(trades)} trading signals via quick_data_fix"
        status_data['timestamp'] = datetime.now().isoformat()
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

    print(f"\n=== System Integration Complete ===")
    print(f"- Updated trades.json: {len(trades)} new trades")
    print(f"- Updated orders.json: {len(new_orders)} new orders")
    print(f"- Updated status: last_signal = {trades[0]['signal'] if trades else 'HOLD'}")
    
    return trades, new_orders

if __name__ == "__main__":
    integrate_trading_signals()