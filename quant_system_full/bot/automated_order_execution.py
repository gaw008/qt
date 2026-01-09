#!/usr/bin/env python3
"""
Automated Order Execution Integration
将智能订单执行集成到自动化交易系统中
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.append('.')

def check_and_execute_pending_orders():
    """
    检查并执行待处理的订单
    这个函数将被集成到 real_trading_task 中
    """
    
    try:
        # 导入执行模块
        from position_manager import PositionManager
        from execution_tiger import create_tiger_execution_engine
        from tradeup_client import build_clients
        
        # 检查是否有待处理订单
        orders_file = Path('../dashboard/state/orders.json')
        if not orders_file.exists():
            return {"status": "no_orders_file", "executed": 0}
        
        with open(orders_file, 'r') as f:
            data = json.load(f)
        
        orders = data.get('orders', [])
        pending_orders = [order for order in orders if order.get('status') == 'SIGNAL_READY']
        
        if not pending_orders:
            return {"status": "no_pending_orders", "executed": 0}
        
        print(f"[AUTO EXECUTION] Found {len(pending_orders)} pending orders")
        
        # 初始化Tiger API客户端
        try:
            quote_client, trade_client = build_clients()
            engine = create_tiger_execution_engine(quote_client, trade_client)
            
            if not engine:
                return {"status": "tiger_api_error", "executed": 0, "error": "Failed to create execution engine"}
        except Exception as e:
            return {"status": "tiger_api_error", "executed": 0, "error": str(e)}
        
        # 执行订单
        executed_count = 0
        failed_count = 0
        
        for i, order in enumerate(pending_orders):
            try:
                symbol = order['symbol']
                side = order['side']  
                quantity = order['quantity']
                order_type = order.get('order_type', 'MARKET')
                price = order.get('price')
                
                print(f"[AUTO EXECUTION] Executing {symbol} {side} {quantity} shares")
                
                # 执行订单
                if order_type == 'MARKET':
                    result = engine.place_market_order(symbol, side, quantity)
                elif order_type == 'LIMIT' and price:
                    result = engine.place_limit_order(symbol, side, quantity, price)
                else:
                    print(f"[AUTO EXECUTION] Unsupported order type: {order_type}")
                    continue
                
                # 更新订单状态
                for j, full_order in enumerate(orders):
                    if full_order['order_id'] == order['order_id']:
                        if result.success:
                            orders[j]['status'] = 'EXECUTED'
                            orders[j]['tiger_order_id'] = result.order_id
                            orders[j]['executed_time'] = datetime.now().isoformat()
                            executed_count += 1
                            print(f"[AUTO EXECUTION] SUCCESS: {symbol} order executed (ID: {result.order_id})")
                        else:
                            orders[j]['status'] = 'FAILED'
                            orders[j]['error_message'] = result.error
                            orders[j]['failed_time'] = datetime.now().isoformat()
                            failed_count += 1
                            print(f"[AUTO EXECUTION] FAILED: {symbol} order failed: {result.error}")
                        break
                
                # 小延迟避免API限制
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[AUTO EXECUTION] ERROR processing order {order.get('order_id', 'unknown')}: {e}")
                failed_count += 1
        
        # 保存更新的订单状态
        with open(orders_file, 'w') as f:
            json.dump({'orders': orders}, f, indent=2)
        
        return {
            "status": "completed",
            "executed": executed_count,
            "failed": failed_count,
            "total_processed": len(pending_orders)
        }
        
    except Exception as e:
        return {"status": "error", "executed": 0, "error": str(e)}

def log_execution_to_trades(executed_orders):
    """将执行结果记录到trades.json"""
    try:
        trades_file = Path('../dashboard/state/trades.json')
        
        # 读取现有交易记录
        if trades_file.exists():
            with open(trades_file, 'r') as f:
                data = json.load(f)
            trades = data.get('trades', [])
        else:
            trades = []
        
        # 添加执行记录
        for order in executed_orders:
            if order.get('status') == 'EXECUTED':
                execution_record = {
                    'symbol': order['symbol'],
                    'signal': order['side'],
                    'quantity': order['quantity'],
                    'price': order.get('price', 0.0),
                    'trade_value': order.get('trade_value', 0.0),
                    'tiger_order_id': order.get('tiger_order_id'),
                    'timestamp': order.get('executed_time'),
                    'status': 'EXECUTED',
                    'source': 'automated_execution'
                }
                trades.append(execution_record)
        
        # 保存更新的交易记录
        with open(trades_file, 'w') as f:
            json.dump({'trades': trades}, f, indent=2)
            
    except Exception as e:
        print(f"[AUTO EXECUTION] Error logging to trades: {e}")

def main():
    """独立测试入口"""
    print("=== Automated Order Execution Test ===")
    result = check_and_execute_pending_orders()
    print(f"Execution result: {result}")

if __name__ == "__main__":
    main()