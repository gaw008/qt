#!/usr/bin/env python3
"""
Signal Execution Engine
Reads trading signals from dashboard state files and executes them as real orders via Tiger API
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.append('.')
sys.path.append('..')

# Import Tiger API components
from execution_tiger import create_tiger_execution_engine
from tradeup_client import build_clients

def load_pending_orders():
    """Load orders with SIGNAL_READY status from orders.json"""
    orders_file = Path('../dashboard/state/orders.json')
    
    if not orders_file.exists():
        print("No orders.json file found")
        return []
    
    try:
        with open(orders_file, 'r') as f:
            data = json.load(f)
        
        orders = data.get('orders', [])
        # Filter orders that are ready for execution
        pending_orders = [order for order in orders if order.get('status') == 'SIGNAL_READY']
        
        print(f"Found {len(pending_orders)} orders ready for execution")
        return pending_orders
    except Exception as e:
        print(f"Error loading orders: {e}")
        return []

def execute_order(engine, order):
    """Execute a single order via Tiger API"""
    symbol = order['symbol']
    side = order['side']
    quantity = order['quantity']
    order_type = order.get('order_type', 'MARKET')
    price = order.get('price')
    
    print(f"\n=== Executing Order ===")
    print(f"Symbol: {symbol}")
    print(f"Side: {side}")
    print(f"Quantity: {quantity}")
    print(f"Order Type: {order_type}")
    if price:
        print(f"Price: ${price:.2f}")
    
    try:
        if order_type == 'MARKET':
            result = engine.place_market_order(symbol, side, quantity)
        elif order_type == 'LIMIT':
            if not price:
                print(f"ERROR: Limit order requires price for {symbol}")
                return False, "Missing price for limit order"
            result = engine.place_limit_order(symbol, side, quantity, price)
        else:
            print(f"ERROR: Unsupported order type: {order_type}")
            return False, f"Unsupported order type: {order_type}"
        
        if result.success:
            print(f"SUCCESS: Order executed - ID: {result.order_id}")
            print(f"Message: {result.message}")
            return True, result.order_id
        else:
            print(f"FAILED: {result.error}")
            return False, result.error
            
    except Exception as e:
        print(f"ERROR: Exception during order execution: {e}")
        return False, str(e)

def update_order_status(orders, order_index, status, tiger_order_id=None, error_msg=None):
    """Update order status in the orders list"""
    order = orders[order_index]
    order['status'] = status
    order['updated_time'] = datetime.now().isoformat()
    
    if tiger_order_id:
        order['tiger_order_id'] = tiger_order_id
    
    if error_msg:
        order['error_message'] = error_msg

def save_updated_orders(orders):
    """Save updated orders back to orders.json"""
    orders_file = Path('../dashboard/state/orders.json')
    
    try:
        with open(orders_file, 'w') as f:
            json.dump({'orders': orders}, f, indent=2)
        print("Updated orders.json with execution results")
    except Exception as e:
        print(f"Error saving orders: {e}")

def log_trade_execution(symbol, side, quantity, price, status, order_id=None):
    """Log executed trade to trades.json"""
    trades_file = Path('../dashboard/state/trades.json')
    
    try:
        # Load existing trades
        if trades_file.exists():
            with open(trades_file, 'r') as f:
                data = json.load(f)
            trades = data.get('trades', [])
        else:
            trades = []
        
        # Add new trade execution record
        trade_record = {
            'symbol': symbol,
            'signal': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'execution_timestamp': datetime.now().isoformat()
        }
        
        if order_id:
            trade_record['tiger_order_id'] = order_id
        
        trades.append(trade_record)
        
        # Save updated trades
        with open(trades_file, 'w') as f:
            json.dump({'trades': trades}, f, indent=2)
            
        print(f"Logged trade execution for {symbol}")
        
    except Exception as e:
        print(f"Error logging trade: {e}")

def execute_all_signals():
    """Main function to execute all pending signals"""
    print("=== Signal Execution Engine Started ===")
    
    # Check market status first
    try:
        from market_time import get_market_manager
        market_manager = get_market_manager()
        is_open = market_manager.is_market_open()
        phase = market_manager.get_market_phase()
        
        print(f"Market Status: {phase.value}")
        print(f"Market Open: {is_open}")
        
        if not is_open:
            print("\nWARNING: Market is currently closed")
            print("Tiger API may reject market orders when market is closed")
            print("The system will attempt to place orders anyway (they may be queued)")
            
    except Exception as e:
        print(f"Could not check market status: {e}")
    
    # Load pending orders
    orders = load_pending_orders()
    if not orders:
        print("No pending orders to execute")
        return
    
    # Initialize Tiger API clients
    try:
        print("Initializing Tiger API clients...")
        quote_client, trade_client = build_clients()
        engine = create_tiger_execution_engine(quote_client, trade_client)
        
        if not engine:
            print("ERROR: Failed to create Tiger execution engine")
            return
            
        print("Tiger API clients initialized successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize Tiger API: {e}")
        print("Make sure your Tiger API credentials are properly configured")
        return
    
    # Execute each order
    executed_count = 0
    failed_count = 0
    
    # Load all orders for updating
    all_orders = []
    orders_file = Path('../dashboard/state/orders.json')
    with open(orders_file, 'r') as f:
        all_orders = json.load(f)['orders']
    
    for i, order in enumerate(orders):
        print(f"\n--- Processing Order {i+1}/{len(orders)} ---")
        
        # Find this order in the full orders list
        order_index = None
        for j, full_order in enumerate(all_orders):
            if full_order['order_id'] == order['order_id']:
                order_index = j
                break
        
        if order_index is None:
            print(f"ERROR: Could not find order {order['order_id']} in full orders list")
            continue
        
        # Execute the order
        success, result = execute_order(engine, order)
        
        if success:
            # Update order status to EXECUTED
            update_order_status(all_orders, order_index, 'EXECUTED', tiger_order_id=result)
            
            # Log trade execution
            log_trade_execution(
                order['symbol'], 
                order['side'], 
                order['quantity'], 
                order.get('price', 0.0), 
                'EXECUTED',
                order_id=result
            )
            
            executed_count += 1
            print(f"SUCCESS: Order {order['order_id']} executed successfully")
            
        else:
            # Update order status to FAILED
            update_order_status(all_orders, order_index, 'FAILED', error_msg=result)
            failed_count += 1
            print(f"FAILED: Order {order['order_id']} failed: {result}")
        
        # Small delay between orders
        time.sleep(1)
    
    # Save updated orders
    save_updated_orders(all_orders)
    
    print(f"\n=== Execution Summary ===")
    print(f"Total orders processed: {len(orders)}")
    print(f"Successfully executed: {executed_count}")
    print(f"Failed: {failed_count}")
    
    if executed_count > 0:
        print(f"\nSUCCESS: {executed_count} trading signals have been successfully transmitted to Tiger API!")
        print("Check your Tiger trading account for order confirmations.")
    else:
        print("\nWARNING: No orders were successfully executed")

if __name__ == "__main__":
    execute_all_signals()