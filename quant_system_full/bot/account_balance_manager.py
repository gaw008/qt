"""
Simple Account Balance Manager for Trading System
Provides basic balance tracking functionality.
"""

import json
import os
from datetime import datetime
from pathlib import Path

class BalanceManager:
    """Simple balance manager for trading operations."""
    
    def __init__(self, initial_balance=12000.0):
        self.balance_file = Path(__file__).parent.parent / 'dashboard' / 'state' / 'account_balance.json'
        self.initial_balance = initial_balance
        self.reserved_funds = {}  # Track reserved funds for pending orders
        
    def get_available_balance(self):
        """Get current available balance."""
        try:
            if self.balance_file.exists():
                with open(self.balance_file, 'r') as f:
                    data = json.load(f)
                    return data.get('available_balance', self.initial_balance)
        except Exception:
            pass
        return self.initial_balance
    
    def update_balance_from_tiger_api(self, execution_engine):
        """Update balance from Tiger API."""
        try:
            # Try to get real account balance if available
            positions = execution_engine.get_account_positions()
            if positions is not None:
                # For now, return True to indicate success
                return True
        except Exception:
            pass
        return False
    
    def reserve_funds_for_order(self, symbol, quantity, estimated_price, order_id):
        """Reserve funds for an order."""
        required_funds = quantity * estimated_price
        available = self.get_available_balance()
        total_reserved = sum(self.reserved_funds.values())
        
        if (available - total_reserved) >= required_funds:
            self.reserved_funds[order_id] = required_funds
            return True
        return False
    
    def complete_order(self, symbol, quantity, actual_price, order_id, success):
        """Complete an order and update balance."""
        if order_id in self.reserved_funds:
            reserved = self.reserved_funds.pop(order_id)
            
            if success:
                # Deduct actual cost from balance
                actual_cost = quantity * actual_price
                current_balance = self.get_available_balance()
                new_balance = current_balance - actual_cost
                
                # Save updated balance
                try:
                    balance_data = {
                        'available_balance': new_balance,
                        'last_updated': datetime.now().isoformat(),
                        'last_transaction': {
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': actual_price,
                            'cost': actual_cost,
                            'order_id': order_id
                        }
                    }
                    
                    os.makedirs(self.balance_file.parent, exist_ok=True)
                    with open(self.balance_file, 'w') as f:
                        json.dump(balance_data, f, indent=2)
                        
                except Exception as e:
                    print(f"Error saving balance: {e}")
    
    def _estimate_stock_price(self, symbol):
        """Estimate stock price for order sizing."""
        # Simple price estimation - in real system this would use market data
        price_estimates = {
            'EDU': 11.5, 'HSBC': 35.2, 'SAN': 4.1, 'PUK': 6.8, 'GME': 15.3,
            'NTNX': 48.7, 'FLEX': 25.6, 'DASH': 62.4, 'ROIV': 12.8, 'CDW': 195.4
        }
        return price_estimates.get(symbol, 50.0)  # Default $50 if unknown

# Global instance
_balance_manager = None

def get_balance_manager():
    """Get the global balance manager instance."""
    global _balance_manager
    if _balance_manager is None:
        _balance_manager = BalanceManager()
    return _balance_manager