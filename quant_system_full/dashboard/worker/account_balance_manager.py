"""
Account Balance Manager

This module manages account balance and ensures orders don't exceed available funds.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AccountBalanceManager:
    """Manages account balance and validates order amounts."""
    
    def __init__(self, balance_file: str = None, initial_balance: float = None):
        self.balance_file = balance_file or "account_balance.json"
        self.account_data = self._load_account_data()
        
        # Initialize with provided balance if file doesn't exist
        if initial_balance and not os.path.exists(self.balance_file):
            self.account_data["available_cash"] = initial_balance
            self.account_data["initial_balance"] = initial_balance
            self._save_account_data()
    
    def _load_account_data(self) -> Dict:
        """Load account data from file."""
        try:
            if os.path.exists(self.balance_file):
                with open(self.balance_file, 'r') as f:
                    data = json.load(f)
                    # Ensure required fields exist
                    if "available_cash" not in data:
                        data["available_cash"] = 10000.0  # Default balance
                    if "initial_balance" not in data:
                        data["initial_balance"] = data["available_cash"]
                    if "reserved_cash" not in data:
                        data["reserved_cash"] = 0.0
                    if "transactions" not in data:
                        data["transactions"] = []
                    return data
        except Exception as e:
            logger.error(f"Error loading account data: {e}")
        
        # DISABLED: No fake balance data allowed
        # User must provide real API connection for balance information
        raise Exception("Account balance file not found and no real Tiger API connection available. Fake balance data has been disabled for safety.")
    
    def _save_account_data(self):
        """Save account data to file."""
        try:
            self.account_data["last_updated"] = datetime.now().isoformat()
            with open(self.balance_file, 'w') as f:
                json.dump(self.account_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving account data: {e}")
    
    def get_available_balance(self) -> float:
        """Get current available cash balance."""
        return self.account_data.get("available_cash", 0.0)
    
    def get_account_summary(self) -> Dict:
        """Get complete account summary."""
        return {
            "available_cash": self.account_data.get("available_cash", 0.0),
            "initial_balance": self.account_data.get("initial_balance", 0.0),
            "reserved_cash": self.account_data.get("reserved_cash", 0.0),
            "total_transactions": len(self.account_data.get("transactions", [])),
            "last_updated": self.account_data.get("last_updated")
        }
    
    def update_balance_from_tiger_api(self, execution_engine=None):
        """Update balance from Tiger API."""
        try:
            if execution_engine is None:
                logger.warning("No execution engine provided, cannot fetch real balance")
                return False
            
            # Get account assets from Tiger API
            assets = execution_engine.get_account_assets()
            
            if assets:
                # Get available cash from the returned dictionary
                available_cash = assets.get('cash_available', 0.0)
                
                old_balance = self.account_data.get("available_cash", 0.0)
                self.account_data["available_cash"] = available_cash
                
                # Record transaction
                transaction = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "tiger_api_sync",
                    "old_balance": old_balance,
                    "new_balance": available_cash,
                    "change": available_cash - old_balance,
                    "reason": "Synced from Tiger API"
                }
                
                self.account_data.setdefault("transactions", []).append(transaction)
                self._save_account_data()
                
                logger.info(f"Balance synced from Tiger API: ${old_balance:.2f} -> ${available_cash:.2f}")
                return True
            else:
                logger.error("Could not get account assets from Tiger API")
                return False
                
        except Exception as e:
            logger.error(f"Error updating balance from Tiger API: {e}")
            return False
    
    def update_balance(self, new_balance: float, reason: str = "Manual update"):
        """Manually update account balance."""
        old_balance = self.account_data.get("available_cash", 0.0)
        self.account_data["available_cash"] = new_balance
        
        # Record transaction
        transaction = {
            "timestamp": datetime.now().isoformat(),
            "type": "balance_update",
            "old_balance": old_balance,
            "new_balance": new_balance,
            "change": new_balance - old_balance,
            "reason": reason
        }
        
        self.account_data.setdefault("transactions", []).append(transaction)
        self._save_account_data()
        
        logger.info(f"Balance updated: ${old_balance:.2f} -> ${new_balance:.2f} ({reason})")
    
    def can_afford_order(self, symbol: str, quantity: int, estimated_price: float = None) -> Tuple[bool, str, int]:
        """
        Check if account can afford an order and suggest adjusted quantity if needed.
        
        Args:
            symbol: Stock symbol
            quantity: Desired quantity
            estimated_price: Estimated price per share (if None, uses recent price estimate)
        
        Returns:
            (can_afford, reason, suggested_quantity)
        """
        available_cash = self.get_available_balance()
        
        # If no price provided, try to estimate based on symbol
        if estimated_price is None:
            estimated_price = self._estimate_stock_price(symbol)
        
        if estimated_price <= 0:
            return False, f"Cannot estimate price for {symbol}", 0
        
        total_cost = quantity * estimated_price
        
        # Add some buffer (2%) for price fluctuations and fees
        total_cost_with_buffer = total_cost * 1.02
        
        if available_cash >= total_cost_with_buffer:
            return True, "Sufficient funds", quantity
        
        # Calculate max affordable quantity
        max_affordable = int(available_cash * 0.98 / estimated_price)  # 98% to leave buffer
        
        if max_affordable <= 0:
            return False, f"Insufficient funds: ${available_cash:.2f} available, need ${total_cost_with_buffer:.2f}", 0
        
        return False, f"Partial funding available: can afford {max_affordable} shares instead of {quantity}", max_affordable
    
    def _estimate_stock_price(self, symbol: str) -> float:
        """Estimate stock price based on symbol (simple heuristics)."""
        # This is a simple estimation - in production you'd fetch real prices
        price_estimates = {
            # Large caps - typically $100-400
            'AAPL': 180.0, 'MSFT': 340.0, 'GOOGL': 140.0, 'AMZN': 145.0, 'NVDA': 120.0,
            'META': 520.0, 'TSLA': 240.0, 'BRK.B': 440.0,
            
            # Mid caps - typically $50-150  
            'ABBV': 170.0, 'PFE': 29.0, 'JNJ': 160.0, 'UNH': 570.0, 'MRK': 115.0,
            'HD': 380.0, 'KO': 63.0, 'PEP': 175.0, 'WMT': 75.0, 'DIS': 95.0,
            
            # Small/Mid caps - typically $20-80
            'AA': 32.0, 'AGG': 104.0, 'AFRM': 35.0, 'AAL': 12.0, 'FITBP': 24.0
        }
        
        return price_estimates.get(symbol, 50.0)  # Default to $50 if unknown
    
    def reserve_funds_for_order(self, symbol: str, quantity: int, estimated_price: float, order_id: str = None):
        """Reserve funds when placing an order."""
        total_cost = quantity * estimated_price * 1.02  # 2% buffer
        
        available = self.account_data.get("available_cash", 0.0)
        reserved = self.account_data.get("reserved_cash", 0.0)
        
        if available >= total_cost:
            self.account_data["available_cash"] = available - total_cost
            self.account_data["reserved_cash"] = reserved + total_cost
            
            # Record reservation
            transaction = {
                "timestamp": datetime.now().isoformat(),
                "type": "funds_reserved",
                "symbol": symbol,
                "quantity": quantity,
                "estimated_price": estimated_price,
                "total_reserved": total_cost,
                "order_id": order_id,
                "status": "reserved"
            }
            
            self.account_data.setdefault("transactions", []).append(transaction)
            self._save_account_data()
            
            logger.info(f"Reserved ${total_cost:.2f} for {symbol} order ({quantity} shares)")
            return True
        
        return False
    
    def complete_order(self, symbol: str, quantity: int, actual_price: float, order_id: str = None, success: bool = True):
        """Complete an order and update balances accordingly."""
        actual_cost = quantity * actual_price
        
        # Find and update the reservation
        transactions = self.account_data.get("transactions", [])
        reserved_amount = 0
        
        for transaction in reversed(transactions):
            if (transaction.get("symbol") == symbol and 
                transaction.get("order_id") == order_id and
                transaction.get("type") == "funds_reserved" and
                transaction.get("status") == "reserved"):
                
                reserved_amount = transaction.get("total_reserved", 0)
                transaction["status"] = "completed" if success else "cancelled"
                break
        
        if success:
            # Successful order - adjust reserved cash by difference
            reserved = self.account_data.get("reserved_cash", 0.0)
            difference = reserved_amount - actual_cost
            
            self.account_data["reserved_cash"] = max(0, reserved - reserved_amount)
            
            # If we reserved more than needed, return the difference to available cash
            if difference > 0:
                self.account_data["available_cash"] += difference
            
            # Record completion
            transaction = {
                "timestamp": datetime.now().isoformat(),
                "type": "order_completed",
                "symbol": symbol,
                "quantity": quantity,
                "actual_price": actual_price,
                "total_cost": actual_cost,
                "order_id": order_id,
                "reserved_amount": reserved_amount,
                "refunded_amount": max(0, difference)
            }
            
            logger.info(f"Order completed: {symbol} {quantity} shares @ ${actual_price:.2f} (${actual_cost:.2f} total)")
            
        else:
            # Failed order - return reserved funds
            reserved = self.account_data.get("reserved_cash", 0.0)
            available = self.account_data.get("available_cash", 0.0)
            
            self.account_data["reserved_cash"] = max(0, reserved - reserved_amount)
            self.account_data["available_cash"] = available + reserved_amount
            
            # Record failure
            transaction = {
                "timestamp": datetime.now().isoformat(),
                "type": "order_failed",
                "symbol": symbol,
                "quantity": quantity,
                "order_id": order_id,
                "refunded_amount": reserved_amount
            }
            
            logger.info(f"Order failed: {symbol} - refunded ${reserved_amount:.2f}")
        
        self.account_data.setdefault("transactions", []).append(transaction)
        self._save_account_data()


# Global balance manager instance
_balance_manager = None

def get_balance_manager(initial_balance: float = None) -> AccountBalanceManager:
    """Get the global balance manager instance."""
    global _balance_manager
    if _balance_manager is None:
        # Store balance file in state directory
        state_dir = Path(__file__).parent.parent / "state"
        state_dir.mkdir(exist_ok=True)
        balance_file = state_dir / "account_balance.json"
        _balance_manager = AccountBalanceManager(str(balance_file), initial_balance)
    
    return _balance_manager


if __name__ == "__main__":
    # Test the balance manager
    manager = get_balance_manager(10000.0)  # $10,000 initial balance
    
    print("Account Summary:")
    summary = manager.get_account_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nTesting order affordability:")
    symbols = ['AAPL', 'ABBV', 'AA', 'NVDA', 'GOOGL']
    
    for symbol in symbols:
        can_afford, reason, suggested_qty = manager.can_afford_order(symbol, 10)
        print(f"  {symbol} (10 shares): {reason}")
        if suggested_qty != 10:
            print(f"    Suggested quantity: {suggested_qty}")