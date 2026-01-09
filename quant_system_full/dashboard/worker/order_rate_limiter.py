"""
Emergency Order Rate Limiter

This module prevents excessive order placement that could lead to continuous trading loops.
"""

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

class OrderRateLimiter:
    """Rate limiter to prevent excessive order placement."""
    
    def __init__(self, rate_limit_file: str = None):
        self.rate_limit_file = rate_limit_file or "order_rate_limit.json"
        self.order_history = self._load_history()
        
        # Rate limits (per symbol)
        self.max_orders_per_hour = 2
        self.max_orders_per_day = 5
        self.cooldown_minutes = 30  # Minimum time between orders for same symbol
        
    def _load_history(self) -> Dict:
        """Load order history from file."""
        try:
            if os.path.exists(self.rate_limit_file):
                with open(self.rate_limit_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load order history: {e}")
        
        return {"orders": []}
    
    def _save_history(self):
        """Save order history to file."""
        try:
            with open(self.rate_limit_file, 'w') as f:
                json.dump(self.order_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save order history: {e}")
    
    def _cleanup_old_orders(self):
        """Remove orders older than 24 hours."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        orders = self.order_history.get("orders", [])
        new_orders = []
        
        for order in orders:
            try:
                order_time = datetime.fromisoformat(order.get("timestamp", ""))
                if order_time > cutoff_time:
                    new_orders.append(order)
            except Exception:
                continue  # Skip malformed orders
        
        self.order_history["orders"] = new_orders
        self._save_history()
    
    def can_place_order(self, symbol: str) -> tuple[bool, str]:
        """
        Check if an order can be placed for the given symbol.
        
        Returns:
            (can_place, reason)
        """
        self._cleanup_old_orders()
        
        now = datetime.now()
        orders = self.order_history.get("orders", [])
        
        # Get orders for this symbol
        symbol_orders = [o for o in orders if o.get("symbol") == symbol]
        
        if not symbol_orders:
            return True, "No previous orders"
        
        # Check cooldown period
        last_order_time = max(
            datetime.fromisoformat(o.get("timestamp", ""))
            for o in symbol_orders
        )
        
        minutes_since_last = (now - last_order_time).total_seconds() / 60
        if minutes_since_last < self.cooldown_minutes:
            return False, f"Cooldown: {self.cooldown_minutes - minutes_since_last:.1f} min remaining"
        
        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        recent_orders = [
            o for o in symbol_orders 
            if datetime.fromisoformat(o.get("timestamp", "")) > hour_ago
        ]
        
        if len(recent_orders) >= self.max_orders_per_hour:
            return False, f"Hourly limit exceeded: {len(recent_orders)}/{self.max_orders_per_hour}"
        
        # Check daily limit
        day_ago = now - timedelta(hours=24)
        daily_orders = [
            o for o in symbol_orders 
            if datetime.fromisoformat(o.get("timestamp", "")) > day_ago
        ]
        
        if len(daily_orders) >= self.max_orders_per_day:
            return False, f"Daily limit exceeded: {len(daily_orders)}/{self.max_orders_per_day}"
        
        return True, "Rate limit check passed"
    
    def record_order(self, symbol: str, side: str, quantity: int, order_id: str = None):
        """Record a new order placement."""
        order_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_id": order_id
        }
        
        self.order_history.setdefault("orders", []).append(order_record)
        self._save_history()
        
        print(f"[RATE_LIMITER] Recorded {side} order: {symbol} x{quantity}")
    
    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get order statistics for a symbol."""
        self._cleanup_old_orders()
        
        now = datetime.now()
        orders = self.order_history.get("orders", [])
        symbol_orders = [o for o in orders if o.get("symbol") == symbol]
        
        if not symbol_orders:
            return {"total_orders": 0, "last_order": None}
        
        # Calculate stats
        last_order_time = max(
            datetime.fromisoformat(o.get("timestamp", ""))
            for o in symbol_orders
        )
        
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)
        
        hourly_count = len([
            o for o in symbol_orders 
            if datetime.fromisoformat(o.get("timestamp", "")) > hour_ago
        ])
        
        daily_count = len([
            o for o in symbol_orders 
            if datetime.fromisoformat(o.get("timestamp", "")) > day_ago
        ])
        
        minutes_since_last = (now - last_order_time).total_seconds() / 60
        
        return {
            "total_orders": len(symbol_orders),
            "hourly_count": hourly_count,
            "daily_count": daily_count,
            "last_order": last_order_time.isoformat(),
            "minutes_since_last": round(minutes_since_last, 1),
            "can_place_order": self.can_place_order(symbol)[0]
        }


# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter() -> OrderRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        # Store rate limit file in state directory
        state_dir = Path(__file__).parent.parent / "state"
        state_dir.mkdir(exist_ok=True)
        rate_limit_file = state_dir / "order_rate_limit.json"
        _rate_limiter = OrderRateLimiter(str(rate_limit_file))
    
    return _rate_limiter


if __name__ == "__main__":
    # Test the rate limiter
    limiter = get_rate_limiter()
    
    # Test ABBV (which was causing issues)
    can_place, reason = limiter.can_place_order("ABBV")
    print(f"Can place ABBV order: {can_place}, Reason: {reason}")
    
    # Show ABBV stats
    stats = limiter.get_symbol_stats("ABBV")
    print(f"ABBV stats: {stats}")
    
    # Show all symbols with recent orders
    orders = limiter.order_history.get("orders", [])
    symbols = set(o.get("symbol", "") for o in orders)
    
    print("\nAll symbols with recent orders:")
    for symbol in sorted(symbols):
        if symbol:
            stats = limiter.get_symbol_stats(symbol)
            print(f"  {symbol}: {stats['daily_count']} orders today, last: {stats['minutes_since_last']:.1f} min ago")