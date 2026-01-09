"""Stop Loss Manager - manages stop-loss and profit-lock rules"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StopLossManager:
    """Manages stop-loss, trailing stop, and profit-lock rules."""

    def __init__(self,
                 stop_loss_pct: float = 0.08,
                 trailing_stop_pct: float = 0.12,
                 profit_lock_pct: float = 0.15):
        """
        Initialize stop-loss manager.

        Args:
            stop_loss_pct: Fixed stop-loss percentage (e.g., 0.08 = 8%)
            trailing_stop_pct: Trailing stop percentage (e.g., 0.12 = 12%)
            profit_lock_pct: Profit lock trigger (e.g., 0.15 = 15%)
        """
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.profit_lock_pct = profit_lock_pct

        self.position_states = {}  # Track highest price for trailing stop

        logger.info(f"StopLossManager initialized: SL={stop_loss_pct:.1%}, "
                    f"Trailing={trailing_stop_pct:.1%}, Lock={profit_lock_pct:.1%}")

    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float) -> tuple:
        """
        Check if stop-loss is triggered.

        Returns:
            (should_exit, reason)
        """
        try:
            if entry_price <= 0 or current_price <= 0:
                return False, ""

            loss = (current_price - entry_price) / entry_price

            # Fixed stop-loss
            if loss <= -self.stop_loss_pct:
                return True, f"Stop-loss triggered ({loss:.2%})"

            return False, ""

        except Exception as e:
            logger.error(f"Error checking stop-loss for {symbol}: {e}")
            return False, ""

    def update_trailing_stop(self, symbol: str, entry_price: float, current_price: float) -> tuple:
        """
        Update and check trailing stop.

        Returns:
            (should_exit, reason)
        """
        try:
            if entry_price <= 0 or current_price <= 0:
                return False, ""

            # Track highest price
            if symbol not in self.position_states:
                self.position_states[symbol] = {'highest_price': current_price}

            highest = self.position_states[symbol]['highest_price']

            # Update highest
            if current_price > highest:
                self.position_states[symbol]['highest_price'] = current_price
                highest = current_price

            # Check if profit locked and trailing stop triggered
            profit = (highest - entry_price) / entry_price

            if profit >= self.profit_lock_pct:
                # Trailing stop active
                drawdown_from_high = (current_price - highest) / highest

                if drawdown_from_high <= -self.trailing_stop_pct:
                    return True, f"Trailing stop triggered (profit: {profit:.2%}, drawdown: {drawdown_from_high:.2%})"

            return False, ""

        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {e}")
            return False, ""

    def should_exit_position(self, symbol: str, entry_price: float, current_price: float) -> tuple:
        """
        Comprehensive exit check.

        Returns:
            (should_exit, reason)
        """
        # Check fixed stop-loss
        exit, reason = self.check_stop_loss(symbol, entry_price, current_price)
        if exit:
            return True, reason

        # Check trailing stop
        exit, reason = self.update_trailing_stop(symbol, entry_price, current_price)
        if exit:
            return True, reason

        return False, ""

    def reset_position(self, symbol: str):
        """Reset tracking for a position."""
        if symbol in self.position_states:
            del self.position_states[symbol]
