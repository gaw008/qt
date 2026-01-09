"""
ExitManager - Standardized exit rules to control single-trade risk.

Purpose: Prevent single-trade losses from expanding.
The system manages entry well, but without standardized exits:
- Entry frequency may decrease
- Single-trade losses may expand
- Net result is still not profitable

Exit Rules:
1. Stop Loss: 1.0 * ATR (intraday)
2. Take Profit: 1.5 * ATR (intraday)
3. Time Stop: 30 min without profit

CRITICAL: ATR Consistency
- Entry (CostBenefitGate) uses: get_intraday_atr(symbol, periods=20)
- Exit (ExitManager) uses: get_intraday_atr(symbol, periods=20) - SAME function
- If mismatch, risk/reward assumption breaks
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

from .position_manager import Position

logger = logging.getLogger(__name__)


class ExitManager:
    """
    Minimum viable exit rules to control single-trade risk.
    Managed at Layer 4 (system level) for uniform execution.
    """

    def __init__(self, data_provider=None):
        """
        Initialize ExitManager.

        Args:
            data_provider: Object that provides:
                - get_intraday_atr(symbol, periods)
                - get_current_price(symbol)
        """
        # ATR-based dynamic stops
        self.stop_loss_atr_multiple = 1.0   # 1.0 * ATR_intraday
        self.take_profit_atr_multiple = 1.5  # 1.5 * ATR_intraday (can increase to 2.0)

        # Time stop
        self.time_stop_minutes = 30  # Exit if no profit after 30 min

        self._data_provider = data_provider

    def check_exit(
        self,
        position: Position,
        current_price: float,
        bar_time: datetime
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            position: Position object with entry details
            current_price: Current market price
            bar_time: Current bar timestamp

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        symbol = position.symbol
        entry_price = position.entry_price
        entry_time = position.entry_time

        # Get ATR for this symbol
        atr = self._get_intraday_atr(symbol, periods=20)

        if atr is None or atr <= 0:
            # No ATR - use 1% of entry price as default
            atr = entry_price * 0.01
            logger.warning(f"{symbol}: No ATR data, using 1% estimate for exits")

        # Calculate P&L per share (direction-aware)
        if position.direction == 'LONG':
            pnl_per_share = current_price - entry_price
        else:  # SHORT
            pnl_per_share = entry_price - current_price

        # 1. Stop Loss: 1.0 * ATR
        stop_loss_threshold = -1 * self.stop_loss_atr_multiple * atr
        if pnl_per_share <= stop_loss_threshold:
            return True, f"Stop loss: ${pnl_per_share:.2f}/sh <= ${stop_loss_threshold:.2f} (1.0 ATR)"

        # 2. Take Profit: 1.5 * ATR
        take_profit_threshold = self.take_profit_atr_multiple * atr
        if pnl_per_share >= take_profit_threshold:
            return True, f"Take profit: ${pnl_per_share:.2f}/sh >= ${take_profit_threshold:.2f} (1.5 ATR)"

        # 3. Time Stop: 30 min without profit
        if entry_time:
            minutes_held = (bar_time - entry_time).total_seconds() / 60
            if minutes_held >= self.time_stop_minutes and pnl_per_share <= 0:
                return True, f"Time stop: {minutes_held:.0f}min held, P&L=${pnl_per_share:.2f}/sh"

        return False, "Hold"

    def get_exit_levels(
        self,
        entry_price: float,
        direction: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Calculate stop loss and take profit price levels for order placement.

        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            symbol: Stock symbol

        Returns:
            Dict with exit levels:
                - stop_loss: Stop loss price
                - take_profit: Take profit price
                - time_stop_minutes: Time stop in minutes
                - atr_used: ATR value used for calculations
        """
        atr = self._get_intraday_atr(symbol, periods=20)

        if atr is None or atr <= 0:
            atr = entry_price * 0.01
            logger.warning(f"{symbol}: No ATR data, using 1% estimate")

        if direction == 'LONG':
            stop_loss_price = entry_price - self.stop_loss_atr_multiple * atr
            take_profit_price = entry_price + self.take_profit_atr_multiple * atr
        else:  # SHORT
            stop_loss_price = entry_price + self.stop_loss_atr_multiple * atr
            take_profit_price = entry_price - self.take_profit_atr_multiple * atr

        return {
            'stop_loss': round(stop_loss_price, 2),
            'take_profit': round(take_profit_price, 2),
            'time_stop_minutes': self.time_stop_minutes,
            'atr_used': round(atr, 4),
            'stop_loss_distance': round(self.stop_loss_atr_multiple * atr, 4),
            'take_profit_distance': round(self.take_profit_atr_multiple * atr, 4),
        }

    def get_exit_status(
        self,
        position: Position,
        current_price: float,
        bar_time: datetime
    ) -> Dict[str, Any]:
        """
        Get detailed exit status for monitoring.

        Args:
            position: Position object
            current_price: Current price
            bar_time: Current timestamp

        Returns:
            Dict with exit status details
        """
        symbol = position.symbol
        entry_price = position.entry_price
        entry_time = position.entry_time

        atr = self._get_intraday_atr(symbol, periods=20) or (entry_price * 0.01)

        # Calculate P&L
        if position.direction == 'LONG':
            pnl_per_share = current_price - entry_price
        else:
            pnl_per_share = entry_price - current_price

        # Calculate thresholds
        stop_loss_threshold = -1 * self.stop_loss_atr_multiple * atr
        take_profit_threshold = self.take_profit_atr_multiple * atr

        # Time held
        minutes_held = (bar_time - entry_time).total_seconds() / 60 if entry_time else 0

        # Distance to thresholds
        distance_to_stop = pnl_per_share - stop_loss_threshold
        distance_to_profit = take_profit_threshold - pnl_per_share

        return {
            'symbol': symbol,
            'direction': position.direction,
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl_per_share': round(pnl_per_share, 4),
            'pnl_total': round(pnl_per_share * position.quantity, 2),
            'atr': round(atr, 4),
            'stop_loss_threshold': round(stop_loss_threshold, 4),
            'take_profit_threshold': round(take_profit_threshold, 4),
            'distance_to_stop': round(distance_to_stop, 4),
            'distance_to_profit': round(distance_to_profit, 4),
            'minutes_held': round(minutes_held, 1),
            'time_stop_remaining': max(0, self.time_stop_minutes - minutes_held),
        }

    def _get_intraday_atr(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Get intraday ATR from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_intraday_atr'):
                return self._data_provider.get_intraday_atr(symbol, periods)
            return None
        except Exception as e:
            logger.error(f"Error getting ATR for {symbol}: {e}")
            return None

    def set_data_provider(self, data_provider) -> None:
        """Set the data provider."""
        self._data_provider = data_provider

    def set_stop_loss_atr(self, multiple: float) -> None:
        """Set stop loss ATR multiple."""
        self.stop_loss_atr_multiple = multiple
        logger.info(f"ExitManager: Stop loss set to {multiple} ATR")

    def set_take_profit_atr(self, multiple: float) -> None:
        """Set take profit ATR multiple."""
        self.take_profit_atr_multiple = multiple
        logger.info(f"ExitManager: Take profit set to {multiple} ATR")

    def set_time_stop(self, minutes: int) -> None:
        """Set time stop in minutes."""
        self.time_stop_minutes = minutes
        logger.info(f"ExitManager: Time stop set to {minutes} minutes")


# Global singleton instance
_exit_manager: Optional[ExitManager] = None


def get_exit_manager() -> ExitManager:
    """Get the global ExitManager singleton."""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = ExitManager()
    return _exit_manager


def set_exit_manager(manager: ExitManager) -> None:
    """Set the global ExitManager singleton."""
    global _exit_manager
    _exit_manager = manager
