"""
PositionSizer - Calculate position sizes with signal/intent separation.

This module implements FIX 17 from the plan:
- Separate signal from intent (OPEN/CLOSE/REDUCE/INCREASE)
- SELL with no position and short disabled is explicitly rejected
- Intent is derived from current position state
- Short positions require explicit allow_short = True

Signal vs Intent:
- Signal: 'BUY' or 'SELL' (direction of trade)
- Intent: 'OPEN' | 'CLOSE' | 'REDUCE' | 'INCREASE' (what we're trying to do)
"""

import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculate position sizes with signal/intent separation.

    FIX 17: Separate signal from intent.
    - Signal: 'BUY' or 'SELL' (direction of trade)
    - Intent: 'OPEN' | 'CLOSE' | 'REDUCE' | 'INCREASE'

    This prevents the bug where SELL = close all (blocking short positions).
    """

    def __init__(
        self,
        base_position_value: float = 10000.0,
        max_position_pct: float = 0.15,
        allow_short: bool = False
    ):
        """
        Initialize PositionSizer.

        Args:
            base_position_value: Base position value in dollars
            max_position_pct: Maximum position as % of portfolio
            allow_short: Whether to allow short positions (MVP: False)
        """
        self.base_position_value = base_position_value
        self.max_position_pct = max_position_pct
        self.allow_short = allow_short  # MVP: Disable shorting

    def derive_intent(
        self,
        signal: str,
        current_shares: int
    ) -> str:
        """
        Derive intent from signal and current position.

        Args:
            signal: 'BUY' or 'SELL'
            current_shares: Current position (positive = long, negative = short)

        Returns:
            Intent: 'OPEN' | 'CLOSE' | 'INCREASE'
        """
        if signal == 'BUY':
            if current_shares < 0:
                return 'CLOSE'  # Buy to cover short
            elif current_shares == 0:
                return 'OPEN'   # Open new long
            else:
                return 'INCREASE'  # Add to long

        elif signal == 'SELL':
            if current_shares > 0:
                return 'CLOSE'  # Close long
            elif current_shares == 0:
                return 'OPEN'   # Open new short (if allowed)
            else:
                return 'INCREASE'  # Add to short

        return 'UNKNOWN'

    def calculate_delta(
        self,
        signal: str,
        intent: str,
        symbol: str,
        current_shares: int,
        score: float,
        regime: Dict[str, Any],
        current_price: float
    ) -> Tuple[int, str]:
        """
        Calculate the share delta for a trade.

        FIX 17: Separate signal from intent.

        Args:
            signal: 'BUY' or 'SELL' (direction of trade)
            intent: 'OPEN' | 'CLOSE' | 'REDUCE' | 'INCREASE'
            symbol: Stock symbol
            current_shares: Current position (positive = long, negative = short)
            score: Signal score (0-100)
            regime: Regime dict with max_position_pct, sector_boost, etc.
            current_price: Current stock price

        Returns:
            Tuple of (delta_shares: int, reason: str)
            - Positive delta = buy shares
            - Negative delta = sell shares
            - Zero = no action
        """
        if signal == 'BUY':
            return self._handle_buy(intent, symbol, current_shares, score, regime, current_price)

        elif signal == 'SELL':
            return self._handle_sell(intent, symbol, current_shares, score, regime, current_price)

        return 0, "Unknown signal"

    def _handle_buy(
        self,
        intent: str,
        symbol: str,
        current_shares: int,
        score: float,
        regime: Dict[str, Any],
        current_price: float
    ) -> Tuple[int, str]:
        """Handle BUY signal based on intent."""
        if intent == 'OPEN':
            # Open new long position
            if current_shares != 0:
                return 0, "Already have position"
            target_shares = self._calculate_target_shares(symbol, score, regime, current_price)
            return target_shares, "Open long"

        elif intent == 'CLOSE':
            # Close short position (buy to cover)
            if current_shares >= 0:
                return 0, "No short to close"
            return -current_shares, "Close short"  # Buy to cover

        elif intent == 'INCREASE':
            # Add to long position
            additional = self._calculate_add_shares(symbol, score, regime, current_price)
            return additional, "Increase long"

        return 0, f"Unknown intent: {intent}"

    def _handle_sell(
        self,
        intent: str,
        symbol: str,
        current_shares: int,
        score: float,
        regime: Dict[str, Any],
        current_price: float
    ) -> Tuple[int, str]:
        """Handle SELL signal based on intent."""
        if intent == 'CLOSE':
            # Close long position
            if current_shares <= 0:
                return 0, "No long to close"  # FIX 17: Explicit rejection
            return -current_shares, "Close long"

        elif intent == 'OPEN':
            # Open short position
            if not self.allow_short:
                return 0, "Short disabled"  # FIX 17: Explicit rejection
            if current_shares != 0:
                return 0, "Already have position"
            target_shares = -self._calculate_target_shares(symbol, score, regime, current_price)
            return target_shares, "Open short"

        elif intent == 'REDUCE':
            # Partial close of long
            reduce_qty = self._calculate_reduce_shares(current_shares, score)
            return -reduce_qty, "Reduce long"

        elif intent == 'INCREASE':
            # Add to short position
            if not self.allow_short:
                return 0, "Short disabled"
            additional = -self._calculate_add_shares(symbol, score, regime, current_price)
            return additional, "Increase short"

        return 0, f"Unknown intent: {intent}"

    def _calculate_target_shares(
        self,
        symbol: str,
        score: float,
        regime: Dict[str, Any],
        current_price: float
    ) -> int:
        """
        Calculate target share count for a new position.

        Args:
            symbol: Stock symbol
            score: Signal score (0-100)
            regime: Regime dict
            current_price: Current stock price

        Returns:
            Target number of shares
        """
        if current_price <= 0:
            return 0

        # Score-based multiplier
        score_multiplier = self._get_score_multiplier(score)

        # Regime adjustments
        max_pos_pct = regime.get('max_position_pct', 1.0)
        sector_boost = regime.get('sector_boost', 1.0)

        # Calculate position value
        position_value = (
            self.base_position_value *
            score_multiplier *
            max_pos_pct *
            sector_boost
        )

        # Convert to shares
        target_shares = int(position_value / current_price)

        return max(1, target_shares)  # At least 1 share

    def _calculate_add_shares(
        self,
        symbol: str,
        score: float,
        regime: Dict[str, Any],
        current_price: float
    ) -> int:
        """
        Calculate shares to add to existing position.

        Uses smaller size (50% of normal) for adds.
        """
        target = self._calculate_target_shares(symbol, score, regime, current_price)
        return max(1, target // 2)  # 50% of normal size for adds

    def _calculate_reduce_shares(
        self,
        current_shares: int,
        score: float
    ) -> int:
        """
        Calculate shares to reduce from position.

        Based on score:
        - Score < 40: Reduce 50%
        - Score < 50: Reduce 30%
        - Otherwise: Reduce 20%
        """
        if current_shares <= 0:
            return 0

        if score < 40:
            reduce_pct = 0.5
        elif score < 50:
            reduce_pct = 0.3
        else:
            reduce_pct = 0.2

        reduce_qty = int(current_shares * reduce_pct)
        return max(1, reduce_qty)

    def _get_score_multiplier(self, score: float) -> float:
        """
        Convert score to position size multiplier.

        Score-to-Position Mapping:
        - >= 80: 1.0 (Full position)
        - 65-79: 0.6 (Normal signal)
        - 50-64: 0.3 (Weak signal)
        - < 50: 0.2 (Exception only)
        """
        if score >= 80:
            return 1.0
        elif score >= 65:
            return 0.6
        elif score >= 50:
            return 0.3
        else:
            return 0.2

    def calculate_position_for_signal(
        self,
        signal: str,
        symbol: str,
        current_shares: int,
        score: float,
        regime: Dict[str, Any],
        current_price: float
    ) -> Tuple[int, str, str]:
        """
        Calculate position delta for a signal (convenience method).

        Automatically derives intent from signal and current position.

        Args:
            signal: 'BUY' or 'SELL'
            symbol: Stock symbol
            current_shares: Current position
            score: Signal score
            regime: Regime dict
            current_price: Current price

        Returns:
            Tuple of (delta_shares, reason, intent)
        """
        # Derive intent based on signal and current position
        intent = self.derive_intent(signal, current_shares)

        # Check for invalid scenarios
        if signal == 'SELL' and current_shares == 0 and not self.allow_short:
            return 0, "SELL with no position and short disabled", intent

        # Calculate delta
        delta_shares, reason = self.calculate_delta(
            signal, intent, symbol, current_shares, score, regime, current_price
        )

        return delta_shares, reason, intent

    def set_allow_short(self, allow: bool) -> None:
        """Enable or disable short selling."""
        self.allow_short = allow
        logger.info(f"PositionSizer: Short selling {'enabled' if allow else 'disabled'}")

    def set_base_position_value(self, value: float) -> None:
        """Set base position value."""
        self.base_position_value = value
        logger.info(f"PositionSizer: Base position value set to ${value:,.2f}")


# Global singleton instance
_position_sizer: Optional[PositionSizer] = None


def get_position_sizer() -> PositionSizer:
    """Get the global PositionSizer singleton."""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer()
    return _position_sizer


def set_position_sizer(sizer: PositionSizer) -> None:
    """Set the global PositionSizer singleton."""
    global _position_sizer
    _position_sizer = sizer
