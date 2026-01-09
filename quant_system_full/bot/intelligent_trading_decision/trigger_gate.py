"""
TriggerGate - Event-triggered gate to convert continuous signals to discrete events.

This module implements FIX 1-4 and FIX 14 from the plan:
- FIX 1: Clear data structure - (symbol, level, direction) triple key
- FIX 2: Direction-aware cross checking
- FIX 3: Volume z-score dual condition with floor protection
- FIX 4: Separate cooldown by direction (BUY/SELL independent)
- FIX 14: Use cooldown_seconds (not bars) - bar-interval independent

Purpose: Convert 5-9 signals/minute into 0-1 signals per price move.
This is the KEY to reducing trade frequency.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

from .key_level_provider import get_key_level_provider

logger = logging.getLogger(__name__)


class TriggerGate:
    """
    Event-triggered gate that filters continuous signals to discrete events.

    Must satisfy ONE of these events to proceed to scoring:
    1. Key Level Cross (direction-aware)
    2. Volume Shock (dual condition: ratio AND z-score)

    This is the KEY to reducing trade frequency from 5-9/min to 0-1/price move.
    """

    def __init__(self, data_provider=None):
        """
        Initialize TriggerGate.

        Args:
            data_provider: Object that provides volume data methods:
                - get_same_time_avg_volume(symbol)
                - get_same_time_std_volume(symbol)
                - get_daily_avg_volume(symbol)
        """
        # FIX 1: Clear data structure - triple key -> datetime
        # FIX 4: Include direction so BUY/SELL cooldowns are independent
        self.last_trigger_time: Dict[Tuple[str, str, str], datetime] = {}

        # Volume thresholds (conservative for first week)
        self.volume_zscore_threshold = 2.5  # Conservative: was 2.0
        self.volume_ratio_threshold = 2.0  # FIX 3: Dual condition

        # FIX 14: Use SECONDS not "bars" - works for any bar interval (1min, 5min, etc.)
        self.cooldown_seconds = 30 * 60  # 30 minutes = 1800 seconds

        self._data_provider = data_provider

    def check(
        self,
        symbol: str,
        signal: str,
        price: float,
        prev_price: float,
        bar_volume: float,
        bar_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trigger event has occurred.

        FIX 2: Now takes `signal` parameter to enforce direction-aware crossing
        FIX 14: Now takes `bar_time` parameter for consistent time comparison

        Args:
            symbol: Stock symbol
            signal: 'BUY' or 'SELL'
            price: Current price
            prev_price: Previous bar's price
            bar_volume: Current bar's volume
            bar_time: Current bar's timestamp

        Returns:
            Tuple of (triggered: bool, trigger_reason: Optional[str])
        """
        key_level_provider = get_key_level_provider()
        key_levels = key_level_provider.get_key_levels(symbol, bar_time)

        triggered = False
        trigger_reason = None

        # Trigger 1: Key Level Cross (direction-aware)
        # FIX 2: Only allow crosses in same direction as signal
        if signal == "BUY":
            # For BUY: only upward crosses trigger
            triggered, trigger_reason = self._check_buy_crosses(
                symbol, price, prev_price, key_levels, bar_time
            )

        elif signal == "SELL":
            # For SELL: only downward crosses trigger
            triggered, trigger_reason = self._check_sell_crosses(
                symbol, price, prev_price, key_levels, bar_time
            )

        # Trigger 2: Volume Shock (if not already triggered)
        if not triggered:
            volume_triggered, volume_reason = self._check_volume_shock(
                symbol, bar_volume, bar_time
            )
            if volume_triggered:
                triggered = True
                trigger_reason = volume_reason

        if triggered:
            logger.info(f"TRIGGER: {symbol} {signal} - {trigger_reason}")

        return triggered, trigger_reason

    def _check_buy_crosses(
        self,
        symbol: str,
        price: float,
        prev_price: float,
        key_levels: Dict[str, float],
        bar_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """Check for upward crosses on key levels for BUY signals."""
        triggered = False
        trigger_reason = None

        # VWAP cross up
        if key_levels.get('vwap') is not None:
            if self._is_cross_up(prev_price, price, key_levels['vwap']):
                if self._can_trigger(symbol, 'vwap', 'BUY', bar_time):
                    triggered = True
                    trigger_reason = "VWAP cross up"
                    self._record_trigger(symbol, 'vwap', 'BUY', bar_time)

        # OR High breakout
        if key_levels.get('or_high') is not None:
            if self._is_cross_up(prev_price, price, key_levels['or_high']):
                if self._can_trigger(symbol, 'or_high', 'BUY', bar_time):
                    triggered = True
                    trigger_reason = "OR High breakout"
                    self._record_trigger(symbol, 'or_high', 'BUY', bar_time)

        # Prev close cross up
        if key_levels.get('prev_close') is not None:
            if self._is_cross_up(prev_price, price, key_levels['prev_close']):
                if self._can_trigger(symbol, 'prev_close', 'BUY', bar_time):
                    triggered = True
                    trigger_reason = "Prev close cross up"
                    self._record_trigger(symbol, 'prev_close', 'BUY', bar_time)

        return triggered, trigger_reason

    def _check_sell_crosses(
        self,
        symbol: str,
        price: float,
        prev_price: float,
        key_levels: Dict[str, float],
        bar_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """Check for downward crosses on key levels for SELL signals."""
        triggered = False
        trigger_reason = None

        # VWAP cross down
        if key_levels.get('vwap') is not None:
            if self._is_cross_down(prev_price, price, key_levels['vwap']):
                if self._can_trigger(symbol, 'vwap', 'SELL', bar_time):
                    triggered = True
                    trigger_reason = "VWAP cross down"
                    self._record_trigger(symbol, 'vwap', 'SELL', bar_time)

        # OR Low breakdown
        if key_levels.get('or_low') is not None:
            if self._is_cross_down(prev_price, price, key_levels['or_low']):
                if self._can_trigger(symbol, 'or_low', 'SELL', bar_time):
                    triggered = True
                    trigger_reason = "OR Low breakdown"
                    self._record_trigger(symbol, 'or_low', 'SELL', bar_time)

        # Prev close cross down
        if key_levels.get('prev_close') is not None:
            if self._is_cross_down(prev_price, price, key_levels['prev_close']):
                if self._can_trigger(symbol, 'prev_close', 'SELL', bar_time):
                    triggered = True
                    trigger_reason = "Prev close cross down"
                    self._record_trigger(symbol, 'prev_close', 'SELL', bar_time)

        return triggered, trigger_reason

    def _check_volume_shock(
        self,
        symbol: str,
        bar_volume: float,
        bar_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for volume shock trigger.

        FIX 3: Dual condition to prevent early morning false triggers
        """
        if not self._data_provider:
            return False, None

        volume_ratio, volume_zscore = self._get_volume_metrics(symbol, bar_volume)

        if volume_ratio is None or volume_zscore is None:
            return False, None

        # FIX 3: Both conditions must be met (dual condition)
        if volume_ratio > self.volume_ratio_threshold and volume_zscore > self.volume_zscore_threshold:
            return True, f"Volume shock ratio={volume_ratio:.1f}x z={volume_zscore:.1f}"

        return False, None

    def _is_cross_up(self, prev_price: float, price: float, level: float) -> bool:
        """True if price just crossed UP through the level."""
        return prev_price <= level < price

    def _is_cross_down(self, prev_price: float, price: float, level: float) -> bool:
        """True if price just crossed DOWN through the level."""
        return prev_price >= level > price

    def _can_trigger(
        self,
        symbol: str,
        level_name: str,
        direction: str,
        bar_time: datetime
    ) -> bool:
        """
        Check if this level+direction can trigger (cooldown check).

        FIX 4: Triple key with direction so BUY/SELL are independent.
        FIX 14: Compare in seconds using bar_time (not datetime.now())
        """
        key = (symbol, level_name, direction)  # FIX 1 & 4: Triple key with direction
        last_trigger = self.last_trigger_time.get(key)

        if last_trigger is None:
            return True

        # FIX 14: Compare in seconds using bar_time (not datetime.now())
        seconds_since = (bar_time - last_trigger).total_seconds()
        return seconds_since >= self.cooldown_seconds

    def _record_trigger(
        self,
        symbol: str,
        level_name: str,
        direction: str,
        bar_time: datetime
    ) -> None:
        """
        Record that a trigger occurred.

        FIX 4: Triple key with direction.
        FIX 14: Use bar_time (not datetime.now()) for consistent time tracking.
        """
        key = (symbol, level_name, direction)  # FIX 1 & 4: Triple key with direction
        # FIX 14: Use bar_time (not datetime.now()) for consistent time tracking
        self.last_trigger_time[key] = bar_time

    def _get_volume_metrics(
        self,
        symbol: str,
        bar_volume: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get volume ratio and z-score metrics.

        FIX 3: Return both ratio and z-score for dual condition check.
        """
        try:
            same_time_avg = self._data_provider.get_same_time_avg_volume(symbol)
            same_time_std = self._data_provider.get_same_time_std_volume(symbol)

            if same_time_avg is None:
                return None, None

            # FIX 3a: Protect against near-zero average
            daily_avg = self._data_provider.get_daily_avg_volume(symbol) or 0
            avg_floor = max(1000, daily_avg / 390 * 0.1)  # 10% of per-minute average
            same_time_avg = max(same_time_avg, avg_floor)

            # FIX 3b: Protect against near-zero std (early morning)
            if same_time_std is None:
                same_time_std = same_time_avg * 0.3
            std_floor = same_time_avg * 0.3  # At least 30% of avg as std
            same_time_std = max(same_time_std, std_floor)

            volume_ratio = bar_volume / same_time_avg
            volume_zscore = (bar_volume - same_time_avg) / same_time_std

            return volume_ratio, volume_zscore

        except Exception as e:
            logger.error(f"Error calculating volume metrics for {symbol}: {e}")
            return None, None

    def reset_daily(self) -> None:
        """
        Reset all trigger cooldowns at market open.

        Call this at the start of each trading day.
        """
        self.last_trigger_time = {}
        logger.info("TriggerGate: Daily reset completed")

    def set_data_provider(self, data_provider) -> None:
        """Set the data provider for volume data access."""
        self._data_provider = data_provider

    def set_cooldown_seconds(self, seconds: int) -> None:
        """
        Set the cooldown period in seconds.

        Args:
            seconds: Cooldown period in seconds (default 1800 = 30 minutes)
        """
        self.cooldown_seconds = seconds
        logger.info(f"TriggerGate: Cooldown set to {seconds} seconds")

    def get_trigger_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get the trigger status for a symbol (for debugging/monitoring).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with trigger timestamps for all level/direction combinations
        """
        status = {}
        for key, trigger_time in self.last_trigger_time.items():
            sym, level, direction = key
            if sym == symbol:
                status[f"{level}_{direction}"] = {
                    'last_trigger': trigger_time.isoformat(),
                    'cooldown_remaining': max(
                        0,
                        self.cooldown_seconds - (datetime.now() - trigger_time).total_seconds()
                    )
                }
        return status


# Global singleton instance
_trigger_gate: Optional[TriggerGate] = None


def get_trigger_gate() -> TriggerGate:
    """Get the global TriggerGate singleton."""
    global _trigger_gate
    if _trigger_gate is None:
        _trigger_gate = TriggerGate()
    return _trigger_gate


def set_trigger_gate(gate: TriggerGate) -> None:
    """Set the global TriggerGate singleton."""
    global _trigger_gate
    _trigger_gate = gate
