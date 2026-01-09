"""
KeyLevelProvider - Provides key price levels for trading decisions.

This module implements FIX 12 and FIX 13 from the plan:
- FIX 12: Auto-reset via session date tracking (O(1) global clear)
- FIX 13: Use America/New_York timezone for OR lock timing

Key Levels Calculation Frequency:
- OR_High/Low: 9:45 AM ET (15min after open), fixed for day
- PrevClose/High/Low: Pre-market, fixed for day
- VWAP: Real-time, updated every minute bar
"""

import logging
from datetime import datetime, time, date, timedelta
from typing import Dict, Optional, Any, Union

try:
    import pytz
    NY_TZ = pytz.timezone('America/New_York')
except ImportError:
    NY_TZ = None
    logging.warning("pytz not installed. Timezone handling may be inaccurate.")


logger = logging.getLogger(__name__)


class KeyLevelProvider:
    """
    Provides key price levels for trading decisions.

    FIX 12: Auto-reset on session date change (O(1) global clear)
    FIX 13: Use America/New_York timezone for OR lock timing

    CRITICAL: Gate1 (TriggerGate) and L3 (PriceActionScorer) must use the SAME
    key_levels object to avoid "triggered but no breakout" inconsistency.
    """

    # Opening Range lock time (15 minutes after market open)
    OR_LOCK_TIME = time(9, 45)  # 9:45 AM ET

    def __init__(self, data_provider=None):
        """
        Initialize KeyLevelProvider.

        Args:
            data_provider: Object that provides market data methods:
                - get_opening_range_high(symbol, minutes)
                - get_opening_range_low(symbol, minutes)
                - get_previous_day_high(symbol)
                - get_previous_day_low(symbol)
                - get_previous_day_close(symbol)
                - get_vwap(symbol)
                - get_intraday_atr(symbol, periods)
        """
        self._cache: Dict[str, Dict[str, float]] = {}  # symbol -> key_levels dict
        self._or_locked: Dict[str, bool] = {}  # symbol -> bool (OR period ended)
        self._current_session_date: Optional[date] = None  # FIX 12: Track trading day
        self._data_provider = data_provider

    def get_key_levels(self, symbol: str, bar_time: Union[datetime, date]) -> Dict[str, float]:
        """
        Get key levels for symbol, auto-resetting on new trading day.

        Args:
            symbol: Stock symbol
            bar_time: Current bar timestamp (datetime with timezone or date)

        Returns:
            Dict with key levels:
                - or_high: Opening range high (first 15 min)
                - or_low: Opening range low (first 15 min)
                - prev_high: Previous day high
                - prev_low: Previous day low
                - prev_close: Previous day close
                - vwap: Current VWAP (real-time)
                - vwap_upper: VWAP + 2 * ATR
                - vwap_lower: VWAP - 2 * ATR
        """
        # FIX 12: Auto-reset when session date changes (O(1) global clear)
        bar_date = self._extract_date(bar_time)
        if self._current_session_date != bar_date:
            logger.info(f"New trading day detected: {bar_date}. Resetting all caches.")
            self._reset_all_caches()
            self._current_session_date = bar_date

        # Check if OR period is locked
        if not self._or_locked.get(symbol, False):
            # FIX 13: Use NY timezone for time comparison
            bar_time_ny = self._extract_time_ny(bar_time)
            if bar_time_ny >= self.OR_LOCK_TIME:
                self._or_locked[symbol] = True
                logger.debug(f"{symbol}: Opening Range locked at {bar_time_ny}")

        # Get or compute static key levels
        if symbol not in self._cache:
            self._cache[symbol] = self._compute_key_levels(symbol)

        # VWAP is real-time, always update
        levels = self._cache[symbol].copy()

        if self._data_provider:
            vwap = self._get_vwap(symbol)
            atr = self._get_intraday_atr(symbol, periods=20)

            levels['vwap'] = vwap
            levels['vwap_upper'] = vwap + 2 * atr if vwap and atr else None
            levels['vwap_lower'] = vwap - 2 * atr if vwap and atr else None
        else:
            levels['vwap'] = None
            levels['vwap_upper'] = None
            levels['vwap_lower'] = None

        return levels

    def _compute_key_levels(self, symbol: str) -> Dict[str, float]:
        """
        Compute static key levels (called once per day per symbol).

        Returns:
            Dict with static key levels (OR and previous day)
        """
        if not self._data_provider:
            logger.warning(f"No data provider set. Returning empty levels for {symbol}")
            return {
                'or_high': None,
                'or_low': None,
                'prev_high': None,
                'prev_low': None,
                'prev_close': None,
            }

        return {
            # Opening Range (first 15 min high/low) - FIXED after 9:45 AM
            'or_high': self._get_opening_range_high(symbol, minutes=15),
            'or_low': self._get_opening_range_low(symbol, minutes=15),
            # Yesterday's levels - FIXED for day
            'prev_high': self._get_previous_day_high(symbol),
            'prev_low': self._get_previous_day_low(symbol),
            'prev_close': self._get_previous_day_close(symbol),
        }

    def _reset_all_caches(self) -> None:
        """
        FIX 12: Clear all caches globally - O(1) operation.

        This is much faster than looping through 2000+ symbols.
        """
        self._cache = {}
        self._or_locked = {}

    def _extract_date(self, bar_time: Union[datetime, date]) -> date:
        """
        Extract date from bar_time (handles datetime and date).

        Args:
            bar_time: datetime or date object

        Returns:
            date object
        """
        if isinstance(bar_time, datetime):
            return bar_time.date()
        return bar_time

    def _extract_time_ny(self, bar_time: Union[datetime, date, time]) -> time:
        """
        FIX 13: Extract time component in NY timezone.

        Args:
            bar_time: datetime, date, or time object

        Returns:
            time object in NY timezone
        """
        if isinstance(bar_time, datetime):
            if bar_time.tzinfo is None:
                # Assume already in NY timezone if no tzinfo
                return bar_time.time()
            # Convert to NY timezone
            if NY_TZ:
                ny_time = bar_time.astimezone(NY_TZ)
                return ny_time.time()
            return bar_time.time()
        if isinstance(bar_time, time):
            return bar_time
        # date object - return market open time as default
        return time(9, 30)

    def is_or_locked(self, symbol: str) -> bool:
        """
        Check if Opening Range is locked for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if OR period has ended (after 9:45 AM ET)
        """
        return self._or_locked.get(symbol, False)

    def invalidate_symbol(self, symbol: str) -> None:
        """
        Invalidate cache for a specific symbol.

        Use this when you know data has changed (e.g., corporate action).

        Args:
            symbol: Stock symbol to invalidate
        """
        if symbol in self._cache:
            del self._cache[symbol]
        if symbol in self._or_locked:
            del self._or_locked[symbol]

    def set_data_provider(self, data_provider) -> None:
        """
        Set the data provider for market data access.

        Args:
            data_provider: Object with market data methods
        """
        self._data_provider = data_provider

    # Data provider wrapper methods
    def _get_opening_range_high(self, symbol: str, minutes: int = 15) -> Optional[float]:
        """Get opening range high from data provider."""
        try:
            if hasattr(self._data_provider, 'get_opening_range_high'):
                return self._data_provider.get_opening_range_high(symbol, minutes)
            return None
        except Exception as e:
            logger.error(f"Error getting OR high for {symbol}: {e}")
            return None

    def _get_opening_range_low(self, symbol: str, minutes: int = 15) -> Optional[float]:
        """Get opening range low from data provider."""
        try:
            if hasattr(self._data_provider, 'get_opening_range_low'):
                return self._data_provider.get_opening_range_low(symbol, minutes)
            return None
        except Exception as e:
            logger.error(f"Error getting OR low for {symbol}: {e}")
            return None

    def _get_previous_day_high(self, symbol: str) -> Optional[float]:
        """Get previous day high from data provider."""
        try:
            if hasattr(self._data_provider, 'get_previous_day_high'):
                return self._data_provider.get_previous_day_high(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting prev high for {symbol}: {e}")
            return None

    def _get_previous_day_low(self, symbol: str) -> Optional[float]:
        """Get previous day low from data provider."""
        try:
            if hasattr(self._data_provider, 'get_previous_day_low'):
                return self._data_provider.get_previous_day_low(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting prev low for {symbol}: {e}")
            return None

    def _get_previous_day_close(self, symbol: str) -> Optional[float]:
        """Get previous day close from data provider."""
        try:
            if hasattr(self._data_provider, 'get_previous_day_close'):
                return self._data_provider.get_previous_day_close(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting prev close for {symbol}: {e}")
            return None

    def _get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP from data provider."""
        try:
            if hasattr(self._data_provider, 'get_vwap'):
                return self._data_provider.get_vwap(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting VWAP for {symbol}: {e}")
            return None

    def _get_intraday_atr(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Get intraday ATR from data provider."""
        try:
            if hasattr(self._data_provider, 'get_intraday_atr'):
                return self._data_provider.get_intraday_atr(symbol, periods)
            return None
        except Exception as e:
            logger.error(f"Error getting intraday ATR for {symbol}: {e}")
            return None


# Global singleton instance
_key_level_provider: Optional[KeyLevelProvider] = None


def get_key_level_provider() -> KeyLevelProvider:
    """
    Get the global KeyLevelProvider singleton.

    Returns:
        KeyLevelProvider instance
    """
    global _key_level_provider
    if _key_level_provider is None:
        _key_level_provider = KeyLevelProvider()
    return _key_level_provider


def set_key_level_provider(provider: KeyLevelProvider) -> None:
    """
    Set the global KeyLevelProvider singleton.

    Args:
        provider: KeyLevelProvider instance to use globally
    """
    global _key_level_provider
    _key_level_provider = provider
