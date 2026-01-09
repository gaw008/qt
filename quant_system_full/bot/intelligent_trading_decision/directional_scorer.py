"""
DirectionalScorer - Layer 3 Signal Execution scoring with directional analysis.

This module implements the Layer 3 scoring system with:
- Separate score_long and score_short calculations
- Signal Stability (25%): Direction consistency + trend strength
- Volume Confirmation (45%): Time-of-day adjusted, VWAP deviation, OBV trend
- Price Action (30%): Edge-trigger key levels with decaying continuation

Bug Fixes Implemented:
- FIX 8: Sparse signal density check (signal_density < 0.4 returns 0)
- FIX 9: Use .get() with default dict to prevent KeyError

Weights (adjusted based on reliability):
- Signal Stability: 25% (reduced - easily fooled by oscillation)
- Volume Confirmation: 45% (increased - hardest to fake)
- Price Action: 30% (unchanged)
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple, Union

from .key_level_provider import get_key_level_provider

logger = logging.getLogger(__name__)


class DirectionalScorer:
    """
    Layer 3 Signal Execution scoring with directional analysis.

    Calculates separate scores for LONG and SHORT signals:
    - score_long for BUY signals
    - score_short for SELL signals

    Each score is composed of:
    - Signal Stability (25%): Direction consistency + trend strength
    - Volume Confirmation (45%): Time-of-day adjusted volume analysis
    - Price Action (30%): Key level breakout/breakdown scoring
    """

    # Factor weights
    WEIGHT_STABILITY = 0.25
    WEIGHT_VOLUME = 0.45
    WEIGHT_PRICE_ACTION = 0.30

    # Stability configuration
    STABILITY_DIRECTION_WEIGHT = 0.60
    STABILITY_TREND_WEIGHT = 0.40
    DEFAULT_LOOKBACK = 10  # N bars for stability calculation

    # Volume scoring thresholds
    VOLUME_RATIO_EXCELLENT = 2.0
    VOLUME_RATIO_GOOD = 1.5
    VOLUME_RATIO_NORMAL = 1.0
    VOLUME_RATIO_LOW = 0.7

    # Price action breakout points
    OR_BREAKOUT_POINTS = 40
    VWAP_CROSS_POINTS = 30
    PREV_CLOSE_CROSS_POINTS = 30

    def __init__(self, data_provider=None):
        """
        Initialize DirectionalScorer.

        Args:
            data_provider: Object that provides market data methods:
                - get_current_bar_volume(symbol)
                - get_same_time_avg_volume(symbol, time, lookback_days)
                - get_daily_avg_volume(symbol)
                - get_vwap(symbol)
                - get_intraday_atr(symbol, periods)
                - get_intraday_obv_trend(symbol)
                - get_signal_history(symbol, N) -> List[int] (-1, 0, 1)
                - get_price_history(symbol, N) -> List[float]
        """
        self._data_provider = data_provider

        # Breakout state tracking per symbol
        # Structure: {symbol: {level_name: {'triggered': bool, 'count': int}}}
        self._breakout_state: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Cache for signal/price history
        self._signal_history: Dict[str, List[int]] = {}
        self._price_history: Dict[str, List[float]] = {}

    def score_long(
        self,
        symbol: str,
        price: float,
        prev_price: float,
        bar_time: datetime
    ) -> float:
        """
        Calculate score for LONG (BUY) signal.

        Args:
            symbol: Stock symbol
            price: Current price
            prev_price: Previous bar price
            bar_time: Current bar timestamp

        Returns:
            Score from 0-100
        """
        # Get key levels from shared provider
        key_level_provider = get_key_level_provider()
        key_levels = key_level_provider.get_key_levels(symbol, bar_time)

        # Calculate each factor
        stability_score = self._calculate_stability(symbol, current_signal=1)
        volume_score = self._calculate_volume_score(symbol, bar_time, price, direction='LONG')
        price_action_score = self._calculate_price_action_score(
            symbol, price, prev_price, signal_direction='BUY', key_levels=key_levels
        )

        # Weighted sum
        final_score = (
            self.WEIGHT_STABILITY * stability_score +
            self.WEIGHT_VOLUME * volume_score +
            self.WEIGHT_PRICE_ACTION * price_action_score
        )

        logger.debug(
            f"{symbol} LONG score: stability={stability_score:.1f}, "
            f"volume={volume_score:.1f}, price_action={price_action_score:.1f}, "
            f"final={final_score:.1f}"
        )

        return final_score

    def score_short(
        self,
        symbol: str,
        price: float,
        prev_price: float,
        bar_time: datetime
    ) -> float:
        """
        Calculate score for SHORT (SELL) signal.

        Args:
            symbol: Stock symbol
            price: Current price
            prev_price: Previous bar price
            bar_time: Current bar timestamp

        Returns:
            Score from 0-100
        """
        # Get key levels from shared provider
        key_level_provider = get_key_level_provider()
        key_levels = key_level_provider.get_key_levels(symbol, bar_time)

        # Calculate each factor
        stability_score = self._calculate_stability(symbol, current_signal=-1)
        volume_score = self._calculate_volume_score(symbol, bar_time, price, direction='SHORT')
        price_action_score = self._calculate_price_action_score(
            symbol, price, prev_price, signal_direction='SELL', key_levels=key_levels
        )

        # Weighted sum
        final_score = (
            self.WEIGHT_STABILITY * stability_score +
            self.WEIGHT_VOLUME * volume_score +
            self.WEIGHT_PRICE_ACTION * price_action_score
        )

        logger.debug(
            f"{symbol} SHORT score: stability={stability_score:.1f}, "
            f"volume={volume_score:.1f}, price_action={price_action_score:.1f}, "
            f"final={final_score:.1f}"
        )

        return final_score

    def calculate_stability(
        self,
        signals: List[int],
        prices: List[float],
        current_signal: int,
        N: int = 10
    ) -> float:
        """
        Calculate signal stability score with sparse signal protection.

        Two-component stability score:
        - 60% direction consistency (ratio of same-direction signals)
        - 40% trend strength (magnitude of price move, ATR-capped)

        FIX 8: Sparse signal density check (signal_density < 0.4 returns 0)

        Args:
            signals: List of recent signals (-1=SELL, 0=HOLD, 1=BUY)
            prices: List of recent prices (aligned with signals)
            current_signal: Current signal (-1, 0, or 1)
            N: Lookback period (default 10)

        Returns:
            Score from 0-100
        """
        # Don't calculate stability for HOLD signals
        if current_signal == 0:
            return 0

        # Get N most recent signals/prices
        recent_signals = signals[-N:] if len(signals) >= N else signals
        recent_prices = prices[-N:] if len(prices) >= N else prices

        # Remove HOLD signals for direction analysis
        non_zero_signals = [s for s in recent_signals if s != 0]

        # Minimum signal count requirement
        if len(non_zero_signals) < 3:
            return 0

        # FIX 8: Sparse signal density check
        signal_density = len(non_zero_signals) / max(len(recent_signals), 1)
        if signal_density < 0.4:
            logger.debug(f"Sparse signals: density={signal_density:.2f} < 0.4, returning 0")
            return 0

        # Component 1: Direction consistency (60%)
        same_direction = sum(1 for s in non_zero_signals if s == current_signal)
        direction_ratio = same_direction / len(non_zero_signals)

        # Component 2: Trend strength (40%)
        # Use price change over N bars
        if len(recent_prices) >= 2:
            ret_N = recent_prices[-1] / recent_prices[0] - 1 if recent_prices[0] > 0 else 0
        else:
            ret_N = 0

        # ATR-based cap for cross-stock comparability
        atr = self._get_intraday_atr_for_stability(recent_prices)
        if atr and len(recent_prices) > 0 and recent_prices[-1] > 0:
            # Cap at 1.5 * ATR relative to price
            cap = 1.5 * atr / recent_prices[-1]
        else:
            # Default cap: 0.5% for minute bars
            cap = 0.005

        trend_strength = min(abs(ret_N), cap) / cap if cap > 0 else 0

        # Combined score
        stability = (
            self.STABILITY_DIRECTION_WEIGHT * direction_ratio +
            self.STABILITY_TREND_WEIGHT * trend_strength
        )

        return stability * 100

    def _calculate_stability(self, symbol: str, current_signal: int) -> float:
        """
        Calculate stability score using cached history for a symbol.

        Args:
            symbol: Stock symbol
            current_signal: Current signal (-1=SELL, 1=BUY)

        Returns:
            Score from 0-100
        """
        # Get history from data provider or cache
        signals = self._get_signal_history(symbol)
        prices = self._get_price_history(symbol)

        return self.calculate_stability(
            signals=signals,
            prices=prices,
            current_signal=current_signal,
            N=self.DEFAULT_LOOKBACK
        )

    def calculate_volume_score(
        self,
        symbol: str,
        current_time: datetime,
        current_price: float
    ) -> float:
        """
        Calculate volume confirmation score.

        Three components:
        - Volume Ratio (40%): Current vs time-of-day historical average
        - VWAP Deviation (30%): Price position relative to VWAP (ATR units)
        - OBV Trend (30%): Intraday On-Balance Volume trend

        FIX: avg_floor protection (max(1000, daily_avg / 390 * 0.1))

        Args:
            symbol: Stock symbol
            current_time: Current bar timestamp
            current_price: Current price

        Returns:
            Score from 0-100
        """
        return self._calculate_volume_score(symbol, current_time, current_price, direction='LONG')

    def _calculate_volume_score(
        self,
        symbol: str,
        current_time: datetime,
        current_price: float,
        direction: str = 'LONG'
    ) -> float:
        """
        Internal volume score calculation with direction awareness.

        Args:
            symbol: Stock symbol
            current_time: Current bar timestamp
            current_price: Current price
            direction: 'LONG' or 'SHORT'

        Returns:
            Score from 0-100
        """
        # Component 1: Volume Ratio (40%)
        volume_ratio_score = self._score_volume_ratio(symbol, current_time)

        # Component 2: VWAP Deviation (30%)
        vwap_score = self._score_vwap_deviation(symbol, current_price, direction)

        # Component 3: OBV Trend (30%)
        obv_score = self._score_obv_trend(symbol, direction)

        total_score = (
            0.4 * volume_ratio_score +
            0.3 * vwap_score +
            0.3 * obv_score
        )

        return total_score

    def _score_volume_ratio(self, symbol: str, current_time: datetime) -> float:
        """
        Score volume ratio vs time-of-day adjusted historical average.

        FIX: avg_floor protection to prevent division by near-zero.

        Args:
            symbol: Stock symbol
            current_time: Current bar timestamp

        Returns:
            Score from 0-100
        """
        current_volume = self._get_current_bar_volume(symbol)
        if current_volume is None or current_volume <= 0:
            return 50  # Neutral score if no data

        # Get time-of-day adjusted historical average
        historical_avg = self._get_same_time_avg_volume(symbol, current_time)

        # FIX: Protect against near-zero historical_avg
        daily_avg = self._get_daily_avg_volume(symbol) or 0
        avg_floor = max(1000, daily_avg / 390 * 0.1)  # At least 1000 shares
        historical_avg = max(historical_avg or avg_floor, avg_floor)

        volume_ratio = current_volume / historical_avg

        # Score based on ratio thresholds
        if volume_ratio >= self.VOLUME_RATIO_EXCELLENT:
            return 100
        elif volume_ratio >= self.VOLUME_RATIO_GOOD:
            return 80
        elif volume_ratio >= self.VOLUME_RATIO_NORMAL:
            return 60
        elif volume_ratio >= self.VOLUME_RATIO_LOW:
            return 40
        else:
            return 20

    def _score_vwap_deviation(
        self,
        symbol: str,
        current_price: float,
        direction: str
    ) -> float:
        """
        Score VWAP deviation in ATR units.

        For BUY: Above VWAP is good
        For SELL: Below VWAP is good

        Args:
            symbol: Stock symbol
            current_price: Current price
            direction: 'LONG' or 'SHORT'

        Returns:
            Score from 0-100
        """
        vwap = self._get_vwap(symbol)
        atr_intraday = self._get_intraday_atr(symbol)

        if vwap is None or atr_intraday is None or atr_intraday <= 0:
            return 50  # Neutral score if no data

        # Deviation in ATR units
        deviation = (current_price - vwap) / atr_intraday

        # Direction-aware scoring
        if direction == 'LONG':
            # For BUY: above VWAP is good
            if deviation >= 1.0:
                return 100
            elif deviation >= 0.5:
                return 80
            elif deviation >= 0.0:
                return 60
            elif deviation >= -0.5:
                return 40
            else:
                return 20
        else:  # SHORT
            # For SELL: below VWAP is good (invert the logic)
            if deviation <= -1.0:
                return 100
            elif deviation <= -0.5:
                return 80
            elif deviation <= 0.0:
                return 60
            elif deviation <= 0.5:
                return 40
            else:
                return 20

    def _score_obv_trend(self, symbol: str, direction: str) -> float:
        """
        Score intraday OBV (On-Balance Volume) trend.

        For BUY: Positive OBV trend is good
        For SELL: Negative OBV trend is good

        Args:
            symbol: Stock symbol
            direction: 'LONG' or 'SHORT'

        Returns:
            Score from 0-100
        """
        obv_trend = self._get_intraday_obv_trend(symbol)

        if obv_trend is None:
            return 50  # Neutral score if no data

        # OBV trend normalized to [-1, 1] range
        # Positive means buying pressure, negative means selling pressure
        if direction == 'LONG':
            if obv_trend >= 0.5:
                return 100
            elif obv_trend >= 0.2:
                return 80
            elif obv_trend >= 0:
                return 60
            elif obv_trend >= -0.3:
                return 40
            else:
                return 20
        else:  # SHORT
            if obv_trend <= -0.5:
                return 100
            elif obv_trend <= -0.2:
                return 80
            elif obv_trend <= 0:
                return 60
            elif obv_trend <= 0.3:
                return 40
            else:
                return 20

    def calculate_price_action_score(
        self,
        price: float,
        prev_price: float,
        signal_direction: str,
        key_levels: Dict[str, Optional[float]]
    ) -> float:
        """
        Calculate price action score with edge-trigger key levels.

        High score only on ACTUAL breakout bar. After breakout,
        switch to continuation mode with decaying scores.

        FIX 9: Use .get() with default dict to prevent KeyError

        Args:
            price: Current price
            prev_price: Previous bar price
            signal_direction: 'BUY' or 'SELL'
            key_levels: Dict with 'or_high', 'or_low', 'vwap', 'prev_close'

        Returns:
            Score from 0-100
        """
        # Use a dummy symbol for external calls
        return self._calculate_price_action_score(
            symbol='_external_',
            price=price,
            prev_price=prev_price,
            signal_direction=signal_direction,
            key_levels=key_levels
        )

    def _calculate_price_action_score(
        self,
        symbol: str,
        price: float,
        prev_price: float,
        signal_direction: str,
        key_levels: Dict[str, Optional[float]]
    ) -> float:
        """
        Internal price action calculation with per-symbol state tracking.

        Decaying continuation score:
        - Fresh breakout: full points (40 for OR, 30 for others)
        - 1st continuation: decay[0]
        - 2nd continuation: decay[1]
        - 3rd+ continuation: 0

        Args:
            symbol: Stock symbol
            price: Current price
            prev_price: Previous bar price
            signal_direction: 'BUY' or 'SELL'
            key_levels: Dict with key price levels

        Returns:
            Score from 0-100
        """
        # FIX 9: Use .get() with default empty dict
        state = self._breakout_state.get(symbol, {})
        score = 0

        if signal_direction == 'BUY':
            # OR High breakout
            or_high = key_levels.get('or_high')
            if or_high is not None:
                score += self._score_level(
                    prev_price, price, or_high,
                    state, 'or_high', direction='up',
                    breakout_pts=self.OR_BREAKOUT_POINTS,
                    continuation_decay=[20, 10, 0]
                )

            # VWAP cross up
            vwap = key_levels.get('vwap')
            if vwap is not None:
                score += self._score_level(
                    prev_price, price, vwap,
                    state, 'vwap', direction='up',
                    breakout_pts=self.VWAP_CROSS_POINTS,
                    continuation_decay=[15, 7, 0]
                )

            # Prev close cross up
            prev_close = key_levels.get('prev_close')
            if prev_close is not None:
                score += self._score_level(
                    prev_price, price, prev_close,
                    state, 'prev_close', direction='up',
                    breakout_pts=self.PREV_CLOSE_CROSS_POINTS,
                    continuation_decay=[15, 7, 0]
                )

        elif signal_direction == 'SELL':
            # OR Low breakdown
            or_low = key_levels.get('or_low')
            if or_low is not None:
                score += self._score_level(
                    prev_price, price, or_low,
                    state, 'or_low', direction='down',
                    breakout_pts=self.OR_BREAKOUT_POINTS,
                    continuation_decay=[20, 10, 0]
                )

            # VWAP cross down
            vwap = key_levels.get('vwap')
            if vwap is not None:
                score += self._score_level(
                    prev_price, price, vwap,
                    state, 'vwap_down', direction='down',
                    breakout_pts=self.VWAP_CROSS_POINTS,
                    continuation_decay=[15, 7, 0]
                )

            # Prev close cross down
            prev_close = key_levels.get('prev_close')
            if prev_close is not None:
                score += self._score_level(
                    prev_price, price, prev_close,
                    state, 'prev_close_down', direction='down',
                    breakout_pts=self.PREV_CLOSE_CROSS_POINTS,
                    continuation_decay=[15, 7, 0]
                )

        # Save state back
        self._breakout_state[symbol] = state

        # Cap at 100
        return min(score, 100)

    def _score_level(
        self,
        prev_price: float,
        price: float,
        level: float,
        state: Dict[str, Dict[str, Any]],
        level_name: str,
        direction: str,
        breakout_pts: int,
        continuation_decay: List[int]
    ) -> int:
        """
        Score a single key level with edge-trigger and decay.

        FIX 9: Use .get() with default dict to prevent KeyError

        Args:
            prev_price: Previous bar price
            price: Current price
            level: Key level price
            state: Breakout state dict (modified in place)
            level_name: Name of level for state tracking
            direction: 'up' or 'down'
            breakout_pts: Points for fresh breakout
            continuation_decay: [1st_cont, 2nd_cont, 3rd+_cont]

        Returns:
            Points scored (0 to breakout_pts)
        """
        # FIX 9: Use .get() with default dict
        lvl = state.get(level_name, {'triggered': False, 'count': 0})

        # Check for cross
        is_cross = (
            (direction == 'up' and prev_price <= level < price) or
            (direction == 'down' and prev_price >= level > price)
        )

        # Check if holding above/below level
        is_holding = (
            (direction == 'up' and price > level) or
            (direction == 'down' and price < level)
        )

        if is_cross:
            # Fresh breakout
            lvl = {'triggered': True, 'count': 0}
            state[level_name] = lvl
            return breakout_pts

        elif lvl.get('triggered', False) and is_holding:
            # Continuation mode with decay
            count = lvl.get('count', 0)
            lvl['count'] = count + 1
            state[level_name] = lvl

            if count < len(continuation_decay):
                return continuation_decay[count]
            return 0

        else:
            # Reset if price retreats back through level
            lvl = {'triggered': False, 'count': 0}
            state[level_name] = lvl
            return 0

    def reset_daily(self, symbol: str) -> None:
        """
        Reset daily state for a symbol.

        Call at market open to clear breakout states.

        Args:
            symbol: Stock symbol to reset
        """
        if symbol in self._breakout_state:
            del self._breakout_state[symbol]
        if symbol in self._signal_history:
            del self._signal_history[symbol]
        if symbol in self._price_history:
            del self._price_history[symbol]

        logger.debug(f"DirectionalScorer: Daily reset for {symbol}")

    def reset_all_daily(self) -> None:
        """Reset daily state for all symbols."""
        self._breakout_state = {}
        self._signal_history = {}
        self._price_history = {}
        logger.info("DirectionalScorer: Daily reset completed for all symbols")

    def update_history(
        self,
        symbol: str,
        signal: int,
        price: float
    ) -> None:
        """
        Update signal and price history for a symbol.

        Args:
            symbol: Stock symbol
            signal: Signal value (-1=SELL, 0=HOLD, 1=BUY)
            price: Current price
        """
        # Initialize if needed
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        # Append new data
        self._signal_history[symbol].append(signal)
        self._price_history[symbol].append(price)

        # Keep only last 50 bars
        max_history = 50
        if len(self._signal_history[symbol]) > max_history:
            self._signal_history[symbol] = self._signal_history[symbol][-max_history:]
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]

    def set_data_provider(self, data_provider) -> None:
        """Set the data provider for market data access."""
        self._data_provider = data_provider

    def get_breakout_state(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Get current breakout state for a symbol (for debugging/monitoring).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with breakout state for each level
        """
        return self._breakout_state.get(symbol, {}).copy()

    def get_score_breakdown(
        self,
        symbol: str,
        price: float,
        prev_price: float,
        bar_time: datetime,
        signal_direction: str
    ) -> Dict[str, Any]:
        """
        Get detailed score breakdown for monitoring/debugging.

        Args:
            symbol: Stock symbol
            price: Current price
            prev_price: Previous bar price
            bar_time: Current bar timestamp
            signal_direction: 'BUY' or 'SELL'

        Returns:
            Dict with all score components
        """
        key_level_provider = get_key_level_provider()
        key_levels = key_level_provider.get_key_levels(symbol, bar_time)

        current_signal = 1 if signal_direction == 'BUY' else -1
        direction = 'LONG' if signal_direction == 'BUY' else 'SHORT'

        stability = self._calculate_stability(symbol, current_signal)
        volume = self._calculate_volume_score(symbol, bar_time, price, direction)

        # Calculate price action without modifying state
        state_backup = self._breakout_state.get(symbol, {}).copy()
        price_action = self._calculate_price_action_score(
            symbol, price, prev_price, signal_direction, key_levels
        )
        # Restore state
        if state_backup:
            self._breakout_state[symbol] = state_backup

        final_score = (
            self.WEIGHT_STABILITY * stability +
            self.WEIGHT_VOLUME * volume +
            self.WEIGHT_PRICE_ACTION * price_action
        )

        return {
            'symbol': symbol,
            'direction': signal_direction,
            'price': price,
            'prev_price': prev_price,
            'components': {
                'stability': round(stability, 2),
                'stability_weighted': round(self.WEIGHT_STABILITY * stability, 2),
                'volume': round(volume, 2),
                'volume_weighted': round(self.WEIGHT_VOLUME * volume, 2),
                'price_action': round(price_action, 2),
                'price_action_weighted': round(self.WEIGHT_PRICE_ACTION * price_action, 2),
            },
            'key_levels': {
                k: round(v, 4) if v is not None else None
                for k, v in key_levels.items()
            },
            'breakout_state': self._breakout_state.get(symbol, {}),
            'final_score': round(final_score, 2),
        }

    # Data provider wrapper methods
    def _get_current_bar_volume(self, symbol: str) -> Optional[float]:
        """Get current bar volume from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_current_bar_volume'):
                return self._data_provider.get_current_bar_volume(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting current bar volume for {symbol}: {e}")
            return None

    def _get_same_time_avg_volume(
        self,
        symbol: str,
        current_time: datetime,
        lookback_days: int = 20
    ) -> Optional[float]:
        """Get time-of-day adjusted historical average volume."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_same_time_avg_volume'):
                return self._data_provider.get_same_time_avg_volume(
                    symbol, current_time, lookback_days
                )
            return None
        except Exception as e:
            logger.error(f"Error getting same-time avg volume for {symbol}: {e}")
            return None

    def _get_daily_avg_volume(self, symbol: str) -> Optional[float]:
        """Get daily average volume from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_daily_avg_volume'):
                return self._data_provider.get_daily_avg_volume(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting daily avg volume for {symbol}: {e}")
            return None

    def _get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_vwap'):
                return self._data_provider.get_vwap(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting VWAP for {symbol}: {e}")
            return None

    def _get_intraday_atr(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Get intraday ATR from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_intraday_atr'):
                return self._data_provider.get_intraday_atr(symbol, periods)
            return None
        except Exception as e:
            logger.error(f"Error getting intraday ATR for {symbol}: {e}")
            return None

    def _get_intraday_atr_for_stability(
        self,
        prices: List[float],
        periods: int = 20
    ) -> Optional[float]:
        """
        Calculate ATR-like metric from price history for stability calculation.

        Args:
            prices: List of recent prices
            periods: Number of periods for calculation

        Returns:
            ATR estimate or None
        """
        if len(prices) < 2:
            return None

        # Simple true range approximation from price changes
        changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        if not changes:
            return None

        # Use average of recent changes as ATR estimate
        recent_changes = changes[-min(periods, len(changes)):]
        return sum(recent_changes) / len(recent_changes)

    def _get_intraday_obv_trend(self, symbol: str) -> Optional[float]:
        """Get intraday OBV trend from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_intraday_obv_trend'):
                return self._data_provider.get_intraday_obv_trend(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting OBV trend for {symbol}: {e}")
            return None

    def _get_signal_history(self, symbol: str) -> List[int]:
        """Get signal history from cache or data provider."""
        # First check local cache
        if symbol in self._signal_history:
            return self._signal_history[symbol]

        # Try data provider
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_signal_history'):
                history = self._data_provider.get_signal_history(symbol, self.DEFAULT_LOOKBACK)
                if history:
                    self._signal_history[symbol] = list(history)
                    return self._signal_history[symbol]
        except Exception as e:
            logger.error(f"Error getting signal history for {symbol}: {e}")

        return []

    def _get_price_history(self, symbol: str) -> List[float]:
        """Get price history from cache or data provider."""
        # First check local cache
        if symbol in self._price_history:
            return self._price_history[symbol]

        # Try data provider
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_price_history'):
                history = self._data_provider.get_price_history(symbol, self.DEFAULT_LOOKBACK)
                if history:
                    self._price_history[symbol] = list(history)
                    return self._price_history[symbol]
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")

        return []


# Global singleton instance
_directional_scorer: Optional[DirectionalScorer] = None


def get_directional_scorer() -> DirectionalScorer:
    """
    Get the global DirectionalScorer singleton.

    Returns:
        DirectionalScorer instance
    """
    global _directional_scorer
    if _directional_scorer is None:
        _directional_scorer = DirectionalScorer()
    return _directional_scorer


def set_directional_scorer(scorer: DirectionalScorer) -> None:
    """
    Set the global DirectionalScorer singleton.

    Args:
        scorer: DirectionalScorer instance to use globally
    """
    global _directional_scorer
    _directional_scorer = scorer
