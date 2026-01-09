"""
StockSelectionFilter - Layer 1 of the Intelligent Trading Decision System.

This module implements the Stock Selection Filter (Layer 1) from the plan:
- 12-1 Month Momentum calculation (CORRECT direction)
- Historical win rate calculation with cold start handling
- Hysteresis buffer (Entry >= 70, Exit < 60)
- Mandatory tradability filters (hard rules)

Frequency: Daily close (NOT hourly - avoid unnecessary churn)

Purpose: Filter the stock universe to a tradeable pool of symbols
that meet momentum, win rate, and tradability criteria.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


class TradabilityFilters:
    """
    Mandatory tradability filters - hard rules that must ALL pass.

    These are non-negotiable requirements for a stock to be tradeable.
    """

    def __init__(self):
        """Initialize with default filter values."""
        self.min_adv = 20_000_000      # Average Daily Volume > $20M
        self.min_price = 5.0            # Price > $5 (avoid penny stock noise)
        self.max_spread_pct = 0.5       # Spread < 0.5%
        self.min_atr_pct = 0.5          # ATR% > 0.5% (enough volatility)
        self.max_atr_pct = 5.0          # ATR% < 5% (not too crazy)

    def to_dict(self) -> Dict[str, float]:
        """Return filters as a dictionary."""
        return {
            'min_adv': self.min_adv,
            'min_price': self.min_price,
            'max_spread_pct': self.max_spread_pct,
            'min_atr_pct': self.min_atr_pct,
            'max_atr_pct': self.max_atr_pct,
        }


class StockSelectionFilter:
    """
    Layer 1: Stock Selection Filter for the Intelligent Trading Decision System.

    Implements:
    - 12-1 Month Momentum scoring
    - Historical win rate calculation with cold start handling
    - Hysteresis buffer for pool entry/exit
    - Mandatory tradability filters

    Entry threshold: Score >= 70
    Exit threshold: Score < 60 (10-point hysteresis buffer)
    """

    # Hysteresis thresholds to avoid flip-flop behavior
    ENTRY_THRESHOLD = 70.0
    EXIT_THRESHOLD = 60.0

    # Weight configuration for selection score
    MOMENTUM_WEIGHT = 0.6   # 60% weight on 12-1 momentum
    WIN_RATE_WEIGHT = 0.4   # 40% weight on historical win rate

    def __init__(self, data_provider=None):
        """
        Initialize StockSelectionFilter.

        Args:
            data_provider: Object that provides market data methods:
                - get_price_history(symbol, days) -> List[float]
                - get_average_daily_volume_dollars(symbol) -> float
                - get_current_price(symbol) -> float
                - get_bid_ask_spread_pct(symbol) -> float
                - get_atr_percent(symbol, periods) -> float
        """
        self._data_provider = data_provider
        self.tradability_filters = TradabilityFilters()

        # Trade history for win rate calculation
        # Structure: {symbol: [{'profit': float, 'timestamp': datetime}, ...]}
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {}

        # Sector-level statistics for cold start
        self._sector_stats: Dict[str, float] = {}

        # System-wide baseline win rate
        self._system_win_rate = 0.50  # Default 50% if no history

        # Last calculated scores for caching
        self._cached_scores: Dict[str, Tuple[float, datetime]] = {}
        self._score_cache_ttl_hours = 4  # Refresh scores every 4 hours

    def filter_universe(self, symbols: List[str]) -> List[str]:
        """
        Apply mandatory tradability filters to a list of symbols.

        These are hard rules - a symbol must pass ALL filters to be considered.

        Args:
            symbols: List of stock symbols to filter

        Returns:
            List of symbols that pass all tradability filters
        """
        passed_symbols = []

        for symbol in symbols:
            passed, reason = self._check_tradability(symbol)
            if passed:
                passed_symbols.append(symbol)
            else:
                logger.debug(f"{symbol}: Failed tradability - {reason}")

        logger.info(
            f"Tradability filter: {len(passed_symbols)}/{len(symbols)} symbols passed"
        )

        return passed_symbols

    def _check_tradability(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a symbol passes all tradability filters.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        filters = self.tradability_filters

        # 1. Check Average Daily Volume (in dollars)
        adv = self._get_adv_dollars(symbol)
        if adv is None:
            return False, "No ADV data"
        if adv < filters.min_adv:
            return False, f"ADV ${adv/1_000_000:.1f}M < ${filters.min_adv/1_000_000:.0f}M"

        # 2. Check price
        price = self._get_current_price(symbol)
        if price is None:
            return False, "No price data"
        if price < filters.min_price:
            return False, f"Price ${price:.2f} < ${filters.min_price:.2f}"

        # 3. Check spread
        spread_pct = self._get_spread_pct(symbol)
        if spread_pct is not None and spread_pct > filters.max_spread_pct:
            return False, f"Spread {spread_pct:.2f}% > {filters.max_spread_pct}%"

        # 4. Check ATR%
        atr_pct = self._get_atr_pct(symbol)
        if atr_pct is not None:
            if atr_pct < filters.min_atr_pct:
                return False, f"ATR% {atr_pct:.2f}% < {filters.min_atr_pct}%"
            if atr_pct > filters.max_atr_pct:
                return False, f"ATR% {atr_pct:.2f}% > {filters.max_atr_pct}%"

        return True, "OK"

    def calculate_momentum_score(self, symbol: str) -> Optional[float]:
        """
        Calculate 12-1 Month Momentum score.

        CRITICAL - CORRECT FORMULA:
        mom_12_1 = (price_21_days_ago / price_252_days_ago) - 1

        This skips the most recent month to avoid short-term reversal effects.

        Args:
            symbol: Stock symbol

        Returns:
            Momentum score (0-100) or None if insufficient data
        """
        # Get price history - need at least 252 trading days
        prices = self._get_price_history(symbol, days=260)  # Extra buffer

        if prices is None or len(prices) < 252:
            logger.debug(f"{symbol}: Insufficient price history for momentum")
            return None

        # CORRECT FORMULA:
        # price_21 = 21 trading days ago (about 1 month)
        # price_252 = 252 trading days ago (about 12 months)
        try:
            price_21_days_ago = prices[-21] if len(prices) >= 21 else prices[0]
            price_252_days_ago = prices[-252] if len(prices) >= 252 else prices[0]

            if price_252_days_ago <= 0:
                logger.warning(f"{symbol}: Invalid historical price")
                return None

            # Calculate raw momentum
            mom_12_1 = (price_21_days_ago / price_252_days_ago) - 1

            # Convert to score (0-100)
            # Typical range: -50% to +100% momentum
            # Map to 0-100 score with 50 as neutral
            score = self._momentum_to_score(mom_12_1)

            logger.debug(f"{symbol}: 12-1 momentum={mom_12_1:.2%}, score={score:.1f}")
            return score

        except (IndexError, ZeroDivisionError) as e:
            logger.warning(f"{symbol}: Error calculating momentum: {e}")
            return None

    def _momentum_to_score(self, momentum: float) -> float:
        """
        Convert raw momentum to a 0-100 score.

        Mapping:
        - momentum <= -0.30 (-30%) -> score = 0
        - momentum = 0 (0%) -> score = 50
        - momentum >= 0.60 (+60%) -> score = 100

        Linear interpolation between these points.

        Args:
            momentum: Raw momentum value (e.g., 0.25 for +25%)

        Returns:
            Score from 0 to 100
        """
        if momentum <= -0.30:
            return 0.0
        elif momentum >= 0.60:
            return 100.0
        elif momentum <= 0:
            # Map [-0.30, 0] to [0, 50]
            return 50.0 * (momentum + 0.30) / 0.30
        else:
            # Map [0, 0.60] to [50, 100]
            return 50.0 + 50.0 * momentum / 0.60

    def calculate_win_rate_score(
        self,
        symbol: str,
        trade_history: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Calculate win rate score with cold start handling.

        Cold Start Hierarchy:
        1. If >= 5 trades: Use stock's actual win rate (full confidence)
        2. If 1-4 trades: Blend stock + system average (partial confidence)
        3. If 0 trades: Use sector average or system average (low confidence)

        Args:
            symbol: Stock symbol
            trade_history: Optional trade history for this symbol
                          If None, uses internal _trade_history

        Returns:
            Win rate score (0-100)
        """
        # Get trade history for this symbol
        if trade_history is None:
            trades = self._trade_history.get(symbol, [])
        else:
            trades = trade_history

        trade_count = len(trades)

        if trade_count >= 5:
            # Full confidence - use stock's actual win rate
            wins = sum(1 for t in trades if t.get('profit', 0) > 0)
            win_rate = wins / trade_count
            confidence = min(1.0, trade_count / 20)  # Full confidence at 20+ trades

        elif trade_count >= 1:
            # Partial confidence - blend stock + system average
            wins = sum(1 for t in trades if t.get('profit', 0) > 0)
            stock_win_rate = wins / trade_count
            win_rate = (stock_win_rate + self._system_win_rate) / 2
            confidence = 0.3 + 0.15 * trade_count  # 0.45 to 0.9

        else:
            # Cold start - use sector average or system average
            sector = self._get_sector(symbol)
            win_rate = self._sector_stats.get(sector, self._system_win_rate)
            confidence = 0.2  # Low confidence for cold start

        # Final score = weighted average of win_rate and system baseline
        final_win_rate = win_rate * confidence + self._system_win_rate * (1 - confidence)

        # Convert to 0-100 score
        # Win rate 30% -> 0, Win rate 70% -> 100
        score = max(0, min(100, (final_win_rate - 0.30) / 0.40 * 100))

        logger.debug(
            f"{symbol}: win_rate={win_rate:.2%}, confidence={confidence:.2f}, "
            f"score={score:.1f}"
        )

        return score

    def calculate_selection_score(self, symbol: str) -> Optional[float]:
        """
        Calculate combined selection score (momentum + win rate).

        Formula:
        selection_score = MOMENTUM_WEIGHT * momentum_score + WIN_RATE_WEIGHT * win_rate_score

        Args:
            symbol: Stock symbol

        Returns:
            Combined selection score (0-100) or None if cannot calculate
        """
        # Check cache first
        cached = self._cached_scores.get(symbol)
        if cached:
            score, timestamp = cached
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            if age_hours < self._score_cache_ttl_hours:
                return score

        # Calculate momentum score
        momentum_score = self.calculate_momentum_score(symbol)
        if momentum_score is None:
            # Use neutral score if no momentum data
            momentum_score = 50.0
            logger.debug(f"{symbol}: Using neutral momentum score due to missing data")

        # Calculate win rate score
        win_rate_score = self.calculate_win_rate_score(symbol)

        # Combined score
        selection_score = (
            self.MOMENTUM_WEIGHT * momentum_score +
            self.WIN_RATE_WEIGHT * win_rate_score
        )

        # Cache the result
        self._cached_scores[symbol] = (selection_score, datetime.now())

        logger.debug(
            f"{symbol}: Selection score={selection_score:.1f} "
            f"(momentum={momentum_score:.1f}, win_rate={win_rate_score:.1f})"
        )

        return selection_score

    def should_enter_pool(self, symbol: str, score: Optional[float] = None) -> bool:
        """
        Check if a symbol should enter the tradeable pool.

        Entry requires score >= ENTRY_THRESHOLD (70).

        Args:
            symbol: Stock symbol
            score: Pre-calculated selection score (optional)

        Returns:
            True if symbol should enter pool
        """
        if score is None:
            score = self.calculate_selection_score(symbol)

        if score is None:
            return False

        return score >= self.ENTRY_THRESHOLD

    def should_exit_pool(self, symbol: str, score: Optional[float] = None) -> bool:
        """
        Check if a symbol should exit the tradeable pool.

        Exit requires score < EXIT_THRESHOLD (60).
        Note: This is different from entry to create hysteresis buffer.

        Args:
            symbol: Stock symbol
            score: Pre-calculated selection score (optional)

        Returns:
            True if symbol should exit pool
        """
        if score is None:
            score = self.calculate_selection_score(symbol)

        if score is None:
            return True  # Exit if we can't calculate score

        return score < self.EXIT_THRESHOLD

    def update_pool(
        self,
        current_pool: Set[str],
        candidates: List[str]
    ) -> Set[str]:
        """
        Update the tradeable pool with hysteresis logic.

        Process:
        1. Filter candidates through tradability filters
        2. For symbols NOT in pool: Add if score >= ENTRY_THRESHOLD
        3. For symbols IN pool: Remove only if score < EXIT_THRESHOLD

        The 10-point hysteresis buffer (70 entry, 60 exit) prevents
        symbols from rapidly entering and exiting the pool.

        Args:
            current_pool: Current set of symbols in the pool
            candidates: List of candidate symbols to consider

        Returns:
            Updated pool set
        """
        new_pool = set()

        # First, filter through tradability requirements
        tradeable = self.filter_universe(candidates)
        tradeable_set = set(tradeable)

        # Track changes for logging
        entries = []
        exits = []
        retained = []

        # Process current pool members
        for symbol in current_pool:
            if symbol not in tradeable_set:
                # Failed tradability - must exit
                exits.append((symbol, "tradability"))
                continue

            score = self.calculate_selection_score(symbol)
            if self.should_exit_pool(symbol, score):
                exits.append((symbol, f"score={score:.1f}<{self.EXIT_THRESHOLD}"))
            else:
                new_pool.add(symbol)
                retained.append(symbol)

        # Consider new entries
        for symbol in tradeable:
            if symbol in new_pool:
                continue  # Already retained from current pool

            score = self.calculate_selection_score(symbol)
            if self.should_enter_pool(symbol, score):
                new_pool.add(symbol)
                entries.append((symbol, f"score={score:.1f}>={self.ENTRY_THRESHOLD}"))

        # Log summary
        logger.info(
            f"Pool update: {len(entries)} entries, {len(exits)} exits, "
            f"{len(retained)} retained | Final pool: {len(new_pool)} symbols"
        )

        if entries:
            logger.info(f"Entries: {[e[0] for e in entries[:10]]}...")
        if exits:
            logger.info(f"Exits: {[e[0] for e in exits[:10]]}...")

        return new_pool

    # Trade history management
    def record_trade(
        self,
        symbol: str,
        profit: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade result for win rate calculation.

        Args:
            symbol: Stock symbol
            profit: Trade profit (positive = win, negative = loss)
            timestamp: Trade timestamp (default: now)
        """
        if symbol not in self._trade_history:
            self._trade_history[symbol] = []

        self._trade_history[symbol].append({
            'profit': profit,
            'timestamp': timestamp or datetime.now()
        })

        # Invalidate cache for this symbol
        if symbol in self._cached_scores:
            del self._cached_scores[symbol]

        # Update system-wide win rate
        self._update_system_win_rate()

    def _update_system_win_rate(self) -> None:
        """Update the system-wide win rate from all trade history."""
        all_trades = []
        for trades in self._trade_history.values():
            all_trades.extend(trades)

        if len(all_trades) >= 10:
            wins = sum(1 for t in all_trades if t.get('profit', 0) > 0)
            self._system_win_rate = wins / len(all_trades)

    def update_sector_stats(self, sector_stats: Dict[str, float]) -> None:
        """
        Update sector-level win rate statistics.

        Args:
            sector_stats: Dict mapping sector name to win rate
        """
        self._sector_stats = sector_stats

    def load_trade_history(self, history: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Load trade history from external source.

        Args:
            history: Dict mapping symbol to list of trade records
        """
        self._trade_history = history
        self._update_system_win_rate()
        logger.info(f"Loaded trade history for {len(history)} symbols")

    # Data provider methods
    def set_data_provider(self, data_provider) -> None:
        """Set the data provider for market data access."""
        self._data_provider = data_provider

    def _get_price_history(self, symbol: str, days: int) -> Optional[List[float]]:
        """Get price history from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_price_history'):
                return self._data_provider.get_price_history(symbol, days)
            return None
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return None

    def _get_adv_dollars(self, symbol: str) -> Optional[float]:
        """Get average daily volume in dollars from data provider."""
        try:
            if self._data_provider:
                # Try specific method first
                if hasattr(self._data_provider, 'get_average_daily_volume_dollars'):
                    return self._data_provider.get_average_daily_volume_dollars(symbol)

                # Fallback: Calculate from volume and price
                if hasattr(self._data_provider, 'get_average_daily_volume'):
                    volume = self._data_provider.get_average_daily_volume(symbol)
                    price = self._get_current_price(symbol)
                    if volume and price:
                        return volume * price
            return None
        except Exception as e:
            logger.error(f"Error getting ADV for {symbol}: {e}")
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_current_price'):
                return self._data_provider.get_current_price(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def _get_spread_pct(self, symbol: str) -> Optional[float]:
        """Get bid-ask spread percentage from data provider."""
        try:
            if self._data_provider:
                # Try specific method first
                if hasattr(self._data_provider, 'get_bid_ask_spread_pct'):
                    return self._data_provider.get_bid_ask_spread_pct(symbol)

                # Fallback: Calculate from bid/ask
                if hasattr(self._data_provider, 'get_bid_ask'):
                    bid, ask = self._data_provider.get_bid_ask(symbol)
                    if bid and ask and bid > 0:
                        return (ask - bid) / bid * 100
            return None
        except Exception as e:
            logger.error(f"Error getting spread for {symbol}: {e}")
            return None

    def _get_atr_pct(self, symbol: str, periods: int = 14) -> Optional[float]:
        """Get ATR as percentage of price from data provider."""
        try:
            if self._data_provider:
                # Try specific method first
                if hasattr(self._data_provider, 'get_atr_percent'):
                    return self._data_provider.get_atr_percent(symbol, periods)

                # Fallback: Calculate from ATR and price
                if hasattr(self._data_provider, 'get_atr'):
                    atr = self._data_provider.get_atr(symbol, periods)
                    price = self._get_current_price(symbol)
                    if atr and price and price > 0:
                        return atr / price * 100
            return None
        except Exception as e:
            logger.error(f"Error getting ATR% for {symbol}: {e}")
            return None

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_sector'):
                sector = self._data_provider.get_sector(symbol)
                if sector:
                    return sector
            return 'Unknown'
        except Exception as e:
            logger.debug(f"Error getting sector for {symbol}: {e}")
            return 'Unknown'

    # Configuration methods
    def set_tradability_filters(
        self,
        min_adv: Optional[float] = None,
        min_price: Optional[float] = None,
        max_spread_pct: Optional[float] = None,
        min_atr_pct: Optional[float] = None,
        max_atr_pct: Optional[float] = None
    ) -> None:
        """
        Update tradability filter parameters.

        Args:
            min_adv: Minimum average daily volume in dollars
            min_price: Minimum stock price
            max_spread_pct: Maximum bid-ask spread percentage
            min_atr_pct: Minimum ATR percentage
            max_atr_pct: Maximum ATR percentage
        """
        if min_adv is not None:
            self.tradability_filters.min_adv = min_adv
        if min_price is not None:
            self.tradability_filters.min_price = min_price
        if max_spread_pct is not None:
            self.tradability_filters.max_spread_pct = max_spread_pct
        if min_atr_pct is not None:
            self.tradability_filters.min_atr_pct = min_atr_pct
        if max_atr_pct is not None:
            self.tradability_filters.max_atr_pct = max_atr_pct

        logger.info(f"Tradability filters updated: {self.tradability_filters.to_dict()}")

    def set_entry_threshold(self, threshold: float) -> None:
        """Set the pool entry threshold."""
        self.ENTRY_THRESHOLD = threshold
        logger.info(f"Entry threshold set to {threshold}")

    def set_exit_threshold(self, threshold: float) -> None:
        """Set the pool exit threshold."""
        self.EXIT_THRESHOLD = threshold
        logger.info(f"Exit threshold set to {threshold}")

    def set_score_weights(
        self,
        momentum_weight: Optional[float] = None,
        win_rate_weight: Optional[float] = None
    ) -> None:
        """
        Set the weights for selection score calculation.

        Args:
            momentum_weight: Weight for momentum score (0-1)
            win_rate_weight: Weight for win rate score (0-1)

        Note: Weights should sum to 1.0
        """
        if momentum_weight is not None:
            self.MOMENTUM_WEIGHT = momentum_weight
        if win_rate_weight is not None:
            self.WIN_RATE_WEIGHT = win_rate_weight

        # Normalize if needed
        total = self.MOMENTUM_WEIGHT + self.WIN_RATE_WEIGHT
        if abs(total - 1.0) > 0.001:
            logger.warning(
                f"Score weights don't sum to 1.0 ({total:.3f}), normalizing..."
            )
            self.MOMENTUM_WEIGHT = self.MOMENTUM_WEIGHT / total
            self.WIN_RATE_WEIGHT = self.WIN_RATE_WEIGHT / total

        logger.info(
            f"Score weights: momentum={self.MOMENTUM_WEIGHT:.2f}, "
            f"win_rate={self.WIN_RATE_WEIGHT:.2f}"
        )

    # Status and monitoring
    def get_status(self) -> Dict[str, Any]:
        """Get current filter status and configuration."""
        return {
            'entry_threshold': self.ENTRY_THRESHOLD,
            'exit_threshold': self.EXIT_THRESHOLD,
            'momentum_weight': self.MOMENTUM_WEIGHT,
            'win_rate_weight': self.WIN_RATE_WEIGHT,
            'tradability_filters': self.tradability_filters.to_dict(),
            'system_win_rate': self._system_win_rate,
            'trade_history_symbols': len(self._trade_history),
            'cached_scores': len(self._cached_scores),
            'sector_stats_available': len(self._sector_stats),
        }

    def get_score_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed score breakdown for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with score components and tradability status
        """
        tradability_passed, tradability_reason = self._check_tradability(symbol)
        momentum_score = self.calculate_momentum_score(symbol)
        win_rate_score = self.calculate_win_rate_score(symbol)
        selection_score = self.calculate_selection_score(symbol)

        in_pool = False
        action = "none"
        if selection_score is not None:
            if selection_score >= self.ENTRY_THRESHOLD:
                action = "entry_eligible"
                in_pool = True
            elif selection_score < self.EXIT_THRESHOLD:
                action = "exit_required"
            else:
                action = "hold"  # In hysteresis buffer zone
                in_pool = True  # Keep if already in pool

        return {
            'symbol': symbol,
            'tradability_passed': tradability_passed,
            'tradability_reason': tradability_reason,
            'momentum_score': momentum_score,
            'win_rate_score': win_rate_score,
            'selection_score': selection_score,
            'entry_threshold': self.ENTRY_THRESHOLD,
            'exit_threshold': self.EXIT_THRESHOLD,
            'action': action,
            'market_data': {
                'adv_dollars': self._get_adv_dollars(symbol),
                'price': self._get_current_price(symbol),
                'spread_pct': self._get_spread_pct(symbol),
                'atr_pct': self._get_atr_pct(symbol),
            }
        }

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._cached_scores = {}
        logger.info("Score cache cleared")


# Global singleton instance
_stock_selection_filter: Optional[StockSelectionFilter] = None


def get_stock_selection_filter() -> StockSelectionFilter:
    """Get the global StockSelectionFilter singleton."""
    global _stock_selection_filter
    if _stock_selection_filter is None:
        _stock_selection_filter = StockSelectionFilter()
    return _stock_selection_filter


def set_stock_selection_filter(filter_instance: StockSelectionFilter) -> None:
    """Set the global StockSelectionFilter singleton."""
    global _stock_selection_filter
    _stock_selection_filter = filter_instance
