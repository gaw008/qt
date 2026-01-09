"""
TradeHistoryAnalyzer - Win rate calculations and performance analysis.

This module provides:
1. Per-symbol win rate calculation with cold start handling
2. Sector-level performance statistics
3. Confidence-weighted win rate scoring for stock selection

Cold Start Handling:
- >= 5 trades: Use actual win rate with high confidence
- 1-4 trades: Blend stock win rate with system average
- 0 trades: Fall back to sector average or system average

Usage:
    from bot.intelligent_trading_decision.trade_history_analyzer import TradeHistoryAnalyzer

    analyzer = TradeHistoryAnalyzer()
    score = analyzer.calculate_win_rate_score('AAPL')
    stats = analyzer.get_symbol_performance('AAPL')
    sector_stats = analyzer.get_sector_stats()
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Supabase client
_supabase_client = None


def _get_supabase():
    """Get Supabase client singleton."""
    global _supabase_client
    if _supabase_client is None:
        try:
            backend_path = str(Path(__file__).parent.parent.parent / "dashboard" / "backend")
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from supabase_client import supabase_client
            _supabase_client = supabase_client
        except ImportError:
            logger.warning("Supabase client not available")
            _supabase_client = False
    return _supabase_client if _supabase_client else None


class TradeHistoryAnalyzer:
    """
    Analyzes trade history to calculate win rates and performance metrics.

    This class implements the historical win rate factor from Layer 1
    of the Intelligent Trading Decision System, with proper cold start
    handling for symbols with limited trade history.
    """

    def __init__(self, lookback_days: int = 90):
        """
        Initialize TradeHistoryAnalyzer.

        Args:
            lookback_days: Number of days to consider for performance analysis
        """
        self._supabase = _get_supabase()
        self.lookback_days = lookback_days

        # Cache for performance data (refreshed periodically)
        self._symbol_cache: Dict[str, Dict] = {}
        self._sector_cache: Dict[str, Dict] = {}
        self._system_stats: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_minutes = 30  # Refresh cache every 30 minutes

    def is_enabled(self) -> bool:
        """Check if analyzer is enabled (Supabase available)."""
        return self._supabase is not None and self._supabase.is_enabled()

    def calculate_win_rate_score(
        self,
        symbol: str,
        trade_history: Optional[List[Dict]] = None,
        sector_stats: Optional[Dict[str, Dict]] = None,
        system_stats: Optional[Dict] = None
    ) -> float:
        """
        Calculate win rate score for a symbol with cold start handling.

        This implements the confidence-weighted win rate calculation from the plan:
        - >= 5 trades: High confidence in stock-specific win rate
        - 1-4 trades: Blend with system average
        - 0 trades: Fall back to sector or system average

        Args:
            symbol: Stock symbol
            trade_history: Optional pre-loaded trade history
            sector_stats: Optional pre-loaded sector statistics
            system_stats: Optional pre-loaded system-wide statistics

        Returns:
            Win rate score (0-100)
        """
        # Load trade history if not provided
        if trade_history is None:
            trade_history = self._get_symbol_trades(symbol)

        # Load stats if not provided
        if sector_stats is None:
            sector_stats = self.get_sector_stats()
        if system_stats is None:
            system_stats = self.get_system_stats()

        trade_count = len(trade_history)
        system_win_rate = system_stats.get('win_rate', 0.5)  # Default 50%

        if trade_count >= 5:
            # High confidence - use actual win rate
            wins = sum(1 for t in trade_history if t.get('was_profitable', False))
            stock_win_rate = wins / trade_count
            # Confidence increases with trade count, max at 20 trades
            confidence = min(1.0, trade_count / 20)

        elif trade_count >= 1:
            # Medium confidence - blend with system average
            wins = sum(1 for t in trade_history if t.get('was_profitable', False))
            stock_win_rate = wins / trade_count
            # Blend: 30% base confidence + 15% per trade
            confidence = 0.30 + 0.15 * trade_count

        else:
            # Cold start - use sector fallback
            sector = self._get_sector(symbol)
            sector_data = sector_stats.get(sector, {})
            stock_win_rate = sector_data.get('win_rate', system_win_rate)
            confidence = 0.20  # Low confidence for sector fallback

        # Calculate final score with confidence weighting
        # Higher confidence = more weight on stock-specific rate
        # Lower confidence = more weight on system average
        final_win_rate = stock_win_rate * confidence + system_win_rate * (1 - confidence)

        # Convert to 0-100 score
        score = final_win_rate * 100

        logger.debug(
            f"{symbol}: trades={trade_count}, win_rate={stock_win_rate:.2f}, "
            f"confidence={confidence:.2f}, score={score:.1f}"
        )

        return score

    def get_symbol_performance(self, symbol: str, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics for a symbol.

        Args:
            symbol: Stock symbol
            days: Number of days to analyze (defaults to lookback_days)

        Returns:
            Dict with performance metrics
        """
        # Check cache
        if symbol in self._symbol_cache and self._is_cache_valid():
            return self._symbol_cache[symbol]

        if not self.is_enabled():
            return self._get_empty_performance()

        lookback = days or self.lookback_days
        trades = self._get_symbol_trades(symbol, lookback)

        if not trades:
            return self._get_empty_performance()

        # Calculate metrics
        wins = [t for t in trades if t.get('was_profitable', False)]
        losses = [t for t in trades if not t.get('was_profitable', False) and t.get('pnl_amount') is not None]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(float(t.get('pnl_amount', 0) or 0) for t in trades)
        avg_win = sum(float(t.get('pnl_amount', 0) or 0) for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(float(t.get('pnl_amount', 0) or 0) for t in losses) / len(losses)) if losses else 0

        # Profit factor
        gross_profit = sum(float(t.get('pnl_amount', 0) or 0) for t in wins)
        gross_loss = abs(sum(float(t.get('pnl_amount', 0) or 0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Hold time metrics
        hold_times = [int(t.get('hold_duration_minutes', 0) or 0) for t in trades if t.get('hold_duration_minutes')]
        avg_hold_minutes = sum(hold_times) / len(hold_times) if hold_times else 0

        # Win rate score
        win_rate_score = self.calculate_win_rate_score(symbol, trades)

        performance = {
            'symbol': symbol,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'win_rate_score': win_rate_score,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor if profit_factor != float('inf') else None,
            'avg_hold_minutes': avg_hold_minutes,
            'lookback_days': lookback,
            'last_trade': trades[0].get('trade_time') if trades else None,
        }

        # Cache result
        self._symbol_cache[symbol] = performance

        return performance

    def get_sector_stats(self) -> Dict[str, Dict]:
        """
        Get performance statistics by sector.

        Returns:
            Dict mapping sector name to performance metrics
        """
        # Check cache
        if self._sector_cache and self._is_cache_valid():
            return self._sector_cache

        if not self.is_enabled():
            return {}

        try:
            # Get closed trades from last N days
            since = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()

            result = self._supabase.client.table('trade_history')\
                .select('sector,was_profitable,pnl_amount')\
                .eq('is_position_closed', True)\
                .gte('trade_time', since)\
                .execute()

            if not result.data:
                return {}

            # Group by sector
            sector_data: Dict[str, List[Dict]] = {}
            for trade in result.data:
                sector = trade.get('sector', 'Other') or 'Other'
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(trade)

            # Calculate stats per sector
            sector_stats = {}
            for sector, trades in sector_data.items():
                total = len(trades)
                wins = sum(1 for t in trades if t.get('was_profitable', False))
                total_pnl = sum(float(t.get('pnl_amount', 0) or 0) for t in trades)

                sector_stats[sector] = {
                    'trade_count': total,
                    'win_count': wins,
                    'win_rate': wins / total if total > 0 else 0.5,
                    'total_pnl': total_pnl,
                    'avg_pnl': total_pnl / total if total > 0 else 0,
                }

            self._sector_cache = sector_stats
            self._cache_time = datetime.utcnow()

            return sector_stats

        except Exception as e:
            logger.error(f"Error getting sector stats: {e}")
            return {}

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide performance statistics.

        Returns:
            Dict with system-wide metrics
        """
        # Check cache
        if self._system_stats and self._is_cache_valid():
            return self._system_stats

        if not self.is_enabled():
            return {'win_rate': 0.5, 'trade_count': 0, 'total_pnl': 0}

        try:
            # Get all closed trades from lookback period
            since = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()

            result = self._supabase.client.table('trade_history')\
                .select('was_profitable,pnl_amount')\
                .eq('is_position_closed', True)\
                .gte('trade_time', since)\
                .execute()

            if not result.data:
                return {'win_rate': 0.5, 'trade_count': 0, 'total_pnl': 0}

            trades = result.data
            total = len(trades)
            wins = sum(1 for t in trades if t.get('was_profitable', False))
            total_pnl = sum(float(t.get('pnl_amount', 0) or 0) for t in trades)

            self._system_stats = {
                'win_rate': wins / total if total > 0 else 0.5,
                'trade_count': total,
                'win_count': wins,
                'loss_count': total - wins,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / total if total > 0 else 0,
                'lookback_days': self.lookback_days,
            }

            self._cache_time = datetime.utcnow()

            return self._system_stats

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'win_rate': 0.5, 'trade_count': 0, 'total_pnl': 0}

    def get_top_performers(self, limit: int = 10, min_trades: int = 3) -> List[Dict]:
        """
        Get top performing symbols by win rate.

        Args:
            limit: Maximum number of symbols to return
            min_trades: Minimum trades required to be considered

        Returns:
            List of symbol performance dicts, sorted by win rate score
        """
        if not self.is_enabled():
            return []

        try:
            # Get symbols with enough trades
            since = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()

            result = self._supabase.client.table('trade_history')\
                .select('symbol')\
                .eq('is_position_closed', True)\
                .gte('trade_time', since)\
                .execute()

            if not result.data:
                return []

            # Count trades per symbol
            symbol_counts: Dict[str, int] = {}
            for trade in result.data:
                symbol = trade.get('symbol')
                if symbol:
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            # Filter by minimum trades
            qualified_symbols = [s for s, c in symbol_counts.items() if c >= min_trades]

            # Get performance for each
            performances = []
            for symbol in qualified_symbols:
                perf = self.get_symbol_performance(symbol)
                if perf.get('total_trades', 0) >= min_trades:
                    performances.append(perf)

            # Sort by win rate score (descending)
            performances.sort(key=lambda x: x.get('win_rate_score', 0), reverse=True)

            return performances[:limit]

        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            return []

    def get_worst_performers(self, limit: int = 10, min_trades: int = 3) -> List[Dict]:
        """
        Get worst performing symbols by win rate (for risk management).

        Args:
            limit: Maximum number of symbols to return
            min_trades: Minimum trades required to be considered

        Returns:
            List of symbol performance dicts, sorted by win rate score (ascending)
        """
        # Reuse top_performers logic, just reverse sort
        performers = self.get_top_performers(limit=100, min_trades=min_trades)
        performers.sort(key=lambda x: x.get('win_rate_score', 100))
        return performers[:limit]

    def refresh_cache(self) -> None:
        """Force refresh of all cached data."""
        self._symbol_cache.clear()
        self._sector_cache.clear()
        self._system_stats = None
        self._cache_time = None
        logger.info("Trade history analyzer cache cleared")

    def _get_symbol_trades(self, symbol: str, days: Optional[int] = None) -> List[Dict]:
        """Get closed trades for a symbol."""
        if not self.is_enabled():
            return []

        try:
            lookback = days or self.lookback_days
            since = (datetime.utcnow() - timedelta(days=lookback)).isoformat()

            result = self._supabase.client.table('trade_history')\
                .select('*')\
                .eq('symbol', symbol.upper())\
                .eq('is_position_closed', True)\
                .gte('trade_time', since)\
                .order('trade_time', desc=True)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {e}")
            return []

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        age = (datetime.utcnow() - self._cache_time).total_seconds() / 60
        return age < self._cache_ttl_minutes

    def _get_empty_performance(self) -> Dict[str, Any]:
        """Return empty performance structure."""
        return {
            'symbol': None,
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0.5,
            'win_rate_score': 50.0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': None,
            'avg_hold_minutes': 0,
            'lookback_days': self.lookback_days,
            'last_trade': None,
        }

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified implementation)."""
        # Technology stocks
        tech_symbols = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NOW'}
        # Financial stocks
        fin_symbols = {'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK'}
        # Healthcare stocks
        health_symbols = {'JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 'TMO', 'DHR', 'AMGN', 'GILD', 'BMY'}
        # Consumer stocks
        consumer_symbols = {'AMZN', 'TSLA', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'NFLX', 'COST'}
        # Energy stocks
        energy_symbols = {'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'HAL'}

        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in fin_symbols:
            return 'Financial'
        elif symbol in health_symbols:
            return 'Healthcare'
        elif symbol in consumer_symbols:
            return 'Consumer'
        elif symbol in energy_symbols:
            return 'Energy'
        else:
            return 'Other'


# Global singleton
_trade_history_analyzer: Optional[TradeHistoryAnalyzer] = None


def get_trade_history_analyzer() -> TradeHistoryAnalyzer:
    """Get the global TradeHistoryAnalyzer singleton."""
    global _trade_history_analyzer
    if _trade_history_analyzer is None:
        _trade_history_analyzer = TradeHistoryAnalyzer()
    return _trade_history_analyzer


def set_trade_history_analyzer(analyzer: TradeHistoryAnalyzer) -> None:
    """Set the global TradeHistoryAnalyzer singleton."""
    global _trade_history_analyzer
    _trade_history_analyzer = analyzer
