"""
TradeHistoryRecorder - Real-time trade recording with full decision context.

This module handles:
1. Recording entry trades with decision context (score, gate reasons, regime info)
2. Recording exit trades and calculating round-trip P&L
3. FIX 15: Properly capture exit_trade_id from insert operations
4. Direction-aware P&L calculation (LONG vs SHORT)

Usage:
    from bot.intelligent_trading_decision.trade_history_recorder import TradeHistoryRecorder

    recorder = TradeHistoryRecorder()
    entry_id = recorder.record_entry(
        symbol='AAPL',
        action='BUY',
        quantity=100,
        fill_price=185.50,
        trade_time=bar_time,
        decision_context={...}
    )

    exit_id = recorder.record_exit(
        entry_trade_id=entry_id,
        exit_price=188.20,
        exit_reason='Take profit',
        bar_time=bar_time
    )
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any
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


class TradeHistoryRecorder:
    """
    Real-time trade recording with full decision context.

    This class records trades as they happen (not historical sync),
    capturing the full decision reasoning for post-trade analysis.

    All costs are in $/share units for consistency with CostBenefitGate.
    """

    def __init__(self):
        """Initialize TradeHistoryRecorder."""
        self._supabase = _get_supabase()

        # Track entry trades for quick exit matching
        self._entry_trades: Dict[str, str] = {}  # position_id -> trade_id

    def is_enabled(self) -> bool:
        """Check if recording is enabled (Supabase available)."""
        return self._supabase is not None and self._supabase.is_enabled()

    def record_entry(
        self,
        symbol: str,
        action: str,
        quantity: int,
        fill_price: float,
        trade_time: datetime,
        decision_context: Optional[Dict[str, Any]] = None,
        position_id: Optional[str] = None,
        tiger_order_id: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Optional[str]:
        """
        Record an entry trade with full decision context.

        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL' (SELL for short entry)
            quantity: Number of shares
            fill_price: Execution price ($/share)
            trade_time: Trade timestamp (use bar_time, not datetime.now())
            decision_context: Dict containing decision reasoning:
                - decision_score: Final score from L3 (0-100)
                - score_components: {stability: 25, volume: 45, price_action: 30}
                - gate1_reason: 'vwap_cross_up', 'or_high_breakout', etc.
                - gate2_edge: Expected edge $/share
                - gate2_cost: Transaction cost $/share
                - edge_multiple: edge/cost ratio used
                - regime: {vix, max_position_pct, threshold_boost, sector_boost}
                - fees_estimated: Commission/fee $/share
                - slippage_estimated: Estimated slippage $/share
                - spread_at_entry: Bid-ask spread at entry
            position_id: Position ID for linking entry/exit
            tiger_order_id: Tiger API order ID
            sector: Stock sector

        Returns:
            The trade_id (UUID) if successful, None otherwise
        """
        if not self.is_enabled():
            logger.warning("Trade history recording disabled (Supabase not available)")
            return None

        try:
            # Determine direction from action
            direction = 'LONG' if action == 'BUY' else 'SHORT'

            # Extract decision context
            ctx = decision_context or {}

            # Prepare trade record
            trade_data = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'fill_price': float(fill_price),
                'total_value': float(fill_price) * int(quantity),
                'trade_time': trade_time.isoformat() if isinstance(trade_time, datetime) else trade_time,
                'tiger_order_id': tiger_order_id,
                'source': 'system',
                'sector': sector or self._get_sector(symbol),
                'is_position_closed': False,

                # Decision reasoning
                'decision_score': ctx.get('decision_score'),
                'adjusted_threshold': ctx.get('adjusted_threshold'),
                'score_components': ctx.get('score_components'),

                # Gate 1: Trigger Info
                'gate1_reason': ctx.get('gate1_reason'),

                # Gate 2: Cost/Benefit (all in $/share)
                'gate2_edge': ctx.get('gate2_edge'),
                'gate2_cost': ctx.get('gate2_cost'),
                'edge_multiple': ctx.get('edge_multiple'),

                # Regime (L2) at trade time
                'regime': ctx.get('regime'),

                # Cost estimates ($/share)
                'fees_estimated': ctx.get('fees_estimated'),
                'slippage_estimated': ctx.get('slippage_estimated'),
                'spread_at_entry': ctx.get('spread_at_entry'),

                # Entry price (same as fill_price for entry)
                'entry_price': float(fill_price),
            }

            # Insert into Supabase
            result = self._supabase.client.table('trade_history').insert(trade_data).execute()

            if result.data and len(result.data) > 0:
                trade_id = result.data[0].get('id')
                logger.info(
                    f"ENTRY RECORDED: {symbol} {action} {quantity} @ ${fill_price:.2f} | "
                    f"Score: {ctx.get('decision_score', 'N/A')} | ID: {trade_id}"
                )

                # Cache for exit matching
                if position_id:
                    self._entry_trades[position_id] = trade_id

                return trade_id
            else:
                logger.error(f"Insert returned no data for {symbol} entry")
                return None

        except Exception as e:
            logger.error(f"Failed to record entry trade: {e}")
            return None

    def record_exit(
        self,
        entry_trade_id: str,
        exit_price: float,
        exit_reason: str,
        bar_time: datetime,
        exit_quantity: Optional[int] = None,
        tiger_order_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Record an exit trade and calculate round-trip P&L.

        FIX 15: Properly capture exit_trade_id from insert operation.

        Args:
            entry_trade_id: The ID of the entry trade being closed
            exit_price: Exit execution price ($/share)
            exit_reason: Reason for exit ('Stop loss hit', 'Take profit', 'Signal reversal', etc.)
            bar_time: Exit timestamp (use bar_time, not datetime.now())
            exit_quantity: Number of shares exited (defaults to entry quantity)
            tiger_order_id: Tiger API order ID for the exit

        Returns:
            The exit_trade_id (UUID) if successful, None otherwise
        """
        if not self.is_enabled():
            logger.warning("Trade history recording disabled")
            return None

        try:
            # Get the entry trade
            entry_result = self._supabase.client.table('trade_history')\
                .select('*')\
                .eq('id', entry_trade_id)\
                .single()\
                .execute()

            if not entry_result.data:
                logger.error(f"Entry trade {entry_trade_id} not found")
                return None

            entry_trade = entry_result.data
            symbol = entry_trade['symbol']
            entry_action = entry_trade['action']
            entry_price = float(entry_trade['fill_price'])
            entry_time_str = entry_trade['trade_time']
            entry_quantity = int(entry_trade['quantity'])

            # Determine direction from entry action
            direction = 'LONG' if entry_action == 'BUY' else 'SHORT'

            # Use exit_quantity or default to entry quantity
            quantity = exit_quantity or entry_quantity

            # Direction-aware P&L calculation
            if direction == 'LONG':
                pnl_amount = (exit_price - entry_price) * quantity
                exit_action = 'SELL'
            else:  # SHORT
                pnl_amount = (entry_price - exit_price) * quantity
                exit_action = 'BUY'

            # Calculate P&L percentage
            pnl_percent = (pnl_amount / (entry_price * quantity)) * 100 if entry_price > 0 else 0

            # Calculate hold duration using bar_time
            if isinstance(entry_time_str, str):
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
            else:
                entry_time = entry_time_str

            if isinstance(bar_time, str):
                exit_time = datetime.fromisoformat(bar_time.replace('Z', '+00:00'))
            else:
                exit_time = bar_time

            hold_duration_minutes = int((exit_time - entry_time).total_seconds() / 60)

            # FIX 15: Insert exit trade and CAPTURE the returned ID
            exit_trade_data = {
                'symbol': symbol,
                'action': exit_action,
                'quantity': quantity,
                'fill_price': float(exit_price),
                'total_value': float(exit_price) * quantity,
                'trade_time': bar_time.isoformat() if isinstance(bar_time, datetime) else bar_time,
                'tiger_order_id': tiger_order_id,
                'source': 'system',
                'sector': entry_trade.get('sector'),
                'paired_trade_id': entry_trade_id,
                'is_position_closed': True,
                'entry_price': entry_price,
                'exit_price': float(exit_price),
                'pnl_amount': pnl_amount,
                'pnl_percent': pnl_percent,
                'hold_duration_minutes': hold_duration_minutes,
                'exit_reason': exit_reason,
                'was_profitable': pnl_amount > 0,
            }

            exit_result = self._supabase.client.table('trade_history').insert(exit_trade_data).execute()

            # FIX 15: Properly extract exit_trade_id
            exit_trade_id = None
            if exit_result.data and len(exit_result.data) > 0:
                exit_trade_id = exit_result.data[0].get('id')
            else:
                logger.error(f"Failed to insert exit trade for {symbol}")
                # Continue to update entry even if exit insert failed

            # Update the entry record with exit info
            update_data = {
                'is_position_closed': True,
                'paired_trade_id': exit_trade_id,  # FIX 15: Now properly defined
                'exit_price': float(exit_price),
                'pnl_amount': pnl_amount,
                'pnl_percent': pnl_percent,
                'hold_duration_minutes': hold_duration_minutes,
                'exit_reason': exit_reason,
                'was_profitable': pnl_amount > 0,
                'decision_quality': 'correct' if pnl_amount > 0 else 'incorrect',
            }

            self._supabase.client.table('trade_history')\
                .update(update_data)\
                .eq('id', entry_trade_id)\
                .execute()

            logger.info(
                f"EXIT RECORDED: {symbol} {exit_action} {quantity} @ ${exit_price:.2f} | "
                f"P&L: ${pnl_amount:.2f} ({pnl_percent:.2f}%) | "
                f"Hold: {hold_duration_minutes}min | Reason: {exit_reason}"
            )

            return exit_trade_id

        except Exception as e:
            logger.error(f"Failed to record exit trade: {e}")
            return None

    def record_exit_by_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        bar_time: datetime,
        tiger_order_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Record an exit trade using position_id to find entry.

        This is a convenience method that looks up the entry trade
        by position_id from the internal cache.

        Args:
            position_id: Position ID used when recording entry
            exit_price: Exit execution price
            exit_reason: Reason for exit
            bar_time: Exit timestamp
            tiger_order_id: Tiger API order ID

        Returns:
            The exit_trade_id if successful, None otherwise
        """
        entry_trade_id = self._entry_trades.get(position_id)
        if not entry_trade_id:
            # Try to find entry by position_id in database
            # This handles cases where recorder was restarted
            logger.warning(f"Entry trade for position {position_id} not in cache, searching database")
            entry_trade_id = self._find_entry_by_position(position_id)

        if not entry_trade_id:
            logger.error(f"No entry trade found for position {position_id}")
            return None

        result = self.record_exit(
            entry_trade_id=entry_trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            bar_time=bar_time,
            tiger_order_id=tiger_order_id
        )

        # Clean up cache
        if position_id in self._entry_trades:
            del self._entry_trades[position_id]

        return result

    def _find_entry_by_position(self, position_id: str) -> Optional[str]:
        """Find entry trade ID by position_id in database."""
        # This would require storing position_id in trade_history
        # For now, return None and let caller handle
        return None

    def get_entry_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent unclosed entry trade for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Entry trade record or None
        """
        if not self.is_enabled():
            return None

        try:
            result = self._supabase.client.table('trade_history')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('is_position_closed', False)\
                .order('trade_time', desc=True)\
                .limit(1)\
                .execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error getting entry for {symbol}: {e}")
            return None

    def get_recent_trades(self, symbol: Optional[str] = None, days: int = 7, limit: int = 100) -> list:
        """
        Get recent trades from history.

        Args:
            symbol: Filter by symbol (optional)
            days: Number of days to look back
            limit: Maximum number of trades to return

        Returns:
            List of trade records
        """
        if not self.is_enabled():
            return []

        try:
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = self._supabase.client.table('trade_history')\
                .select('*')\
                .gte('trade_time', since)\
                .order('trade_time', desc=True)\
                .limit(limit)

            if symbol:
                query = query.eq('symbol', symbol.upper())

            result = query.execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

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


# Import timedelta for get_recent_trades
from datetime import timedelta


# Global singleton
_trade_history_recorder: Optional[TradeHistoryRecorder] = None


def get_trade_history_recorder() -> TradeHistoryRecorder:
    """Get the global TradeHistoryRecorder singleton."""
    global _trade_history_recorder
    if _trade_history_recorder is None:
        _trade_history_recorder = TradeHistoryRecorder()
    return _trade_history_recorder


def set_trade_history_recorder(recorder: TradeHistoryRecorder) -> None:
    """Set the global TradeHistoryRecorder singleton."""
    global _trade_history_recorder
    _trade_history_recorder = recorder
