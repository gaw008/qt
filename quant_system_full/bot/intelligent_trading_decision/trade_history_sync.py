"""
TradeHistorySync - Sync historical trades from Tiger API to Supabase.

This module handles:
1. Syncing filled orders from Tiger API (up to 90 days)
2. Matching buy/sell pairs to calculate round-trip P&L
3. Persisting to trade_history table in Supabase

Usage:
    from bot.intelligent_trading_decision.trade_history_sync import TradeHistorySync

    sync = TradeHistorySync(trade_client=tiger_trade_client)
    result = sync.sync_from_tiger(days=90)
    print(f"Synced {result['synced']} trades")
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Tiger SDK
try:
    from tigeropen.trade.trade_client import TradeClient
    from tigeropen.common.consts import Market, SecurityType, OrderStatus
    TIGER_SDK_AVAILABLE = True
except ImportError:
    logger.warning("Tiger SDK not available for trade history sync")
    TIGER_SDK_AVAILABLE = False
    TradeClient = None
    OrderStatus = None

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


class TradeHistorySync:
    """
    Sync historical trades from Tiger API to Supabase.

    This class handles the historical sync mode where we pull filled orders
    from Tiger API and persist them to the trade_history table for analysis.
    """

    def __init__(self, trade_client: Optional[Any] = None, account: Optional[str] = None):
        """
        Initialize TradeHistorySync.

        Args:
            trade_client: Tiger TradeClient instance (optional, can be set later)
            account: Tiger account number (optional, will try to extract from client)
        """
        self._trade_client = trade_client
        self._account = account or os.getenv('ACCOUNT')
        self._supabase = _get_supabase()

        # Track synced order IDs to avoid duplicates
        self._synced_order_ids: set = set()

    def set_trade_client(self, trade_client: Any, account: Optional[str] = None) -> None:
        """Set the Tiger trade client."""
        self._trade_client = trade_client
        if account:
            self._account = account

    def sync_from_tiger(self, days: int = 90) -> Dict[str, Any]:
        """
        Sync filled orders from Tiger API to Supabase.

        Args:
            days: Number of days to look back (max 90 for Tiger API)

        Returns:
            Dict with sync results: {synced, skipped, errors, paired}
        """
        if not TIGER_SDK_AVAILABLE:
            logger.error("Tiger SDK not available")
            return {'synced': 0, 'skipped': 0, 'errors': 1, 'paired': 0, 'error': 'Tiger SDK not available'}

        if not self._trade_client:
            logger.error("Trade client not set")
            return {'synced': 0, 'skipped': 0, 'errors': 1, 'paired': 0, 'error': 'Trade client not set'}

        if not self._supabase or not self._supabase.is_enabled():
            logger.error("Supabase not available")
            return {'synced': 0, 'skipped': 0, 'errors': 1, 'paired': 0, 'error': 'Supabase not available'}

        logger.info(f"Starting trade history sync for {days} days")

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Fetch filled orders from Tiger API
        try:
            orders = self._fetch_filled_orders(start_time, end_time)
            logger.info(f"Fetched {len(orders)} filled orders from Tiger API")
        except Exception as e:
            logger.error(f"Failed to fetch orders from Tiger API: {e}")
            return {'synced': 0, 'skipped': 0, 'errors': 1, 'paired': 0, 'error': str(e)}

        # Load existing order IDs from Supabase to avoid duplicates
        self._load_existing_order_ids()

        # Sync orders to Supabase
        synced = 0
        skipped = 0
        errors = 0

        for order in orders:
            try:
                order_id = str(order.get('order_id', order.get('id')))

                if order_id in self._synced_order_ids:
                    skipped += 1
                    continue

                success = self._insert_trade(order)
                if success:
                    synced += 1
                    self._synced_order_ids.add(order_id)
                else:
                    errors += 1

            except Exception as e:
                logger.error(f"Error syncing order: {e}")
                errors += 1

        logger.info(f"Sync complete: synced={synced}, skipped={skipped}, errors={errors}")

        # Calculate round-trip P&L for paired trades
        paired = self._calculate_round_trip_pnl()

        return {
            'synced': synced,
            'skipped': skipped,
            'errors': errors,
            'paired': paired,
            'total_orders': len(orders)
        }

    def _fetch_filled_orders(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Fetch filled orders from Tiger API.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of order dictionaries
        """
        orders = []

        try:
            # Use Tiger API to get orders
            # Note: Tiger API returns orders in different formats based on SDK version
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)

            tiger_orders = self._trade_client.get_orders(
                sec_type=SecurityType.STK,
                market=Market.US,
                start_time=start_ms,
                end_time=end_ms,
                is_brief=False
            )

            if not tiger_orders:
                return orders

            for order in tiger_orders:
                # Only process filled orders
                status = getattr(order, 'status', None)
                if status not in [OrderStatus.FILLED, 'FILLED']:
                    continue

                order_dict = {
                    'order_id': getattr(order, 'id', None),
                    'symbol': getattr(order.contract, 'symbol', None) if hasattr(order, 'contract') else None,
                    'action': getattr(order, 'action', None),
                    'quantity': getattr(order, 'filled', 0) or getattr(order, 'quantity', 0),
                    'fill_price': getattr(order, 'avg_fill_price', 0),
                    'order_time': self._parse_order_time(getattr(order, 'trade_time', None) or getattr(order, 'order_time', None)),
                    'status': str(status),
                    'commission': getattr(order, 'commission', 0) or 0,
                }

                # Filter out invalid orders
                if order_dict['symbol'] and order_dict['fill_price'] and order_dict['fill_price'] > 0:
                    orders.append(order_dict)

        except Exception as e:
            logger.error(f"Error fetching orders from Tiger API: {e}")
            raise

        return orders

    def _parse_order_time(self, order_time: Any) -> Optional[datetime]:
        """Parse order time from Tiger API response."""
        if order_time is None:
            return None

        if isinstance(order_time, datetime):
            return order_time

        if isinstance(order_time, (int, float)):
            # Timestamp in milliseconds
            try:
                return datetime.fromtimestamp(order_time / 1000)
            except (ValueError, OSError):
                return None

        if isinstance(order_time, str):
            # Try parsing as ISO format
            try:
                return datetime.fromisoformat(order_time.replace('Z', '+00:00'))
            except ValueError:
                pass

        return None

    def _load_existing_order_ids(self) -> None:
        """Load existing Tiger order IDs from Supabase to avoid duplicates."""
        try:
            # Query trade_history for existing tiger_order_ids
            result = self._supabase.client.table('trade_history')\
                .select('tiger_order_id')\
                .not_.is_('tiger_order_id', 'null')\
                .execute()

            if result.data:
                self._synced_order_ids = {
                    str(r['tiger_order_id']) for r in result.data if r.get('tiger_order_id')
                }
                logger.info(f"Loaded {len(self._synced_order_ids)} existing order IDs")

        except Exception as e:
            logger.warning(f"Failed to load existing order IDs: {e}")
            self._synced_order_ids = set()

    def _insert_trade(self, order: Dict) -> bool:
        """
        Insert a trade record into Supabase.

        Args:
            order: Order dictionary from Tiger API

        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine sector (can be enhanced with actual sector lookup)
            symbol = order.get('symbol', '')
            sector = self._get_sector(symbol)

            # Prepare trade record
            trade_data = {
                'symbol': symbol,
                'action': order.get('action', 'UNKNOWN'),
                'quantity': order.get('quantity', 0),
                'fill_price': float(order.get('fill_price', 0)),
                'total_value': float(order.get('fill_price', 0)) * int(order.get('quantity', 0)),
                'trade_time': order.get('order_time').isoformat() if order.get('order_time') else datetime.utcnow().isoformat(),
                'tiger_order_id': str(order.get('order_id')),
                'source': 'sync_from_tiger',
                'sector': sector,
                # Historical syncs don't have decision context
                'decision_score': None,
                'score_components': None,
                'gate1_reason': None,
                'gate2_edge': None,
                'gate2_cost': None,
            }

            # Insert into Supabase
            result = self._supabase.client.table('trade_history').insert(trade_data).execute()

            if result.data:
                logger.debug(f"Inserted trade: {symbol} {order.get('action')} {order.get('quantity')} @ ${order.get('fill_price'):.2f}")
                return True
            else:
                logger.warning(f"Insert returned no data for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            return False

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

    def _calculate_round_trip_pnl(self) -> int:
        """
        Match buy/sell pairs and calculate round-trip P&L.

        This finds unpaired entry trades and matches them with corresponding
        exit trades to calculate the round-trip P&L.

        Returns:
            Number of trades paired
        """
        paired = 0

        try:
            # Get unpaired entry trades (BUY actions without paired_trade_id and not closed)
            result = self._supabase.client.table('trade_history')\
                .select('*')\
                .is_('paired_trade_id', 'null')\
                .eq('is_position_closed', False)\
                .order('trade_time', desc=False)\
                .execute()

            if not result.data:
                logger.info("No unpaired trades to match")
                return 0

            # Group by symbol
            trades_by_symbol: Dict[str, List[Dict]] = {}
            for trade in result.data:
                symbol = trade['symbol']
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)

            # Match buy/sell pairs (FIFO)
            for symbol, trades in trades_by_symbol.items():
                buys = [t for t in trades if t['action'] == 'BUY']
                sells = [t for t in trades if t['action'] == 'SELL']

                # Sort by time
                buys.sort(key=lambda x: x['trade_time'])
                sells.sort(key=lambda x: x['trade_time'])

                for buy in buys:
                    # Find first sell after this buy
                    for sell in sells:
                        if sell['trade_time'] > buy['trade_time']:
                            # Match found - calculate P&L
                            entry_price = float(buy['fill_price'])
                            exit_price = float(sell['fill_price'])
                            quantity = min(int(buy['quantity']), int(sell['quantity']))

                            # P&L calculation (LONG direction for BUY entry)
                            pnl_amount = (exit_price - entry_price) * quantity
                            pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                            # Calculate hold duration in minutes
                            entry_time = datetime.fromisoformat(buy['trade_time'].replace('Z', '+00:00'))
                            exit_time = datetime.fromisoformat(sell['trade_time'].replace('Z', '+00:00'))
                            hold_minutes = int((exit_time - entry_time).total_seconds() / 60)

                            # Update entry trade with exit info
                            self._supabase.client.table('trade_history').update({
                                'is_position_closed': True,
                                'paired_trade_id': sell['id'],
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl_amount': pnl_amount,
                                'pnl_percent': pnl_percent,
                                'hold_duration_minutes': hold_minutes,
                                'exit_reason': 'matched_from_sync',
                                'was_profitable': pnl_amount > 0,
                            }).eq('id', buy['id']).execute()

                            # Update exit trade to link back
                            self._supabase.client.table('trade_history').update({
                                'paired_trade_id': buy['id'],
                            }).eq('id', sell['id']).execute()

                            paired += 1
                            sells.remove(sell)
                            logger.debug(f"Paired {symbol}: ${entry_price:.2f} -> ${exit_price:.2f}, P&L: ${pnl_amount:.2f}")
                            break

            logger.info(f"Paired {paired} round-trip trades")

        except Exception as e:
            logger.error(f"Error calculating round-trip P&L: {e}")

        return paired

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status and statistics."""
        if not self._supabase or not self._supabase.is_enabled():
            return {'error': 'Supabase not available'}

        try:
            # Get counts
            total = self._supabase.client.table('trade_history')\
                .select('id', count='exact').execute()

            synced = self._supabase.client.table('trade_history')\
                .select('id', count='exact')\
                .eq('source', 'sync_from_tiger').execute()

            paired = self._supabase.client.table('trade_history')\
                .select('id', count='exact')\
                .eq('is_position_closed', True).execute()

            return {
                'total_trades': total.count if hasattr(total, 'count') else len(total.data or []),
                'synced_from_tiger': synced.count if hasattr(synced, 'count') else len(synced.data or []),
                'paired_trades': paired.count if hasattr(paired, 'count') else len(paired.data or []),
                'tiger_sdk_available': TIGER_SDK_AVAILABLE,
                'supabase_enabled': True,
            }

        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {'error': str(e)}


# Global singleton
_trade_history_sync: Optional[TradeHistorySync] = None


def get_trade_history_sync() -> TradeHistorySync:
    """Get the global TradeHistorySync singleton."""
    global _trade_history_sync
    if _trade_history_sync is None:
        _trade_history_sync = TradeHistorySync()
    return _trade_history_sync


def set_trade_history_sync(sync: TradeHistorySync) -> None:
    """Set the global TradeHistorySync singleton."""
    global _trade_history_sync
    _trade_history_sync = sync
