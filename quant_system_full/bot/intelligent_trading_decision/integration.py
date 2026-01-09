"""
Integration helper for Intelligent Trading Decision System.

This module provides easy integration with auto_trading_engine.py
without requiring major refactoring.

Usage in auto_trading_engine.py:
    from bot.intelligent_trading_decision.integration import (
        init_decision_system,
        filter_signals_through_decision_chain,
        check_position_exits,
    )

    # In __init__:
    self.decision_system = init_decision_system(data_provider=self)

    # In analyze_trading_opportunities:
    filtered_signals = filter_signals_through_decision_chain(
        trading_signals,
        decision_system=self.decision_system,
        bar_time=datetime.now()
    )
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .decision_chain import DecisionChain, get_decision_chain, set_decision_chain
from .key_level_provider import get_key_level_provider
from .trigger_gate import get_trigger_gate
from .position_manager import get_position_manager, Position
from .position_sizer import get_position_sizer
from .cost_benefit_gate import get_cost_benefit_gate
from .exit_manager import get_exit_manager

logger = logging.getLogger(__name__)


def init_decision_system(
    data_provider=None,
    config: Optional[Dict[str, Any]] = None,
    stock_pool: Optional[set] = None
) -> DecisionChain:
    """
    Initialize the decision system.

    Args:
        data_provider: Object that provides market data methods
        config: Optional configuration overrides
        stock_pool: Initial stock pool (symbols to trade)

    Returns:
        Initialized DecisionChain instance
    """
    # Default config (conservative for first week)
    default_config = {
        'base_threshold': 65,
        'cooldown_minutes': 20,
        'max_daily_loss': 2000,
        'max_trades_per_symbol_day': 2,
        'max_trades_per_day': 20,
    }

    if config:
        default_config.update(config)

    # Create and configure decision chain
    chain = DecisionChain(config=default_config)

    if data_provider:
        chain.set_data_provider(data_provider)

    if stock_pool:
        chain.set_stock_pool(stock_pool)

    # Set as global singleton
    set_decision_chain(chain)

    logger.info(
        f"Decision system initialized: "
        f"threshold={default_config['base_threshold']}, "
        f"cooldown={default_config['cooldown_minutes']}min, "
        f"max_trades/day={default_config['max_trades_per_day']}"
    )

    return chain


def filter_signals_through_decision_chain(
    trading_signals: Dict[str, List[Dict]],
    decision_system: Optional[DecisionChain] = None,
    bar_time: Optional[datetime] = None,
    current_positions: Optional[Dict[str, int]] = None
) -> Dict[str, List[Dict]]:
    """
    Filter trading signals through the decision chain.

    This is the main integration point. Call this after generating raw signals
    to filter them through the intelligent decision system.

    Args:
        trading_signals: Dict with 'buy', 'sell', 'hold' lists
        decision_system: DecisionChain instance (uses global if not provided)
        bar_time: Current timestamp (uses now if not provided)
        current_positions: Dict of symbol -> share count (optional)

    Returns:
        Filtered trading signals dict
    """
    if decision_system is None:
        decision_system = get_decision_chain()

    if bar_time is None:
        bar_time = datetime.now()

    if current_positions is None:
        current_positions = {}

    filtered_signals = {
        'buy': [],
        'sell': [],
        'hold': trading_signals.get('hold', []),
        'blocked': [],  # Track blocked signals for logging
    }

    # Filter buy signals
    for signal in trading_signals.get('buy', []):
        approved, reason = _evaluate_signal(
            signal, 'BUY', decision_system, bar_time, current_positions
        )
        if approved:
            signal['decision_reason'] = reason
            filtered_signals['buy'].append(signal)
        else:
            signal['blocked_reason'] = reason
            filtered_signals['blocked'].append(signal)
            logger.info(f"BLOCKED BUY {signal.get('symbol')}: {reason}")

    # Filter sell signals
    for signal in trading_signals.get('sell', []):
        approved, reason = _evaluate_signal(
            signal, 'SELL', decision_system, bar_time, current_positions
        )
        if approved:
            signal['decision_reason'] = reason
            filtered_signals['sell'].append(signal)
        else:
            signal['blocked_reason'] = reason
            filtered_signals['blocked'].append(signal)
            logger.info(f"BLOCKED SELL {signal.get('symbol')}: {reason}")

    # Log summary
    original_buy = len(trading_signals.get('buy', []))
    original_sell = len(trading_signals.get('sell', []))
    approved_buy = len(filtered_signals['buy'])
    approved_sell = len(filtered_signals['sell'])
    blocked = len(filtered_signals['blocked'])

    logger.info(
        f"Decision filter: BUY {approved_buy}/{original_buy}, "
        f"SELL {approved_sell}/{original_sell}, "
        f"BLOCKED {blocked}"
    )

    return filtered_signals


def _evaluate_signal(
    signal: Dict[str, Any],
    action: str,
    decision_system: DecisionChain,
    bar_time: datetime,
    current_positions: Dict[str, int]
) -> tuple:
    """
    Evaluate a single signal through the decision chain.

    Returns:
        Tuple of (approved: bool, reason: str)
    """
    symbol = signal.get('symbol')
    if not symbol:
        return False, "No symbol"

    price = signal.get('price', 0)
    if price <= 0:
        return False, "Invalid price"

    # Get signal score (use existing score or default)
    signal_score = signal.get('score', 70)  # Default 70 if not provided

    # Get current position
    current_shares = current_positions.get(symbol, 0)

    # For simplicity in integration, use a simplified check
    # The full decision chain requires prev_price and bar_volume
    # which may not be available in the current signal format

    # Check if in pool
    if symbol not in decision_system.stock_pool:
        # Auto-add to pool if not present (for gradual rollout)
        if len(decision_system.stock_pool) == 0:
            decision_system.add_to_pool(symbol)
        else:
            return False, "Not in pool"

    # Check regime
    if not decision_system.trade_enabled:
        return False, "Trading disabled"

    # Check score threshold
    adjusted_threshold = decision_system.base_threshold + decision_system.threshold_boost
    if signal_score < adjusted_threshold:
        return False, f"Score {signal_score:.1f} < {adjusted_threshold}"

    # Check risk control
    risk_passed, risk_reason = decision_system._check_risk_control(
        symbol, action, signal_score, bar_time
    )
    if not risk_passed:
        return False, risk_reason

    # Record the decision
    decision_system._record_trade_decision(symbol, bar_time)

    return True, f"Approved: score={signal_score:.1f}"


def check_position_exits(
    positions: List[Dict],
    bar_time: Optional[datetime] = None,
    price_getter=None
) -> List[Dict]:
    """
    Check all positions for exit conditions.

    Args:
        positions: List of position dicts with symbol, entry_price, entry_time, direction, quantity
        bar_time: Current timestamp
        price_getter: Function that returns current price for a symbol

    Returns:
        List of positions that should be exited with exit reasons
    """
    if bar_time is None:
        bar_time = datetime.now()

    exit_manager = get_exit_manager()
    exits = []

    for pos_dict in positions:
        symbol = pos_dict.get('symbol')
        if not symbol:
            continue

        # Get current price
        if price_getter:
            current_price = price_getter(symbol)
        else:
            current_price = pos_dict.get('market_price') or pos_dict.get('price')

        if not current_price or current_price <= 0:
            continue

        # Create Position object for exit check
        position = Position(
            position_id=pos_dict.get('position_id', symbol),
            symbol=symbol,
            direction=pos_dict.get('direction', 'LONG'),
            entry_price=pos_dict.get('entry_price', pos_dict.get('avg_cost', 0)),
            quantity=pos_dict.get('quantity', 0),
            entry_time=pos_dict.get('entry_time', bar_time)
        )

        # Check exit conditions
        should_exit, reason = exit_manager.check_exit(position, current_price, bar_time)

        if should_exit:
            exits.append({
                'symbol': symbol,
                'action': 'SELL' if position.direction == 'LONG' else 'BUY',
                'qty': position.quantity,
                'price': current_price,
                'reason': reason,
                'exit_type': 'auto',
                'position_id': position.position_id,
            })
            logger.info(f"EXIT SIGNAL: {symbol} - {reason}")

    return exits


def update_regime_from_market(decision_system: Optional[DecisionChain] = None, vix: Optional[float] = None):
    """
    Update regime based on current market conditions.

    Args:
        decision_system: DecisionChain instance
        vix: Current VIX value (if available)
    """
    if decision_system is None:
        decision_system = get_decision_chain()

    if vix is not None:
        decision_system.update_regime_from_vix(vix)
        logger.info(f"Regime updated for VIX={vix:.1f}")


def reset_daily(decision_system: Optional[DecisionChain] = None):
    """
    Reset all daily counters. Call at market open.

    Args:
        decision_system: DecisionChain instance
    """
    if decision_system is None:
        decision_system = get_decision_chain()

    decision_system.reset_daily()
    logger.info("Decision system daily reset completed")


def get_decision_status(decision_system: Optional[DecisionChain] = None) -> Dict[str, Any]:
    """
    Get current decision system status.

    Args:
        decision_system: DecisionChain instance

    Returns:
        Status dict
    """
    if decision_system is None:
        decision_system = get_decision_chain()

    return decision_system.get_status()


# Convenience functions for enabling/disabling
def enable_decision_filter():
    """Enable the decision filter (default)."""
    chain = get_decision_chain()
    chain.trade_enabled = True
    logger.info("Decision filter ENABLED")


def disable_decision_filter():
    """Disable the decision filter (pass all signals)."""
    chain = get_decision_chain()
    chain.trade_enabled = False
    logger.info("Decision filter DISABLED - all signals will pass")


def set_score_threshold(threshold: int):
    """Set the base score threshold."""
    chain = get_decision_chain()
    chain.base_threshold = threshold
    logger.info(f"Score threshold set to {threshold}")


def add_symbols_to_pool(symbols: List[str]):
    """Add symbols to the trading pool."""
    chain = get_decision_chain()
    for symbol in symbols:
        chain.add_to_pool(symbol)
    logger.info(f"Added {len(symbols)} symbols to pool (total: {len(chain.stock_pool)})")
