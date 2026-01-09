"""
DecisionChain - Main orchestrator for the Intelligent Trading Decision System.

This module implements the final execution chain with two critical gates:
1. TriggerGate - Convert continuous signals to discrete events
2. CostBenefitGate - Only trade when Edge > Cost

Execution Flow:
1. Pool check (Layer 1) - Is symbol in tradeable pool?
2. Regime check (Layer 2) - Is trading enabled? What's max position?
3. TriggerGate (Gate 1) - Did a trigger event occur?
4. Signal scoring (Layer 3) - Score the signal direction
5. CostBenefitGate (Gate 2) - Does edge justify cost?
6. Risk control (Layer 4) - Hard rules check
7. Position sizing - Calculate trade size

Expected Result: Reduce signals from 5-9/min to 0-1 per price move.
"""

import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

from .key_level_provider import get_key_level_provider
from .trigger_gate import get_trigger_gate
from .position_manager import get_position_manager
from .position_sizer import get_position_sizer
from .cost_benefit_gate import get_cost_benefit_gate
from .exit_manager import get_exit_manager

logger = logging.getLogger(__name__)


class DecisionChain:
    """
    Main orchestrator for trade decisions.

    Implements the 4-layer architecture with 2 critical gates.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DecisionChain.

        Args:
            config: Optional configuration dict with:
                - base_threshold: Base score threshold (default 65)
                - cooldown_minutes: Trade cooldown per symbol (default 20)
                - max_daily_loss: Max daily loss limit (default 2000)
                - max_trades_per_symbol_day: Max trades per symbol per day (default 2)
                - max_trades_per_day: Max total trades per day (default 20)
        """
        self.config = config or {}

        # Layer 3: Score threshold
        self.base_threshold = self.config.get('base_threshold', 65)

        # Layer 4: Risk control parameters
        self.cooldown_minutes = self.config.get('cooldown_minutes', 20)
        self.max_daily_loss = self.config.get('max_daily_loss', 2000)
        self.max_trades_per_symbol_day = self.config.get('max_trades_per_symbol_day', 2)
        self.max_trades_per_day = self.config.get('max_trades_per_day', 20)

        # Trade tracking
        self.last_trade_time: Dict[str, datetime] = {}  # symbol -> datetime
        self.trades_today: Dict[str, int] = {}  # symbol -> count
        self.total_trades_today = 0

        # Stock pool (Layer 1) - symbols that pass selection criteria
        self.stock_pool: set = set()

        # Regime state (Layer 2)
        self.trade_enabled = True
        self.max_position_pct = 1.0
        self.threshold_boost = 0
        self.sector_boost: Dict[str, float] = {}  # sector -> boost multiplier

        # Data provider for market data
        self._data_provider = None

    def execute(
        self,
        symbol: str,
        signal: str,
        price: float,
        prev_price: float,
        bar_volume: float,
        bar_time: datetime,
        signal_score: float,
        current_shares: int = 0
    ) -> Tuple[bool, int, str]:
        """
        Execute the full decision chain.

        Args:
            symbol: Stock symbol
            signal: 'BUY' or 'SELL'
            price: Current price
            prev_price: Previous bar price
            bar_volume: Current bar volume
            bar_time: Current bar timestamp
            signal_score: Pre-calculated signal score (0-100)
            current_shares: Current position (positive = long, negative = short)

        Returns:
            Tuple of (should_trade: bool, target_shares: int, reason: str)
        """
        # Layer 1: Pool check
        if symbol not in self.stock_pool:
            return False, 0, f"Not in pool ({len(self.stock_pool)} symbols)"

        # Layer 2: Regime check
        if not self.trade_enabled:
            return False, 0, "Trading disabled by regime"

        # Gate 1: Trigger Gate
        trigger_gate = get_trigger_gate()
        triggered, trigger_reason = trigger_gate.check(
            symbol, signal, price, prev_price, bar_volume, bar_time
        )
        if not triggered:
            return False, 0, "No trigger event"

        # Layer 3: Score threshold check
        adjusted_threshold = self.base_threshold + self.threshold_boost
        if signal_score < adjusted_threshold:
            return False, 0, f"Score {signal_score:.1f} < threshold {adjusted_threshold}"

        # Gate 2: Cost/Benefit check
        cost_benefit_gate = get_cost_benefit_gate()
        passed, cost_reason = cost_benefit_gate.check(
            symbol, price, signal_score, shares=100  # Estimate
        )
        if not passed:
            return False, 0, cost_reason

        # Layer 4: Risk control
        risk_passed, risk_reason = self._check_risk_control(
            symbol, signal, signal_score, bar_time
        )
        if not risk_passed:
            return False, 0, risk_reason

        # Position sizing
        position_sizer = get_position_sizer()
        regime = {
            'max_position_pct': self.max_position_pct,
            'sector_boost': self.sector_boost.get(self._get_sector(symbol), 1.0),
        }

        delta_shares, size_reason, intent = position_sizer.calculate_position_for_signal(
            signal, symbol, current_shares, signal_score, regime, price
        )

        if delta_shares == 0:
            return False, 0, size_reason

        # Record the trade decision
        self._record_trade_decision(symbol, bar_time)

        logger.info(
            f"DECISION APPROVED: {symbol} {signal} | "
            f"Trigger: {trigger_reason} | Score: {signal_score:.1f} | "
            f"Intent: {intent} | Delta: {delta_shares}"
        )

        return True, delta_shares, f"Approved: {trigger_reason}, score={signal_score:.1f}"

    def _check_risk_control(
        self,
        symbol: str,
        signal: str,
        signal_score: float,
        bar_time: datetime
    ) -> Tuple[bool, str]:
        """
        Layer 4: Risk control hard rules.

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        # 1. Global daily trade limit
        if self.total_trades_today >= self.max_trades_per_day:
            return False, f"Daily cap: {self.total_trades_today}/{self.max_trades_per_day}"

        # 2. Daily loss limit (requires P&L data)
        daily_pnl = self._get_daily_pnl()
        if daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit: ${daily_pnl:.2f}"

        # 3. Trade cooldown
        last_trade = self.last_trade_time.get(symbol)
        if last_trade:
            minutes_since = (bar_time - last_trade).total_seconds() / 60
            if minutes_since < self.cooldown_minutes:
                remaining = self.cooldown_minutes - minutes_since
                return False, f"Cooldown: {remaining:.1f} min left"

        # 4. Max trades per symbol per day
        today_count = self.trades_today.get(symbol, 0)
        if today_count >= self.max_trades_per_symbol_day:
            return False, f"Symbol cap: {today_count}/{self.max_trades_per_symbol_day}"

        # 5. Earnings blackout
        if self._is_earnings_window(symbol):
            return False, "Earnings blackout"

        return True, "OK"

    def _record_trade_decision(self, symbol: str, bar_time: datetime) -> None:
        """Record that a trade was approved."""
        self.last_trade_time[symbol] = bar_time
        self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
        self.total_trades_today += 1

    def reset_daily(self) -> None:
        """Reset daily counters at market open."""
        self.last_trade_time = {}
        self.trades_today = {}
        self.total_trades_today = 0

        # Reset component states
        get_trigger_gate().reset_daily()
        get_position_manager().reset_daily()

        logger.info("DecisionChain: Daily reset completed")

    # Pool management (Layer 1)
    def set_stock_pool(self, symbols: set) -> None:
        """Set the tradeable stock pool."""
        self.stock_pool = symbols
        logger.info(f"Stock pool updated: {len(symbols)} symbols")

    def add_to_pool(self, symbol: str) -> None:
        """Add a symbol to the pool."""
        self.stock_pool.add(symbol)

    def remove_from_pool(self, symbol: str) -> None:
        """Remove a symbol from the pool."""
        self.stock_pool.discard(symbol)

    # Regime management (Layer 2)
    def update_regime(
        self,
        trade_enabled: bool = True,
        max_position_pct: float = 1.0,
        threshold_boost: int = 0,
        sector_boost: Optional[Dict[str, float]] = None
    ) -> None:
        """Update regime parameters."""
        self.trade_enabled = trade_enabled
        self.max_position_pct = max_position_pct
        self.threshold_boost = threshold_boost
        if sector_boost:
            self.sector_boost = sector_boost

        logger.info(
            f"Regime updated: enabled={trade_enabled}, "
            f"max_pos={max_position_pct:.1%}, threshold_boost={threshold_boost}"
        )

    def update_regime_from_vix(self, vix: float) -> None:
        """
        Update regime based on VIX level.

        VIX-based controls:
        - Position sizing: linear decay from 100% at VIX=15 to 0% at VIX=35
        - Threshold boost: +5 for 20-25, +10 for 25-30, +15 for >30
        """
        # Position sizing (continuous)
        self.max_position_pct = max(0, min(1, (35 - vix) / 20))

        # Threshold boost
        if vix < 20:
            self.threshold_boost = 0
        elif vix < 25:
            self.threshold_boost = 5
        elif vix < 30:
            self.threshold_boost = 10
        else:
            self.threshold_boost = 15

        # Trade enabled
        self.trade_enabled = vix < 40

        logger.info(
            f"Regime updated for VIX={vix:.1f}: "
            f"max_pos={self.max_position_pct:.1%}, "
            f"threshold_boost={self.threshold_boost}, "
            f"enabled={self.trade_enabled}"
        )

    # Data provider
    def set_data_provider(self, data_provider) -> None:
        """Set data provider for all components."""
        self._data_provider = data_provider

        # Set on all components
        get_key_level_provider().set_data_provider(data_provider)
        get_trigger_gate().set_data_provider(data_provider)
        get_cost_benefit_gate().set_data_provider(data_provider)
        get_exit_manager().set_data_provider(data_provider)

    # Helper methods
    def _get_daily_pnl(self) -> float:
        """Get daily P&L from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_daily_pnl'):
                return self._data_provider.get_daily_pnl()
            return 0.0
        except Exception:
            return 0.0

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_sector'):
                return self._data_provider.get_sector(symbol)
            return 'Unknown'
        except Exception:
            return 'Unknown'

    def _is_earnings_window(self, symbol: str, days_before: int = 1, days_after: int = 1) -> bool:
        """Check if symbol is in earnings blackout window."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'is_earnings_window'):
                return self._data_provider.is_earnings_window(symbol, days_before, days_after)
            return False
        except Exception:
            return False

    # Status and monitoring
    def get_status(self) -> Dict[str, Any]:
        """Get current decision chain status."""
        return {
            'trade_enabled': self.trade_enabled,
            'pool_size': len(self.stock_pool),
            'max_position_pct': self.max_position_pct,
            'threshold_boost': self.threshold_boost,
            'adjusted_threshold': self.base_threshold + self.threshold_boost,
            'cooldown_minutes': self.cooldown_minutes,
            'total_trades_today': self.total_trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'symbols_traded_today': len(self.trades_today),
        }


# Global singleton instance
_decision_chain: Optional[DecisionChain] = None


def get_decision_chain() -> DecisionChain:
    """Get the global DecisionChain singleton."""
    global _decision_chain
    if _decision_chain is None:
        _decision_chain = DecisionChain()
    return _decision_chain


def set_decision_chain(chain: DecisionChain) -> None:
    """Set the global DecisionChain singleton."""
    global _decision_chain
    _decision_chain = chain
