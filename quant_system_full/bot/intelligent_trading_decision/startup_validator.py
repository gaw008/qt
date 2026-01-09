"""
Startup Validator - Validates all decision system modules on startup.

Call validate_startup() in runner.py to verify all components are working.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class StartupValidationError(Exception):
    """Raised when startup validation fails."""
    pass


def validate_startup(raise_on_error: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Validate all decision system modules on startup.

    Args:
        raise_on_error: If True, raise StartupValidationError on failure

    Returns:
        Tuple of (success: bool, passed: List[str], failed: List[str])
    """
    passed = []
    failed = []

    # 1. Validate module imports
    import_result = _validate_imports()
    if import_result[0]:
        passed.append("Module imports")
    else:
        failed.append(f"Module imports: {import_result[1]}")

    # 2. Validate KeyLevelProvider
    klp_result = _validate_key_level_provider()
    if klp_result[0]:
        passed.append("KeyLevelProvider")
    else:
        failed.append(f"KeyLevelProvider: {klp_result[1]}")

    # 3. Validate TriggerGate
    tg_result = _validate_trigger_gate()
    if tg_result[0]:
        passed.append("TriggerGate")
    else:
        failed.append(f"TriggerGate: {tg_result[1]}")

    # 4. Validate PositionManager
    pm_result = _validate_position_manager()
    if pm_result[0]:
        passed.append("PositionManager")
    else:
        failed.append(f"PositionManager: {pm_result[1]}")

    # 5. Validate PositionSizer
    ps_result = _validate_position_sizer()
    if ps_result[0]:
        passed.append("PositionSizer")
    else:
        failed.append(f"PositionSizer: {ps_result[1]}")

    # 6. Validate CostBenefitGate
    cbg_result = _validate_cost_benefit_gate()
    if cbg_result[0]:
        passed.append("CostBenefitGate")
    else:
        failed.append(f"CostBenefitGate: {cbg_result[1]}")

    # 7. Validate ExitManager
    em_result = _validate_exit_manager()
    if em_result[0]:
        passed.append("ExitManager")
    else:
        failed.append(f"ExitManager: {em_result[1]}")

    # 8. Validate DecisionChain
    dc_result = _validate_decision_chain()
    if dc_result[0]:
        passed.append("DecisionChain")
    else:
        failed.append(f"DecisionChain: {dc_result[1]}")

    # 9. Validate Integration helpers
    int_result = _validate_integration()
    if int_result[0]:
        passed.append("Integration helpers")
    else:
        failed.append(f"Integration: {int_result[1]}")

    success = len(failed) == 0

    # Log results
    _log_validation_results(passed, failed)

    if not success and raise_on_error:
        raise StartupValidationError(f"Startup validation failed: {failed}")

    return success, passed, failed


def _validate_imports() -> Tuple[bool, str]:
    """Validate all module imports."""
    try:
        from . import (
            KeyLevelProvider,
            TriggerGate,
            PositionManager,
            PositionSizer,
            CostBenefitGate,
            ExitManager,
            DecisionChain,
            init_decision_system,
            filter_signals_through_decision_chain,
        )
        return True, "OK"
    except ImportError as e:
        return False, str(e)


def _validate_key_level_provider() -> Tuple[bool, str]:
    """Validate KeyLevelProvider functionality."""
    try:
        from .key_level_provider import KeyLevelProvider, get_key_level_provider

        # Test instantiation
        klp = KeyLevelProvider()

        # Test get_key_levels with mock time
        bar_time = datetime.now()
        levels = klp.get_key_levels("AAPL", bar_time)

        # Verify structure
        required_keys = ['or_high', 'or_low', 'prev_high', 'prev_low', 'prev_close', 'vwap']
        for key in required_keys:
            if key not in levels:
                return False, f"Missing key: {key}"

        # Test session reset (FIX 12)
        tomorrow = bar_time + timedelta(days=1)
        klp.get_key_levels("AAPL", tomorrow)
        if klp._current_session_date != tomorrow.date():
            return False, "FIX 12: Session reset failed"

        # Test singleton
        singleton = get_key_level_provider()
        if singleton is None:
            return False, "Singleton not created"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_trigger_gate() -> Tuple[bool, str]:
    """Validate TriggerGate functionality."""
    try:
        from .trigger_gate import TriggerGate, get_trigger_gate

        # Test instantiation
        tg = TriggerGate()

        # Verify FIX 14: cooldown_seconds (not bars)
        if not hasattr(tg, 'cooldown_seconds'):
            return False, "FIX 14: Missing cooldown_seconds"
        if tg.cooldown_seconds != 1800:  # 30 minutes
            return False, f"FIX 14: Wrong cooldown value: {tg.cooldown_seconds}"

        # Test check method signature (FIX 14: requires bar_time)
        bar_time = datetime.now()
        try:
            triggered, reason = tg.check(
                symbol="AAPL",
                signal="BUY",
                price=150.0,
                prev_price=149.0,
                bar_volume=100000,
                bar_time=bar_time
            )
        except TypeError as e:
            return False, f"FIX 14: bar_time parameter issue: {e}"

        # Test reset
        tg.reset_daily()
        if len(tg.last_trigger_time) != 0:
            return False, "reset_daily failed"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_position_manager() -> Tuple[bool, str]:
    """Validate PositionManager functionality."""
    try:
        from .position_manager import PositionManager, Position, get_position_manager

        # Test instantiation
        pm = PositionManager(single_position_per_symbol=True)

        # Test position opening (FIX 16: keyed by position_id)
        bar_time = datetime.now()
        position = pm.open_position(
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            quantity=100,
            entry_time=bar_time
        )

        # Verify position_id is used
        if position.position_id not in pm.active_positions:
            return False, "FIX 16: Position not keyed by position_id"

        # Verify symbol lookup works
        found = pm.get_position_by_symbol("AAPL")
        if found is None:
            return False, "FIX 16: Symbol lookup failed"

        # Test duplicate prevention
        try:
            pm.open_position(
                symbol="AAPL",
                direction="LONG",
                entry_price=151.0,
                quantity=50,
                entry_time=bar_time
            )
            return False, "FIX 16: Should prevent duplicate symbol"
        except ValueError:
            pass  # Expected

        # Test closing
        closed = pm.close_position(
            position_id=position.position_id,
            exit_price=155.0,
            exit_reason="Test",
            exit_time=bar_time
        )
        if closed.pnl != 500.0:  # (155-150) * 100
            return False, f"P&L calculation wrong: {closed.pnl}"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_position_sizer() -> Tuple[bool, str]:
    """Validate PositionSizer functionality."""
    try:
        from .position_sizer import PositionSizer, get_position_sizer

        # Test instantiation
        ps = PositionSizer(allow_short=False)

        # Verify FIX 17: allow_short default is False
        if ps.allow_short != False:
            return False, "FIX 17: allow_short should default to False"

        # Test intent derivation (FIX 17)
        intent = ps.derive_intent(signal='BUY', current_shares=0)
        if intent != 'OPEN':
            return False, f"FIX 17: Wrong intent for BUY with 0 shares: {intent}"

        intent = ps.derive_intent(signal='SELL', current_shares=100)
        if intent != 'CLOSE':
            return False, f"FIX 17: Wrong intent for SELL with long: {intent}"

        # Test SELL with no position and short disabled (FIX 17)
        delta, reason, intent = ps.calculate_position_for_signal(
            signal='SELL',
            symbol='AAPL',
            current_shares=0,
            score=75,
            regime={'max_position_pct': 1.0},
            current_price=150.0
        )
        if delta != 0:
            return False, "FIX 17: SELL with no position should return 0"
        if "short disabled" not in reason.lower():
            return False, f"FIX 17: Wrong reason: {reason}"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_cost_benefit_gate() -> Tuple[bool, str]:
    """Validate CostBenefitGate functionality."""
    try:
        from .cost_benefit_gate import CostBenefitGate, get_cost_benefit_gate

        # Test instantiation
        cbg = CostBenefitGate()

        # Verify FIX 5-7 parameters
        if cbg.edge_multiple != 2.5:
            return False, f"Wrong edge_multiple: {cbg.edge_multiple}"
        if cbg.fee_per_share != 0.005:
            return False, "FIX 7: Wrong fee_per_share"
        if cbg.min_order_fee != 1.0:
            return False, "FIX 7: Wrong min_order_fee"

        # Test check method (without data provider, should use defaults)
        passed, reason = cbg.check(
            symbol="AAPL",
            price=150.0,
            signal_score=75,
            shares=100
        )
        # Should return a result (True or False with reason)
        if not isinstance(passed, bool):
            return False, "check() should return bool"

        # Test cost breakdown
        breakdown = cbg.get_cost_breakdown("AAPL", 150.0, 100)
        required_keys = ['half_spread', 'slippage', 'fee', 'total_cost_per_share']
        for key in required_keys:
            if key not in breakdown:
                return False, f"FIX 5: Missing {key} in breakdown"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_exit_manager() -> Tuple[bool, str]:
    """Validate ExitManager functionality."""
    try:
        from .exit_manager import ExitManager, get_exit_manager
        from .position_manager import Position

        # Test instantiation
        em = ExitManager()

        # Verify ATR multiples
        if em.stop_loss_atr_multiple != 1.0:
            return False, f"Wrong stop_loss_atr_multiple: {em.stop_loss_atr_multiple}"
        if em.take_profit_atr_multiple != 1.5:
            return False, f"Wrong take_profit_atr_multiple: {em.take_profit_atr_multiple}"

        # Test check_exit with mock position
        bar_time = datetime.now()
        position = Position(
            position_id="test",
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            quantity=100,
            entry_time=bar_time - timedelta(minutes=10)
        )

        # Test stop loss hit
        should_exit, reason = em.check_exit(position, 148.0, bar_time)
        # Without data provider, uses 1% default ATR = $1.50
        # Stop loss at 150 - 1.5 = 148.5, so 148.0 should trigger
        if not should_exit:
            return False, "Stop loss should trigger"

        # Test get_exit_levels
        levels = em.get_exit_levels(150.0, "LONG", "AAPL")
        if 'stop_loss' not in levels or 'take_profit' not in levels:
            return False, "Missing exit levels"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_decision_chain() -> Tuple[bool, str]:
    """Validate DecisionChain functionality."""
    try:
        from .decision_chain import DecisionChain, get_decision_chain

        # Test instantiation
        dc = DecisionChain()

        # Verify default config
        if dc.base_threshold != 65:
            return False, f"Wrong base_threshold: {dc.base_threshold}"
        if dc.cooldown_minutes != 20:
            return False, f"Wrong cooldown_minutes: {dc.cooldown_minutes}"
        if dc.max_trades_per_day != 20:
            return False, f"Wrong max_trades_per_day: {dc.max_trades_per_day}"

        # Test pool management
        dc.set_stock_pool({"AAPL", "MSFT", "GOOGL"})
        if len(dc.stock_pool) != 3:
            return False, "Pool management failed"

        # Test regime update
        # VIX=25.0 is in range [25, 30) so threshold_boost should be 10
        dc.update_regime_from_vix(25.0)
        if dc.threshold_boost != 10:
            return False, f"VIX regime boost wrong: {dc.threshold_boost} (expected 10 for VIX=25)"

        # Test reset
        dc.reset_daily()
        if dc.total_trades_today != 0:
            return False, "reset_daily failed"

        # Test status
        status = dc.get_status()
        if 'trade_enabled' not in status:
            return False, "Missing status fields"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _validate_integration() -> Tuple[bool, str]:
    """Validate integration helpers."""
    try:
        from .integration import (
            init_decision_system,
            filter_signals_through_decision_chain,
            check_position_exits,
            reset_daily,
            get_decision_status,
            enable_decision_filter,
            disable_decision_filter,
        )

        # Test init
        ds = init_decision_system(
            config={'base_threshold': 70},
            stock_pool={'AAPL', 'MSFT'}
        )
        if ds.base_threshold != 70:
            return False, "Config override failed"

        # Test signal filtering
        test_signals = {
            'buy': [{'symbol': 'AAPL', 'price': 150.0, 'score': 80}],
            'sell': [],
            'hold': []
        }
        filtered = filter_signals_through_decision_chain(test_signals)
        if 'blocked' not in filtered:
            return False, "Missing blocked list"

        # Test status
        status = get_decision_status()
        if not isinstance(status, dict):
            return False, "get_decision_status failed"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def _log_validation_results(passed: List[str], failed: List[str]):
    """Log validation results."""
    total = len(passed) + len(failed)

    logger.info("=" * 60)
    logger.info("INTELLIGENT TRADING DECISION SYSTEM - STARTUP VALIDATION")
    logger.info("=" * 60)

    if passed:
        logger.info(f"PASSED ({len(passed)}/{total}):")
        for item in passed:
            logger.info(f"  [OK] {item}")

    if failed:
        logger.error(f"FAILED ({len(failed)}/{total}):")
        for item in failed:
            logger.error(f"  [FAIL] {item}")

    if not failed:
        logger.info("-" * 60)
        logger.info("ALL MODULES VALIDATED SUCCESSFULLY")
        logger.info("-" * 60)
    else:
        logger.error("-" * 60)
        logger.error(f"VALIDATION FAILED: {len(failed)} modules have issues")
        logger.error("-" * 60)


def get_system_info() -> Dict[str, Any]:
    """Get system information for logging."""
    try:
        from . import __version__
        version = __version__
    except ImportError:
        version = "unknown"

    return {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'modules': [
            'KeyLevelProvider',
            'TriggerGate',
            'PositionManager',
            'PositionSizer',
            'CostBenefitGate',
            'ExitManager',
            'DecisionChain',
        ],
        'bug_fixes': [
            'FIX 1-4: TriggerGate improvements',
            'FIX 5-7: CostBenefitGate units',
            'FIX 12: KeyLevelProvider auto-reset',
            'FIX 13: NY timezone handling',
            'FIX 14: cooldown_seconds',
            'FIX 16: PositionManager by position_id',
            'FIX 17: PositionSizer signal/intent',
        ]
    }
