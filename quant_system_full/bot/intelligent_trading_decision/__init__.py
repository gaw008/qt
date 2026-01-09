"""
Intelligent Trading Decision System

A 4-layer architecture with 2 critical gates to reduce excessive trading:
- Layer 1: Stock Selection (Daily) - Universe Filter
- Layer 2: Daily Regime (Daily) - Position/Trade Control
- Layer 3: Signal Execution (Minute) - Directional Scoring
- Layer 4: Risk Control (System) - Hard Rules

Critical Gates:
- Gate 1: TriggerGate - Convert continuous signals to discrete events
- Gate 2: CostBenefitGate - Only trade when Edge > Cost

Phase 0: Trade History (Prerequisite for Learning)
- TradeHistorySync: Sync filled orders from Tiger API (90 days)
- TradeHistoryRecorder: Real-time trade recording with decision context
- TradeHistoryAnalyzer: Win rate calculations with cold start handling

Bug Fixes Implemented:
- FIX 1-4: TriggerGate data structure, direction-aware, volume protection
- FIX 5-7: CostBenefitGate consistent units, half-spread, fee handling
- FIX 8-9: Stability sparse signal, PriceAction .get() default
- FIX 10-11: ATR consistency, key level timing
- FIX 12-13: KeyLevelProvider auto-reset, NY timezone
- FIX 14: TriggerGate cooldown_seconds
- FIX 15: record_exit exit_trade_id capture
- FIX 16: PositionManager keyed by position_id
- FIX 17: PositionSizer signal/intent separation
"""

from .key_level_provider import (
    KeyLevelProvider,
    get_key_level_provider,
    set_key_level_provider,
)

from .stock_selection_filter import (
    StockSelectionFilter,
    TradabilityFilters,
    get_stock_selection_filter,
    set_stock_selection_filter,
)

from .trigger_gate import (
    TriggerGate,
    get_trigger_gate,
    set_trigger_gate,
)

from .position_manager import (
    Position,
    PositionManager,
    get_position_manager,
    set_position_manager,
)

from .position_sizer import (
    PositionSizer,
    get_position_sizer,
    set_position_sizer,
)

from .cost_benefit_gate import (
    CostBenefitGate,
    get_cost_benefit_gate,
    set_cost_benefit_gate,
)

from .exit_manager import (
    ExitManager,
    get_exit_manager,
    set_exit_manager,
)

from .directional_scorer import (
    DirectionalScorer,
    get_directional_scorer,
    set_directional_scorer,
)

from .decision_chain import (
    DecisionChain,
    get_decision_chain,
    set_decision_chain,
)

from .integration import (
    init_decision_system,
    filter_signals_through_decision_chain,
    check_position_exits,
    update_regime_from_market,
    reset_daily,
    get_decision_status,
    enable_decision_filter,
    disable_decision_filter,
    set_score_threshold,
    add_symbols_to_pool,
)

from .startup_validator import (
    validate_startup,
    get_system_info,
    StartupValidationError,
)

from .adaptive_learner import (
    AdaptiveLearner,
    get_adaptive_learner,
    set_adaptive_learner,
)

from .trade_history_sync import (
    TradeHistorySync,
    get_trade_history_sync,
    set_trade_history_sync,
)

from .trade_history_recorder import (
    TradeHistoryRecorder,
    get_trade_history_recorder,
    set_trade_history_recorder,
)

from .trade_history_analyzer import (
    TradeHistoryAnalyzer,
    get_trade_history_analyzer,
    set_trade_history_analyzer,
)

__all__ = [
    # KeyLevelProvider
    'KeyLevelProvider',
    'get_key_level_provider',
    'set_key_level_provider',
    # StockSelectionFilter
    'StockSelectionFilter',
    'TradabilityFilters',
    'get_stock_selection_filter',
    'set_stock_selection_filter',
    # TriggerGate
    'TriggerGate',
    'get_trigger_gate',
    'set_trigger_gate',
    # PositionManager
    'Position',
    'PositionManager',
    'get_position_manager',
    'set_position_manager',
    # PositionSizer
    'PositionSizer',
    'get_position_sizer',
    'set_position_sizer',
    # CostBenefitGate
    'CostBenefitGate',
    'get_cost_benefit_gate',
    'set_cost_benefit_gate',
    # ExitManager
    'ExitManager',
    'get_exit_manager',
    'set_exit_manager',
    # DirectionalScorer
    'DirectionalScorer',
    'get_directional_scorer',
    'set_directional_scorer',
    # DecisionChain
    'DecisionChain',
    'get_decision_chain',
    'set_decision_chain',
    # Integration helpers
    'init_decision_system',
    'filter_signals_through_decision_chain',
    'check_position_exits',
    'update_regime_from_market',
    'reset_daily',
    'get_decision_status',
    'enable_decision_filter',
    'disable_decision_filter',
    'set_score_threshold',
    'add_symbols_to_pool',
    # Startup validation
    'validate_startup',
    'get_system_info',
    'StartupValidationError',
    # AdaptiveLearner
    'AdaptiveLearner',
    'get_adaptive_learner',
    'set_adaptive_learner',
    # TradeHistorySync
    'TradeHistorySync',
    'get_trade_history_sync',
    'set_trade_history_sync',
    # TradeHistoryRecorder
    'TradeHistoryRecorder',
    'get_trade_history_recorder',
    'set_trade_history_recorder',
    # TradeHistoryAnalyzer
    'TradeHistoryAnalyzer',
    'get_trade_history_analyzer',
    'set_trade_history_analyzer',
]

__version__ = '1.1.0'
