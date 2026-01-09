"""
Integration module for Phase 1 improvements
Provides adapters and wrappers for integrating enhancement modules with the main trading system
"""

from .cost_aware_trading_adapter import CostAwareTradingAdapter, create_cost_aware_trading_engine

__all__ = [
    'CostAwareTradingAdapter',
    'create_cost_aware_trading_engine'
]