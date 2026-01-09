"""
Market Regime Detection Module

This module provides market regime detection capabilities for adaptive trading strategies.
"""

from .market_regime_detector import MarketRegimeDetector, MarketRegime
from .regime_strategy_adapter import RegimeStrategyAdapter

__all__ = [
    'MarketRegimeDetector',
    'MarketRegime',
    'RegimeStrategyAdapter'
]