"""
Risk Management V2 - Independent Module

Comprehensive risk management layer for improved selection strategies.

Components:
- StopLossManager: Stop-loss and profit-lock mechanisms
- MarketRegimeFilter: Market environment detection and exposure adjustment
- PortfolioRiskControl: Portfolio-level risk constraints

This module is completely independent and will not affect the original
system even if it encounters errors.
"""

from .stop_loss_manager import StopLossManager
from .market_regime_filter import MarketRegimeFilter
from .portfolio_risk_control import PortfolioRiskControl

__all__ = [
    'StopLossManager',
    'MarketRegimeFilter',
    'PortfolioRiskControl'
]

__version__ = '2.0.0'
