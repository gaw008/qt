"""
Intelligent Monitoring and Alerting Module

This module provides comprehensive monitoring and alerting capabilities for the trading system,
including TCA analysis, risk exposure monitoring, performance attribution, and intelligent alerts.
"""

from .trading_cost_analyzer import TradingCostAnalyzer
from .risk_exposure_monitor import RiskExposureMonitor
from .performance_attribution import PerformanceAttributor
# from .alert_system import AlertSystem  # Temporarily disabled due to email import issues

__all__ = [
    'TradingCostAnalyzer',
    'RiskExposureMonitor',
    'PerformanceAttributor',
    # 'AlertSystem'
]