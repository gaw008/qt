"""
Quantitative Trading System Improvement Modules

This package contains enhanced modules for:
- Cost modeling and slippage estimation
- Portfolio risk management and optimization
- Execution robustness and reliability
- System monitoring and alerting
- Backtesting and validation frameworks
"""

__version__ = "1.0.0"
__author__ = "Quant Trading System"

# Module imports
from . import cost_models
from . import risk_management
from . import execution
from . import monitoring
from . import backtest

__all__ = [
    'cost_models',
    'risk_management',
    'execution',
    'monitoring',
    'backtest'
]