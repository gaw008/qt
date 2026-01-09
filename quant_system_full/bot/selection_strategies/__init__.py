"""
Stock Selection Strategies Package

This package contains various stock selection strategies for the quantitative trading system.
Each strategy implements the BaseSelectionStrategy interface for consistent integration.

Available Strategies:
- ValueMomentumStrategy: Combines value metrics with momentum indicators
- TechnicalBreakoutStrategy: Identifies stocks with technical breakout patterns
- EarningsMomentumStrategy: Focuses on earnings momentum and surprises

Usage:
    from bot.selection_strategies import ValueMomentumStrategy
    
    strategy = ValueMomentumStrategy()
    selected_stocks = strategy.select_stocks(universe=['AAPL', 'MSFT', 'GOOGL'])
"""

from .base_strategy import BaseSelectionStrategy, SelectionResult, SelectionCriteria
from .value_momentum import ValueMomentumStrategy
from .technical_breakout import TechnicalBreakoutStrategy
from .earnings_momentum import EarningsMomentumStrategy

__all__ = [
    'BaseSelectionStrategy',
    'SelectionResult', 
    'SelectionCriteria',
    'ValueMomentumStrategy',
    'TechnicalBreakoutStrategy',
    'EarningsMomentumStrategy',
]

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'value_momentum': ValueMomentumStrategy,
    'technical_breakout': TechnicalBreakoutStrategy,
    'earnings_momentum': EarningsMomentumStrategy,
}


def get_strategy(strategy_name: str, **kwargs) -> BaseSelectionStrategy:
    """
    Factory function to create strategy instances by name.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy-specific configuration parameters
        
    Returns:
        Instantiated strategy object
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)


def list_available_strategies() -> list[str]:
    """
    Get list of available strategy names.
    
    Returns:
        List of strategy names that can be used with get_strategy()
    """
    return list(STRATEGY_REGISTRY.keys())