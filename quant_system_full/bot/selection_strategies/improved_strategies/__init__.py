"""
Improved Selection Strategies V2 - Independent Module

This module contains improved versions of stock selection strategies
that address the momentum-chasing issues in the original strategies.

Key Improvements:
- 12-1 month momentum (skips recent period to avoid buying at peaks)
- Enhanced value weighting (reduced momentum bias)
- Overbought filtering (RSI, Bollinger Bands)
- Style diversification (value, momentum, balanced)

This module is completely independent and will not affect the original
system even if it encounters errors.
"""

from .improved_value_momentum_v2 import ImprovedValueMomentumV2
from .defensive_value import DefensiveValue
from .balanced_momentum import BalancedMomentum

__all__ = [
    'ImprovedValueMomentumV2',
    'DefensiveValue',
    'BalancedMomentum'
]

__version__ = '2.0.0'
