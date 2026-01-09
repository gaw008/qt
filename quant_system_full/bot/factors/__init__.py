"""
Multi-factor scoring system for quantitative trading.

This package provides comprehensive factor analysis modules including:
- Valuation factors (P/E, P/B, EV/EBITDA, etc.)
- Volume factors (OBV, VWAP, MFI, volume ratios)
- Momentum factors (RSI, ROC, price/volume momentum)
- Technical factors (MACD, Bollinger Bands, KDJ, ADX)
- Market sentiment factors (VIX, breadth, sector rotation)

All factors are designed to work together through the scoring_engine module.
"""

# Import main factor functions for convenience
try:
    from .valuation import valuation_score
    from .volume_factors import volume_features, cross_section_volume_score
    from .momentum_factors import momentum_features, cross_section_momentum_score
    from .technical_factors import technical_features, cross_section_technical_score
    from .market_factors import market_sentiment_features, cross_section_market_score
except ImportError as e:
    # Handle potential missing dependencies gracefully
    import warnings
    warnings.warn(f"Some factor modules could not be imported: {e}")

__version__ = "1.0.0"
__all__ = [
    "valuation_score",
    "volume_features", 
    "cross_section_volume_score",
    "momentum_features",
    "cross_section_momentum_score", 
    "technical_features",
    "cross_section_technical_score",
    "market_sentiment_features",
    "cross_section_market_score"
]