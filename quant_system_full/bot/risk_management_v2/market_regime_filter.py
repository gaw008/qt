"""Market Regime Filter - detects market environment and adjusts exposure"""

import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class MarketRegimeFilter:
    """Detects market regime and recommends exposure adjustments."""

    def __init__(self,
                 vix_threshold_high: float = 25.0,
                 vix_threshold_low: float = 15.0):
        """
        Initialize market regime filter.

        Args:
            vix_threshold_high: VIX above this = high volatility/bearish
            vix_threshold_low: VIX below this = low volatility/bullish
        """
        self.vix_threshold_high = vix_threshold_high
        self.vix_threshold_low = vix_threshold_low

        logger.info(f"MarketRegimeFilter initialized: VIX thresholds [{vix_threshold_low}, {vix_threshold_high}]")

    def get_market_regime(self) -> MarketRegime:
        """
        Determine current market regime.

        Simplified version - can be enhanced with real SPY/VIX data.

        Returns:
            MarketRegime enum
        """
        try:
            # Simplified: assume neutral market
            # In production, would fetch SPY MA200, VIX, breadth indicators
            return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return MarketRegime.UNKNOWN

    def should_reduce_exposure(self) -> bool:
        """
        Determine if exposure should be reduced.

        Returns:
            True if should reduce positions
        """
        try:
            regime = self.get_market_regime()

            if regime == MarketRegime.BEAR:
                logger.warning("Bear market detected - recommend reducing exposure")
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking exposure: {e}")
            return False

    def get_recommended_max_stocks(self, default_max: int = 10) -> int:
        """
        Get recommended max stocks based on regime.

        Args:
            default_max: Default maximum stocks

        Returns:
            Adjusted maximum stocks
        """
        try:
            regime = self.get_market_regime()

            if regime == MarketRegime.BEAR:
                return min(5, default_max)  # Reduce to 5 in bear market
            elif regime == MarketRegime.BULL:
                return default_max  # Full allocation in bull market
            else:
                return default_max  # Normal allocation

        except Exception as e:
            logger.error(f"Error getting recommended max stocks: {e}")
            return default_max
