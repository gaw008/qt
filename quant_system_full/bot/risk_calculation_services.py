#!/usr/bin/env python3
"""
Risk Calculation Services - Refactored Risk Management Components

This module contains specialized services for risk calculations,
following Single Responsibility Principle.

Services:
- TailRiskCalculator: Expected Shortfall and tail risk metrics
- RegimeDetectionService: Market regime detection and adjustment
- DrawdownManager: Drawdown tier management and actions
- CorrelationAnalyzer: Portfolio correlation and dependence analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states for dynamic risk adjustment"""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    TRENDING = "TRENDING"
    CRISIS = "CRISIS"


@dataclass
class TailRiskMetrics:
    """Tail risk and extreme event metrics"""
    es_97_5: float = 0.0        # Expected Shortfall @ 97.5%
    es_99: float = 0.0          # Expected Shortfall @ 99%
    tail_ratio: float = 0.0     # Ratio of gains to losses in tails
    skewness: float = 0.0       # Return distribution skewness
    kurtosis: float = 0.0       # Return distribution kurtosis
    max_drawdown: float = 0.0   # Maximum historical drawdown
    calmar_ratio: float = 0.0   # Return / Max Drawdown
    tail_dependence: float = 0.0 # Tail dependence with market


class TailRiskCalculator:
    """
    Specialized service for tail risk calculations.
    Handles Expected Shortfall and extreme event analysis.
    """

    def calculate_expected_shortfall(self,
                                   returns: np.ndarray,
                                   confidence_level: float = 0.975) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR) at specified confidence level.

        ES is superior to VaR as it measures the expected loss in the tail beyond VaR.
        ES@97.5% = E[Loss | Loss > VaR@97.5%]

        Args:
            returns: Array of return values
            confidence_level: Confidence level for ES calculation

        Returns:
            Expected Shortfall as positive value
        """
        if len(returns) == 0:
            return 0.0

        # Sort returns (losses are negative)
        sorted_returns = np.sort(returns)

        # Find VaR cutoff point
        var_index = int((1 - confidence_level) * len(sorted_returns))
        if var_index == 0:
            var_index = 1

        # ES is the mean of returns below VaR
        tail_returns = sorted_returns[:var_index]
        if len(tail_returns) == 0:
            return 0.0

        expected_shortfall = np.mean(tail_returns)
        return abs(expected_shortfall)  # Return as positive value

    def calculate_tail_dependence(self,
                                portfolio_returns: np.ndarray,
                                market_returns: np.ndarray,
                                threshold: float = 0.95) -> float:
        """
        Calculate tail dependence between portfolio and market.
        Measures correlation during extreme market events.

        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market benchmark return series
            threshold: Threshold for extreme events

        Returns:
            Tail dependence coefficient
        """
        if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 20:
            return 0.0

        # Find extreme market events (bottom 5%)
        market_threshold = np.percentile(market_returns, (1 - threshold) * 100)
        extreme_market_mask = market_returns <= market_threshold

        if np.sum(extreme_market_mask) < 5:
            return 0.0

        # Calculate correlation during extreme events
        extreme_portfolio = portfolio_returns[extreme_market_mask]
        extreme_market = market_returns[extreme_market_mask]

        if len(extreme_portfolio) < 3:
            return 0.0

        try:
            correlation, _ = pearsonr(extreme_portfolio, extreme_market)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def calculate_comprehensive_tail_metrics(self, returns: np.ndarray) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics.

        Args:
            returns: Return series for analysis

        Returns:
            TailRiskMetrics object with all tail risk measures
        """
        if len(returns) < 10:
            return TailRiskMetrics()

        try:
            # Basic statistics
            from scipy import stats
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

            # Expected Shortfall calculations
            es_97_5 = self.calculate_expected_shortfall(returns, 0.975)
            es_99 = self.calculate_expected_shortfall(returns, 0.99)

            # Tail ratio calculation
            tail_ratio = self._calculate_tail_ratio(returns)

            # Drawdown metrics
            max_drawdown, calmar_ratio = self._calculate_drawdown_metrics(returns)

            return TailRiskMetrics(
                es_97_5=es_97_5,
                es_99=es_99,
                tail_ratio=tail_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio
            )

        except Exception as e:
            logger.warning(f"Error calculating tail metrics: {e}")
            return TailRiskMetrics()

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (average gain / average loss in tails)."""
        positive_tail = returns[returns > np.percentile(returns, 95)]
        negative_tail = returns[returns < np.percentile(returns, 5)]

        if len(positive_tail) > 0 and len(negative_tail) > 0:
            avg_gain = np.mean(positive_tail)
            avg_loss = abs(np.mean(negative_tail))
            return avg_gain / avg_loss if avg_loss > 0 else 0.0
        return 0.0

    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate drawdown and Calmar ratio."""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # Calmar ratio (annualized return / max drawdown)
        ann_return = np.mean(returns) * 252  # Assuming daily returns
        calmar_ratio = ann_return / max_drawdown if max_drawdown > 0 else 0.0

        return max_drawdown, calmar_ratio


class RegimeDetectionService:
    """
    Service for detecting market regimes and adjusting risk parameters.
    """

    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Detect current market regime based on volatility, correlation, and sentiment.

        Regime classification:
        - NORMAL: VIX < 20, correlation < 0.5
        - VOLATILE: VIX 20-30, correlation 0.5-0.7
        - TRENDING: VIX < 25, strong directional momentum
        - CRISIS: VIX > 30, correlation > 0.7

        Args:
            market_data: Dictionary containing market indicators

        Returns:
            Detected market regime
        """
        vix = market_data.get('vix', 20)
        correlation = market_data.get('market_correlation', 0.5)
        momentum = market_data.get('momentum_strength', 0.0)

        if vix > 30 and correlation > 0.7:
            return MarketRegime.CRISIS
        elif vix > 20 and correlation > 0.5:
            return MarketRegime.VOLATILE
        elif abs(momentum) > 0.5 and vix < 25:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.NORMAL

    def get_regime_multipliers(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get risk parameter multipliers for the current market regime.

        Args:
            regime: Current market regime

        Returns:
            Dictionary of multipliers for risk parameters
        """
        regime_multipliers = {
            MarketRegime.NORMAL: {"var": 1.0, "position": 1.0, "es": 1.0},
            MarketRegime.VOLATILE: {"var": 0.8, "position": 0.8, "es": 1.2},
            MarketRegime.TRENDING: {"var": 1.1, "position": 1.1, "es": 0.9},
            MarketRegime.CRISIS: {"var": 0.5, "position": 0.5, "es": 2.0}
        }

        return regime_multipliers.get(regime, {"var": 1.0, "position": 1.0, "es": 1.0})

    def apply_regime_adjustments(self,
                               base_limits: Dict[str, float],
                               regime: MarketRegime) -> Dict[str, float]:
        """
        Apply regime-based adjustments to risk limits.

        Args:
            base_limits: Base risk limits
            regime: Current market regime

        Returns:
            Adjusted risk limits
        """
        multipliers = self.get_regime_multipliers(regime)

        adjusted_limits = base_limits.copy()

        # Apply multipliers to relevant limits
        if 'max_portfolio_var' in adjusted_limits:
            adjusted_limits['max_portfolio_var'] *= multipliers.get('var', 1.0)

        if 'max_single_position' in adjusted_limits:
            adjusted_limits['max_single_position'] *= multipliers.get('position', 1.0)

        if 'es_97_5_limit' in adjusted_limits:
            adjusted_limits['es_97_5_limit'] *= multipliers.get('es', 1.0)

        return adjusted_limits


@dataclass
class DrawdownTier:
    """Configuration for a single drawdown tier."""
    threshold: float
    actions: List[str]
    severity: str


class DrawdownManager:
    """
    Service for managing portfolio drawdown tiers and actions.
    """

    def __init__(self):
        self.tiers = [
            DrawdownTier(0.08, ["reduce_position_size_10%", "increase_stop_loss_tightness", "pause_new_positions"], "TIER_1"),
            DrawdownTier(0.12, ["reduce_position_size_25%", "reduce_sector_concentration", "increase_cash_allocation"], "TIER_2"),
            DrawdownTier(0.15, ["reduce_position_size_50%", "close_high_correlation_positions", "emergency_risk_off"], "TIER_3")
        ]

    def check_drawdown_tier(self, current_drawdown: float) -> Tuple[int, List[str], str]:
        """
        Check which drawdown tier is activated and return appropriate actions.

        Args:
            current_drawdown: Current portfolio drawdown level

        Returns:
            Tuple of (tier_level, suggested_actions, severity)
        """
        for i, tier in enumerate(reversed(self.tiers), 1):
            if current_drawdown >= tier.threshold:
                tier_level = len(self.tiers) - i + 1
                return tier_level, tier.actions, tier.severity

        return 0, [], "NORMAL"

    def get_tier_configuration(self) -> List[Dict[str, Any]]:
        """Get current tier configuration for reporting."""
        return [
            {
                "tier": i + 1,
                "threshold": tier.threshold,
                "actions": tier.actions,
                "severity": tier.severity
            }
            for i, tier in enumerate(self.tiers)
        ]

    def update_tier_threshold(self, tier_level: int, new_threshold: float) -> bool:
        """
        Update threshold for a specific tier.

        Args:
            tier_level: Tier level (1, 2, or 3)
            new_threshold: New threshold value

        Returns:
            True if update successful, False otherwise
        """
        try:
            if 1 <= tier_level <= len(self.tiers):
                self.tiers[tier_level - 1].threshold = new_threshold
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating tier threshold: {e}")
            return False


class CorrelationAnalyzer:
    """
    Service for analyzing portfolio correlations and dependencies.
    """

    def calculate_portfolio_correlation_matrix(self,
                                             returns_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio assets.

        Args:
            returns_data: Dictionary of symbol -> returns array

        Returns:
            Correlation matrix as pandas DataFrame
        """
        if not returns_data:
            return pd.DataFrame()

        try:
            # Align all return series to same length
            min_length = min(len(returns) for returns in returns_data.values())

            aligned_data = {}
            for symbol, returns in returns_data.items():
                if len(returns) >= min_length:
                    aligned_data[symbol] = returns[-min_length:]

            if not aligned_data:
                return pd.DataFrame()

            # Create DataFrame and calculate correlation
            df = pd.DataFrame(aligned_data)
            correlation_matrix = df.corr()

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def identify_high_correlation_pairs(self,
                                      correlation_matrix: pd.DataFrame,
                                      threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of assets with high correlation.

        Args:
            correlation_matrix: Asset correlation matrix
            threshold: Correlation threshold for high correlation

        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        high_corr_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                asset1 = correlation_matrix.columns[i]
                asset2 = correlation_matrix.columns[j]
                corr = abs(correlation_matrix.iloc[i, j])

                if not np.isnan(corr) and corr >= threshold:
                    high_corr_pairs.append((asset1, asset2, corr))

        return high_corr_pairs

    def calculate_diversification_ratio(self,
                                      weights: np.ndarray,
                                      volatilities: np.ndarray,
                                      correlation_matrix: np.ndarray) -> float:
        """
        Calculate portfolio diversification ratio.

        Args:
            weights: Portfolio weights
            volatilities: Individual asset volatilities
            correlation_matrix: Asset correlation matrix

        Returns:
            Diversification ratio (higher is better)
        """
        try:
            # Weighted average volatility
            weighted_avg_vol = np.sum(weights * volatilities)

            # Portfolio volatility
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
            portfolio_vol = np.sqrt(portfolio_variance)

            # Diversification ratio
            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 0.0

    def analyze_concentration_risk(self,
                                 weights: np.ndarray,
                                 sectors: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze portfolio concentration risk.

        Args:
            weights: Portfolio weights
            sectors: Optional sector classifications

        Returns:
            Dictionary with concentration metrics
        """
        try:
            # Herfindahl-Hirschman Index for concentration
            hhi = np.sum(weights ** 2)

            # Maximum weight
            max_weight = np.max(weights) if len(weights) > 0 else 0.0

            # Effective number of positions
            effective_positions = 1.0 / hhi if hhi > 0 else 0.0

            concentration_metrics = {
                'herfindahl_index': hhi,
                'max_weight': max_weight,
                'effective_positions': effective_positions,
                'concentration_ratio_top5': np.sum(np.sort(weights)[-5:]) if len(weights) >= 5 else np.sum(weights)
            }

            # Sector concentration if sector data provided
            if sectors and len(sectors) == len(weights):
                sector_weights = {}
                for sector, weight in zip(sectors, weights):
                    sector_weights[sector] = sector_weights.get(sector, 0) + weight

                concentration_metrics['sector_concentration'] = sector_weights
                concentration_metrics['max_sector_weight'] = max(sector_weights.values()) if sector_weights else 0.0

            return concentration_metrics

        except Exception as e:
            logger.error(f"Error analyzing concentration risk: {e}")
            return {}