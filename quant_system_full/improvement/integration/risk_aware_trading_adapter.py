#!/usr/bin/env python3
"""
Risk-Aware Trading Adapter

Integrates advanced portfolio risk management into the trading system
without disrupting existing workflows. Provides real-time risk monitoring,
VaR calculations, and position concentration limits.

This adapter wraps the existing AutoTradingEngine with risk awareness.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bot"))
sys.path.insert(0, str(project_root / "improvement"))

from risk_management.portfolio_risk_manager import (
    PortfolioRiskManager, RiskBudget, RiskMetrics, create_risk_manager
)

logger = logging.getLogger(__name__)


@dataclass
class RiskConstraints:
    """Risk constraints for trading decisions"""
    max_portfolio_var_95: float = 0.05  # 5% daily VaR
    max_position_concentration: float = 0.15  # 15% max position weight
    max_correlation_threshold: float = 0.8  # 80% max correlation
    min_diversification_count: int = 5  # Minimum 5 positions
    max_sector_concentration: float = 0.25  # 25% max sector weight


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    approved: bool
    risk_score: float  # 0-100 scale
    var_95: float
    concentration_risk: float
    warnings: List[str]
    recommendations: List[str]
    position_limits: Dict[str, float]


class RiskAwareTradingAdapter:
    """
    Risk-aware trading adapter that integrates portfolio risk management

    This adapter provides:
    - Real-time portfolio risk monitoring
    - VaR-based position sizing
    - Concentration risk controls
    - Dynamic risk limit enforcement
    """

    def __init__(self,
                 auto_trading_engine=None,
                 dry_run: bool = True,
                 risk_constraints: Optional[RiskConstraints] = None,
                 enable_risk_monitoring: bool = True):
        """
        Initialize risk-aware trading adapter

        Args:
            auto_trading_engine: Existing AutoTradingEngine instance
            dry_run: Whether running in dry run mode
            risk_constraints: Risk constraint parameters
            enable_risk_monitoring: Enable real-time risk monitoring
        """
        self.auto_trading_engine = auto_trading_engine
        self.dry_run = dry_run
        self.enable_risk_monitoring = enable_risk_monitoring
        self.risk_constraints = risk_constraints or RiskConstraints()

        # Initialize risk manager
        risk_config = {
            'confidence_levels': [0.95, 0.99],
            'risk_horizon_days': 1,
            'lookback_days': 252,
            'min_observations': 30
        }
        self.risk_manager = create_risk_manager(risk_config)

        # Risk budget setup
        self.risk_budget = RiskBudget(
            total_budget=self.risk_constraints.max_portfolio_var_95,
            sector_limits={},  # Will be populated dynamically
            position_limits={},  # Will be calculated based on concentration limits
            var_limit=self.risk_constraints.max_portfolio_var_95,
            correlation_limit=self.risk_constraints.max_correlation_threshold,
            concentration_limit=self.risk_constraints.max_position_concentration ** 2  # HHI format
        )

        # Cache for performance
        self._last_price_update = None
        self._price_data_cache = {}
        self._risk_metrics_cache = None
        self._cache_expiry = None

        logger.info("Risk-Aware Trading Adapter initialized")

    def update_market_data(self, market_data: Dict[str, Any]):
        """
        Update market data for risk calculations

        Args:
            market_data: Dictionary with symbol -> price data
        """
        try:
            # Convert market data to DataFrame format
            price_data = {}
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'price' in data:
                    price_data[symbol] = data['price']
                elif isinstance(data, (int, float)):
                    price_data[symbol] = data

            # Create price DataFrame with historical simulation
            # For real implementation, this should use actual historical price data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            prices_df = pd.DataFrame(index=dates)

            for symbol, current_price in price_data.items():
                # Simulate historical prices with random walk (for demonstration)
                # In production, use actual historical data
                np.random.seed(hash(symbol) % 2**32)  # Reproducible random seed per symbol
                returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
                prices = [current_price]
                for ret in reversed(returns[1:]):
                    prices.append(prices[-1] / (1 + ret))
                prices_df[symbol] = list(reversed(prices))

            # Update risk manager
            self.risk_manager.update_price_data(prices_df)
            self._last_price_update = datetime.now()
            self._price_data_cache = price_data

            # Clear cache
            self._risk_metrics_cache = None

            logger.debug(f"Updated market data for {len(price_data)} symbols")

        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            raise

    def assess_portfolio_risk(self,
                            current_positions: Dict[str, float],
                            proposed_positions: Dict[str, float]) -> RiskAssessment:
        """
        Assess risk of proposed portfolio changes

        Args:
            current_positions: Current portfolio positions {symbol: value}
            proposed_positions: Proposed portfolio positions {symbol: value}

        Returns:
            RiskAssessment with risk metrics and recommendations
        """
        try:
            warnings = []
            recommendations = []

            # Convert positions to weights
            total_value = sum(abs(v) for v in proposed_positions.values())
            if total_value == 0:
                return RiskAssessment(
                    approved=True,
                    risk_score=0.0,
                    var_95=0.0,
                    concentration_risk=0.0,
                    warnings=[],
                    recommendations=[],
                    position_limits={}
                )

            weights = {symbol: value / total_value for symbol, value in proposed_positions.items()}

            # Calculate risk metrics
            risk_metrics = self._get_cached_risk_metrics(weights)

            # Risk scoring (0-100 scale)
            risk_score = self._calculate_risk_score(risk_metrics, weights)

            # Check concentration risk
            concentration_hhi = risk_metrics.concentration_hhi
            max_position = max(abs(w) for w in weights.values()) if weights else 0

            # Position limit recommendations
            position_limits = self._calculate_position_limits(weights, risk_metrics)

            # Risk assessment
            approved = True

            # Check VaR limit
            if risk_metrics.portfolio_var_95 > self.risk_constraints.max_portfolio_var_95:
                approved = False
                warnings.append(f"Portfolio VaR ({risk_metrics.portfolio_var_95:.3f}) exceeds limit ({self.risk_constraints.max_portfolio_var_95:.3f})")
                recommendations.append("Reduce position sizes or diversify holdings")

            # Check concentration risk
            if max_position > self.risk_constraints.max_position_concentration:
                approved = False
                warnings.append(f"Maximum position weight ({max_position:.3f}) exceeds limit ({self.risk_constraints.max_position_concentration:.3f})")
                recommendations.append("Reduce concentrated positions")

            # Check diversification
            active_positions = len([w for w in weights.values() if abs(w) > 0.01])
            if active_positions < self.risk_constraints.min_diversification_count:
                warnings.append(f"Low diversification: only {active_positions} significant positions")
                recommendations.append("Increase portfolio diversification")

            # Check correlation risk
            if risk_metrics.max_correlation > self.risk_constraints.max_correlation_threshold:
                warnings.append(f"High correlation risk ({risk_metrics.max_correlation:.3f})")
                recommendations.append("Consider less correlated assets")

            return RiskAssessment(
                approved=approved,
                risk_score=risk_score,
                var_95=risk_metrics.portfolio_var_95,
                concentration_risk=concentration_hhi,
                warnings=warnings,
                recommendations=recommendations,
                position_limits=position_limits
            )

        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskAssessment(
                approved=False,
                risk_score=100.0,
                var_95=0.0,
                concentration_risk=0.0,
                warnings=[f"Risk assessment failed: {e}"],
                recommendations=["Manual review required"],
                position_limits={}
            )

    def optimize_position_sizes(self,
                              target_positions: Dict[str, float],
                              max_portfolio_value: float) -> Dict[str, float]:
        """
        Optimize position sizes based on risk constraints

        Args:
            target_positions: Target positions {symbol: value}
            max_portfolio_value: Maximum total portfolio value

        Returns:
            Risk-adjusted position sizes
        """
        try:
            if not target_positions:
                return {}

            # Convert to weights
            total_target = sum(abs(v) for v in target_positions.values())
            if total_target == 0:
                return target_positions

            weights = {symbol: value / total_target for symbol, value in target_positions.items()}

            # Assess current risk
            risk_assessment = self.assess_portfolio_risk({}, target_positions)

            # If already compliant, return as-is
            if risk_assessment.approved:
                return target_positions

            # Apply risk-based scaling
            optimized_weights = self._apply_risk_scaling(weights, risk_assessment)

            # Convert back to absolute values
            optimized_positions = {}
            for symbol, weight in optimized_weights.items():
                optimized_positions[symbol] = weight * max_portfolio_value

            # Verify optimization worked
            final_assessment = self.assess_portfolio_risk({}, optimized_positions)
            if not final_assessment.approved:
                logger.warning("Risk optimization could not achieve full compliance")

            return optimized_positions

        except Exception as e:
            logger.error(f"Error optimizing position sizes: {e}")
            return target_positions  # Return original if optimization fails

    def analyze_trading_opportunities(self,
                                    current_positions: Dict[str, Any],
                                    recommended_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trading opportunities with risk awareness

        This method wraps the original analyze_trading_opportunities with risk checks
        """
        try:
            # Convert current positions to value format
            current_values = {}
            for pos in current_positions:
                symbol = pos.get('symbol')
                value = pos.get('value', 0.0)
                if symbol:
                    current_values[symbol] = value

            # Convert recommended positions to value format
            recommended_values = {}
            for pos in recommended_positions:
                symbol = pos.get('symbol')
                value = pos.get('value', 0.0)
                if symbol:
                    recommended_values[symbol] = value

            # Assess risk of recommended portfolio
            risk_assessment = self.assess_portfolio_risk(current_values, recommended_values)

            # If risk monitoring is disabled, proceed with original logic
            if not self.enable_risk_monitoring:
                if self.auto_trading_engine:
                    return self.auto_trading_engine.analyze_trading_opportunities(
                        current_positions, recommended_positions
                    )
                else:
                    return self._default_trading_analysis(current_positions, recommended_positions)

            # Risk-adjusted analysis
            trading_signals = {
                'buy': [],
                'sell': [],
                'hold': []
            }

            # Apply risk-based filtering
            if risk_assessment.approved:
                # Portfolio passes risk checks - proceed with recommendations
                for pos in recommended_positions:
                    symbol = pos.get('symbol', '')
                    action = pos.get('action', 'hold')

                    # Check individual position limits
                    if symbol in risk_assessment.position_limits:
                        max_allowed = risk_assessment.position_limits[symbol]
                        current_weight = abs(pos.get('value', 0.0)) / sum(abs(p.get('value', 0.0)) for p in recommended_positions)

                        if current_weight > max_allowed:
                            # Reduce position size
                            scale_factor = max_allowed / current_weight
                            pos = pos.copy()
                            pos['qty'] = int(pos.get('qty', 0) * scale_factor)
                            pos['value'] = pos.get('value', 0.0) * scale_factor
                            action = 'reduced_buy' if action == 'buy' else action

                    # Categorize signals
                    if action in ['buy', 'strong_buy', 'reduced_buy']:
                        trading_signals['buy'].append(pos)
                    elif action in ['sell', 'strong_sell']:
                        trading_signals['sell'].append(pos)
                    else:
                        trading_signals['hold'].append(pos)

            else:
                # Portfolio fails risk checks - apply conservative approach
                logger.warning("Portfolio fails risk assessment - applying conservative approach")

                # Keep current positions, avoid new buys
                for pos in current_positions:
                    trading_signals['hold'].append(pos)

            # Add risk metadata
            result = {
                'trading_signals': trading_signals,
                'risk_assessment': {
                    'approved': risk_assessment.approved,
                    'risk_score': risk_assessment.risk_score,
                    'var_95_percent': risk_assessment.var_95 * 100,
                    'concentration_risk': risk_assessment.concentration_risk,
                    'warnings': risk_assessment.warnings,
                    'recommendations': risk_assessment.recommendations
                },
                'execution_results': [],
                'trading_summary': {
                    'risk_adjusted': True,
                    'risk_compliance': risk_assessment.approved,
                    'total_risk_score': risk_assessment.risk_score
                }
            }

            return result

        except Exception as e:
            logger.error(f"Error in risk-aware trading analysis: {e}")
            # Fallback to original method if available
            if self.auto_trading_engine:
                return self.auto_trading_engine.analyze_trading_opportunities(
                    current_positions, recommended_positions
                )
            else:
                return self._default_trading_analysis(current_positions, recommended_positions)

    def _get_cached_risk_metrics(self, weights: Dict[str, float]) -> RiskMetrics:
        """Get cached risk metrics or calculate new ones"""
        current_time = datetime.now()

        # Check cache validity (5 minute expiry)
        if (self._risk_metrics_cache is not None and
            self._cache_expiry is not None and
            current_time < self._cache_expiry):
            return self._risk_metrics_cache

        # Calculate new metrics
        risk_metrics = self.risk_manager.generate_risk_report(weights, self.risk_budget)

        # Update cache
        self._risk_metrics_cache = risk_metrics
        self._cache_expiry = current_time + timedelta(minutes=5)

        return risk_metrics

    def _calculate_risk_score(self, risk_metrics: RiskMetrics, weights: Dict[str, float]) -> float:
        """Calculate composite risk score (0-100 scale)"""
        try:
            score = 0.0

            # VaR component (0-40 points)
            var_ratio = risk_metrics.portfolio_var_95 / self.risk_constraints.max_portfolio_var_95
            var_score = min(40.0, var_ratio * 40.0)
            score += var_score

            # Concentration component (0-30 points)
            max_weight = max(abs(w) for w in weights.values()) if weights else 0
            concentration_ratio = max_weight / self.risk_constraints.max_position_concentration
            concentration_score = min(30.0, concentration_ratio * 30.0)
            score += concentration_score

            # Diversification component (0-20 points)
            active_positions = len([w for w in weights.values() if abs(w) > 0.01])
            if active_positions < self.risk_constraints.min_diversification_count:
                diversification_score = 20.0 * (1 - active_positions / self.risk_constraints.min_diversification_count)
            else:
                diversification_score = 0.0
            score += diversification_score

            # Correlation component (0-10 points)
            if risk_metrics.max_correlation > self.risk_constraints.max_correlation_threshold:
                correlation_score = 10.0 * (risk_metrics.max_correlation - self.risk_constraints.max_correlation_threshold) / (1.0 - self.risk_constraints.max_correlation_threshold)
            else:
                correlation_score = 0.0
            score += correlation_score

            return min(100.0, score)

        except Exception as e:
            logger.warning(f"Error calculating risk score: {e}")
            return 50.0  # Default moderate risk score

    def _calculate_position_limits(self, weights: Dict[str, float], risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Calculate individual position limits based on risk contribution"""
        try:
            position_limits = {}

            # Base limit from risk constraints
            base_limit = self.risk_constraints.max_position_concentration

            # Get position risk contributions
            position_risks = self.risk_manager.calculate_position_risk_contributions(weights)

            for pos_risk in position_risks:
                symbol = pos_risk.symbol

                # Adjust limit based on volatility and correlation
                volatility_adjustment = max(0.5, 1.0 - pos_risk.volatility)
                correlation_adjustment = max(0.5, 1.0 - abs(pos_risk.correlation_with_market))

                adjusted_limit = base_limit * volatility_adjustment * correlation_adjustment
                position_limits[symbol] = min(base_limit, adjusted_limit)

            return position_limits

        except Exception as e:
            logger.warning(f"Error calculating position limits: {e}")
            # Return default limits for all symbols
            return {symbol: self.risk_constraints.max_position_concentration for symbol in weights.keys()}

    def _apply_risk_scaling(self, weights: Dict[str, float], risk_assessment: RiskAssessment) -> Dict[str, float]:
        """Apply risk-based scaling to position weights"""
        try:
            if risk_assessment.approved:
                return weights

            # Calculate scaling factor based on risk score
            if risk_assessment.risk_score <= 50:
                scale_factor = 1.0
            elif risk_assessment.risk_score <= 75:
                scale_factor = 0.8
            else:
                scale_factor = 0.6

            # Apply uniform scaling
            scaled_weights = {symbol: weight * scale_factor for symbol, weight in weights.items()}

            # Apply individual position limits
            for symbol, limit in risk_assessment.position_limits.items():
                if symbol in scaled_weights and abs(scaled_weights[symbol]) > limit:
                    scaled_weights[symbol] = limit * (1 if scaled_weights[symbol] > 0 else -1)

            return scaled_weights

        except Exception as e:
            logger.warning(f"Error applying risk scaling: {e}")
            return weights

    def _default_trading_analysis(self, current_positions: List[Dict], recommended_positions: List[Dict]) -> Dict[str, Any]:
        """Default trading analysis when no auto trading engine is available"""
        return {
            'trading_signals': {
                'buy': [pos for pos in recommended_positions if pos.get('action') in ['buy', 'strong_buy']],
                'sell': [],
                'hold': current_positions
            },
            'execution_results': [],
            'trading_summary': {
                'risk_adjusted': True,
                'total_trades': 0
            }
        }


def create_risk_aware_trading_engine(auto_trading_engine=None,
                                   dry_run: bool = True,
                                   enable_risk_monitoring: bool = True,
                                   risk_constraints: Optional[RiskConstraints] = None) -> RiskAwareTradingAdapter:
    """
    Factory function to create risk-aware trading engine

    Args:
        auto_trading_engine: Optional existing AutoTradingEngine
        dry_run: Whether to run in dry run mode
        enable_risk_monitoring: Enable risk monitoring features
        risk_constraints: Custom risk constraints

    Returns:
        Configured RiskAwareTradingAdapter
    """
    return RiskAwareTradingAdapter(
        auto_trading_engine=auto_trading_engine,
        dry_run=dry_run,
        risk_constraints=risk_constraints,
        enable_risk_monitoring=enable_risk_monitoring
    )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create risk-aware adapter
    adapter = create_risk_aware_trading_engine(dry_run=True)

    # Example market data
    market_data = {
        'AAPL': {'price': 150.0},
        'GOOGL': {'price': 2500.0},
        'TSLA': {'price': 800.0},
        'MSFT': {'price': 300.0}
    }

    # Update market data
    adapter.update_market_data(market_data)

    # Example portfolio
    current_positions = []
    recommended_positions = [
        {'symbol': 'AAPL', 'qty': 100, 'price': 150.0, 'value': 15000.0, 'action': 'buy'},
        {'symbol': 'GOOGL', 'qty': 10, 'price': 2500.0, 'value': 25000.0, 'action': 'buy'},
        {'symbol': 'TSLA', 'qty': 50, 'price': 800.0, 'value': 40000.0, 'action': 'buy'}
    ]

    # Assess risk
    total_value = sum(pos['value'] for pos in recommended_positions)
    proposed_positions = {pos['symbol']: pos['value'] for pos in recommended_positions}

    risk_assessment = adapter.assess_portfolio_risk({}, proposed_positions)

    print(f"Risk Assessment:")
    print(f"  Approved: {risk_assessment.approved}")
    print(f"  Risk Score: {risk_assessment.risk_score:.1f}/100")
    print(f"  VaR 95%: {risk_assessment.var_95:.3f}")
    print(f"  Warnings: {risk_assessment.warnings}")
    print(f"  Recommendations: {risk_assessment.recommendations}")

    # Analyze trading opportunities
    analysis = adapter.analyze_trading_opportunities(current_positions, recommended_positions)
    print(f"\nTrading Analysis:")
    print(f"  Buy signals: {len(analysis['trading_signals']['buy'])}")
    print(f"  Risk compliance: {analysis['risk_assessment']['approved']}")
    print(f"  Risk score: {analysis['risk_assessment']['risk_score']:.1f}")