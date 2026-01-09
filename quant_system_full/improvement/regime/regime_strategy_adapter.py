"""
Regime Strategy Adapter

This module adapts trading strategy parameters based on detected market regime.
Different market conditions require different risk management and position sizing approaches.

Key Features:
- Dynamic stop-loss adjustment based on market regime
- Position sizing adaptation (target allocation per stock)
- Portfolio size management (number of holdings)
- Factor weight optimization for different market conditions
- Risk parameter scaling
- Regime-aware execution adjustments
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from .market_regime_detector import MarketRegime, RegimeAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegimeStrategyConfig:
    """Strategy configuration for different market regimes."""

    # Stop-loss levels (negative percentages)
    bull_stop_loss: float = -0.07  # -7%
    bear_stop_loss: float = -0.03  # -3%
    sideways_stop_loss: float = -0.05  # -5%

    # Target position sizes (as fraction of portfolio)
    bull_position_size: float = 0.06  # 6%
    bear_position_size: float = 0.03  # 3%
    sideways_position_size: float = 0.04  # 4%

    # Number of holdings
    bull_max_positions: int = 18
    bear_max_positions: int = 12
    sideways_max_positions: int = 15

    # Factor weights (momentum, value, quality, etc.)
    bull_factor_weights: Dict[str, float] = None
    bear_factor_weights: Dict[str, float] = None
    sideways_factor_weights: Dict[str, float] = None

    # Risk scaling factors
    bull_risk_multiplier: float = 1.2  # More aggressive
    bear_risk_multiplier: float = 0.6  # More conservative
    sideways_risk_multiplier: float = 1.0  # Neutral

    # Execution parameters
    bull_urgency_factor: float = 1.1  # Slightly more urgent
    bear_urgency_factor: float = 0.8  # More patient
    sideways_urgency_factor: float = 1.0  # Standard

    def __post_init__(self):
        if self.bull_factor_weights is None:
            self.bull_factor_weights = {
                'momentum': 0.35,
                'quality': 0.25,
                'value': 0.15,
                'growth': 0.25
            }

        if self.bear_factor_weights is None:
            self.bear_factor_weights = {
                'value': 0.40,
                'quality': 0.35,
                'momentum': 0.10,
                'growth': 0.15
            }

        if self.sideways_factor_weights is None:
            self.sideways_factor_weights = {
                'momentum': 0.25,
                'value': 0.25,
                'quality': 0.25,
                'growth': 0.25
            }


@dataclass
class AdaptedStrategyParams:
    """Adapted strategy parameters for current market regime."""
    regime: MarketRegime
    confidence: float

    # Core parameters
    stop_loss_level: float
    position_size: float
    max_positions: int

    # Factor weights
    factor_weights: Dict[str, float]

    # Risk parameters
    risk_multiplier: float
    urgency_factor: float

    # Derived parameters
    portfolio_turnover_limit: float
    sector_concentration_limit: float
    volatility_adjustment: float

    # Metadata
    regime_strength: float
    adaptation_timestamp: str = ""

    def __post_init__(self):
        if not self.adaptation_timestamp:
            self.adaptation_timestamp = datetime.now().isoformat()


class RegimeStrategyAdapter:
    """
    Adapts trading strategy parameters based on market regime detection.
    """

    def __init__(self, config: Optional[RegimeStrategyConfig] = None):
        """
        Initialize strategy adapter.

        Args:
            config: Configuration for regime-specific parameters
        """
        self.config = config or RegimeStrategyConfig()

        # Parameter history
        self.adaptation_history: List[Dict] = []
        self.current_params: Optional[AdaptedStrategyParams] = None

        # Adaptation settings
        self.min_confidence_for_adaptation = 0.6
        self.smoothing_factor = 0.3  # For gradual transitions

        logger.info("[regime_adapter] Strategy adapter initialized")

    def adapt_strategy(self, regime_analysis: RegimeAnalysis,
                      market_volatility: Optional[float] = None,
                      current_portfolio_params: Optional[Dict] = None) -> AdaptedStrategyParams:
        """
        Adapt strategy parameters based on regime analysis.

        Args:
            regime_analysis: Current regime analysis
            market_volatility: Current market volatility (for adjustments)
            current_portfolio_params: Current portfolio state for smooth transitions

        Returns:
            Adapted strategy parameters
        """
        try:
            regime = regime_analysis.current_regime
            confidence = regime_analysis.confidence
            strength = regime_analysis.regime_strength

            logger.info(f"[regime_adapter] Adapting strategy for {regime.value} regime (confidence: {confidence:.2f})")

            # Get base parameters for regime
            base_params = self._get_base_parameters(regime)

            # Apply confidence-based adjustments
            adjusted_params = self._apply_confidence_adjustments(base_params, confidence, strength)

            # Apply volatility adjustments if available
            if market_volatility is not None:
                adjusted_params = self._apply_volatility_adjustments(adjusted_params, market_volatility)

            # Apply smooth transitions if we have previous parameters
            if (self.current_params is not None and
                current_portfolio_params is not None and
                confidence > self.min_confidence_for_adaptation):
                adjusted_params = self._apply_smooth_transition(adjusted_params, current_portfolio_params)

            # Create final adapted parameters
            adapted_params = AdaptedStrategyParams(
                regime=regime,
                confidence=confidence,
                stop_loss_level=adjusted_params['stop_loss'],
                position_size=adjusted_params['position_size'],
                max_positions=adjusted_params['max_positions'],
                factor_weights=adjusted_params['factor_weights'],
                risk_multiplier=adjusted_params['risk_multiplier'],
                urgency_factor=adjusted_params['urgency_factor'],
                portfolio_turnover_limit=adjusted_params['turnover_limit'],
                sector_concentration_limit=adjusted_params['sector_limit'],
                volatility_adjustment=adjusted_params.get('volatility_adjustment', 1.0),
                regime_strength=strength
            )

            # Update state
            self._update_adaptation_history(adapted_params)
            self.current_params = adapted_params

            logger.info(f"[regime_adapter] Strategy adapted: stop_loss={adapted_params.stop_loss_level:.1%}, "
                       f"position_size={adapted_params.position_size:.1%}, max_positions={adapted_params.max_positions}")

            return adapted_params

        except Exception as e:
            logger.error(f"[regime_adapter] Strategy adaptation failed: {e}")
            # Return conservative default parameters
            return self._get_default_parameters()

    def _get_base_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get base parameters for specific regime."""
        try:
            if regime == MarketRegime.BULL:
                return {
                    'stop_loss': self.config.bull_stop_loss,
                    'position_size': self.config.bull_position_size,
                    'max_positions': self.config.bull_max_positions,
                    'factor_weights': self.config.bull_factor_weights.copy(),
                    'risk_multiplier': self.config.bull_risk_multiplier,
                    'urgency_factor': self.config.bull_urgency_factor,
                    'turnover_limit': 0.30,  # Higher turnover OK in bull markets
                    'sector_limit': 0.25     # Standard sector limits
                }

            elif regime == MarketRegime.BEAR:
                return {
                    'stop_loss': self.config.bear_stop_loss,
                    'position_size': self.config.bear_position_size,
                    'max_positions': self.config.bear_max_positions,
                    'factor_weights': self.config.bear_factor_weights.copy(),
                    'risk_multiplier': self.config.bear_risk_multiplier,
                    'urgency_factor': self.config.bear_urgency_factor,
                    'turnover_limit': 0.15,  # Lower turnover in bear markets
                    'sector_limit': 0.20     # Tighter sector limits
                }

            elif regime == MarketRegime.SIDEWAYS:
                return {
                    'stop_loss': self.config.sideways_stop_loss,
                    'position_size': self.config.sideways_position_size,
                    'max_positions': self.config.sideways_max_positions,
                    'factor_weights': self.config.sideways_factor_weights.copy(),
                    'risk_multiplier': self.config.sideways_risk_multiplier,
                    'urgency_factor': self.config.sideways_urgency_factor,
                    'turnover_limit': 0.20,  # Moderate turnover
                    'sector_limit': 0.25     # Standard sector limits
                }

            else:  # UNKNOWN
                return self._get_conservative_parameters()

        except Exception as e:
            logger.error(f"[regime_adapter] Failed to get base parameters: {e}")
            return self._get_conservative_parameters()

    def _get_conservative_parameters(self) -> Dict[str, Any]:
        """Get conservative default parameters for unknown regime."""
        return {
            'stop_loss': -0.04,  # -4%
            'position_size': 0.03,  # 3%
            'max_positions': 12,
            'factor_weights': {
                'value': 0.35,
                'quality': 0.35,
                'momentum': 0.15,
                'growth': 0.15
            },
            'risk_multiplier': 0.8,
            'urgency_factor': 0.9,
            'turnover_limit': 0.10,
            'sector_limit': 0.15
        }

    def _apply_confidence_adjustments(self, base_params: Dict, confidence: float, strength: float) -> Dict:
        """Apply adjustments based on regime confidence and strength."""
        try:
            adjusted = base_params.copy()

            # If confidence is low, move towards more conservative parameters
            if confidence < self.min_confidence_for_adaptation:
                conservative = self._get_conservative_parameters()

                # Blend towards conservative based on how low confidence is
                blend_factor = 1 - (confidence / self.min_confidence_for_adaptation)

                adjusted['stop_loss'] = (base_params['stop_loss'] * (1 - blend_factor) +
                                       conservative['stop_loss'] * blend_factor)
                adjusted['position_size'] = (base_params['position_size'] * (1 - blend_factor) +
                                           conservative['position_size'] * blend_factor)
                adjusted['risk_multiplier'] = (base_params['risk_multiplier'] * (1 - blend_factor) +
                                             conservative['risk_multiplier'] * blend_factor)

            # Adjust based on regime strength
            strength_factor = min(1.2, 0.8 + (strength * 0.4))  # Scale between 0.8 and 1.2

            # Apply strength adjustments (stronger regime = more aggressive)
            if base_params['risk_multiplier'] > 1.0:  # Bull market parameters
                adjusted['risk_multiplier'] *= strength_factor
                adjusted['position_size'] *= strength_factor
            else:  # Bear/sideways parameters
                adjusted['risk_multiplier'] *= (2 - strength_factor)  # Inverse relationship
                adjusted['stop_loss'] *= (2 - strength_factor)  # Less aggressive stop in weak bear

            return adjusted

        except Exception as e:
            logger.error(f"[regime_adapter] Confidence adjustment failed: {e}")
            return base_params

    def _apply_volatility_adjustments(self, params: Dict, volatility: float) -> Dict:
        """Apply adjustments based on market volatility."""
        try:
            adjusted = params.copy()

            # Normalize volatility (assume typical range 0.1 to 0.5)
            vol_factor = np.clip(volatility / 0.2, 0.5, 2.5)

            # High volatility = tighter stops, smaller positions
            adjusted['stop_loss'] *= (1 / vol_factor)  # Tighter stops in high vol
            adjusted['position_size'] *= (1 / vol_factor)  # Smaller positions in high vol
            adjusted['turnover_limit'] *= (1 / vol_factor)  # Less turnover in high vol

            # Store volatility adjustment for reference
            adjusted['volatility_adjustment'] = vol_factor

            logger.debug(f"[regime_adapter] Volatility adjustment applied: factor={vol_factor:.2f}")

            return adjusted

        except Exception as e:
            logger.error(f"[regime_adapter] Volatility adjustment failed: {e}")
            return params

    def _apply_smooth_transition(self, new_params: Dict, portfolio_state: Dict) -> Dict:
        """Apply smooth transition to avoid dramatic parameter changes."""
        try:
            if self.current_params is None:
                return new_params

            smoothed = new_params.copy()

            # Get current parameters for comparison
            current = {
                'stop_loss': self.current_params.stop_loss_level,
                'position_size': self.current_params.position_size,
                'risk_multiplier': self.current_params.risk_multiplier
            }

            # Apply smoothing to key parameters
            for param in ['stop_loss', 'position_size', 'risk_multiplier']:
                if param in new_params and param in current:
                    # Calculate smoothed value
                    smoothed[param] = (current[param] * self.smoothing_factor +
                                     new_params[param] * (1 - self.smoothing_factor))

            # Portfolio-specific adjustments
            if 'current_positions' in portfolio_state:
                current_positions = portfolio_state['current_positions']
                target_positions = new_params['max_positions']

                # If we need to reduce positions significantly, do it gradually
                if current_positions > target_positions * 1.2:
                    # Gradual reduction
                    smoothed['max_positions'] = max(target_positions,
                                                  int(current_positions * 0.9))
                else:
                    smoothed['max_positions'] = new_params['max_positions']

            logger.debug(f"[regime_adapter] Smooth transition applied")

            return smoothed

        except Exception as e:
            logger.error(f"[regime_adapter] Smooth transition failed: {e}")
            return new_params

    def _update_adaptation_history(self, adapted_params: AdaptedStrategyParams):
        """Update adaptation history."""
        try:
            history_entry = {
                'regime': adapted_params.regime.value,
                'confidence': adapted_params.confidence,
                'stop_loss': adapted_params.stop_loss_level,
                'position_size': adapted_params.position_size,
                'max_positions': adapted_params.max_positions,
                'risk_multiplier': adapted_params.risk_multiplier,
                'timestamp': adapted_params.adaptation_timestamp
            }

            self.adaptation_history.append(history_entry)

            # Keep only recent history (last 30 adaptations)
            if len(self.adaptation_history) > 30:
                self.adaptation_history = self.adaptation_history[-30:]

        except Exception as e:
            logger.error(f"[regime_adapter] Failed to update adaptation history: {e}")

    def _get_default_parameters(self) -> AdaptedStrategyParams:
        """Get default conservative parameters."""
        conservative = self._get_conservative_parameters()

        return AdaptedStrategyParams(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            stop_loss_level=conservative['stop_loss'],
            position_size=conservative['position_size'],
            max_positions=conservative['max_positions'],
            factor_weights=conservative['factor_weights'],
            risk_multiplier=conservative['risk_multiplier'],
            urgency_factor=conservative['urgency_factor'],
            portfolio_turnover_limit=conservative['turnover_limit'],
            sector_concentration_limit=conservative['sector_limit'],
            volatility_adjustment=1.0,
            regime_strength=0.0
        )

    def get_parameter_impact_analysis(self,
                                    baseline_params: Dict,
                                    adapted_params: AdaptedStrategyParams) -> Dict[str, Any]:
        """Analyze the impact of parameter changes."""
        try:
            analysis = {
                'regime_change': {
                    'from': baseline_params.get('regime', 'unknown'),
                    'to': adapted_params.regime.value
                },
                'parameter_changes': {},
                'risk_impact': {},
                'expected_portfolio_changes': {}
            }

            # Parameter changes
            if 'stop_loss' in baseline_params:
                stop_loss_change = adapted_params.stop_loss_level - baseline_params['stop_loss']
                analysis['parameter_changes']['stop_loss'] = {
                    'absolute_change': stop_loss_change,
                    'relative_change': stop_loss_change / abs(baseline_params['stop_loss']),
                    'interpretation': 'tighter' if stop_loss_change > 0 else 'looser'
                }

            if 'position_size' in baseline_params:
                size_change = adapted_params.position_size - baseline_params['position_size']
                analysis['parameter_changes']['position_size'] = {
                    'absolute_change': size_change,
                    'relative_change': size_change / baseline_params['position_size'],
                    'interpretation': 'larger' if size_change > 0 else 'smaller'
                }

            # Risk impact assessment
            analysis['risk_impact'] = {
                'overall_risk_change': adapted_params.risk_multiplier - baseline_params.get('risk_multiplier', 1.0),
                'portfolio_concentration': 1.0 / adapted_params.max_positions,
                'stop_loss_protection': abs(adapted_params.stop_loss_level),
                'risk_level': self._assess_risk_level(adapted_params)
            }

            # Expected portfolio changes
            analysis['expected_portfolio_changes'] = {
                'position_count_target': adapted_params.max_positions,
                'individual_position_target': adapted_params.position_size,
                'turnover_limit': adapted_params.portfolio_turnover_limit,
                'factor_emphasis': max(adapted_params.factor_weights, key=adapted_params.factor_weights.get)
            }

            return analysis

        except Exception as e:
            logger.error(f"[regime_adapter] Impact analysis failed: {e}")
            return {'error': str(e)}

    def _assess_risk_level(self, params: AdaptedStrategyParams) -> str:
        """Assess overall risk level of adapted parameters."""
        try:
            # Calculate risk score based on multiple factors
            risk_score = 0

            # Position size contribution (higher = more risky)
            risk_score += (params.position_size / 0.05) * 25  # Normalize to 5% base

            # Stop loss contribution (tighter = less risky)
            risk_score += (1 / abs(params.stop_loss_level) * 0.05) * 25  # Normalize to 5% base

            # Portfolio concentration (fewer positions = more risky)
            risk_score += (20 / params.max_positions) * 25  # Normalize to 20 positions base

            # Risk multiplier contribution
            risk_score += (params.risk_multiplier - 1.0) * 25

            # Classify risk level
            if risk_score < 75:
                return 'conservative'
            elif risk_score < 125:
                return 'moderate'
            elif risk_score < 175:
                return 'aggressive'
            else:
                return 'very_aggressive'

        except Exception as e:
            logger.error(f"[regime_adapter] Risk assessment failed: {e}")
            return 'unknown'

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of current adaptation state."""
        try:
            summary = {
                'current_parameters': asdict(self.current_params) if self.current_params else None,
                'adaptation_history_length': len(self.adaptation_history),
                'recent_adaptations': self.adaptation_history[-5:] if len(self.adaptation_history) >= 5 else self.adaptation_history,
                'configuration': {
                    'min_confidence_threshold': self.min_confidence_for_adaptation,
                    'smoothing_factor': self.smoothing_factor
                },
                'timestamp': datetime.now().isoformat()
            }

            # Add parameter trends if we have history
            if len(self.adaptation_history) >= 2:
                recent = self.adaptation_history[-5:]
                summary['parameter_trends'] = {
                    'stop_loss_trend': self._calculate_trend([h['stop_loss'] for h in recent]),
                    'position_size_trend': self._calculate_trend([h['position_size'] for h in recent]),
                    'risk_multiplier_trend': self._calculate_trend([h['risk_multiplier'] for h in recent])
                }

            return summary

        except Exception as e:
            logger.error(f"[regime_adapter] Failed to get adaptation summary: {e}")
            return {'error': str(e)}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        try:
            if len(values) < 2:
                return 'stable'

            slope = (values[-1] - values[0]) / len(values)

            if slope > 0.001:
                return 'increasing'
            elif slope < -0.001:
                return 'decreasing'
            else:
                return 'stable'

        except Exception as e:
            return 'unknown'


def create_regime_strategy_adapter(custom_config: Optional[Dict] = None) -> RegimeStrategyAdapter:
    """
    Create and configure a regime strategy adapter.

    Args:
        custom_config: Custom configuration parameters

    Returns:
        Configured RegimeStrategyAdapter instance
    """
    config = RegimeStrategyConfig()

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return RegimeStrategyAdapter(config)


if __name__ == "__main__":
    # Test strategy adaptation
    print("=== Regime Strategy Adapter Test ===")

    from .market_regime_detector import RegimeAnalysis, MarketRegime

    # Create sample regime analysis
    sample_analysis = RegimeAnalysis(
        current_regime=MarketRegime.BULL,
        confidence=0.8,
        indicators=[],
        consensus_score={'bull': 0.7, 'bear': 0.1, 'sideways': 0.2},
        days_in_regime=5,
        last_regime_change=None,
        regime_strength=0.75
    )

    # Create adapter and adapt strategy
    adapter = create_regime_strategy_adapter()
    adapted_params = adapter.adapt_strategy(sample_analysis, market_volatility=0.15)

    print(f"Adapted Strategy for {adapted_params.regime.value.upper()} Market:")
    print(f"  Stop Loss: {adapted_params.stop_loss_level:.1%}")
    print(f"  Position Size: {adapted_params.position_size:.1%}")
    print(f"  Max Positions: {adapted_params.max_positions}")
    print(f"  Risk Multiplier: {adapted_params.risk_multiplier:.2f}")
    print(f"  Factor Weights: {adapted_params.factor_weights}")

    # Test impact analysis
    baseline = {'regime': 'sideways', 'stop_loss': -0.05, 'position_size': 0.04, 'risk_multiplier': 1.0}
    impact = adapter.get_parameter_impact_analysis(baseline, adapted_params)
    print(f"\nImpact Analysis:")
    print(f"  Risk Level: {impact['risk_impact']['risk_level']}")
    print(f"  Position Size Change: {impact['parameter_changes']['position_size']['interpretation']}")
    print(f"  Stop Loss Change: {impact['parameter_changes']['stop_loss']['interpretation']}")