#!/usr/bin/env python3
"""
Market Regime and Risk Management Integration Module

This module provides seamless integration between the sophisticated Market Regime
Classification System and the existing Enhanced Risk Manager. It enables:

- Automatic regime detection feeding into risk limit adjustments
- Dynamic risk parameter updates based on regime transitions
- Enhanced portfolio risk assessment with regime-aware metrics
- Backtesting validation with regime-specific performance analysis
- Real-time monitoring and alerting integration

Key Features:
- Regime-aware risk limit scaling
- Automatic model retraining and updates
- Integration with existing portfolio management
- Enhanced monitoring and alert systems
- Performance attribution by market regime
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Import existing system components
try:
    from enhanced_risk_manager import (
        EnhancedRiskManager,
        RiskLimits,
        MarketRegime as ExistingMarketRegime,
        TailRiskMetrics,
        RiskAlert,
        RiskLevel
    )
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

try:
    from market_regime_classifier import (
        MarketRegimeClassifier,
        MarketRegime,
        RegimePrediction,
        RegimeTransition
    )
    REGIME_CLASSIFIER_AVAILABLE = True
except ImportError:
    REGIME_CLASSIFIER_AVAILABLE = False

try:
    from config import SETTINGS
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RegimeRiskConfig:
    """Configuration for regime-risk integration"""
    # Model update intervals
    regime_update_interval: int = 300  # 5 minutes
    model_retrain_interval: int = 86400  # 24 hours

    # Risk limit multipliers by regime
    normal_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'max_portfolio_var': 1.0,
        'max_single_position': 1.0,
        'es_97_5_limit': 1.0,
        'daily_loss_limit': 1.0
    })

    volatile_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'max_portfolio_var': 0.7,
        'max_single_position': 0.8,
        'es_97_5_limit': 1.5,
        'daily_loss_limit': 0.8
    })

    crisis_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'max_portfolio_var': 0.5,
        'max_single_position': 0.6,
        'es_97_5_limit': 2.0,
        'daily_loss_limit': 0.6
    })

    # Confidence thresholds for regime changes
    regime_change_threshold: float = 0.7
    low_confidence_threshold: float = 0.5

    # Performance tracking
    enable_regime_attribution: bool = True
    attribution_lookback_days: int = 252

    # Monitoring settings
    enable_real_time_monitoring: bool = True
    alert_on_regime_change: bool = True
    alert_on_low_confidence: bool = True


@dataclass
class RegimeRiskState:
    """Current state of regime-risk integration"""
    current_regime: MarketRegime
    regime_confidence: float
    last_regime_update: datetime
    last_model_retrain: datetime
    risk_limits: RiskLimits
    active_alerts: List[RiskAlert]
    performance_attribution: Dict[str, Any]
    monitoring_enabled: bool = True


class RegimeRiskIntegration:
    """
    Integrated Market Regime and Risk Management System

    This class provides seamless integration between regime detection and
    risk management, enabling dynamic risk controls based on market conditions.
    """

    def __init__(self,
                 config: Optional[RegimeRiskConfig] = None,
                 base_risk_limits: Optional[RiskLimits] = None,
                 regime_classifier: Optional[MarketRegimeClassifier] = None,
                 risk_manager: Optional[EnhancedRiskManager] = None):
        """
        Initialize integrated regime-risk system

        Args:
            config: Configuration for integration
            base_risk_limits: Base risk limits to scale
            regime_classifier: Regime classification system
            risk_manager: Enhanced risk manager
        """
        if not RISK_MANAGER_AVAILABLE or not REGIME_CLASSIFIER_AVAILABLE:
            raise ImportError("Required components not available")

        self.config = config or RegimeRiskConfig()
        self.base_risk_limits = base_risk_limits or RiskLimits()
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()
        self.risk_manager = risk_manager or EnhancedRiskManager()

        # Initialize state
        self.state = RegimeRiskState(
            current_regime=MarketRegime.NORMAL,
            regime_confidence=0.0,
            last_regime_update=datetime.min,
            last_model_retrain=datetime.min,
            risk_limits=self.base_risk_limits,
            active_alerts=[],
            performance_attribution={}
        )

        # Monitoring components
        self.monitoring_thread = None
        self.monitoring_active = False
        self.performance_tracker = RegimePerformanceTracker()

        # Integration mappings
        self.regime_mapping = {
            MarketRegime.NORMAL: ExistingMarketRegime.NORMAL,
            MarketRegime.VOLATILE: ExistingMarketRegime.VOLATILE,
            MarketRegime.CRISIS: ExistingMarketRegime.CRISIS
        }

        logger.info("Regime-Risk Integration System initialized")

    def start_monitoring(self) -> bool:
        """
        Start real-time regime monitoring and risk adjustment

        Returns:
            True if monitoring started successfully
        """
        if not self.config.enable_real_time_monitoring:
            logger.info("Real-time monitoring disabled in config")
            return False

        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return False

        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

            logger.info("Real-time regime monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
            return False

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            logger.info("Real-time regime monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Regime monitoring loop started")

        while self.monitoring_active:
            try:
                # Update regime detection
                self.update_regime_assessment()

                # Check if model retraining is needed
                if self._should_retrain_models():
                    self._retrain_models_async()

                # Sleep until next update
                time.sleep(self.config.regime_update_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

        logger.info("Regime monitoring loop stopped")

    def update_regime_assessment(self) -> RegimePrediction:
        """
        Update current regime assessment and adjust risk parameters

        Returns:
            Current regime prediction
        """
        try:
            # Get current regime prediction
            prediction = self.regime_classifier.predict_regime()

            # Update state
            old_regime = self.state.current_regime
            self.state.current_regime = prediction.regime
            self.state.regime_confidence = prediction.confidence
            self.state.last_regime_update = datetime.now()

            # Check for regime change
            if old_regime != prediction.regime:
                logger.info(f"Regime change detected: {old_regime.value} -> {prediction.regime.value}")

                # Generate alert if configured
                if self.config.alert_on_regime_change:
                    self._generate_regime_change_alert(old_regime, prediction)

                # Update risk limits
                self._update_risk_limits(prediction)

                # Update risk manager regime
                mapped_regime = self.regime_mapping[prediction.regime]
                self.risk_manager.current_regime = mapped_regime

            # Check confidence level
            if prediction.confidence < self.config.low_confidence_threshold:
                if self.config.alert_on_low_confidence:
                    self._generate_low_confidence_alert(prediction)

            # Update performance attribution
            if self.config.enable_regime_attribution:
                self._update_performance_attribution(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Failed to update regime assessment: {e}")
            # Return default prediction on error
            return RegimePrediction(
                regime=MarketRegime.NORMAL,
                confidence=0.5,
                probability_normal=0.6,
                probability_volatile=0.3,
                probability_crisis=0.1,
                indicators=None,
                method='fallback'
            )

    def _update_risk_limits(self, prediction: RegimePrediction):
        """Update risk limits based on regime prediction"""
        try:
            # Get appropriate multipliers
            if prediction.regime == MarketRegime.NORMAL:
                multipliers = self.config.normal_multipliers
            elif prediction.regime == MarketRegime.VOLATILE:
                multipliers = self.config.volatile_multipliers
            else:  # CRISIS
                multipliers = self.config.crisis_multipliers

            # Apply confidence weighting
            confidence_factor = max(0.5, prediction.confidence)

            # Create adjusted risk limits
            adjusted_limits = RiskLimits()
            adjusted_limits.max_portfolio_var = (
                self.base_risk_limits.max_portfolio_var *
                multipliers['max_portfolio_var'] * confidence_factor
            )
            adjusted_limits.max_single_position = (
                self.base_risk_limits.max_single_position *
                multipliers['max_single_position'] * confidence_factor
            )
            adjusted_limits.es_97_5_limit = (
                self.base_risk_limits.es_97_5_limit *
                multipliers['es_97_5_limit'] / confidence_factor  # Inverse for ES
            )
            adjusted_limits.daily_loss_limit = (
                self.base_risk_limits.daily_loss_limit *
                multipliers['daily_loss_limit'] * confidence_factor
            )

            # Keep other limits unchanged
            adjusted_limits.max_sector_weight = self.base_risk_limits.max_sector_weight
            adjusted_limits.max_correlation = self.base_risk_limits.max_correlation
            adjusted_limits.max_drawdown_budget = self.base_risk_limits.max_drawdown_budget

            # Update state and risk manager
            self.state.risk_limits = adjusted_limits
            self.risk_manager.risk_limits = adjusted_limits

            logger.info(f"Risk limits updated for {prediction.regime.value} regime (confidence: {prediction.confidence:.3f})")

        except Exception as e:
            logger.error(f"Failed to update risk limits: {e}")

    def _generate_regime_change_alert(self, old_regime: MarketRegime, prediction: RegimePrediction):
        """Generate alert for regime change"""
        alert = RiskAlert(
            timestamp=datetime.now().isoformat(),
            level=RiskLevel.HIGH if prediction.regime == MarketRegime.CRISIS else RiskLevel.MEDIUM,
            category="Regime Change",
            message=f"Market regime changed from {old_regime.value} to {prediction.regime.value}",
            metric_value=prediction.confidence,
            limit_value=self.config.regime_change_threshold,
            suggested_actions=[
                "review_portfolio_positions",
                "adjust_position_sizes",
                "increase_monitoring_frequency",
                "validate_risk_models"
            ]
        )

        self.state.active_alerts.append(alert)
        self.risk_manager.alerts.append(alert)

        logger.warning(f"Regime change alert: {alert.message}")

    def _generate_low_confidence_alert(self, prediction: RegimePrediction):
        """Generate alert for low confidence prediction"""
        alert = RiskAlert(
            timestamp=datetime.now().isoformat(),
            level=RiskLevel.MEDIUM,
            category="Low Confidence",
            message=f"Low confidence regime prediction: {prediction.confidence:.3f}",
            metric_value=prediction.confidence,
            limit_value=self.config.low_confidence_threshold,
            suggested_actions=[
                "increase_data_collection",
                "manual_regime_review",
                "conservative_risk_settings",
                "model_diagnostic_check"
            ]
        )

        self.state.active_alerts.append(alert)
        logger.warning(f"Low confidence alert: {alert.message}")

    def _update_performance_attribution(self, prediction: RegimePrediction):
        """Update performance attribution by regime"""
        try:
            attribution = self.performance_tracker.update_attribution(
                regime=prediction.regime,
                confidence=prediction.confidence,
                timestamp=datetime.now()
            )

            self.state.performance_attribution = attribution

        except Exception as e:
            logger.error(f"Failed to update performance attribution: {e}")

    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        time_since_retrain = (datetime.now() - self.state.last_model_retrain).total_seconds()
        return time_since_retrain > self.config.model_retrain_interval

    def _retrain_models_async(self):
        """Retrain models asynchronously"""
        def retrain_task():
            try:
                logger.info("Starting model retraining...")
                self.regime_classifier.fit_models()
                self.state.last_model_retrain = datetime.now()
                logger.info("Model retraining completed")

            except Exception as e:
                logger.error(f"Model retraining failed: {e}")

        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(retrain_task)

    def assess_portfolio_risk_with_regime(self,
                                        portfolio: Dict[str, Any],
                                        market_data: Dict[str, Any],
                                        returns_history: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment with regime awareness

        Args:
            portfolio: Portfolio data
            market_data: Market data
            returns_history: Historical returns

        Returns:
            Enhanced risk assessment with regime information
        """
        try:
            # Get current regime assessment
            regime_prediction = self.update_regime_assessment()

            # Perform standard risk assessment
            standard_assessment = self.risk_manager.assess_portfolio_risk(
                portfolio, market_data, returns_history
            )

            # Add regime-specific information
            regime_assessment = {
                'regime_analysis': {
                    'current_regime': regime_prediction.regime.value,
                    'regime_confidence': regime_prediction.confidence,
                    'regime_probabilities': {
                        'normal': regime_prediction.probability_normal,
                        'volatile': regime_prediction.probability_volatile,
                        'crisis': regime_prediction.probability_crisis
                    },
                    'regime_indicators': {
                        'vix_level': regime_prediction.indicators.vix_level if regime_prediction.indicators else 0,
                        'volatility_percentile': regime_prediction.indicators.volatility_percentile if regime_prediction.indicators else 0,
                        'correlation_level': regime_prediction.indicators.correlation_level if regime_prediction.indicators else 0
                    }
                },
                'regime_adjusted_limits': {
                    'max_portfolio_var': self.state.risk_limits.max_portfolio_var,
                    'max_single_position': self.state.risk_limits.max_single_position,
                    'es_97_5_limit': self.state.risk_limits.es_97_5_limit,
                    'daily_loss_limit': self.state.risk_limits.daily_loss_limit
                },
                'regime_risk_score': self._calculate_regime_risk_score(regime_prediction, portfolio),
                'regime_recommendations': self._generate_regime_recommendations(regime_prediction, standard_assessment)
            }

            # Combine assessments
            enhanced_assessment = {**standard_assessment, **regime_assessment}

            return enhanced_assessment

        except Exception as e:
            logger.error(f"Failed to assess portfolio risk with regime: {e}")
            # Fallback to standard assessment
            return self.risk_manager.assess_portfolio_risk(portfolio, market_data, returns_history)

    def _calculate_regime_risk_score(self,
                                   prediction: RegimePrediction,
                                   portfolio: Dict[str, Any]) -> float:
        """Calculate regime-specific risk score"""
        try:
            base_score = 0.5

            # Regime contribution
            if prediction.regime == MarketRegime.CRISIS:
                regime_score = 0.9
            elif prediction.regime == MarketRegime.VOLATILE:
                regime_score = 0.7
            else:
                regime_score = 0.3

            # Confidence adjustment
            confidence_adjustment = 1 - (prediction.confidence - 0.5)

            # Portfolio concentration factor
            positions = portfolio.get('positions', [])
            total_value = portfolio.get('total_value', 1)

            if positions and total_value > 0:
                position_weights = [pos.get('market_value', 0) / total_value for pos in positions]
                concentration = max(position_weights) if position_weights else 0
                concentration_factor = 1 + concentration  # Higher concentration = higher risk
            else:
                concentration_factor = 1

            # Combined score
            final_score = min(1.0, regime_score * confidence_adjustment * concentration_factor)

            return final_score

        except Exception as e:
            logger.error(f"Failed to calculate regime risk score: {e}")
            return 0.5

    def _generate_regime_recommendations(self,
                                       prediction: RegimePrediction,
                                       standard_assessment: Dict[str, Any]) -> List[str]:
        """Generate regime-specific recommendations"""
        recommendations = []

        try:
            # Regime-specific recommendations
            if prediction.regime == MarketRegime.CRISIS:
                recommendations.extend([
                    "Reduce position sizes significantly",
                    "Increase cash allocation",
                    "Consider defensive sector rotation",
                    "Implement dynamic hedging strategies",
                    "Monitor intraday volatility closely"
                ])

            elif prediction.regime == MarketRegime.VOLATILE:
                recommendations.extend([
                    "Reduce position concentration",
                    "Increase stop-loss discipline",
                    "Consider volatility-adjusted position sizing",
                    "Monitor correlation changes",
                    "Prepare for potential regime shift"
                ])

            else:  # NORMAL
                recommendations.extend([
                    "Standard risk management procedures",
                    "Monitor for early regime change signals",
                    "Consider strategic rebalancing",
                    "Maintain diversification discipline"
                ])

            # Confidence-based recommendations
            if prediction.confidence < self.config.low_confidence_threshold:
                recommendations.extend([
                    "Apply conservative risk parameters",
                    "Increase monitoring frequency",
                    "Consider manual regime validation",
                    "Review model inputs and data quality"
                ])

            # Standard assessment integration
            violations = standard_assessment.get('risk_violations', [])
            if violations:
                recommendations.append("Address identified risk violations immediately")

            drawdown_tier = standard_assessment.get('drawdown_tier', 0)
            if drawdown_tier > 0:
                recommendations.append(f"Implement Tier {drawdown_tier} drawdown protocols")

        except Exception as e:
            logger.error(f"Failed to generate regime recommendations: {e}")
            recommendations.append("Review risk parameters and portfolio allocation")

        return recommendations

    def get_regime_dashboard_data(self) -> Dict[str, Any]:
        """Get data for regime-risk dashboard"""
        try:
            # Get risk dashboard data
            risk_dashboard = self.risk_manager.get_risk_dashboard_data()

            # Add regime-specific data
            regime_dashboard = {
                'regime_state': {
                    'current_regime': self.state.current_regime.value,
                    'confidence': self.state.regime_confidence,
                    'last_update': self.state.last_regime_update.isoformat() if self.state.last_regime_update != datetime.min else None,
                    'monitoring_active': self.monitoring_active
                },
                'risk_adjustments': {
                    'base_limits': {
                        'max_portfolio_var': self.base_risk_limits.max_portfolio_var,
                        'max_single_position': self.base_risk_limits.max_single_position,
                        'es_97_5_limit': self.base_risk_limits.es_97_5_limit
                    },
                    'adjusted_limits': {
                        'max_portfolio_var': self.state.risk_limits.max_portfolio_var,
                        'max_single_position': self.state.risk_limits.max_single_position,
                        'es_97_5_limit': self.state.risk_limits.es_97_5_limit
                    },
                    'adjustment_ratios': {
                        'portfolio_var': self.state.risk_limits.max_portfolio_var / self.base_risk_limits.max_portfolio_var,
                        'single_position': self.state.risk_limits.max_single_position / self.base_risk_limits.max_single_position,
                        'es_limit': self.state.risk_limits.es_97_5_limit / self.base_risk_limits.es_97_5_limit
                    }
                },
                'performance_attribution': self.state.performance_attribution,
                'regime_alerts': [
                    {
                        'timestamp': alert.timestamp,
                        'level': alert.level.value,
                        'category': alert.category,
                        'message': alert.message
                    } for alert in self.state.active_alerts[-5:]  # Last 5 regime alerts
                ]
            }

            # Combine dashboards
            combined_dashboard = {**risk_dashboard, **regime_dashboard}

            return combined_dashboard

        except Exception as e:
            logger.error(f"Failed to get regime dashboard data: {e}")
            return {'error': str(e)}

    def export_integration_report(self, filepath: str) -> bool:
        """Export comprehensive integration report"""
        try:
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'integration_config': {
                    'regime_update_interval': self.config.regime_update_interval,
                    'model_retrain_interval': self.config.model_retrain_interval,
                    'risk_multipliers': {
                        'normal': self.config.normal_multipliers,
                        'volatile': self.config.volatile_multipliers,
                        'crisis': self.config.crisis_multipliers
                    }
                },
                'current_state': {
                    'current_regime': self.state.current_regime.value,
                    'regime_confidence': self.state.regime_confidence,
                    'last_regime_update': self.state.last_regime_update.isoformat() if self.state.last_regime_update != datetime.min else None,
                    'monitoring_active': self.monitoring_active,
                    'active_alerts_count': len(self.state.active_alerts)
                },
                'risk_adjustments': {
                    'base_limits': {
                        'max_portfolio_var': self.base_risk_limits.max_portfolio_var,
                        'max_single_position': self.base_risk_limits.max_single_position,
                        'es_97_5_limit': self.base_risk_limits.es_97_5_limit
                    },
                    'current_limits': {
                        'max_portfolio_var': self.state.risk_limits.max_portfolio_var,
                        'max_single_position': self.state.risk_limits.max_single_position,
                        'es_97_5_limit': self.state.risk_limits.es_97_5_limit
                    }
                },
                'performance_attribution': self.state.performance_attribution,
                'regime_classifier_summary': self.regime_classifier.get_regime_summary(),
                'risk_manager_summary': self.risk_manager.get_risk_dashboard_data()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Integration report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export integration report: {e}")
            return False


class RegimePerformanceTracker:
    """Track portfolio performance attribution by market regime"""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.regime_performance = {
            MarketRegime.NORMAL: [],
            MarketRegime.VOLATILE: [],
            MarketRegime.CRISIS: []
        }
        self.regime_periods = []

    def update_attribution(self,
                         regime: MarketRegime,
                         confidence: float,
                         timestamp: datetime,
                         portfolio_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Update performance attribution

        Args:
            regime: Current regime
            confidence: Regime confidence
            timestamp: Current timestamp
            portfolio_return: Portfolio return (if available)

        Returns:
            Updated attribution data
        """
        # Record regime period
        self.regime_periods.append({
            'regime': regime,
            'confidence': confidence,
            'timestamp': timestamp,
            'portfolio_return': portfolio_return
        })

        # Keep only recent periods
        cutoff_date = timestamp - timedelta(days=self.lookback_days)
        self.regime_periods = [
            p for p in self.regime_periods
            if p['timestamp'] >= cutoff_date
        ]

        # Calculate attribution
        attribution = self._calculate_attribution()

        return attribution

    def _calculate_attribution(self) -> Dict[str, Any]:
        """Calculate performance attribution by regime"""
        attribution = {
            'regime_statistics': {},
            'total_periods': len(self.regime_periods),
            'period_distribution': {},
            'average_confidence': {}
        }

        if not self.regime_periods:
            return attribution

        # Group by regime
        regime_groups = {}
        for period in self.regime_periods:
            regime = period['regime']
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(period)

        # Calculate statistics for each regime
        for regime, periods in regime_groups.items():
            regime_name = regime.value

            # Basic statistics
            count = len(periods)
            avg_confidence = np.mean([p['confidence'] for p in periods])

            # Performance statistics (if returns available)
            returns = [p['portfolio_return'] for p in periods if p['portfolio_return'] is not None]
            if returns:
                avg_return = np.mean(returns)
                return_volatility = np.std(returns)
                sharpe_ratio = avg_return / (return_volatility + 1e-9)
            else:
                avg_return = None
                return_volatility = None
                sharpe_ratio = None

            attribution['regime_statistics'][regime_name] = {
                'period_count': count,
                'period_percentage': count / len(self.regime_periods) * 100,
                'average_confidence': avg_confidence,
                'average_return': avg_return,
                'return_volatility': return_volatility,
                'sharpe_ratio': sharpe_ratio
            }

            attribution['period_distribution'][regime_name] = count
            attribution['average_confidence'][regime_name] = avg_confidence

        return attribution


# Factory function for easy integration
def create_regime_risk_integration(
    config: Optional[RegimeRiskConfig] = None,
    base_risk_limits: Optional[RiskLimits] = None
) -> RegimeRiskIntegration:
    """
    Factory function to create integrated regime-risk system

    Args:
        config: Integration configuration
        base_risk_limits: Base risk limits

    Returns:
        RegimeRiskIntegration instance
    """
    return RegimeRiskIntegration(config, base_risk_limits)


# Compatibility function for existing systems
def get_enhanced_risk_manager_with_regime() -> EnhancedRiskManager:
    """
    Get enhanced risk manager with regime integration

    Returns:
        Enhanced risk manager with regime awareness
    """
    integration = RegimeRiskIntegration()
    integration.start_monitoring()
    return integration.risk_manager


if __name__ == "__main__":
    # Example usage and demonstration
    print("Regime-Risk Integration System")
    print("=" * 40)

    if not RISK_MANAGER_AVAILABLE or not REGIME_CLASSIFIER_AVAILABLE:
        print("Error: Required components not available")
        exit(1)

    # Create integration system
    integration = RegimeRiskIntegration()

    # Start monitoring
    print("Starting real-time monitoring...")
    if integration.start_monitoring():
        print("Monitoring started successfully")

        # Simulate some activity
        print("\nRunning regime assessment...")

        # Update regime
        prediction = integration.update_regime_assessment()
        print(f"Current regime: {prediction.regime.value} (confidence: {prediction.confidence:.3f})")

        # Mock portfolio for testing
        mock_portfolio = {
            'total_value': 1000000,
            'positions': [
                {'symbol': 'AAPL', 'market_value': 150000, 'sector': 'Technology'},
                {'symbol': 'GOOGL', 'market_value': 120000, 'sector': 'Technology'},
                {'symbol': 'JPM', 'market_value': 100000, 'sector': 'Financial'}
            ]
        }

        mock_market_data = {'vix': 25.0, 'market_correlation': 0.6}
        mock_returns = np.random.normal(0.001, 0.02, 100)

        # Assess portfolio risk with regime
        print("\nAssessing portfolio risk with regime awareness...")
        risk_assessment = integration.assess_portfolio_risk_with_regime(
            mock_portfolio, mock_market_data, mock_returns
        )

        print(f"Regime risk score: {risk_assessment.get('regime_risk_score', 'N/A')}")
        print(f"Adjusted ES limit: {risk_assessment.get('regime_adjusted_limits', {}).get('es_97_5_limit', 'N/A')}")

        # Get dashboard data
        print("\nGenerating dashboard data...")
        dashboard_data = integration.get_regime_dashboard_data()
        print(f"Current regime: {dashboard_data.get('regime_state', {}).get('current_regime', 'Unknown')}")

        # Export report
        print("\nExporting integration report...")
        report_success = integration.export_integration_report("regime_risk_integration_report.json")
        if report_success:
            print("Report exported successfully")

        # Stop monitoring
        print("\nStopping monitoring...")
        integration.stop_monitoring()

    else:
        print("Failed to start monitoring")

    print("\nRegime-Risk Integration demonstration complete!")