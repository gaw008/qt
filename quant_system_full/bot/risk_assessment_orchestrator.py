#!/usr/bin/env python3
"""
Risk Assessment Orchestrator - Investment Grade Risk Management

This module provides the main orchestration layer for risk management,
coordinating between specialized risk calculation services.

Features:
- Orchestrates tail risk calculations, regime detection, and drawdown management
- Generates comprehensive risk assessments
- Manages risk alerts and reporting
- Provides real-time risk monitoring dashboard data
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

# Import specialized services
from risk_calculation_services import (
    TailRiskCalculator,
    RegimeDetectionService,
    DrawdownManager,
    CorrelationAnalyzer,
    MarketRegime,
    TailRiskMetrics
)

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk alert levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimits:
    """Dynamic risk limits configuration"""
    max_portfolio_var: float = 0.20  # Maximum portfolio variance
    max_single_position: float = 0.10  # Maximum single position weight
    max_sector_weight: float = 0.25   # Maximum sector concentration
    max_correlation: float = 0.80     # Maximum pairwise correlation
    es_97_5_limit: float = 0.05       # Expected Shortfall @ 97.5% daily
    daily_loss_limit: float = 0.03    # Daily loss circuit breaker
    max_drawdown_budget: float = 0.15 # Maximum allowable drawdown


@dataclass
class RiskAlert:
    """Risk alert structure"""
    timestamp: str
    level: RiskLevel
    category: str
    message: str
    metric_value: float
    limit_value: float
    suggested_actions: List[str]


@dataclass
class RiskAssessmentResult:
    """Comprehensive risk assessment result"""
    timestamp: str
    market_regime: str
    tail_risk_metrics: Dict[str, float]
    concentration_risk: Dict[str, float]
    correlation_analysis: Dict[str, Any]
    risk_violations: List[str]
    active_alerts: List[RiskAlert]
    drawdown_tier: int
    suggested_actions: List[str]
    risk_limits: Dict[str, float]
    overall_risk_score: float


class RiskAssessmentOrchestrator:
    """
    Main orchestrator for investment-grade risk management.

    Coordinates between specialized risk services to provide comprehensive
    portfolio risk assessment and monitoring.
    """

    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize the risk assessment orchestrator.

        Args:
            risk_limits: Custom risk limits configuration
        """
        self.risk_limits = risk_limits or RiskLimits()

        # Initialize specialized services
        self.tail_calculator = TailRiskCalculator()
        self.regime_detector = RegimeDetectionService()
        self.drawdown_manager = DrawdownManager()
        self.correlation_analyzer = CorrelationAnalyzer()

        # State tracking
        self.current_regime = MarketRegime.NORMAL
        self.alerts: List[RiskAlert] = []
        self.assessment_history: List[RiskAssessmentResult] = []
        self.active_tier = 0

        logger.info("Risk Assessment Orchestrator initialized with investment-grade controls")

    def assess_portfolio_risk(self,
                            portfolio: Dict[str, Any],
                            market_data: Dict[str, Any],
                            returns_history: np.ndarray,
                            returns_by_asset: Optional[Dict[str, np.ndarray]] = None) -> RiskAssessmentResult:
        """
        Conduct comprehensive portfolio risk assessment.

        Args:
            portfolio: Portfolio data including positions and values
            market_data: Market indicators and benchmarks
            returns_history: Portfolio return history
            returns_by_asset: Individual asset return series

        Returns:
            Comprehensive risk assessment result
        """
        logger.info("Starting comprehensive portfolio risk assessment")

        # 1. Detect market regime and adjust limits
        regime_assessment = self._assess_market_regime(market_data)

        # 2. Calculate tail risk metrics
        tail_risk_assessment = self._assess_tail_risk(returns_history, market_data)

        # 3. Analyze concentration and correlation risk
        concentration_assessment = self._assess_concentration_risk(portfolio, returns_by_asset)

        # 4. Check drawdown tiers
        drawdown_assessment = self._assess_drawdown_risk(tail_risk_assessment['metrics'])

        # 5. Generate risk alerts
        alerts = self._generate_risk_alerts(
            regime_assessment,
            tail_risk_assessment,
            concentration_assessment,
            drawdown_assessment
        )

        # 6. Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk_score(
            tail_risk_assessment['metrics'],
            concentration_assessment['metrics'],
            len([a for a in alerts if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        )

        # 7. Compile assessment result
        assessment_result = RiskAssessmentResult(
            timestamp=datetime.now().isoformat(),
            market_regime=regime_assessment['regime'].value,
            tail_risk_metrics=tail_risk_assessment['metrics_dict'],
            concentration_risk=concentration_assessment['metrics'],
            correlation_analysis=concentration_assessment['correlation_analysis'],
            risk_violations=[],  # Will be populated from alerts
            active_alerts=alerts,
            drawdown_tier=drawdown_assessment['tier'],
            suggested_actions=drawdown_assessment['actions'],
            risk_limits=regime_assessment['adjusted_limits'],
            overall_risk_score=overall_risk_score
        )

        # Extract violations from alerts
        assessment_result.risk_violations = [
            alert.category.upper().replace(" ", "_") + "_VIOLATION"
            for alert in alerts if alert.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]

        # Store assessment in history
        self.assessment_history.append(assessment_result)

        # Keep only recent history (last 100 assessments)
        if len(self.assessment_history) > 100:
            self.assessment_history = self.assessment_history[-100:]

        logger.info(f"Risk assessment completed - Overall score: {overall_risk_score:.2f}")

        return assessment_result

    def _assess_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market regime and adjust risk limits."""
        regime = self.regime_detector.detect_market_regime(market_data)
        self.current_regime = regime

        # Convert risk limits to dict for adjustment
        base_limits = {
            'max_portfolio_var': self.risk_limits.max_portfolio_var,
            'max_single_position': self.risk_limits.max_single_position,
            'max_sector_weight': self.risk_limits.max_sector_weight,
            'max_correlation': self.risk_limits.max_correlation,
            'es_97_5_limit': self.risk_limits.es_97_5_limit,
            'daily_loss_limit': self.risk_limits.daily_loss_limit,
            'max_drawdown_budget': self.risk_limits.max_drawdown_budget
        }

        adjusted_limits = self.regime_detector.apply_regime_adjustments(base_limits, regime)

        return {
            'regime': regime,
            'base_limits': base_limits,
            'adjusted_limits': adjusted_limits
        }

    def _assess_tail_risk(self, returns_history: np.ndarray, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess tail risk metrics."""
        metrics = self.tail_calculator.calculate_comprehensive_tail_metrics(returns_history)

        # Calculate tail dependence if market data available
        if 'benchmark_returns' in market_data:
            try:
                benchmark_returns = np.array(market_data['benchmark_returns'])
                if len(benchmark_returns) == len(returns_history):
                    metrics.tail_dependence = self.tail_calculator.calculate_tail_dependence(
                        returns_history, benchmark_returns
                    )
            except Exception as e:
                logger.warning(f"Could not calculate tail dependence: {e}")

        # Convert to dict for JSON serialization
        metrics_dict = {
            'es_97_5': metrics.es_97_5,
            'es_99': metrics.es_99,
            'tail_ratio': metrics.tail_ratio,
            'max_drawdown': metrics.max_drawdown,
            'calmar_ratio': metrics.calmar_ratio,
            'skewness': metrics.skewness,
            'kurtosis': metrics.kurtosis,
            'tail_dependence': metrics.tail_dependence
        }

        return {
            'metrics': metrics,
            'metrics_dict': metrics_dict
        }

    def _assess_concentration_risk(self, portfolio: Dict[str, Any],
                                 returns_by_asset: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Assess portfolio concentration and correlation risk."""
        positions = portfolio.get('positions', [])
        total_value = portfolio.get('total_value', 0)

        if not positions or total_value <= 0:
            return {
                'metrics': {},
                'correlation_analysis': {},
                'high_correlation_pairs': []
            }

        # Calculate position weights and sectors
        weights = []
        sectors = []
        symbols = []

        for position in positions:
            weight = position.get('market_value', 0) / total_value
            weights.append(weight)
            sectors.append(position.get('sector', 'Unknown'))
            symbols.append(position.get('symbol', ''))

        weights = np.array(weights)

        # Calculate concentration metrics
        concentration_metrics = self.correlation_analyzer.analyze_concentration_risk(weights, sectors)

        # Add position-level metrics
        concentration_metrics.update({
            'max_position_weight': np.max(weights) if len(weights) > 0 else 0.0,
            'position_count': len(positions),
            'effective_positions': 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0.0
        })

        # Correlation analysis if return data available
        correlation_analysis = {}
        high_correlation_pairs = []

        if returns_by_asset:
            # Filter returns data to match current positions
            position_returns = {
                symbol: returns_by_asset[symbol]
                for symbol in symbols if symbol in returns_by_asset
            }

            if position_returns:
                correlation_matrix = self.correlation_analyzer.calculate_portfolio_correlation_matrix(position_returns)

                if not correlation_matrix.empty:
                    high_correlation_pairs = self.correlation_analyzer.identify_high_correlation_pairs(
                        correlation_matrix, self.risk_limits.max_correlation
                    )

                    correlation_analysis = {
                        'avg_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
                        'max_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()) if len(correlation_matrix) > 1 else 0.0,
                        'correlation_matrix_size': len(correlation_matrix)
                    }

        return {
            'metrics': concentration_metrics,
            'correlation_analysis': correlation_analysis,
            'high_correlation_pairs': high_correlation_pairs
        }

    def _assess_drawdown_risk(self, tail_metrics: TailRiskMetrics) -> Dict[str, Any]:
        """Assess drawdown tier and required actions."""
        current_drawdown = abs(tail_metrics.max_drawdown)
        tier, actions, severity = self.drawdown_manager.check_drawdown_tier(current_drawdown)

        # Update active tier if escalated
        if tier > self.active_tier:
            self.active_tier = tier
            logger.warning(f"Drawdown tier escalated to {tier}: {current_drawdown:.3f}")

        return {
            'tier': tier,
            'actions': actions,
            'severity': severity,
            'current_drawdown': current_drawdown
        }

    def _generate_risk_alerts(self, regime_assessment: Dict[str, Any],
                            tail_risk_assessment: Dict[str, Any],
                            concentration_assessment: Dict[str, Any],
                            drawdown_assessment: Dict[str, Any]) -> List[RiskAlert]:
        """Generate risk alerts based on assessment results."""
        alerts = []
        adjusted_limits = regime_assessment['adjusted_limits']
        tail_metrics = tail_risk_assessment['metrics']
        concentration_metrics = concentration_assessment['metrics']

        # ES limit violation
        if tail_metrics.es_97_5 > adjusted_limits['es_97_5_limit']:
            alerts.append(RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.HIGH,
                category="Tail Risk",
                message=f"ES@97.5% ({tail_metrics.es_97_5:.3f}) exceeds limit ({adjusted_limits['es_97_5_limit']:.3f})",
                metric_value=tail_metrics.es_97_5,
                limit_value=adjusted_limits['es_97_5_limit'],
                suggested_actions=["reduce_position_sizes", "increase_diversification", "implement_hedging"]
            ))

        # Position concentration violation
        max_position_weight = concentration_metrics.get('max_position_weight', 0.0)
        if max_position_weight > adjusted_limits['max_single_position']:
            alerts.append(RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.MEDIUM,
                category="Concentration Risk",
                message=f"Maximum position weight ({max_position_weight:.3f}) exceeds limit ({adjusted_limits['max_single_position']:.3f})",
                metric_value=max_position_weight,
                limit_value=adjusted_limits['max_single_position'],
                suggested_actions=["reduce_concentrated_positions", "rebalance_portfolio"]
            ))

        # Sector concentration violation
        max_sector_weight = concentration_metrics.get('max_sector_weight', 0.0)
        if max_sector_weight > adjusted_limits['max_sector_weight']:
            alerts.append(RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.MEDIUM,
                category="Sector Risk",
                message=f"Maximum sector weight ({max_sector_weight:.3f}) exceeds limit ({adjusted_limits['max_sector_weight']:.3f})",
                metric_value=max_sector_weight,
                limit_value=adjusted_limits['max_sector_weight'],
                suggested_actions=["diversify_sectors", "reduce_sector_exposure"]
            ))

        # Drawdown tier activation
        if drawdown_assessment['tier'] > 0:
            level = RiskLevel.CRITICAL if drawdown_assessment['tier'] >= 3 else RiskLevel.HIGH
            alerts.append(RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=level,
                category="Drawdown Budget",
                message=f"Drawdown tier {drawdown_assessment['tier']} activated: {drawdown_assessment['current_drawdown']:.3f}",
                metric_value=drawdown_assessment['current_drawdown'],
                limit_value=0.08,  # Tier 1 threshold
                suggested_actions=drawdown_assessment['actions']
            ))

        # High correlation alerts
        high_corr_pairs = concentration_assessment.get('high_correlation_pairs', [])
        if high_corr_pairs:
            alerts.append(RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.MEDIUM,
                category="Correlation Risk",
                message=f"Found {len(high_corr_pairs)} high correlation pairs (>{self.risk_limits.max_correlation})",
                metric_value=len(high_corr_pairs),
                limit_value=0,
                suggested_actions=["review_position_correlations", "consider_hedging", "diversify_holdings"]
            ))

        # Store alerts
        self.alerts.extend(alerts)

        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        return alerts

    def _calculate_overall_risk_score(self, tail_metrics: TailRiskMetrics,
                                    concentration_metrics: Dict[str, float],
                                    critical_alert_count: int) -> float:
        """
        Calculate overall portfolio risk score (0-100, higher is riskier).

        Args:
            tail_metrics: Tail risk metrics
            concentration_metrics: Concentration risk metrics
            critical_alert_count: Number of critical/high alerts

        Returns:
            Overall risk score (0-100)
        """
        try:
            # Tail risk component (0-40 points)
            es_component = min(tail_metrics.es_97_5 / self.risk_limits.es_97_5_limit * 20, 20)
            drawdown_component = min(tail_metrics.max_drawdown / self.risk_limits.max_drawdown_budget * 20, 20)
            tail_score = es_component + drawdown_component

            # Concentration risk component (0-30 points)
            max_position_weight = concentration_metrics.get('max_position_weight', 0.0)
            hhi = concentration_metrics.get('herfindahl_index', 0.0)
            position_concentration = min(max_position_weight / self.risk_limits.max_single_position * 15, 15)
            portfolio_concentration = min(hhi * 15, 15)
            concentration_score = position_concentration + portfolio_concentration

            # Alert-based component (0-30 points)
            alert_score = min(critical_alert_count * 10, 30)

            # Total score
            total_score = tail_score + concentration_score + alert_score

            return min(total_score, 100.0)

        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 50.0  # Default moderate risk score

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time risk dashboard data."""
        recent_alerts = self.alerts[-10:] if self.alerts else []

        alert_summary = {
            'critical': len([a for a in recent_alerts if a.level == RiskLevel.CRITICAL]),
            'high': len([a for a in recent_alerts if a.level == RiskLevel.HIGH]),
            'medium': len([a for a in recent_alerts if a.level == RiskLevel.MEDIUM]),
            'low': len([a for a in recent_alerts if a.level == RiskLevel.LOW])
        }

        return {
            'current_regime': self.current_regime.value,
            'active_tier': self.active_tier,
            'alert_summary': alert_summary,
            'recent_alerts': [
                {
                    'timestamp': a.timestamp,
                    'level': a.level.value,
                    'category': a.category,
                    'message': a.message,
                    'suggested_actions': a.suggested_actions[:3]  # Top 3 actions
                } for a in recent_alerts
            ],
            'assessment_history_count': len(self.assessment_history),
            'last_assessment': self.assessment_history[-1].timestamp if self.assessment_history else None
        }

    def export_risk_report(self, filepath: str) -> bool:
        """Export comprehensive risk report for compliance and audit."""
        try:
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'risk_manager_config': {
                    'es_confidence_level': 0.975,
                    'drawdown_tiers': self.drawdown_manager.get_tier_configuration(),
                    'current_regime': self.current_regime.value,
                    'risk_limits': {
                        'max_portfolio_var': self.risk_limits.max_portfolio_var,
                        'max_single_position': self.risk_limits.max_single_position,
                        'max_sector_weight': self.risk_limits.max_sector_weight,
                        'es_97_5_limit': self.risk_limits.es_97_5_limit,
                        'daily_loss_limit': self.risk_limits.daily_loss_limit,
                        'max_drawdown_budget': self.risk_limits.max_drawdown_budget
                    }
                },
                'current_status': self.get_risk_dashboard_data(),
                'recent_assessments': [
                    {
                        'timestamp': assessment.timestamp,
                        'market_regime': assessment.market_regime,
                        'overall_risk_score': assessment.overall_risk_score,
                        'drawdown_tier': assessment.drawdown_tier,
                        'alert_count': len(assessment.active_alerts)
                    } for assessment in self.assessment_history[-10:]
                ],
                'all_alerts': [
                    {
                        'timestamp': a.timestamp,
                        'level': a.level.value,
                        'category': a.category,
                        'message': a.message,
                        'metric_value': a.metric_value,
                        'limit_value': a.limit_value,
                        'suggested_actions': a.suggested_actions
                    } for a in self.alerts
                ]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Risk report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export risk report: {e}")
            return False