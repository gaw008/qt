#!/usr/bin/env python3
"""
Enhanced Risk Management System
增强风险管理系统

Investment-grade risk management with Expected Shortfall (ES), drawdown budgeting,
tail dependence analysis, and dynamic risk controls.

Features:
- ES@97.5% as primary risk metric (replacing VaR)
- Tiered drawdown budgeting with automatic de-leveraging
- Tail dependence and correlation clustering
- Market regime-aware risk limits
- Real-time risk monitoring and alerts

投资级风险管理功能：
- ES@97.5%作为主要风险指标（替代VaR）
- 分层回撤预算与自动减杠杆
- 尾部相关性与相关性聚类分析
- 市场状态感知风险限制
- 实时风险监控与预警
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from scipy import stats
from scipy.stats import pearsonr
import json

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk alert levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class MarketRegime(Enum):
    """Market regime states for dynamic risk adjustment"""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    TRENDING = "TRENDING"
    CRISIS = "CRISIS"

@dataclass
class RiskLimits:
    """Dynamic risk limits based on market regime"""
    max_portfolio_var: float = 0.20  # Maximum portfolio variance
    max_single_position: float = 0.10  # Maximum single position weight
    max_sector_weight: float = 0.25   # Maximum sector concentration
    max_correlation: float = 0.80     # Maximum pairwise correlation
    es_97_5_limit: float = 0.05       # Expected Shortfall @ 97.5% daily
    daily_loss_limit: float = 0.03    # Daily loss circuit breaker
    max_drawdown_budget: float = 0.15 # Maximum allowable drawdown

    # Regime-specific multipliers
    regime_multipliers: Dict[MarketRegime, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.NORMAL: {"var": 1.0, "position": 1.0, "es": 1.0},
        MarketRegime.VOLATILE: {"var": 0.8, "position": 0.8, "es": 1.2},
        MarketRegime.TRENDING: {"var": 1.1, "position": 1.1, "es": 0.9},
        MarketRegime.CRISIS: {"var": 0.5, "position": 0.5, "es": 2.0}
    })

@dataclass
class DrawdownBudget:
    """Tiered drawdown budgeting system"""
    tier_1_threshold: float = 0.08  # 8% drawdown triggers Tier 1
    tier_2_threshold: float = 0.12  # 12% drawdown triggers Tier 2
    tier_3_threshold: float = 0.15  # 15% drawdown triggers Tier 3

    tier_1_actions: List[str] = field(default_factory=lambda: [
        "reduce_position_size_10%", "increase_stop_loss_tightness", "pause_new_positions"
    ])
    tier_2_actions: List[str] = field(default_factory=lambda: [
        "reduce_position_size_25%", "reduce_sector_concentration", "increase_cash_allocation"
    ])
    tier_3_actions: List[str] = field(default_factory=lambda: [
        "reduce_position_size_50%", "close_high_correlation_positions", "emergency_risk_off"
    ])

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

class EnhancedRiskManager:
    """
    Investment-grade risk management system with ES, drawdown budgeting,
    and dynamic risk controls
    """

    def __init__(self,
                 risk_limits: Optional[RiskLimits] = None,
                 drawdown_budget: Optional[DrawdownBudget] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.drawdown_budget = drawdown_budget or DrawdownBudget()
        self.current_regime = MarketRegime.NORMAL
        self.alerts: List[RiskAlert] = []
        self.performance_history: List[float] = []
        self.position_history: List[Dict] = []
        self.risk_metrics_history: List[TailRiskMetrics] = []

        # State tracking
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.active_tier = 0
        self.last_regime_update = datetime.now()

        logger.info("Enhanced Risk Manager initialized with ES@97.5% and drawdown budgeting")

    def calculate_expected_shortfall(self,
                                   returns: np.ndarray,
                                   confidence_level: float = 0.975) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR) at specified confidence level

        ES is superior to VaR as it measures the expected loss in the tail beyond VaR
        ES@97.5% = E[Loss | Loss > VaR@97.5%]
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

    def calculate_tail_risk_metrics(self, returns: np.ndarray) -> TailRiskMetrics:
        """Calculate comprehensive tail risk metrics"""
        if len(returns) < 10:
            return TailRiskMetrics()

        # Basic statistics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

        # Expected Shortfall calculations
        es_97_5 = self.calculate_expected_shortfall(returns, 0.975)
        es_99 = self.calculate_expected_shortfall(returns, 0.99)

        # Tail ratio (average gain / average loss in tails)
        positive_tail = returns[returns > np.percentile(returns, 95)]
        negative_tail = returns[returns < np.percentile(returns, 5)]

        tail_ratio = 0.0
        if len(positive_tail) > 0 and len(negative_tail) > 0:
            avg_gain = np.mean(positive_tail)
            avg_loss = abs(np.mean(negative_tail))
            tail_ratio = avg_gain / avg_loss if avg_loss > 0 else 0.0

        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # Calmar ratio (annualized return / max drawdown)
        ann_return = np.mean(returns) * 252  # Assuming daily returns
        calmar_ratio = ann_return / max_drawdown if max_drawdown > 0 else 0.0

        return TailRiskMetrics(
            es_97_5=es_97_5,
            es_99=es_99,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio
        )

    def calculate_tail_dependence(self,
                                portfolio_returns: np.ndarray,
                                market_returns: np.ndarray,
                                threshold: float = 0.95) -> float:
        """
        Calculate tail dependence between portfolio and market
        Measures correlation during extreme market events
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

        correlation, _ = pearsonr(extreme_portfolio, extreme_market)
        return correlation if not np.isnan(correlation) else 0.0

    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Detect current market regime based on volatility, correlation, and sentiment

        Regime classification:
        - NORMAL: VIX < 20, correlation < 0.5
        - VOLATILE: VIX 20-30, correlation 0.5-0.7
        - TRENDING: VIX < 25, strong directional momentum
        - CRISIS: VIX > 30, correlation > 0.7
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

    def get_regime_adjusted_limits(self, regime: MarketRegime) -> RiskLimits:
        """Get risk limits adjusted for current market regime"""
        multipliers = self.risk_limits.regime_multipliers.get(regime, {})

        adjusted_limits = RiskLimits()
        adjusted_limits.max_portfolio_var = self.risk_limits.max_portfolio_var * multipliers.get('var', 1.0)
        adjusted_limits.max_single_position = self.risk_limits.max_single_position * multipliers.get('position', 1.0)
        adjusted_limits.es_97_5_limit = self.risk_limits.es_97_5_limit * multipliers.get('es', 1.0)
        adjusted_limits.max_sector_weight = self.risk_limits.max_sector_weight
        adjusted_limits.max_correlation = self.risk_limits.max_correlation
        adjusted_limits.daily_loss_limit = self.risk_limits.daily_loss_limit
        adjusted_limits.max_drawdown_budget = self.risk_limits.max_drawdown_budget

        return adjusted_limits

    def check_drawdown_tiers(self, current_drawdown: float) -> Tuple[int, List[str]]:
        """
        Check drawdown tiers and return appropriate tier and actions

        Returns:
            Tuple of (tier_level, suggested_actions)
        """
        if current_drawdown >= self.drawdown_budget.tier_3_threshold:
            return 3, self.drawdown_budget.tier_3_actions
        elif current_drawdown >= self.drawdown_budget.tier_2_threshold:
            return 2, self.drawdown_budget.tier_2_actions
        elif current_drawdown >= self.drawdown_budget.tier_1_threshold:
            return 1, self.drawdown_budget.tier_1_actions
        else:
            return 0, []

    def assess_portfolio_risk(self,
                            portfolio: Dict[str, Any],
                            market_data: Dict[str, Any],
                            returns_history: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment with ES and tail risk analysis
        """
        # Update market regime
        self.current_regime = self.detect_market_regime(market_data)
        adjusted_limits = self.get_regime_adjusted_limits(self.current_regime)

        # Calculate tail risk metrics
        tail_metrics = self.calculate_tail_risk_metrics(returns_history)

        # Portfolio-level metrics
        positions = portfolio.get('positions', [])
        total_value = portfolio.get('total_value', 0)

        # Position concentration analysis
        position_weights = []
        sector_weights = {}
        correlations = []

        for position in positions:
            weight = position.get('market_value', 0) / total_value if total_value > 0 else 0
            position_weights.append(weight)

            sector = position.get('sector', 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Risk violations
        violations = []
        alerts = []

        # Check ES limit
        if tail_metrics.es_97_5 > adjusted_limits.es_97_5_limit:
            alert = RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.HIGH,
                category="Tail Risk",
                message=f"ES@97.5% ({tail_metrics.es_97_5:.3f}) exceeds limit ({adjusted_limits.es_97_5_limit:.3f})",
                metric_value=tail_metrics.es_97_5,
                limit_value=adjusted_limits.es_97_5_limit,
                suggested_actions=["reduce_position_sizes", "increase_diversification", "implement_hedging"]
            )
            alerts.append(alert)
            violations.append("ES_VIOLATION")

        # Check position concentration
        max_position_weight = max(position_weights) if position_weights else 0
        if max_position_weight > adjusted_limits.max_single_position:
            alert = RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.MEDIUM,
                category="Concentration Risk",
                message=f"Maximum position weight ({max_position_weight:.3f}) exceeds limit ({adjusted_limits.max_single_position:.3f})",
                metric_value=max_position_weight,
                limit_value=adjusted_limits.max_single_position,
                suggested_actions=["reduce_concentrated_positions", "rebalance_portfolio"]
            )
            alerts.append(alert)
            violations.append("CONCENTRATION_VIOLATION")

        # Check sector concentration
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        if max_sector_weight > adjusted_limits.max_sector_weight:
            alert = RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.MEDIUM,
                category="Sector Risk",
                message=f"Maximum sector weight ({max_sector_weight:.3f}) exceeds limit ({adjusted_limits.max_sector_weight:.3f})",
                metric_value=max_sector_weight,
                limit_value=adjusted_limits.max_sector_weight,
                suggested_actions=["diversify_sectors", "reduce_sector_exposure"]
            )
            alerts.append(alert)
            violations.append("SECTOR_VIOLATION")

        # Check drawdown tiers
        current_drawdown = abs(tail_metrics.max_drawdown)
        tier, tier_actions = self.check_drawdown_tiers(current_drawdown)

        if tier > self.active_tier:
            alert = RiskAlert(
                timestamp=datetime.now().isoformat(),
                level=RiskLevel.CRITICAL if tier >= 3 else RiskLevel.HIGH,
                category="Drawdown Budget",
                message=f"Drawdown tier {tier} activated: {current_drawdown:.3f}",
                metric_value=current_drawdown,
                limit_value=self.drawdown_budget.tier_1_threshold,
                suggested_actions=tier_actions
            )
            alerts.append(alert)
            self.active_tier = tier

        # Store alerts
        self.alerts.extend(alerts)

        # Risk assessment summary
        risk_assessment = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': self.current_regime.value,
            'tail_risk_metrics': {
                'es_97_5': tail_metrics.es_97_5,
                'es_99': tail_metrics.es_99,
                'tail_ratio': tail_metrics.tail_ratio,
                'max_drawdown': tail_metrics.max_drawdown,
                'calmar_ratio': tail_metrics.calmar_ratio,
                'skewness': tail_metrics.skewness,
                'kurtosis': tail_metrics.kurtosis
            },
            'concentration_risk': {
                'max_position_weight': max_position_weight,
                'max_sector_weight': max_sector_weight,
                'position_count': len(positions)
            },
            'risk_violations': violations,
            'active_alerts': len([a for a in alerts if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]),
            'drawdown_tier': tier,
            'suggested_actions': tier_actions,
            'risk_limits': {
                'es_97_5_limit': adjusted_limits.es_97_5_limit,
                'max_position_limit': adjusted_limits.max_single_position,
                'max_sector_limit': adjusted_limits.max_sector_weight
            }
        }

        return risk_assessment

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time risk dashboard data"""
        recent_alerts = [a for a in self.alerts[-10:]]  # Last 10 alerts

        alert_summary = {
            'critical': len([a for a in recent_alerts if a.level == RiskLevel.CRITICAL]),
            'high': len([a for a in recent_alerts if a.level == RiskLevel.HIGH]),
            'medium': len([a for a in recent_alerts if a.level == RiskLevel.MEDIUM]),
            'low': len([a for a in recent_alerts if a.level == RiskLevel.LOW])
        }

        return {
            'current_regime': self.current_regime.value,
            'active_tier': self.active_tier,
            'current_drawdown': self.current_drawdown,
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
            'risk_metrics_trend': self.risk_metrics_history[-30:] if self.risk_metrics_history else []
        }

    def export_risk_report(self, filepath: str) -> bool:
        """Export comprehensive risk report for compliance and audit"""
        try:
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'risk_manager_config': {
                    'es_confidence_level': 0.975,
                    'drawdown_tiers': {
                        'tier_1': self.drawdown_budget.tier_1_threshold,
                        'tier_2': self.drawdown_budget.tier_2_threshold,
                        'tier_3': self.drawdown_budget.tier_3_threshold
                    },
                    'current_regime': self.current_regime.value
                },
                'current_status': self.get_risk_dashboard_data(),
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
                ],
                'performance_metrics': {
                    'total_periods': len(self.performance_history),
                    'risk_metrics_count': len(self.risk_metrics_history)
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Risk report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export risk report: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Risk Manager - Investment Grade Risk Controls")
    print("=" * 60)

    # Initialize enhanced risk manager
    risk_manager = EnhancedRiskManager()

    # Simulate some return data for testing
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    returns[50:55] = -0.05  # Simulate a crisis period

    # Calculate tail risk metrics
    tail_metrics = risk_manager.calculate_tail_risk_metrics(returns)
    print("Tail Risk Metrics:")
    print(f"ES@97.5%: {tail_metrics.es_97_5:.4f}")
    print(f"ES@99%: {tail_metrics.es_99:.4f}")
    print(f"Max Drawdown: {tail_metrics.max_drawdown:.4f}")
    print(f"Calmar Ratio: {tail_metrics.calmar_ratio:.4f}")
    print(f"Skewness: {tail_metrics.skewness:.4f}")
    print(f"Kurtosis: {tail_metrics.kurtosis:.4f}")

    # Simulate portfolio data
    mock_portfolio = {
        'total_value': 1000000,
        'positions': [
            {'symbol': 'AAPL', 'market_value': 150000, 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'market_value': 120000, 'sector': 'Technology'},
            {'symbol': 'JPM', 'market_value': 100000, 'sector': 'Financial'},
            {'symbol': 'JNJ', 'market_value': 80000, 'sector': 'Healthcare'}
        ]
    }

    # Simulate market data
    mock_market_data = {
        'vix': 25.0,
        'market_correlation': 0.6,
        'momentum_strength': 0.3
    }

    # Assess portfolio risk
    risk_assessment = risk_manager.assess_portfolio_risk(
        mock_portfolio, mock_market_data, returns
    )

    print(f"\nRisk Assessment Summary:")
    print(f"Market Regime: {risk_assessment['market_regime']}")
    print(f"ES@97.5%: {risk_assessment['tail_risk_metrics']['es_97_5']:.4f}")
    print(f"Max Position Weight: {risk_assessment['concentration_risk']['max_position_weight']:.3f}")
    print(f"Risk Violations: {risk_assessment['risk_violations']}")
    print(f"Active Alerts: {risk_assessment['active_alerts']}")
    print(f"Drawdown Tier: {risk_assessment['drawdown_tier']}")

    # Export risk report
    risk_manager.export_risk_report("risk_assessment_report.json")
    print(f"\nRisk report exported successfully!")