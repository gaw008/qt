#!/usr/bin/env python3
"""
Factor Crowding Detection and Monitoring System
因子拥挤度检测与监控系统

Investment-grade factor crowding detection to prevent systematic risks:
- Cross-sectional factor exposure analysis
- Crowding metrics based on factor loadings distribution
- Dynamic de-crowding mechanisms and alerts
- ETF/CTA holdings proxy analysis
- Regime-dependent crowding thresholds

投资级因子拥挤度检测：
- 横截面因子暴露分析
- 基于因子载荷分布的拥挤度指标
- 动态去拥挤机制与预警
- ETF/CTA持仓代理分析
- 状态依赖的拥挤阈值
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
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class CrowdingLevel(Enum):
    """Factor crowding severity levels"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class MarketRegime(Enum):
    """Market regime for crowding threshold adjustment"""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"

@dataclass
class CrowdingThresholds:
    """Dynamic crowding thresholds by market regime"""
    normal_regime: Dict[str, float] = field(default_factory=lambda: {
        'concentration_hhi': 0.15,      # Herfindahl-Hirschman Index threshold
        'factor_skewness': 2.0,         # Factor loading distribution skewness
        'cross_correlation': 0.7,       # Cross-factor correlation threshold
        'pca_concentration': 0.8,       # First PC explains >80% variance
        'active_share_threshold': 0.6   # Active share relative to benchmark
    })

    volatile_regime: Dict[str, float] = field(default_factory=lambda: {
        'concentration_hhi': 0.12,      # Stricter in volatile markets
        'factor_skewness': 1.5,
        'cross_correlation': 0.6,
        'pca_concentration': 0.75,
        'active_share_threshold': 0.5
    })

    crisis_regime: Dict[str, float] = field(default_factory=lambda: {
        'concentration_hhi': 0.10,      # Very strict during crisis
        'factor_skewness': 1.0,
        'cross_correlation': 0.5,
        'pca_concentration': 0.7,
        'active_share_threshold': 0.4
    })

@dataclass
class FactorCrowdingMetrics:
    """Comprehensive factor crowding metrics"""
    timestamp: str
    factor_name: str

    # Concentration metrics
    herfindahl_index: float = 0.0       # HHI of factor loadings
    gini_coefficient: float = 0.0       # Inequality in factor exposure
    top_decile_share: float = 0.0       # Share of top 10% exposures

    # Distribution metrics
    loading_skewness: float = 0.0       # Skewness of factor loadings
    loading_kurtosis: float = 0.0       # Kurtosis (tail heaviness)
    effective_breadth: float = 0.0      # Effective number of positions

    # Cross-factor metrics
    avg_correlation: float = 0.0        # Average correlation with other factors
    max_correlation: float = 0.0        # Maximum correlation with any factor
    pca_explained_variance: float = 0.0 # Variance explained by first PC

    # Market-based metrics
    active_share: float = 0.0           # Active share vs market
    tracking_error: float = 0.0         # Portfolio tracking error

    # Crowding assessment
    crowding_level: CrowdingLevel = CrowdingLevel.LOW
    crowding_score: float = 0.0         # Composite crowding score (0-100)
    risk_contribution: float = 0.0      # Contribution to portfolio risk

@dataclass
class CrowdingAlert:
    """Crowding alert structure"""
    timestamp: str
    factor_name: str
    alert_level: CrowdingLevel
    metric_name: str
    current_value: float
    threshold_value: float
    recommended_actions: List[str]
    impact_assessment: str

class FactorCrowdingMonitor:
    """
    Comprehensive factor crowding detection and monitoring system

    Features:
    - Multi-dimensional crowding metrics calculation
    - Dynamic threshold adjustment by market regime
    - Real-time crowding alerts and recommendations
    - Historical crowding trend analysis
    - Portfolio de-crowding optimization
    """

    def __init__(self, thresholds: Optional[CrowdingThresholds] = None):
        self.thresholds = thresholds or CrowdingThresholds()
        self.current_regime = MarketRegime.NORMAL
        self.crowding_history: List[FactorCrowdingMetrics] = []
        self.alerts: List[CrowdingAlert] = []
        self.factor_returns_history: Dict[str, List[float]] = {}

        logger.info("Factor Crowding Monitor initialized")

    def calculate_herfindahl_index(self, exposures: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index for factor concentration

        HHI = Σ(weight_i)²
        Higher values indicate more concentration
        """
        if len(exposures) == 0:
            return 0.0

        # Normalize exposures to weights (handle negative exposures)
        abs_exposures = np.abs(exposures)
        total_exposure = np.sum(abs_exposures)

        if total_exposure == 0:
            return 0.0

        weights = abs_exposures / total_exposure
        hhi = np.sum(weights ** 2)

        return hhi

    def calculate_gini_coefficient(self, exposures: np.ndarray) -> float:
        """
        Calculate Gini coefficient for inequality in factor exposures

        Values closer to 1 indicate higher inequality (more crowding)
        """
        if len(exposures) <= 1:
            return 0.0

        # Use absolute exposures
        abs_exposures = np.abs(exposures)
        abs_exposures = np.sort(abs_exposures)

        n = len(abs_exposures)
        index = np.arange(1, n + 1)

        # Gini coefficient formula
        gini = (2 * np.sum(index * abs_exposures)) / (n * np.sum(abs_exposures)) - (n + 1) / n

        return max(0.0, gini)

    def calculate_effective_breadth(self, exposures: np.ndarray) -> float:
        """
        Calculate effective breadth (effective number of positions)

        Effective Breadth = 1 / Σ(weight_i)²
        Higher values indicate better diversification
        """
        if len(exposures) == 0:
            return 0.0

        hhi = self.calculate_herfindahl_index(exposures)
        return 1.0 / hhi if hhi > 0 else 0.0

    def analyze_factor_distribution(self,
                                  factor_exposures: np.ndarray,
                                  factor_name: str) -> Dict[str, float]:
        """
        Analyze the distribution characteristics of factor exposures
        """
        if len(factor_exposures) == 0:
            return {}

        # Basic statistics
        factor_skewness = skew(factor_exposures)
        factor_kurtosis = kurtosis(factor_exposures, fisher=True)  # Excess kurtosis

        # Concentration metrics
        hhi = self.calculate_herfindahl_index(factor_exposures)
        gini = self.calculate_gini_coefficient(factor_exposures)
        effective_breadth = self.calculate_effective_breadth(factor_exposures)

        # Top decile analysis
        abs_exposures = np.abs(factor_exposures)
        top_decile_threshold = np.percentile(abs_exposures, 90)
        top_decile_exposures = abs_exposures[abs_exposures >= top_decile_threshold]
        top_decile_share = np.sum(top_decile_exposures) / np.sum(abs_exposures) if np.sum(abs_exposures) > 0 else 0

        return {
            'herfindahl_index': hhi,
            'gini_coefficient': gini,
            'effective_breadth': effective_breadth,
            'loading_skewness': factor_skewness,
            'loading_kurtosis': factor_kurtosis,
            'top_decile_share': top_decile_share
        }

    def calculate_cross_factor_correlations(self,
                                          factor_exposures_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate cross-factor correlation metrics
        """
        if factor_exposures_matrix.shape[1] < 2:
            return {'avg_correlation': 0.0, 'max_correlation': 0.0}

        # Calculate correlation matrix
        corr_matrix = factor_exposures_matrix.corr()

        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        correlations = upper_triangle.stack().values
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) == 0:
            return {'avg_correlation': 0.0, 'max_correlation': 0.0}

        avg_correlation = np.mean(np.abs(correlations))
        max_correlation = np.max(np.abs(correlations))

        return {
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation
        }

    def perform_pca_analysis(self, factor_exposures_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Perform PCA to detect factor clustering and concentration
        """
        if factor_exposures_matrix.shape[1] < 2:
            return {'pca_explained_variance': 0.0, 'effective_factors': 1.0}

        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(factor_exposures_matrix.fillna(0))

        # Perform PCA
        pca = PCA()
        pca.fit(standardized_data)

        # Calculate cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        first_pc_variance = explained_variance_ratio[0]

        # Calculate effective number of factors (inverse participation ratio)
        weights = explained_variance_ratio
        effective_factors = 1.0 / np.sum(weights ** 2) if len(weights) > 0 else 1.0

        return {
            'pca_explained_variance': first_pc_variance,
            'effective_factors': effective_factors,
            'cumulative_variance_80': np.where(np.cumsum(explained_variance_ratio) >= 0.8)[0][0] + 1 if len(explained_variance_ratio) > 0 else 1
        }

    def calculate_active_share(self,
                             portfolio_weights: np.ndarray,
                             benchmark_weights: np.ndarray) -> float:
        """
        Calculate active share relative to benchmark

        Active Share = 0.5 * Σ|w_portfolio - w_benchmark|
        """
        if len(portfolio_weights) != len(benchmark_weights):
            logger.warning("Portfolio and benchmark weights have different lengths")
            return 0.0

        # Ensure weights sum to 1
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights) if np.sum(portfolio_weights) > 0 else portfolio_weights
        benchmark_weights = benchmark_weights / np.sum(benchmark_weights) if np.sum(benchmark_weights) > 0 else benchmark_weights

        active_share = 0.5 * np.sum(np.abs(portfolio_weights - benchmark_weights))
        return active_share

    def assess_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Assess current market regime for dynamic threshold adjustment
        """
        vix = market_data.get('vix', 20)
        market_stress = market_data.get('market_stress_index', 0.5)
        correlation_spike = market_data.get('correlation_spike', False)

        if vix > 30 or market_stress > 0.8 or correlation_spike:
            return MarketRegime.CRISIS
        elif vix > 20 or market_stress > 0.6:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.NORMAL

    def get_current_thresholds(self) -> Dict[str, float]:
        """Get thresholds for current market regime"""
        if self.current_regime == MarketRegime.CRISIS:
            return self.thresholds.crisis_regime
        elif self.current_regime == MarketRegime.VOLATILE:
            return self.thresholds.volatile_regime
        else:
            return self.thresholds.normal_regime

    def calculate_crowding_score(self, metrics: FactorCrowdingMetrics) -> float:
        """
        Calculate composite crowding score (0-100 scale)

        Weighted combination of various crowding metrics
        """
        thresholds = self.get_current_thresholds()

        # Normalize metrics to 0-1 scale
        scores = []
        weights = []

        # HHI component (30% weight)
        if thresholds['concentration_hhi'] > 0:
            hhi_score = min(1.0, metrics.herfindahl_index / thresholds['concentration_hhi'])
            scores.append(hhi_score)
            weights.append(0.30)

        # Skewness component (20% weight)
        if thresholds['factor_skewness'] > 0:
            skew_score = min(1.0, abs(metrics.loading_skewness) / thresholds['factor_skewness'])
            scores.append(skew_score)
            weights.append(0.20)

        # Correlation component (25% weight)
        if thresholds['cross_correlation'] > 0:
            corr_score = min(1.0, metrics.max_correlation / thresholds['cross_correlation'])
            scores.append(corr_score)
            weights.append(0.25)

        # PCA concentration component (15% weight)
        if thresholds['pca_concentration'] > 0:
            pca_score = min(1.0, metrics.pca_explained_variance / thresholds['pca_concentration'])
            scores.append(pca_score)
            weights.append(0.15)

        # Active share component (10% weight)
        if thresholds['active_share_threshold'] > 0:
            # Lower active share indicates more crowding (following benchmark)
            active_score = max(0.0, 1.0 - metrics.active_share / thresholds['active_share_threshold'])
            scores.append(active_score)
            weights.append(0.10)

        if not scores:
            return 0.0

        # Calculate weighted average
        weighted_score = np.average(scores, weights=weights)
        return weighted_score * 100  # Convert to 0-100 scale

    def determine_crowding_level(self, crowding_score: float) -> CrowdingLevel:
        """Determine crowding level based on composite score"""
        if crowding_score >= 80:
            return CrowdingLevel.EXTREME
        elif crowding_score >= 60:
            return CrowdingLevel.HIGH
        elif crowding_score >= 40:
            return CrowdingLevel.MODERATE
        else:
            return CrowdingLevel.LOW

    def _calculate_crowding_score(self, factor_hhi: float, max_correlation: float, regime: str) -> float:
        """
        Simplified crowding score calculation for real-time monitoring

        Args:
            factor_hhi: Herfindahl-Hirschman Index for factor concentration
            max_correlation: Maximum correlation between factors
            regime: Market regime string ("Normal", "Volatile", "Crisis")

        Returns:
            Crowding score (0-100 scale)
        """
        try:
            # Set regime-based thresholds
            if regime.upper() == "CRISIS":
                hhi_threshold = 0.10
                corr_threshold = 0.5
            elif regime.upper() == "VOLATILE":
                hhi_threshold = 0.12
                corr_threshold = 0.6
            else:  # Normal
                hhi_threshold = 0.15
                corr_threshold = 0.7

            # Calculate component scores (0-1 scale)
            hhi_score = min(1.0, factor_hhi / hhi_threshold) if hhi_threshold > 0 else 0.0
            corr_score = min(1.0, max_correlation / corr_threshold) if corr_threshold > 0 else 0.0

            # Weighted combination (HHI 60%, Correlation 40% for simplified version)
            weighted_score = 0.6 * hhi_score + 0.4 * corr_score

            return weighted_score * 100  # Convert to 0-100 scale

        except Exception as e:
            logger.error(f"Simplified crowding score calculation failed: {e}")
            return 0.0

    def monitor_factor_crowding(self,
                              factor_exposures: Dict[str, np.ndarray],
                              portfolio_weights: Optional[np.ndarray] = None,
                              benchmark_weights: Optional[np.ndarray] = None,
                              market_data: Optional[Dict[str, Any]] = None) -> List[FactorCrowdingMetrics]:
        """
        Monitor crowding across all factors

        Args:
            factor_exposures: Dict mapping factor names to exposure arrays
            portfolio_weights: Portfolio weights for active share calculation
            benchmark_weights: Benchmark weights for comparison
            market_data: Market data for regime assessment
        """
        # Update market regime
        if market_data:
            self.current_regime = self.assess_market_regime(market_data)

        # Create factor exposure matrix
        factor_df = pd.DataFrame(factor_exposures)

        # Calculate cross-factor metrics
        cross_factor_metrics = self.calculate_cross_factor_correlations(factor_df)
        pca_metrics = self.perform_pca_analysis(factor_df)

        # Calculate active share if weights provided
        active_share = 0.0
        if portfolio_weights is not None and benchmark_weights is not None:
            active_share = self.calculate_active_share(portfolio_weights, benchmark_weights)

        # Analyze each factor
        factor_metrics = []

        for factor_name, exposures in factor_exposures.items():
            # Factor-specific distribution analysis
            dist_metrics = self.analyze_factor_distribution(exposures, factor_name)

            # Create comprehensive metrics object
            metrics = FactorCrowdingMetrics(
                timestamp=datetime.now().isoformat(),
                factor_name=factor_name,
                herfindahl_index=dist_metrics.get('herfindahl_index', 0.0),
                gini_coefficient=dist_metrics.get('gini_coefficient', 0.0),
                top_decile_share=dist_metrics.get('top_decile_share', 0.0),
                loading_skewness=dist_metrics.get('loading_skewness', 0.0),
                loading_kurtosis=dist_metrics.get('loading_kurtosis', 0.0),
                effective_breadth=dist_metrics.get('effective_breadth', 0.0),
                avg_correlation=cross_factor_metrics.get('avg_correlation', 0.0),
                max_correlation=cross_factor_metrics.get('max_correlation', 0.0),
                pca_explained_variance=pca_metrics.get('pca_explained_variance', 0.0),
                active_share=active_share
            )

            # Calculate composite crowding score
            metrics.crowding_score = self.calculate_crowding_score(metrics)
            metrics.crowding_level = self.determine_crowding_level(metrics.crowding_score)

            factor_metrics.append(metrics)

            # Check for alerts
            self._check_crowding_alerts(metrics)

        # Store in history
        self.crowding_history.extend(factor_metrics)

        # Log summary
        high_crowding_factors = [m.factor_name for m in factor_metrics if m.crowding_level in [CrowdingLevel.HIGH, CrowdingLevel.EXTREME]]
        if high_crowding_factors:
            logger.warning(f"High crowding detected in factors: {high_crowding_factors}")

        return factor_metrics

    def _check_crowding_alerts(self, metrics: FactorCrowdingMetrics):
        """Check for crowding alerts and generate recommendations"""
        thresholds = self.get_current_thresholds()
        timestamp = datetime.now().isoformat()

        # Check HHI threshold
        if metrics.herfindahl_index > thresholds['concentration_hhi']:
            alert = CrowdingAlert(
                timestamp=timestamp,
                factor_name=metrics.factor_name,
                alert_level=CrowdingLevel.HIGH,
                metric_name="Herfindahl Index",
                current_value=metrics.herfindahl_index,
                threshold_value=thresholds['concentration_hhi'],
                recommended_actions=[
                    "Reduce position concentration in top holdings",
                    "Diversify factor exposure across more securities",
                    "Consider factor rotation to alternative strategies"
                ],
                impact_assessment="High concentration risk - potential for correlated losses"
            )
            self.alerts.append(alert)

        # Check skewness threshold
        if abs(metrics.loading_skewness) > thresholds['factor_skewness']:
            alert = CrowdingAlert(
                timestamp=timestamp,
                factor_name=metrics.factor_name,
                alert_level=CrowdingLevel.MODERATE,
                metric_name="Loading Skewness",
                current_value=abs(metrics.loading_skewness),
                threshold_value=thresholds['factor_skewness'],
                recommended_actions=[
                    "Rebalance factor exposures to reduce skewness",
                    "Add contrarian positions to balance distribution",
                    "Monitor for factor momentum exhaustion"
                ],
                impact_assessment="Skewed factor distribution indicates potential crowding"
            )
            self.alerts.append(alert)

        # Check correlation threshold
        if metrics.max_correlation > thresholds['cross_correlation']:
            alert = CrowdingAlert(
                timestamp=timestamp,
                factor_name=metrics.factor_name,
                alert_level=CrowdingLevel.HIGH,
                metric_name="Cross-Factor Correlation",
                current_value=metrics.max_correlation,
                threshold_value=thresholds['cross_correlation'],
                recommended_actions=[
                    "Reduce exposure to highly correlated factors",
                    "Implement factor orthogonalization",
                    "Consider regime-specific factor allocation"
                ],
                impact_assessment="High factor correlation increases systematic risk"
            )
            self.alerts.append(alert)

    def generate_decrowding_recommendations(self,
                                          factor_metrics: List[FactorCrowdingMetrics],
                                          target_crowding_reduction: float = 0.2) -> Dict[str, Any]:
        """
        Generate specific recommendations for reducing factor crowding

        Args:
            factor_metrics: Current factor crowding metrics
            target_crowding_reduction: Target reduction in crowding score (0.2 = 20%)
        """
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'target_reduction': target_crowding_reduction,
            'factor_recommendations': {},
            'portfolio_actions': [],
            'urgency_level': 'LOW'
        }

        # Identify most crowded factors
        crowded_factors = [m for m in factor_metrics if m.crowding_level in [CrowdingLevel.HIGH, CrowdingLevel.EXTREME]]

        if not crowded_factors:
            recommendations['portfolio_actions'].append("Current factor crowding is within acceptable levels")
            return recommendations

        # Set urgency level
        extreme_factors = [m for m in crowded_factors if m.crowding_level == CrowdingLevel.EXTREME]
        if extreme_factors:
            recommendations['urgency_level'] = 'HIGH'
        elif len(crowded_factors) > 2:
            recommendations['urgency_level'] = 'MEDIUM'

        # Generate factor-specific recommendations
        for factor_metric in crowded_factors:
            factor_recs = []

            if factor_metric.herfindahl_index > self.get_current_thresholds()['concentration_hhi']:
                factor_recs.append({
                    'action': 'Reduce concentration',
                    'target': f"Reduce HHI from {factor_metric.herfindahl_index:.3f} to {self.get_current_thresholds()['concentration_hhi']:.3f}",
                    'method': 'Diversify top holdings, trim largest positions'
                })

            if factor_metric.max_correlation > self.get_current_thresholds()['cross_correlation']:
                factor_recs.append({
                    'action': 'Reduce correlation',
                    'target': f"Reduce max correlation from {factor_metric.max_correlation:.3f} to {self.get_current_thresholds()['cross_correlation']:.3f}",
                    'method': 'Orthogonalize factors, reduce overlapping exposures'
                })

            if abs(factor_metric.loading_skewness) > self.get_current_thresholds()['factor_skewness']:
                factor_recs.append({
                    'action': 'Balance distribution',
                    'target': f"Reduce skewness from {factor_metric.loading_skewness:.3f} to ±{self.get_current_thresholds()['factor_skewness']:.3f}",
                    'method': 'Add contrarian positions, rebalance extremes'
                })

            recommendations['factor_recommendations'][factor_metric.factor_name] = factor_recs

        # Generate portfolio-level actions
        if extreme_factors:
            recommendations['portfolio_actions'].extend([
                "URGENT: Implement immediate de-risking measures",
                "Reduce position sizes in most crowded factors by 30-50%",
                "Activate alternative factor strategies",
                "Consider temporary market-neutral positioning"
            ])
        elif len(crowded_factors) > 1:
            recommendations['portfolio_actions'].extend([
                "Gradual reduction in crowded factor exposures",
                "Increase diversification across factor categories",
                "Monitor for factor rotation opportunities",
                "Implement regime-aware factor allocation"
            ])

        return recommendations

    def export_crowding_report(self, filepath: str) -> bool:
        """Export comprehensive crowding analysis report"""
        try:
            # Get recent metrics
            recent_metrics = self.crowding_history[-20:] if self.crowding_history else []
            recent_alerts = self.alerts[-10:] if self.alerts else []

            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'current_regime': self.current_regime.value,
                'thresholds': {
                    'normal': self.thresholds.normal_regime,
                    'volatile': self.thresholds.volatile_regime,
                    'crisis': self.thresholds.crisis_regime
                },
                'recent_crowding_metrics': [
                    {
                        'timestamp': m.timestamp,
                        'factor_name': m.factor_name,
                        'crowding_level': m.crowding_level.value,
                        'crowding_score': m.crowding_score,
                        'herfindahl_index': m.herfindahl_index,
                        'loading_skewness': m.loading_skewness,
                        'max_correlation': m.max_correlation,
                        'effective_breadth': m.effective_breadth
                    }
                    for m in recent_metrics
                ],
                'recent_alerts': [
                    {
                        'timestamp': a.timestamp,
                        'factor_name': a.factor_name,
                        'alert_level': a.alert_level.value,
                        'metric_name': a.metric_name,
                        'current_value': a.current_value,
                        'threshold_value': a.threshold_value,
                        'recommended_actions': a.recommended_actions[:3]  # Top 3 actions
                    }
                    for a in recent_alerts
                ],
                'summary_statistics': {
                    'total_factors_monitored': len(set(m.factor_name for m in recent_metrics)),
                    'high_crowding_factors': len([m for m in recent_metrics if m.crowding_level in [CrowdingLevel.HIGH, CrowdingLevel.EXTREME]]),
                    'avg_crowding_score': np.mean([m.crowding_score for m in recent_metrics]) if recent_metrics else 0,
                    'max_factor_correlation': max([m.max_correlation for m in recent_metrics]) if recent_metrics else 0
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Crowding analysis report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export crowding report: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Factor Crowding Monitor - Investment Grade Crowding Detection")
    print("=" * 70)

    # Initialize crowding monitor
    monitor = FactorCrowdingMonitor()

    # Generate synthetic factor exposure data
    np.random.seed(42)
    n_stocks = 100

    # Create realistic factor exposures with some crowding
    factor_exposures = {
        'value': np.random.normal(0, 1, n_stocks),
        'momentum': np.random.normal(0, 0.8, n_stocks),
        'quality': np.random.normal(0, 1.2, n_stocks),
        'size': np.random.normal(0, 0.5, n_stocks),
        'volatility': np.random.exponential(0.5, n_stocks)
    }

    # Add some correlation and concentration to simulate crowding
    factor_exposures['momentum'] = 0.7 * factor_exposures['value'] + 0.3 * factor_exposures['momentum']  # High correlation
    factor_exposures['value'][0:10] = 3.0  # Concentration in top stocks

    # Synthetic portfolio and benchmark weights
    portfolio_weights = np.random.dirichlet(np.ones(n_stocks) * 0.5)  # Concentrated portfolio
    benchmark_weights = np.ones(n_stocks) / n_stocks  # Equal weight benchmark

    # Market data for regime assessment
    market_data = {
        'vix': 25.0,  # Moderate volatility
        'market_stress_index': 0.4,
        'correlation_spike': False
    }

    # Monitor factor crowding
    crowding_results = monitor.monitor_factor_crowding(
        factor_exposures=factor_exposures,
        portfolio_weights=portfolio_weights,
        benchmark_weights=benchmark_weights,
        market_data=market_data
    )

    print(f"Factor Crowding Analysis Results:")
    print(f"Market Regime: {monitor.current_regime.value}")
    print(f"Factors Analyzed: {len(crowding_results)}")

    for result in crowding_results:
        print(f"\n{result.factor_name}:")
        print(f"  Crowding Level: {result.crowding_level.value}")
        print(f"  Crowding Score: {result.crowding_score:.1f}/100")
        print(f"  HHI: {result.herfindahl_index:.3f}")
        print(f"  Max Correlation: {result.max_correlation:.3f}")
        print(f"  Effective Breadth: {result.effective_breadth:.1f}")

    # Generate decrowding recommendations
    recommendations = monitor.generate_decrowding_recommendations(crowding_results)
    print(f"\nDecrowding Recommendations:")
    print(f"Urgency Level: {recommendations['urgency_level']}")
    print(f"Crowded Factors: {len(recommendations['factor_recommendations'])}")

    for factor, recs in recommendations['factor_recommendations'].items():
        print(f"\n{factor}:")
        for rec in recs[:2]:  # Show top 2 recommendations
            print(f"  - {rec['action']}: {rec['method']}")

    # Export crowding report
    monitor.export_crowding_report("factor_crowding_analysis_report.json")
    print(f"\nCrowding analysis report exported successfully!")