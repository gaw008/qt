#!/usr/bin/env python3
"""
Scoring Orchestrator - Investment Grade Multi-Factor Scoring System

This module provides the main orchestration layer for multi-factor scoring,
coordinating between specialized scoring services.

Features:
- Orchestrates factor calculation, normalization, and correlation analysis
- Manages dynamic weight optimization
- Generates composite scores and trading signals
- Provides explainable scoring results and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging

# Import specialized services
from scoring_services import (
    FactorCalculationService,
    FactorNormalizationService,
    CorrelationAnalysisService,
    WeightOptimizationService,
    NormalizationConfig
)

logger = logging.getLogger(__name__)


@dataclass
class FactorWeights:
    """Configuration for factor weights and parameters."""
    # Factor weights (should sum to 1.0)
    valuation: float = 0.25
    volume: float = 0.15
    momentum: float = 0.20
    technical: float = 0.25
    market_sentiment: float = 0.15

    # Dynamic adjustment parameters
    enable_dynamic_weights: bool = True
    weight_adjustment_period: int = 60  # Days to look back
    min_weight: float = 0.05
    max_weight: float = 0.50

    # Correlation parameters
    high_correlation_threshold: float = 0.8
    redundancy_penalty: float = 0.1

    def __post_init__(self):
        """Validate and normalize weights."""
        weights = [self.valuation, self.volume, self.momentum, self.technical, self.market_sentiment]
        total = sum(weights)

        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total:.4f}, normalizing to 1.0")
            factor = 1.0 / total
            self.valuation *= factor
            self.volume *= factor
            self.momentum *= factor
            self.technical *= factor
            self.market_sentiment *= factor

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'valuation': self.valuation,
            'volume': self.volume,
            'momentum': self.momentum,
            'technical': self.technical,
            'market_sentiment': self.market_sentiment
        }


@dataclass
class ScoringResult:
    """Comprehensive scoring result."""
    scores: pd.DataFrame
    factor_contributions: pd.DataFrame
    factor_correlations: pd.DataFrame
    weights_used: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ScoringOrchestrator:
    """
    Main orchestrator for investment-grade multi-factor scoring system.

    Coordinates between specialized services to provide comprehensive
    factor-based scoring and signal generation.
    """

    def __init__(self,
                 factor_weights: Optional[FactorWeights] = None,
                 normalization_config: Optional[NormalizationConfig] = None):
        """
        Initialize the scoring orchestrator.

        Args:
            factor_weights: Factor weight configuration
            normalization_config: Normalization configuration
        """
        self.factor_weights = factor_weights or FactorWeights()
        self.normalization_config = normalization_config or NormalizationConfig()

        # Initialize specialized services
        self.factor_calculator = FactorCalculationService()
        self.factor_normalizer = FactorNormalizationService(self.normalization_config)
        self.correlation_analyzer = CorrelationAnalysisService(
            self.factor_weights.high_correlation_threshold
        )
        self.weight_optimizer = WeightOptimizationService(
            self.factor_weights.min_weight,
            self.factor_weights.max_weight,
            self.factor_weights.redundancy_penalty
        )

        # State tracking
        self.scoring_history: List[ScoringResult] = []
        self.performance_history: Dict[str, List[float]] = {}

        logger.info("Scoring Orchestrator initialized with investment-grade multi-factor system")

    def calculate_composite_scores(self,
                                 data: Dict[str, pd.DataFrame],
                                 market_data: Optional[Dict[str, Any]] = None,
                                 sector_mapping: Optional[Dict[str, str]] = None,
                                 custom_weights: Optional[Dict[str, float]] = None) -> ScoringResult:
        """
        Calculate comprehensive composite scores for all symbols.

        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            market_data: Optional market-level data
            sector_mapping: Optional sector classification
            custom_weights: Optional custom factor weights

        Returns:
            ScoringResult with comprehensive scoring analysis
        """
        logger.info(f"Starting composite scoring for {len(data)} symbols")

        # Step 1: Calculate individual factor scores
        factor_scores = self._calculate_individual_factors(data, market_data)

        if factor_scores.empty:
            return self._create_empty_result()

        # Step 2: Normalize factor scores
        normalized_factors = self._normalize_factors(factor_scores, sector_mapping)

        # Step 3: Analyze factor correlations
        correlation_analysis = self._analyze_correlations(normalized_factors)

        # Step 4: Optimize factor weights
        optimized_weights = self._optimize_weights(custom_weights, correlation_analysis)

        # Step 5: Calculate composite scores
        composite_result = self._calculate_weighted_scores(normalized_factors, optimized_weights)

        # Step 6: Generate final scoring result
        result = self._compile_scoring_result(
            composite_result,
            normalized_factors,
            correlation_analysis,
            optimized_weights
        )

        # Store in history
        self._update_history(result)

        logger.info(f"Composite scoring completed for {len(result.scores)} symbols")
        return result

    def _calculate_individual_factors(self,
                                    data: Dict[str, pd.DataFrame],
                                    market_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Calculate individual factor scores for all symbols."""
        logger.debug("Calculating individual factor scores")

        try:
            # Enhanced kwargs for factor calculation
            calculation_kwargs = {
                'market_data': market_data,
                'all_data': data  # For market sentiment calculations
            }

            factor_df = self.factor_calculator.calculate_all_factors(data, **calculation_kwargs)

            if factor_df.empty:
                logger.warning("No factor scores calculated")
                return pd.DataFrame()

            logger.debug(f"Calculated {len(factor_df.columns)-1} factors for {len(factor_df)} symbols")
            return factor_df

        except Exception as e:
            logger.error(f"Error calculating individual factors: {e}")
            return pd.DataFrame()

    def _normalize_factors(self,
                         factor_scores: pd.DataFrame,
                         sector_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Normalize factor scores and apply sector neutrality if configured."""
        logger.debug("Normalizing factor scores")

        try:
            # Basic normalization
            normalized = self.factor_normalizer.normalize_factors(factor_scores)

            # Apply sector neutrality if configured and mapping provided
            if sector_mapping:
                normalized = self.factor_normalizer.apply_sector_neutrality(normalized, sector_mapping)

            # Replace infinite values and fill NaN
            numeric_cols = normalized.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'symbol':
                    normalized[col] = normalized[col].replace([np.inf, -np.inf], np.nan).fillna(0)

            logger.debug("Factor normalization completed")
            return normalized

        except Exception as e:
            logger.error(f"Error normalizing factors: {e}")
            return factor_scores

    def _analyze_correlations(self, normalized_factors: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factor correlations and detect redundancy."""
        logger.debug("Analyzing factor correlations")

        try:
            # Calculate correlation matrix
            correlation_matrix = self.correlation_analyzer.calculate_factor_correlations(normalized_factors)

            # Detect redundant factors
            redundant_pairs = self.correlation_analyzer.detect_redundant_factors(correlation_matrix)

            # Calculate factor loadings
            factor_loadings = self.correlation_analyzer.calculate_factor_loadings(normalized_factors)

            analysis = {
                'correlation_matrix': correlation_matrix,
                'redundant_pairs': redundant_pairs,
                'factor_loadings': factor_loadings,
                'num_redundant_pairs': len(redundant_pairs)
            }

            logger.debug(f"Found {len(redundant_pairs)} highly correlated factor pairs")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {
                'correlation_matrix': pd.DataFrame(),
                'redundant_pairs': [],
                'factor_loadings': {},
                'num_redundant_pairs': 0
            }

    def _optimize_weights(self,
                        custom_weights: Optional[Dict[str, float]],
                        correlation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Optimize factor weights based on configuration and analysis."""
        logger.debug("Optimizing factor weights")

        try:
            # Start with base weights
            if custom_weights:
                base_weights = custom_weights.copy()
            else:
                base_weights = self.factor_weights.to_dict()

            # Skip optimization if dynamic weights disabled
            if not self.factor_weights.enable_dynamic_weights:
                return base_weights

            # Get correlation data
            correlation_matrix = correlation_analysis.get('correlation_matrix', pd.DataFrame())
            redundant_pairs = correlation_analysis.get('redundant_pairs', [])

            # Get historical performance if available
            historical_performance = self._get_historical_performance()

            # Optimize weights
            optimized_weights = self.weight_optimizer.optimize_weights(
                base_weights,
                correlation_matrix,
                redundant_pairs,
                historical_performance
            )

            logger.debug("Weight optimization completed")
            return optimized_weights

        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
            return self.factor_weights.to_dict()

    def _calculate_weighted_scores(self,
                                 normalized_factors: pd.DataFrame,
                                 weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate weighted composite scores."""
        logger.debug("Calculating weighted composite scores")

        try:
            if normalized_factors.empty:
                return {'scores': pd.DataFrame(), 'contributions': pd.DataFrame()}

            # Initialize result DataFrames
            composite_scores = pd.DataFrame({'symbol': normalized_factors['symbol']})
            factor_contributions = pd.DataFrame({'symbol': normalized_factors['symbol']})

            composite_scores['composite_score'] = 0.0

            # Calculate weighted contributions
            score_columns = [col for col in normalized_factors.columns if col.endswith('_score')]

            for score_col in score_columns:
                # Extract factor name from column name
                factor_name = score_col.replace('_score', '')

                if factor_name in weights:
                    weight = weights[factor_name]
                    contribution = normalized_factors[score_col] * weight
                    composite_scores['composite_score'] += contribution
                    factor_contributions[factor_name] = contribution
                else:
                    factor_contributions[factor_name] = 0.0

            # Calculate rankings and percentiles
            composite_scores['rank'] = composite_scores['composite_score'].rank(ascending=False)
            composite_scores['percentile'] = composite_scores['composite_score'].rank(pct=True)

            logger.debug(f"Calculated composite scores for {len(composite_scores)} symbols")

            return {
                'scores': composite_scores,
                'contributions': factor_contributions
            }

        except Exception as e:
            logger.error(f"Error calculating weighted scores: {e}")
            return {'scores': pd.DataFrame(), 'contributions': pd.DataFrame()}

    def _compile_scoring_result(self,
                              composite_result: Dict[str, Any],
                              normalized_factors: pd.DataFrame,
                              correlation_analysis: Dict[str, Any],
                              weights: Dict[str, float]) -> ScoringResult:
        """Compile comprehensive scoring result."""
        try:
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(composite_result['scores'])

            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'num_symbols': len(composite_result['scores']),
                'num_factors': len([col for col in normalized_factors.columns if col.endswith('_score')]),
                'redundant_pairs': correlation_analysis.get('redundant_pairs', []),
                'normalization_method': self.normalization_config.method,
                'sector_neutral': self.normalization_config.enable_sector_neutrality,
                'dynamic_weights_enabled': self.factor_weights.enable_dynamic_weights
            }

            return ScoringResult(
                scores=composite_result['scores'],
                factor_contributions=composite_result['contributions'],
                factor_correlations=correlation_analysis.get('correlation_matrix', pd.DataFrame()),
                weights_used=weights,
                metadata=metadata,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            logger.error(f"Error compiling scoring result: {e}")
            return self._create_empty_result()

    def _calculate_performance_metrics(self, scores: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the scoring system."""
        if scores.empty or 'composite_score' not in scores.columns:
            return {}

        try:
            score_series = scores['composite_score']

            return {
                'mean_score': float(score_series.mean()),
                'std_score': float(score_series.std()),
                'min_score': float(score_series.min()),
                'max_score': float(score_series.max()),
                'median_score': float(score_series.median()),
                'skewness': float(score_series.skew()) if len(score_series) > 0 else 0.0,
                'kurtosis': float(score_series.kurtosis()) if len(score_series) > 0 else 0.0,
                'score_range': float(score_series.max() - score_series.min()),
                'iqr': float(score_series.quantile(0.75) - score_series.quantile(0.25))
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _get_historical_performance(self) -> Optional[Dict[str, float]]:
        """Get historical performance data for weight optimization."""
        if not self.performance_history:
            return None

        try:
            # Simple average of recent performance
            performance = {}
            for factor, history in self.performance_history.items():
                if history:
                    performance[factor] = np.mean(history[-self.factor_weights.weight_adjustment_period:])

            return performance if performance else None

        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return None

    def _update_history(self, result: ScoringResult):
        """Update scoring and performance history."""
        try:
            # Add to scoring history
            self.scoring_history.append(result)

            # Keep only recent history
            if len(self.scoring_history) > self.factor_weights.weight_adjustment_period:
                self.scoring_history = self.scoring_history[-self.factor_weights.weight_adjustment_period:]

            # Update performance history (placeholder for actual performance tracking)
            # In practice, this would track factor performance vs returns
            for factor in result.weights_used:
                if factor not in self.performance_history:
                    self.performance_history[factor] = []

                # Placeholder performance score
                perf_score = 1.0  # Would be actual performance calculation
                self.performance_history[factor].append(perf_score)

                # Keep only recent performance
                if len(self.performance_history[factor]) > self.factor_weights.weight_adjustment_period:
                    self.performance_history[factor] = self.performance_history[factor][-self.factor_weights.weight_adjustment_period:]

        except Exception as e:
            logger.error(f"Error updating history: {e}")

    def _create_empty_result(self) -> ScoringResult:
        """Create empty scoring result for error cases."""
        return ScoringResult(
            scores=pd.DataFrame(),
            factor_contributions=pd.DataFrame(),
            factor_correlations=pd.DataFrame(),
            weights_used={},
            metadata={'error': 'No data available'},
            performance_metrics={}
        )

    def generate_trading_signals(self,
                               result: ScoringResult,
                               buy_threshold: float = 0.7,
                               sell_threshold: float = 0.3,
                               max_positions: int = 10) -> pd.DataFrame:
        """
        Generate trading signals based on composite scores.

        Args:
            result: ScoringResult from calculate_composite_scores
            buy_threshold: Percentile threshold for buy signals
            sell_threshold: Percentile threshold for sell signals
            max_positions: Maximum number of positions

        Returns:
            DataFrame with trading signals
        """
        if result.scores.empty:
            return pd.DataFrame()

        try:
            signals = result.scores.copy()
            signals['signal'] = 0

            # Buy signals (top performers)
            buy_mask = signals['percentile'] >= buy_threshold
            top_buys = signals[buy_mask].nlargest(max_positions, 'composite_score')
            signals.loc[signals['symbol'].isin(top_buys['symbol']), 'signal'] = 1

            # Sell signals (bottom performers) - only if we have short capability
            sell_mask = signals['percentile'] <= sell_threshold
            bottom_sells = signals[sell_mask].nsmallest(max_positions // 2, 'composite_score')  # Fewer short positions
            signals.loc[signals['symbol'].isin(bottom_sells['symbol']), 'signal'] = -1

            # Add signal strength
            signals['signal_strength'] = signals['percentile'].apply(
                lambda x: abs(x - 0.5) * 2  # Scale to 0-1 where 1 is strongest
            )

            return signals[['symbol', 'composite_score', 'rank', 'percentile', 'signal', 'signal_strength']]

        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return pd.DataFrame()

    def explain_scores(self, result: ScoringResult, top_n: int = 10) -> Dict[str, Any]:
        """
        Generate explanation of scoring results.

        Args:
            result: ScoringResult from calculate_composite_scores
            top_n: Number of top/bottom stocks to explain

        Returns:
            Dictionary with explanatory analysis
        """
        if result.scores.empty:
            return {}

        try:
            analysis = {}

            # Top and bottom performers
            top_stocks = result.scores.nlargest(top_n, 'composite_score')
            bottom_stocks = result.scores.nsmallest(top_n, 'composite_score')

            analysis['top_stocks'] = top_stocks[['symbol', 'composite_score', 'rank', 'percentile']].to_dict('records')
            analysis['bottom_stocks'] = bottom_stocks[['symbol', 'composite_score', 'rank', 'percentile']].to_dict('records')

            # Factor importance
            analysis['factor_weights'] = result.weights_used

            # Performance metrics
            analysis['performance_metrics'] = result.performance_metrics

            # Factor correlation summary
            if not result.factor_correlations.empty:
                avg_correlations = {}
                for col in result.factor_correlations.columns:
                    other_cols = [c for c in result.factor_correlations.columns if c != col]
                    if other_cols:
                        avg_corr = result.factor_correlations.loc[col, other_cols].abs().mean()
                        avg_correlations[col] = float(avg_corr) if not np.isnan(avg_corr) else 0.0

                analysis['average_factor_correlations'] = avg_correlations

            # Factor contribution analysis for top stocks
            if not result.factor_contributions.empty:
                top_contributions = {}
                for _, row in top_stocks.iterrows():
                    symbol = row['symbol']
                    contrib_row = result.factor_contributions[
                        result.factor_contributions['symbol'] == symbol
                    ]
                    if not contrib_row.empty:
                        contrib_dict = {}
                        for factor in result.weights_used.keys():
                            if factor in contrib_row.columns:
                                contrib_dict[factor] = float(contrib_row[factor].iloc[0])
                        top_contributions[symbol] = contrib_dict

                analysis['top_stock_contributions'] = top_contributions

            # Metadata
            analysis['metadata'] = result.metadata

            return analysis

        except Exception as e:
            logger.error(f"Error explaining scores: {e}")
            return {}

    def save_configuration(self, filepath: str) -> bool:
        """Save current configuration to JSON file."""
        try:
            config = {
                'factor_weights': {
                    'valuation': self.factor_weights.valuation,
                    'volume': self.factor_weights.volume,
                    'momentum': self.factor_weights.momentum,
                    'technical': self.factor_weights.technical,
                    'market_sentiment': self.factor_weights.market_sentiment,
                    'enable_dynamic_weights': self.factor_weights.enable_dynamic_weights,
                    'weight_adjustment_period': self.factor_weights.weight_adjustment_period,
                    'min_weight': self.factor_weights.min_weight,
                    'max_weight': self.factor_weights.max_weight,
                    'high_correlation_threshold': self.factor_weights.high_correlation_threshold,
                    'redundancy_penalty': self.factor_weights.redundancy_penalty
                },
                'normalization_config': {
                    'method': self.normalization_config.method,
                    'winsorize_percentile': self.normalization_config.winsorize_percentile,
                    'enable_sector_neutrality': self.normalization_config.enable_sector_neutrality,
                    'outlier_threshold': self.normalization_config.outlier_threshold
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    @classmethod
    def load_configuration(cls, filepath: str) -> 'ScoringOrchestrator':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)

            factor_weights = FactorWeights(**config.get('factor_weights', {}))
            normalization_config = NormalizationConfig(**config.get('normalization_config', {}))

            return cls(factor_weights, normalization_config)

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return cls()  # Return default configuration