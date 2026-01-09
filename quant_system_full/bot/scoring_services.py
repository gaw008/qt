#!/usr/bin/env python3
"""
Scoring Services - Refactored Multi-Factor Scoring Components

This module contains specialized services for factor scoring,
following Single Responsibility Principle.

Services:
- FactorCalculationService: Individual factor computation
- FactorNormalizationService: Data normalization and outlier handling
- CorrelationAnalysisService: Factor correlation and redundancy detection
- WeightOptimizationService: Dynamic weight optimization
- ScoringOrchestrator: Main coordination service
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class FactorScore:
    """Individual factor score result."""
    symbol: str
    factor_name: str
    raw_score: float
    normalized_score: float
    percentile: float
    metadata: Dict[str, Any] = None


@dataclass
class NormalizationConfig:
    """Configuration for factor normalization."""
    method: str = "robust"  # "standard", "robust", "winsorize"
    winsorize_percentile: float = 0.05
    enable_sector_neutrality: bool = False
    outlier_threshold: float = 3.0


class FactorCalculationStrategy(ABC):
    """Abstract base class for factor calculation strategies."""

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """Calculate factor score for given data."""
        pass

    @abstractmethod
    def get_factor_name(self) -> str:
        """Get the name of this factor."""
        pass


class FactorCalculationService:
    """
    Service for calculating individual factor scores.
    Supports multiple factor calculation strategies.
    """

    def __init__(self):
        self.strategies: Dict[str, FactorCalculationStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default factor calculation strategies."""
        # Import factor modules with fallbacks
        try:
            self._try_import_factor_modules()
        except ImportError as e:
            logger.warning(f"Could not import all factor modules: {e}")

    def _try_import_factor_modules(self):
        """Try to import factor calculation modules."""
        try:
            from .factors.valuation import valuation_score
            from .factors.volume_factors import volume_features
            from .factors.momentum_factors import momentum_features
            from .factors.technical_factors import technical_features
            from .factors.market_factors import market_sentiment_features

            # Register strategies
            self.strategies['valuation'] = ValuationFactorStrategy(valuation_score)
            self.strategies['volume'] = VolumeFactorStrategy(volume_features)
            self.strategies['momentum'] = MomentumFactorStrategy(momentum_features)
            self.strategies['technical'] = TechnicalFactorStrategy(technical_features)
            self.strategies['market_sentiment'] = MarketSentimentFactorStrategy(market_sentiment_features)

        except ImportError:
            # Fallback strategies
            logger.warning("Using fallback factor strategies")
            self.strategies = {
                'valuation': FallbackFactorStrategy('valuation'),
                'volume': FallbackFactorStrategy('volume'),
                'momentum': FallbackFactorStrategy('momentum'),
                'technical': FallbackFactorStrategy('technical'),
                'market_sentiment': FallbackFactorStrategy('market_sentiment')
            }

    def register_strategy(self, name: str, strategy: FactorCalculationStrategy):
        """Register a new factor calculation strategy."""
        self.strategies[name] = strategy

    def calculate_factor(self, factor_name: str, symbol: str, data: pd.DataFrame, **kwargs) -> FactorScore:
        """
        Calculate a single factor score.

        Args:
            factor_name: Name of the factor to calculate
            symbol: Stock symbol
            data: OHLCV data
            **kwargs: Additional parameters for factor calculation

        Returns:
            FactorScore object with calculation result
        """
        if factor_name not in self.strategies:
            logger.warning(f"Unknown factor: {factor_name}")
            return FactorScore(symbol, factor_name, 0.0, 0.0, 0.5)

        try:
            strategy = self.strategies[factor_name]
            raw_score = strategy.calculate(data, **kwargs)

            return FactorScore(
                symbol=symbol,
                factor_name=factor_name,
                raw_score=raw_score,
                normalized_score=raw_score,  # Will be normalized later
                percentile=0.5,  # Will be calculated later
                metadata={'calculation_timestamp': pd.Timestamp.now().isoformat()}
            )

        except Exception as e:
            logger.error(f"Error calculating {factor_name} for {symbol}: {e}")
            return FactorScore(symbol, factor_name, 0.0, 0.0, 0.5)

    def calculate_all_factors(self, symbols_data: Dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Calculate all factors for all symbols.

        Args:
            symbols_data: Dictionary of symbol -> OHLCV DataFrame
            **kwargs: Additional parameters

        Returns:
            DataFrame with factor scores for all symbols
        """
        all_scores = []

        for symbol, data in symbols_data.items():
            if data is None or data.empty:
                continue

            symbol_scores = {'symbol': symbol}

            for factor_name in self.strategies.keys():
                try:
                    factor_score = self.calculate_factor(factor_name, symbol, data, **kwargs)
                    symbol_scores[f'{factor_name}_score'] = factor_score.raw_score
                except Exception as e:
                    logger.warning(f"Error calculating {factor_name} for {symbol}: {e}")
                    symbol_scores[f'{factor_name}_score'] = 0.0

            all_scores.append(symbol_scores)

        return pd.DataFrame(all_scores) if all_scores else pd.DataFrame()


class FactorNormalizationService:
    """
    Service for normalizing factor scores and handling outliers.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()

    def normalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize factor scores using configured method.

        Args:
            factor_df: DataFrame with raw factor scores

        Returns:
            DataFrame with normalized factor scores
        """
        if factor_df.empty:
            return factor_df

        result = factor_df.copy()

        # Get numeric columns only
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'symbol']

        for col in numeric_cols:
            result[col] = self._normalize_column(result[col], self.config.method)

        return result

    def _normalize_column(self, series: pd.Series, method: str) -> pd.Series:
        """Normalize a single column using the specified method."""
        if series.empty or series.isna().all():
            return series

        if method == "standard":
            return self._standard_normalize(series)
        elif method == "robust":
            return self._robust_normalize(series)
        elif method == "winsorize":
            return self._winsorize_normalize(series)
        else:
            logger.warning(f"Unknown normalization method: {method}, using robust")
            return self._robust_normalize(series)

    def _standard_normalize(self, series: pd.Series) -> pd.Series:
        """Standard z-score normalization."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    def _robust_normalize(self, series: pd.Series) -> pd.Series:
        """Robust normalization using median and MAD."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(0, index=series.index)
        return (series - median) / (mad * 1.4826)  # 1.4826 for normal consistency

    def _winsorize_normalize(self, series: pd.Series) -> pd.Series:
        """Winsorize outliers then standardize."""
        lower_bound = series.quantile(self.config.winsorize_percentile)
        upper_bound = series.quantile(1 - self.config.winsorize_percentile)
        winsorized = series.clip(lower_bound, upper_bound)
        return self._standard_normalize(winsorized)

    def apply_sector_neutrality(self, scores_df: pd.DataFrame,
                              sector_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Apply sector neutrality to scores.

        Args:
            scores_df: DataFrame with scores
            sector_mapping: Mapping of symbol -> sector

        Returns:
            Sector-neutral scores
        """
        if not self.config.enable_sector_neutrality or not sector_mapping:
            return scores_df

        result = scores_df.copy()

        # Add sector column
        result['sector'] = result['symbol'].map(sector_mapping)

        # Neutralize scores within each sector
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['symbol', 'sector']]

        for col in numeric_cols:
            if result['sector'].notna().any():
                # Within-sector z-score
                result[col] = result.groupby('sector')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-9) if x.std() > 0 else 0
                )

        result = result.drop('sector', axis=1, errors='ignore')
        return result


class CorrelationAnalysisService:
    """
    Service for analyzing factor correlations and detecting redundancy.
    """

    def __init__(self, high_correlation_threshold: float = 0.8):
        self.high_correlation_threshold = high_correlation_threshold

    def calculate_factor_correlations(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.

        Args:
            factor_df: DataFrame with factor scores

        Returns:
            Factor correlation matrix
        """
        if factor_df.empty:
            return pd.DataFrame()

        numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'symbol']

        if len(numeric_cols) == 0:
            return pd.DataFrame()

        factor_data = factor_df[numeric_cols]
        return factor_data.corr()

    def detect_redundant_factors(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Detect pairs of factors with high correlation.

        Args:
            correlation_matrix: Factor correlation matrix

        Returns:
            List of (factor1, factor2, correlation) tuples
        """
        if correlation_matrix.empty:
            return []

        redundant_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                factor1 = correlation_matrix.columns[i]
                factor2 = correlation_matrix.columns[j]
                corr = abs(correlation_matrix.iloc[i, j])

                if not np.isnan(corr) and corr >= self.high_correlation_threshold:
                    redundant_pairs.append((factor1, factor2, corr))

        return redundant_pairs

    def calculate_factor_loadings(self, factor_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate factor loading statistics.

        Args:
            factor_df: DataFrame with factor scores

        Returns:
            Dictionary with factor loading statistics
        """
        if factor_df.empty:
            return {}

        numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'symbol']

        loadings = {}
        for col in numeric_cols:
            series = factor_df[col]
            loadings[col] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'skewness': float(series.skew()) if len(series) > 0 else 0.0,
                'kurtosis': float(series.kurtosis()) if len(series) > 0 else 0.0,
                'non_zero_ratio': float((series != 0).sum() / len(series)) if len(series) > 0 else 0.0
            }

        return loadings


class WeightOptimizationService:
    """
    Service for optimizing factor weights based on historical performance.
    """

    def __init__(self,
                 min_weight: float = 0.05,
                 max_weight: float = 0.50,
                 redundancy_penalty: float = 0.1):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.redundancy_penalty = redundancy_penalty

    def optimize_weights(self,
                       base_weights: Dict[str, float],
                       correlation_matrix: pd.DataFrame,
                       redundant_pairs: List[Tuple[str, str, float]],
                       historical_performance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize factor weights considering correlations and performance.

        Args:
            base_weights: Initial factor weights
            correlation_matrix: Factor correlation matrix
            redundant_pairs: List of highly correlated factor pairs
            historical_performance: Optional historical performance data

        Returns:
            Optimized weights dictionary
        """
        optimized_weights = base_weights.copy()

        # Apply correlation penalty
        optimized_weights = self._apply_correlation_penalty(optimized_weights, redundant_pairs)

        # Apply performance adjustment if available
        if historical_performance:
            optimized_weights = self._apply_performance_adjustment(optimized_weights, historical_performance)

        # Ensure weight constraints
        optimized_weights = self._enforce_weight_constraints(optimized_weights)

        return optimized_weights

    def _apply_correlation_penalty(self,
                                 weights: Dict[str, float],
                                 redundant_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """Apply penalty for highly correlated factors."""
        adjusted_weights = weights.copy()

        for factor1, factor2, corr in redundant_pairs:
            penalty = self.redundancy_penalty * corr

            # Apply penalty to both factors proportionally
            factor1_key = self._find_weight_key(factor1, adjusted_weights)
            factor2_key = self._find_weight_key(factor2, adjusted_weights)

            if factor1_key:
                adjusted_weights[factor1_key] *= (1 - penalty / 2)
            if factor2_key:
                adjusted_weights[factor2_key] *= (1 - penalty / 2)

        return adjusted_weights

    def _apply_performance_adjustment(self,
                                    weights: Dict[str, float],
                                    performance: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on historical performance."""
        adjusted_weights = weights.copy()

        # Simple performance-based adjustment
        # In practice, this could use more sophisticated optimization
        for factor, weight in adjusted_weights.items():
            perf_score = performance.get(factor, 1.0)
            # Adjust weight by performance (bounded adjustment)
            adjustment = min(max(perf_score, 0.5), 2.0)
            adjusted_weights[factor] = weight * adjustment

        return adjusted_weights

    def _enforce_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce minimum and maximum weight constraints."""
        # Apply bounds
        for factor in weights:
            weights[factor] = np.clip(weights[factor], self.min_weight, self.max_weight)

        # Renormalize to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for factor in weights:
                weights[factor] /= total_weight

        return weights

    def _find_weight_key(self, factor_name: str, weights: Dict[str, float]) -> Optional[str]:
        """Find the corresponding weight key for a factor name."""
        # Direct match
        if factor_name in weights:
            return factor_name

        # Try with '_score' suffix removed
        base_name = factor_name.replace('_score', '')
        if base_name in weights:
            return base_name

        # Try adding '_weight' suffix
        weight_key = f"{base_name}_weight"
        if weight_key in weights:
            return weight_key

        return None


# Factor Strategy Implementations

class ValuationFactorStrategy(FactorCalculationStrategy):
    """Strategy for valuation factor calculation."""

    def __init__(self, calculation_func):
        self.calculation_func = calculation_func

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        try:
            if 'market_cap' in data.columns:
                result = self.calculation_func(data.iloc[[-1]])
                return result['ValuationScore'].iloc[0] if not result.empty else 0.0
            return 0.0
        except Exception:
            return 0.0

    def get_factor_name(self) -> str:
        return "valuation"


class VolumeFactorStrategy(FactorCalculationStrategy):
    """Strategy for volume factor calculation."""

    def __init__(self, calculation_func):
        self.calculation_func = calculation_func

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        try:
            result = self.calculation_func(data)
            if not result.empty and 'vol_score' in result.columns:
                return result['vol_score'].iloc[-1]
            return 0.0
        except Exception:
            return 0.0

    def get_factor_name(self) -> str:
        return "volume"


class MomentumFactorStrategy(FactorCalculationStrategy):
    """Strategy for momentum factor calculation."""

    def __init__(self, calculation_func):
        self.calculation_func = calculation_func

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        try:
            result = self.calculation_func(data)
            if not result.empty and 'momentum_score' in result.columns:
                return result['momentum_score'].iloc[-1]
            return 0.0
        except Exception:
            return 0.0

    def get_factor_name(self) -> str:
        return "momentum"


class TechnicalFactorStrategy(FactorCalculationStrategy):
    """Strategy for technical factor calculation."""

    def __init__(self, calculation_func):
        self.calculation_func = calculation_func

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        try:
            result = self.calculation_func(data)
            if not result.empty and 'technical_score' in result.columns:
                return result['technical_score'].iloc[-1]
            return 0.0
        except Exception:
            return 0.0

    def get_factor_name(self) -> str:
        return "technical"


class MarketSentimentFactorStrategy(FactorCalculationStrategy):
    """Strategy for market sentiment factor calculation."""

    def __init__(self, calculation_func):
        self.calculation_func = calculation_func

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        try:
            market_data = kwargs.get('market_data', {})
            symbol = kwargs.get('symbol', '')
            all_data = kwargs.get('all_data', {})

            if market_data and all_data:
                result = self.calculation_func(
                    all_data,
                    volume_data=market_data.get('volume_data'),
                    vix_data=market_data.get('vix_data'),
                    benchmark_data=market_data.get('benchmark_data'),
                    symbol=symbol
                )
                if not result.empty and 'market_sentiment_score' in result.columns:
                    return result['market_sentiment_score'].iloc[-1]
            return 0.0
        except Exception:
            return 0.0

    def get_factor_name(self) -> str:
        return "market_sentiment"


class FallbackFactorStrategy(FactorCalculationStrategy):
    """Fallback strategy when factor modules are not available."""

    def __init__(self, factor_name: str):
        self.factor_name = factor_name

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        # Simple fallback calculation based on price momentum
        try:
            if len(data) < 2:
                return 0.0

            price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
            return float(np.clip(price_change * 10, -3, 3))  # Bounded between -3 and 3
        except Exception:
            return 0.0

    def get_factor_name(self) -> str:
        return self.factor_name