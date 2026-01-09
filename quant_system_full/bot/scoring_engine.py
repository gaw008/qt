"""
Comprehensive multi-factor scoring engine for quantitative trading system.

This module provides:
- Multi-factor weight configuration and dynamic adjustment
- Factor normalization and composite scoring algorithms  
- Explainable scoring results analysis
- Factor correlation analysis and redundancy detection

Integrates all factor modules: valuation, volume, momentum, technical, and market sentiment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import warnings

# Optional imports with fallbacks
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, some advanced features disabled")

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available, some normalization features disabled")

# Import factor modules
try:
    from .factors.valuation import valuation_score
    from .factors.volume_factors import volume_features, cross_section_volume_score
    from .factors.momentum_factors import momentum_features, cross_section_momentum_score
    from .factors.technical_factors import technical_features, cross_section_technical_score
    from .factors.market_factors import market_sentiment_features, cross_section_market_score
    HAS_FACTORS = True
except ImportError as e:
    try:
        # Fallback to absolute imports
        from bot.factors.valuation import valuation_score
        from bot.factors.volume_factors import volume_features, cross_section_volume_score
        from bot.factors.momentum_factors import momentum_features, cross_section_momentum_score
        from bot.factors.technical_factors import technical_features, cross_section_technical_score
        from bot.factors.market_factors import market_sentiment_features, cross_section_market_score
        HAS_FACTORS = True
    except ImportError as e2:
        HAS_FACTORS = False
        warnings.warn(f"Could not import factor modules: {e2}")


@dataclass
class FactorWeights:
    """Configuration class for factor weights and parameters."""
    
    # Factor weights (should sum to 1.0)
    valuation_weight: float = 0.25
    volume_weight: float = 0.15
    momentum_weight: float = 0.20
    technical_weight: float = 0.25
    market_sentiment_weight: float = 0.15
    
    # Dynamic adjustment parameters
    enable_dynamic_weights: bool = True
    weight_adjustment_period: int = 60  # Days to look back for weight optimization
    min_weight: float = 0.05  # Minimum weight for any factor
    max_weight: float = 0.50  # Maximum weight for any factor
    
    # Correlation thresholds
    high_correlation_threshold: float = 0.8
    redundancy_penalty: float = 0.1
    
    # Scoring parameters
    outlier_method: str = "robust"  # "standard", "robust", "winsorize"
    winsorize_percentile: float = 0.05
    enable_sector_neutrality: bool = False
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total_weight = (self.valuation_weight + self.volume_weight + 
                       self.momentum_weight + self.technical_weight + 
                       self.market_sentiment_weight)
        
        if abs(total_weight - 1.0) > 1e-6:
            warnings.warn(f"Weights sum to {total_weight:.4f}, normalizing to 1.0")
            # Normalize weights
            factor = 1.0 / total_weight
            self.valuation_weight *= factor
            self.volume_weight *= factor
            self.momentum_weight *= factor
            self.technical_weight *= factor
            self.market_sentiment_weight *= factor


@dataclass
class ScoringResult:
    """Results from the scoring engine."""
    
    scores: pd.DataFrame
    factor_contributions: pd.DataFrame
    factor_correlations: pd.DataFrame
    weights_used: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiFactorScoringEngine:
    """
    Comprehensive scoring engine that combines multiple factor modules.
    """
    
    def __init__(self, weights: Optional[FactorWeights] = None):
        """
        Initialize the scoring engine.
        
        Args:
            weights: Factor weights configuration
        """
        self.weights = weights or FactorWeights()
        self.factor_history = []
        self.weight_history = []
        self.correlation_matrix = None
        
    def _normalize_factors(self, factor_df: pd.DataFrame, 
                          method: str = "robust") -> pd.DataFrame:
        """
        Normalize factors using specified method.
        
        Args:
            factor_df: DataFrame with factor scores
            method: Normalization method ("standard", "robust", "winsorize")
            
        Returns:
            Normalized factor DataFrame
        """
        result = factor_df.copy()
        
        for col in result.columns:
            if col == 'symbol':
                continue
                
            series = result[col]
            if series.dtype in ['object', 'string']:
                continue
                
            if method == "standard":
                # Standard z-score normalization
                result[col] = (series - series.mean()) / (series.std() + 1e-9)
                
            elif method == "robust":
                # Robust normalization using median and MAD
                median = series.median()
                mad = (series - median).abs().median()
                result[col] = (series - median) / (mad * 1.4826 + 1e-9)  # 1.4826 for normal consistency
                
            elif method == "winsorize":
                # Winsorize outliers then standardize
                lower_bound = series.quantile(self.weights.winsorize_percentile)
                upper_bound = series.quantile(1 - self.weights.winsorize_percentile)
                winsorized = series.clip(lower_bound, upper_bound)
                result[col] = (winsorized - winsorized.mean()) / (winsorized.std() + 1e-9)
            
            # Replace infinite values
            result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return result
    
    def _calculate_factor_correlations(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.
        
        Args:
            factor_df: DataFrame with factor scores
            
        Returns:
            Correlation matrix
        """
        numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
        factor_only_df = factor_df[numeric_cols]
        
        if factor_only_df.empty:
            return pd.DataFrame()
            
        correlation_matrix = factor_only_df.corr()
        self.correlation_matrix = correlation_matrix
        
        return correlation_matrix
    
    def _detect_redundant_factors(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Detect pairs of factors with high correlation.
        
        Args:
            correlation_matrix: Factor correlation matrix
            
        Returns:
            List of (factor1, factor2, correlation) tuples for highly correlated pairs
        """
        redundant_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                factor1 = correlation_matrix.columns[i]
                factor2 = correlation_matrix.columns[j]
                corr = abs(correlation_matrix.iloc[i, j])
                
                if corr >= self.weights.high_correlation_threshold:
                    redundant_pairs.append((factor1, factor2, corr))
        
        return redundant_pairs
    
    def _adjust_weights_for_correlation(self, base_weights: Dict[str, float],
                                      redundant_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Adjust factor weights to account for high correlations.
        
        Args:
            base_weights: Initial factor weights
            redundant_pairs: List of highly correlated factor pairs
            
        Returns:
            Adjusted weights dictionary
        """
        adjusted_weights = base_weights.copy()
        
        # Penalize redundant factors
        for factor1, factor2, corr in redundant_pairs:
            penalty = self.weights.redundancy_penalty * corr
            
            # Apply penalty to both factors proportionally
            if factor1 in adjusted_weights:
                adjusted_weights[factor1] *= (1 - penalty / 2)
            if factor2 in adjusted_weights:
                adjusted_weights[factor2] *= (1 - penalty / 2)
        
        # Renormalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for factor in adjusted_weights:
                adjusted_weights[factor] /= total_weight
        
        # Ensure weights are within bounds
        for factor in adjusted_weights:
            adjusted_weights[factor] = np.clip(
                adjusted_weights[factor], 
                self.weights.min_weight, 
                self.weights.max_weight
            )
        
        # Final normalization
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for factor in adjusted_weights:
                adjusted_weights[factor] /= total_weight
        
        return adjusted_weights
    
    def _optimize_weights_dynamically(self, historical_factors: List[pd.DataFrame],
                                    historical_returns: Optional[List[pd.Series]] = None) -> Dict[str, float]:
        """
        Optimize factor weights based on historical performance using IC-based weighting.
        
        CRITICAL FIX #9: Implemented actual optimization using Information Coefficient (IC).
        Factors with higher IC (correlation with forward returns) get higher weights.
        
        Args:
            historical_factors: List of historical factor DataFrames
            historical_returns: Optional historical returns for optimization
            
        Returns:
            Optimized weights dictionary
        """
        base_weights = {
            'valuation': self.weights.valuation_weight,
            'volume': self.weights.volume_weight,
            'momentum': self.weights.momentum_weight,
            'technical': self.weights.technical_weight,
            'market_sentiment': self.weights.market_sentiment_weight
        }
        
        # Need at least 10 periods for meaningful optimization
        min_periods = 10
        if len(historical_factors) < min_periods:
            logger.info(f"[WEIGHT_OPT] Not enough history ({len(historical_factors)} < {min_periods}), using default weights")
            return base_weights
        
        if historical_returns is None or len(historical_returns) < min_periods:
            logger.info("[WEIGHT_OPT] No return data available, using default weights")
            return base_weights
        
        try:
            # Calculate Information Coefficient (IC) for each factor
            # IC = correlation between factor scores and forward returns
            factor_ics = {}
            factor_cols = ['valuation', 'volume', 'momentum', 'technical', 'market_sentiment']
            
            for factor in factor_cols:
                ics = []
                for i in range(len(historical_factors) - 1):
                    current_factors = historical_factors[i]
                    forward_returns = historical_returns[i + 1] if i + 1 < len(historical_returns) else None
                    
                    if forward_returns is None or current_factors is None:
                        continue
                    
                    if factor not in current_factors.columns:
                        continue
                    
                    # Align indices
                    common_symbols = current_factors.index.intersection(forward_returns.index)
                    if len(common_symbols) < 5:  # Need at least 5 stocks
                        continue
                    
                    factor_values = current_factors.loc[common_symbols, factor]
                    returns = forward_returns.loc[common_symbols]
                    
                    # Calculate Spearman rank correlation (more robust)
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(factor_values.values, returns.values)
                    
                    if not np.isnan(ic):
                        ics.append(ic)
                
                # Average IC for this factor
                if ics:
                    factor_ics[factor] = np.mean(ics)
                else:
                    factor_ics[factor] = 0.0
            
            # Convert ICs to weights using IC-IR weighting
            # Higher absolute IC = higher weight (we use abs because negative IC can be useful too)
            ic_weights = {}
            total_ic = sum(abs(ic) for ic in factor_ics.values())
            
            if total_ic > 0.01:  # Threshold to avoid division by near-zero
                for factor in factor_cols:
                    # Blend IC-based weight with base weight (50/50)
                    ic_based = abs(factor_ics.get(factor, 0)) / total_ic
                    base = base_weights.get(factor, 0.2)
                    ic_weights[factor] = 0.5 * ic_based + 0.5 * base
                
                # Normalize weights to sum to 1
                total_weight = sum(ic_weights.values())
                if total_weight > 0:
                    for factor in ic_weights:
                        ic_weights[factor] /= total_weight
                
                logger.info(f"[WEIGHT_OPT] Optimized weights based on IC: {ic_weights}")
                logger.info(f"[WEIGHT_OPT] Factor ICs: {factor_ics}")
                return ic_weights
            else:
                logger.info("[WEIGHT_OPT] Total IC too low, using default weights")
                return base_weights
                
        except Exception as e:
            logger.warning(f"[WEIGHT_OPT] Optimization failed: {e}, using default weights")
            return base_weights
    
    def _calculate_individual_factors(self, 
                                    data: Dict[str, pd.DataFrame],
                                    market_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculate all individual factor scores.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            market_data: Optional market-level data (VIX, benchmarks, etc.)
            
        Returns:
            DataFrame with all factor scores for each symbol
        """
        all_factors = []
        
        for symbol, df in data.items():
            if df is None or df.empty:
                continue
                
            try:
                factor_row = {'symbol': symbol}
                
                # Valuation factors (requires fundamental data)
                try:
                    if HAS_FACTORS and 'market_cap' in df.columns:  # Has fundamental data
                        val_score = valuation_score(df.iloc[[-1]])  # Latest data only
                        if not val_score.empty:
                            factor_row['valuation_score'] = val_score['ValuationScore'].iloc[0]
                    else:
                        factor_row['valuation_score'] = 0
                except Exception:
                    factor_row['valuation_score'] = 0
                
                # Volume factors
                try:
                    if HAS_FACTORS:
                        vol_features = volume_features(df)
                        if not vol_features.empty and 'vol_score' in vol_features.columns:
                            factor_row['volume_score'] = vol_features['vol_score'].iloc[-1]
                        else:
                            factor_row['volume_score'] = 0
                    else:
                        factor_row['volume_score'] = 0
                except Exception:
                    factor_row['volume_score'] = 0
                
                # Momentum factors
                try:
                    if HAS_FACTORS:
                        mom_features = momentum_features(df)
                        if not mom_features.empty and 'momentum_score' in mom_features.columns:
                            factor_row['momentum_score'] = mom_features['momentum_score'].iloc[-1]
                        else:
                            factor_row['momentum_score'] = 0
                    else:
                        factor_row['momentum_score'] = 0
                except Exception:
                    factor_row['momentum_score'] = 0
                
                # Technical factors
                try:
                    if HAS_FACTORS:
                        tech_features = technical_features(df)
                        if not tech_features.empty and 'technical_score' in tech_features.columns:
                            factor_row['technical_score'] = tech_features['technical_score'].iloc[-1]
                        else:
                            factor_row['technical_score'] = 0
                    else:
                        factor_row['technical_score'] = 0
                except Exception:
                    factor_row['technical_score'] = 0
                
                # Market sentiment factors
                try:
                    if HAS_FACTORS and market_data:
                        market_features = market_sentiment_features(
                            data, 
                            volume_data=market_data.get('volume_data'),
                            vix_data=market_data.get('vix_data'),
                            benchmark_data=market_data.get('benchmark_data'),
                            symbol=symbol
                        )
                        if not market_features.empty and 'market_sentiment_score' in market_features.columns:
                            factor_row['market_sentiment_score'] = market_features['market_sentiment_score'].iloc[-1]
                        else:
                            factor_row['market_sentiment_score'] = 0
                    else:
                        factor_row['market_sentiment_score'] = 0
                except Exception:
                    factor_row['market_sentiment_score'] = 0
                
                all_factors.append(factor_row)
                
            except Exception as e:
                warnings.warn(f"Error calculating factors for {symbol}: {e}")
                continue
        
        if not all_factors:
            return pd.DataFrame()
            
        factors_df = pd.DataFrame(all_factors)
        return factors_df
    
    def _apply_sector_neutrality(self, scores_df: pd.DataFrame,
                               sector_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Apply sector neutrality to scores.
        
        Args:
            scores_df: DataFrame with scores
            sector_mapping: Mapping of symbol -> sector
            
        Returns:
            Sector-neutral scores
        """
        result = scores_df.copy()
        
        if not self.weights.enable_sector_neutrality or not sector_mapping:
            return result
            
        # Add sector column
        result['sector'] = result['symbol'].map(sector_mapping)
        
        # Neutralize scores within each sector
        for col in result.columns:
            if col in ['symbol', 'sector']:
                continue
                
            if result['sector'].notna().any():
                # Within-sector z-score
                result[col] = result.groupby('sector')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-9)
                )
        
        result = result.drop('sector', axis=1)
        return result
    
    def calculate_composite_scores(self, 
                                  data: Dict[str, pd.DataFrame],
                                  market_data: Optional[Dict[str, Any]] = None,
                                  sector_mapping: Optional[Dict[str, str]] = None,
                                  custom_weights: Optional[Dict[str, float]] = None) -> ScoringResult:
        """
        Calculate composite scores for all symbols.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            market_data: Optional market-level data
            sector_mapping: Optional sector classification
            custom_weights: Optional custom factor weights
            
        Returns:
            ScoringResult with composite scores and analysis
        """
        # Calculate individual factors
        factors_df = self._calculate_individual_factors(data, market_data)
        
        if factors_df.empty:
            return ScoringResult(
                scores=pd.DataFrame(),
                factor_contributions=pd.DataFrame(),
                factor_correlations=pd.DataFrame(),
                weights_used={}
            )
        
        # Normalize factors
        normalized_factors = self._normalize_factors(factors_df, self.weights.outlier_method)
        
        # Calculate correlations
        correlation_matrix = self._calculate_factor_correlations(normalized_factors)
        
        # Detect redundant factors
        redundant_pairs = self._detect_redundant_factors(correlation_matrix)
        
        # Determine weights to use
        if custom_weights:
            weights_to_use = custom_weights
        elif self.weights.enable_dynamic_weights and len(self.factor_history) > 0:
            weights_to_use = self._optimize_weights_dynamically(self.factor_history)
        else:
            weights_to_use = {
                'valuation': self.weights.valuation_weight,
                'volume': self.weights.volume_weight,
                'momentum': self.weights.momentum_weight,
                'technical': self.weights.technical_weight,
                'market_sentiment': self.weights.market_sentiment_weight
            }
        
        # Adjust weights for correlations
        final_weights = self._adjust_weights_for_correlation(weights_to_use, redundant_pairs)
        
        # Calculate weighted composite scores
        score_columns = ['valuation_score', 'volume_score', 'momentum_score', 
                        'technical_score', 'market_sentiment_score']
        
        composite_scores = pd.DataFrame({'symbol': normalized_factors['symbol']})
        factor_contributions = pd.DataFrame({'symbol': normalized_factors['symbol']})
        
        composite_scores['composite_score'] = 0
        
        for factor_name, weight in final_weights.items():
            score_col = f'{factor_name}_score'
            if score_col in normalized_factors.columns:
                contribution = normalized_factors[score_col] * weight
                composite_scores['composite_score'] += contribution
                factor_contributions[factor_name] = contribution
            else:
                factor_contributions[factor_name] = 0
        
        # Apply sector neutrality if enabled
        if sector_mapping:
            composite_scores = self._apply_sector_neutrality(composite_scores, sector_mapping)
            factor_contributions = self._apply_sector_neutrality(factor_contributions, sector_mapping)
        
        # Final ranking
        composite_scores['rank'] = composite_scores['composite_score'].rank(ascending=False)
        composite_scores['percentile'] = composite_scores['composite_score'].rank(pct=True)
        
        # Store history
        self.factor_history.append(normalized_factors)
        self.weight_history.append(final_weights)
        
        # Keep only recent history
        if len(self.factor_history) > self.weights.weight_adjustment_period:
            self.factor_history = self.factor_history[-self.weights.weight_adjustment_period:]
            self.weight_history = self.weight_history[-self.weights.weight_adjustment_period:]
        
        # Create result
        result = ScoringResult(
            scores=composite_scores,
            factor_contributions=factor_contributions,
            factor_correlations=correlation_matrix,
            weights_used=final_weights,
            metadata={
                'redundant_pairs': redundant_pairs,
                'normalization_method': self.weights.outlier_method,
                'sector_neutral': self.weights.enable_sector_neutrality,
                'num_symbols': len(composite_scores)
            }
        )
        
        return result
    
    def explain_scores(self, result: ScoringResult, 
                      top_n: int = 10) -> Dict[str, Any]:
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
        
        analysis = {}
        
        # Top and bottom performers
        top_stocks = result.scores.nlargest(top_n, 'composite_score')
        bottom_stocks = result.scores.nsmallest(top_n, 'composite_score')
        
        analysis['top_stocks'] = top_stocks[['symbol', 'composite_score', 'rank', 'percentile']].to_dict('records')
        analysis['bottom_stocks'] = bottom_stocks[['symbol', 'composite_score', 'rank', 'percentile']].to_dict('records')
        
        # Factor importance
        factor_weights = result.weights_used
        analysis['factor_weights'] = factor_weights
        
        # Factor correlations summary
        if not result.factor_correlations.empty:
            # Average correlation per factor
            avg_correlations = {}
            for col in result.factor_correlations.columns:
                other_cols = [c for c in result.factor_correlations.columns if c != col]
                if other_cols:
                    avg_corr = result.factor_correlations.loc[col, other_cols].abs().mean()
                    avg_correlations[col] = avg_corr
            
            analysis['average_factor_correlations'] = avg_correlations
        
        # Score distribution statistics
        scores = result.scores['composite_score']
        analysis['score_statistics'] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(scores.median()),
            'skewness': float(scores.skew()) if len(scores) > 0 else 0,
            'kurtosis': float(scores.kurtosis()) if len(scores) > 0 else 0
        }
        
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
                    for factor in factor_weights.keys():
                        if factor in contrib_row.columns:
                            contrib_dict[factor] = float(contrib_row[factor].iloc[0])
                    top_contributions[symbol] = contrib_dict
            
            analysis['top_stock_contributions'] = top_contributions
        
        return analysis
    
    def get_trading_signals(self, result: ScoringResult,
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
        
        signals = result.scores.copy()
        signals['signal'] = 0
        
        # Buy signals (top performers)
        buy_mask = signals['percentile'] >= buy_threshold
        top_buys = signals[buy_mask].nlargest(max_positions, 'composite_score')
        signals.loc[signals['symbol'].isin(top_buys['symbol']), 'signal'] = 1
        
        # Sell signals (bottom performers)
        sell_mask = signals['percentile'] <= sell_threshold
        bottom_sells = signals[sell_mask].nsmallest(max_positions, 'composite_score')
        signals.loc[signals['symbol'].isin(bottom_sells['symbol']), 'signal'] = -1
        
        return signals[['symbol', 'composite_score', 'rank', 'percentile', 'signal']]
    
    def save_configuration(self, filepath: str):
        """Save current configuration to JSON file."""
        config = {
            'weights': {
                'valuation_weight': self.weights.valuation_weight,
                'volume_weight': self.weights.volume_weight,
                'momentum_weight': self.weights.momentum_weight,
                'technical_weight': self.weights.technical_weight,
                'market_sentiment_weight': self.weights.market_sentiment_weight,
                'enable_dynamic_weights': self.weights.enable_dynamic_weights,
                'weight_adjustment_period': self.weights.weight_adjustment_period,
                'min_weight': self.weights.min_weight,
                'max_weight': self.weights.max_weight,
                'high_correlation_threshold': self.weights.high_correlation_threshold,
                'redundancy_penalty': self.weights.redundancy_penalty,
                'outlier_method': self.weights.outlier_method,
                'winsorize_percentile': self.weights.winsorize_percentile,
                'enable_sector_neutrality': self.weights.enable_sector_neutrality
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_configuration(cls, filepath: str) -> 'MultiFactorScoringEngine':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        weights = FactorWeights(**config['weights'])
        return cls(weights)


# Utility functions for backwards compatibility
def calculate_multi_factor_score(data: Dict[str, pd.DataFrame], 
                               weights: Optional[Dict[str, float]] = None,
                               **kwargs) -> pd.DataFrame:
    """
    Legacy function for calculating multi-factor scores.
    
    Args:
        data: Dictionary of symbol -> OHLCV DataFrame
        weights: Optional custom weights
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with composite scores
    """
    engine = MultiFactorScoringEngine()
    result = engine.calculate_composite_scores(data, custom_weights=weights)
    return result.scores


def get_factor_signals(data: Dict[str, pd.DataFrame],
                      buy_threshold: float = 0.7,
                      sell_threshold: float = 0.3) -> pd.DataFrame:
    """
    Legacy function for generating factor-based signals.
    
    Args:
        data: Dictionary of symbol -> OHLCV DataFrame
        buy_threshold: Buy signal threshold
        sell_threshold: Sell signal threshold
        
    Returns:
        DataFrame with trading signals
    """
    engine = MultiFactorScoringEngine()
    result = engine.calculate_composite_scores(data)
    signals = engine.get_trading_signals(result, buy_threshold, sell_threshold)
    return signals