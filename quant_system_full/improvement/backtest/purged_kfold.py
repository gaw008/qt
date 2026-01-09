#!/usr/bin/env python3
"""
Purged K-Fold Cross-Validation for Time Series

Implements purged and combinatorial cross-validation specifically designed for
financial time series to prevent data leakage and overfitting.

Key Features:
- Purged K-Fold: Removes samples around test period to prevent label leakage
- Combinatorial Purged CV: Multiple train/test combinations for robust evaluation
- Embargo periods: Additional buffer to prevent forward-looking bias
- Time-aware splitting: Respects temporal order of financial data
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

@dataclass
class CVConfig:
    """Configuration for cross-validation"""
    n_splits: int = 5
    purge_pct: float = 0.02  # Percentage of data to purge around test set
    embargo_pct: float = 0.01  # Additional embargo period
    test_size: float = 0.2  # Size of each test fold
    min_train_size: float = 0.3  # Minimum training set size
    combinatorial: bool = False  # Whether to use combinatorial CV
    max_combinations: int = 10  # Max combinations for combinatorial CV

class PurgedTimeSeriesSplit:
    """
    Time series cross-validator with purging and embargo periods

    Prevents data leakage by:
    1. Purging samples around test period
    2. Adding embargo periods
    3. Maintaining temporal order
    4. Avoiding overlapping train/test periods
    """

    def __init__(self, config: CVConfig):
        self.config = config
        self.fold_results = []

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for purged time series cross-validation

        Args:
            X: Feature matrix with DatetimeIndex
            y: Target variable (optional)
            groups: Group labels (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex for time series CV")

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold sizes
        test_size = int(n_samples * self.config.test_size)
        purge_size = int(n_samples * self.config.purge_pct)
        embargo_size = int(n_samples * self.config.embargo_pct)

        # Generate fold start positions
        fold_starts = self._get_fold_starts(n_samples, test_size)

        for fold_idx, test_start in enumerate(fold_starts):
            test_end = min(test_start + test_size, n_samples)

            # Define purge and embargo regions
            purge_start = max(0, test_start - purge_size)
            purge_end = min(n_samples, test_end + embargo_size)

            # Training indices: everything except test and purged regions
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:purge_end] = False
            train_indices = indices[train_mask]

            # Test indices
            test_indices = indices[test_start:test_end]

            # Check minimum training size
            if len(train_indices) < int(n_samples * self.config.min_train_size):
                warnings.warn(f"Fold {fold_idx}: Training set too small, skipping")
                continue

            # Store fold information
            fold_info = {
                'fold': fold_idx,
                'train_start': X.index[train_indices[0]] if len(train_indices) > 0 else None,
                'train_end': X.index[train_indices[-1]] if len(train_indices) > 0 else None,
                'test_start': X.index[test_start],
                'test_end': X.index[test_end - 1],
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'purge_size': purge_size,
                'embargo_size': embargo_size
            }
            self.fold_results.append(fold_info)

            yield train_indices, test_indices

    def _get_fold_starts(self, n_samples: int, test_size: int) -> List[int]:
        """Calculate fold start positions"""
        if self.config.combinatorial:
            return self._get_combinatorial_starts(n_samples, test_size)
        else:
            return self._get_sequential_starts(n_samples, test_size)

    def _get_sequential_starts(self, n_samples: int, test_size: int) -> List[int]:
        """Sequential fold starts for standard K-Fold"""
        step_size = (n_samples - test_size) // (self.config.n_splits - 1)
        starts = [i * step_size for i in range(self.config.n_splits)]
        return [s for s in starts if s + test_size <= n_samples]

    def _get_combinatorial_starts(self, n_samples: int, test_size: int) -> List[int]:
        """Combinatorial fold starts for more robust evaluation"""
        min_gap = int(n_samples * (self.config.purge_pct + self.config.embargo_pct) * 2)
        possible_starts = list(range(0, n_samples - test_size, min_gap))

        if len(possible_starts) <= self.config.max_combinations:
            return possible_starts
        else:
            # Randomly sample combinations
            import random
            random.seed(42)  # For reproducibility
            return random.sample(possible_starts, self.config.max_combinations)

    def get_fold_info(self) -> pd.DataFrame:
        """Get information about all folds"""
        if not self.fold_results:
            return pd.DataFrame()
        return pd.DataFrame(self.fold_results)

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation

    Generates multiple train/test combinations to provide more robust
    performance estimates and detect overfitting.
    """

    def __init__(self, config: CVConfig):
        self.config = config
        self.results = []

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series,
                      scoring_func, fit_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform combinatorial cross-validation

        Args:
            model: Model to evaluate (must have fit/predict methods)
            X: Feature matrix
            y: Target variable
            scoring_func: Function to calculate performance scores
            fit_params: Additional parameters for model fitting

        Returns:
            Dictionary with CV results and statistics
        """
        fit_params = fit_params or {}

        # Initialize splitter
        splitter = PurgedTimeSeriesSplit(self.config)

        fold_scores = []
        fold_details = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            try:
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Fit model
                model.fit(X_train, y_train, **fit_params)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate scores
                score = scoring_func(y_test, y_pred)
                fold_scores.append(score)

                # Store fold details
                fold_detail = {
                    'fold': fold_idx,
                    'score': score,
                    'train_period': f"{X_train.index[0]} to {X_train.index[-1]}",
                    'test_period': f"{X_test.index[0]} to {X_test.index[-1]}",
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                fold_details.append(fold_detail)

            except Exception as e:
                warnings.warn(f"Fold {fold_idx} failed: {str(e)}")
                continue

        if not fold_scores:
            raise ValueError("All folds failed - check data and model")

        # Calculate statistics
        scores_array = np.array(fold_scores)
        results = {
            'scores': fold_scores,
            'mean_score': np.mean(scores_array),
            'std_score': np.std(scores_array),
            'min_score': np.min(scores_array),
            'max_score': np.max(scores_array),
            'median_score': np.median(scores_array),
            'q25_score': np.percentile(scores_array, 25),
            'q75_score': np.percentile(scores_array, 75),
            'fold_details': fold_details,
            'fold_info': splitter.get_fold_info(),
            'config': self.config
        }

        self.results.append(results)
        return results

def deflated_sharpe_ratio(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                         trials: int = 1, freq: int = 252) -> Dict[str, float]:
    """
    Calculate Deflated Sharpe Ratio (DSR) to adjust for multiple testing

    The DSR adjusts the Sharpe ratio for the number of trials/backtests
    performed, helping to identify genuine alpha vs. data mining luck.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns (optional)
        trials: Number of strategy trials/backtests performed
        freq: Frequency of returns (252 for daily)

    Returns:
        Dictionary with Sharpe ratio, DSR, and related statistics
    """
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
    else:
        excess_returns = returns

    # Basic Sharpe ratio calculation
    mean_return = excess_returns.mean() * freq
    vol_return = excess_returns.std() * np.sqrt(freq)
    sharpe_ratio = mean_return / vol_return if vol_return != 0 else 0

    # Calculate DSR adjustment
    n_obs = len(returns)

    # Estimated Sharpe ratio variance
    sharpe_var = (1 + sharpe_ratio**2 / 2) / n_obs

    # Deflation factor based on number of trials
    if trials > 1:
        # Expected maximum Sharpe ratio under null hypothesis
        expected_max_sr = np.sqrt(2 * np.log(trials))

        # DSR calculation
        dsr = (sharpe_ratio - expected_max_sr) / np.sqrt(sharpe_var)
    else:
        dsr = sharpe_ratio / np.sqrt(sharpe_var)

    # Additional statistics
    results = {
        'sharpe_ratio': sharpe_ratio,
        'deflated_sharpe_ratio': dsr,
        'annualized_return': mean_return,
        'annualized_volatility': vol_return,
        'trials': trials,
        'observations': n_obs,
        'sharpe_variance': sharpe_var,
        'is_significant': dsr > 0.1,  # Common threshold
        'confidence_level': 2 * (1 - stats.norm.cdf(abs(dsr))) if 'stats' in globals() else None
    }

    return results

def validate_strategy_robustness(strategy_func, data: pd.DataFrame,
                                target_col: str, feature_cols: List[str],
                                cv_config: Optional[CVConfig] = None) -> Dict[str, Any]:
    """
    Comprehensive strategy robustness validation

    Args:
        strategy_func: Function that returns fitted model
        data: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature column names
        cv_config: Cross-validation configuration

    Returns:
        Comprehensive robustness analysis results
    """
    cv_config = cv_config or CVConfig()

    # Prepare data
    X = data[feature_cols].copy()
    y = data[target_col].copy()

    # Remove any missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X, y = X[mask], y[mask]

    print(f"Robustness validation on {len(X)} samples, {len(feature_cols)} features")

    # Define scoring function
    def sharpe_scoring(y_true, y_pred):
        """Calculate Sharpe ratio as score"""
        returns = pd.Series(y_pred, index=y_true.index)
        return deflated_sharpe_ratio(returns)['sharpe_ratio']

    # Run combinatorial CV
    cv = CombinatorialPurgedCV(cv_config)

    try:
        # Get model instance
        model = strategy_func()

        # Perform cross-validation
        cv_results = cv.cross_validate(model, X, y, sharpe_scoring)

        # Calculate deflated Sharpe ratio
        all_scores = cv_results['scores']
        dsr_results = deflated_sharpe_ratio(
            pd.Series(all_scores),
            trials=len(all_scores)
        )

        # Combine results
        robustness_results = {
            'cross_validation': cv_results,
            'deflated_sharpe': dsr_results,
            'summary': {
                'mean_sharpe': cv_results['mean_score'],
                'std_sharpe': cv_results['std_score'],
                'dsr': dsr_results['deflated_sharpe_ratio'],
                'is_robust': dsr_results['is_significant'],
                'n_folds': len(all_scores),
                'worst_performance': cv_results['min_score'],
                'best_performance': cv_results['max_score']
            }
        }

        return robustness_results

    except Exception as e:
        return {
            'error': str(e),
            'traceback': str(e.__class__.__name__)
        }

# Example usage functions
def create_simple_strategy():
    """Example strategy for testing"""
    from sklearn.linear_model import LinearRegression
    return LinearRegression()

if __name__ == "__main__":
    # Example usage
    import scipy.stats as stats

    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates)),
        'feature3': np.random.randn(len(dates)),
        'returns': np.random.randn(len(dates)) * 0.02
    }, index=dates)

    # Test configuration
    config = CVConfig(
        n_splits=5,
        purge_pct=0.02,
        embargo_pct=0.01,
        combinatorial=True,
        max_combinations=8
    )

    # Run robustness validation
    results = validate_strategy_robustness(
        create_simple_strategy,
        sample_data,
        'returns',
        ['feature1', 'feature2', 'feature3'],
        config
    )

    print("Robustness Validation Results:")
    print(f"Mean Sharpe: {results['summary']['mean_sharpe']:.3f}")
    print(f"Deflated Sharpe: {results['summary']['dsr']:.3f}")
    print(f"Is Robust: {results['summary']['is_robust']}")
    print(f"Number of folds: {results['summary']['n_folds']}")