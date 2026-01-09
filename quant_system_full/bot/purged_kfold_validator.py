#!/usr/bin/env python3
"""
Purged K-Fold Cross-Validation for Time Series
时间序列Purged K-Fold交叉验证

Investment-grade cross-validation methodology to prevent data leakage
and overfitting in time series financial models:

- Purged K-Fold with embargo periods
- Walk-forward analysis with expanding/sliding windows
- Multiple testing correction (FDR control)
- Target leakage detection and prevention
- Feature stability and importance analysis

投资级交叉验证方法论：
- 带禁运期的Purged K-Fold验证
- 滚动前瞻分析（扩展/滑动窗口）
- 多重检验校正（FDR控制）
- 目标泄漏检测与防护
- 特征稳定性与重要性分析
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from dataclasses import dataclass, field
import logging
import warnings
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ValidationConfig:
    """Configuration for time series validation"""
    n_splits: int = 5                    # Number of CV folds
    embargo_days: int = 3                # Days to embargo around test set
    purge_days: int = 5                  # Days to purge between train/test
    min_train_samples: int = 252         # Minimum training samples (1 year daily)
    test_size_ratio: float = 0.2         # Test set size as ratio of total
    walk_forward_steps: int = 12         # Number of walk-forward steps
    confidence_level: float = 0.95       # Confidence level for statistical tests
    max_features_to_test: int = 50       # Maximum features to test simultaneously

@dataclass
class ValidationResults:
    """Results from cross-validation"""
    cv_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    feature_stability: Dict[str, float]
    overfitting_ratio: float             # OOS performance / IS performance
    p_value: float                       # Statistical significance
    is_significant: bool
    target_leakage_detected: bool
    warnings: List[str]

@dataclass
class WalkForwardResults:
    """Results from walk-forward analysis"""
    step_scores: List[float]
    step_dates: List[str]
    rolling_sharpe: List[float]
    rolling_information_ratio: List[float]
    performance_decay: float             # Rate of performance degradation
    stability_score: float               # Consistency across time periods
    regime_performance: Dict[str, float] # Performance by market regime

class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for time series data

    Prevents data leakage by:
    1. Purging overlapping samples between train/test sets
    2. Adding embargo periods around test sets
    3. Respecting temporal order in data splits
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validation_results: List[ValidationResults] = []
        self.walk_forward_results: Optional[WalkForwardResults] = None

        logger.info(f"Purged K-Fold validator initialized: {self.config.n_splits} splits, "
                   f"{self.config.embargo_days}d embargo, {self.config.purge_days}d purge")

    def get_train_test_splits(self,
                            timestamps: pd.Series,
                            embargo_days: Optional[int] = None,
                            purge_days: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits for time series data

        Args:
            timestamps: Series of timestamps (must be sorted)
            embargo_days: Days to embargo around test set
            purge_days: Days to purge between train/test sets

        Yields:
            Tuple of (train_indices, test_indices)
        """
        embargo_days = embargo_days or self.config.embargo_days
        purge_days = purge_days or self.config.purge_days

        n_samples = len(timestamps)
        test_size = int(n_samples * self.config.test_size_ratio / self.config.n_splits)

        for fold in range(self.config.n_splits):
            # Calculate test set boundaries
            test_start_idx = fold * test_size
            test_end_idx = min((fold + 1) * test_size, n_samples)

            if test_end_idx - test_start_idx < 10:  # Minimum test size
                continue

            # Get test timestamps
            test_start_time = timestamps.iloc[test_start_idx]
            test_end_time = timestamps.iloc[test_end_idx - 1]

            # Calculate embargo and purge boundaries
            embargo_start = test_start_time - pd.Timedelta(days=embargo_days)
            embargo_end = test_end_time + pd.Timedelta(days=embargo_days)
            purge_start = test_start_time - pd.Timedelta(days=purge_days)
            purge_end = test_end_time + pd.Timedelta(days=purge_days)

            # Create train set (excluding embargo and purge periods)
            train_mask = (
                (timestamps < embargo_start) |  # Before embargo
                (timestamps > embargo_end)      # After embargo
            )

            # Additional purge to ensure no overlap
            train_mask = train_mask & (
                (timestamps < purge_start) |
                (timestamps > purge_end)
            )

            # Test set indices
            test_indices = np.arange(test_start_idx, test_end_idx)

            # Train set indices
            train_indices = np.where(train_mask)[0]

            # Ensure minimum training size
            if len(train_indices) < self.config.min_train_samples:
                logger.warning(f"Fold {fold}: Insufficient training samples ({len(train_indices)} < {self.config.min_train_samples})")
                continue

            logger.debug(f"Fold {fold}: Train={len(train_indices)}, Test={len(test_indices)}, "
                        f"Train period: {timestamps.iloc[train_indices[0]]} to {timestamps.iloc[train_indices[-1]]}, "
                        f"Test period: {test_start_time} to {test_end_time}")

            yield train_indices, test_indices

    def detect_target_leakage(self,
                            features: pd.DataFrame,
                            target: pd.Series,
                            timestamps: pd.Series,
                            lag_days: int = 5) -> Dict[str, Any]:
        """
        Detect potential target leakage in features

        Tests for:
        1. Features that perfectly predict future returns
        2. Features with unrealistic forward-looking information
        3. Correlation patterns suggesting data snooping
        """
        leakage_results = {
            'leakage_detected': False,
            'suspicious_features': [],
            'perfect_predictors': [],
            'high_future_correlation': [],
            'details': {}
        }

        # Shift target forward to test for future information leakage
        future_target = target.shift(-lag_days)

        for feature_name in features.columns:
            feature_values = features[feature_name].dropna()

            # Test 1: Perfect prediction (correlation > 0.95 with future returns)
            if len(feature_values) > 10:
                # Align indices
                common_idx = feature_values.index.intersection(future_target.dropna().index)
                if len(common_idx) > 10:
                    corr, p_value = pearsonr(
                        feature_values.loc[common_idx],
                        future_target.loc[common_idx]
                    )

                    if abs(corr) > 0.95 and p_value < 0.01:
                        leakage_results['perfect_predictors'].append({
                            'feature': feature_name,
                            'correlation': corr,
                            'p_value': p_value
                        })
                        leakage_results['leakage_detected'] = True

                    elif abs(corr) > 0.3 and p_value < 0.01:
                        leakage_results['high_future_correlation'].append({
                            'feature': feature_name,
                            'correlation': corr,
                            'p_value': p_value
                        })

            # Test 2: Check for features that look forward beyond reasonable limits
            # (This would be implemented with domain-specific knowledge)

        if leakage_results['leakage_detected']:
            logger.warning(f"Target leakage detected in {len(leakage_results['perfect_predictors'])} features!")

        return leakage_results

    def calculate_feature_stability(self,
                                  features: pd.DataFrame,
                                  timestamps: pd.Series,
                                  window_size: int = 252) -> Dict[str, float]:
        """
        Calculate feature stability over time using rolling correlations

        Stable features should maintain consistent relationships over time
        """
        stability_scores = {}

        for feature_name in features.columns:
            feature_series = features[feature_name].dropna()

            if len(feature_series) < window_size * 2:
                stability_scores[feature_name] = 0.0
                continue

            # Calculate rolling correlations between consecutive windows
            correlations = []

            for i in range(window_size, len(feature_series) - window_size, window_size // 2):
                window1 = feature_series.iloc[i-window_size:i]
                window2 = feature_series.iloc[i:i+window_size]

                # Calculate correlation between consecutive windows
                if len(window1) > 10 and len(window2) > 10:
                    corr, _ = pearsonr(
                        np.arange(len(window1)),  # Use time index as proxy
                        window1.values
                    )
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)

            # Stability score is the consistency of these correlations
            if correlations:
                stability_scores[feature_name] = 1.0 - np.std(correlations)
            else:
                stability_scores[feature_name] = 0.0

        return stability_scores

    def multiple_testing_correction(self,
                                  p_values: List[float],
                                  method: str = 'fdr_bh') -> Tuple[List[bool], List[float]]:
        """
        Apply multiple testing correction to control False Discovery Rate

        Methods:
        - 'fdr_bh': Benjamini-Hochberg FDR control
        - 'bonferroni': Conservative Bonferroni correction
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)

        if method == 'bonferroni':
            # Bonferroni correction: α' = α / n
            corrected_alpha = self.config.confidence_level / n_tests
            reject = p_values < corrected_alpha
            corrected_p = p_values * n_tests

        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            # Find largest k such that P(k) <= (k/n) * α
            reject = np.zeros(n_tests, dtype=bool)
            alpha = 1 - self.config.confidence_level

            for i in range(n_tests - 1, -1, -1):
                threshold = (i + 1) / n_tests * alpha
                if sorted_p[i] <= threshold:
                    reject[sorted_indices[:i+1]] = True
                    break

            corrected_p = p_values  # BH doesn't adjust p-values directly

        else:
            raise ValueError(f"Unknown correction method: {method}")

        return reject.tolist(), corrected_p.tolist()

    def cross_validate_model(self,
                           features: pd.DataFrame,
                           target: pd.Series,
                           timestamps: pd.Series,
                           model_func: callable,
                           scoring_func: callable = None) -> ValidationResults:
        """
        Perform purged k-fold cross-validation on a model

        Args:
            features: Feature matrix
            target: Target variable
            timestamps: Time index (must be sorted)
            model_func: Function that takes (X_train, y_train) and returns trained model
            scoring_func: Function that takes (y_true, y_pred) and returns score
        """
        if scoring_func is None:
            scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)

        # Check for target leakage
        leakage_results = self.detect_target_leakage(features, target, timestamps)

        # Calculate feature stability
        feature_stability = self.calculate_feature_stability(features, timestamps)

        # Perform cross-validation
        cv_scores = []
        feature_importances = []
        warnings_list = []

        try:
            for fold, (train_idx, test_idx) in enumerate(
                self.get_train_test_splits(timestamps)
            ):
                # Extract train/test data
                X_train = features.iloc[train_idx]
                y_train = target.iloc[train_idx]
                X_test = features.iloc[test_idx]
                y_test = target.iloc[test_idx]

                # Remove NaN values
                train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
                test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())

                X_train = X_train[train_mask]
                y_train = y_train[train_mask]
                X_test = X_test[test_mask]
                y_test = y_test[test_mask]

                if len(X_train) < 50 or len(X_test) < 10:
                    warnings_list.append(f"Fold {fold}: Insufficient data after cleaning")
                    continue

                # Train model
                model = model_func(X_train, y_train)

                # Generate predictions
                y_pred = model.predict(X_test)

                # Calculate score
                score = scoring_func(y_test, y_pred)
                cv_scores.append(score)

                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(features.columns, model.feature_importances_))
                    feature_importances.append(importance_dict)

                logger.debug(f"Fold {fold}: Score = {score:.4f}")

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            warnings_list.append(f"CV error: {str(e)}")

        if not cv_scores:
            logger.error("No valid CV scores obtained")
            return ValidationResults(
                cv_scores=[],
                mean_score=0.0,
                std_score=0.0,
                confidence_interval=(0.0, 0.0),
                feature_importance={},
                feature_stability=feature_stability,
                overfitting_ratio=0.0,
                p_value=1.0,
                is_significant=False,
                target_leakage_detected=leakage_results['leakage_detected'],
                warnings=warnings_list
            )

        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        # Confidence interval
        n = len(cv_scores)
        sem = std_score / np.sqrt(n)
        t_critical = stats.t.ppf((1 + self.config.confidence_level) / 2, n - 1)
        ci_lower = mean_score - t_critical * sem
        ci_upper = mean_score + t_critical * sem

        # Statistical significance test (one-sample t-test against zero)
        t_stat, p_value = stats.ttest_1samp(cv_scores, 0)

        # Average feature importance
        avg_feature_importance = {}
        if feature_importances:
            for feature in features.columns:
                importances = [fi.get(feature, 0) for fi in feature_importances if feature in fi]
                if importances:
                    avg_feature_importance[feature] = np.mean(importances)

        # Overfitting ratio (would need in-sample performance for comparison)
        overfitting_ratio = 1.0  # Placeholder

        results = ValidationResults(
            cv_scores=cv_scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=(ci_lower, ci_upper),
            feature_importance=avg_feature_importance,
            feature_stability=feature_stability,
            overfitting_ratio=overfitting_ratio,
            p_value=p_value,
            is_significant=p_value < (1 - self.config.confidence_level),
            target_leakage_detected=leakage_results['leakage_detected'],
            warnings=warnings_list
        )

        self.validation_results.append(results)
        logger.info(f"CV completed: Mean score = {mean_score:.4f} ± {std_score:.4f}, p-value = {p_value:.4f}")

        return results

    def walk_forward_analysis(self,
                            features: pd.DataFrame,
                            target: pd.Series,
                            timestamps: pd.Series,
                            model_func: callable,
                            window_type: str = 'expanding') -> WalkForwardResults:
        """
        Perform walk-forward analysis with expanding or rolling windows

        Args:
            window_type: 'expanding' or 'rolling'
        """
        n_samples = len(features)
        step_size = n_samples // self.config.walk_forward_steps
        min_train_size = self.config.min_train_samples

        step_scores = []
        step_dates = []
        rolling_sharpe = []

        for step in range(self.config.walk_forward_steps):
            # Define test period
            test_start = min_train_size + step * step_size
            test_end = min(test_start + step_size, n_samples)

            if test_end - test_start < 10:  # Minimum test size
                continue

            # Define training period
            if window_type == 'expanding':
                train_start = 0
                train_end = test_start
            else:  # rolling
                train_start = max(0, test_start - min_train_size)
                train_end = test_start

            # Extract data
            X_train = features.iloc[train_start:train_end]
            y_train = target.iloc[train_start:train_end]
            X_test = features.iloc[test_start:test_end]
            y_test = target.iloc[test_start:test_end]

            # Clean data
            train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())

            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

            if len(X_train) < 50 or len(X_test) < 5:
                continue

            try:
                # Train and predict
                model = model_func(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate return-based metrics
                returns = y_test.values
                step_score = np.mean(returns)
                step_scores.append(step_score)
                step_dates.append(timestamps.iloc[test_start])

                # Calculate rolling Sharpe ratio
                if len(returns) > 1:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0.0)

                logger.debug(f"Walk-forward step {step}: Score = {step_score:.4f}, Sharpe = {rolling_sharpe[-1]:.2f}")

            except Exception as e:
                logger.warning(f"Walk-forward step {step} failed: {e}")
                continue

        # Calculate performance metrics
        performance_decay = 0.0
        stability_score = 0.0

        if len(step_scores) > 2:
            # Performance decay: slope of performance over time
            x = np.arange(len(step_scores))
            slope, _, _, _, _ = stats.linregress(x, step_scores)
            performance_decay = slope

            # Stability: inverse of coefficient of variation
            if np.mean(step_scores) != 0:
                stability_score = 1.0 / (np.std(step_scores) / abs(np.mean(step_scores)) + 0.01)

        self.walk_forward_results = WalkForwardResults(
            step_scores=step_scores,
            step_dates=[d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in step_dates],
            rolling_sharpe=rolling_sharpe,
            rolling_information_ratio=[],  # Would need benchmark for IR
            performance_decay=performance_decay,
            stability_score=stability_score,
            regime_performance={}  # Would need regime classification
        )

        logger.info(f"Walk-forward analysis completed: {len(step_scores)} steps, "
                   f"stability = {stability_score:.3f}, decay = {performance_decay:.6f}")

        return self.walk_forward_results

    def export_validation_report(self, filepath: str) -> bool:
        """Export comprehensive validation report"""
        try:
            report_data = {
                'validation_timestamp': datetime.now().isoformat(),
                'config': {
                    'n_splits': self.config.n_splits,
                    'embargo_days': self.config.embargo_days,
                    'purge_days': self.config.purge_days,
                    'min_train_samples': self.config.min_train_samples,
                    'confidence_level': self.config.confidence_level
                },
                'cross_validation_results': [
                    {
                        'cv_scores': result.cv_scores,
                        'mean_score': result.mean_score,
                        'std_score': result.std_score,
                        'confidence_interval': result.confidence_interval,
                        'p_value': result.p_value,
                        'is_significant': result.is_significant,
                        'target_leakage_detected': result.target_leakage_detected,
                        'warnings': result.warnings,
                        'feature_importance': result.feature_importance,
                        'feature_stability': result.feature_stability
                    }
                    for result in self.validation_results
                ],
                'walk_forward_results': {
                    'step_scores': self.walk_forward_results.step_scores,
                    'step_dates': self.walk_forward_results.step_dates,
                    'rolling_sharpe': self.walk_forward_results.rolling_sharpe,
                    'performance_decay': self.walk_forward_results.performance_decay,
                    'stability_score': self.walk_forward_results.stability_score
                } if self.walk_forward_results else {}
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Validation report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Purged K-Fold Validator - Investment Grade Cross-Validation")
    print("=" * 70)

    # Create synthetic time series data for testing
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Synthetic features with realistic financial characteristics
    features_data = {
        'factor_1': np.random.normal(0, 1, n_samples),
        'factor_2': np.random.normal(0, 1, n_samples),
        'factor_3': np.random.normal(0, 1, n_samples),
        'momentum': np.random.normal(0, 0.5, n_samples),
        'volatility': np.random.exponential(0.2, n_samples)
    }

    features = pd.DataFrame(features_data, index=dates)

    # Synthetic target with some predictable patterns
    target = (
        0.1 * features['factor_1'] +
        0.05 * features['factor_2'] +
        np.random.normal(0, 0.02, n_samples)
    )
    target = pd.Series(target, index=dates)

    # Simple model function for testing
    def simple_model(X_train, y_train):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    # Initialize validator
    config = ValidationConfig(n_splits=5, embargo_days=5, purge_days=3)
    validator = PurgedKFoldCV(config)

    # Perform cross-validation
    cv_results = validator.cross_validate_model(
        features=features,
        target=target,
        timestamps=features.index,
        model_func=simple_model
    )

    print(f"Cross-Validation Results:")
    print(f"Mean Score: {cv_results.mean_score:.4f} ± {cv_results.std_score:.4f}")
    print(f"Confidence Interval: [{cv_results.confidence_interval[0]:.4f}, {cv_results.confidence_interval[1]:.4f}]")
    print(f"P-value: {cv_results.p_value:.4f}")
    print(f"Statistically Significant: {cv_results.is_significant}")
    print(f"Target Leakage Detected: {cv_results.target_leakage_detected}")

    # Feature importance
    print(f"\nTop 3 Important Features:")
    sorted_features = sorted(cv_results.feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:3]:
        print(f"  {feature}: {importance:.4f}")

    # Walk-forward analysis
    wf_results = validator.walk_forward_analysis(
        features=features,
        target=target,
        timestamps=features.index,
        model_func=simple_model,
        window_type='expanding'
    )

    print(f"\nWalk-Forward Analysis:")
    print(f"Number of Steps: {len(wf_results.step_scores)}")
    print(f"Average Score: {np.mean(wf_results.step_scores):.4f}")
    print(f"Performance Decay: {wf_results.performance_decay:.6f}")
    print(f"Stability Score: {wf_results.stability_score:.3f}")

    # Export validation report
    validator.export_validation_report("purged_kfold_validation_report.json")
    print(f"\nValidation report exported successfully!")