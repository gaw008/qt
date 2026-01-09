#!/usr/bin/env python3
"""
Rolling Out-of-Sample Validation Framework

Implements rigorous out-of-sample testing to evaluate strategy performance
on truly unseen data and detect overfitting.

Key Features:
- Rolling window validation (3 years train -> 6 months test)
- Walk-forward analysis with parameter reoptimization
- Performance degradation detection
- Out-of-sample vs in-sample comparison
- Regime-aware validation splits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

@dataclass
class RollingConfig:
    """Configuration for rolling validation"""
    train_period_months: int = 36  # 3 years training
    test_period_months: int = 6   # 6 months testing
    step_months: int = 6          # Move forward by 6 months
    min_train_samples: int = 500  # Minimum training samples
    min_test_samples: int = 100   # Minimum test samples
    reoptimize_params: bool = True  # Whether to reoptimize parameters each window
    save_detailed_results: bool = True
    results_dir: str = "reports/rolling_validation"

@dataclass
class ValidationWindow:
    """Single validation window information"""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_samples: int
    test_samples: int
    market_regime: Optional[str] = None

class RollingValidator:
    """
    Implements rolling out-of-sample validation for trading strategies

    Features:
    - Walk-forward analysis with configurable windows
    - Parameter reoptimization for each window
    - Performance tracking across market regimes
    - Degradation analysis and early stopping
    """

    def __init__(self, config: RollingConfig):
        self.config = config
        self.windows = []
        self.results = []
        self.summary_stats = {}

    def validate_strategy(self, strategy_func: Callable, data: pd.DataFrame,
                         param_optimizer: Optional[Callable] = None,
                         base_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform rolling out-of-sample validation

        Args:
            strategy_func: Function that takes (data, **params) and returns performance
            data: Market data with DatetimeIndex
            param_optimizer: Function to optimize parameters on training data
            base_params: Base parameters if no optimization is used

        Returns:
            Comprehensive validation results
        """
        print("Starting Rolling Out-of-Sample Validation...")

        # Generate validation windows
        self.windows = self._generate_windows(data)
        print(f"Generated {len(self.windows)} validation windows")

        # Process each window
        window_results = []
        for window in self.windows:
            print(f"Processing window {window.window_id}: {window.test_start} to {window.test_end}")

            try:
                result = self._process_window(strategy_func, data, window,
                                            param_optimizer, base_params)
                window_results.append(result)
            except Exception as e:
                print(f"Window {window.window_id} failed: {str(e)}")
                continue

        if not window_results:
            raise ValueError("All validation windows failed")

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(window_results)

        # Analyze performance degradation
        degradation_analysis = self._analyze_performance_degradation(window_results)

        # Generate regime analysis
        regime_analysis = self._analyze_by_regime(window_results)

        # Compile results
        validation_results = {
            'config': self.config,
            'windows': [w.__dict__ for w in self.windows],
            'window_results': window_results,
            'summary_stats': summary_stats,
            'degradation_analysis': degradation_analysis,
            'regime_analysis': regime_analysis,
            'validation_passed': self._check_validation_criteria(summary_stats)
        }

        # Save detailed results
        if self.config.save_detailed_results:
            self._save_results(validation_results)

        # Generate plots
        self._generate_plots(validation_results)

        self.results = validation_results
        return validation_results

    def _generate_windows(self, data: pd.DataFrame) -> List[ValidationWindow]:
        """Generate rolling validation windows"""
        windows = []
        start_date = data.index[0]
        end_date = data.index[-1]

        window_id = 0
        current_date = start_date

        while True:
            # Calculate train period
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.config.train_period_months)

            # Calculate test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.config.test_period_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Count actual samples
            train_mask = (data.index >= train_start) & (data.index < train_end)
            test_mask = (data.index >= test_start) & (data.index < test_end)

            train_samples = train_mask.sum()
            test_samples = test_mask.sum()

            # Check minimum sample requirements
            if train_samples < self.config.min_train_samples or test_samples < self.config.min_test_samples:
                current_date += pd.DateOffset(months=self.config.step_months)
                continue

            # Detect market regime for test period
            regime = self._detect_market_regime(data[test_mask])

            window = ValidationWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=train_samples,
                test_samples=test_samples,
                market_regime=regime
            )

            windows.append(window)
            window_id += 1

            # Move to next window
            current_date += pd.DateOffset(months=self.config.step_months)

        return windows

    def _process_window(self, strategy_func: Callable, data: pd.DataFrame,
                       window: ValidationWindow, param_optimizer: Optional[Callable],
                       base_params: Optional[Dict]) -> Dict[str, Any]:
        """Process a single validation window"""

        # Split data
        train_mask = (data.index >= window.train_start) & (data.index < window.train_end)
        test_mask = (data.index >= window.test_start) & (data.index < window.test_end)

        train_data = data[train_mask].copy()
        test_data = data[test_mask].copy()

        # Optimize parameters on training data
        if param_optimizer and self.config.reoptimize_params:
            optimal_params = param_optimizer(train_data)
        else:
            optimal_params = base_params or {}

        # Calculate in-sample performance
        train_performance = self._evaluate_strategy(strategy_func, train_data, optimal_params)

        # Calculate out-of-sample performance
        test_performance = self._evaluate_strategy(strategy_func, test_data, optimal_params)

        # Calculate performance metrics
        window_result = {
            'window_id': window.window_id,
            'window_info': window.__dict__,
            'optimal_params': optimal_params,
            'in_sample_performance': train_performance,
            'out_of_sample_performance': test_performance,
            'performance_degradation': self._calculate_degradation(train_performance, test_performance),
            'market_regime': window.market_regime
        }

        return window_result

    def _evaluate_strategy(self, strategy_func: Callable, data: pd.DataFrame,
                          params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy performance"""
        try:
            # Run strategy
            results = strategy_func(data, **params)

            # Extract returns
            if isinstance(results, pd.Series):
                returns = results
            elif isinstance(results, dict) and 'returns' in results:
                returns = results['returns']
            else:
                raise ValueError("Strategy must return returns series or dict with 'returns'")

            # Calculate metrics
            performance = self._calculate_metrics(returns)
            return performance

        except Exception as e:
            warnings.warn(f"Strategy evaluation failed: {str(e)}")
            return {'sharpe_ratio': np.nan, 'total_return': np.nan, 'max_drawdown': np.nan}

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(returns) == 0:
            return {'sharpe_ratio': np.nan, 'total_return': np.nan, 'max_drawdown': np.nan}

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': annualized_vol,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio
        }

    def _calculate_degradation(self, in_sample: Dict, out_sample: Dict) -> Dict[str, float]:
        """Calculate performance degradation from in-sample to out-of-sample"""
        degradation = {}

        for metric in ['sharpe_ratio', 'total_return', 'calmar_ratio']:
            is_value = in_sample.get(metric, 0)
            oos_value = out_sample.get(metric, 0)

            if is_value != 0:
                degradation[f'{metric}_degradation'] = (oos_value - is_value) / abs(is_value)
            else:
                degradation[f'{metric}_degradation'] = 0

        return degradation

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Simple market regime detection"""
        if 'close' not in data.columns:
            return 'unknown'

        # Calculate returns
        returns = data['close'].pct_change().dropna()

        if len(returns) < 10:
            return 'unknown'

        # Simple regime classification
        mean_return = returns.mean()
        volatility = returns.std()

        # VIX-like measure (rolling volatility)
        vol_threshold = 0.02  # 2% daily volatility threshold

        if mean_return > 0.001 and volatility < vol_threshold:
            return 'bull_low_vol'
        elif mean_return > 0.001 and volatility >= vol_threshold:
            return 'bull_high_vol'
        elif mean_return < -0.001 and volatility >= vol_threshold:
            return 'bear_high_vol'
        elif abs(mean_return) <= 0.001:
            return 'sideways'
        else:
            return 'bear_low_vol'

    def _calculate_summary_stats(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics across all windows"""
        if not window_results:
            return {}

        # Extract metrics
        is_sharpe = [r['in_sample_performance']['sharpe_ratio'] for r in window_results]
        oos_sharpe = [r['out_of_sample_performance']['sharpe_ratio'] for r in window_results]

        is_return = [r['in_sample_performance']['total_return'] for r in window_results]
        oos_return = [r['out_of_sample_performance']['total_return'] for r in window_results]

        # Remove NaN values
        is_sharpe = [x for x in is_sharpe if not np.isnan(x)]
        oos_sharpe = [x for x in oos_sharpe if not np.isnan(x)]
        is_return = [x for x in is_return if not np.isnan(x)]
        oos_return = [x for x in oos_return if not np.isnan(x)]

        summary = {
            'total_windows': len(window_results),
            'successful_windows': len(oos_sharpe),
            'in_sample_sharpe': {
                'mean': np.mean(is_sharpe) if is_sharpe else np.nan,
                'std': np.std(is_sharpe) if is_sharpe else np.nan,
                'median': np.median(is_sharpe) if is_sharpe else np.nan
            },
            'out_of_sample_sharpe': {
                'mean': np.mean(oos_sharpe) if oos_sharpe else np.nan,
                'std': np.std(oos_sharpe) if oos_sharpe else np.nan,
                'median': np.median(oos_sharpe) if oos_sharpe else np.nan
            },
            'sharpe_degradation': {
                'mean': np.mean(oos_sharpe) - np.mean(is_sharpe) if is_sharpe and oos_sharpe else np.nan,
                'consistency': len([i for i, (is_s, oos_s) in enumerate(zip(is_sharpe, oos_sharpe)) if oos_s > 0]) / len(oos_sharpe) if oos_sharpe else 0
            }
        }

        return summary

    def _analyze_performance_degradation(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance degradation patterns"""
        degradations = []

        for result in window_results:
            degradation = result.get('performance_degradation', {})
            if degradation:
                degradations.append(degradation)

        if not degradations:
            return {}

        # Aggregate degradation statistics
        sharpe_degradations = [d.get('sharpe_ratio_degradation', 0) for d in degradations]
        return_degradations = [d.get('total_return_degradation', 0) for d in degradations]

        analysis = {
            'average_sharpe_degradation': np.mean(sharpe_degradations),
            'sharpe_degradation_std': np.std(sharpe_degradations),
            'proportion_positive_sharpe': sum(1 for d in sharpe_degradations if d > 0) / len(sharpe_degradations),
            'worst_sharpe_degradation': min(sharpe_degradations),
            'overfitting_risk': 'High' if np.mean(sharpe_degradations) < -0.3 else 'Medium' if np.mean(sharpe_degradations) < -0.1 else 'Low'
        }

        return analysis

    def _analyze_by_regime(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        regime_performance = {}

        for result in window_results:
            regime = result.get('market_regime', 'unknown')
            oos_perf = result.get('out_of_sample_performance', {})

            if regime not in regime_performance:
                regime_performance[regime] = []

            regime_performance[regime].append(oos_perf)

        # Calculate statistics by regime
        regime_stats = {}
        for regime, performances in regime_performance.items():
            if not performances:
                continue

            sharpe_ratios = [p.get('sharpe_ratio', np.nan) for p in performances]
            sharpe_ratios = [x for x in sharpe_ratios if not np.isnan(x)]

            if sharpe_ratios:
                regime_stats[regime] = {
                    'count': len(performances),
                    'mean_sharpe': np.mean(sharpe_ratios),
                    'std_sharpe': np.std(sharpe_ratios),
                    'win_rate': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios)
                }

        return regime_stats

    def _check_validation_criteria(self, summary_stats: Dict) -> bool:
        """Check if strategy passes validation criteria"""
        if not summary_stats:
            return False

        oos_sharpe = summary_stats.get('out_of_sample_sharpe', {}).get('mean', 0)
        consistency = summary_stats.get('sharpe_degradation', {}).get('consistency', 0)

        # Validation criteria
        criteria = [
            oos_sharpe > 0.5,  # Minimum Sharpe ratio
            consistency > 0.6,  # At least 60% of windows profitable
            summary_stats.get('successful_windows', 0) >= 3  # At least 3 successful windows
        ]

        return all(criteria)

    def _save_results(self, results: Dict):
        """Save detailed results"""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rolling_validation_{timestamp}.json"

        def convert_for_json(obj):
            """Convert numpy types for JSON serialization"""
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        # Clean results for JSON
        cleaned_results = self._deep_convert(results, convert_for_json)

        with open(results_dir / filename, 'w') as f:
            json.dump(cleaned_results, f, indent=2, default=str)

        print(f"Results saved to {results_dir / filename}")

    def _deep_convert(self, obj, converter):
        """Recursively apply converter to nested structures"""
        if isinstance(obj, dict):
            return {k: self._deep_convert(v, converter) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_convert(v, converter) for v in obj]
        else:
            return converter(obj)

    def _generate_plots(self, results: Dict):
        """Generate validation plots"""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Performance over time
        self._plot_performance_over_time(results, results_dir)

        # In-sample vs out-of-sample
        self._plot_is_vs_oos(results, results_dir)

        # Regime analysis
        self._plot_regime_analysis(results, results_dir)

    def _plot_performance_over_time(self, results: Dict, results_dir: Path):
        """Plot performance metrics over time"""
        window_results = results.get('window_results', [])
        if not window_results:
            return

        # Extract data
        test_dates = []
        oos_sharpe = []
        is_sharpe = []

        for result in window_results:
            window_info = result.get('window_info', {})
            test_start = pd.to_datetime(window_info.get('test_start'))
            if test_start:
                test_dates.append(test_start)
                oos_sharpe.append(result.get('out_of_sample_performance', {}).get('sharpe_ratio', np.nan))
                is_sharpe.append(result.get('in_sample_performance', {}).get('sharpe_ratio', np.nan))

        if not test_dates:
            return

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, is_sharpe, 'o-', label='In-Sample Sharpe', alpha=0.7)
        plt.plot(test_dates, oos_sharpe, 'o-', label='Out-of-Sample Sharpe', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Test Period Start')
        plt.ylabel('Sharpe Ratio')
        plt.title('Performance Over Time: Rolling Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / 'performance_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_is_vs_oos(self, results: Dict, results_dir: Path):
        """Plot in-sample vs out-of-sample performance"""
        window_results = results.get('window_results', [])
        if not window_results:
            return

        is_sharpe = []
        oos_sharpe = []

        for result in window_results:
            is_perf = result.get('in_sample_performance', {}).get('sharpe_ratio', np.nan)
            oos_perf = result.get('out_of_sample_performance', {}).get('sharpe_ratio', np.nan)

            if not (np.isnan(is_perf) or np.isnan(oos_perf)):
                is_sharpe.append(is_perf)
                oos_sharpe.append(oos_perf)

        if len(is_sharpe) < 2:
            return

        # Create scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(is_sharpe, oos_sharpe, alpha=0.7, s=60)

        # Add diagonal line (perfect consistency)
        min_val = min(min(is_sharpe), min(oos_sharpe))
        max_val = max(max(is_sharpe), max(oos_sharpe))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Consistency')

        plt.xlabel('In-Sample Sharpe Ratio')
        plt.ylabel('Out-of-Sample Sharpe Ratio')
        plt.title('In-Sample vs Out-of-Sample Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'is_vs_oos_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_regime_analysis(self, results: Dict, results_dir: Path):
        """Plot performance by market regime"""
        regime_analysis = results.get('regime_analysis', {})
        if not regime_analysis:
            return

        regimes = list(regime_analysis.keys())
        mean_sharpe = [regime_analysis[r]['mean_sharpe'] for r in regimes]
        win_rates = [regime_analysis[r]['win_rate'] for r in regimes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Mean Sharpe by regime
        bars1 = ax1.bar(regimes, mean_sharpe, alpha=0.7)
        ax1.set_ylabel('Mean Sharpe Ratio')
        ax1.set_title('Performance by Market Regime')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Win rate by regime
        bars2 = ax2.bar(regimes, win_rates, alpha=0.7, color='orange')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate by Market Regime')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(results_dir / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example usage
def example_optimizer(data: pd.DataFrame) -> Dict[str, Any]:
    """Example parameter optimizer"""
    # Simple grid search for momentum strategy
    best_params = {'lookback': 20, 'threshold': 0.02}
    best_sharpe = -999

    for lookback in [10, 15, 20, 25, 30]:
        for threshold in [0.01, 0.015, 0.02, 0.025, 0.03]:
            returns = example_strategy(data, lookback, threshold)
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'lookback': lookback, 'threshold': threshold}

    return best_params

def example_strategy(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.02) -> pd.Series:
    """Example strategy for testing"""
    returns = data['close'].pct_change()
    momentum = returns.rolling(lookback).mean()

    signals = np.where(momentum > threshold, 1, np.where(momentum < -threshold, -1, 0))
    strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns

    return strategy_returns.fillna(0)

if __name__ == "__main__":
    # Example usage
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    # Create sample data with regime changes
    returns = np.random.randn(len(dates)) * 0.01
    returns[:500] += 0.0005  # Bull market
    returns[500:1000] -= 0.0003  # Bear market
    returns[1000:] += 0.0002  # Recovery

    sample_data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns))
    }, index=dates)

    # Configure validation
    config = RollingConfig(
        train_period_months=24,  # 2 years training
        test_period_months=6,    # 6 months testing
        step_months=6,           # 6 months step
        reoptimize_params=True
    )

    # Run validation
    validator = RollingValidator(config)
    results = validator.validate_strategy(
        example_strategy,
        sample_data,
        example_optimizer,
        {'lookback': 20, 'threshold': 0.02}
    )

    print("Rolling Validation Complete!")
    print(f"Validation passed: {results['validation_passed']}")
    summary = results['summary_stats']
    print(f"Out-of-sample Sharpe: {summary['out_of_sample_sharpe']['mean']:.3f}")
    print(f"Performance consistency: {summary['sharpe_degradation']['consistency']:.1%}")