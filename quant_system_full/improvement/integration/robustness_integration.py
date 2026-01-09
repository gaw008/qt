#!/usr/bin/env python3
"""
Robustness Testing Integration Module

Integrates all Phase 2 robustness testing components with the existing trading system:
- Purged K-Fold cross-validation
- Parameter sensitivity analysis
- Rolling out-of-sample validation
- Deflated Sharpe ratio calculations

This module provides a unified interface for comprehensive strategy validation.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import warnings

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "improvement" / "backtest"))
sys.path.insert(0, str(project_root / "improvement" / "analysis"))
sys.path.insert(0, str(project_root / "improvement" / "validation"))

# Import robustness components
try:
    from purged_kfold import validate_strategy_robustness, CVConfig, deflated_sharpe_ratio
    from parameter_sensitivity import ParameterSensitivityAnalyzer, SensitivityConfig, ParameterRange
    from rolling_validation import RollingValidator, RollingConfig
except ImportError as e:
    print(f"Warning: Could not import robustness modules: {e}")

class RobustnessTestSuite:
    """
    Comprehensive robustness testing suite for trading strategies

    Combines all Phase 2 validation methods into a single interface
    that integrates with the existing trading system.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.results = {}
        self.reports_dir = Path("reports/robustness")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _default_config(self) -> Dict:
        """Default configuration for robustness testing"""
        return {
            'cv_config': {
                'n_splits': 5,
                'purge_pct': 0.02,
                'embargo_pct': 0.01,
                'combinatorial': True,
                'max_combinations': 8
            },
            'sensitivity_config': {
                'n_jobs': 4,
                'save_plots': True,
                'plot_dir': str(self.reports_dir / "sensitivity")
            },
            'rolling_config': {
                'train_period_months': 36,
                'test_period_months': 6,
                'step_months': 6,
                'reoptimize_params': True,
                'results_dir': str(self.reports_dir / "rolling")
            },
            'deflated_sharpe_trials': 20,  # Number of strategy trials for DSR
            'min_data_points': 252  # Minimum data points for reliable testing
        }

    def validate_current_strategies(self, data_source: str = "yahoo") -> Dict[str, Any]:
        """
        Validate current trading strategies using all robustness tests

        Args:
            data_source: Data source to use ("yahoo", "tiger", "cache")

        Returns:
            Comprehensive validation results
        """
        print("Starting Comprehensive Strategy Robustness Validation...")
        print("=" * 70)

        # Load market data
        market_data = self._load_market_data(data_source)
        if market_data is None or len(market_data) < self.config['min_data_points']:
            raise ValueError(f"Insufficient market data: need {self.config['min_data_points']} points")

        # Get current strategy configurations
        strategy_configs = self._get_current_strategy_configs()

        all_results = {}

        for strategy_name, strategy_config in strategy_configs.items():
            print(f"\nValidating strategy: {strategy_name}")
            print("-" * 50)

            try:
                # Create strategy function
                strategy_func = self._create_strategy_function(strategy_name, strategy_config)

                # Run comprehensive validation
                strategy_results = self._validate_single_strategy(
                    strategy_name, strategy_func, market_data, strategy_config
                )

                all_results[strategy_name] = strategy_results

            except Exception as e:
                print(f"Strategy {strategy_name} validation failed: {str(e)}")
                all_results[strategy_name] = {'error': str(e)}

        # Generate consolidated report
        consolidated_results = self._consolidate_results(all_results)

        # Save comprehensive results
        self._save_comprehensive_results(consolidated_results)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(consolidated_results)

        self.results = {
            'individual_strategies': all_results,
            'consolidated_results': consolidated_results,
            'executive_summary': executive_summary,
            'validation_timestamp': datetime.now().isoformat()
        }

        return self.results

    def _load_market_data(self, data_source: str) -> Optional[pd.DataFrame]:
        """Load market data for validation"""
        try:
            if data_source == "cache":
                # Try to load from data cache
                cache_dir = project_root / "data_cache"
                if cache_dir.exists():
                    # Load a representative stock for testing
                    cache_files = list(cache_dir.glob("*.csv"))
                    if cache_files:
                        data = pd.read_csv(cache_files[0], index_col=0, parse_dates=True)
                        return data.tail(1000)  # Last 1000 days

            elif data_source == "yahoo":
                # Load data using yahoo finance
                try:
                    import yfinance as yf
                    # Use SPY as market proxy
                    spy = yf.download("SPY", start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d"))
                    return spy
                except ImportError:
                    print("yfinance not available, trying alternative")

            # Fallback: generate synthetic data for testing
            print("Using synthetic data for robustness testing")
            return self._generate_synthetic_data()

        except Exception as e:
            print(f"Error loading market data: {e}")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        # Create 3 years of synthetic daily data
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
        np.random.seed(42)

        # Simulate realistic market returns with regime changes
        returns = np.random.randn(len(dates)) * 0.015
        returns[:300] += 0.0005  # Bull market
        returns[300:600] -= 0.0003  # Bear market
        returns[600:] += 0.0002  # Recovery

        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.005),
            'High': prices * (1 + abs(np.random.randn(len(dates))) * 0.01),
            'Low': prices * (1 - abs(np.random.randn(len(dates))) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)

        return data

    def _get_current_strategy_configs(self) -> Dict[str, Dict]:
        """Get current strategy configurations from the system"""
        # This would integrate with the actual strategy configuration
        # For now, return test configurations based on existing strategies

        strategies = {
            'ValueMomentum': {
                'base_params': {
                    'value_weight': 0.4,
                    'momentum_weight': 0.6,
                    'lookback_period': 60
                },
                'param_ranges': {
                    'value_weight': ParameterRange('value_weight', 0.4, 0.3, 5),
                    'momentum_weight': ParameterRange('momentum_weight', 0.6, 0.3, 5),
                    'lookback_period': ParameterRange('lookback_period', 60, 0.4, 5, discrete=True)
                }
            },
            'TechnicalBreakout': {
                'base_params': {
                    'sma_short': 20,
                    'sma_long': 50,
                    'volume_threshold': 1.5
                },
                'param_ranges': {
                    'sma_short': ParameterRange('sma_short', 20, 0.5, 5, discrete=True),
                    'sma_long': ParameterRange('sma_long', 50, 0.4, 5, discrete=True),
                    'volume_threshold': ParameterRange('volume_threshold', 1.5, 0.3, 5)
                }
            }
        }

        return strategies

    def _create_strategy_function(self, strategy_name: str, config: Dict) -> Callable:
        """Create strategy function for testing"""
        def strategy_func(data: pd.DataFrame, **params) -> pd.Series:
            """Generic strategy function"""
            if strategy_name == 'ValueMomentum':
                return self._value_momentum_strategy(data, **params)
            elif strategy_name == 'TechnicalBreakout':
                return self._technical_breakout_strategy(data, **params)
            else:
                # Default simple momentum strategy
                return self._simple_momentum_strategy(data, **params)

        return strategy_func

    def _value_momentum_strategy(self, data: pd.DataFrame, value_weight: float = 0.4,
                                momentum_weight: float = 0.6, lookback_period: int = 60) -> pd.Series:
        """Simplified value momentum strategy"""
        returns = data['Close'].pct_change()

        # Momentum component
        momentum = returns.rolling(lookback_period).mean()

        # Value component (using simple price-to-average ratio)
        avg_price = data['Close'].rolling(lookback_period * 2).mean()
        value_signal = (avg_price - data['Close']) / avg_price

        # Combined signal
        combined_signal = momentum_weight * momentum + value_weight * value_signal

        # Generate positions
        positions = np.where(combined_signal > combined_signal.quantile(0.7), 1,
                           np.where(combined_signal < combined_signal.quantile(0.3), -1, 0))

        strategy_returns = pd.Series(positions, index=data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def _technical_breakout_strategy(self, data: pd.DataFrame, sma_short: int = 20,
                                   sma_long: int = 50, volume_threshold: float = 1.5) -> pd.Series:
        """Simplified technical breakout strategy"""
        returns = data['Close'].pct_change()

        # Moving averages
        short_ma = data['Close'].rolling(sma_short).mean()
        long_ma = data['Close'].rolling(sma_long).mean()

        # Volume signal
        avg_volume = data['Volume'].rolling(20).mean()
        volume_signal = data['Volume'] > (avg_volume * volume_threshold)

        # Breakout signal
        ma_signal = short_ma > long_ma

        # Combined signal
        positions = np.where(ma_signal & volume_signal, 1,
                           np.where(~ma_signal, -1, 0))

        strategy_returns = pd.Series(positions, index=data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def _simple_momentum_strategy(self, data: pd.DataFrame, lookback: int = 20,
                                threshold: float = 0.02) -> pd.Series:
        """Simple momentum strategy for testing"""
        returns = data['Close'].pct_change()
        momentum = returns.rolling(lookback).mean()

        positions = np.where(momentum > threshold, 1,
                           np.where(momentum < -threshold, -1, 0))

        strategy_returns = pd.Series(positions, index=data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def _validate_single_strategy(self, strategy_name: str, strategy_func: Callable,
                                 data: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """Comprehensive validation of a single strategy"""
        validation_results = {
            'strategy_name': strategy_name,
            'data_period': f"{data.index[0]} to {data.index[-1]}",
            'data_points': len(data)
        }

        # 1. Cross-validation with Purged K-Fold
        print(f"  Running Purged K-Fold validation...")
        try:
            cv_config = CVConfig(**self.config['cv_config'])

            # Create a wrapper for the strategy function
            def cv_strategy_func():
                class StrategyModel:
                    def __init__(self):
                        self.params = config['base_params']

                    def fit(self, X, y, **kwargs):
                        pass

                    def predict(self, X):
                        # Convert DataFrame back for strategy
                        strategy_data = pd.DataFrame({
                            'Close': X.iloc[:, 0],  # Assume first column is close price
                            'Volume': X.iloc[:, 1] if X.shape[1] > 1 else pd.Series(1000000, index=X.index)
                        })
                        strategy_returns = strategy_func(strategy_data, **self.params)
                        return strategy_returns.values

                return StrategyModel()

            # Prepare data for CV
            cv_data = pd.DataFrame({
                'Close': data['Close'],
                'Volume': data['Volume'],
                'returns': data['Close'].pct_change()
            }).dropna()

            X_cv = cv_data[['Close', 'Volume']]
            y_cv = cv_data['returns']

            cv_results = validate_strategy_robustness(
                cv_strategy_func, cv_data, 'returns', ['Close', 'Volume'], cv_config
            )

            validation_results['cross_validation'] = cv_results

        except Exception as e:
            print(f"    Cross-validation failed: {str(e)}")
            validation_results['cross_validation'] = {'error': str(e)}

        # 2. Parameter Sensitivity Analysis
        print(f"  Running parameter sensitivity analysis...")
        try:
            sensitivity_config = SensitivityConfig(
                parameter_ranges=config['param_ranges'],
                **self.config['sensitivity_config']
            )

            analyzer = ParameterSensitivityAnalyzer(sensitivity_config)
            sensitivity_results = analyzer.analyze_strategy(
                strategy_func, data, config['base_params']
            )

            validation_results['parameter_sensitivity'] = sensitivity_results

        except Exception as e:
            print(f"    Parameter sensitivity failed: {str(e)}")
            validation_results['parameter_sensitivity'] = {'error': str(e)}

        # 3. Rolling Out-of-Sample Validation
        print(f"  Running rolling validation...")
        try:
            rolling_config = RollingConfig(**self.config['rolling_config'])

            # Create parameter optimizer
            def param_optimizer(train_data):
                # Simple grid search optimization
                best_params = config['base_params'].copy()
                best_sharpe = -999

                # Test a few parameter combinations
                param_ranges = config['param_ranges']
                for param_name, param_range in param_ranges.items():
                    values = param_range.get_values()
                    mid_values = values[len(values)//2-1:len(values)//2+2]  # Test around middle

                    for value in mid_values:
                        test_params = config['base_params'].copy()
                        test_params[param_name] = value

                        returns = strategy_func(train_data, **test_params)
                        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params[param_name] = value

                return best_params

            validator = RollingValidator(rolling_config)
            rolling_results = validator.validate_strategy(
                strategy_func, data, param_optimizer, config['base_params']
            )

            validation_results['rolling_validation'] = rolling_results

        except Exception as e:
            print(f"    Rolling validation failed: {str(e)}")
            validation_results['rolling_validation'] = {'error': str(e)}

        # 4. Calculate Deflated Sharpe Ratio
        print(f"  Calculating Deflated Sharpe Ratio...")
        try:
            strategy_returns = strategy_func(data, **config['base_params'])
            dsr_results = deflated_sharpe_ratio(
                strategy_returns,
                trials=self.config['deflated_sharpe_trials']
            )

            validation_results['deflated_sharpe'] = dsr_results

        except Exception as e:
            print(f"    DSR calculation failed: {str(e)}")
            validation_results['deflated_sharpe'] = {'error': str(e)}

        return validation_results

    def _consolidate_results(self, all_results: Dict) -> Dict[str, Any]:
        """Consolidate results across all strategies"""
        consolidation = {
            'total_strategies': len(all_results),
            'successful_validations': 0,
            'overall_robustness_score': 0,
            'strategy_rankings': [],
            'best_strategy': None,
            'robustness_summary': {}
        }

        strategy_scores = []

        for strategy_name, results in all_results.items():
            if 'error' in results:
                continue

            consolidation['successful_validations'] += 1

            # Calculate robustness score for this strategy
            robustness_score = self._calculate_strategy_robustness_score(results)
            strategy_scores.append((strategy_name, robustness_score))

            # Add to rankings
            consolidation['strategy_rankings'].append({
                'strategy': strategy_name,
                'robustness_score': robustness_score,
                'key_metrics': self._extract_key_metrics(results)
            })

        # Sort by robustness score
        consolidation['strategy_rankings'].sort(key=lambda x: x['robustness_score'], reverse=True)

        if strategy_scores:
            consolidation['overall_robustness_score'] = np.mean([score for _, score in strategy_scores])
            consolidation['best_strategy'] = max(strategy_scores, key=lambda x: x[1])[0]

        return consolidation

    def _calculate_strategy_robustness_score(self, results: Dict) -> float:
        """Calculate overall robustness score for a strategy"""
        scores = []

        # Cross-validation score
        cv_results = results.get('cross_validation', {})
        if 'summary' in cv_results and cv_results['summary'].get('is_robust', False):
            scores.append(0.8)  # High weight for passing cross-validation
        elif 'summary' in cv_results:
            dsr = cv_results['summary'].get('dsr', 0)
            scores.append(max(0, min(1, (dsr + 1) / 2)))  # Normalize DSR to 0-1

        # Parameter sensitivity score
        sensitivity_results = results.get('parameter_sensitivity', {})
        if 'robustness_scores' in sensitivity_results:
            robustness_scores = sensitivity_results['robustness_scores']
            avg_robustness = np.mean([scores['combined_score'] for scores in robustness_scores.values()])
            scores.append(avg_robustness)

        # Rolling validation score
        rolling_results = results.get('rolling_validation', {})
        if rolling_results.get('validation_passed', False):
            scores.append(0.9)
        elif 'summary_stats' in rolling_results:
            consistency = rolling_results['summary_stats'].get('sharpe_degradation', {}).get('consistency', 0)
            scores.append(consistency)

        # Deflated Sharpe score
        dsr_results = results.get('deflated_sharpe', {})
        if dsr_results.get('is_significant', False):
            scores.append(0.8)
        else:
            dsr = dsr_results.get('deflated_sharpe_ratio', 0)
            scores.append(max(0, min(1, (dsr + 1) / 2)))

        return np.mean(scores) if scores else 0

    def _extract_key_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract key metrics from validation results"""
        key_metrics = {}

        # From deflated Sharpe
        dsr_results = results.get('deflated_sharpe', {})
        key_metrics['sharpe_ratio'] = dsr_results.get('sharpe_ratio', np.nan)
        key_metrics['deflated_sharpe'] = dsr_results.get('deflated_sharpe_ratio', np.nan)

        # From rolling validation
        rolling_results = results.get('rolling_validation', {})
        if 'summary_stats' in rolling_results:
            summary = rolling_results['summary_stats']
            key_metrics['oos_sharpe_mean'] = summary.get('out_of_sample_sharpe', {}).get('mean', np.nan)
            key_metrics['consistency'] = summary.get('sharpe_degradation', {}).get('consistency', np.nan)

        # From parameter sensitivity
        sensitivity_results = results.get('parameter_sensitivity', {})
        if 'summary' in sensitivity_results:
            key_metrics['base_performance'] = sensitivity_results['summary'].get('mean_sharpe', np.nan)

        return key_metrics

    def _generate_executive_summary(self, consolidated_results: Dict) -> Dict[str, Any]:
        """Generate executive summary of robustness validation"""
        summary = {
            'validation_date': datetime.now().strftime("%Y-%m-%d"),
            'overall_assessment': 'UNKNOWN',
            'key_findings': [],
            'recommendations': [],
            'risk_alerts': []
        }

        # Overall assessment
        overall_score = consolidated_results.get('overall_robustness_score', 0)
        if overall_score >= 0.7:
            summary['overall_assessment'] = 'ROBUST'
            summary['key_findings'].append(f"Strategies show strong robustness (score: {overall_score:.2f})")
        elif overall_score >= 0.5:
            summary['overall_assessment'] = 'MODERATE'
            summary['key_findings'].append(f"Strategies show moderate robustness (score: {overall_score:.2f})")
        else:
            summary['overall_assessment'] = 'WEAK'
            summary['risk_alerts'].append(f"Strategies show weak robustness (score: {overall_score:.2f})")

        # Best strategy
        best_strategy = consolidated_results.get('best_strategy')
        if best_strategy:
            summary['recommendations'].append(f"Focus on {best_strategy} strategy for implementation")

        # Strategy-specific findings
        rankings = consolidated_results.get('strategy_rankings', [])
        for ranking in rankings[:2]:  # Top 2 strategies
            strategy = ranking['strategy']
            score = ranking['robustness_score']
            if score >= 0.6:
                summary['key_findings'].append(f"{strategy} strategy passes robustness tests")
            else:
                summary['risk_alerts'].append(f"{strategy} strategy needs parameter optimization")

        return summary

    def _save_comprehensive_results(self, results: Dict):
        """Save comprehensive results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robustness_validation_{timestamp}.json"

        def convert_for_json(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return convert_for_json(d)

        cleaned_results = clean_dict(results)

        with open(self.reports_dir / filename, 'w') as f:
            json.dump(cleaned_results, f, indent=2, default=str)

        print(f"Comprehensive results saved to {self.reports_dir / filename}")

    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        if not self.results:
            return "No validation results available"

        exec_summary = self.results.get('executive_summary', {})
        consolidated = self.results.get('consolidated_results', {})

        report = []
        report.append("STRATEGY ROBUSTNESS VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Validation Date: {exec_summary.get('validation_date', 'Unknown')}")
        report.append(f"Overall Assessment: {exec_summary.get('overall_assessment', 'Unknown')}")
        report.append("")

        # Key findings
        report.append("KEY FINDINGS:")
        for finding in exec_summary.get('key_findings', []):
            report.append(f"  ? {finding}")

        # Recommendations
        if exec_summary.get('recommendations'):
            report.append("\nRECOMMENDATIONS:")
            for rec in exec_summary['recommendations']:
                report.append(f"  ? {rec}")

        # Risk alerts
        if exec_summary.get('risk_alerts'):
            report.append("\nRISK ALERTS:")
            for alert in exec_summary['risk_alerts']:
                report.append(f"  [WARNING]  {alert}")

        # Strategy rankings
        rankings = consolidated.get('strategy_rankings', [])
        if rankings:
            report.append("\nSTRATEGY RANKINGS:")
            for i, ranking in enumerate(rankings, 1):
                strategy = ranking['strategy']
                score = ranking['robustness_score']
                report.append(f"  {i}. {strategy}: {score:.3f}")

        return "\n".join(report)

# Integration with existing system
def run_phase2_robustness_validation():
    """Main function to run Phase 2 robustness validation"""
    print("Phase 2: Robustness Testing & Validation")
    print("=" * 50)

    try:
        # Initialize test suite
        test_suite = RobustnessTestSuite()

        # Run comprehensive validation
        results = test_suite.validate_current_strategies()

        # Print summary
        print("\n" + "=" * 50)
        print("VALIDATION COMPLETE")
        print("=" * 50)
        print(test_suite.generate_summary_report())

        return results

    except Exception as e:
        print(f"Robustness validation failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the validation
    results = run_phase2_robustness_validation()

    if results:
        print(f"\nDetailed results available in: reports/robustness/")
        print("Phase 2 robustness testing completed successfully!")
    else:
        print("Phase 2 robustness testing failed - check logs for details")