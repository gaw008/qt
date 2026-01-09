#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis Framework

Analyzes how strategy performance varies with parameter changes to:
1. Identify robust parameter ranges
2. Detect overfitting to specific parameter values
3. Generate sensitivity heatmaps and reports
4. Calculate elasticity metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

@dataclass
class ParameterRange:
    """Define parameter variation range"""
    name: str
    base_value: Union[int, float]
    variation_pct: float = 0.20  # +/-20% variation
    steps: int = 5
    log_scale: bool = False
    discrete: bool = False
    custom_values: Optional[List] = None

    def get_values(self) -> List[Union[int, float]]:
        """Generate parameter values to test"""
        if self.custom_values:
            return self.custom_values

        if self.discrete:
            # For discrete parameters, use integer steps
            base = int(self.base_value)
            variation = max(1, int(base * self.variation_pct))
            return list(range(max(1, base - variation), base + variation + 1))

        # Continuous parameters
        if self.log_scale:
            log_base = np.log10(self.base_value)
            log_range = log_base * self.variation_pct
            log_values = np.linspace(log_base - log_range, log_base + log_range, self.steps)
            return [10 ** lv for lv in log_values]
        else:
            variation = self.base_value * self.variation_pct
            return list(np.linspace(self.base_value - variation, self.base_value + variation, self.steps))

@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis"""
    parameter_ranges: Dict[str, ParameterRange] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=lambda: ['sharpe_ratio', 'max_drawdown', 'total_return'])
    n_jobs: int = 4
    save_plots: bool = True
    plot_dir: str = "reports/sensitivity"
    min_samples: int = 100  # Minimum samples required for analysis

class ParameterSensitivityAnalyzer:
    """
    Analyzes parameter sensitivity for trading strategies

    Features:
    - Grid search across parameter combinations
    - Individual parameter sensitivity curves
    - Interaction effects between parameters
    - Robustness scoring and ranking
    """

    def __init__(self, config: SensitivityConfig):
        self.config = config
        self.results = {}
        self.base_performance = None

    def analyze_strategy(self, strategy_func: Callable, data: pd.DataFrame,
                        base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive parameter sensitivity analysis

        Args:
            strategy_func: Function that returns strategy results given parameters
            data: Market data for backtesting
            base_params: Base parameter configuration

        Returns:
            Complete sensitivity analysis results
        """
        print("Starting Parameter Sensitivity Analysis...")
        print(f"Base parameters: {base_params}")

        # Calculate base performance
        print("Calculating base performance...")
        self.base_performance = self._evaluate_strategy(strategy_func, data, base_params)
        print(f"Base Sharpe: {self.base_performance.get('sharpe_ratio', 0):.3f}")

        # Individual parameter sensitivity
        print("Analyzing individual parameter sensitivity...")
        individual_results = self._analyze_individual_parameters(strategy_func, data, base_params)

        # Parameter interaction analysis
        print("Analyzing parameter interactions...")
        interaction_results = self._analyze_parameter_interactions(strategy_func, data, base_params)

        # Robustness scoring
        print("Calculating robustness scores...")
        robustness_scores = self._calculate_robustness_scores(individual_results)

        # Generate comprehensive results
        results = {
            'base_performance': self.base_performance,
            'individual_sensitivity': individual_results,
            'parameter_interactions': interaction_results,
            'robustness_scores': robustness_scores,
            'recommendations': self._generate_recommendations(individual_results, robustness_scores),
            'config': self.config
        }

        # Generate visualizations
        if self.config.save_plots:
            self._generate_plots(results)

        self.results = results
        return results

    def _analyze_individual_parameters(self, strategy_func: Callable, data: pd.DataFrame,
                                     base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze each parameter individually"""
        individual_results = {}

        for param_name, param_range in self.config.parameter_ranges.items():
            print(f"  Analyzing {param_name}...")

            param_values = param_range.get_values()
            param_results = []

            # Test each parameter value
            for value in param_values:
                test_params = base_params.copy()
                test_params[param_name] = value

                performance = self._evaluate_strategy(strategy_func, data, test_params)
                performance['parameter_value'] = value

                param_results.append(performance)

            # Calculate sensitivity metrics
            sensitivity_metrics = self._calculate_sensitivity_metrics(param_results, param_name)

            individual_results[param_name] = {
                'parameter_range': param_range,
                'results': param_results,
                'sensitivity_metrics': sensitivity_metrics
            }

        return individual_results

    def _analyze_parameter_interactions(self, strategy_func: Callable, data: pd.DataFrame,
                                      base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interactions between parameter pairs"""
        param_names = list(self.config.parameter_ranges.keys())
        interaction_results = {}

        # Analyze all parameter pairs
        for i, param1 in enumerate(param_names):
            for param2 in param_names[i+1:]:
                print(f"  Analyzing {param1} x {param2} interaction...")

                interaction_key = f"{param1}_x_{param2}"

                # Get parameter value grids
                values1 = self.config.parameter_ranges[param1].get_values()
                values2 = self.config.parameter_ranges[param2].get_values()

                # Limit grid size for performance
                if len(values1) * len(values2) > 25:
                    values1 = values1[::max(1, len(values1)//5)]
                    values2 = values2[::max(1, len(values2)//5)]

                # Grid search
                grid_results = []
                for v1, v2 in itertools.product(values1, values2):
                    test_params = base_params.copy()
                    test_params[param1] = v1
                    test_params[param2] = v2

                    performance = self._evaluate_strategy(strategy_func, data, test_params)
                    performance[param1] = v1
                    performance[param2] = v2

                    grid_results.append(performance)

                # Calculate interaction metrics
                interaction_metrics = self._calculate_interaction_metrics(grid_results, param1, param2)

                interaction_results[interaction_key] = {
                    'param1': param1,
                    'param2': param2,
                    'grid_results': grid_results,
                    'interaction_metrics': interaction_metrics
                }

        return interaction_results

    def _evaluate_strategy(self, strategy_func: Callable, data: pd.DataFrame,
                          params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy with given parameters"""
        try:
            # Run strategy
            results = strategy_func(data, **params)

            # Calculate performance metrics
            if isinstance(results, pd.Series):
                returns = results
            elif isinstance(results, dict) and 'returns' in results:
                returns = results['returns']
            else:
                raise ValueError("Strategy function must return returns series or dict with 'returns' key")

            # Calculate standard metrics
            performance = self._calculate_performance_metrics(returns)

            return performance

        except Exception as e:
            warnings.warn(f"Strategy evaluation failed with params {params}: {str(e)}")
            return {metric: np.nan for metric in self.config.metrics}

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) < self.config.min_samples:
            return {metric: np.nan for metric in self.config.metrics}

        metrics = {}

        # Returns-based metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        annualized_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()

        # Volatility
        volatility = annualized_vol

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0

        # Standard metrics
        metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        })

        return metrics

    def _calculate_sensitivity_metrics(self, param_results: List[Dict], param_name: str) -> Dict[str, float]:
        """Calculate sensitivity metrics for a parameter"""
        # Extract values and performance
        values = [r['parameter_value'] for r in param_results]
        performances = {metric: [r.get(metric, np.nan) for r in param_results]
                       for metric in self.config.metrics}

        sensitivity_metrics = {}

        for metric in self.config.metrics:
            metric_values = np.array(performances[metric])

            # Remove NaN values
            valid_mask = ~np.isnan(metric_values)
            if valid_mask.sum() < 2:
                continue

            valid_values = np.array(values)[valid_mask]
            valid_metrics = metric_values[valid_mask]

            # Calculate sensitivity
            param_range = max(valid_values) - min(valid_values)
            metric_range = max(valid_metrics) - min(valid_metrics)

            # Elasticity: % change in metric / % change in parameter
            if param_range > 0 and min(valid_values) > 0:
                param_pct_change = param_range / min(valid_values)
                base_metric = self.base_performance.get(metric, 0)
                if base_metric != 0:
                    metric_pct_change = metric_range / abs(base_metric)
                    elasticity = metric_pct_change / param_pct_change
                else:
                    elasticity = 0
            else:
                elasticity = 0

            # Volatility of performance across parameter range
            metric_volatility = np.std(valid_metrics)

            # Robustness: percentage of parameter values that beat base performance
            base_value = self.base_performance.get(metric, 0)
            if metric in ['sharpe_ratio', 'total_return', 'annualized_return', 'win_rate', 'calmar_ratio', 'sortino_ratio']:
                better_count = (valid_metrics > base_value).sum()
            else:  # For drawdown and volatility, lower is better
                better_count = (valid_metrics < base_value).sum()

            robustness = better_count / len(valid_metrics) if len(valid_metrics) > 0 else 0

            sensitivity_metrics[f'{metric}_elasticity'] = elasticity
            sensitivity_metrics[f'{metric}_volatility'] = metric_volatility
            sensitivity_metrics[f'{metric}_robustness'] = robustness

        return sensitivity_metrics

    def _calculate_interaction_metrics(self, grid_results: List[Dict], param1: str, param2: str) -> Dict[str, float]:
        """Calculate interaction effect metrics"""
        if len(grid_results) < 4:
            return {}

        df = pd.DataFrame(grid_results)

        interaction_metrics = {}

        for metric in self.config.metrics:
            if metric not in df.columns:
                continue

            # Create pivot table
            try:
                pivot = df.pivot_table(values=metric, index=param1, columns=param2, aggfunc='mean')

                # Calculate interaction strength (variance explained by interaction)
                overall_mean = df[metric].mean()
                param1_effects = df.groupby(param1)[metric].mean() - overall_mean
                param2_effects = df.groupby(param2)[metric].mean() - overall_mean

                # Total variation
                total_var = df[metric].var()

                # Interaction variation (residual after removing main effects)
                interaction_var = 0
                for i, row in df.iterrows():
                    expected = overall_mean + param1_effects.get(row[param1], 0) + param2_effects.get(row[param2], 0)
                    interaction_var += (row[metric] - expected) ** 2

                interaction_var /= len(df)

                # Interaction strength
                interaction_strength = interaction_var / total_var if total_var > 0 else 0

                interaction_metrics[f'{metric}_interaction_strength'] = interaction_strength

            except Exception as e:
                warnings.warn(f"Could not calculate interaction metrics for {metric}: {str(e)}")

        return interaction_metrics

    def _calculate_robustness_scores(self, individual_results: Dict) -> Dict[str, float]:
        """Calculate overall robustness scores for parameters"""
        robustness_scores = {}

        for param_name, param_data in individual_results.items():
            sensitivity_metrics = param_data['sensitivity_metrics']

            # Aggregate robustness across metrics
            robustness_values = []
            for key, value in sensitivity_metrics.items():
                if 'robustness' in key and not np.isnan(value):
                    robustness_values.append(value)

            # Overall robustness score
            overall_robustness = np.mean(robustness_values) if robustness_values else 0

            # Stability score (inverse of elasticity)
            elasticity_values = []
            for key, value in sensitivity_metrics.items():
                if 'elasticity' in key and not np.isnan(value):
                    elasticity_values.append(abs(value))

            stability_score = 1 / (1 + np.mean(elasticity_values)) if elasticity_values else 0

            # Combined robustness score
            combined_score = (overall_robustness + stability_score) / 2

            robustness_scores[param_name] = {
                'overall_robustness': overall_robustness,
                'stability_score': stability_score,
                'combined_score': combined_score
            }

        return robustness_scores

    def _generate_recommendations(self, individual_results: Dict, robustness_scores: Dict) -> List[Dict]:
        """Generate parameter tuning recommendations"""
        recommendations = []

        # Rank parameters by robustness
        param_ranking = sorted(robustness_scores.items(),
                             key=lambda x: x[1]['combined_score'], reverse=True)

        for param_name, scores in param_ranking:
            param_data = individual_results[param_name]
            base_value = self.config.parameter_ranges[param_name].base_value

            # Find best performing parameter value
            results = param_data['results']
            best_result = max(results, key=lambda x: x.get('sharpe_ratio', -999))
            best_value = best_result['parameter_value']

            recommendation = {
                'parameter': param_name,
                'current_value': base_value,
                'recommended_value': best_value,
                'improvement': best_result.get('sharpe_ratio', 0) - self.base_performance.get('sharpe_ratio', 0),
                'robustness_score': scores['combined_score'],
                'priority': 'High' if scores['combined_score'] > 0.7 else 'Medium' if scores['combined_score'] > 0.5 else 'Low'
            }

            recommendations.append(recommendation)

        return recommendations

    def _generate_plots(self, results: Dict):
        """Generate sensitivity analysis plots"""
        plot_dir = Path(self.config.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Individual parameter sensitivity plots
        self._plot_individual_sensitivity(results['individual_sensitivity'], plot_dir)

        # Robustness heatmap
        self._plot_robustness_heatmap(results['robustness_scores'], plot_dir)

        # Parameter interaction heatmaps
        self._plot_interaction_heatmaps(results['parameter_interactions'], plot_dir)

        print(f"Plots saved to {plot_dir}")

    def _plot_individual_sensitivity(self, individual_results: Dict, plot_dir: Path):
        """Plot individual parameter sensitivity curves"""
        n_params = len(individual_results)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_params > 1 else [axes]

        for idx, (param_name, param_data) in enumerate(individual_results.items()):
            ax = axes[idx]

            results = param_data['results']
            values = [r['parameter_value'] for r in results]
            sharpe_ratios = [r.get('sharpe_ratio', np.nan) for r in results]

            # Remove NaN values
            valid_data = [(v, s) for v, s in zip(values, sharpe_ratios) if not np.isnan(s)]
            if not valid_data:
                continue

            values, sharpe_ratios = zip(*valid_data)

            ax.plot(values, sharpe_ratios, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=self.base_performance.get('sharpe_ratio', 0),
                      color='red', linestyle='--', label='Base Performance')

            ax.set_xlabel(param_name)
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title(f'{param_name} Sensitivity')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(plot_dir / 'individual_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_robustness_heatmap(self, robustness_scores: Dict, plot_dir: Path):
        """Plot robustness scores heatmap"""
        # Prepare data for heatmap
        params = list(robustness_scores.keys())
        metrics = ['overall_robustness', 'stability_score', 'combined_score']

        data = []
        for param in params:
            row = [robustness_scores[param][metric] for metric in metrics]
            data.append(row)

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(data, xticklabels=metrics, yticklabels=params,
                   annot=True, fmt='.3f', cmap='RdYlGn', center=0.5)
        plt.title('Parameter Robustness Scores')
        plt.tight_layout()
        plt.savefig(plot_dir / 'robustness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_interaction_heatmaps(self, interaction_results: Dict, plot_dir: Path):
        """Plot parameter interaction heatmaps"""
        for interaction_key, interaction_data in interaction_results.items():
            df = pd.DataFrame(interaction_data['grid_results'])
            param1, param2 = interaction_data['param1'], interaction_data['param2']

            # Create pivot table for Sharpe ratio
            if 'sharpe_ratio' in df.columns:
                pivot = df.pivot_table(values='sharpe_ratio', index=param1, columns=param2, aggfunc='mean')

                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn')
                plt.title(f'Sharpe Ratio: {param1} vs {param2} Interaction')
                plt.tight_layout()
                plt.savefig(plot_dir / f'interaction_{interaction_key}.png', dpi=300, bbox_inches='tight')
                plt.close()

    def save_results(self, filepath: str):
        """Save analysis results to JSON"""
        if self.results:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [clean_dict(v) for v in d]
                else:
                    return convert_types(d)

            cleaned_results = clean_dict(self.results)

            with open(filepath, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)

# Example usage
def example_strategy(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.02) -> pd.Series:
    """Example strategy for testing sensitivity analysis"""
    # Simple momentum strategy
    returns = data['close'].pct_change()
    momentum = returns.rolling(lookback).mean()

    signals = np.where(momentum > threshold, 1, np.where(momentum < -threshold, -1, 0))
    strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns

    return strategy_returns.fillna(0)

if __name__ == "__main__":
    # Example usage
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
    }, index=dates)

    # Define parameter ranges
    config = SensitivityConfig(
        parameter_ranges={
            'lookback': ParameterRange(name='lookback', base_value=20, variation_pct=0.5, steps=7, discrete=True),
            'threshold': ParameterRange(name='threshold', base_value=0.02, variation_pct=0.5, steps=5)
        }
    )

    # Run sensitivity analysis
    analyzer = ParameterSensitivityAnalyzer(config)
    results = analyzer.analyze_strategy(example_strategy, sample_data, {'lookback': 20, 'threshold': 0.02})

    print("Sensitivity Analysis Complete!")
    print(f"Base Sharpe: {results['base_performance']['sharpe_ratio']:.3f}")
    print(f"Recommendations: {len(results['recommendations'])}")

    for rec in results['recommendations']:
        print(f"  {rec['parameter']}: {rec['current_value']} -> {rec['recommended_value']} "
              f"(improvement: {rec['improvement']:.3f}, priority: {rec['priority']})")