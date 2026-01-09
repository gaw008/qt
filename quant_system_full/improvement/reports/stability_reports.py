#!/usr/bin/env python3
"""
Stability Analysis Report Generator

Generates comprehensive reports for strategy stability and robustness analysis.
Creates both technical reports for analysts and executive summaries for decision makers.

Key Features:
- Multi-format reports (HTML, PDF, JSON)
- Interactive visualizations
- Executive dashboard
- Risk alerts and recommendations
- Automated report scheduling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
from dataclasses import dataclass
import base64
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    output_dir: str = "reports/stability"
    include_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    report_formats: List[str] = None
    executive_summary: bool = True
    technical_details: bool = True

    def __post_init__(self):
        if self.report_formats is None:
            self.report_formats = ["html", "json"]

class StabilityReportGenerator:
    """
    Generates comprehensive stability analysis reports

    Creates detailed reports combining results from:
    - Purged K-Fold cross-validation
    - Parameter sensitivity analysis
    - Rolling out-of-sample validation
    - Deflated Sharpe ratio analysis
    """

    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def generate_comprehensive_report(self, validation_results: Dict[str, Any],
                                    report_title: str = "Strategy Stability Analysis") -> Dict[str, str]:
        """
        Generate comprehensive stability analysis report

        Args:
            validation_results: Results from robustness validation
            report_title: Title for the report

        Returns:
            Dictionary with paths to generated report files
        """
        print("Generating Comprehensive Stability Report...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}

        # Generate plots
        plot_paths = self._generate_all_plots(validation_results, timestamp)

        # Generate HTML report
        if "html" in self.config.report_formats:
            html_path = self._generate_html_report(validation_results, plot_paths, report_title, timestamp)
            report_files["html"] = str(html_path)

        # Generate JSON report
        if "json" in self.config.report_formats:
            json_path = self._generate_json_report(validation_results, timestamp)
            report_files["json"] = str(json_path)

        # Generate executive summary
        if self.config.executive_summary:
            exec_path = self._generate_executive_summary(validation_results, timestamp)
            report_files["executive"] = str(exec_path)

        # Generate dashboard data
        dashboard_path = self._generate_dashboard_data(validation_results, timestamp)
        report_files["dashboard"] = str(dashboard_path)

        print(f"Reports generated in: {self.output_dir}")
        return report_files

    def _generate_all_plots(self, validation_results: Dict, timestamp: str) -> Dict[str, str]:
        """Generate all visualization plots"""
        if not self.config.include_plots:
            return {}

        plot_dir = self.output_dir / f"plots_{timestamp}"
        plot_dir.mkdir(exist_ok=True)

        plot_paths = {}

        try:
            # Strategy comparison plot
            plot_paths['strategy_comparison'] = self._plot_strategy_comparison(validation_results, plot_dir)

            # Robustness scores heatmap
            plot_paths['robustness_heatmap'] = self._plot_robustness_heatmap(validation_results, plot_dir)

            # Performance degradation analysis
            plot_paths['degradation_analysis'] = self._plot_degradation_analysis(validation_results, plot_dir)

            # Risk-return scatter
            plot_paths['risk_return'] = self._plot_risk_return_analysis(validation_results, plot_dir)

            # Time series performance
            plot_paths['time_series'] = self._plot_time_series_performance(validation_results, plot_dir)

            # Parameter sensitivity summary
            plot_paths['parameter_sensitivity'] = self._plot_parameter_sensitivity_summary(validation_results, plot_dir)

        except Exception as e:
            print(f"Warning: Some plots could not be generated: {e}")

        return plot_paths

    def _plot_strategy_comparison(self, validation_results: Dict, plot_dir: Path) -> str:
        """Plot strategy comparison chart"""
        consolidated = validation_results.get('consolidated_results', {})
        rankings = consolidated.get('strategy_rankings', [])

        if not rankings:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Robustness scores
        strategies = [r['strategy'] for r in rankings]
        scores = [r['robustness_score'] for r in rankings]

        bars1 = ax1.bar(strategies, scores, alpha=0.7, color='steelblue')
        ax1.set_ylabel('Robustness Score')
        ax1.set_title('Strategy Robustness Comparison')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom')

        # Key metrics comparison (if available)
        sharpe_ratios = []
        for ranking in rankings:
            metrics = ranking.get('key_metrics', {})
            sharpe = metrics.get('sharpe_ratio', np.nan)
            if not np.isnan(sharpe):
                sharpe_ratios.append(sharpe)
            else:
                sharpe_ratios.append(0)

        if any(s != 0 for s in sharpe_ratios):
            bars2 = ax2.bar(strategies, sharpe_ratios, alpha=0.7, color='orange')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Strategy Performance Comparison')
            ax2.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, sharpe in zip(bars2, sharpe_ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{sharpe:.3f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Sharpe Ratio Data Available',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Strategy Performance Comparison')

        plt.tight_layout()

        filename = plot_dir / f'strategy_comparison.{self.config.plot_format}'
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

        return str(filename)

    def _plot_robustness_heatmap(self, validation_results: Dict, plot_dir: Path) -> str:
        """Plot robustness metrics heatmap"""
        individual_results = validation_results.get('individual_strategies', {})

        if not individual_results:
            return ""

        # Collect robustness metrics
        strategies = []
        metrics_data = []

        for strategy_name, results in individual_results.items():
            if 'error' in results:
                continue

            strategies.append(strategy_name)

            # Collect different robustness indicators
            row_data = []

            # Cross-validation robustness
            cv_results = results.get('cross_validation', {})
            cv_robust = 1.0 if cv_results.get('summary', {}).get('is_robust', False) else 0.0
            row_data.append(cv_robust)

            # Parameter sensitivity robustness
            sensitivity_results = results.get('parameter_sensitivity', {})
            if 'robustness_scores' in sensitivity_results:
                robustness_scores = sensitivity_results['robustness_scores']
                avg_robustness = np.mean([scores['combined_score'] for scores in robustness_scores.values()])
                row_data.append(avg_robustness)
            else:
                row_data.append(0.0)

            # Rolling validation robustness
            rolling_results = results.get('rolling_validation', {})
            rolling_robust = 1.0 if rolling_results.get('validation_passed', False) else 0.0
            row_data.append(rolling_robust)

            # Deflated Sharpe significance
            dsr_results = results.get('deflated_sharpe', {})
            dsr_significant = 1.0 if dsr_results.get('is_significant', False) else 0.0
            row_data.append(dsr_significant)

            metrics_data.append(row_data)

        if not metrics_data:
            return ""

        # Create heatmap
        metrics_labels = ['Cross-Validation', 'Parameter Sensitivity', 'Rolling Validation', 'Deflated Sharpe']

        plt.figure(figsize=(10, max(6, len(strategies) * 0.8)))
        sns.heatmap(metrics_data,
                   xticklabels=metrics_labels,
                   yticklabels=strategies,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   center=0.5,
                   cbar_kws={'label': 'Robustness Score'})

        plt.title('Strategy Robustness Metrics Heatmap')
        plt.tight_layout()

        filename = plot_dir / f'robustness_heatmap.{self.config.plot_format}'
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

        return str(filename)

    def _plot_degradation_analysis(self, validation_results: Dict, plot_dir: Path) -> str:
        """Plot performance degradation analysis"""
        individual_results = validation_results.get('individual_strategies', {})

        degradation_data = []
        strategies = []

        for strategy_name, results in individual_results.items():
            if 'error' in results:
                continue

            rolling_results = results.get('rolling_validation', {})
            degradation_analysis = rolling_results.get('degradation_analysis', {})

            if degradation_analysis:
                strategies.append(strategy_name)
                avg_degradation = degradation_analysis.get('average_sharpe_degradation', 0)
                degradation_data.append(avg_degradation)

        if not degradation_data:
            return ""

        plt.figure(figsize=(10, 6))
        colors = ['red' if d < -0.2 else 'orange' if d < 0 else 'green' for d in degradation_data]

        bars = plt.bar(strategies, degradation_data, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5, label='Caution Threshold')
        plt.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')

        plt.ylabel('Average Sharpe Degradation')
        plt.title('In-Sample vs Out-of-Sample Performance Degradation')
        plt.xticks(rotation=45)
        plt.legend()

        # Add value labels
        for bar, value in zip(bars, degradation_data):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.,
                    height + (0.01 if height >= 0 else -0.02),
                    f'{value:.3f}', ha='center',
                    va='bottom' if height >= 0 else 'top')

        plt.tight_layout()

        filename = plot_dir / f'degradation_analysis.{self.config.plot_format}'
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

        return str(filename)

    def _plot_risk_return_analysis(self, validation_results: Dict, plot_dir: Path) -> str:
        """Plot risk-return analysis"""
        individual_results = validation_results.get('individual_strategies', {})

        returns_data = []
        volatility_data = []
        strategy_names = []

        for strategy_name, results in individual_results.items():
            if 'error' in results:
                continue

            dsr_results = results.get('deflated_sharpe', {})
            if 'annualized_return' in dsr_results and 'annualized_volatility' in dsr_results:
                strategy_names.append(strategy_name)
                returns_data.append(dsr_results['annualized_return'])
                volatility_data.append(dsr_results['annualized_volatility'])

        if not returns_data:
            return ""

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(volatility_data, returns_data, s=100, alpha=0.7, c=range(len(strategy_names)), cmap='viridis')

        # Add strategy labels
        for i, name in enumerate(strategy_names):
            plt.annotate(name, (volatility_data[i], returns_data[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Risk-Return Analysis')
        plt.grid(True, alpha=0.3)

        # Add quadrant lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=np.mean(volatility_data), color='gray', linestyle='--', alpha=0.5, label='Avg Volatility')

        plt.legend()
        plt.tight_layout()

        filename = plot_dir / f'risk_return.{self.config.plot_format}'
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

        return str(filename)

    def _plot_time_series_performance(self, validation_results: Dict, plot_dir: Path) -> str:
        """Plot time series performance from rolling validation"""
        individual_results = validation_results.get('individual_strategies', {})

        plt.figure(figsize=(12, 8))

        for strategy_name, results in individual_results.items():
            if 'error' in results:
                continue

            rolling_results = results.get('rolling_validation', {})
            window_results = rolling_results.get('window_results', [])

            if not window_results:
                continue

            # Extract time series data
            dates = []
            oos_performance = []

            for window_result in window_results:
                window_info = window_result.get('window_info', {})
                test_start = window_info.get('test_start')

                if test_start:
                    dates.append(pd.to_datetime(test_start))
                    oos_perf = window_result.get('out_of_sample_performance', {})
                    sharpe = oos_perf.get('sharpe_ratio', np.nan)
                    oos_performance.append(sharpe)

            if dates and oos_performance:
                plt.plot(dates, oos_performance, marker='o', label=strategy_name, alpha=0.7)

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Test Period Start')
        plt.ylabel('Out-of-Sample Sharpe Ratio')
        plt.title('Rolling Out-of-Sample Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = plot_dir / f'time_series_performance.{self.config.plot_format}'
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

        return str(filename)

    def _plot_parameter_sensitivity_summary(self, validation_results: Dict, plot_dir: Path) -> str:
        """Plot parameter sensitivity summary"""
        individual_results = validation_results.get('individual_strategies', {})

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        plot_idx = 0
        for strategy_name, results in individual_results.items():
            if 'error' in results or plot_idx >= 4:
                continue

            sensitivity_results = results.get('parameter_sensitivity', {})
            recommendations = sensitivity_results.get('recommendations', [])

            if not recommendations:
                continue

            ax = axes[plot_idx]

            # Plot parameter improvement potential
            params = [rec['parameter'] for rec in recommendations]
            improvements = [rec['improvement'] for rec in recommendations]
            priorities = [rec['priority'] for rec in recommendations]

            colors = ['red' if p == 'High' else 'orange' if p == 'Medium' else 'green' for p in priorities]

            bars = ax.bar(params, improvements, color=colors, alpha=0.7)
            ax.set_title(f'{strategy_name} Parameter Sensitivity')
            ax.set_ylabel('Sharpe Improvement')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.,
                       height + (0.001 if height >= 0 else -0.001),
                       f'{improvement:.3f}', ha='center',
                       va='bottom' if height >= 0 else 'top', fontsize=8)

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, 4):
            axes[idx].set_visible(False)

        plt.tight_layout()

        filename = plot_dir / f'parameter_sensitivity.{self.config.plot_format}'
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

        return str(filename)

    def _generate_html_report(self, validation_results: Dict, plot_paths: Dict,
                            report_title: str, timestamp: str) -> Path:
        """Generate HTML report"""
        html_content = self._create_html_template(validation_results, plot_paths, report_title)

        filename = self.output_dir / f"stability_report_{timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filename

    def _create_html_template(self, validation_results: Dict, plot_paths: Dict, report_title: str) -> str:
        """Create HTML report template"""
        executive_summary = validation_results.get('executive_summary', {})
        consolidated = validation_results.get('consolidated_results', {})

        # Convert plots to base64 for embedding
        embedded_plots = {}
        for plot_name, plot_path in plot_paths.items():
            if plot_path and Path(plot_path).exists():
                with open(plot_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()
                    embedded_plots[plot_name] = f"data:image/png;base64,{encoded}"

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }}
        .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .alert-success {{ background-color: #d4edda; border-color: #c3e6cb; color: #155724; }}
        .alert-warning {{ background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }}
        .alert-danger {{ background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .robustness-score {{ font-weight: bold; }}
        .score-high {{ color: #27ae60; }}
        .score-medium {{ color: #f39c12; }}
        .score-low {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="alert alert-{self._get_alert_class(executive_summary.get('overall_assessment', 'UNKNOWN'))}">
            <strong>Overall Assessment: {executive_summary.get('overall_assessment', 'Unknown')}</strong>
        </div>

        <div class="metric">
            <strong>Total Strategies Tested:</strong> {consolidated.get('total_strategies', 0)}
        </div>
        <div class="metric">
            <strong>Successful Validations:</strong> {consolidated.get('successful_validations', 0)}
        </div>
        <div class="metric">
            <strong>Overall Robustness Score:</strong>
            <span class="robustness-score {self._get_score_class(consolidated.get('overall_robustness_score', 0))}">
                {consolidated.get('overall_robustness_score', 0):.3f}
            </span>
        </div>

        <h3>Key Findings</h3>
        <ul>
        {self._format_list_items(executive_summary.get('key_findings', []))}
        </ul>

        <h3>Recommendations</h3>
        <ul>
        {self._format_list_items(executive_summary.get('recommendations', []))}
        </ul>

        {self._format_risk_alerts(executive_summary.get('risk_alerts', []))}
    </div>

    <div class="section">
        <h2>Strategy Rankings</h2>
        {self._create_strategy_rankings_table(consolidated.get('strategy_rankings', []))}
    </div>

    <div class="section">
        <h2>Visualizations</h2>
        {self._create_plots_section(embedded_plots)}
    </div>

    <div class="section">
        <h2>Technical Details</h2>
        <h3>Validation Methods</h3>
        <ul>
            <li><strong>Purged K-Fold Cross-Validation:</strong> Time-series aware cross-validation preventing data leakage</li>
            <li><strong>Parameter Sensitivity Analysis:</strong> Robustness testing across parameter ranges</li>
            <li><strong>Rolling Out-of-Sample Validation:</strong> Walk-forward analysis with true out-of-sample testing</li>
            <li><strong>Deflated Sharpe Ratio:</strong> Multiple testing adjustment for strategy significance</li>
        </ul>

        <h3>Robustness Criteria</h3>
        <ul>
            <li>Out-of-sample Sharpe ratio > 0.5</li>
            <li>Performance consistency > 60%</li>
            <li>Minimum 3 successful validation windows</li>
            <li>Deflated Sharpe ratio significance > 0.1</li>
        </ul>
    </div>

    <div class="section">
        <h2>Data Summary</h2>
        <p><strong>Validation Timestamp:</strong> {validation_results.get('validation_timestamp', 'Unknown')}</p>
        <p><strong>Report Generation:</strong> Automated stability analysis system</p>
    </div>
</body>
</html>
        """

        return html_template

    def _get_alert_class(self, assessment: str) -> str:
        """Get CSS alert class based on assessment"""
        if assessment == 'ROBUST':
            return 'success'
        elif assessment == 'MODERATE':
            return 'warning'
        else:
            return 'danger'

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on robustness score"""
        if score >= 0.7:
            return 'score-high'
        elif score >= 0.5:
            return 'score-medium'
        else:
            return 'score-low'

    def _format_list_items(self, items: List[str]) -> str:
        """Format list items for HTML"""
        return '\n'.join([f'<li>{item}</li>' for item in items])

    def _format_risk_alerts(self, alerts: List[str]) -> str:
        """Format risk alerts section"""
        if not alerts:
            return ""

        alert_html = '<h3>Risk Alerts</h3>'
        for alert in alerts:
            alert_html += f'<div class="alert alert-danger">[WARNING] {alert}</div>'

        return alert_html

    def _create_strategy_rankings_table(self, rankings: List[Dict]) -> str:
        """Create strategy rankings table"""
        if not rankings:
            return "<p>No strategy rankings available.</p>"

        table_html = """
        <table>
            <tr>
                <th>Rank</th>
                <th>Strategy</th>
                <th>Robustness Score</th>
                <th>Sharpe Ratio</th>
                <th>Status</th>
            </tr>
        """

        for i, ranking in enumerate(rankings, 1):
            strategy = ranking['strategy']
            score = ranking['robustness_score']
            metrics = ranking.get('key_metrics', {})
            sharpe = metrics.get('sharpe_ratio', 'N/A')

            if isinstance(sharpe, (int, float)) and not np.isnan(sharpe):
                sharpe_str = f"{sharpe:.3f}"
            else:
                sharpe_str = "N/A"

            status = "Strong" if score >= 0.7 else "Moderate" if score >= 0.5 else "Weak"
            status_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.5 else "score-low"

            table_html += f"""
            <tr>
                <td>{i}</td>
                <td>{strategy}</td>
                <td><span class="robustness-score {self._get_score_class(score)}">{score:.3f}</span></td>
                <td>{sharpe_str}</td>
                <td><span class="{status_class}">{status}</span></td>
            </tr>
            """

        table_html += "</table>"
        return table_html

    def _create_plots_section(self, embedded_plots: Dict) -> str:
        """Create plots section for HTML"""
        if not embedded_plots:
            return "<p>No visualizations available.</p>"

        plots_html = ""
        plot_titles = {
            'strategy_comparison': 'Strategy Comparison',
            'robustness_heatmap': 'Robustness Metrics Heatmap',
            'degradation_analysis': 'Performance Degradation Analysis',
            'risk_return': 'Risk-Return Analysis',
            'time_series': 'Time Series Performance',
            'parameter_sensitivity': 'Parameter Sensitivity Summary'
        }

        for plot_name, plot_data in embedded_plots.items():
            title = plot_titles.get(plot_name, plot_name.replace('_', ' ').title())
            plots_html += f"""
            <div class="plot">
                <h3>{title}</h3>
                <img src="{plot_data}" alt="{title}">
            </div>
            """

        return plots_html

    def _generate_json_report(self, validation_results: Dict, timestamp: str) -> Path:
        """Generate JSON report"""
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

        cleaned_results = clean_dict(validation_results)

        filename = self.output_dir / f"stability_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(cleaned_results, f, indent=2, default=str)

        return filename

    def _generate_executive_summary(self, validation_results: Dict, timestamp: str) -> Path:
        """Generate executive summary"""
        executive_summary = validation_results.get('executive_summary', {})
        consolidated = validation_results.get('consolidated_results', {})

        summary_text = []
        summary_text.append("STRATEGY STABILITY EXECUTIVE SUMMARY")
        summary_text.append("=" * 50)
        summary_text.append(f"Date: {executive_summary.get('validation_date', 'Unknown')}")
        summary_text.append(f"Overall Assessment: {executive_summary.get('overall_assessment', 'Unknown')}")
        summary_text.append("")

        # Key metrics
        summary_text.append("KEY METRICS:")
        summary_text.append(f"  ? Strategies Tested: {consolidated.get('total_strategies', 0)}")
        summary_text.append(f"  ? Successful Validations: {consolidated.get('successful_validations', 0)}")
        summary_text.append(f"  ? Overall Robustness Score: {consolidated.get('overall_robustness_score', 0):.3f}")
        summary_text.append("")

        # Top strategy
        best_strategy = consolidated.get('best_strategy')
        if best_strategy:
            summary_text.append(f"RECOMMENDED STRATEGY: {best_strategy}")
            summary_text.append("")

        # Key findings
        findings = executive_summary.get('key_findings', [])
        if findings:
            summary_text.append("KEY FINDINGS:")
            for finding in findings:
                summary_text.append(f"  ? {finding}")
            summary_text.append("")

        # Recommendations
        recommendations = executive_summary.get('recommendations', [])
        if recommendations:
            summary_text.append("RECOMMENDATIONS:")
            for rec in recommendations:
                summary_text.append(f"  ? {rec}")
            summary_text.append("")

        # Risk alerts
        alerts = executive_summary.get('risk_alerts', [])
        if alerts:
            summary_text.append("RISK ALERTS:")
            for alert in alerts:
                summary_text.append(f"  [WARNING]  {alert}")
            summary_text.append("")

        # Strategy rankings
        rankings = consolidated.get('strategy_rankings', [])
        if rankings:
            summary_text.append("STRATEGY RANKINGS:")
            for i, ranking in enumerate(rankings, 1):
                strategy = ranking['strategy']
                score = ranking['robustness_score']
                summary_text.append(f"  {i}. {strategy}: {score:.3f}")

        filename = self.output_dir / f"executive_summary_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write('\n'.join(summary_text))

        return filename

    def _generate_dashboard_data(self, validation_results: Dict, timestamp: str) -> Path:
        """Generate dashboard data for web interface"""
        dashboard_data = {
            'timestamp': timestamp,
            'last_updated': datetime.now().isoformat(),
            'executive_summary': validation_results.get('executive_summary', {}),
            'consolidated_results': validation_results.get('consolidated_results', {}),
            'alert_level': self._determine_alert_level(validation_results),
            'quick_stats': self._extract_quick_stats(validation_results)
        }

        filename = self.output_dir / f"dashboard_data_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        return filename

    def _determine_alert_level(self, validation_results: Dict) -> str:
        """Determine overall alert level"""
        executive_summary = validation_results.get('executive_summary', {})
        assessment = executive_summary.get('overall_assessment', 'UNKNOWN')

        if assessment == 'ROBUST':
            return 'success'
        elif assessment == 'MODERATE':
            return 'warning'
        else:
            return 'danger'

    def _extract_quick_stats(self, validation_results: Dict) -> Dict:
        """Extract quick statistics for dashboard"""
        consolidated = validation_results.get('consolidated_results', {})
        executive_summary = validation_results.get('executive_summary', {})

        return {
            'total_strategies': consolidated.get('total_strategies', 0),
            'successful_validations': consolidated.get('successful_validations', 0),
            'robustness_score': consolidated.get('overall_robustness_score', 0),
            'best_strategy': consolidated.get('best_strategy', 'None'),
            'risk_alerts_count': len(executive_summary.get('risk_alerts', [])),
            'recommendations_count': len(executive_summary.get('recommendations', []))
        }

# Example usage
if __name__ == "__main__":
    # Create sample validation results for testing
    sample_results = {
        'individual_strategies': {
            'ValueMomentum': {
                'cross_validation': {'summary': {'is_robust': True, 'dsr': 0.8}},
                'parameter_sensitivity': {'robustness_scores': {'param1': {'combined_score': 0.7}}},
                'rolling_validation': {'validation_passed': True},
                'deflated_sharpe': {'sharpe_ratio': 1.2, 'deflated_sharpe_ratio': 0.9, 'is_significant': True}
            }
        },
        'consolidated_results': {
            'total_strategies': 1,
            'successful_validations': 1,
            'overall_robustness_score': 0.8,
            'best_strategy': 'ValueMomentum',
            'strategy_rankings': [
                {'strategy': 'ValueMomentum', 'robustness_score': 0.8, 'key_metrics': {'sharpe_ratio': 1.2}}
            ]
        },
        'executive_summary': {
            'validation_date': '2024-01-01',
            'overall_assessment': 'ROBUST',
            'key_findings': ['Strong robustness across all tests'],
            'recommendations': ['Deploy ValueMomentum strategy'],
            'risk_alerts': []
        },
        'validation_timestamp': '2024-01-01T00:00:00'
    }

    # Generate report
    config = ReportConfig()
    generator = StabilityReportGenerator(config)

    report_files = generator.generate_comprehensive_report(
        sample_results,
        "Sample Strategy Stability Analysis"
    )

    print("Sample report generated:")
    for format_type, filepath in report_files.items():
        print(f"  {format_type}: {filepath}")