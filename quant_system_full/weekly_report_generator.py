"""
Weekly Report Generator - Automated Weekly Performance Reports

Analyzes performance tracking data and generates comprehensive weekly reports.
Includes comparisons, trend analysis, and actionable recommendations.

Usage:
    python weekly_report_generator.py
    python weekly_report_generator.py --csv custom_tracking.csv --output reports/
"""

import os
import sys
import json
import csv
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import argparse
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeeklyReportGenerator:
    """Generates comprehensive weekly performance reports."""

    def __init__(self, csv_file: str = None, output_dir: str = None):
        """
        Initialize weekly report generator.

        Args:
            csv_file: Path to performance tracking CSV
            output_dir: Directory for output reports
        """
        if csv_file is None:
            csv_file = os.path.join(
                os.path.dirname(__file__),
                'performance_tracking.csv'
            )

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__),
                'reports'
            )

        self.csv_file = csv_file
        self.output_dir = output_dir

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initialized WeeklyReportGenerator")
        logger.info(f"  CSV file: {csv_file}")
        logger.info(f"  Output dir: {output_dir}")

    def load_data(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Load performance data from CSV.

        Args:
            days: Number of days to load (default 7 for weekly)

        Returns:
            List of daily performance records
        """
        try:
            if not os.path.exists(self.csv_file):
                logger.warning(f"CSV file not found: {self.csv_file}")
                return []

            data = []
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)

            # Get last N days
            recent_data = data[-days:] if len(data) >= days else data

            logger.info(f"Loaded {len(recent_data)} days of data")
            return recent_data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []

    def calculate_weekly_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate weekly performance metrics.

        Args:
            data: List of daily performance records

        Returns:
            Weekly metrics dictionary
        """
        try:
            if not data:
                return {}

            # Extract values
            portfolio_values = [float(d.get('portfolio_value', 0)) for d in data if d.get('portfolio_value')]
            daily_returns = [float(d.get('daily_return', 0)) for d in data if d.get('daily_return')]
            sharpe_ratios = [float(d.get('sharpe_ratio_ytd', 0)) for d in data if d.get('sharpe_ratio_ytd')]
            drawdowns = [float(d.get('current_drawdown', 0)) for d in data if d.get('current_drawdown')]

            # Weekly return
            if len(portfolio_values) >= 2:
                weekly_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            else:
                weekly_return = 0.0

            # Daily statistics
            avg_daily_return = statistics.mean(daily_returns) if daily_returns else 0.0
            daily_volatility = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0

            # Annualized volatility (assuming 252 trading days)
            annualized_volatility = daily_volatility * (252 ** 0.5) if daily_volatility > 0 else 0.0

            # Weekly Sharpe ratio
            if daily_volatility > 0:
                weekly_sharpe = avg_daily_return / daily_volatility * (5 ** 0.5)  # 5 trading days per week
            else:
                weekly_sharpe = 0.0

            # Win rate
            winning_days = sum(1 for r in daily_returns if r > 0)
            win_rate = winning_days / len(daily_returns) if daily_returns else 0.0

            # Max drawdown this week
            max_drawdown = min(drawdowns) if drawdowns else 0.0

            # Latest Sharpe
            latest_sharpe = sharpe_ratios[-1] if sharpe_ratios else 0.0

            # Portfolio growth
            if len(portfolio_values) >= 2:
                portfolio_growth = portfolio_values[-1] - portfolio_values[0]
            else:
                portfolio_growth = 0.0

            # Average metrics
            avg_sharpe = statistics.mean(sharpe_ratios) if sharpe_ratios else 0.0

            # Extract other metrics from latest day
            latest = data[-1] if data else {}

            metrics = {
                # Period info
                'start_date': data[0].get('date', '') if data else '',
                'end_date': data[-1].get('date', '') if data else '',
                'num_days': len(data),

                # Portfolio metrics
                'start_value': portfolio_values[0] if portfolio_values else 0.0,
                'end_value': portfolio_values[-1] if portfolio_values else 0.0,
                'portfolio_growth': portfolio_growth,
                'weekly_return': weekly_return,

                # Performance metrics
                'avg_daily_return': avg_daily_return,
                'daily_volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'weekly_sharpe': weekly_sharpe,
                'avg_sharpe_ytd': avg_sharpe,
                'latest_sharpe_ytd': latest_sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'winning_days': winning_days,
                'losing_days': len(daily_returns) - winning_days,

                # Latest status
                'num_selections': int(latest.get('num_selections', 0)),
                'active_positions': int(latest.get('active_positions', 0)),
                'avg_selection_score': float(latest.get('avg_selection_score', 0)),
                'compliance_status': latest.get('compliance_status', 'UNKNOWN'),
                'compliance_violations': int(latest.get('compliance_violations', 0)),
                'crowding_level': latest.get('crowding_level', 'UNKNOWN'),
                'market_regime': latest.get('market_regime', 'unknown'),

                # Risk metrics
                'portfolio_es_975': float(latest.get('portfolio_es_975', 0)),
                'risk_budget_utilization': float(latest.get('risk_budget_utilization', 0)),
                'factor_hhi': float(latest.get('factor_hhi', 0)),
                'max_correlation': float(latest.get('max_correlation', 0))
            }

            logger.info(f"Calculated weekly metrics for {metrics['num_days']} days")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on metrics.

        Args:
            metrics: Weekly metrics dictionary

        Returns:
            List of recommendation strings
        """
        try:
            recommendations = []

            # Performance recommendations
            weekly_return = metrics.get('weekly_return', 0)
            if weekly_return < -0.02:
                recommendations.append("ALERT: Weekly return below -2%. Review risk management and position sizing.")
            elif weekly_return < 0:
                recommendations.append("Weekly return negative. Consider reviewing selection strategy and market conditions.")
            elif weekly_return > 0.05:
                recommendations.append("Strong weekly performance (+5%+). Monitor for profit-taking opportunities.")

            # Sharpe ratio recommendations
            weekly_sharpe = metrics.get('weekly_sharpe', 0)
            if weekly_sharpe < 0:
                recommendations.append("Negative risk-adjusted return. Review strategy effectiveness and consider reducing exposure.")
            elif weekly_sharpe < 1.0:
                recommendations.append("Low Sharpe ratio (<1.0). Consider strategy optimization or risk reduction.")
            elif weekly_sharpe > 2.0:
                recommendations.append("Excellent risk-adjusted return (Sharpe >2.0). Current strategy performing well.")

            # Volatility recommendations
            daily_vol = metrics.get('daily_volatility', 0)
            if daily_vol > 0.03:
                recommendations.append("High daily volatility (>3%). Consider position size reduction and tighter stop-losses.")
            elif daily_vol > 0.02:
                recommendations.append("Elevated volatility. Monitor market regime and adjust risk parameters.")

            # Drawdown recommendations
            max_dd = metrics.get('max_drawdown', 0)
            if max_dd < -0.10:
                recommendations.append("CRITICAL: Drawdown exceeds -10%. Immediate risk review required.")
            elif max_dd < -0.05:
                recommendations.append("Significant drawdown (-5% to -10%). Review stop-loss levels and position sizing.")

            # Win rate recommendations
            win_rate = metrics.get('win_rate', 0)
            if win_rate < 0.40:
                recommendations.append("Low win rate (<40%). Review selection criteria and entry timing.")
            elif win_rate > 0.60:
                recommendations.append("Strong win rate (>60%). Current selection strategy effective.")

            # Compliance recommendations
            compliance_violations = metrics.get('compliance_violations', 0)
            if compliance_violations > 0:
                recommendations.append(f"WARNING: {compliance_violations} compliance violation(s). Review and address immediately.")

            # Crowding recommendations
            crowding_level = metrics.get('crowding_level', 'UNKNOWN')
            if crowding_level == 'HIGH':
                recommendations.append("High factor crowding detected. Consider diversifying across different factors.")
            elif crowding_level == 'MODERATE':
                recommendations.append("Moderate factor crowding. Monitor position concentration.")

            # Risk budget recommendations
            risk_budget_util = metrics.get('risk_budget_utilization', 0)
            if risk_budget_util > 0.90:
                recommendations.append("Risk budget utilization >90%. Consider reducing position sizes.")
            elif risk_budget_util < 0.50:
                recommendations.append("Low risk budget utilization (<50%). Opportunity to increase position sizes if market conditions favorable.")

            # General recommendations
            if not recommendations:
                recommendations.append("No immediate concerns. Continue monitoring performance and market conditions.")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

    def compare_with_baseline(self, metrics: Dict[str, Any], baseline: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare metrics with baseline (e.g., previous week, benchmark).

        Args:
            metrics: Current week metrics
            baseline: Baseline metrics for comparison

        Returns:
            Comparison dictionary
        """
        try:
            if not baseline:
                return {}

            comparison = {
                'return_change': metrics.get('weekly_return', 0) - baseline.get('weekly_return', 0),
                'sharpe_change': metrics.get('weekly_sharpe', 0) - baseline.get('weekly_sharpe', 0),
                'volatility_change': metrics.get('daily_volatility', 0) - baseline.get('daily_volatility', 0),
                'win_rate_change': metrics.get('win_rate', 0) - baseline.get('win_rate', 0),
                'better_metrics': {
                    'return': metrics.get('weekly_return', 0) > baseline.get('weekly_return', 0),
                    'sharpe': metrics.get('weekly_sharpe', 0) > baseline.get('weekly_sharpe', 0),
                    'volatility': metrics.get('daily_volatility', 0) < baseline.get('daily_volatility', 0),
                    'win_rate': metrics.get('win_rate', 0) > baseline.get('win_rate', 0)
                }
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return {}

    def generate_markdown_report(self, metrics: Dict[str, Any], recommendations: List[str],
                                  comparison: Dict[str, Any] = None) -> str:
        """
        Generate Markdown format report.

        Args:
            metrics: Weekly metrics
            recommendations: List of recommendations
            comparison: Optional comparison data

        Returns:
            Markdown formatted report string
        """
        try:
            report_lines = [
                "# Weekly Performance Report",
                "",
                f"**Period**: {metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}",
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "---",
                "",
                "## Portfolio Summary",
                "",
                f"- **Starting Value**: ${metrics.get('start_value', 0):,.2f}",
                f"- **Ending Value**: ${metrics.get('end_value', 0):,.2f}",
                f"- **Growth**: ${metrics.get('portfolio_growth', 0):,.2f} ({metrics.get('weekly_return', 0):.2%})",
                f"- **Active Positions**: {metrics.get('active_positions', 0)}",
                f"- **Total Selections**: {metrics.get('num_selections', 0)}",
                "",
                "## Performance Metrics",
                "",
                f"- **Weekly Return**: {metrics.get('weekly_return', 0):.2%}",
                f"- **Avg Daily Return**: {metrics.get('avg_daily_return', 0):.4%}",
                f"- **Daily Volatility**: {metrics.get('daily_volatility', 0):.4%}",
                f"- **Annualized Volatility**: {metrics.get('annualized_volatility', 0):.2%}",
                f"- **Weekly Sharpe Ratio**: {metrics.get('weekly_sharpe', 0):.2f}",
                f"- **Latest Sharpe YTD**: {metrics.get('latest_sharpe_ytd', 0):.2f}",
                f"- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2%}",
                "",
                "## Trading Statistics",
                "",
                f"- **Win Rate**: {metrics.get('win_rate', 0):.2%}",
                f"- **Winning Days**: {metrics.get('winning_days', 0)}",
                f"- **Losing Days**: {metrics.get('losing_days', 0)}",
                f"- **Average Selection Score**: {metrics.get('avg_selection_score', 0):.2f}",
                "",
                "## Risk Metrics",
                "",
                f"- **ES@97.5%**: {metrics.get('portfolio_es_975', 0):.4f}",
                f"- **Risk Budget Utilization**: {metrics.get('risk_budget_utilization', 0):.2%}",
                f"- **Factor HHI**: {metrics.get('factor_hhi', 0):.4f}",
                f"- **Max Correlation**: {metrics.get('max_correlation', 0):.4f}",
                f"- **Crowding Level**: {metrics.get('crowding_level', 'UNKNOWN')}",
                "",
                "## System Status",
                "",
                f"- **Market Regime**: {metrics.get('market_regime', 'unknown')}",
                f"- **Compliance Status**: {metrics.get('compliance_status', 'UNKNOWN')}",
                f"- **Compliance Violations**: {metrics.get('compliance_violations', 0)}",
                ""
            ]

            # Add comparison section if available
            if comparison:
                report_lines.extend([
                    "## Week-over-Week Comparison",
                    "",
                    f"- **Return Change**: {comparison.get('return_change', 0):+.2%}",
                    f"- **Sharpe Change**: {comparison.get('sharpe_change', 0):+.2f}",
                    f"- **Volatility Change**: {comparison.get('volatility_change', 0):+.4%}",
                    f"- **Win Rate Change**: {comparison.get('win_rate_change', 0):+.2%}",
                    ""
                ])

            # Add recommendations
            report_lines.extend([
                "## Recommendations",
                ""
            ])

            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")

            report_lines.extend([
                "",
                "---",
                "",
                "*Report generated automatically by Weekly Report Generator*"
            ])

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
            return ""

    def generate_json_report(self, metrics: Dict[str, Any], recommendations: List[str],
                            comparison: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate JSON format report.

        Args:
            metrics: Weekly metrics
            recommendations: List of recommendations
            comparison: Optional comparison data

        Returns:
            Report dictionary
        """
        try:
            report = {
                'report_type': 'weekly_performance',
                'generated_at': datetime.now().isoformat(),
                'period': {
                    'start_date': metrics.get('start_date', ''),
                    'end_date': metrics.get('end_date', ''),
                    'num_days': metrics.get('num_days', 0)
                },
                'metrics': metrics,
                'recommendations': recommendations
            }

            if comparison:
                report['comparison'] = comparison

            return report

        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return {}

    def save_reports(self, markdown_report: str, json_report: Dict[str, Any]):
        """
        Save reports to files.

        Args:
            markdown_report: Markdown formatted report
            json_report: JSON report dictionary
        """
        try:
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y-%m-%d')

            # Save Markdown report
            md_file = os.path.join(self.output_dir, f'weekly_report_{timestamp}.md')
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            logger.info(f"Markdown report saved: {md_file}")

            # Save JSON report
            json_file = os.path.join(self.output_dir, f'weekly_report_{timestamp}.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON report saved: {json_file}")

        except Exception as e:
            logger.error(f"Error saving reports: {e}")

    def run(self, days: int = 7):
        """
        Generate weekly report.

        Args:
            days: Number of days to analyze (default 7)
        """
        try:
            logger.info(f"Starting weekly report generation for last {days} days")

            # Load data
            data = self.load_data(days)

            if not data:
                logger.error("No data available for report")
                print("ERROR: No performance data available")
                print(f"Please ensure {self.csv_file} exists with data")
                return False

            # Calculate metrics
            metrics = self.calculate_weekly_metrics(data)

            if not metrics:
                logger.error("Failed to calculate metrics")
                return False

            # Generate recommendations
            recommendations = self.generate_recommendations(metrics)

            # Generate reports
            markdown_report = self.generate_markdown_report(metrics, recommendations)
            json_report = self.generate_json_report(metrics, recommendations)

            # Save reports
            self.save_reports(markdown_report, json_report)

            # Print summary to console
            print("\n" + "=" * 80)
            print("WEEKLY PERFORMANCE REPORT GENERATED")
            print("=" * 80)
            print(f"\nPeriod: {metrics.get('start_date')} to {metrics.get('end_date')}")
            print(f"Days: {metrics.get('num_days')}")
            print(f"\nWeekly Return: {metrics.get('weekly_return', 0):.2%}")
            print(f"Weekly Sharpe: {metrics.get('weekly_sharpe', 0):.2f}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
            print(f"\nReports saved to: {self.output_dir}")
            print("=" * 80 + "\n")

            logger.info("Weekly report generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error running report generation: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Weekly Performance Report Generator')
    parser.add_argument('--csv', type=str, help='Path to performance tracking CSV')
    parser.add_argument('--output', type=str, help='Output directory for reports')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze (default: 7)')

    args = parser.parse_args()

    print("\nWeekly Performance Report Generator")
    print(f"Analyzing last {args.days} days\n")

    # Initialize generator
    generator = WeeklyReportGenerator(csv_file=args.csv, output_dir=args.output)

    # Generate report
    success = generator.run(days=args.days)

    if success:
        print("Report generation completed successfully!")
    else:
        print("ERROR: Failed to generate report")
        sys.exit(1)


if __name__ == '__main__':
    main()
