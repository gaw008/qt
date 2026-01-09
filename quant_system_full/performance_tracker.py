"""
Performance Tracker - Daily Performance Metrics Recording

Tracks key performance metrics daily and maintains a time series CSV file.
Designed to run automatically (e.g., daily at market close) or manually.

Usage:
    python performance_tracker.py
    python performance_tracker.py --output custom_tracking.csv
"""

import os
import sys
import json
import logging
import csv
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks and records daily performance metrics."""

    def __init__(self, status_file: str = None, output_csv: str = None):
        """
        Initialize performance tracker.

        Args:
            status_file: Path to status.json file
            output_csv: Path to output CSV file
        """
        if status_file is None:
            status_file = os.path.join(
                os.path.dirname(__file__),
                'dashboard', 'state', 'status.json'
            )

        if output_csv is None:
            output_csv = os.path.join(
                os.path.dirname(__file__),
                'performance_tracking.csv'
            )

        self.status_file = status_file
        self.output_csv = output_csv

        logger.info(f"Initialized PerformanceTracker")
        logger.info(f"  Status file: {status_file}")
        logger.info(f"  Output CSV: {output_csv}")

    def load_status(self) -> Dict[str, Any]:
        """Load current system status."""
        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            logger.info("Successfully loaded status.json")
            return status
        except Exception as e:
            logger.error(f"Error loading status file: {e}")
            return {}

    def extract_metrics(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from status.

        Args:
            status: Status dictionary

        Returns:
            Metrics dictionary
        """
        try:
            # Extract real-time metrics
            rt_metrics = status.get('real_time_metrics', {})

            # Extract portfolio values
            real_portfolio_value = status.get('real_portfolio_value', 0)
            recommended_portfolio_value = status.get('recommended_portfolio_value', 0)

            # Extract selection info
            selection_results = status.get('selection_results', {})
            top_picks = selection_results.get('top_picks', [])

            # Extract compliance status
            compliance_status = status.get('compliance_status', 'UNKNOWN')
            compliance_violations = len(status.get('compliance_violations', []))

            # Extract factor crowding
            crowding = status.get('factor_crowding_analysis', {})
            crowding_level = crowding.get('crowding_level', 'UNKNOWN')

            # Extract market regime
            market_regime = status.get('market_regime', {})
            regime = market_regime.get('regime', 'unknown')

            # Calculate daily return (if previous value available)
            # This would require reading previous day's data
            daily_return = 0.0  # Placeholder

            # Build metrics dictionary
            metrics = {
                'date': date.today().isoformat(),
                'timestamp': datetime.now().isoformat(),

                # Portfolio metrics
                'portfolio_value': real_portfolio_value,
                'recommended_value': recommended_portfolio_value,
                'daily_pnl': rt_metrics.get('daily_pnl', 0),
                'daily_return': daily_return,

                # Performance metrics
                'sharpe_ratio_ytd': rt_metrics.get('sharpe_ratio_ytd', 0),
                'max_drawdown_ytd': rt_metrics.get('max_drawdown_ytd', 0),
                'current_drawdown': rt_metrics.get('current_drawdown', 0),

                # Risk metrics
                'portfolio_es_975': rt_metrics.get('portfolio_es_975', 0),
                'risk_budget_utilization': rt_metrics.get('risk_budget_utilization', 0),
                'tail_dependence': rt_metrics.get('tail_dependence', 0),

                # Selection metrics
                'num_selections': selection_results.get('total_selections', 0),
                'avg_selection_score': sum(p['avg_score'] for p in top_picks) / len(top_picks) if top_picks else 0,
                'active_positions': rt_metrics.get('active_positions', 0),

                # Cost metrics
                'daily_transaction_costs': rt_metrics.get('daily_transaction_costs', 0),
                'implementation_shortfall': rt_metrics.get('implementation_shortfall', 0),
                'capacity_utilization': rt_metrics.get('capacity_utilization', 0),

                # Crowding metrics
                'factor_hhi': rt_metrics.get('factor_hhi', 0),
                'max_correlation': rt_metrics.get('max_correlation', 0),
                'crowding_risk_score': rt_metrics.get('crowding_risk_score', 0),
                'crowding_level': crowding_level,

                # Compliance
                'compliance_status': compliance_status,
                'compliance_violations': compliance_violations,

                # Market
                'market_regime': regime,
                'system_uptime': rt_metrics.get('system_uptime', 0),
                'data_freshness': rt_metrics.get('data_freshness', 0)
            }

            logger.info(f"Extracted metrics for {metrics['date']}")
            return metrics

        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return {}

    def calculate_daily_return(self, current_value: float) -> float:
        """
        Calculate daily return by comparing with previous day.

        Args:
            current_value: Current portfolio value

        Returns:
            Daily return as decimal (e.g., 0.02 for 2%)
        """
        try:
            if not os.path.exists(self.output_csv):
                return 0.0

            # Read last row from CSV
            with open(self.output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    return 0.0

                last_row = rows[-1]
                previous_value = float(last_row.get('portfolio_value', 0))

                if previous_value > 0:
                    daily_return = (current_value - previous_value) / previous_value
                    return daily_return

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating daily return: {e}")
            return 0.0

    def append_metrics(self, metrics: Dict[str, Any]):
        """
        Append metrics to CSV file.

        Args:
            metrics: Metrics dictionary
        """
        try:
            # Check if file exists
            file_exists = os.path.exists(self.output_csv)

            # Define CSV columns
            columns = [
                'date', 'timestamp',
                'portfolio_value', 'recommended_value', 'daily_pnl', 'daily_return',
                'sharpe_ratio_ytd', 'max_drawdown_ytd', 'current_drawdown',
                'portfolio_es_975', 'risk_budget_utilization', 'tail_dependence',
                'num_selections', 'avg_selection_score', 'active_positions',
                'daily_transaction_costs', 'implementation_shortfall', 'capacity_utilization',
                'factor_hhi', 'max_correlation', 'crowding_risk_score', 'crowding_level',
                'compliance_status', 'compliance_violations',
                'market_regime', 'system_uptime', 'data_freshness'
            ]

            # Calculate daily return if possible
            if 'portfolio_value' in metrics:
                metrics['daily_return'] = self.calculate_daily_return(metrics['portfolio_value'])

            # Check if entry for today already exists
            if file_exists:
                today = metrics['date']
                updated = False

                # Read existing data
                with open(self.output_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                # Check if today's entry exists
                for i, row in enumerate(rows):
                    if row['date'] == today:
                        # Update existing entry
                        rows[i] = metrics
                        updated = True
                        logger.info(f"Updated existing entry for {today}")
                        break

                if updated:
                    # Rewrite entire file
                    with open(self.output_csv, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    # Append new entry
                    with open(self.output_csv, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=columns)
                        writer.writerow(metrics)
                    logger.info(f"Appended new entry for {today}")

            else:
                # Create new file with header
                with open(self.output_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerow(metrics)
                logger.info(f"Created new CSV file: {self.output_csv}")

            logger.info(f"Successfully recorded metrics for {metrics['date']}")

        except Exception as e:
            logger.error(f"Error appending metrics to CSV: {e}")

    def print_summary(self, metrics: Dict[str, Any]):
        """Print metrics summary to console."""
        try:
            print("\n" + "=" * 80)
            print("DAILY PERFORMANCE METRICS")
            print("=" * 80)

            print(f"\nDate: {metrics.get('date')}")
            print(f"Timestamp: {metrics.get('timestamp')}")

            print("\n--- Portfolio ---")
            print(f"  Portfolio Value: ${metrics.get('portfolio_value', 0):,.2f}")
            print(f"  Daily P&L: ${metrics.get('daily_pnl', 0):,.2f}")
            print(f"  Daily Return: {metrics.get('daily_return', 0):.2%}")

            print("\n--- Performance ---")
            print(f"  Sharpe Ratio YTD: {metrics.get('sharpe_ratio_ytd', 0):.2f}")
            print(f"  Max Drawdown YTD: {metrics.get('max_drawdown_ytd', 0):.2%}")
            print(f"  Current Drawdown: {metrics.get('current_drawdown', 0):.2%}")

            print("\n--- Risk ---")
            print(f"  ES@97.5%: {metrics.get('portfolio_es_975', 0):.4f}")
            print(f"  Risk Budget Utilization: {metrics.get('risk_budget_utilization', 0):.2%}")
            print(f"  Tail Dependence: {metrics.get('tail_dependence', 0):.4f}")

            print("\n--- Selection ---")
            print(f"  Num Selections: {metrics.get('num_selections', 0)}")
            print(f"  Avg Score: {metrics.get('avg_selection_score', 0):.2f}")
            print(f"  Active Positions: {metrics.get('active_positions', 0)}")

            print("\n--- Crowding ---")
            print(f"  Factor HHI: {metrics.get('factor_hhi', 0):.4f}")
            print(f"  Max Correlation: {metrics.get('max_correlation', 0):.4f}")
            print(f"  Crowding Level: {metrics.get('crowding_level', 'UNKNOWN')}")

            print("\n--- Compliance ---")
            print(f"  Status: {metrics.get('compliance_status', 'UNKNOWN')}")
            print(f"  Violations: {metrics.get('compliance_violations', 0)}")

            print("\n--- System ---")
            print(f"  Market Regime: {metrics.get('market_regime', 'unknown')}")
            print(f"  System Uptime: {metrics.get('system_uptime', 0):.1f}h")

            print("\n" + "=" * 80)

        except Exception as e:
            logger.error(f"Error printing summary: {e}")

    def run(self):
        """Run performance tracking."""
        try:
            logger.info("Starting performance tracking run")

            # Load status
            status = self.load_status()

            if not status:
                logger.error("Could not load status")
                return False

            # Extract metrics
            metrics = self.extract_metrics(status)

            if not metrics:
                logger.error("Could not extract metrics")
                return False

            # Append to CSV
            self.append_metrics(metrics)

            # Print summary
            self.print_summary(metrics)

            logger.info("Performance tracking completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error running performance tracker: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Daily Performance Metrics Tracker')
    parser.add_argument('--status', type=str, help='Path to status.json file')
    parser.add_argument('--output', type=str, default='performance_tracking.csv', help='Output CSV file')

    args = parser.parse_args()

    print("\nDaily Performance Metrics Tracker")
    print(f"Output: {args.output}\n")

    # Initialize tracker
    tracker = PerformanceTracker(status_file=args.status, output_csv=args.output)

    # Run tracking
    success = tracker.run()

    if success:
        print(f"\nMetrics successfully recorded to: {args.output}")
        print("Daily tracking completed!")
    else:
        print("\nERROR: Failed to record metrics")
        sys.exit(1)


if __name__ == '__main__':
    main()
