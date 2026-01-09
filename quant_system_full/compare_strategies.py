"""
Strategy Comparison Tool - Compare Original vs Improved Strategy Selections

Analyzes and compares stock selections from different strategies.
Provides detailed comparison reports on selection quality and characteristics.

Usage:
    python compare_strategies.py
    python compare_strategies.py --output comparison_report.json
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyComparer:
    """Compares stock selections from different strategies."""

    def __init__(self, status_file: str = None):
        """
        Initialize strategy comparer.

        Args:
            status_file: Path to status.json file
        """
        if status_file is None:
            status_file = os.path.join(
                os.path.dirname(__file__),
                'dashboard', 'state', 'status.json'
            )

        self.status_file = status_file
        logger.info(f"Initialized StrategyComparer with status file: {status_file}")

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

    def extract_selection_info(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract selection information from status.

        Args:
            status: Status dictionary

        Returns:
            Selection information dictionary
        """
        try:
            selection_results = status.get('selection_results', {})

            if not selection_results:
                logger.warning("No selection_results found in status")
                return {}

            top_picks = selection_results.get('top_picks', [])

            # Extract key information
            selection_info = {
                'timestamp': selection_results.get('timestamp'),
                'total_selections': selection_results.get('total_selections', 0),
                'top_picks': top_picks,
                'symbols': [pick['symbol'] for pick in top_picks],
                'avg_score': sum(pick['avg_score'] for pick in top_picks) / len(top_picks) if top_picks else 0,
                'score_range': {
                    'min': min(pick['avg_score'] for pick in top_picks) if top_picks else 0,
                    'max': max(pick['avg_score'] for pick in top_picks) if top_picks else 0
                },
                'strategy_distribution': self._analyze_strategy_distribution(top_picks),
                'action_distribution': self._analyze_action_distribution(top_picks)
            }

            return selection_info

        except Exception as e:
            logger.error(f"Error extracting selection info: {e}")
            return {}

    def _analyze_strategy_distribution(self, top_picks: List[Dict]) -> Dict[str, int]:
        """Analyze which strategies selected which stocks."""
        try:
            strategy_counts = {}

            for pick in top_picks:
                count = pick.get('strategy_count', 1)
                strategy_counts[pick['symbol']] = count

            # Count how many stocks from each strategy type
            single_strategy = sum(1 for c in strategy_counts.values() if c == 1)
            multi_strategy = sum(1 for c in strategy_counts.values() if c > 1)

            return {
                'single_strategy_picks': single_strategy,
                'multi_strategy_picks': multi_strategy,
                'total': len(top_picks)
            }

        except Exception as e:
            logger.error(f"Error analyzing strategy distribution: {e}")
            return {}

    def _analyze_action_distribution(self, top_picks: List[Dict]) -> Dict[str, int]:
        """Analyze action distribution."""
        try:
            actions = [pick.get('dominant_action', 'watch') for pick in top_picks]
            action_counts = Counter(actions)

            return dict(action_counts)

        except Exception as e:
            logger.error(f"Error analyzing action distribution: {e}")
            return {}

    def compare_selections(self, current_info: Dict[str, Any], previous_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare current selection with previous (if available).

        Args:
            current_info: Current selection info
            previous_info: Previous selection info (optional)

        Returns:
            Comparison dictionary
        """
        try:
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'current_selection': current_info
            }

            if previous_info:
                # Symbol overlap
                current_symbols = set(current_info.get('symbols', []))
                previous_symbols = set(previous_info.get('symbols', []))

                overlap = current_symbols & previous_symbols
                new_symbols = current_symbols - previous_symbols
                removed_symbols = previous_symbols - current_symbols

                comparison['comparison'] = {
                    'overlap_count': len(overlap),
                    'overlap_symbols': list(overlap),
                    'new_symbols': list(new_symbols),
                    'removed_symbols': list(removed_symbols),
                    'stability_rate': len(overlap) / len(previous_symbols) if previous_symbols else 0,
                    'avg_score_change': current_info['avg_score'] - previous_info['avg_score']
                }

                logger.info(f"Compared selections: {len(overlap)} overlapping, {len(new_symbols)} new, {len(removed_symbols)} removed")

            else:
                comparison['comparison'] = None
                logger.info("No previous selection for comparison")

            return comparison

        except Exception as e:
            logger.error(f"Error comparing selections: {e}")
            return {}

    def analyze_style_characteristics(self, top_picks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze style characteristics of selections.

        Args:
            top_picks: List of top picks

        Returns:
            Style analysis dictionary
        """
        try:
            # Extract style information from reasoning
            value_keywords = ['value', 'undervalued', 'P/E', 'P/B', 'dividend']
            momentum_keywords = ['momentum', 'breakout', 'trend', 'moving average']
            technical_keywords = ['technical', 'Bollinger', 'MA', 'volume']

            style_counts = {
                'value': 0,
                'momentum': 0,
                'technical': 0,
                'mixed': 0
            }

            for pick in top_picks:
                reasoning = pick.get('reasoning', '').lower()

                is_value = any(kw in reasoning for kw in value_keywords)
                is_momentum = any(kw in reasoning for kw in momentum_keywords)
                is_technical = any(kw in reasoning for kw in technical_keywords)

                style_count = sum([is_value, is_momentum, is_technical])

                if style_count > 1:
                    style_counts['mixed'] += 1
                elif is_value:
                    style_counts['value'] += 1
                elif is_momentum:
                    style_counts['momentum'] += 1
                elif is_technical:
                    style_counts['technical'] += 1

            # Calculate percentages
            total = len(top_picks)
            style_percentages = {
                style: (count / total * 100) if total > 0 else 0
                for style, count in style_counts.items()
            }

            return {
                'style_counts': style_counts,
                'style_percentages': style_percentages
            }

        except Exception as e:
            logger.error(f"Error analyzing style characteristics: {e}")
            return {}

    def generate_report(self, comparison: Dict[str, Any], output_file: str = None):
        """
        Generate comparison report.

        Args:
            comparison: Comparison dictionary
            output_file: Output file path (optional)
        """
        try:
            # Add style analysis
            current_info = comparison.get('current_selection', {})
            top_picks = current_info.get('top_picks', [])

            if top_picks:
                comparison['style_analysis'] = self.analyze_style_characteristics(top_picks)

            # Save to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(comparison, f, indent=2, ensure_ascii=False)
                logger.info(f"Report saved to {output_file}")

            # Print console summary
            self._print_console_summary(comparison)

        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def _print_console_summary(self, comparison: Dict[str, Any]):
        """Print comparison summary to console."""
        try:
            print("\n" + "=" * 80)
            print("STRATEGY SELECTION COMPARISON REPORT")
            print("=" * 80)

            current_info = comparison.get('current_selection', {})

            print(f"\nTimestamp: {current_info.get('timestamp', 'N/A')}")
            print(f"Total Selections: {current_info.get('total_selections', 0)}")
            print(f"Average Score: {current_info.get('avg_score', 0):.2f}")
            print(f"Score Range: {current_info.get('score_range', {}).get('min', 0):.2f} - {current_info.get('score_range', {}).get('max', 0):.2f}")

            # Strategy distribution
            print("\n--- Strategy Distribution ---")
            strat_dist = current_info.get('strategy_distribution', {})
            print(f"  Single Strategy: {strat_dist.get('single_strategy_picks', 0)}")
            print(f"  Multi Strategy: {strat_dist.get('multi_strategy_picks', 0)}")

            # Action distribution
            print("\n--- Action Distribution ---")
            action_dist = current_info.get('action_distribution', {})
            for action, count in action_dist.items():
                print(f"  {action.upper()}: {count}")

            # Style analysis
            style_analysis = comparison.get('style_analysis', {})
            if style_analysis:
                print("\n--- Style Analysis ---")
                style_pcts = style_analysis.get('style_percentages', {})
                for style, pct in style_pcts.items():
                    print(f"  {style.capitalize()}: {pct:.1f}%")

            # Comparison with previous
            comp = comparison.get('comparison')
            if comp:
                print("\n--- Comparison with Previous ---")
                print(f"  Overlapping: {comp.get('overlap_count', 0)} stocks")
                print(f"  New: {len(comp.get('new_symbols', []))} stocks")
                print(f"  Removed: {len(comp.get('removed_symbols', []))} stocks")
                print(f"  Stability Rate: {comp.get('stability_rate', 0):.2%}")
                print(f"  Avg Score Change: {comp.get('avg_score_change', 0):+.2f}")

            # Top picks
            print("\n--- Top 10 Selections ---")
            top_picks = current_info.get('top_picks', [])[:10]
            for i, pick in enumerate(top_picks, 1):
                print(f"  #{i:2d} {pick['symbol']:6s} - Score: {pick['avg_score']:5.1f} - Action: {pick['dominant_action']:12s}")

            print("\n" + "=" * 80)

        except Exception as e:
            logger.error(f"Error printing console summary: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Strategy Selection Comparison Tool')
    parser.add_argument('--status', type=str, help='Path to status.json file')
    parser.add_argument('--output', type=str, default='comparison_report.json', help='Output file')
    parser.add_argument('--previous', type=str, help='Path to previous comparison report for diff')

    args = parser.parse_args()

    print("\nStrategy Selection Comparison Tool")
    print(f"Output: {args.output}\n")

    # Initialize comparer
    comparer = StrategyComparer(status_file=args.status)

    # Load current status
    status = comparer.load_status()

    if not status:
        print("ERROR: Could not load status file")
        return

    # Extract current selection info
    current_info = comparer.extract_selection_info(status)

    if not current_info:
        print("ERROR: Could not extract selection information")
        return

    # Load previous info if specified
    previous_info = None
    if args.previous and os.path.exists(args.previous):
        try:
            with open(args.previous, 'r', encoding='utf-8') as f:
                previous_report = json.load(f)
                previous_info = previous_report.get('current_selection')
            logger.info(f"Loaded previous report: {args.previous}")
        except Exception as e:
            logger.warning(f"Could not load previous report: {e}")

    # Compare
    comparison = comparer.compare_selections(current_info, previous_info)

    # Generate report
    comparer.generate_report(comparison, args.output)

    print(f"\nComparison report saved to: {args.output}")


if __name__ == '__main__':
    main()
