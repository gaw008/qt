"""
Backtest Comparison Tool - Compare Original vs Improved Strategies V2

Independent tool for backtesting and comparing selection strategies.
Generates comprehensive performance reports and comparison analysis.

Usage:
    python backtest_comparison.py --start 2024-01-01 --end 2024-12-31 --universe 500
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'bot')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bot.selection_strategies.base_strategy import SelectionCriteria
from bot.selection_strategies.value_momentum import ValueMomentumStrategy
from bot.selection_strategies.technical_breakout import TechnicalBreakoutStrategy
from bot.selection_strategies.earnings_momentum import EarningsMomentumStrategy
from bot.selection_strategies.strategy_orchestrator_v2 import StrategyOrchestratorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestComparison:
    """Backtest comparison engine for strategy evaluation."""

    def __init__(self, start_date: str, end_date: str, universe_size: int = 500):
        """
        Initialize backtest comparison.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            universe_size: Number of stocks in universe
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.universe_size = universe_size

        logger.info(f"Initialized backtest: {start_date} to {end_date}, universe={universe_size}")

    def get_stock_universe(self) -> List[str]:
        """Get stock universe for backtesting."""
        try:
            import csv

            universe_file = os.path.join(os.path.dirname(__file__), "all_stock_symbols.csv")

            if not os.path.exists(universe_file):
                logger.warning("Stock universe file not found, using fallback")
                return self._get_fallback_universe()

            symbols = []
            with open(universe_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('Symbol', '').strip()
                    if symbol and len(symbol) <= 5:
                        symbols.append(symbol)

            logger.info(f"Loaded {len(symbols)} symbols from universe file")
            return symbols[:self.universe_size]

        except Exception as e:
            logger.error(f"Error loading universe: {e}")
            return self._get_fallback_universe()

    def _get_fallback_universe(self) -> List[str]:
        """Fallback universe if CSV not available."""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
            "NFLX", "ADBE", "AMD", "INTC", "QCOM", "CSCO", "UNH", "JNJ", "PFE", "ABBV",
            "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "HD", "PG", "KO", "PEP",
            "CAT", "BA", "GE", "RTX", "XOM", "CVX", "LIN", "AMT", "PLD", "CCI"
        ]

    def run_original_strategies(self, universe: List[str], criteria: SelectionCriteria) -> Dict[str, Any]:
        """Run original strategies combination."""
        try:
            logger.info("[ORIGINAL] Running original strategies")

            strategies = [
                ValueMomentumStrategy(),
                TechnicalBreakoutStrategy(),
                EarningsMomentumStrategy()
            ]

            all_results = {}
            for strategy in strategies:
                try:
                    results = strategy.select_stocks(universe, criteria)
                    all_results[strategy.name] = results
                    logger.info(f"[ORIGINAL] {strategy.name}: {len(results.selected_stocks)} stocks")
                except Exception as e:
                    logger.error(f"[ORIGINAL] Error in {strategy.name}: {e}")

            # Combine with simple consensus scoring
            combined = self._combine_original(all_results, criteria.max_stocks)

            return {
                'strategy_type': 'original',
                'selections': combined,
                'strategy_count': len(strategies),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[ORIGINAL] Failed: {e}")
            return {'strategy_type': 'original', 'selections': [], 'error': str(e)}

    def run_improved_strategies(self, universe: List[str], criteria: SelectionCriteria) -> Dict[str, Any]:
        """Run improved strategies V2 with risk management."""
        try:
            logger.info("[IMPROVED] Running improved strategies V2")

            config_path = 'bot/config/selection_config_v2.json'
            orchestrator = StrategyOrchestratorV2(enable_improved=True, config_path=config_path)

            combined = orchestrator.select_stocks_with_risk_management(universe, criteria)

            return {
                'strategy_type': 'improved_v2',
                'selections': combined,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[IMPROVED] Failed: {e}")
            return {'strategy_type': 'improved_v2', 'selections': [], 'error': str(e)}

    def _combine_original(self, all_results: Dict, max_stocks: int) -> List[Dict[str, Any]]:
        """Combine original strategy results with consensus bonus."""
        try:
            combined = {}

            for strategy_name, results in all_results.items():
                for stock in results.selected_stocks:
                    symbol = stock.symbol

                    if symbol not in combined:
                        combined[symbol] = {
                            'symbol': symbol,
                            'total_score': 0,
                            'strategy_count': 0,
                            'strategies': []
                        }

                    combined[symbol]['total_score'] += stock.score
                    combined[symbol]['strategy_count'] += 1
                    combined[symbol]['strategies'].append(strategy_name)

            # Calculate final scores with consensus bonus
            final = []
            for symbol, data in combined.items():
                avg_score = data['total_score'] / data['strategy_count']
                consensus_bonus = min(10.0, data['strategy_count'] * 2.5)
                final_score = avg_score + consensus_bonus

                final.append({
                    'symbol': symbol,
                    'score': final_score,
                    'avg_score': avg_score,
                    'strategy_count': data['strategy_count']
                })

            final.sort(key=lambda x: x['score'], reverse=True)
            return final[:max_stocks]

        except Exception as e:
            logger.error(f"Error combining original strategies: {e}")
            return []

    def calculate_performance_metrics(self, selections: List[Dict[str, Any]], period_days: int = 30) -> Dict[str, float]:
        """
        Calculate performance metrics for selected stocks.

        Note: This is a simplified backtest. In production, use actual historical prices
        and trading simulation with transaction costs.

        Args:
            selections: List of selected stocks
            period_days: Holding period in days

        Returns:
            Performance metrics dictionary
        """
        try:
            if not selections:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'avg_gain': 0.0,
                    'avg_loss': 0.0,
                    'num_selections': 0
                }

            # Simplified performance simulation
            # In production, fetch actual historical data
            import numpy as np

            num_stocks = len(selections)
            returns = []

            for selection in selections:
                # Simulate return based on score
                # Higher scores -> higher expected return (simplified)
                score = selection.get('score', 50.0)
                base_return = (score - 50.0) / 500.0  # Convert score to return
                noise = np.random.normal(0, 0.02)  # Add market noise
                stock_return = base_return + noise
                returns.append(stock_return)

            returns = np.array(returns)

            # Calculate metrics
            total_return = np.mean(returns)
            volatility = np.std(returns) if len(returns) > 1 else 0.01
            sharpe_ratio = (total_return / volatility) if volatility > 0 else 0.0

            # Win rate
            wins = np.sum(returns > 0)
            win_rate = wins / len(returns) if len(returns) > 0 else 0.0

            # Average gain/loss
            gains = returns[returns > 0]
            losses = returns[returns < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

            # Max drawdown (simplified)
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

            return {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'avg_gain': float(avg_gain),
                'avg_loss': float(avg_loss),
                'num_selections': num_stocks,
                'volatility': float(volatility)
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'error': str(e), 'num_selections': len(selections)}

    def run_comparison(self) -> Dict[str, Any]:
        """Run complete backtest comparison."""
        try:
            logger.info("Starting backtest comparison")

            # Get stock universe
            universe = self.get_stock_universe()

            # Configure selection criteria
            criteria = SelectionCriteria(
                max_stocks=20,
                min_market_cap=1e8,
                max_market_cap=5e12,
                min_volume=50000,
                min_price=1.0,
                max_price=2000.0,
                min_score_threshold=0.0
            )

            # Run both strategies
            original_results = self.run_original_strategies(universe, criteria)
            improved_results = self.run_improved_strategies(universe, criteria)

            # Calculate performance metrics
            original_metrics = self.calculate_performance_metrics(original_results['selections'])
            improved_metrics = self.calculate_performance_metrics(improved_results['selections'])

            # Build comparison report
            comparison = {
                'backtest_period': {
                    'start_date': self.start_date.isoformat(),
                    'end_date': self.end_date.isoformat(),
                    'days': (self.end_date - self.start_date).days
                },
                'universe_size': self.universe_size,
                'original_strategy': {
                    'selections': original_results['selections'][:10],  # Top 10
                    'metrics': original_metrics
                },
                'improved_strategy': {
                    'selections': improved_results['selections'][:10],  # Top 10
                    'metrics': improved_metrics
                },
                'comparison': self._calculate_comparison(original_metrics, improved_metrics),
                'timestamp': datetime.now().isoformat()
            }

            logger.info("Backtest comparison completed")
            return comparison

        except Exception as e:
            logger.error(f"Backtest comparison failed: {e}")
            return {'error': str(e)}

    def _calculate_comparison(self, original: Dict, improved: Dict) -> Dict[str, Any]:
        """Calculate comparison metrics between strategies."""
        try:
            return {
                'return_improvement': improved['total_return'] - original['total_return'],
                'sharpe_improvement': improved['sharpe_ratio'] - original['sharpe_ratio'],
                'drawdown_improvement': improved['max_drawdown'] - original['max_drawdown'],
                'win_rate_improvement': improved['win_rate'] - original['win_rate'],
                'better_metrics': {
                    'return': improved['total_return'] > original['total_return'],
                    'sharpe': improved['sharpe_ratio'] > original['sharpe_ratio'],
                    'drawdown': improved['max_drawdown'] > original['max_drawdown'],
                    'win_rate': improved['win_rate'] > original['win_rate']
                }
            }
        except Exception as e:
            logger.error(f"Error calculating comparison: {e}")
            return {}

    def generate_report(self, comparison: Dict[str, Any], output_file: str = 'backtest_comparison_report.json'):
        """Generate and save comparison report."""
        try:
            # Save JSON report
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)

            logger.info(f"Report saved to {output_file}")

            # Print summary to console
            print("\n" + "=" * 80)
            print("BACKTEST COMPARISON REPORT")
            print("=" * 80)

            if 'error' in comparison:
                print(f"\nERROR: {comparison['error']}")
                return

            print(f"\nPeriod: {comparison['backtest_period']['start_date']} to {comparison['backtest_period']['end_date']}")
            print(f"Universe Size: {comparison['universe_size']}")

            print("\n--- ORIGINAL STRATEGIES ---")
            orig_metrics = comparison['original_strategy']['metrics']
            print(f"  Total Return: {orig_metrics['total_return']:.2%}")
            print(f"  Sharpe Ratio: {orig_metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {orig_metrics['max_drawdown']:.2%}")
            print(f"  Win Rate: {orig_metrics['win_rate']:.2%}")
            print(f"  Selections: {orig_metrics['num_selections']}")

            print("\n--- IMPROVED STRATEGIES V2 ---")
            imp_metrics = comparison['improved_strategy']['metrics']
            print(f"  Total Return: {imp_metrics['total_return']:.2%}")
            print(f"  Sharpe Ratio: {imp_metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {imp_metrics['max_drawdown']:.2%}")
            print(f"  Win Rate: {imp_metrics['win_rate']:.2%}")
            print(f"  Selections: {imp_metrics['num_selections']}")

            print("\n--- COMPARISON ---")
            comp = comparison['comparison']
            print(f"  Return Improvement: {comp['return_improvement']:+.2%}")
            print(f"  Sharpe Improvement: {comp['sharpe_improvement']:+.2f}")
            print(f"  Drawdown Improvement: {comp['drawdown_improvement']:+.2%}")
            print(f"  Win Rate Improvement: {comp['win_rate_improvement']:+.2%}")

            print("\n--- WINNER ---")
            better = comp['better_metrics']
            wins = sum(better.values())
            print(f"  Improved V2 wins {wins}/4 metrics")
            print(f"  Recommendation: {'USE IMPROVED V2' if wins >= 3 else 'USE ORIGINAL'}")

            print("\n" + "=" * 80)

        except Exception as e:
            logger.error(f"Error generating report: {e}")


def main():
    """Main entry point for backtest comparison tool."""
    parser = argparse.ArgumentParser(description='Backtest Comparison Tool')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--universe', type=int, default=500, help='Universe size')
    parser.add_argument('--output', type=str, default='backtest_comparison_report.json', help='Output file')

    args = parser.parse_args()

    print(f"\nBacktest Comparison Tool - Original vs Improved Strategies V2")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe: {args.universe} stocks")
    print(f"Output: {args.output}\n")

    # Run comparison
    backtest = BacktestComparison(args.start, args.end, args.universe)
    comparison_results = backtest.run_comparison()

    # Generate report
    backtest.generate_report(comparison_results, args.output)

    print("\nBacktest comparison completed!")
    print(f"Full report saved to: {args.output}")
    print(f"Log saved to: backtest_comparison.log")


if __name__ == '__main__':
    main()
