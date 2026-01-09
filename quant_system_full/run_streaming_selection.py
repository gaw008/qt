#!/usr/bin/env python3
"""
Streaming Stock Selection Entry Point

This script serves as the main entry point for streaming stock selection,
designed to be called by the worker runner with command line arguments.

Usage:
    python run_streaming_selection.py --batch-size 100 --top-n 20 --universe-file all_stock_symbols.csv
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add bot path for imports
BOT_PATH = Path(__file__).parent / 'bot'
sys.path.insert(0, str(BOT_PATH))

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    from config import SETTINGS
    from selection_strategies.streaming_strategy import StreamingSelectionStrategy
    from selection_strategies.streaming_value_momentum import StreamingValueMomentumStrategy
    from data import fetch_batch_history, get_data_cache_stats
    print("[STREAMING] Successfully imported bot modules")
except ImportError as e:
    print(f"[STREAMING] Failed to import bot modules: {e}")
    sys.exit(1)

def load_stock_universe(universe_file: str) -> List[str]:
    """Load stock symbols from universe file."""
    try:
        if not os.path.exists(universe_file):
            raise FileNotFoundError(f"Universe file not found: {universe_file}")

        with open(universe_file, 'r') as f:
            # Read CSV file, assuming first column contains symbols
            import csv
            reader = csv.reader(f)
            symbols = []
            for row in reader:
                if row and row[0].strip():  # Skip empty rows
                    symbols.append(row[0].strip().upper())

        print(f"[STREAMING] Loaded {len(symbols)} symbols from {universe_file}")
        return symbols

    except Exception as e:
        print(f"[STREAMING] Error loading universe file: {e}")
        # Fallback to predefined universe
        fallback_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'MU', 'AMAT', 'LRCX'
        ]
        print(f"[STREAMING] Using fallback universe: {len(fallback_symbols)} symbols")
        return fallback_symbols

def main():
    """Main streaming selection execution."""
    parser = argparse.ArgumentParser(description='Streaming Stock Selection')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top stocks to select')
    parser.add_argument('--universe-file', type=str, default='all_stock_symbols.csv', help='Stock universe CSV file')
    parser.add_argument('--strategy', type=str, default='value_momentum', choices=['value_momentum', 'streaming'], help='Selection strategy')
    parser.add_argument('--output-dir', type=str, default='dashboard/state', help='Output directory for results')

    args = parser.parse_args()

    print(f"[STREAMING] Starting streaming selection with:")
    print(f"[STREAMING]   - Batch size: {args.batch_size}")
    print(f"[STREAMING]   - Top N: {args.top_n}")
    print(f"[STREAMING]   - Universe file: {args.universe_file}")
    print(f"[STREAMING]   - Strategy: {args.strategy}")
    print(f"[STREAMING]   - Output dir: {args.output_dir}")

    start_time = time.time()

    try:
        # Load stock universe
        universe = load_stock_universe(args.universe_file)

        if not universe:
            print("[STREAMING] ERROR: No symbols loaded from universe file")
            sys.exit(1)

        # Initialize streaming strategy
        if args.strategy == 'value_momentum':
            strategy = StreamingValueMomentumStrategy(
                candidate_pool_size=args.top_n * 5,  # Keep more candidates than needed
                min_score_threshold=30.0
            )
        else:
            strategy = StreamingSelectionStrategy()

        print(f"[STREAMING] Initialized strategy: {strategy.__class__.__name__}")

        # Configure strategy parameters
        from selection_strategies.base_strategy import SelectionCriteria
        criteria = SelectionCriteria(
            max_stocks=args.top_n,
            min_market_cap=SETTINGS.min_market_cap,
            max_market_cap=SETTINGS.max_market_cap,
            min_volume=SETTINGS.min_daily_volume,
            min_price=SETTINGS.min_stock_price,
            max_price=SETTINGS.max_stock_price
        )

        # Execute streaming selection
        print(f"[STREAMING] Starting analysis of {len(universe)} stocks...")

        results = strategy.select_stocks(
            universe=universe,
            criteria=criteria
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final results
        final_results_file = output_dir / 'streaming_selection_final.json'
        final_results = {
            "timestamp": results.timestamp.isoformat(),
            "strategy_name": results.strategy_name,
            "execution_time": results.execution_time,
            "total_processed": results.total_candidates,
            "selected_stocks": [stock.to_dict() for stock in results.selected_stocks],
            "metadata": results.metadata,
            "errors": results.errors
        }
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"[STREAMING] Results saved to: {final_results_file}")

        # Save selection results for dashboard
        selection_results_file = output_dir / 'selection_results.json'
        dashboard_results = {
            "timestamp": datetime.now().isoformat(),
            "strategy": args.strategy,
            "total_analyzed": len(universe),
            "selected_count": len(results.selected_stocks),
            "execution_time_seconds": time.time() - start_time,
            "stocks": [
                {
                    "symbol": stock.symbol,
                    "score": stock.score,
                    "action": stock.action.value,
                    "reasoning": stock.reasoning
                }
                for stock in results.selected_stocks
            ]
        }

        with open(selection_results_file, 'w') as f:
            json.dump(dashboard_results, f, indent=2)

        print(f"[STREAMING] Dashboard results saved to: {selection_results_file}")

        # Print summary
        execution_time = time.time() - start_time
        print(f"\n[STREAMING] === SELECTION COMPLETE ===")
        print(f"[STREAMING] Total stocks analyzed: {len(universe)}")
        print(f"[STREAMING] Stocks selected: {len(results.selected_stocks)}")
        print(f"[STREAMING] Execution time: {execution_time:.1f} seconds")
        print(f"[STREAMING] Average time per stock: {execution_time/len(universe):.3f} seconds")

        # Print top selections
        print(f"[STREAMING] Top selections:")
        for i, stock in enumerate(results.selected_stocks[:10], 1):
            print(f"[STREAMING]   {i:2d}. {stock.symbol:6s} - Score: {stock.score:6.1f} - {stock.action.value}")

        print(f"[STREAMING] SUCCESS: Streaming selection completed")

    except Exception as e:
        print(f"[STREAMING] ERROR: {e}")
        import traceback
        print(f"[STREAMING] Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()