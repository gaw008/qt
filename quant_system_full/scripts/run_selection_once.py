"""
Run stock selection once using CSV-only universe.

This script instantiates the MarketAwareScheduler and directly runs the
stock_selection_task(), which now uses the CSV-only universe builder.
It is useful for forcing a fresh selection after code changes without
starting a long-running scheduler loop.
"""

import os
import sys
from pathlib import Path


def main():
    # Ensure we can import the worker runner
    # Project root = quant_system_full
    root = Path(__file__).resolve().parent.parent
    worker_dir = root / 'dashboard' / 'worker'
    if str(worker_dir) not in sys.path:
        sys.path.append(str(worker_dir))

    # Enforce CSV-only minimum size
    os.environ.setdefault('CSV_MIN_UNIVERSE', '5000')
    # Optional: cap to 5000
    os.environ.setdefault('SELECTION_UNIVERSE_SIZE', '5000')
    # Ensure CSV path uses project default when env var is empty
    if not os.environ.get('STOCK_UNIVERSE_FILE'):
        os.environ['STOCK_UNIVERSE_FILE'] = 'all_stock_symbols.csv'

    from runner import MarketAwareScheduler

    scheduler = MarketAwareScheduler(market_type=os.getenv('PRIMARY_MARKET', 'US'))

    try:
        scheduler.stock_selection_task()
        print('Selection run completed.')
    except Exception as e:
        print(f'Selection run failed: {e}')


if __name__ == '__main__':
    main()
