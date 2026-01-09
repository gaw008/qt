import sys, os
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is importable when running from repo root
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from bot.tradeup_client import build_clients
from bot.data import fetch_history


def is_us_market_open_now(now: datetime | None = None) -> bool:
    # Simple heuristic: US equities Mon-Fri, 09:30-16:00 America/New_York
    # Avoid tz libs to keep deps minimal; assume local UTC offset roughly; this is a heuristic only.
    now = now or datetime.utcnow()
    # Market closed on weekends
    if now.weekday() >= 5:
        return False
    # Approximate NY time by shifting -4 hours (DST) as heuristic
    ny = now - timedelta(hours=4)
    minutes = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


def main():
    quote, _ = build_clients()
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    df_m = fetch_history(quote, symbol, period='1min', limit=50, dry_run=False)
    df_d = fetch_history(quote, symbol, period='day', limit=10, dry_run=False)
    print('[diag] 1min rows=', len(df_m), ' last=', df_m['time'].iloc[-1] if len(df_m) else None)
    print('[diag] day  rows=', len(df_d), ' last=', df_d['time'].iloc[-1] if len(df_d) else None)
    print('[diag] market_open_guess=', is_us_market_open_now())


if __name__ == '__main__':
    main()


