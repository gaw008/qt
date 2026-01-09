from __future__ import annotations

from typing import Iterable
import pandas as pd

def _safe_first(items: list, default=None):
    return items[0] if items else default

def fetch_fundamentals(symbols: Iterable[str]) -> pd.DataFrame:
    """Fetch basic fundamentals via yfinance for valuation factors.

    Returns columns: symbol, market_cap, revenue_ttm, ebitda_ttm, fcf_ttm, book_equity,
    cash_equiv, total_debt, industry
    """
    import yfinance as yf

    records = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            # Cash flow: freeCashflow may be trailing twelve months or recent; fallback to 0
            fcf = info.get('freeCashflow') or 0
            # Balance sheet approximations
            book_equity = info.get('bookValue')
            if book_equity is not None and isinstance(book_equity, (int, float)):
                # bookValue reported per share; multiply by shares if available
                shares = info.get('sharesOutstanding') or 0
                book_equity = book_equity * shares if shares else None
            market_cap = info.get('marketCap')
            revenue_ttm = info.get('totalRevenue')
            ebitda_ttm = info.get('ebitda')
            total_debt = info.get('totalDebt') or 0
            cash_equiv = info.get('totalCash') or 0
            industry = info.get('industry') or 'Unknown'
            records.append({
                'symbol': sym,
                'market_cap': market_cap or 0,
                'revenue_ttm': revenue_ttm or 0,
                'ebitda_ttm': ebitda_ttm or 0,
                'fcf_ttm': fcf or 0,
                'book_equity': book_equity or 0,
                'cash_equiv': cash_equiv or 0,
                'total_debt': total_debt or 0,
                'industry': industry,
            })
        except Exception:
            records.append({
                'symbol': sym,
                'market_cap': 0,
                'revenue_ttm': 0,
                'ebitda_ttm': 0,
                'fcf_ttm': 0,
                'book_equity': 0,
                'cash_equiv': 0,
                'total_debt': 0,
                'industry': 'Unknown',
            })
    return pd.DataFrame(records)


