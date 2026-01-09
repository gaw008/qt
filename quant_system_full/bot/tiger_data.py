"""
Tiger API data fetching module.
Provides Tiger Brokers API data access with fallback support.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

_quote_client_instance = None


def _get_quote_client():
    """Get or create Tiger quote client instance."""
    global _quote_client_instance

    if _quote_client_instance is not None:
        return _quote_client_instance

    try:
        from tigeropen.quote.quote_client import QuoteClient
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from pathlib import Path
        import os

        # Use props configuration file (same as runner.py)
        # Find props directory relative to bot module
        bot_dir = Path(__file__).parent
        props_dir = str((bot_dir.parent / "props").resolve())

        if not Path(props_dir).exists():
            logger.error(f"[tiger_data] Props directory not found: {props_dir}")
            return None

        props_file = Path(props_dir) / "tiger_openapi_config.properties"
        if not props_file.exists():
            logger.error(f"[tiger_data] Props file not found: {props_file}")
            return None

        # Initialize using props configuration
        config = TigerOpenClientConfig(props_path=props_dir)
        _quote_client_instance = QuoteClient(config)

        logger.info("[tiger_data] Quote client initialized successfully using props config")
        return _quote_client_instance

    except Exception as e:
        logger.error(f"[tiger_data] Failed to initialize quote client: {e}")
        return None


def fetch_tiger_ticker_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch ticker information from Tiger API.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with ticker info or None if unavailable
    """
    try:
        # Get quote client
        quote_client = _get_quote_client()
        if not quote_client:
            logger.warning("[tiger_data] Quote client not available")
            return None

        # Fetch stock brief from Tiger API
        briefs = quote_client.get_stock_briefs([symbol])

        if not briefs or len(briefs) == 0:
            logger.warning(f"[tiger_data] No brief data for {symbol}")
            return None

        brief = briefs[0]

        # Extract available fields from Tiger brief
        info = {
            'symbol': symbol,
            'shortName': getattr(brief, 'name', symbol),
            'longName': getattr(brief, 'name', symbol),
            'regularMarketPrice': float(getattr(brief, 'latest_price', 0) or 0),
            'regularMarketVolume': int(getattr(brief, 'volume', 0) or 0),
            'marketCap': float(getattr(brief, 'market_cap', 0) or 0),
            'fiftyTwoWeekHigh': float(getattr(brief, 'high_52', 0) or 0),
            'fiftyTwoWeekLow': float(getattr(brief, 'low_52', 0) or 0),
            'averageVolume': int(getattr(brief, 'avg_volume', 0) or 0),
            'previousClose': float(getattr(brief, 'pre_close', 0) or 0),
        }

        # Try to get additional financial data if available
        try:
            # Get financial data
            financials = quote_client.get_financial_report([symbol])
            if financials and len(financials) > 0:
                fin = financials[0]
                info['trailingPE'] = float(getattr(fin, 'pe_ttm', 0) or 0)
                info['priceToBook'] = float(getattr(fin, 'pb', 0) or 0)
                info['dividendYield'] = float(getattr(fin, 'dividend_yield', 0) or 0)
        except Exception as e:
            logger.debug(f"[tiger_data] Financial data not available for {symbol}: {e}")

        logger.info(f"[tiger_data] Successfully fetched info for {symbol} from Tiger API")
        return info

    except Exception as e:
        logger.error(f"[tiger_data] Failed to fetch ticker info for {symbol}: {e}")
        return None


def fetch_tiger_price_history(symbol: str, period: str = 'day', limit: int = 300):
    """
    Fetch price history from Tiger API.
    This is a placeholder - actual implementation is in data.py fetch_history.

    Args:
        symbol: Stock ticker symbol
        period: Time period ('day', 'min1', 'min5', etc.)
        limit: Number of data points

    Returns:
        DataFrame with price history or None
    """
    try:
        from bot.data import fetch_history

        quote_client = _get_quote_client()
        return fetch_history(quote_client, symbol, period, limit, dry_run=False)

    except Exception as e:
        logger.error(f"[tiger_data] Failed to fetch price history for {symbol}: {e}")
        return None