"""
Yahoo Finance API Direct Integration

This module provides direct integration with Yahoo Finance API using yfinance library.
It offers reliable market data access without external dependencies.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import time
import logging
from bot.config import SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_yahoo_price_history(
    symbol: str,
    period: str = 'day',
    limit: int = 300,
    max_retries: int = 3,
    retry_delay: float = 3.0
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data directly from Yahoo Finance.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        period: Time period ('day', '1min', '5min', '15min', '30min', '1h')
        limit: Number of data points to fetch
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        DataFrame with columns [time, open, high, low, close, volume] or None if failed
    """
    
    for attempt in range(max_retries):
        try:
            # Map period to Yahoo Finance parameters
            yahoo_period, yahoo_interval = _map_to_yahoo_params(period, limit)
            
            logger.info(f"[yahoo_data] Fetching {symbol}: period={yahoo_period}, interval={yahoo_interval}, attempt={attempt+1}")
            
            # Create ticker and fetch data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=yahoo_period, interval=yahoo_interval, auto_adjust=True, prepost=True)
            
            if hist.empty:
                logger.warning(f"[yahoo_data] No data returned for {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            
            # Convert to standard format
            df = pd.DataFrame({
                'time': hist.index,
                'open': hist['Open'].astype(float),
                'high': hist['High'].astype(float),
                'low': hist['Low'].astype(float),
                'close': hist['Close'].astype(float),
                'volume': hist['Volume'].astype(int)
            })
            
            # Sort and limit to requested number of rows
            df = df.sort_values('time').tail(limit).reset_index(drop=True)
            
            # Ensure timezone-naive datetime for consistency
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_convert(None)
            
            logger.info(f"[yahoo_data] Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"[yahoo_data] Attempt {attempt+1} failed for {symbol}: {type(e).__name__}: {e}")
            
            if attempt < max_retries - 1:
                # Exponential backoff with extended base delay (3-hour selection cycle allows longer waits)
                delay = retry_delay * (2 ** attempt)
                logger.info(f"[yahoo_data] Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"[yahoo_data] All {max_retries} attempts failed for {symbol}")
    
    return None


def fetch_yahoo_ticker_info(symbol: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive ticker information from Yahoo Finance.
    
    Returns company details, financial metrics, and market data.
    """
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                if attempt < max_retries - 1:
                    time.sleep(2.0)
                    continue
                return None
            
            # Extract key information
            ticker_data = {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'earnings_date': info.get('earningsDate', []),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"[yahoo_data] Successfully fetched info for {symbol}")
            return ticker_data
            
        except Exception as e:
            logger.error(f"[yahoo_data] Info fetch attempt {attempt+1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    
    return None


def fetch_yahoo_multiple_symbols(
    symbols: List[str], 
    period: str = 'day', 
    limit: int = 300
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch data for multiple symbols efficiently.
    
    Returns a dictionary mapping symbols to their data DataFrames.
    """
    results = {}
    
    # Use batch processing for efficiency
    yahoo_period, yahoo_interval = _map_to_yahoo_params(period, limit)
    
    for symbol in symbols:
        logger.info(f"[yahoo_data] Processing {symbol} in batch")
        results[symbol] = fetch_yahoo_price_history(
            symbol, period, limit, max_retries=2, retry_delay=2.0
        )

        # Extended delay to avoid rate limiting (3-hour selection cycle allows longer waits)
        time.sleep(0.5)
    
    return results


def get_market_status() -> Dict[str, Any]:
    """
    Get current market status and trading hours.
    """
    try:
        # Use SPY as a proxy for US market status
        spy = yf.Ticker("SPY")
        info = spy.info
        
        return {
            'market_state': info.get('marketState', 'UNKNOWN'),
            'regular_market_time': info.get('regularMarketTime', 0),
            'pre_market_time': info.get('preMarketTime', 0),
            'post_market_time': info.get('postMarketTime', 0),
            'timezone': info.get('timeZone', 'America/New_York'),
            'is_market_open': info.get('marketState') == 'REGULAR',
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[yahoo_data] Failed to get market status: {e}")
        return {
            'market_state': 'UNKNOWN',
            'is_market_open': False,
            'last_updated': datetime.now().isoformat()
        }


def _map_to_yahoo_params(period: str, limit: int) -> tuple[str, str]:
    """
    Map internal period format to Yahoo Finance API parameters.
    
    Returns: (yahoo_period, yahoo_interval)
    """
    p = period.lower()
    
    # Map intervals
    if p in ('1min', '1m', 'min1', 'minute1'):
        interval = '1m'
        # For 1-minute data, determine appropriate period
        if limit <= 60:
            yahoo_period = '1d'
        elif limit <= 420:  # 7 hours * 60 minutes
            yahoo_period = '2d'
        elif limit <= 1800:  # ~5 days of trading
            yahoo_period = '7d'
        else:
            yahoo_period = '1mo'
            
    elif p in ('2min', '2m', 'min2', 'minute2'):
        interval = '2m'
        yahoo_period = '1mo' if limit > 900 else '7d'
        
    elif p in ('5min', '5m', 'min5', 'minute5'):
        interval = '5m'
        yahoo_period = '1mo' if limit > 300 else '7d'
        
    elif p in ('15min', '15m', 'min15', 'minute15'):
        interval = '15m'
        yahoo_period = '2mo' if limit > 200 else '1mo'
        
    elif p in ('30min', '30m', 'min30', 'minute30'):
        interval = '30m'
        yahoo_period = '3mo' if limit > 100 else '2mo'
        
    elif p in ('60min', '60m', '1h', 'hour1', 'hourly'):
        interval = '1h'
        yahoo_period = '6mo' if limit > 200 else '3mo'
        
    elif p in ('1d', 'day', 'daily'):
        interval = '1d'
        if limit <= 30:
            yahoo_period = '1mo'
        elif limit <= 90:
            yahoo_period = '3mo'
        elif limit <= 180:
            yahoo_period = '6mo'
        elif limit <= 365:
            yahoo_period = '1y'
        elif limit <= 730:
            yahoo_period = '2y'
        else:
            yahoo_period = '5y'
            
    elif p in ('1wk', 'week', 'weekly'):
        interval = '1wk'
        yahoo_period = '2y' if limit > 100 else '1y'
        
    elif p in ('1mo', 'month', 'monthly'):
        interval = '1mo'
        yahoo_period = '10y' if limit > 60 else '5y'
        
    else:
        # Default to daily
        interval = '1d'
        yahoo_period = '1y'
    
    return yahoo_period, interval


def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol exists and has data available.
    
    Returns True if symbol is valid, False otherwise.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return bool(info and 'symbol' in info)
    except Exception:
        return False


def search_symbols(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search for symbols matching the query.
    
    Note: This is a basic implementation. For production use,
    consider using a dedicated symbol search API.
    """
    common_symbols = [
        # Tech stocks
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE',
        # Financial
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'V', 'MA',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'MDT', 'AMGN', 'GILD',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'GLD'
    ]
    
    query_upper = query.upper()
    matches = []
    
    for symbol in common_symbols:
        if query_upper in symbol:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                matches.append({
                    'symbol': symbol,
                    'name': info.get('longName', info.get('shortName', symbol)),
                    'type': 'stock'
                })
                if len(matches) >= max_results:
                    break
            except Exception:
                continue
    
    return matches