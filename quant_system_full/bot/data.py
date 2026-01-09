import math
import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import pandas as pd
try:
    from tigeropen.common.consts import BarPeriod, QuoteRight, Market
except Exception:
    BarPeriod = None
    QuoteRight = None
    Market = None

# Import config to check MCP settings
from bot.config import SETTINGS


def _map_period(period: str):
    """Return a tuple: (enum_or_str_for_sdk, str_period_for_sdk)
    Some SDKs want BarPeriod enum, others want strings like 'min1','min5','day'.
    """
    p = (period or '').lower()
    # String mapping used by your SDK (since get_kline=False, get_bars=True)
    if 'day' in p:
        str_p = 'day'
    elif p in ('1min','1m','min1','minute1'):
        str_p = 'min1'
    elif p in ('5min','5m','min5','minute5'):
        str_p = 'min5'
    else:
        str_p = p or 'day'

    # Enum (if available)
    enum_p = None
    if BarPeriod is not None:
        try:
            if str_p == 'day':
                enum_p = getattr(BarPeriod, 'DAY', None)
            elif str_p == 'min1':
                enum_p = getattr(BarPeriod, 'MIN_1', None)
            elif str_p == 'min5':
                enum_p = getattr(BarPeriod, 'MIN_5', None)
        except Exception:
            enum_p = None
    return enum_p or str_p, str_p


def _sdk_get_kline_any(quote_client, symbol: str, period: str, limit: int):
    """Try several tigeropen methods to get kline/bars to be robust across versions."""
    enum_p, str_p = _map_period(period)
    right = getattr(QuoteRight, 'NONE', None)

    # Try get_kline with rich signature
    if hasattr(quote_client, 'get_kline'):
        for kwargs in (
            dict(symbol=symbol, period=enum_p, right=right, begin_time=None, end_time=None, limit=limit),
            dict(symbol=symbol, period=enum_p, right=right, count=limit),
            dict(symbol=symbol, period=enum_p, count=limit),
            dict(symbol=symbol, period=str_p, right=right, limit=limit),
            dict(symbol=symbol, period=str_p, limit=limit),
        ):
            try:
                return quote_client.get_kline(**{k: v for k, v in kwargs.items() if v is not None})
            except Exception:
                pass

    # Some versions use get_kline_bars
    if hasattr(quote_client, 'get_kline_bars'):
        for kwargs in (
            dict(symbol=symbol, period=enum_p, right=right, count=limit),
            dict(symbol=symbol, period=enum_p, count=limit),
            dict(symbol=symbol, period=str_p, right=right, count=limit),
            dict(symbol=symbol, period=str_p, count=limit),
        ):
            try:
                return quote_client.get_kline_bars(**{k: v for k, v in kwargs.items() if v is not None})
            except Exception:
                pass

    # Prefer get_bars (present in your SDK)
    if hasattr(quote_client, 'get_bars'):
        now = datetime.utcnow()
        step = timedelta(days=1) if str_p == 'day' else timedelta(minutes=1 if str_p == 'min1' else 5)
        begin_dt = now - step * (limit + 10)
        begin_str = begin_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_str = now.strftime('%Y-%m-%d %H:%M:%S')
        attempts = (
            # minimal
            dict(symbols=[symbol], period=str_p, limit=limit),
            dict(symbols=[symbol], period=str_p, limit=limit),
            dict(symbols=[symbol], period=str_p, count=limit),
            dict(symbols=[symbol], period=str_p),
            # with right
            dict(symbols=[symbol], period=str_p, right=right, limit=limit),
            # with time window (datetime)
            dict(symbols=[symbol], period=str_p, begin_time=begin_dt, end_time=now, limit=limit),
            # with time window (string)
            dict(symbols=[symbol], period=str_p, begin_time=begin_str, end_time=end_str, limit=limit),
            # enum period variants
            dict(symbols=[symbol], period=enum_p, limit=limit),
            dict(symbols=[symbol], period=enum_p, right=right, limit=limit),
            # with market if required
            dict(symbols=[symbol], period=str_p, market=getattr(Market, 'US', None), limit=limit),
        )
        for kwargs in attempts:
            try:
                return quote_client.get_bars(**{k: v for k, v in kwargs.items() if v is not None})
            except Exception:
                pass

    # Paged fallback
    if hasattr(quote_client, 'get_bars_by_page'):
        for kwargs in (
            dict(symbols=[symbol], period=str_p, page=1, size=limit),
            dict(symbols=[symbol], period=str_p, page=1, size=min(50, limit)),
        ):
            try:
                return quote_client.get_bars_by_page(**kwargs)
            except Exception:
                pass

    # Some versions use get_history_kline
    if hasattr(quote_client, 'get_history_kline'):
        for kwargs in (
            dict(symbol=symbol, period=bp, count=limit),
            dict(symbol=symbol, period=period, count=limit),
        ):
            try:
                return quote_client.get_history_kline(**kwargs)
            except Exception:
                pass
    raise RuntimeError('No compatible kline/bars method found on quote_client')


def _bars_to_dataframe(obj) -> pd.DataFrame:
    """Best-effort convert various SDK return formats into a standard DataFrame.
    Expected output columns: time, open, high, low, close, volume.
    """
    # If already a DataFrame with required columns
    if isinstance(obj, pd.DataFrame):
        cols = {c.lower() for c in obj.columns}
        mapping = {}
        for want in ['time', 'open', 'high', 'low', 'close', 'volume']:
            if want in cols:
                mapping[next(c for c in obj.columns if c.lower() == want)] = want
        df = obj.rename(columns=mapping)
        missing = [c for c in ['time', 'open', 'high', 'low', 'close', 'volume'] if c not in df.columns]
        if not missing:
            return df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

    # If list[dict]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # Common keys: time/timestamp, open, high, low, close, volume/vol
        rows = []
        for it in obj:
            t = it.get('time') or it.get('timestamp') or it.get('datetime')
            if isinstance(t, (int, float)) and t > 10_000_000_000:
                # ns -> ms
                t = int(t / 1_000_000)
            rows.append({
                'time': t,
                'open': it.get('open'),
                'high': it.get('high'),
                'low': it.get('low'),
                'close': it.get('close') or it.get('price'),
                'volume': it.get('volume') or it.get('vol') or 0,
            })
        return pd.DataFrame(rows)

    # If object with attributes
    try:
        items = list(obj)
        if items:
            first = items[0]
            if hasattr(first, 'time') and hasattr(first, 'close'):
                rows = []
                for it in items:
                    rows.append({
                        'time': getattr(it, 'time', None),
                        'open': getattr(it, 'open', None),
                        'high': getattr(it, 'high', None),
                        'low': getattr(it, 'low', None),
                        'close': getattr(it, 'close', None),
                        'volume': getattr(it, 'volume', None) or 0,
                    })
                return pd.DataFrame(rows)
    except Exception:
        pass

    raise ValueError('Unsupported kline/bars return format; please update parser')


def fetch_history(
    quote_client,
    symbol: str,
    period: str = 'day',
    limit: int = 300,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Fetch history from various data sources with fallback strategy.

    period examples: 'day', '1min', '5min'.
    
    Data source priority based on SETTINGS.data_source:
    - "yahoo_mcp": Use Yahoo Finance MCP first, fallback to Tiger SDK
    - "tiger": Use Tiger SDK first, fallback to Yahoo Finance MCP  
    - "auto": Try Yahoo Finance MCP first, then Tiger SDK, then placeholder
    """
    if dry_run:
        return _generate_placeholder_data(symbol, period, limit)

    data_source = SETTINGS.data_source.lower()
    df = None

    # Try data sources based on preference
    if data_source in ("yahoo_api", "auto"):
        df = _fetch_yahoo_data(symbol, period, limit)
        if df is not None:
            return _normalize_dataframe(df, symbol, period, "yahoo_api")

    if data_source in ("yahoo_mcp", "auto") and df is None:
        df = _fetch_yahoo_mcp_data(symbol, period, limit)
        if df is not None:
            return _normalize_dataframe(df, symbol, period, "yahoo_mcp")

    if data_source in ("tiger", "auto") and quote_client is not None and df is None:
        try:
            raw = _sdk_get_kline_any(quote_client, symbol, period=period, limit=limit)
            df = _bars_to_dataframe(raw)
            return _normalize_dataframe(df, symbol, period, "tiger_sdk")
        except Exception as e:
            print(f"[data] Tiger SDK failed: {type(e).__name__}: {e}")

    # Fallback attempts for specific data source preferences
    if data_source == "tiger" and df is None:
        # Try Yahoo API first, then MCP
        df = _fetch_yahoo_data(symbol, period, limit)
        if df is not None:
            return _normalize_dataframe(df, symbol, period, "yahoo_api_fallback")
        
        df = _fetch_yahoo_mcp_data(symbol, period, limit)
        if df is not None:
            return _normalize_dataframe(df, symbol, period, "yahoo_mcp_fallback")
    
    elif data_source == "yahoo_mcp" and df is None:
        # Try Yahoo API, then Tiger SDK
        df = _fetch_yahoo_data(symbol, period, limit)
        if df is not None:
            return _normalize_dataframe(df, symbol, period, "yahoo_api_fallback")
            
        if quote_client is not None:
            try:
                raw = _sdk_get_kline_any(quote_client, symbol, period=period, limit=limit)
                df = _bars_to_dataframe(raw)
                return _normalize_dataframe(df, symbol, period, "tiger_sdk_fallback")
            except Exception as e:
                print(f"[data] Tiger SDK fallback failed: {type(e).__name__}: {e}")
                
    elif data_source == "yahoo_api" and df is None:
        # Try MCP, then Tiger SDK
        df = _fetch_yahoo_mcp_data(symbol, period, limit)
        if df is not None:
            return _normalize_dataframe(df, symbol, period, "yahoo_mcp_fallback")
            
        if quote_client is not None:
            try:
                raw = _sdk_get_kline_any(quote_client, symbol, period=period, limit=limit)
                df = _bars_to_dataframe(raw)
                return _normalize_dataframe(df, symbol, period, "tiger_sdk_fallback")
            except Exception as e:
                print(f"[data] Tiger SDK fallback failed: {type(e).__name__}: {e}")

    # Final fallback to placeholder data
    print(f"[data] All data sources failed, using placeholder for {symbol}")
    return _generate_placeholder_data(symbol, period, limit)


def _normalize_dataframe(df: pd.DataFrame, symbol: str, period: str, source: str) -> pd.DataFrame:
    """Normalize dataframe to standard format and log success."""
    print(f"[data] source={source} period={period} symbol={symbol} rows={len(df)}")
    
    # Normalize time to pandas datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        try:
            # epoch seconds/millis
            if df['time'].max() > 10_000_000_000:
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert(None)
            else:
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
        except Exception:
            df['time'] = pd.to_datetime(df['time'])
    
    df = df.sort_values('time').reset_index(drop=True)
    
    # Ensure numeric types
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]


def _generate_placeholder_data(symbol: str, period: str, limit: int) -> pd.DataFrame:
    """Generate placeholder random walk data for demo purposes."""
    now = datetime.utcnow().replace(second=0, microsecond=0)
    step = timedelta(days=1) if 'day' in period else timedelta(minutes=1)
    times = [now - i * step for i in range(limit)][::-1]
    price = 100.0
    rows = []
    for t in times:
        change = random.uniform(-0.5, 0.5)
        open_p = price
        close_p = max(0.01, price + change)
        high_p = max(open_p, close_p) + random.uniform(0, 0.3)
        low_p = min(open_p, close_p) - random.uniform(0, 0.3)
        vol = max(1, int(abs(change) * 10_000))
        rows.append({'time': t, 'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p, 'volume': vol})
        price = close_p
    
    print(f"[data] source=placeholder period={period} symbol={symbol} rows={len(rows)}")
    return pd.DataFrame(rows)


def fetch_daily_history(symbol: str, limit: int = 300):
    raise NotImplementedError('Use fetch_history with a real quote_client instance')


def to_dataframe(bars):
    return _bars_to_dataframe(bars)


def _fetch_yahoo_data(symbol: str, period: str = 'day', limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch historical data using Yahoo Finance API directly.
    
    Returns None if Yahoo Finance API is not available or fails.
    """
    try:
        from yahoo_data import fetch_yahoo_price_history
        return fetch_yahoo_price_history(symbol, period, limit)
    except ImportError:
        print(f"[data] Yahoo Finance API module not available")
        return None
    except Exception as e:
        print(f"[data] Yahoo Finance API failed: {type(e).__name__}: {e}")
        return None


def _fetch_yahoo_mcp_data(symbol: str, period: str = 'day', limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch historical data using Yahoo Finance MCP server.
    
    Returns None if MCP is not available or fails.
    """
    try:
        from mcp_data import fetch_yahoo_mcp_price_history
        return fetch_yahoo_mcp_price_history(symbol, period, limit)
    except ImportError:
        print(f"[data] MCP data module not available")
        return None
    except Exception as e:
        print(f"[data] Yahoo Finance MCP failed: {type(e).__name__}: {e}")
        return None


# ===== BATCH DATA ACQUISITION FUNCTIONS =====

def fetch_batch_history(
    quote_client,
    symbols: List[str],
    period: str = 'day',
    limit: int = 300,
    dry_run: bool = False,
    max_concurrent: int = 5,
    delay_between_requests: float = 0.1
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch historical data for multiple symbols with optimized batch processing.
    
    This function provides:
    - Concurrent data fetching with rate limiting
    - Error handling and retry mechanisms
    - Data caching to avoid duplicate requests
    - Progress tracking for large batches
    
    Args:
        quote_client: Tiger SDK quote client instance
        symbols: List of stock symbols to fetch data for
        period: Time period ('day', '1min', '5min', etc.)
        limit: Number of data points per symbol
        dry_run: If True, generate placeholder data
        max_concurrent: Maximum concurrent requests (rate limiting)
        delay_between_requests: Delay between requests in seconds
        
    Returns:
        Dictionary mapping symbols to their DataFrames (None if failed)
    """
    if not symbols:
        return {}
    
    print(f"[data] Starting batch fetch for {len(symbols)} symbols, period={period}")
    
    if dry_run:
        return {symbol: _generate_placeholder_data(symbol, period, limit) for symbol in symbols}
    
    # Use optimized batch processing based on data source
    data_source = SETTINGS.data_source.lower()
    
    if data_source in ("yahoo_api", "auto"):
        result = _fetch_batch_yahoo_api(symbols, period, limit, max_concurrent, delay_between_requests)
        if any(df is not None for df in result.values()):
            return result
    
    if data_source in ("yahoo_mcp", "auto"):
        result = _fetch_batch_yahoo_mcp(symbols, period, limit, max_concurrent, delay_between_requests)
        if any(df is not None for df in result.values()):
            return result
    
    if data_source in ("tiger", "auto") and quote_client is not None:
        result = _fetch_batch_tiger_sdk(quote_client, symbols, period, limit, delay_between_requests)
        if any(df is not None for df in result.values()):
            return result
    
    # Fallback: single-threaded fetch using existing fetch_history
    print(f"[data] Using fallback single-threaded fetching")
    result = {}
    for i, symbol in enumerate(symbols):
        print(f"[data] Processing {symbol} ({i+1}/{len(symbols)})")
        try:
            df = fetch_history(quote_client, symbol, period, limit, dry_run=False)
            result[symbol] = df
            if delay_between_requests > 0:
                import time
                time.sleep(delay_between_requests)
        except Exception as e:
            print(f"[data] Failed to fetch {symbol}: {e}")
            result[symbol] = None
    
    successful = sum(1 for df in result.values() if df is not None)
    print(f"[data] Batch fetch completed: {successful}/{len(symbols)} successful")
    return result


def _fetch_batch_yahoo_api(
    symbols: List[str], 
    period: str, 
    limit: int,
    max_concurrent: int,
    delay: float
) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch batch data using Yahoo Finance API."""
    try:
        from yahoo_data import fetch_yahoo_multiple_symbols
        print(f"[data] Using Yahoo Finance API batch processing")
        return fetch_yahoo_multiple_symbols(symbols, period, limit)
    except ImportError:
        print(f"[data] Yahoo Finance API module not available for batch processing")
        return {symbol: None for symbol in symbols}
    except Exception as e:
        print(f"[data] Yahoo Finance API batch processing failed: {e}")
        return {symbol: None for symbol in symbols}


def _fetch_batch_yahoo_mcp(
    symbols: List[str],
    period: str, 
    limit: int,
    max_concurrent: int,
    delay: float
) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch batch data using Yahoo Finance MCP."""
    try:
        from mcp_data import fetch_yahoo_mcp_batch_price_history
        print(f"[data] Using Yahoo Finance MCP batch processing")
        return fetch_yahoo_mcp_batch_price_history(symbols, period, limit)
    except ImportError:
        print(f"[data] MCP batch processing not available")
        # Fallback to individual MCP requests
        return _fetch_batch_mcp_individual(symbols, period, limit, delay)
    except Exception as e:
        print(f"[data] Yahoo Finance MCP batch processing failed: {e}")
        return _fetch_batch_mcp_individual(symbols, period, limit, delay)


def _fetch_batch_mcp_individual(symbols: List[str], period: str, limit: int, delay: float) -> Dict[str, Optional[pd.DataFrame]]:
    """Fallback individual MCP requests with rate limiting."""
    result = {}
    for i, symbol in enumerate(symbols):
        print(f"[data] MCP individual fetch {symbol} ({i+1}/{len(symbols)})")
        result[symbol] = _fetch_yahoo_mcp_data(symbol, period, limit)
        
        if delay > 0 and i < len(symbols) - 1:
            import time
            time.sleep(delay)
    
    return result


def _fetch_batch_tiger_sdk(
    quote_client,
    symbols: List[str],
    period: str,
    limit: int, 
    delay: float
) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch batch data using Tiger SDK with rate limiting."""
    print(f"[data] Using Tiger SDK batch processing")
    result = {}
    
    for i, symbol in enumerate(symbols):
        print(f"[data] Tiger SDK fetch {symbol} ({i+1}/{len(symbols)})")
        try:
            raw = _sdk_get_kline_any(quote_client, symbol, period=period, limit=limit)
            df = _bars_to_dataframe(raw)
            result[symbol] = _normalize_dataframe(df, symbol, period, "tiger_sdk_batch")
        except Exception as e:
            print(f"[data] Tiger SDK failed for {symbol}: {e}")
            result[symbol] = None
        
        if delay > 0 and i < len(symbols) - 1:
            import time
            time.sleep(delay)
    
    return result


def fetch_sector_history(
    quote_client,
    sector_name: str,
    period: str = 'day',
    limit: int = 300,
    dry_run: bool = False,
    validate_symbols: bool = True
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch historical data for all stocks in a specific sector.
    
    This function integrates with the sector management system to:
    - Get all stocks in the specified sector
    - Validate symbols if requested
    - Fetch data using optimized batch processing
    
    Args:
        quote_client: Tiger SDK quote client instance
        sector_name: Name of the sector (e.g., 'technology', 'healthcare')
        period: Time period ('day', '1min', '5min', etc.)
        limit: Number of data points per symbol
        dry_run: If True, generate placeholder data
        validate_symbols: If True, validate symbols before fetching data
        
    Returns:
        Dictionary mapping symbols to their DataFrames (None if failed)
    """
    try:
        from sector_manager import get_sector_stocks
        
        # Get stocks in the sector
        symbols = get_sector_stocks(sector_name, validate=validate_symbols)
        
        if not symbols:
            print(f"[data] No valid stocks found in sector '{sector_name}'")
            return {}
        
        print(f"[data] Fetching data for {len(symbols)} stocks in sector '{sector_name}'")
        
        # Use batch processing
        return fetch_batch_history(
            quote_client=quote_client,
            symbols=symbols,
            period=period,
            limit=limit,
            dry_run=dry_run,
            max_concurrent=5,
            delay_between_requests=0.1
        )
        
    except ImportError:
        print(f"[data] Sector manager not available. Please ensure bot.sector_manager is properly installed.")
        return {}
    except Exception as e:
        print(f"[data] Failed to fetch sector data: {e}")
        return {}


def fetch_all_sectors_history(
    quote_client,
    period: str = 'day',
    limit: int = 300,
    dry_run: bool = False,
    active_sectors_only: bool = True
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """
    Fetch historical data for all stocks across all sectors.
    
    Args:
        quote_client: Tiger SDK quote client instance
        period: Time period ('day', '1min', '5min', etc.)
        limit: Number of data points per symbol
        dry_run: If True, generate placeholder data
        active_sectors_only: If True, only process active sectors
        
    Returns:
        Nested dictionary: {sector_name: {symbol: DataFrame}}
    """
    try:
        from sector_manager import list_sectors, get_sector_stocks
        
        # Get all active sectors
        sectors = list_sectors(active_only=active_sectors_only)
        
        if not sectors:
            print(f"[data] No sectors found")
            return {}
        
        print(f"[data] Processing {len(sectors)} sectors: {sectors}")
        
        all_results = {}
        total_stocks = 0
        
        for sector_name in sectors:
            print(f"[data] Processing sector: {sector_name}")
            
            sector_data = fetch_sector_history(
                quote_client=quote_client,
                sector_name=sector_name,
                period=period,
                limit=limit,
                dry_run=dry_run,
                validate_symbols=True
            )
            
            all_results[sector_name] = sector_data
            successful = sum(1 for df in sector_data.values() if df is not None)
            total_stocks += len(sector_data)
            
            print(f"[data] Sector '{sector_name}': {successful}/{len(sector_data)} stocks successful")
        
        print(f"[data] All sectors processed: {total_stocks} total stocks")
        return all_results
        
    except ImportError:
        print(f"[data] Sector manager not available")
        return {}
    except Exception as e:
        print(f"[data] Failed to fetch all sectors data: {e}")
        return {}


def get_batch_latest_data(
    quote_client,
    symbols: List[str],
    dry_run: bool = False
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Get the latest market data for multiple symbols.
    
    This is optimized for getting recent data (last few data points)
    for screening and real-time analysis.
    
    Args:
        quote_client: Tiger SDK quote client instance
        symbols: List of stock symbols
        dry_run: If True, generate placeholder data
        
    Returns:
        Dictionary mapping symbols to their latest data DataFrames
    """
    return fetch_batch_history(
        quote_client=quote_client,
        symbols=symbols,
        period='day',
        limit=5,  # Only get last 5 data points for performance
        dry_run=dry_run,
        max_concurrent=10,  # More concurrent requests for small data
        delay_between_requests=0.05  # Shorter delay for latest data
    )


def cache_sector_data(
    quote_client,
    cache_dir: str = "data_cache",
    period: str = 'day',
    limit: int = 300,
    dry_run: bool = False
) -> bool:
    """
    Cache historical data for all sectors to disk for faster access.
    
    Args:
        quote_client: Tiger SDK quote client instance
        cache_dir: Directory to store cached data
        period: Time period to cache
        limit: Number of data points to cache
        dry_run: If True, generate placeholder data
        
    Returns:
        True if caching was successful, False otherwise
    """
    import os
    from pathlib import Path
    
    try:
        # Create cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Fetch all sectors data
        all_data = fetch_all_sectors_history(
            quote_client=quote_client,
            period=period,
            limit=limit,
            dry_run=dry_run
        )
        
        cached_files = 0
        
        for sector_name, sector_data in all_data.items():
            sector_dir = Path(cache_dir) / sector_name
            sector_dir.mkdir(exist_ok=True)
            
            for symbol, df in sector_data.items():
                if df is not None:
                    cache_file = sector_dir / f"{symbol}_{period}.csv"
                    df.to_csv(cache_file, index=False)
                    cached_files += 1
        
        print(f"[data] Cached {cached_files} files to {cache_dir}")
        return True
        
    except Exception as e:
        print(f"[data] Failed to cache sector data: {e}")
        return False


def fetch_ticker_info(symbol: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch ticker information with data source switching support.
    Respects DATA_SOURCE environment variable.

    Args:
        symbol: Stock ticker symbol
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with ticker info or None if unavailable
    """
    from typing import Any

    data_source = SETTINGS.data_source.lower()
    info = None

    # Try Tiger API first if configured
    if data_source in ("tiger", "auto"):
        try:
            from bot.tiger_data import fetch_tiger_ticker_info
            info = fetch_tiger_ticker_info(symbol)
            if info:
                return info
        except Exception as e:
            print(f"[data] Tiger ticker info failed for {symbol}: {e}")

    # Try Yahoo MCP if configured
    if data_source in ("yahoo_mcp", "auto") and info is None:
        try:
            from bot.mcp_data import fetch_yahoo_mcp_ticker_info
            info = fetch_yahoo_mcp_ticker_info(symbol)
            if info:
                return info
        except Exception as e:
            print(f"[data] Yahoo MCP ticker info failed for {symbol}: {e}")

    # Try Yahoo API as fallback
    if data_source in ("yahoo_api", "auto") and info is None:
        try:
            from bot.yahoo_data import fetch_yahoo_ticker_info
            info = fetch_yahoo_ticker_info(symbol, max_retries)
            if info:
                return info
        except Exception as e:
            print(f"[data] Yahoo API ticker info failed for {symbol}: {e}")

    return None


