"""
Historical Data API

This module provides a unified API interface for accessing historical market data
with built-in caching, quality validation, and corporate actions adjustments.

Key Features:
- Unified API for multiple data sources
- Intelligent caching with TTL management
- Real-time data quality validation
- Automatic corporate actions adjustments
- Flexible query interface with filters
- Performance optimized for backtesting workloads
- Integration with existing data management systems
- RESTful API endpoints for external access
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path
import hashlib
import time
import threading
from functools import wraps
import warnings

# Import our data management components
from bot.historical_data_manager import HistoricalDataManager, DataSourceType
from bot.data_quality_framework import DataQualityValidator, validate_symbol_data
from bot.corporate_actions_processor import CorporateActionsProcessor, process_corporate_actions_for_symbol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrequency(Enum):
    """Data frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class AdjustmentType(Enum):
    """Price adjustment types."""
    NONE = "none"              # Raw prices
    SPLITS = "splits"          # Split-adjusted only
    SPLITS_DIVIDENDS = "all"   # Split and dividend adjusted
    FULL = "full"              # All corporate actions


@dataclass
class DataQuery:
    """Data query specification."""
    symbols: List[str]
    start_date: str
    end_date: str
    frequency: DataFrequency = DataFrequency.DAILY
    adjustment_type: AdjustmentType = AdjustmentType.FULL
    include_volume: bool = True
    validate_quality: bool = True
    fill_missing: bool = True
    remove_outliers: bool = False
    min_quality_score: float = 0.8

    def to_cache_key(self) -> str:
        """Generate cache key for this query."""
        key_data = {
            'symbols': sorted(self.symbols),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'frequency': self.frequency.value,
            'adjustment_type': self.adjustment_type.value,
            'include_volume': self.include_volume,
            'validate_quality': self.validate_quality,
            'fill_missing': self.fill_missing,
            'remove_outliers': self.remove_outliers,
            'min_quality_score': self.min_quality_score
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class QueryResult:
    """Result of a data query."""
    data: Dict[str, pd.DataFrame]  # Symbol -> DataFrame mapping
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    cache_hit: bool = False
    query_time_ms: float = 0.0
    total_records: int = 0
    symbols_found: List[str] = None
    symbols_missing: List[str] = None

    def __post_init__(self):
        if self.symbols_found is None:
            self.symbols_found = list(self.data.keys())
        if self.symbols_missing is None:
            self.symbols_missing = []
        self.total_records = sum(len(df) for df in self.data.values())


class PerformanceMonitor:
    """Monitor API performance and usage."""

    def __init__(self):
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'total_records_served': 0
        }
        self.recent_queries = []
        self.max_recent_queries = 1000

    def record_query(self, query: DataQuery, result: QueryResult):
        """Record query performance statistics."""
        self.query_stats['total_queries'] += 1
        if result.cache_hit:
            self.query_stats['cache_hits'] += 1

        # Update average response time
        total_time = (self.query_stats['avg_response_time'] *
                     (self.query_stats['total_queries'] - 1) +
                     result.query_time_ms)
        self.query_stats['avg_response_time'] = total_time / self.query_stats['total_queries']

        self.query_stats['total_records_served'] += result.total_records

        # Store recent query info
        query_info = {
            'timestamp': datetime.now().isoformat(),
            'symbols_count': len(query.symbols),
            'date_range_days': (datetime.strptime(query.end_date, '%Y-%m-%d') -
                               datetime.strptime(query.start_date, '%Y-%m-%d')).days,
            'response_time_ms': result.query_time_ms,
            'cache_hit': result.cache_hit,
            'records_returned': result.total_records
        }

        self.recent_queries.append(query_info)
        if len(self.recent_queries) > self.max_recent_queries:
            self.recent_queries.pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (self.query_stats['cache_hits'] /
                         max(1, self.query_stats['total_queries']) * 100)

        recent_avg_time = 0.0
        if len(self.recent_queries) > 0:
            recent_times = [q['response_time_ms'] for q in self.recent_queries[-100:]]
            recent_avg_time = sum(recent_times) / len(recent_times)

        return {
            'total_queries': self.query_stats['total_queries'],
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time_ms': self.query_stats['avg_response_time'],
            'recent_avg_response_time_ms': recent_avg_time,
            'total_records_served': self.query_stats['total_records_served'],
            'queries_last_hour': len([q for q in self.recent_queries
                                    if datetime.fromisoformat(q['timestamp']) >
                                    datetime.now() - timedelta(hours=1)])
        }


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Add timing info to result if it's a QueryResult
        if isinstance(result, QueryResult):
            result.query_time_ms = execution_time

        return result
    return wrapper


class HistoricalDataAPI:
    """
    Unified API for historical market data access.

    Provides high-level interface with caching, validation, and adjustments.
    """

    def __init__(
        self,
        data_manager: Optional[HistoricalDataManager] = None,
        quality_validator: Optional[DataQualityValidator] = None,
        corporate_actions: Optional[CorporateActionsProcessor] = None,
        cache_ttl: int = 3600,  # 1 hour default
        max_cache_size: int = 1000
    ):
        """
        Initialize Historical Data API.

        Args:
            data_manager: Historical data manager instance
            quality_validator: Data quality validator instance
            corporate_actions: Corporate actions processor instance
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cached queries
        """
        self.data_manager = data_manager or HistoricalDataManager()
        self.quality_validator = quality_validator or DataQualityValidator()
        self.corporate_actions = corporate_actions or CorporateActionsProcessor()

        # Caching
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._query_cache = {}
        self._cache_timestamps = {}
        self._cache_lock = threading.RLock()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Supported frequencies
        self._frequency_handlers = {
            DataFrequency.DAILY: self._get_daily_data,
            DataFrequency.WEEKLY: self._get_weekly_data,
            DataFrequency.MONTHLY: self._get_monthly_data,
            DataFrequency.QUARTERLY: self._get_quarterly_data
        }

        logger.info("[api] Historical Data API initialized")

    @timing_decorator
    def get_data(self, query: DataQuery) -> QueryResult:
        """
        Get historical data based on query specification.

        Args:
            query: Data query specification

        Returns:
            Query result with data and metadata
        """
        logger.info(f"[api] Processing query for {len(query.symbols)} symbols "
                   f"from {query.start_date} to {query.end_date}")

        # Check cache first
        cache_key = query.to_cache_key()
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.info("[api] Returning cached result")
            self.performance_monitor.record_query(query, cached_result)
            return cached_result

        # Process query
        start_time = time.perf_counter()

        try:
            # Get raw data for all symbols
            raw_data = {}
            symbols_found = []
            symbols_missing = []

            for symbol in query.symbols:
                try:
                    df = self._get_symbol_data(symbol, query)
                    if df is not None and len(df) > 0:
                        raw_data[symbol] = df
                        symbols_found.append(symbol)
                    else:
                        symbols_missing.append(symbol)
                        logger.warning(f"[api] No data found for symbol: {symbol}")
                except Exception as e:
                    logger.error(f"[api] Error fetching data for {symbol}: {e}")
                    symbols_missing.append(symbol)

            # Process data based on query parameters
            processed_data = {}
            quality_metrics = {}

            for symbol, df in raw_data.items():
                try:
                    processed_df = self._process_symbol_data(df, symbol, query)
                    processed_data[symbol] = processed_df

                    # Calculate quality metrics if requested
                    if query.validate_quality:
                        _, metrics = validate_symbol_data(processed_df, symbol, perform_fixes=False)
                        quality_metrics[symbol] = metrics.to_dict()

                except Exception as e:
                    logger.error(f"[api] Error processing data for {symbol}: {e}")
                    symbols_missing.append(symbol)
                    if symbol in symbols_found:
                        symbols_found.remove(symbol)

            # Create result
            result = QueryResult(
                data=processed_data,
                metadata={
                    'query': asdict(query),
                    'execution_timestamp': datetime.now().isoformat(),
                    'data_sources_used': ['historical_data_manager'],
                    'adjustments_applied': query.adjustment_type.value,
                    'quality_validation': query.validate_quality
                },
                quality_metrics=quality_metrics,
                cache_hit=False,
                symbols_found=symbols_found,
                symbols_missing=symbols_missing
            )

            # Cache the result
            self._store_in_cache(cache_key, result)

            # Record performance metrics
            self.performance_monitor.record_query(query, result)

            logger.info(f"[api] Query completed: {len(symbols_found)} symbols found, "
                       f"{len(symbols_missing)} missing, {result.total_records} total records")

            return result

        except Exception as e:
            logger.error(f"[api] Query processing failed: {e}")
            raise

    def _get_symbol_data(self, symbol: str, query: DataQuery) -> Optional[pd.DataFrame]:
        """Get raw data for a single symbol."""
        try:
            # Use appropriate frequency handler
            frequency_handler = self._frequency_handlers.get(query.frequency, self._get_daily_data)
            return frequency_handler(symbol, query.start_date, query.end_date)

        except Exception as e:
            logger.error(f"[api] Failed to get data for {symbol}: {e}")
            return None

    def _get_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get daily frequency data."""
        return self.data_manager.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjusted=True
        )

    def _get_weekly_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get weekly frequency data."""
        daily_df = self._get_daily_data(symbol, start_date, end_date)
        if len(daily_df) == 0:
            return daily_df

        # Resample to weekly (Friday close)
        daily_df.set_index('date', inplace=True)
        weekly_df = daily_df.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return weekly_df.reset_index()

    def _get_monthly_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get monthly frequency data."""
        daily_df = self._get_daily_data(symbol, start_date, end_date)
        if len(daily_df) == 0:
            return daily_df

        # Resample to monthly (month end)
        daily_df.set_index('date', inplace=True)
        monthly_df = daily_df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return monthly_df.reset_index()

    def _get_quarterly_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get quarterly frequency data."""
        daily_df = self._get_daily_data(symbol, start_date, end_date)
        if len(daily_df) == 0:
            return daily_df

        # Resample to quarterly (quarter end)
        daily_df.set_index('date', inplace=True)
        quarterly_df = daily_df.resample('Q').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return quarterly_df.reset_index()

    def _process_symbol_data(self, df: pd.DataFrame, symbol: str, query: DataQuery) -> pd.DataFrame:
        """Process data for a single symbol based on query parameters."""
        processed_df = df.copy()

        # Apply corporate actions adjustments
        if query.adjustment_type != AdjustmentType.NONE:
            processed_df = self.corporate_actions.apply_corporate_actions(
                processed_df, symbol
            )

        # Quality validation and fixes
        if query.validate_quality:
            processed_df, metrics = validate_symbol_data(
                processed_df, symbol, perform_fixes=True
            )

            # Filter by quality score if specified
            if hasattr(metrics, 'overall_quality_score'):
                if metrics.overall_quality_score < query.min_quality_score:
                    logger.warning(f"[api] {symbol} quality score {metrics.overall_quality_score:.3f} "
                                 f"below threshold {query.min_quality_score}")

        # Fill missing data if requested
        if query.fill_missing:
            processed_df = self._fill_missing_data(processed_df)

        # Remove outliers if requested
        if query.remove_outliers:
            processed_df = self._remove_outliers(processed_df)

        # Remove volume column if not requested
        if not query.include_volume and 'volume' in processed_df.columns:
            processed_df = processed_df.drop(columns=['volume'])

        return processed_df

    def _fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing data using forward fill method."""
        if len(df) == 0:
            return df

        # Sort by date
        df_sorted = df.sort_values('date').copy()

        # Forward fill missing values
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df_sorted.columns:
                df_sorted[col] = df_sorted[col].fillna(method='ffill')

        # For volume, use interpolation or zero
        if 'volume' in df_sorted.columns:
            df_sorted['volume'] = df_sorted['volume'].fillna(0)

        return df_sorted

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from data."""
        if len(df) < 10:  # Need enough data
            return df

        df_clean = df.copy()

        # Calculate returns for outlier detection
        df_clean = df_clean.sort_values('date')
        df_clean['returns'] = df_clean['close'].pct_change()

        # Remove return outliers using IQR method
        Q1 = df_clean['returns'].quantile(0.25)
        Q3 = df_clean['returns'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep only non-outlier records
        outlier_mask = (
            (df_clean['returns'] >= lower_bound) &
            (df_clean['returns'] <= upper_bound)
        ) | df_clean['returns'].isna()

        df_clean = df_clean[outlier_mask].drop(columns=['returns'])

        removed_count = len(df) - len(df_clean)
        if removed_count > 0:
            logger.info(f"[api] Removed {removed_count} outlier records")

        return df_clean

    def _get_from_cache(self, cache_key: str) -> Optional[QueryResult]:
        """Get result from cache if available and not expired."""
        with self._cache_lock:
            if cache_key not in self._query_cache:
                return None

            # Check if expired
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time > self.cache_ttl:
                del self._query_cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None

            # Return cached result with cache_hit flag
            cached_result = self._query_cache[cache_key]
            cached_result.cache_hit = True
            return cached_result

    def _store_in_cache(self, cache_key: str, result: QueryResult):
        """Store result in cache."""
        with self._cache_lock:
            # Implement LRU eviction if cache is full
            if len(self._query_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self._cache_timestamps.keys(),
                               key=lambda k: self._cache_timestamps[k])
                del self._query_cache[oldest_key]
                del self._cache_timestamps[oldest_key]

            # Store new result
            self._query_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

    def clear_cache(self):
        """Clear all cached results."""
        with self._cache_lock:
            self._query_cache.clear()
            self._cache_timestamps.clear()
        logger.info("[api] Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            cache_size = len(self._query_cache)

            # Calculate cache memory usage (approximate)
            total_records = sum(
                sum(len(df) for df in result.data.values())
                for result in self._query_cache.values()
            )

            return {
                'cache_size': cache_size,
                'max_cache_size': self.max_cache_size,
                'cache_utilization': cache_size / self.max_cache_size,
                'total_cached_records': total_records,
                'cache_ttl_seconds': self.cache_ttl
            }

    # Convenience methods for common queries

    def get_price_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        adjusted: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get price data for symbols (convenience method).

        Args:
            symbols: Symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency
            adjusted: Whether to apply corporate action adjustments

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        adjustment_type = AdjustmentType.FULL if adjusted else AdjustmentType.NONE

        query = DataQuery(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustment_type=adjustment_type,
            include_volume=True,
            validate_quality=True
        )

        result = self.get_data(query)
        return result.data

    def get_returns_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> Dict[str, pd.DataFrame]:
        """
        Get returns data for symbols.

        Args:
            symbols: Symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency

        Returns:
            Dictionary mapping symbols to DataFrames with returns
        """
        price_data = self.get_price_data(symbols, start_date, end_date, frequency, adjusted=True)

        returns_data = {}
        for symbol, df in price_data.items():
            if len(df) > 1:
                df_returns = df.copy()
                df_returns['returns'] = df_returns['close'].pct_change()
                df_returns['log_returns'] = np.log(df_returns['close'] / df_returns['close'].shift(1))
                returns_data[symbol] = df_returns
            else:
                returns_data[symbol] = df

        return returns_data

    def get_ohlcv_matrix(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        field: str = 'close'
    ) -> pd.DataFrame:
        """
        Get OHLCV data as a matrix (dates x symbols).

        Args:
            symbols: List of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            field: OHLCV field to extract ('open', 'high', 'low', 'close', 'volume')

        Returns:
            DataFrame with dates as index and symbols as columns
        """
        price_data = self.get_price_data(symbols, start_date, end_date)

        # Create matrix
        matrix_data = {}
        for symbol, df in price_data.items():
            if field in df.columns:
                matrix_data[symbol] = df.set_index('date')[field]

        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            return matrix_df.sort_index()
        else:
            return pd.DataFrame()

    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and statistics."""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'performance': self.performance_monitor.get_performance_stats(),
            'cache': self.get_cache_stats(),
            'components': {
                'data_manager': 'healthy',
                'quality_validator': 'healthy',
                'corporate_actions': 'healthy'
            }
        }


# Global API instance for easy access
_api_instance = None

def get_api() -> HistoricalDataAPI:
    """Get global API instance (singleton pattern)."""
    global _api_instance
    if _api_instance is None:
        _api_instance = HistoricalDataAPI()
    return _api_instance


# Convenience functions
def get_historical_data(
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    frequency: str = "daily",
    adjusted: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to get historical data.

    Args:
        symbols: Symbol or list of symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        frequency: 'daily', 'weekly', 'monthly', or 'quarterly'
        adjusted: Whether to apply corporate action adjustments

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    api = get_api()
    freq_enum = DataFrequency(frequency.lower())
    return api.get_price_data(symbols, start_date, end_date, freq_enum, adjusted)


def get_returns(
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    frequency: str = "daily"
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to get returns data.
    """
    api = get_api()
    freq_enum = DataFrequency(frequency.lower())
    return api.get_returns_data(symbols, start_date, end_date, freq_enum)


def get_price_matrix(
    symbols: List[str],
    start_date: str,
    end_date: str,
    field: str = 'close'
) -> pd.DataFrame:
    """
    Convenience function to get price matrix.
    """
    api = get_api()
    return api.get_ohlcv_matrix(symbols, start_date, end_date, field)


if __name__ == "__main__":
    # Example usage
    print("Testing Historical Data API...")

    # Initialize API
    api = HistoricalDataAPI()

    # Test basic query
    symbols = ["AAPL", "MSFT", "GOOGL"]
    query = DataQuery(
        symbols=symbols,
        start_date="2020-01-01",
        end_date="2023-12-31",
        frequency=DataFrequency.DAILY,
        adjustment_type=AdjustmentType.FULL,
        validate_quality=True
    )

    print(f"Querying data for {len(symbols)} symbols...")
    result = api.get_data(query)

    print(f"Query Results:")
    print(f"  Symbols found: {len(result.symbols_found)}")
    print(f"  Symbols missing: {len(result.symbols_missing)}")
    print(f"  Total records: {result.total_records}")
    print(f"  Query time: {result.query_time_ms:.1f}ms")
    print(f"  Cache hit: {result.cache_hit}")

    # Test convenience functions
    print("\nTesting convenience functions...")
    price_data = get_historical_data("AAPL", "2023-01-01", "2023-12-31")
    if "AAPL" in price_data:
        print(f"AAPL data: {len(price_data['AAPL'])} records")

    # Test price matrix
    matrix = get_price_matrix(["AAPL", "MSFT"], "2023-01-01", "2023-03-31")
    print(f"Price matrix shape: {matrix.shape}")

    # Show API status
    status = api.get_api_status()
    print(f"\nAPI Status:")
    print(f"  Performance: {status['performance']}")
    print(f"  Cache: {status['cache']}")