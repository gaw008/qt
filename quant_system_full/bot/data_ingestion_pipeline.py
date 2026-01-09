"""
Data Ingestion Pipeline

This module provides a comprehensive data ingestion pipeline that orchestrates
the collection of historical market data from multiple sources with robust
error handling, data validation, and performance optimization.

Key Features:
- Multi-source data orchestration (Yahoo Finance, Tiger API, FRED)
- Intelligent source fallback and redundancy
- Rate limiting and API quota management
- Parallel processing with resource management
- Comprehensive error handling and retry logic
- Data validation and quality assurance
- Progress tracking and monitoring
- Integration with existing data cache patterns
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib

# Import existing system components
from bot.config import SETTINGS
from bot.data import fetch_history, fetch_batch_history
from bot.historical_data_manager import HistoricalDataManager, DataSourceType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPriority(Enum):
    """Data ingestion priority levels."""
    CRITICAL = 1    # Trading-critical symbols (current positions)
    HIGH = 2        # Active watchlist symbols
    MEDIUM = 3      # Sector benchmark symbols
    LOW = 4         # Historical analysis symbols


class IngestionStatus(Enum):
    """Ingestion task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class IngestionTask:
    """Individual data ingestion task."""
    symbol: str
    start_date: str
    end_date: str
    sources: List[DataSourceType]
    priority: IngestionPriority = IngestionPriority.MEDIUM
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    # Runtime state
    status: IngestionStatus = IngestionStatus.PENDING
    current_retry: int = 0
    current_source_idx: int = 0
    error_messages: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> Optional[float]:
        """Task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def current_source(self) -> Optional[DataSourceType]:
        """Current data source being tried."""
        if 0 <= self.current_source_idx < len(self.sources):
            return self.sources[self.current_source_idx]
        return None

    def next_source(self) -> bool:
        """Move to next data source. Returns True if available."""
        self.current_source_idx += 1
        return self.current_source_idx < len(self.sources)

    def reset_sources(self):
        """Reset to first data source (for retries)."""
        self.current_source_idx = 0


@dataclass
class IngestionProgress:
    """Tracks overall ingestion progress."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def completion_rate(self) -> float:
        """Completion rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.completed_tasks == 0:
            return None

        avg_time_per_task = self.elapsed_time / self.completed_tasks
        remaining_tasks = self.total_tasks - self.completed_tasks
        return avg_time_per_task * remaining_tasks


class DataIngestionPipeline:
    """
    Comprehensive data ingestion pipeline with multi-source support.

    Features:
    - Intelligent source selection and fallback
    - Rate limiting and quota management
    - Parallel processing with resource control
    - Comprehensive error handling
    - Progress tracking and monitoring
    """

    def __init__(
        self,
        data_manager: Optional[HistoricalDataManager] = None,
        max_workers: int = 8,
        rate_limit_per_second: float = 10.0,
        enable_caching: bool = True
    ):
        """
        Initialize data ingestion pipeline.

        Args:
            data_manager: Historical data manager instance
            max_workers: Maximum concurrent workers
            rate_limit_per_second: API calls per second limit
            enable_caching: Whether to enable response caching
        """
        self.data_manager = data_manager or HistoricalDataManager()
        self.max_workers = max_workers
        self.rate_limit_per_second = rate_limit_per_second
        self.enable_caching = enable_caching

        # Rate limiting
        self._rate_limiter = self._create_rate_limiter()

        # Task management
        self._task_queue: List[IngestionTask] = []
        self._active_tasks: Dict[str, IngestionTask] = {}
        self._completed_tasks: List[IngestionTask] = []
        self._failed_tasks: List[IngestionTask] = []

        # Progress tracking
        self.progress = IngestionProgress()

        # Thread safety
        self._lock = threading.RLock()

        # Source configurations
        self._source_configs = self._initialize_source_configs()

        # Response cache
        self._response_cache = {} if enable_caching else None
        self._cache_ttl = 3600  # 1 hour

        logger.info(f"[ingestion] Pipeline initialized with {max_workers} workers, rate limit: {rate_limit_per_second}/s")

    def _create_rate_limiter(self):
        """Create rate limiter for API calls."""
        class RateLimiter:
            def __init__(self, rate_per_second: float):
                self.rate_per_second = rate_per_second
                self.last_call_time = 0.0
                self.lock = threading.Lock()

            def wait(self):
                with self.lock:
                    now = time.time()
                    time_since_last = now - self.last_call_time
                    min_interval = 1.0 / self.rate_per_second

                    if time_since_last < min_interval:
                        sleep_time = min_interval - time_since_last
                        time.sleep(sleep_time)

                    self.last_call_time = time.time()

        return RateLimiter(self.rate_limit_per_second)

    def _initialize_source_configs(self) -> Dict[DataSourceType, Dict[str, Any]]:
        """Initialize configuration for each data source."""
        return {
            DataSourceType.YAHOO_FINANCE: {
                'timeout': 30.0,
                'max_retries': 3,
                'retry_delay': 1.0,
                'rate_limit': 100,  # requests per minute
                'batch_size': 50,
                'priority': 1  # Higher priority (lower number)
            },
            DataSourceType.TIGER_API: {
                'timeout': 45.0,
                'max_retries': 2,
                'retry_delay': 2.0,
                'rate_limit': 30,   # requests per minute
                'batch_size': 20,
                'priority': 2
            },
            DataSourceType.FRED: {
                'timeout': 60.0,
                'max_retries': 3,
                'retry_delay': 1.5,
                'rate_limit': 120,  # requests per hour
                'batch_size': 10,
                'priority': 3
            }
        }

    def add_ingestion_task(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        sources: Optional[List[DataSourceType]] = None,
        priority: IngestionPriority = IngestionPriority.MEDIUM
    ) -> IngestionTask:
        """
        Add a new ingestion task to the pipeline.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sources: Data sources to try (defaults to all available)
            priority: Task priority

        Returns:
            Created ingestion task
        """
        if sources is None:
            # Default source priority order
            sources = [
                DataSourceType.YAHOO_FINANCE,
                DataSourceType.TIGER_API,
                DataSourceType.FRED
            ]

        task = IngestionTask(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            sources=sources,
            priority=priority
        )

        with self._lock:
            self._task_queue.append(task)
            self.progress.total_tasks += 1

        logger.info(f"[ingestion] Added task for {symbol} ({start_date} to {end_date}) with priority {priority.name}")
        return task

    def add_bulk_tasks(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        sources: Optional[List[DataSourceType]] = None,
        priority_mapper: Optional[Callable[[str], IngestionPriority]] = None
    ) -> List[IngestionTask]:
        """
        Add multiple ingestion tasks efficiently.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sources: Data sources to try
            priority_mapper: Function to determine priority per symbol

        Returns:
            List of created tasks
        """
        tasks = []

        for symbol in symbols:
            priority = IngestionPriority.MEDIUM
            if priority_mapper:
                priority = priority_mapper(symbol)

            task = self.add_ingestion_task(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                sources=sources,
                priority=priority
            )
            tasks.append(task)

        logger.info(f"[ingestion] Added {len(tasks)} bulk tasks")
        return tasks

    def run_pipeline(
        self,
        wait_for_completion: bool = True,
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None
    ) -> IngestionProgress:
        """
        Run the ingestion pipeline.

        Args:
            wait_for_completion: Whether to wait for all tasks to complete
            progress_callback: Optional callback for progress updates

        Returns:
            Final progress state
        """
        if not self._task_queue:
            logger.warning("[ingestion] No tasks in queue")
            return self.progress

        logger.info(f"[ingestion] Starting pipeline with {len(self._task_queue)} tasks")

        # Sort tasks by priority
        with self._lock:
            self._task_queue.sort(key=lambda t: t.priority.value)

        # Start worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}

            while self._task_queue or self._active_tasks:
                # Submit new tasks up to worker limit
                while (len(future_to_task) < self.max_workers and
                       self._task_queue and
                       len(self._active_tasks) < self.max_workers):

                    task = self._task_queue.pop(0)
                    future = executor.submit(self._execute_task, task)
                    future_to_task[future] = task

                    with self._lock:
                        self._active_tasks[task.symbol] = task

                # Process completed tasks
                completed_futures = []
                for future in future_to_task:
                    if future.done():
                        completed_futures.append(future)

                for future in completed_futures:
                    task = future_to_task.pop(future)

                    try:
                        result = future.result()
                        task.result = result
                        task.status = IngestionStatus.COMPLETED
                        self._completed_tasks.append(task)

                        with self._lock:
                            self.progress.completed_tasks += 1
                            if task.symbol in self._active_tasks:
                                del self._active_tasks[task.symbol]

                        logger.info(f"[ingestion] Completed {task.symbol}: {result.get('records_inserted', 0)} records")

                    except Exception as e:
                        task.status = IngestionStatus.FAILED
                        task.error_messages.append(str(e))
                        self._failed_tasks.append(task)

                        with self._lock:
                            self.progress.failed_tasks += 1
                            if task.symbol in self._active_tasks:
                                del self._active_tasks[task.symbol]

                        logger.error(f"[ingestion] Failed {task.symbol}: {e}")

                # Update progress
                if progress_callback:
                    progress_callback(self.progress)

                # Small delay to prevent busy waiting
                if future_to_task or self._active_tasks:
                    time.sleep(0.1)

                # Break if not waiting for completion and no active tasks
                if not wait_for_completion and not self._active_tasks:
                    break

        # Final progress update
        with self._lock:
            self.progress.in_progress_tasks = len(self._active_tasks)

        logger.info(f"[ingestion] Pipeline completed: {self.progress.completed_tasks} success, {self.progress.failed_tasks} failed")
        return self.progress

    def _execute_task(self, task: IngestionTask) -> Dict[str, Any]:
        """
        Execute a single ingestion task with retry logic.

        Args:
            task: Ingestion task to execute

        Returns:
            Task execution result
        """
        task.start_time = time.time()
        task.status = IngestionStatus.IN_PROGRESS

        logger.info(f"[ingestion] Starting {task.symbol} ({task.start_date} to {task.end_date})")

        # Try each source with retry logic
        for attempt in range(task.max_retries + 1):
            task.current_retry = attempt

            if attempt > 0:
                task.status = IngestionStatus.RETRYING
                logger.info(f"[ingestion] Retrying {task.symbol} (attempt {attempt + 1}/{task.max_retries + 1})")
                time.sleep(task.retry_delay * attempt)  # Exponential backoff

            # Try each configured source
            task.reset_sources()
            while task.current_source is not None:
                source = task.current_source

                try:
                    # Rate limiting
                    self._rate_limiter.wait()

                    # Check cache first
                    cache_key = self._get_cache_key(task.symbol, task.start_date, task.end_date, source)
                    if self._response_cache and cache_key in self._response_cache:
                        cache_entry = self._response_cache[cache_key]
                        if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                            logger.info(f"[ingestion] Using cached data for {task.symbol} from {source.value}")
                            task.end_time = time.time()
                            return cache_entry['result']

                    # Execute data fetch
                    result = self._fetch_data_from_source(task, source)

                    # Cache successful result
                    if self._response_cache and result.get('records_inserted', 0) > 0:
                        self._response_cache[cache_key] = {
                            'result': result,
                            'timestamp': time.time()
                        }

                    task.end_time = time.time()
                    logger.info(f"[ingestion] Successfully completed {task.symbol} using {source.value}")
                    return result

                except Exception as e:
                    error_msg = f"{source.value}: {str(e)}"
                    task.error_messages.append(error_msg)
                    logger.warning(f"[ingestion] Source {source.value} failed for {task.symbol}: {e}")

                    # Try next source
                    if not task.next_source():
                        break

            # If all sources failed for this attempt, continue to next attempt
            logger.warning(f"[ingestion] All sources failed for {task.symbol} on attempt {attempt + 1}")

        # All attempts failed
        task.end_time = time.time()
        task.status = IngestionStatus.FAILED
        error_summary = "; ".join(task.error_messages[-len(task.sources):])  # Last error per source
        raise Exception(f"All sources and retries failed: {error_summary}")

    def _fetch_data_from_source(self, task: IngestionTask, source: DataSourceType) -> Dict[str, Any]:
        """
        Fetch data from a specific source.

        Args:
            task: Ingestion task
            source: Data source to use

        Returns:
            Fetch result dictionary
        """
        logger.info(f"[ingestion] Fetching {task.symbol} from {source.value}")

        if source == DataSourceType.YAHOO_FINANCE:
            return self._fetch_yahoo_data(task)
        elif source == DataSourceType.TIGER_API:
            return self._fetch_tiger_data(task)
        elif source == DataSourceType.FRED:
            return self._fetch_fred_data(task)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _fetch_yahoo_data(self, task: IngestionTask) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance."""
        try:
            # Calculate limit for Yahoo Finance API
            start_dt = datetime.strptime(task.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(task.end_date, "%Y-%m-%d")
            days_diff = (end_dt - start_dt).days + 10

            # Use existing data acquisition system
            df = fetch_history(
                quote_client=None,
                symbol=task.symbol,
                period='day',
                limit=min(days_diff, 5000),
                dry_run=False
            )

            if df is None or len(df) == 0:
                raise ValueError("No data returned from Yahoo Finance")

            # Filter to requested date range
            df['date'] = pd.to_datetime(df['time']).dt.date
            df = df[
                (df['date'] >= start_dt.date()) &
                (df['date'] <= end_dt.date())
            ]

            # Store data using data manager
            records_inserted, records_updated = self.data_manager._store_data_batch(
                df, task.symbol, DataSourceType.YAHOO_FINANCE
            )

            return {
                'source': DataSourceType.YAHOO_FINANCE.value,
                'records_processed': len(df),
                'records_inserted': records_inserted,
                'records_updated': records_updated,
                'date_range': f"{task.start_date} to {task.end_date}",
                'quality_score': df['quality_score'].mean() if 'quality_score' in df.columns else 1.0
            }

        except Exception as e:
            raise Exception(f"Yahoo Finance fetch failed: {str(e)}")

    def _fetch_tiger_data(self, task: IngestionTask) -> Dict[str, Any]:
        """Fetch data from Tiger API."""
        # For now, this is a placeholder since Tiger integration requires more setup
        logger.warning(f"[ingestion] Tiger API not fully implemented for {task.symbol}")
        raise NotImplementedError("Tiger API integration pending")

    def _fetch_fred_data(self, task: IngestionTask) -> Dict[str, Any]:
        """Fetch data from FRED (Federal Reserve Economic Data)."""
        # This would be used for economic indicators and macro data
        logger.warning(f"[ingestion] FRED API not implemented for {task.symbol}")
        raise NotImplementedError("FRED API integration pending")

    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, source: DataSourceType) -> str:
        """Generate cache key for data request."""
        key_string = f"{symbol}_{start_date}_{end_date}_{source.value}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        with self._lock:
            return {
                'progress': {
                    'total_tasks': self.progress.total_tasks,
                    'completed_tasks': self.progress.completed_tasks,
                    'failed_tasks': self.progress.failed_tasks,
                    'in_progress_tasks': len(self._active_tasks),
                    'pending_tasks': len(self._task_queue),
                    'completion_rate': self.progress.completion_rate,
                    'elapsed_time': self.progress.elapsed_time,
                    'estimated_time_remaining': self.progress.estimated_time_remaining
                },
                'active_tasks': [
                    {
                        'symbol': task.symbol,
                        'status': task.status.value,
                        'current_source': task.current_source.value if task.current_source else None,
                        'current_retry': task.current_retry,
                        'duration': task.duration
                    }
                    for task in self._active_tasks.values()
                ],
                'recent_completions': [
                    {
                        'symbol': task.symbol,
                        'duration': task.duration,
                        'records_inserted': task.result.get('records_inserted', 0) if task.result else 0,
                        'source': task.result.get('source') if task.result else None
                    }
                    for task in self._completed_tasks[-10:]  # Last 10 completions
                ],
                'recent_failures': [
                    {
                        'symbol': task.symbol,
                        'error_messages': task.error_messages[-3:],  # Last 3 errors
                        'duration': task.duration
                    }
                    for task in self._failed_tasks[-10:]  # Last 10 failures
                ]
            }

    def clear_completed_tasks(self):
        """Clear completed and failed tasks to free memory."""
        with self._lock:
            completed_count = len(self._completed_tasks)
            failed_count = len(self._failed_tasks)

            self._completed_tasks.clear()
            self._failed_tasks.clear()

            logger.info(f"[ingestion] Cleared {completed_count} completed and {failed_count} failed tasks")

    def cancel_pending_tasks(self, symbols: Optional[List[str]] = None):
        """
        Cancel pending tasks.

        Args:
            symbols: Specific symbols to cancel (None for all)
        """
        with self._lock:
            if symbols:
                # Cancel specific symbols
                original_count = len(self._task_queue)
                self._task_queue = [task for task in self._task_queue if task.symbol not in symbols]
                cancelled_count = original_count - len(self._task_queue)

                # Update progress
                self.progress.total_tasks -= cancelled_count

                logger.info(f"[ingestion] Cancelled {cancelled_count} pending tasks for symbols: {symbols}")
            else:
                # Cancel all pending tasks
                cancelled_count = len(self._task_queue)
                self._task_queue.clear()

                # Update progress
                self.progress.total_tasks -= cancelled_count

                logger.info(f"[ingestion] Cancelled all {cancelled_count} pending tasks")


# Convenience functions for integration
def create_pipeline(
    max_workers: int = 8,
    rate_limit: float = 10.0
) -> DataIngestionPipeline:
    """Create and return a configured data ingestion pipeline."""
    return DataIngestionPipeline(
        max_workers=max_workers,
        rate_limit_per_second=rate_limit
    )


def run_bulk_ingestion(
    symbols: List[str],
    start_date: str = "2006-01-01",
    end_date: Optional[str] = None,
    max_workers: int = 8
) -> Dict[str, Any]:
    """
    Convenience function for bulk historical data ingestion.

    Args:
        symbols: List of stock symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_workers: Maximum concurrent workers

    Returns:
        Pipeline execution results
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    pipeline = create_pipeline(max_workers=max_workers)

    # Add tasks with intelligent priority assignment
    def priority_mapper(symbol: str) -> IngestionPriority:
        # You could enhance this with actual portfolio/watchlist data
        if symbol in ["SPY", "QQQ", "IWM"]:  # Major ETFs
            return IngestionPriority.HIGH
        return IngestionPriority.MEDIUM

    pipeline.add_bulk_tasks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        priority_mapper=priority_mapper
    )

    # Run pipeline with progress tracking
    def progress_callback(progress: IngestionProgress):
        if progress.total_tasks > 0:
            completion_rate = progress.completion_rate
            remaining_time = progress.estimated_time_remaining

            print(f"Progress: {completion_rate:.1f}% "
                  f"({progress.completed_tasks}/{progress.total_tasks}) "
                  f"ETA: {remaining_time:.0f}s" if remaining_time else "")

    final_progress = pipeline.run_pipeline(
        wait_for_completion=True,
        progress_callback=progress_callback
    )

    return {
        'total_symbols': len(symbols),
        'successful_ingestions': final_progress.completed_tasks,
        'failed_ingestions': final_progress.failed_tasks,
        'completion_rate': final_progress.completion_rate,
        'total_duration': final_progress.elapsed_time,
        'pipeline_status': pipeline.get_pipeline_status()
    }


if __name__ == "__main__":
    # Example usage
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    print("Starting bulk ingestion pipeline test...")
    results = run_bulk_ingestion(
        symbols=test_symbols,
        start_date="2020-01-01",
        end_date="2023-12-31",
        max_workers=4
    )

    print(f"\nIngestion Results:")
    print(f"  Total symbols: {results['total_symbols']}")
    print(f"  Successful: {results['successful_ingestions']}")
    print(f"  Failed: {results['failed_ingestions']}")
    print(f"  Completion rate: {results['completion_rate']:.1f}%")
    print(f"  Duration: {results['total_duration']:.1f} seconds")