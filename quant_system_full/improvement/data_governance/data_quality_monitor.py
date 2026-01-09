"""
Data Quality Monitoring System

This module provides comprehensive data quality monitoring for financial data,
including missing data detection, anomaly identification, and data integrity validation.

Key Features:
- Real-time data quality assessment
- Missing data detection and reporting
- Price jump and anomaly detection
- Volume consistency validation
- Survivorship bias detection
- Data completeness scoring
- Quality trend analysis
- Automated quality reporting
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of data anomalies."""
    MISSING_DATA = "missing_data"
    PRICE_JUMP = "price_jump"
    VOLUME_ANOMALY = "volume_anomaly"
    NEGATIVE_PRICE = "negative_price"
    ZERO_VOLUME = "zero_volume"
    OUTLIER_RETURN = "outlier_return"
    STALE_DATA = "stale_data"
    DUPLICATE_DATA = "duplicate_data"
    INCONSISTENT_OHLC = "inconsistent_ohlc"


@dataclass
class DataQualityIssue:
    """Individual data quality issue."""
    symbol: str
    date: str
    anomaly_type: AnomalyType
    severity: DataQualityLevel
    description: str
    value: Optional[float] = None
    expected_value: Optional[float] = None
    confidence: float = 1.0
    auto_fixable: bool = False
    fixed: bool = False
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


@dataclass
class DataQualityMetrics:
    """Data quality metrics for a symbol."""
    symbol: str
    assessment_date: str
    data_period_start: str
    data_period_end: str

    # Completeness metrics
    total_expected_days: int
    actual_data_days: int
    missing_days: int
    completeness_ratio: float

    # Quality scores
    overall_quality_score: float
    price_quality_score: float
    volume_quality_score: float
    consistency_score: float

    # Issue counts
    total_issues: int
    critical_issues: int
    auto_fixable_issues: int

    # Data characteristics
    price_volatility: float
    average_volume: float
    data_freshness_hours: float

    # Quality level
    quality_level: DataQualityLevel


@dataclass
class DataQualityConfig:
    """Configuration for data quality monitoring."""
    # Price jump thresholds
    price_jump_threshold: float = 0.20  # 20% single-day jump
    extreme_price_jump_threshold: float = 0.50  # 50% extreme jump

    # Volume thresholds
    volume_spike_threshold: float = 5.0  # 5x normal volume
    zero_volume_tolerance: int = 3  # Max consecutive zero volume days

    # Missing data thresholds
    missing_data_threshold: float = 0.05  # 5% missing data tolerance
    critical_missing_threshold: float = 0.20  # 20% critical threshold

    # Staleness thresholds
    max_staleness_hours: float = 48.0  # 48 hours for daily data
    critical_staleness_hours: float = 120.0  # 5 days critical

    # Quality scoring weights
    completeness_weight: float = 0.30
    consistency_weight: float = 0.25
    freshness_weight: float = 0.20
    accuracy_weight: float = 0.25

    # Outlier detection parameters
    outlier_std_threshold: float = 4.0  # Standard deviations for outliers
    lookback_days: int = 60  # Days to look back for statistics


class DataQualityMonitor:
    """
    Monitors and assesses data quality for financial data.
    """

    def __init__(self, config: Optional[DataQualityConfig] = None,
                 data_dir: str = "data_cache/data_quality"):
        """
        Initialize data quality monitor.

        Args:
            config: Quality monitoring configuration
            data_dir: Directory for storing quality data
        """
        self.config = config or DataQualityConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Database for storing quality data
        self.db_path = self.data_dir / "data_quality.db"

        # Quality cache
        self.quality_cache: Dict[str, DataQualityMetrics] = {}
        self.issues_cache: Dict[str, List[DataQualityIssue]] = {}

        # Initialize database
        self._init_database()

        logger.info(f"[data_quality] Monitor initialized with data dir: {self.data_dir}")

    def _init_database(self):
        """Initialize SQLite database for data quality tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Quality metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        assessment_date TEXT NOT NULL,
                        data_period_start TEXT NOT NULL,
                        data_period_end TEXT NOT NULL,
                        total_expected_days INTEGER,
                        actual_data_days INTEGER,
                        missing_days INTEGER,
                        completeness_ratio REAL,
                        overall_quality_score REAL,
                        price_quality_score REAL,
                        volume_quality_score REAL,
                        consistency_score REAL,
                        total_issues INTEGER,
                        critical_issues INTEGER,
                        auto_fixable_issues INTEGER,
                        price_volatility REAL,
                        average_volume REAL,
                        data_freshness_hours REAL,
                        quality_level TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(symbol, assessment_date)
                    )
                """)

                # Quality issues table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_issues (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        anomaly_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        description TEXT,
                        value REAL,
                        expected_value REAL,
                        confidence REAL DEFAULT 1.0,
                        auto_fixable INTEGER DEFAULT 0,
                        fixed INTEGER DEFAULT 0,
                        detected_at TEXT NOT NULL,
                        fixed_at TEXT
                    )
                """)

                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_symbol_date ON quality_metrics(symbol, assessment_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_issues_symbol_date ON quality_issues(symbol, date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_issues_type ON quality_issues(anomaly_type)")

                conn.commit()

            logger.info("[data_quality] Database initialized successfully")

        except Exception as e:
            logger.error(f"[data_quality] Database initialization failed: {e}")
            raise

    def assess_data_quality(self, symbol: str, data: pd.DataFrame,
                          assessment_date: Optional[str] = None) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment.

        Args:
            symbol: Stock symbol
            data: OHLCV data to assess
            assessment_date: Date of assessment

        Returns:
            Data quality metrics
        """
        try:
            if assessment_date is None:
                assessment_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"[data_quality] Assessing data quality for {symbol}")

            # Basic data validation
            if data.empty:
                return self._create_empty_metrics(symbol, assessment_date, "No data available")

            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Sort by date
            data = data.sort_index()

            # Calculate time period
            period_start = data.index[0].strftime('%Y-%m-%d')
            period_end = data.index[-1].strftime('%Y-%m-%d')

            # Detect and store issues
            issues = self._detect_all_anomalies(symbol, data)
            self._store_quality_issues(issues)

            # Calculate metrics
            metrics = self._calculate_quality_metrics(symbol, data, issues, assessment_date,
                                                    period_start, period_end)

            # Store metrics
            self._store_quality_metrics(metrics)

            # Cache results
            self.quality_cache[f"{symbol}_{assessment_date}"] = metrics
            self.issues_cache[f"{symbol}_{assessment_date}"] = issues

            logger.info(f"[data_quality] Assessment complete for {symbol}: "
                       f"{metrics.quality_level.value} quality ({metrics.overall_quality_score:.2f})")

            return metrics

        except Exception as e:
            logger.error(f"[data_quality] Quality assessment failed for {symbol}: {e}")
            return self._create_empty_metrics(symbol, assessment_date or datetime.now().strftime('%Y-%m-%d'),
                                            f"Assessment error: {str(e)}")

    def _detect_all_anomalies(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect all types of data quality anomalies."""
        issues = []

        # Missing data detection
        issues.extend(self._detect_missing_data(symbol, data))

        # Price anomalies
        issues.extend(self._detect_price_anomalies(symbol, data))

        # Volume anomalies
        issues.extend(self._detect_volume_anomalies(symbol, data))

        # OHLC consistency
        issues.extend(self._detect_ohlc_inconsistencies(symbol, data))

        # Outlier detection
        issues.extend(self._detect_outliers(symbol, data))

        # Duplicate detection
        issues.extend(self._detect_duplicates(symbol, data))

        # Staleness detection
        issues.extend(self._detect_stale_data(symbol, data))

        return issues

    def _detect_missing_data(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect missing data points."""
        issues = []

        try:
            # Create complete date range
            start_date = data.index[0]
            end_date = data.index[-1]
            complete_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Find missing business days
            business_days = pd.bdate_range(start=start_date, end=end_date)
            missing_dates = business_days.difference(data.index)

            for missing_date in missing_dates:
                issue = DataQualityIssue(
                    symbol=symbol,
                    date=missing_date.strftime('%Y-%m-%d'),
                    anomaly_type=AnomalyType.MISSING_DATA,
                    severity=DataQualityLevel.FAIR,
                    description=f"Missing data for business day",
                    auto_fixable=True
                )
                issues.append(issue)

            # Check for consecutive missing data
            if len(missing_dates) > 5:
                # Mark as critical if too much missing data
                for issue in issues[-5:]:
                    issue.severity = DataQualityLevel.CRITICAL

        except Exception as e:
            logger.error(f"[data_quality] Missing data detection failed: {e}")

        return issues

    def _detect_price_anomalies(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect price-related anomalies."""
        issues = []

        try:
            if 'close' not in data.columns:
                return issues

            prices = data['close']

            # Negative prices
            negative_mask = prices <= 0
            for date in data.index[negative_mask]:
                issue = DataQualityIssue(
                    symbol=symbol,
                    date=date.strftime('%Y-%m-%d'),
                    anomaly_type=AnomalyType.NEGATIVE_PRICE,
                    severity=DataQualityLevel.CRITICAL,
                    description=f"Negative or zero price: {prices.loc[date]:.4f}",
                    value=prices.loc[date],
                    auto_fixable=False
                )
                issues.append(issue)

            # Price jumps
            returns = prices.pct_change()
            jump_mask = returns.abs() > self.config.price_jump_threshold

            for date in data.index[jump_mask]:
                if pd.notna(returns.loc[date]):
                    severity = (DataQualityLevel.CRITICAL
                              if abs(returns.loc[date]) > self.config.extreme_price_jump_threshold
                              else DataQualityLevel.POOR)

                    issue = DataQualityIssue(
                        symbol=symbol,
                        date=date.strftime('%Y-%m-%d'),
                        anomaly_type=AnomalyType.PRICE_JUMP,
                        severity=severity,
                        description=f"Large price jump: {returns.loc[date]:.1%}",
                        value=returns.loc[date],
                        auto_fixable=False
                    )
                    issues.append(issue)

        except Exception as e:
            logger.error(f"[data_quality] Price anomaly detection failed: {e}")

        return issues

    def _detect_volume_anomalies(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect volume-related anomalies."""
        issues = []

        try:
            if 'volume' not in data.columns:
                return issues

            volume = data['volume']

            # Zero volume
            zero_mask = volume == 0
            zero_consecutive = 0

            for date in data.index:
                if zero_mask.loc[date]:
                    zero_consecutive += 1
                    if zero_consecutive > self.config.zero_volume_tolerance:
                        issue = DataQualityIssue(
                            symbol=symbol,
                            date=date.strftime('%Y-%m-%d'),
                            anomaly_type=AnomalyType.ZERO_VOLUME,
                            severity=DataQualityLevel.POOR,
                            description=f"Zero volume for {zero_consecutive} consecutive days",
                            value=0,
                            auto_fixable=True
                        )
                        issues.append(issue)
                else:
                    zero_consecutive = 0

            # Volume spikes
            volume_ma = volume.rolling(20).mean()
            spike_mask = volume > (volume_ma * self.config.volume_spike_threshold)

            for date in data.index[spike_mask]:
                if pd.notna(volume_ma.loc[date]) and volume_ma.loc[date] > 0:
                    spike_ratio = volume.loc[date] / volume_ma.loc[date]
                    issue = DataQualityIssue(
                        symbol=symbol,
                        date=date.strftime('%Y-%m-%d'),
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        severity=DataQualityLevel.FAIR,
                        description=f"Volume spike: {spike_ratio:.1f}x normal volume",
                        value=spike_ratio,
                        auto_fixable=False
                    )
                    issues.append(issue)

        except Exception as e:
            logger.error(f"[data_quality] Volume anomaly detection failed: {e}")

        return issues

    def _detect_ohlc_inconsistencies(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect OHLC data inconsistencies."""
        issues = []

        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                return issues

            for date in data.index:
                row = data.loc[date]
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                # Check OHLC relationships
                if not (low_price <= open_price <= high_price and
                       low_price <= close_price <= high_price):
                    issue = DataQualityIssue(
                        symbol=symbol,
                        date=date.strftime('%Y-%m-%d'),
                        anomaly_type=AnomalyType.INCONSISTENT_OHLC,
                        severity=DataQualityLevel.CRITICAL,
                        description=f"OHLC inconsistency: O:{open_price:.2f} H:{high_price:.2f} L:{low_price:.2f} C:{close_price:.2f}",
                        auto_fixable=True
                    )
                    issues.append(issue)

        except Exception as e:
            logger.error(f"[data_quality] OHLC consistency check failed: {e}")

        return issues

    def _detect_outliers(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect statistical outliers in returns."""
        issues = []

        try:
            if 'close' not in data.columns:
                return issues

            returns = data['close'].pct_change()
            returns_clean = returns.dropna()

            if len(returns_clean) < 10:
                return issues

            # Calculate rolling statistics
            lookback = min(self.config.lookback_days, len(returns_clean))
            returns_std = returns_clean.rolling(lookback).std()
            returns_mean = returns_clean.rolling(lookback).mean()

            # Detect outliers
            for date in data.index[1:]:  # Skip first date (no return)
                if date in returns.index and pd.notna(returns.loc[date]):
                    if pd.notna(returns_std.loc[date]) and returns_std.loc[date] > 0:
                        z_score = abs((returns.loc[date] - returns_mean.loc[date]) / returns_std.loc[date])

                        if z_score > self.config.outlier_std_threshold:
                            issue = DataQualityIssue(
                                symbol=symbol,
                                date=date.strftime('%Y-%m-%d'),
                                anomaly_type=AnomalyType.OUTLIER_RETURN,
                                severity=DataQualityLevel.FAIR,
                                description=f"Outlier return: {returns.loc[date]:.1%} (z-score: {z_score:.1f})",
                                value=returns.loc[date],
                                confidence=min(1.0, z_score / 10),
                                auto_fixable=False
                            )
                            issues.append(issue)

        except Exception as e:
            logger.error(f"[data_quality] Outlier detection failed: {e}")

        return issues

    def _detect_duplicates(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect duplicate data entries."""
        issues = []

        try:
            # Check for duplicate dates
            duplicate_dates = data.index.duplicated()

            for i, is_duplicate in enumerate(duplicate_dates):
                if is_duplicate:
                    date = data.index[i]
                    issue = DataQualityIssue(
                        symbol=symbol,
                        date=date.strftime('%Y-%m-%d'),
                        anomaly_type=AnomalyType.DUPLICATE_DATA,
                        severity=DataQualityLevel.POOR,
                        description="Duplicate data entry for date",
                        auto_fixable=True
                    )
                    issues.append(issue)

        except Exception as e:
            logger.error(f"[data_quality] Duplicate detection failed: {e}")

        return issues

    def _detect_stale_data(self, symbol: str, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect stale data based on freshness."""
        issues = []

        try:
            if data.empty:
                return issues

            latest_data_date = data.index[-1]
            now = datetime.now()

            # Calculate staleness in hours
            staleness_hours = (now - latest_data_date).total_seconds() / 3600

            if staleness_hours > self.config.max_staleness_hours:
                severity = (DataQualityLevel.CRITICAL
                          if staleness_hours > self.config.critical_staleness_hours
                          else DataQualityLevel.POOR)

                issue = DataQualityIssue(
                    symbol=symbol,
                    date=latest_data_date.strftime('%Y-%m-%d'),
                    anomaly_type=AnomalyType.STALE_DATA,
                    severity=severity,
                    description=f"Data is {staleness_hours:.1f} hours old",
                    value=staleness_hours,
                    auto_fixable=False
                )
                issues.append(issue)

        except Exception as e:
            logger.error(f"[data_quality] Stale data detection failed: {e}")

        return issues

    def _calculate_quality_metrics(self, symbol: str, data: pd.DataFrame,
                                 issues: List[DataQualityIssue],
                                 assessment_date: str, period_start: str,
                                 period_end: str) -> DataQualityMetrics:
        """Calculate comprehensive quality metrics."""
        try:
            # Basic statistics
            start_date = pd.to_datetime(period_start)
            end_date = pd.to_datetime(period_end)
            total_expected_days = len(pd.bdate_range(start=start_date, end=end_date))
            actual_data_days = len(data)
            missing_days = total_expected_days - actual_data_days
            completeness_ratio = actual_data_days / total_expected_days if total_expected_days > 0 else 0

            # Issue statistics
            total_issues = len(issues)
            critical_issues = len([i for i in issues if i.severity == DataQualityLevel.CRITICAL])
            auto_fixable_issues = len([i for i in issues if i.auto_fixable])

            # Calculate component scores
            completeness_score = completeness_ratio
            consistency_score = self._calculate_consistency_score(data, issues)
            freshness_score = self._calculate_freshness_score(data)
            accuracy_score = self._calculate_accuracy_score(issues, total_expected_days)

            # Overall quality score
            overall_quality_score = (
                completeness_score * self.config.completeness_weight +
                consistency_score * self.config.consistency_weight +
                freshness_score * self.config.freshness_weight +
                accuracy_score * self.config.accuracy_weight
            )

            # Price and volume specific scores
            price_quality_score = self._calculate_price_quality_score(data, issues)
            volume_quality_score = self._calculate_volume_quality_score(data, issues)

            # Data characteristics
            price_volatility = data['close'].pct_change().std() if 'close' in data.columns else 0
            average_volume = data['volume'].mean() if 'volume' in data.columns else 0
            data_freshness_hours = self._calculate_data_freshness_hours(data)

            # Determine quality level
            quality_level = self._determine_quality_level(overall_quality_score, critical_issues)

            return DataQualityMetrics(
                symbol=symbol,
                assessment_date=assessment_date,
                data_period_start=period_start,
                data_period_end=period_end,
                total_expected_days=total_expected_days,
                actual_data_days=actual_data_days,
                missing_days=missing_days,
                completeness_ratio=completeness_ratio,
                overall_quality_score=overall_quality_score,
                price_quality_score=price_quality_score,
                volume_quality_score=volume_quality_score,
                consistency_score=consistency_score,
                total_issues=total_issues,
                critical_issues=critical_issues,
                auto_fixable_issues=auto_fixable_issues,
                price_volatility=price_volatility,
                average_volume=average_volume,
                data_freshness_hours=data_freshness_hours,
                quality_level=quality_level
            )

        except Exception as e:
            logger.error(f"[data_quality] Quality metrics calculation failed: {e}")
            return self._create_empty_metrics(symbol, assessment_date, "Calculation error")

    def _calculate_consistency_score(self, data: pd.DataFrame, issues: List[DataQualityIssue]) -> float:
        """Calculate data consistency score."""
        try:
            consistency_issues = [i for i in issues if i.anomaly_type in [
                AnomalyType.INCONSISTENT_OHLC, AnomalyType.NEGATIVE_PRICE
            ]]

            if len(data) == 0:
                return 0.0

            consistency_ratio = 1.0 - (len(consistency_issues) / len(data))
            return max(0.0, consistency_ratio)

        except Exception:
            return 0.0

    def _calculate_freshness_score(self, data: pd.DataFrame) -> float:
        """Calculate data freshness score."""
        try:
            if data.empty:
                return 0.0

            latest_date = data.index[-1]
            staleness_hours = (datetime.now() - latest_date).total_seconds() / 3600

            if staleness_hours <= self.config.max_staleness_hours:
                return 1.0
            elif staleness_hours <= self.config.critical_staleness_hours:
                return 1.0 - ((staleness_hours - self.config.max_staleness_hours) /
                            (self.config.critical_staleness_hours - self.config.max_staleness_hours))
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_accuracy_score(self, issues: List[DataQualityIssue], total_days: int) -> float:
        """Calculate data accuracy score."""
        try:
            if total_days == 0:
                return 0.0

            accuracy_issues = [i for i in issues if i.anomaly_type in [
                AnomalyType.PRICE_JUMP, AnomalyType.VOLUME_ANOMALY, AnomalyType.OUTLIER_RETURN
            ]]

            accuracy_ratio = 1.0 - (len(accuracy_issues) / total_days)
            return max(0.0, accuracy_ratio)

        except Exception:
            return 0.0

    def _calculate_price_quality_score(self, data: pd.DataFrame, issues: List[DataQualityIssue]) -> float:
        """Calculate price-specific quality score."""
        try:
            if 'close' not in data.columns or len(data) == 0:
                return 0.0

            price_issues = [i for i in issues if i.anomaly_type in [
                AnomalyType.PRICE_JUMP, AnomalyType.NEGATIVE_PRICE, AnomalyType.INCONSISTENT_OHLC
            ]]

            price_quality = 1.0 - (len(price_issues) / len(data))
            return max(0.0, price_quality)

        except Exception:
            return 0.0

    def _calculate_volume_quality_score(self, data: pd.DataFrame, issues: List[DataQualityIssue]) -> float:
        """Calculate volume-specific quality score."""
        try:
            if 'volume' not in data.columns or len(data) == 0:
                return 0.0

            volume_issues = [i for i in issues if i.anomaly_type in [
                AnomalyType.VOLUME_ANOMALY, AnomalyType.ZERO_VOLUME
            ]]

            volume_quality = 1.0 - (len(volume_issues) / len(data))
            return max(0.0, volume_quality)

        except Exception:
            return 0.0

    def _calculate_data_freshness_hours(self, data: pd.DataFrame) -> float:
        """Calculate data freshness in hours."""
        try:
            if data.empty:
                return float('inf')

            latest_date = data.index[-1]
            return (datetime.now() - latest_date).total_seconds() / 3600

        except Exception:
            return float('inf')

    def _determine_quality_level(self, overall_score: float, critical_issues: int) -> DataQualityLevel:
        """Determine overall quality level."""
        if critical_issues > 0:
            return DataQualityLevel.CRITICAL
        elif overall_score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif overall_score >= 0.8:
            return DataQualityLevel.GOOD
        elif overall_score >= 0.6:
            return DataQualityLevel.FAIR
        else:
            return DataQualityLevel.POOR

    def _create_empty_metrics(self, symbol: str, assessment_date: str, reason: str) -> DataQualityMetrics:
        """Create empty metrics for error cases."""
        return DataQualityMetrics(
            symbol=symbol,
            assessment_date=assessment_date,
            data_period_start=assessment_date,
            data_period_end=assessment_date,
            total_expected_days=0,
            actual_data_days=0,
            missing_days=0,
            completeness_ratio=0.0,
            overall_quality_score=0.0,
            price_quality_score=0.0,
            volume_quality_score=0.0,
            consistency_score=0.0,
            total_issues=1,
            critical_issues=1,
            auto_fixable_issues=0,
            price_volatility=0.0,
            average_volume=0.0,
            data_freshness_hours=float('inf'),
            quality_level=DataQualityLevel.CRITICAL
        )

    def _store_quality_metrics(self, metrics: DataQualityMetrics):
        """Store quality metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO quality_metrics
                    (symbol, assessment_date, data_period_start, data_period_end,
                     total_expected_days, actual_data_days, missing_days, completeness_ratio,
                     overall_quality_score, price_quality_score, volume_quality_score, consistency_score,
                     total_issues, critical_issues, auto_fixable_issues,
                     price_volatility, average_volume, data_freshness_hours, quality_level, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.symbol, metrics.assessment_date, metrics.data_period_start, metrics.data_period_end,
                    metrics.total_expected_days, metrics.actual_data_days, metrics.missing_days, metrics.completeness_ratio,
                    metrics.overall_quality_score, metrics.price_quality_score, metrics.volume_quality_score, metrics.consistency_score,
                    metrics.total_issues, metrics.critical_issues, metrics.auto_fixable_issues,
                    metrics.price_volatility, metrics.average_volume, metrics.data_freshness_hours,
                    metrics.quality_level.value, datetime.now().isoformat()
                ))

        except Exception as e:
            logger.error(f"[data_quality] Failed to store quality metrics: {e}")

    def _store_quality_issues(self, issues: List[DataQualityIssue]):
        """Store quality issues in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for issue in issues:
                    conn.execute("""
                        INSERT OR REPLACE INTO quality_issues
                        (symbol, date, anomaly_type, severity, description, value, expected_value,
                         confidence, auto_fixable, fixed, detected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        issue.symbol, issue.date, issue.anomaly_type.value, issue.severity.value,
                        issue.description, issue.value, issue.expected_value,
                        issue.confidence, int(issue.auto_fixable), int(issue.fixed), issue.detected_at
                    ))

        except Exception as e:
            logger.error(f"[data_quality] Failed to store quality issues: {e}")

    def get_quality_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get summary of data quality across all monitored symbols."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent assessments
                cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as total_assessments,
                           AVG(overall_quality_score) as avg_quality_score,
                           COUNT(CASE WHEN quality_level = 'excellent' THEN 1 END) as excellent_count,
                           COUNT(CASE WHEN quality_level = 'good' THEN 1 END) as good_count,
                           COUNT(CASE WHEN quality_level = 'fair' THEN 1 END) as fair_count,
                           COUNT(CASE WHEN quality_level = 'poor' THEN 1 END) as poor_count,
                           COUNT(CASE WHEN quality_level = 'critical' THEN 1 END) as critical_count,
                           COUNT(DISTINCT symbol) as unique_symbols
                    FROM quality_metrics
                    WHERE assessment_date >= ?
                """, (cutoff_date,))

                metrics_summary = cursor.fetchone()

                # Get issue summary
                cursor.execute("""
                    SELECT anomaly_type, COUNT(*) as count
                    FROM quality_issues
                    WHERE detected_at >= ?
                    GROUP BY anomaly_type
                    ORDER BY count DESC
                """, (cutoff_date,))

                issues_by_type = dict(cursor.fetchall())

                return {
                    'assessment_period_days': days_back,
                    'total_assessments': metrics_summary[0] if metrics_summary else 0,
                    'average_quality_score': metrics_summary[1] if metrics_summary and metrics_summary[1] else 0,
                    'quality_distribution': {
                        'excellent': metrics_summary[2] if metrics_summary else 0,
                        'good': metrics_summary[3] if metrics_summary else 0,
                        'fair': metrics_summary[4] if metrics_summary else 0,
                        'poor': metrics_summary[5] if metrics_summary else 0,
                        'critical': metrics_summary[6] if metrics_summary else 0
                    },
                    'unique_symbols_assessed': metrics_summary[7] if metrics_summary else 0,
                    'issues_by_type': issues_by_type,
                    'cache_size': len(self.quality_cache),
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"[data_quality] Failed to get quality summary: {e}")
            return {'error': str(e)}


def create_data_quality_monitor(custom_config: Optional[Dict] = None,
                              data_dir: Optional[str] = None) -> DataQualityMonitor:
    """
    Create and configure a data quality monitor.

    Args:
        custom_config: Custom configuration parameters
        data_dir: Custom data directory

    Returns:
        Configured DataQualityMonitor instance
    """
    config = DataQualityConfig()

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return DataQualityMonitor(config, data_dir or "data_cache/data_quality")


if __name__ == "__main__":
    # Test data quality monitoring
    print("=== Data Quality Monitor Test ===")

    # Create monitor
    monitor = create_data_quality_monitor()

    # Create sample data with quality issues
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)) * 2,
        'high': 105 + np.random.randn(len(dates)) * 2,
        'low': 95 + np.random.randn(len(dates)) * 2,
        'close': 100 + np.random.randn(len(dates)) * 2,
        'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
    }, index=dates)

    # Introduce some quality issues
    sample_data.loc[dates[10], 'close'] = -5.0  # Negative price
    sample_data.loc[dates[20], 'close'] = sample_data.loc[dates[19], 'close'] * 1.5  # Price jump
    sample_data.loc[dates[30:35], 'volume'] = 0  # Zero volume

    # Remove some data to simulate missing data
    sample_data = sample_data.drop(dates[40:45])

    # Assess quality
    metrics = monitor.assess_data_quality("TEST", sample_data)

    print(f"Quality Assessment for TEST:")
    print(f"  Overall Quality: {metrics.quality_level.value} ({metrics.overall_quality_score:.2f})")
    print(f"  Completeness: {metrics.completeness_ratio:.1%}")
    print(f"  Total Issues: {metrics.total_issues}")
    print(f"  Critical Issues: {metrics.critical_issues}")
    print(f"  Data Freshness: {metrics.data_freshness_hours:.1f} hours")

    # Get summary
    summary = monitor.get_quality_summary()
    print(f"\nQuality Summary:")
    print(f"  Assessments: {summary['total_assessments']}")
    print(f"  Average Score: {summary['average_quality_score']:.2f}")
    print(f"  Quality Distribution: {summary['quality_distribution']}")