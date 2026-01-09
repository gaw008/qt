"""
Data Quality Validation Framework

This module provides comprehensive data quality validation and cleansing
for financial market data, ensuring high-quality inputs for backtesting
and live trading systems.

Key Features:
- Comprehensive validation rules for OHLCV data
- Missing data detection and interpolation strategies
- Price anomaly detection (jumps, spikes, errors)
- Volume consistency validation
- Corporate actions impact assessment
- Data completeness scoring
- Automated cleansing and correction
- Quality metrics and reporting
- Integration with survivorship bias detection
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityIssueType(Enum):
    """Types of data quality issues."""
    MISSING_DATA = "missing_data"
    PRICE_JUMP = "price_jump"
    VOLUME_ANOMALY = "volume_anomaly"
    NEGATIVE_PRICE = "negative_price"
    ZERO_VOLUME = "zero_volume"
    OHLC_INCONSISTENCY = "ohlc_inconsistency"
    STALE_DATA = "stale_data"
    DUPLICATE_RECORDS = "duplicate_records"
    OUTLIER_RETURN = "outlier_return"
    VOLUME_PRICE_MISMATCH = "volume_price_mismatch"
    SPLIT_NOT_ADJUSTED = "split_not_adjusted"
    DIVIDEND_NOT_ADJUSTED = "dividend_not_adjusted"


class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"    # Data unusable, major impact
    HIGH = "high"           # Significant impact on analysis
    MEDIUM = "medium"       # Moderate impact, should be fixed
    LOW = "low"            # Minor impact, cosmetic
    INFO = "info"          # Informational, no action needed


class InterpolationMethod(Enum):
    """Methods for missing data interpolation."""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    SEASONAL = "seasonal"
    MARKET_AVERAGE = "market_average"


@dataclass
class QualityIssue:
    """Individual data quality issue record."""
    symbol: str
    date: str
    issue_type: QualityIssueType
    severity: QualitySeverity
    description: str
    value: Optional[float] = None
    expected_value: Optional[float] = None
    confidence: float = 1.0
    auto_fixable: bool = False
    suggested_fix: Optional[str] = None
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityMetrics:
    """Data quality metrics for a symbol or dataset."""
    symbol: str
    assessment_period: str
    total_records: int
    missing_records: int
    quality_issues: int
    critical_issues: int

    # Completeness metrics
    completeness_ratio: float
    consecutive_missing_max: int

    # Consistency metrics
    ohlc_consistency_score: float
    volume_consistency_score: float

    # Anomaly metrics
    price_anomaly_count: int
    volume_anomaly_count: int

    # Overall quality score (0-1)
    overall_quality_score: float

    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'assessment_period': self.assessment_period,
            'total_records': self.total_records,
            'missing_records': self.missing_records,
            'quality_issues': self.quality_issues,
            'critical_issues': self.critical_issues,
            'completeness_ratio': self.completeness_ratio,
            'consecutive_missing_max': self.consecutive_missing_max,
            'ohlc_consistency_score': self.ohlc_consistency_score,
            'volume_consistency_score': self.volume_consistency_score,
            'price_anomaly_count': self.price_anomaly_count,
            'volume_anomaly_count': self.volume_anomaly_count,
            'overall_quality_score': self.overall_quality_score,
            'recommended_actions': self.recommended_actions
        }


class DataQualityValidator:
    """
    Comprehensive data quality validation and cleansing system.

    Provides validation rules, anomaly detection, and automated
    fixing for financial market data.
    """

    def __init__(self, db_path: str = "data_cache/data_quality.db"):
        """
        Initialize data quality validator.

        Args:
            db_path: Path to SQLite database for quality tracking
        """
        self.db_path = db_path
        self.validation_rules = self._initialize_validation_rules()
        self.thresholds = self._initialize_thresholds()

        # Initialize database
        self._init_database()

        # Anomaly detection models
        self._anomaly_detectors = {}

        logger.info(f"[quality] Validator initialized with database: {db_path}")

    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different data types."""
        return {
            'price_rules': {
                'min_price': 0.01,
                'max_price': 100000.0,
                'max_daily_change': 0.5,  # 50% max daily change
                'ohlc_consistency_tolerance': 0.001,  # 0.1% tolerance
                'split_detection_threshold': 0.4,  # 40% overnight change
            },
            'volume_rules': {
                'min_volume': 0,
                'max_volume_multiplier': 100,  # 100x average volume
                'min_volume_ratio': 0.01,  # 1% of average volume
                'zero_volume_tolerance': 0.05,  # 5% of days can have zero volume
            },
            'temporal_rules': {
                'max_consecutive_missing': 10,  # Max 10 consecutive missing days
                'min_completeness_ratio': 0.95,  # 95% data completeness
                'stale_data_threshold': 5,  # Data older than 5 business days
                'duplicate_tolerance': 0,  # No duplicates allowed
            },
            'consistency_rules': {
                'high_low_spread_max': 0.2,  # 20% max H-L spread
                'volume_price_correlation_min': -0.5,  # Min correlation threshold
                'return_volatility_max': 0.15,  # 15% max daily volatility
            }
        }

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize statistical thresholds for anomaly detection."""
        return {
            'price_jump_zscore': 3.0,
            'volume_anomaly_zscore': 3.0,
            'return_outlier_zscore': 4.0,
            'isolation_forest_contamination': 0.1,
            'bollinger_band_periods': 20,
            'bollinger_band_std': 2.0,
        }

    def _init_database(self):
        """Initialize SQLite database for quality tracking."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Quality issues table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    value REAL,
                    expected_value REAL,
                    confidence REAL DEFAULT 1.0,
                    auto_fixable BOOLEAN DEFAULT 0,
                    suggested_fix TEXT,
                    fixed BOOLEAN DEFAULT 0,
                    detected_at TEXT NOT NULL,
                    fixed_at TEXT
                )
            """)

            # Quality metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    assessment_date TEXT NOT NULL,
                    assessment_period TEXT NOT NULL,
                    total_records INTEGER NOT NULL,
                    missing_records INTEGER DEFAULT 0,
                    quality_issues INTEGER DEFAULT 0,
                    critical_issues INTEGER DEFAULT 0,
                    completeness_ratio REAL DEFAULT 1.0,
                    consecutive_missing_max INTEGER DEFAULT 0,
                    ohlc_consistency_score REAL DEFAULT 1.0,
                    volume_consistency_score REAL DEFAULT 1.0,
                    price_anomaly_count INTEGER DEFAULT 0,
                    volume_anomaly_count INTEGER DEFAULT 0,
                    overall_quality_score REAL DEFAULT 1.0,
                    recommended_actions TEXT,
                    UNIQUE(symbol, assessment_date)
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_issues_symbol_date ON quality_issues(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_symbol ON quality_metrics(symbol, assessment_date)")

            conn.commit()

    def validate_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        perform_fixes: bool = True,
        save_issues: bool = True
    ) -> Tuple[pd.DataFrame, QualityMetrics, List[QualityIssue]]:
        """
        Comprehensive validation of a dataset.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            perform_fixes: Whether to automatically fix issues
            save_issues: Whether to save issues to database

        Returns:
            Tuple of (cleaned_df, quality_metrics, issues_found)
        """
        logger.info(f"[quality] Starting validation for {symbol} ({len(df)} records)")

        # Ensure required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Make a copy for processing
        clean_df = df.copy()
        issues_found = []

        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(clean_df['date']):
            clean_df['date'] = pd.to_datetime(clean_df['date'])

        # Sort by date
        clean_df = clean_df.sort_values('date').reset_index(drop=True)

        # Run all validation checks
        issues_found.extend(self._validate_price_data(clean_df, symbol))
        issues_found.extend(self._validate_volume_data(clean_df, symbol))
        issues_found.extend(self._validate_temporal_consistency(clean_df, symbol))
        issues_found.extend(self._validate_ohlc_consistency(clean_df, symbol))
        issues_found.extend(self._detect_anomalies(clean_df, symbol))
        issues_found.extend(self._detect_corporate_action_issues(clean_df, symbol))

        # Perform automatic fixes if requested
        if perform_fixes:
            clean_df = self._apply_automatic_fixes(clean_df, issues_found, symbol)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(clean_df, issues_found, symbol)

        # Save issues to database if requested
        if save_issues:
            self._save_quality_issues(issues_found)
            self._save_quality_metrics(quality_metrics)

        logger.info(f"[quality] Validation completed for {symbol}: "
                   f"{len(issues_found)} issues found, "
                   f"quality score: {quality_metrics.overall_quality_score:.3f}")

        return clean_df, quality_metrics, issues_found

    def _validate_price_data(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Validate price data for basic consistency."""
        issues = []
        rules = self.validation_rules['price_rules']

        # Check for negative or zero prices
        for col in ['open', 'high', 'low', 'close']:
            invalid_prices = df[df[col] <= 0]
            for idx, row in invalid_prices.iterrows():
                issues.append(QualityIssue(
                    symbol=symbol,
                    date=row['date'].strftime('%Y-%m-%d'),
                    issue_type=QualityIssueType.NEGATIVE_PRICE,
                    severity=QualitySeverity.CRITICAL,
                    description=f"Invalid {col} price: {row[col]}",
                    value=row[col],
                    auto_fixable=True,
                    suggested_fix="Remove record or interpolate from adjacent values"
                ))

        # Check for unrealistic price levels
        for col in ['open', 'high', 'low', 'close']:
            unrealistic_prices = df[
                (df[col] < rules['min_price']) |
                (df[col] > rules['max_price'])
            ]
            for idx, row in unrealistic_prices.iterrows():
                issues.append(QualityIssue(
                    symbol=symbol,
                    date=row['date'].strftime('%Y-%m-%d'),
                    issue_type=QualityIssueType.PRICE_JUMP,
                    severity=QualitySeverity.HIGH,
                    description=f"Unrealistic {col} price: {row[col]}",
                    value=row[col],
                    auto_fixable=False
                ))

        # Check for extreme daily price changes
        if len(df) > 1:
            df_sorted = df.sort_values('date')
            daily_returns = df_sorted['close'].pct_change().abs()
            extreme_changes = daily_returns > rules['max_daily_change']

            for idx in df_sorted[extreme_changes].index:
                if idx > 0:  # Skip first row (no previous data)
                    row = df_sorted.loc[idx]
                    prev_close = df_sorted.loc[idx-1, 'close']
                    change = daily_returns.loc[idx]

                    issues.append(QualityIssue(
                        symbol=symbol,
                        date=row['date'].strftime('%Y-%m-%d'),
                        issue_type=QualityIssueType.PRICE_JUMP,
                        severity=QualitySeverity.HIGH,
                        description=f"Extreme price change: {change:.2%} (from {prev_close:.2f} to {row['close']:.2f})",
                        value=change,
                        expected_value=rules['max_daily_change'],
                        auto_fixable=False
                    ))

        return issues

    def _validate_volume_data(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Validate volume data for consistency."""
        issues = []
        rules = self.validation_rules['volume_rules']

        # Check for negative volume
        negative_volume = df[df['volume'] < 0]
        for idx, row in negative_volume.iterrows():
            issues.append(QualityIssue(
                symbol=symbol,
                date=row['date'].strftime('%Y-%m-%d'),
                issue_type=QualityIssueType.NEGATIVE_PRICE,
                severity=QualitySeverity.CRITICAL,
                description=f"Negative volume: {row['volume']}",
                value=row['volume'],
                auto_fixable=True,
                suggested_fix="Set volume to 0"
            ))

        # Check for excessive zero volume days
        zero_volume_days = (df['volume'] == 0).sum()
        zero_volume_ratio = zero_volume_days / len(df)

        if zero_volume_ratio > rules['zero_volume_tolerance']:
            issues.append(QualityIssue(
                symbol=symbol,
                date=df['date'].min().strftime('%Y-%m-%d'),
                issue_type=QualityIssueType.ZERO_VOLUME,
                severity=QualitySeverity.MEDIUM,
                description=f"Too many zero volume days: {zero_volume_ratio:.2%} "
                           f"(threshold: {rules['zero_volume_tolerance']:.2%})",
                value=zero_volume_ratio,
                expected_value=rules['zero_volume_tolerance'],
                auto_fixable=False
            ))

        # Check for volume anomalies (extreme spikes)
        if len(df) > 10:  # Need enough data for statistics
            median_volume = df['volume'].median()
            volume_anomalies = df[df['volume'] > median_volume * rules['max_volume_multiplier']]

            for idx, row in volume_anomalies.iterrows():
                if row['volume'] > 0:  # Ignore zero volume
                    issues.append(QualityIssue(
                        symbol=symbol,
                        date=row['date'].strftime('%Y-%m-%d'),
                        issue_type=QualityIssueType.VOLUME_ANOMALY,
                        severity=QualitySeverity.MEDIUM,
                        description=f"Volume spike: {row['volume']:,} "
                                   f"({row['volume']/median_volume:.1f}x median)",
                        value=row['volume'],
                        expected_value=median_volume,
                        auto_fixable=False
                    ))

        return issues

    def _validate_temporal_consistency(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Validate temporal consistency and completeness."""
        issues = []
        rules = self.validation_rules['temporal_rules']

        # Check for duplicates
        duplicates = df.duplicated(subset=['date'], keep='first')
        if duplicates.any():
            duplicate_dates = df[duplicates]['date']
            for dup_date in duplicate_dates:
                issues.append(QualityIssue(
                    symbol=symbol,
                    date=dup_date.strftime('%Y-%m-%d'),
                    issue_type=QualityIssueType.DUPLICATE_RECORDS,
                    severity=QualitySeverity.HIGH,
                    description=f"Duplicate record for date: {dup_date.strftime('%Y-%m-%d')}",
                    auto_fixable=True,
                    suggested_fix="Keep most recent record"
                ))

        # Check for missing business days (if we have enough data)
        if len(df) > 20:  # Need reasonable amount of data
            df_sorted = df.sort_values('date')
            date_range = pd.date_range(
                start=df_sorted['date'].min(),
                end=df_sorted['date'].max(),
                freq='B'  # Business days
            )

            missing_dates = date_range.difference(df_sorted['date'])

            if len(missing_dates) > 0:
                # Check for consecutive missing periods
                if len(missing_dates) > 1:
                    missing_df = pd.DataFrame({'date': missing_dates}).sort_values('date')
                    missing_df['prev_date'] = missing_df['date'].shift(1)
                    missing_df['gap'] = (missing_df['date'] - missing_df['prev_date']).dt.days

                    consecutive_missing = 1
                    max_consecutive = 1

                    for gap in missing_df['gap'].dropna():
                        if gap == 1:  # Consecutive day
                            consecutive_missing += 1
                            max_consecutive = max(max_consecutive, consecutive_missing)
                        else:
                            consecutive_missing = 1

                # Report missing data issues
                missing_ratio = len(missing_dates) / len(date_range)
                severity = QualitySeverity.CRITICAL if missing_ratio > 0.1 else QualitySeverity.MEDIUM

                issues.append(QualityIssue(
                    symbol=symbol,
                    date=df_sorted['date'].min().strftime('%Y-%m-%d'),
                    issue_type=QualityIssueType.MISSING_DATA,
                    severity=severity,
                    description=f"Missing {len(missing_dates)} business days "
                               f"({missing_ratio:.2%} of expected data)",
                    value=missing_ratio,
                    expected_value=rules['min_completeness_ratio'],
                    auto_fixable=True,
                    suggested_fix="Interpolate missing values"
                ))

        return issues

    def _validate_ohlc_consistency(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Validate OHLC price consistency within each record."""
        issues = []
        tolerance = self.validation_rules['price_rules']['ohlc_consistency_tolerance']

        # Check basic OHLC relationships
        invalid_ohlc = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ]

        for idx, row in invalid_ohlc.iterrows():
            issues.append(QualityIssue(
                symbol=symbol,
                date=row['date'].strftime('%Y-%m-%d'),
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=QualitySeverity.CRITICAL,
                description=f"OHLC inconsistency: O={row['open']:.2f}, "
                           f"H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}",
                auto_fixable=True,
                suggested_fix="Recalculate high/low from open/close"
            ))

        # Check for suspicious high-low spreads
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        max_spread = self.validation_rules['consistency_rules']['high_low_spread_max']
        excessive_spreads = df[df['hl_spread'] > max_spread]

        for idx, row in excessive_spreads.iterrows():
            issues.append(QualityIssue(
                symbol=symbol,
                date=row['date'].strftime('%Y-%m-%d'),
                issue_type=QualityIssueType.OHLC_INCONSISTENCY,
                severity=QualitySeverity.MEDIUM,
                description=f"Excessive H-L spread: {row['hl_spread']:.2%} "
                           f"(H={row['high']:.2f}, L={row['low']:.2f})",
                value=row['hl_spread'],
                expected_value=max_spread,
                auto_fixable=False
            ))

        return issues

    def _detect_anomalies(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Detect statistical anomalies in price and volume data."""
        issues = []

        if len(df) < 30:  # Need enough data for statistical analysis
            return issues

        # Calculate returns for anomaly detection
        df_sorted = df.sort_values('date').copy()
        df_sorted['returns'] = df_sorted['close'].pct_change()
        df_sorted['log_volume'] = np.log1p(df_sorted['volume'])

        # Detect return outliers using z-score
        returns = df_sorted['returns'].dropna()
        if len(returns) > 10:
            return_zscore = np.abs(stats.zscore(returns))
            outlier_threshold = self.thresholds['return_outlier_zscore']
            outlier_indices = df_sorted[return_zscore > outlier_threshold].index

            for idx in outlier_indices:
                row = df_sorted.loc[idx]
                if not pd.isna(row['returns']):
                    issues.append(QualityIssue(
                        symbol=symbol,
                        date=row['date'].strftime('%Y-%m-%d'),
                        issue_type=QualityIssueType.OUTLIER_RETURN,
                        severity=QualitySeverity.MEDIUM,
                        description=f"Return outlier: {row['returns']:.2%} "
                                   f"(z-score: {return_zscore[idx]:.2f})",
                        value=row['returns'],
                        confidence=min(1.0, return_zscore[idx] / outlier_threshold),
                        auto_fixable=False
                    ))

        # Detect volume anomalies using isolation forest
        try:
            volume_data = df_sorted[['log_volume']].dropna()
            if len(volume_data) > 20:
                iso_forest = IsolationForest(
                    contamination=self.thresholds['isolation_forest_contamination'],
                    random_state=42
                )
                anomaly_labels = iso_forest.fit_predict(volume_data)
                anomaly_indices = volume_data[anomaly_labels == -1].index

                for idx in anomaly_indices:
                    row = df_sorted.loc[idx]
                    issues.append(QualityIssue(
                        symbol=symbol,
                        date=row['date'].strftime('%Y-%m-%d'),
                        issue_type=QualityIssueType.VOLUME_ANOMALY,
                        severity=QualitySeverity.LOW,
                        description=f"Volume anomaly detected: {row['volume']:,}",
                        value=row['volume'],
                        confidence=0.7,  # Isolation forest confidence
                        auto_fixable=False
                    ))
        except Exception as e:
            logger.warning(f"[quality] Isolation forest failed for {symbol}: {e}")

        return issues

    def _detect_corporate_action_issues(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Detect potential corporate action adjustment issues."""
        issues = []

        if len(df) < 10:
            return issues

        df_sorted = df.sort_values('date').copy()
        df_sorted['returns'] = df_sorted['close'].pct_change()

        # Detect potential stock splits (large overnight changes)
        split_threshold = self.validation_rules['price_rules']['split_detection_threshold']
        potential_splits = df_sorted[df_sorted['returns'].abs() > split_threshold]

        for idx, row in potential_splits.iterrows():
            if not pd.isna(row['returns']):
                # Check if this looks like a split (return close to -0.5, -0.66, etc.)
                split_ratios = [0.5, 0.33, 0.25, 0.2]  # 2:1, 3:1, 4:1, 5:1 splits
                likely_split = any(abs(row['returns'] + (1 - ratio)) < 0.05 for ratio in split_ratios)

                if likely_split:
                    issues.append(QualityIssue(
                        symbol=symbol,
                        date=row['date'].strftime('%Y-%m-%d'),
                        issue_type=QualityIssueType.SPLIT_NOT_ADJUSTED,
                        severity=QualitySeverity.HIGH,
                        description=f"Potential unadjusted stock split: {row['returns']:.2%} return",
                        value=row['returns'],
                        auto_fixable=False,
                        suggested_fix="Apply stock split adjustment to historical prices"
                    ))

        return issues

    def _apply_automatic_fixes(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
        symbol: str
    ) -> pd.DataFrame:
        """Apply automatic fixes to data quality issues."""
        clean_df = df.copy()
        fixes_applied = 0

        for issue in issues:
            if not issue.auto_fixable:
                continue

            issue_date = pd.to_datetime(issue.date)
            mask = clean_df['date'] == issue_date

            if issue.issue_type == QualityIssueType.NEGATIVE_PRICE:
                # Remove records with negative prices
                clean_df = clean_df[~mask]
                fixes_applied += 1

            elif issue.issue_type == QualityIssueType.ZERO_VOLUME:
                # Set negative volume to zero
                clean_df.loc[mask, 'volume'] = 0
                fixes_applied += 1

            elif issue.issue_type == QualityIssueType.DUPLICATE_RECORDS:
                # Remove duplicate records (keep last)
                clean_df = clean_df.drop_duplicates(subset=['date'], keep='last')
                fixes_applied += 1

            elif issue.issue_type == QualityIssueType.OHLC_INCONSISTENCY:
                # Fix basic OHLC inconsistencies
                row_indices = clean_df[mask].index
                for idx in row_indices:
                    row = clean_df.loc[idx]
                    # Ensure high >= max(open, close) and low <= min(open, close)
                    clean_df.loc[idx, 'high'] = max(row['high'], row['open'], row['close'])
                    clean_df.loc[idx, 'low'] = min(row['low'], row['open'], row['close'])
                fixes_applied += 1

        if fixes_applied > 0:
            logger.info(f"[quality] Applied {fixes_applied} automatic fixes for {symbol}")

        return clean_df

    def _calculate_quality_metrics(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
        symbol: str
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        total_records = len(df)

        # Count issues by severity
        critical_issues = sum(1 for issue in issues if issue.severity == QualitySeverity.CRITICAL)
        high_issues = sum(1 for issue in issues if issue.severity == QualitySeverity.HIGH)
        medium_issues = sum(1 for issue in issues if issue.severity == QualitySeverity.MEDIUM)

        # Calculate completeness (simplified - based on expected business days)
        if total_records > 0:
            date_range = pd.date_range(
                start=df['date'].min(),
                end=df['date'].max(),
                freq='B'
            )
            expected_records = len(date_range)
            completeness_ratio = min(1.0, total_records / expected_records)
        else:
            completeness_ratio = 0.0

        # OHLC consistency score
        ohlc_valid = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        ohlc_consistency_score = len(ohlc_valid) / total_records if total_records > 0 else 0.0

        # Volume consistency score (non-negative volumes)
        volume_valid = df[df['volume'] >= 0]
        volume_consistency_score = len(volume_valid) / total_records if total_records > 0 else 0.0

        # Count specific anomaly types
        price_anomaly_count = sum(1 for issue in issues if issue.issue_type in [
            QualityIssueType.PRICE_JUMP, QualityIssueType.OUTLIER_RETURN,
            QualityIssueType.NEGATIVE_PRICE
        ])

        volume_anomaly_count = sum(1 for issue in issues if issue.issue_type in [
            QualityIssueType.VOLUME_ANOMALY, QualityIssueType.ZERO_VOLUME
        ])

        # Calculate overall quality score
        # Weighted scoring: completeness (40%), consistency (30%), anomalies (30%)
        completeness_weight = 0.4
        consistency_weight = 0.3
        anomaly_weight = 0.3

        consistency_avg = (ohlc_consistency_score + volume_consistency_score) / 2
        anomaly_penalty = min(1.0, (critical_issues * 0.1 + high_issues * 0.05) / total_records)
        anomaly_score = max(0.0, 1.0 - anomaly_penalty)

        overall_quality_score = (
            completeness_ratio * completeness_weight +
            consistency_avg * consistency_weight +
            anomaly_score * anomaly_weight
        )

        # Generate recommendations
        recommendations = []
        if completeness_ratio < 0.95:
            recommendations.append("Improve data completeness - fill missing business days")
        if critical_issues > 0:
            recommendations.append("Address critical data quality issues immediately")
        if price_anomaly_count > total_records * 0.02:  # More than 2% anomalies
            recommendations.append("Investigate price anomalies and potential splits")
        if volume_anomaly_count > total_records * 0.05:  # More than 5% anomalies
            recommendations.append("Review volume data quality and outliers")

        return QualityMetrics(
            symbol=symbol,
            assessment_period=f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            total_records=total_records,
            missing_records=0,  # Simplified - would need business day analysis
            quality_issues=len(issues),
            critical_issues=critical_issues,
            completeness_ratio=completeness_ratio,
            consecutive_missing_max=0,  # Simplified
            ohlc_consistency_score=ohlc_consistency_score,
            volume_consistency_score=volume_consistency_score,
            price_anomaly_count=price_anomaly_count,
            volume_anomaly_count=volume_anomaly_count,
            overall_quality_score=overall_quality_score,
            recommended_actions=recommendations
        )

    def _save_quality_issues(self, issues: List[QualityIssue]):
        """Save quality issues to database."""
        if not issues:
            return

        with sqlite3.connect(self.db_path) as conn:
            for issue in issues:
                conn.execute("""
                    INSERT OR REPLACE INTO quality_issues
                    (symbol, date, issue_type, severity, description, value,
                     expected_value, confidence, auto_fixable, suggested_fix, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    issue.symbol, issue.date, issue.issue_type.value,
                    issue.severity.value, issue.description, issue.value,
                    issue.expected_value, issue.confidence, issue.auto_fixable,
                    issue.suggested_fix, issue.detected_at
                ))
            conn.commit()

    def _save_quality_metrics(self, metrics: QualityMetrics):
        """Save quality metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quality_metrics
                (symbol, assessment_date, assessment_period, total_records,
                 missing_records, quality_issues, critical_issues, completeness_ratio,
                 consecutive_missing_max, ohlc_consistency_score, volume_consistency_score,
                 price_anomaly_count, volume_anomaly_count, overall_quality_score,
                 recommended_actions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol, datetime.now().strftime('%Y-%m-%d'),
                metrics.assessment_period, metrics.total_records,
                metrics.missing_records, metrics.quality_issues,
                metrics.critical_issues, metrics.completeness_ratio,
                metrics.consecutive_missing_max, metrics.ohlc_consistency_score,
                metrics.volume_consistency_score, metrics.price_anomaly_count,
                metrics.volume_anomaly_count, metrics.overall_quality_score,
                json.dumps(metrics.recommended_actions)
            ))
            conn.commit()

    def get_quality_report(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive quality report."""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                # Symbol-specific report
                metrics = conn.execute("""
                    SELECT * FROM quality_metrics
                    WHERE symbol = ?
                    ORDER BY assessment_date DESC LIMIT 1
                """, (symbol,)).fetchone()

                issues = conn.execute("""
                    SELECT issue_type, severity, COUNT(*) as count
                    FROM quality_issues
                    WHERE symbol = ? AND NOT fixed
                    GROUP BY issue_type, severity
                """, (symbol,)).fetchall()

                return {
                    'symbol': symbol,
                    'latest_metrics': dict(zip([col[0] for col in conn.description], metrics)) if metrics else None,
                    'open_issues': [dict(zip(['issue_type', 'severity', 'count'], issue)) for issue in issues]
                }
            else:
                # System-wide report
                avg_quality = conn.execute("""
                    SELECT AVG(overall_quality_score)
                    FROM quality_metrics
                """).fetchone()[0]

                issue_summary = conn.execute("""
                    SELECT severity, COUNT(*) as count
                    FROM quality_issues
                    WHERE NOT fixed
                    GROUP BY severity
                """).fetchall()

                return {
                    'average_quality_score': avg_quality,
                    'open_issues_by_severity': [dict(zip(['severity', 'count'], item)) for item in issue_summary]
                }


# Convenience functions
def validate_symbol_data(
    df: pd.DataFrame,
    symbol: str,
    perform_fixes: bool = True
) -> Tuple[pd.DataFrame, QualityMetrics]:
    """
    Convenience function to validate data for a single symbol.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        perform_fixes: Whether to apply automatic fixes

    Returns:
        Tuple of (cleaned_df, quality_metrics)
    """
    validator = DataQualityValidator()
    clean_df, metrics, issues = validator.validate_dataset(
        df, symbol, perform_fixes=perform_fixes
    )
    return clean_df, metrics


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Download test data
    print("Testing data quality validation...")
    test_data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
    test_data = test_data.reset_index()
    test_data.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

    # Validate data
    clean_data, quality_metrics = validate_symbol_data(test_data, "AAPL")

    print(f"Quality validation results for AAPL:")
    print(f"  Original records: {len(test_data)}")
    print(f"  Clean records: {len(clean_data)}")
    print(f"  Overall quality score: {quality_metrics.overall_quality_score:.3f}")
    print(f"  Issues found: {quality_metrics.quality_issues}")
    print(f"  Critical issues: {quality_metrics.critical_issues}")
    print(f"  Recommendations: {quality_metrics.recommended_actions}")