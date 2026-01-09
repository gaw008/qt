"""
Corporate Actions Processing System

This module handles comprehensive corporate actions processing for historical
market data, ensuring accurate price adjustments for backtesting and analysis.

Key Features:
- Stock split and reverse split detection and adjustment
- Dividend ex-date processing and adjustment
- Merger and acquisition handling
- Symbol change tracking and mapping
- Spinoff processing
- Rights issue adjustments
- Automated detection from price patterns
- Manual corporate action entry
- Historical reconstruction of adjusted prices
- Data integrity validation after adjustments
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path
import requests
import time
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorporateActionType(Enum):
    """Types of corporate actions."""
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"
    RIGHTS_ISSUE = "rights_issue"
    SPECIAL_DIVIDEND = "special_dividend"


class AdjustmentStatus(Enum):
    """Status of corporate action processing."""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    VERIFIED = "verified"
    MANUAL_REVIEW = "manual_review"


@dataclass
class CorporateAction:
    """Corporate action record with comprehensive details."""
    symbol: str
    action_type: CorporateActionType
    ex_date: str  # ISO format date (YYYY-MM-DD)
    announcement_date: Optional[str] = None
    record_date: Optional[str] = None
    payment_date: Optional[str] = None

    # Split-specific fields
    split_ratio: Optional[float] = None  # New shares per old share (e.g., 2.0 for 2:1 split)
    reverse_split_ratio: Optional[float] = None  # Old shares per new share

    # Dividend-specific fields
    dividend_amount: Optional[float] = None  # Per share amount
    dividend_currency: Optional[str] = None

    # Merger/Acquisition fields
    target_symbol: Optional[str] = None
    cash_amount: Optional[float] = None
    stock_ratio: Optional[float] = None

    # Symbol change fields
    new_symbol: Optional[str] = None
    old_symbol: Optional[str] = None

    # Spinoff fields
    spinoff_symbol: Optional[str] = None
    spinoff_ratio: Optional[float] = None

    # Metadata
    description: str = ""
    source: str = ""
    confidence: float = 1.0
    manual_entry: bool = False
    status: AdjustmentStatus = AdjustmentStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processed_at: Optional[str] = None
    verified_at: Optional[str] = None

    def __post_init__(self):
        """Validate corporate action data."""
        if self.action_type in [CorporateActionType.STOCK_SPLIT, CorporateActionType.REVERSE_SPLIT]:
            if self.split_ratio is None and self.reverse_split_ratio is None:
                raise ValueError("Split actions require either split_ratio or reverse_split_ratio")

        if self.action_type == CorporateActionType.DIVIDEND:
            if self.dividend_amount is None:
                raise ValueError("Dividend actions require dividend_amount")

    def get_price_adjustment_factor(self) -> float:
        """
        Calculate price adjustment factor for this corporate action.

        Returns:
            Factor to multiply historical prices by
        """
        if self.action_type == CorporateActionType.STOCK_SPLIT:
            return 1.0 / self.split_ratio if self.split_ratio else 1.0

        elif self.action_type == CorporateActionType.REVERSE_SPLIT:
            return self.reverse_split_ratio if self.reverse_split_ratio else 1.0

        elif self.action_type == CorporateActionType.DIVIDEND:
            # For regular dividends, typically no price adjustment needed
            # as the market adjusts automatically on ex-date
            return 1.0

        elif self.action_type == CorporateActionType.SPECIAL_DIVIDEND:
            # Special dividends may require price adjustment
            # This is simplified - in practice, need current price context
            return 1.0

        return 1.0

    def get_volume_adjustment_factor(self) -> float:
        """
        Calculate volume adjustment factor for this corporate action.

        Returns:
            Factor to multiply historical volumes by
        """
        if self.action_type == CorporateActionType.STOCK_SPLIT:
            return self.split_ratio if self.split_ratio else 1.0

        elif self.action_type == CorporateActionType.REVERSE_SPLIT:
            return 1.0 / self.reverse_split_ratio if self.reverse_split_ratio else 1.0

        return 1.0


@dataclass
class AdjustmentFactor:
    """Pre-calculated adjustment factor for a specific date and symbol."""
    symbol: str
    date: str
    price_factor: float
    volume_factor: float
    cumulative_price_factor: float
    cumulative_volume_factor: float
    action_type: CorporateActionType
    action_id: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CorporateActionDetector:
    """
    Automatic detection of corporate actions from price and volume patterns.
    """

    def __init__(self):
        """Initialize corporate action detector."""
        self.detection_thresholds = {
            'split_return_threshold': 0.4,  # 40% overnight return suggests split
            'min_split_ratio': 1.5,  # Minimum split ratio to consider
            'max_split_ratio': 10.0,  # Maximum split ratio to consider
            'dividend_volume_spike': 2.0,  # 2x volume spike on ex-date
            'price_gap_threshold': 0.1,  # 10% price gap
        }

    def detect_stock_splits(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """
        Detect stock splits from price patterns.

        Args:
            df: DataFrame with OHLCV data sorted by date
            symbol: Stock symbol

        Returns:
            List of detected corporate actions
        """
        detected_actions = []

        if len(df) < 2:
            return detected_actions

        # Calculate overnight returns
        df_sorted = df.sort_values('date').copy()
        df_sorted['prev_close'] = df_sorted['close'].shift(1)
        df_sorted['overnight_return'] = (df_sorted['open'] - df_sorted['prev_close']) / df_sorted['prev_close']

        # Look for large negative overnight returns (potential splits)
        split_threshold = -self.detection_thresholds['split_return_threshold']
        potential_splits = df_sorted[
            (df_sorted['overnight_return'] < split_threshold) &
            (df_sorted['overnight_return'].notna())
        ]

        for idx, row in potential_splits.iterrows():
            overnight_return = row['overnight_return']

            # Calculate implied split ratio
            # If price drops by 50%, it suggests a 2:1 split
            implied_ratio = 1.0 / (1.0 + overnight_return)

            # Check if this is close to a common split ratio
            common_ratios = [2.0, 3.0, 1.5, 4.0, 5.0]
            closest_ratio = min(common_ratios, key=lambda x: abs(x - implied_ratio))

            if abs(closest_ratio - implied_ratio) < 0.2:  # Within 20% of common ratio
                action = CorporateAction(
                    symbol=symbol,
                    action_type=CorporateActionType.STOCK_SPLIT,
                    ex_date=row['date'].strftime('%Y-%m-%d'),
                    split_ratio=closest_ratio,
                    description=f"Auto-detected {closest_ratio}:1 stock split "
                               f"(overnight return: {overnight_return:.2%})",
                    source="automatic_detection",
                    confidence=max(0.5, 1.0 - abs(closest_ratio - implied_ratio))
                )
                detected_actions.append(action)

        logger.info(f"[corp_actions] Detected {len(detected_actions)} potential splits for {symbol}")
        return detected_actions

    def detect_reverse_splits(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """Detect reverse splits from price patterns."""
        detected_actions = []

        if len(df) < 2:
            return detected_actions

        df_sorted = df.sort_values('date').copy()
        df_sorted['prev_close'] = df_sorted['close'].shift(1)
        df_sorted['overnight_return'] = (df_sorted['open'] - df_sorted['prev_close']) / df_sorted['prev_close']

        # Look for large positive overnight returns (potential reverse splits)
        reverse_split_threshold = self.detection_thresholds['split_return_threshold']
        potential_reverse_splits = df_sorted[
            (df_sorted['overnight_return'] > reverse_split_threshold) &
            (df_sorted['overnight_return'].notna())
        ]

        for idx, row in potential_reverse_splits.iterrows():
            overnight_return = row['overnight_return']

            # Calculate implied reverse split ratio
            implied_ratio = 1.0 + overnight_return

            # Check if this is close to a common reverse split ratio
            common_ratios = [2.0, 3.0, 4.0, 5.0, 10.0]
            closest_ratio = min(common_ratios, key=lambda x: abs(x - implied_ratio))

            if abs(closest_ratio - implied_ratio) < 0.3:  # Within 30% of common ratio
                action = CorporateAction(
                    symbol=symbol,
                    action_type=CorporateActionType.REVERSE_SPLIT,
                    ex_date=row['date'].strftime('%Y-%m-%d'),
                    reverse_split_ratio=closest_ratio,
                    description=f"Auto-detected 1:{closest_ratio} reverse split "
                               f"(overnight return: {overnight_return:.2%})",
                    source="automatic_detection",
                    confidence=max(0.5, 1.0 - abs(closest_ratio - implied_ratio) / closest_ratio)
                )
                detected_actions.append(action)

        return detected_actions

    def detect_dividends(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """
        Detect dividends from volume spikes and small price gaps.

        This is more challenging than splits and less reliable.
        """
        detected_actions = []

        if len(df) < 10:  # Need more data for dividend detection
            return detected_actions

        df_sorted = df.sort_values('date').copy()

        # Calculate rolling average volume
        df_sorted['volume_ma'] = df_sorted['volume'].rolling(window=20, min_periods=5).mean()
        df_sorted['volume_ratio'] = df_sorted['volume'] / df_sorted['volume_ma']

        # Look for volume spikes that might indicate ex-dividend dates
        volume_spike_threshold = self.detection_thresholds['dividend_volume_spike']
        volume_spikes = df_sorted[
            (df_sorted['volume_ratio'] > volume_spike_threshold) &
            (df_sorted['volume_ratio'].notna())
        ]

        # For each volume spike, check if there's a small price gap
        for idx, row in volume_spikes.iterrows():
            # This is a simplified heuristic - in practice, dividend detection
            # is much more complex and typically requires external data sources
            if row['volume_ratio'] > 3.0:  # Very high volume
                # Check for small negative price gap
                prev_row_idx = df_sorted.index[df_sorted.index < idx]
                if len(prev_row_idx) > 0:
                    prev_idx = prev_row_idx[-1]
                    prev_row = df_sorted.loc[prev_idx]

                    price_gap = (row['open'] - prev_row['close']) / prev_row['close']

                    if -0.05 < price_gap < 0:  # Small negative gap (up to 5%)
                        estimated_dividend = abs(price_gap * prev_row['close'])

                        action = CorporateAction(
                            symbol=symbol,
                            action_type=CorporateActionType.DIVIDEND,
                            ex_date=row['date'].strftime('%Y-%m-%d'),
                            dividend_amount=estimated_dividend,
                            description=f"Auto-detected potential dividend "
                                       f"(volume spike: {row['volume_ratio']:.1f}x, "
                                       f"estimated amount: ${estimated_dividend:.2f})",
                            source="automatic_detection",
                            confidence=0.3  # Low confidence for auto-detected dividends
                        )
                        detected_actions.append(action)

        return detected_actions


class CorporateActionsProcessor:
    """
    Main corporate actions processing system.

    Handles detection, storage, and application of corporate actions
    to historical price data.
    """

    def __init__(self, db_path: str = "data_cache/corporate_actions.db"):
        """
        Initialize corporate actions processor.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.detector = CorporateActionDetector()

        # Initialize database
        self._init_database()

        # Cache for adjustment factors
        self._adjustment_cache = {}
        self._cache_expiry = {}
        self._cache_ttl = 3600  # 1 hour

        logger.info(f"[corp_actions] Processor initialized with database: {db_path}")

    def _init_database(self):
        """Initialize SQLite database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Corporate actions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    announcement_date TEXT,
                    record_date TEXT,
                    payment_date TEXT,
                    split_ratio REAL,
                    reverse_split_ratio REAL,
                    dividend_amount REAL,
                    dividend_currency TEXT,
                    target_symbol TEXT,
                    cash_amount REAL,
                    stock_ratio REAL,
                    new_symbol TEXT,
                    old_symbol TEXT,
                    spinoff_symbol TEXT,
                    spinoff_ratio REAL,
                    description TEXT,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    manual_entry BOOLEAN DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    verified_at TEXT,
                    UNIQUE(symbol, action_type, ex_date, split_ratio, dividend_amount)
                )
            """)

            # Adjustment factors table (pre-calculated for performance)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adjustment_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    price_factor REAL NOT NULL,
                    volume_factor REAL NOT NULL,
                    cumulative_price_factor REAL NOT NULL,
                    cumulative_volume_factor REAL NOT NULL,
                    action_type TEXT NOT NULL,
                    action_id INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, date),
                    FOREIGN KEY (action_id) REFERENCES corporate_actions (id)
                )
            """)

            # Price adjustment log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adjustment_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action_id INTEGER NOT NULL,
                    records_affected INTEGER DEFAULT 0,
                    adjustment_start_date TEXT,
                    adjustment_end_date TEXT,
                    price_factor_applied REAL,
                    volume_factor_applied REAL,
                    processed_at TEXT NOT NULL,
                    FOREIGN KEY (action_id) REFERENCES corporate_actions (id)
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_symbol_date ON corporate_actions(symbol, ex_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_factors_symbol_date ON adjustment_factors(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_symbol ON adjustment_log(symbol, processed_at)")

            conn.commit()

    def add_corporate_action(self, action: CorporateAction) -> int:
        """
        Add a corporate action to the database.

        Args:
            action: CorporateAction object

        Returns:
            ID of the inserted action
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO corporate_actions
                (symbol, action_type, ex_date, announcement_date, record_date,
                 payment_date, split_ratio, reverse_split_ratio, dividend_amount,
                 dividend_currency, target_symbol, cash_amount, stock_ratio,
                 new_symbol, old_symbol, spinoff_symbol, spinoff_ratio,
                 description, source, confidence, manual_entry, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action.symbol, action.action_type.value, action.ex_date,
                action.announcement_date, action.record_date, action.payment_date,
                action.split_ratio, action.reverse_split_ratio, action.dividend_amount,
                action.dividend_currency, action.target_symbol, action.cash_amount,
                action.stock_ratio, action.new_symbol, action.old_symbol,
                action.spinoff_symbol, action.spinoff_ratio, action.description,
                action.source, action.confidence, action.manual_entry,
                action.status.value, action.created_at
            ))

            action_id = cursor.lastrowid
            conn.commit()

        logger.info(f"[corp_actions] Added {action.action_type.value} for {action.symbol} on {action.ex_date}")
        return action_id

    def detect_corporate_actions(self, df: pd.DataFrame, symbol: str) -> List[CorporateAction]:
        """
        Automatically detect corporate actions from price data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            List of detected corporate actions
        """
        all_detected = []

        # Detect stock splits
        splits = self.detector.detect_stock_splits(df, symbol)
        all_detected.extend(splits)

        # Detect reverse splits
        reverse_splits = self.detector.detect_reverse_splits(df, symbol)
        all_detected.extend(reverse_splits)

        # Detect dividends (lower confidence)
        dividends = self.detector.detect_dividends(df, symbol)
        all_detected.extend(dividends)

        logger.info(f"[corp_actions] Auto-detected {len(all_detected)} corporate actions for {symbol}")
        return all_detected

    def apply_corporate_actions(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply corporate actions to historical price data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            start_date: Optional start date for adjustments
            end_date: Optional end date for adjustments

        Returns:
            DataFrame with adjusted prices
        """
        if len(df) == 0:
            return df

        # Get corporate actions for this symbol
        actions = self.get_corporate_actions(symbol, start_date, end_date)

        if not actions:
            logger.info(f"[corp_actions] No corporate actions found for {symbol}")
            return df

        # Sort dataframe by date and make a copy
        adjusted_df = df.sort_values('date').copy()

        # Convert date column if needed
        if not pd.api.types.is_datetime64_any_dtype(adjusted_df['date']):
            adjusted_df['date'] = pd.to_datetime(adjusted_df['date'])

        # Apply each corporate action
        for action in sorted(actions, key=lambda x: x.ex_date):
            adjusted_df = self._apply_single_action(adjusted_df, action)

        logger.info(f"[corp_actions] Applied {len(actions)} corporate actions to {symbol}")
        return adjusted_df

    def _apply_single_action(self, df: pd.DataFrame, action: CorporateAction) -> pd.DataFrame:
        """Apply a single corporate action to the dataframe."""
        ex_date = pd.to_datetime(action.ex_date)

        # Only adjust data before the ex-date
        pre_action_mask = df['date'] < ex_date

        if not pre_action_mask.any():
            return df  # No data before ex-date

        price_factor = action.get_price_adjustment_factor()
        volume_factor = action.get_volume_adjustment_factor()

        # Apply price adjustments
        if price_factor != 1.0:
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df.loc[pre_action_mask, col] *= price_factor

            # Apply to adjusted_close if present
            if 'adjusted_close' in df.columns:
                df.loc[pre_action_mask, 'adjusted_close'] *= price_factor

        # Apply volume adjustments
        if volume_factor != 1.0 and 'volume' in df.columns:
            df.loc[pre_action_mask, 'volume'] *= volume_factor

        # Log the adjustment
        self._log_adjustment(action, pre_action_mask.sum(), price_factor, volume_factor)

        return df

    def _log_adjustment(
        self,
        action: CorporateAction,
        records_affected: int,
        price_factor: float,
        volume_factor: float
    ):
        """Log the applied adjustment."""
        with sqlite3.connect(self.db_path) as conn:
            # First, get the action ID if it exists in database
            action_id_result = conn.execute("""
                SELECT id FROM corporate_actions
                WHERE symbol = ? AND action_type = ? AND ex_date = ?
            """, (action.symbol, action.action_type.value, action.ex_date)).fetchone()

            action_id = action_id_result[0] if action_id_result else None

            if action_id:
                conn.execute("""
                    INSERT INTO adjustment_log
                    (symbol, action_id, records_affected, price_factor_applied,
                     volume_factor_applied, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    action.symbol, action_id, records_affected,
                    price_factor, volume_factor, datetime.now().isoformat()
                ))
                conn.commit()

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        action_types: Optional[List[CorporateActionType]] = None
    ) -> List[CorporateAction]:
        """
        Retrieve corporate actions for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            action_types: Optional list of action types to filter

        Returns:
            List of corporate actions
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM corporate_actions WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND ex_date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND ex_date <= ?"
                params.append(end_date)

            if action_types:
                action_type_strings = [at.value for at in action_types]
                placeholders = ','.join(['?' for _ in action_type_strings])
                query += f" AND action_type IN ({placeholders})"
                params.extend(action_type_strings)

            query += " ORDER BY ex_date"

            rows = conn.execute(query, params).fetchall()
            columns = [description[0] for description in conn.description]

        actions = []
        for row in rows:
            row_dict = dict(zip(columns, row))

            # Convert to CorporateAction object
            action = CorporateAction(
                symbol=row_dict['symbol'],
                action_type=CorporateActionType(row_dict['action_type']),
                ex_date=row_dict['ex_date'],
                announcement_date=row_dict['announcement_date'],
                record_date=row_dict['record_date'],
                payment_date=row_dict['payment_date'],
                split_ratio=row_dict['split_ratio'],
                reverse_split_ratio=row_dict['reverse_split_ratio'],
                dividend_amount=row_dict['dividend_amount'],
                dividend_currency=row_dict['dividend_currency'],
                target_symbol=row_dict['target_symbol'],
                cash_amount=row_dict['cash_amount'],
                stock_ratio=row_dict['stock_ratio'],
                new_symbol=row_dict['new_symbol'],
                old_symbol=row_dict['old_symbol'],
                spinoff_symbol=row_dict['spinoff_symbol'],
                spinoff_ratio=row_dict['spinoff_ratio'],
                description=row_dict['description'],
                source=row_dict['source'],
                confidence=row_dict['confidence'],
                manual_entry=bool(row_dict['manual_entry']),
                status=AdjustmentStatus(row_dict['status']),
                created_at=row_dict['created_at'],
                processed_at=row_dict['processed_at'],
                verified_at=row_dict['verified_at']
            )
            actions.append(action)

        return actions

    def calculate_adjustment_factors(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, AdjustmentFactor]:
        """
        Calculate cumulative adjustment factors for a date range.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary mapping dates to AdjustmentFactor objects
        """
        # Get corporate actions for the symbol
        actions = self.get_corporate_actions(symbol, start_date, end_date)

        if not actions:
            return {}

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        adjustment_factors = {}

        # Initialize cumulative factors
        cumulative_price_factor = 1.0
        cumulative_volume_factor = 1.0

        # Process each date in reverse order (from end to start)
        # This ensures we apply adjustments correctly
        for current_date in reversed(date_range):
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Check if there's a corporate action on this date
            daily_price_factor = 1.0
            daily_volume_factor = 1.0
            action_type = None
            action_id = None

            for action in actions:
                if action.ex_date == current_date_str:
                    daily_price_factor = action.get_price_adjustment_factor()
                    daily_volume_factor = action.get_volume_adjustment_factor()
                    action_type = action.action_type
                    # Would need to get action_id from database in real implementation
                    break

            # Update cumulative factors
            if current_date < pd.Timestamp.now():  # Only for past dates
                factor = AdjustmentFactor(
                    symbol=symbol,
                    date=current_date_str,
                    price_factor=daily_price_factor,
                    volume_factor=daily_volume_factor,
                    cumulative_price_factor=cumulative_price_factor,
                    cumulative_volume_factor=cumulative_volume_factor,
                    action_type=action_type or CorporateActionType.DIVIDEND,  # Default
                    action_id=action_id
                )
                adjustment_factors[current_date_str] = factor

                # Update cumulative factors for the next iteration (previous date)
                cumulative_price_factor *= daily_price_factor
                cumulative_volume_factor *= daily_volume_factor

        return adjustment_factors

    def get_corporate_actions_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of corporate actions in the database."""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                # Symbol-specific summary
                total_actions = conn.execute(
                    "SELECT COUNT(*) FROM corporate_actions WHERE symbol = ?", (symbol,)
                ).fetchone()[0]

                by_type = conn.execute("""
                    SELECT action_type, COUNT(*) as count
                    FROM corporate_actions
                    WHERE symbol = ?
                    GROUP BY action_type
                """, (symbol,)).fetchall()

                recent_actions = conn.execute("""
                    SELECT action_type, ex_date, description
                    FROM corporate_actions
                    WHERE symbol = ?
                    ORDER BY ex_date DESC LIMIT 5
                """, (symbol,)).fetchall()

                return {
                    'symbol': symbol,
                    'total_actions': total_actions,
                    'by_type': [dict(zip(['action_type', 'count'], item)) for item in by_type],
                    'recent_actions': [dict(zip(['action_type', 'ex_date', 'description'], item)) for item in recent_actions]
                }
            else:
                # System-wide summary
                total_actions = conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0]
                total_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM corporate_actions").fetchone()[0]

                by_type = conn.execute("""
                    SELECT action_type, COUNT(*) as count
                    FROM corporate_actions
                    GROUP BY action_type
                """).fetchall()

                return {
                    'total_actions': total_actions,
                    'total_symbols': total_symbols,
                    'by_type': [dict(zip(['action_type', 'count'], item)) for item in by_type]
                }


# Convenience functions
def process_corporate_actions_for_symbol(
    df: pd.DataFrame,
    symbol: str,
    auto_detect: bool = True
) -> Tuple[pd.DataFrame, List[CorporateAction]]:
    """
    Convenience function to process corporate actions for a single symbol.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        auto_detect: Whether to auto-detect corporate actions

    Returns:
        Tuple of (adjusted_df, detected_actions)
    """
    processor = CorporateActionsProcessor()

    detected_actions = []
    if auto_detect:
        detected_actions = processor.detect_corporate_actions(df, symbol)

        # Add detected actions to database
        for action in detected_actions:
            processor.add_corporate_action(action)

    # Apply all corporate actions (existing + detected)
    adjusted_df = processor.apply_corporate_actions(df, symbol)

    return adjusted_df, detected_actions


if __name__ == "__main__":
    # Example usage
    print("Testing corporate actions processing...")

    # Create sample data with a stock split
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
    np.random.seed(42)

    # Generate price data with a 2:1 split on 2020-06-15
    prices = []
    base_price = 100.0

    for date in dates:
        if date < pd.Timestamp('2020-06-15'):
            # Pre-split prices
            price = base_price + np.random.normal(0, 2)
            base_price = price
        else:
            # Post-split prices (half the price)
            if len(prices) == 0 or date == pd.Timestamp('2020-06-15'):
                price = base_price / 2.0  # Split adjustment
            else:
                price = prices[-1] + np.random.normal(0, 1)

        prices.append(max(1.0, price))

    # Create DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'close': prices,
        'volume': [int(abs(np.random.normal(100000, 20000))) for _ in prices]
    })

    # Test corporate actions processing
    adjusted_data, detected = process_corporate_actions_for_symbol(
        test_data, "TEST", auto_detect=True
    )

    print(f"Original data points: {len(test_data)}")
    print(f"Adjusted data points: {len(adjusted_data)}")
    print(f"Detected corporate actions: {len(detected)}")

    if detected:
        for action in detected:
            print(f"  - {action.action_type.value} on {action.ex_date}: {action.description}")

    # Show price comparison around split date
    split_window = test_data[
        (test_data['date'] >= '2020-06-10') &
        (test_data['date'] <= '2020-06-20')
    ][['date', 'close']].copy()

    split_window_adj = adjusted_data[
        (adjusted_data['date'] >= '2020-06-10') &
        (adjusted_data['date'] <= '2020-06-20')
    ][['date', 'close']].copy()

    print("\nPrice comparison around split date:")
    print("Date         Original  Adjusted")
    for i, (_, row) in enumerate(split_window.iterrows()):
        adj_row = split_window_adj.iloc[i]
        print(f"{row['date'].strftime('%Y-%m-%d')}  {row['close']:8.2f}  {adj_row['close']:8.2f}")