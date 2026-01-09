"""
Corporate Actions Processing System

This module handles corporate actions for financial data including stock splits,
dividends, mergers, spinoffs, and symbol changes. It maintains both raw and
adjusted price series to ensure accurate backtesting and analysis.

Key Features:
- Stock split and reverse split adjustments
- Dividend ex-date processing
- Merger and acquisition handling
- Symbol change tracking
- Dual-track data storage (raw/adjusted)
- Historical action replay capability
- Data consistency validation
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path

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


@dataclass
class CorporateAction:
    """Individual corporate action record."""
    symbol: str
    action_type: CorporateActionType
    ex_date: str  # ISO format date
    announcement_date: Optional[str] = None
    effective_date: Optional[str] = None

    # Action-specific parameters
    ratio: Optional[float] = None  # For splits (e.g., 2.0 for 2:1 split)
    amount: Optional[float] = None  # For dividends
    new_symbol: Optional[str] = None  # For symbol changes
    target_symbol: Optional[str] = None  # For mergers

    # Metadata
    description: str = ""
    source: str = ""
    confidence: float = 1.0
    processed: bool = False
    created_date: str = ""

    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()


@dataclass
class AdjustmentFactor:
    """Price adjustment factor for a specific date."""
    symbol: str
    date: str
    price_factor: float  # Multiply raw price by this
    volume_factor: float  # Multiply raw volume by this
    action_type: CorporateActionType
    action_id: Optional[str] = None


class CorporateActionsProcessor:
    """
    Processes corporate actions and maintains adjusted price series.
    """

    def __init__(self, data_dir: str = "data_cache/corporate_actions"):
        """
        Initialize corporate actions processor.

        Args:
            data_dir: Directory for storing corporate actions data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Database for storing actions
        self.db_path = self.data_dir / "corporate_actions.db"
        self.adjustment_factors_cache: Dict[str, List[AdjustmentFactor]] = {}

        # Initialize database
        self._init_database()

        logger.info(f"[corp_actions] Processor initialized with data dir: {self.data_dir}")

    def _init_database(self):
        """Initialize SQLite database for corporate actions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Corporate actions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS corporate_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        ex_date TEXT NOT NULL,
                        announcement_date TEXT,
                        effective_date TEXT,
                        ratio REAL,
                        amount REAL,
                        new_symbol TEXT,
                        target_symbol TEXT,
                        description TEXT,
                        source TEXT,
                        confidence REAL DEFAULT 1.0,
                        processed INTEGER DEFAULT 0,
                        created_date TEXT NOT NULL,
                        UNIQUE(symbol, action_type, ex_date, ratio, amount)
                    )
                """)

                # Adjustment factors table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS adjustment_factors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        price_factor REAL NOT NULL,
                        volume_factor REAL NOT NULL,
                        action_type TEXT NOT NULL,
                        action_id INTEGER,
                        created_date TEXT NOT NULL,
                        UNIQUE(symbol, date, action_type),
                        FOREIGN KEY (action_id) REFERENCES corporate_actions (id)
                    )
                """)

                # Indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_symbol_date ON corporate_actions(symbol, ex_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_factors_symbol_date ON adjustment_factors(symbol, date)")

                conn.commit()

            logger.info("[corp_actions] Database initialized successfully")

        except Exception as e:
            logger.error(f"[corp_actions] Database initialization failed: {e}")
            raise

    def add_corporate_action(self, action: CorporateAction) -> bool:
        """
        Add a new corporate action.

        Args:
            action: Corporate action to add

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO corporate_actions
                    (symbol, action_type, ex_date, announcement_date, effective_date,
                     ratio, amount, new_symbol, target_symbol, description, source,
                     confidence, processed, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    action.symbol,
                    action.action_type.value,
                    action.ex_date,
                    action.announcement_date,
                    action.effective_date,
                    action.ratio,
                    action.amount,
                    action.new_symbol,
                    action.target_symbol,
                    action.description,
                    action.source,
                    action.confidence,
                    int(action.processed),
                    action.created_date
                ))

                action_id = cursor.lastrowid
                conn.commit()

            logger.info(f"[corp_actions] Added action: {action.symbol} {action.action_type.value} on {action.ex_date}")

            # Process the action if it's ready
            if not action.processed:
                self._process_single_action(action_id)

            return True

        except Exception as e:
            logger.error(f"[corp_actions] Failed to add corporate action: {e}")
            return False

    def load_actions_from_csv(self, csv_path: str) -> int:
        """
        Load corporate actions from CSV file.

        Args:
            csv_path: Path to CSV file with corporate actions

        Returns:
            Number of actions loaded
        """
        try:
            df = pd.read_csv(csv_path)
            loaded_count = 0

            required_columns = ['symbol', 'action_type', 'ex_date']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            for _, row in df.iterrows():
                try:
                    action = CorporateAction(
                        symbol=row['symbol'],
                        action_type=CorporateActionType(row['action_type']),
                        ex_date=row['ex_date'],
                        announcement_date=row.get('announcement_date'),
                        effective_date=row.get('effective_date'),
                        ratio=row.get('ratio'),
                        amount=row.get('amount'),
                        new_symbol=row.get('new_symbol'),
                        target_symbol=row.get('target_symbol'),
                        description=row.get('description', ''),
                        source=row.get('source', 'csv_import'),
                        confidence=row.get('confidence', 1.0)
                    )

                    if self.add_corporate_action(action):
                        loaded_count += 1

                except Exception as e:
                    logger.warning(f"[corp_actions] Failed to load action from row: {e}")
                    continue

            logger.info(f"[corp_actions] Loaded {loaded_count} actions from CSV")
            return loaded_count

        except Exception as e:
            logger.error(f"[corp_actions] Failed to load actions from CSV: {e}")
            return 0

    def get_actions_for_symbol(self, symbol: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> List[CorporateAction]:
        """
        Get corporate actions for a specific symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)

        Returns:
            List of corporate actions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM corporate_actions WHERE symbol = ?"
                params = [symbol]

                if start_date:
                    query += " AND ex_date >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND ex_date <= ?"
                    params.append(end_date)

                query += " ORDER BY ex_date"

                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

                actions = []
                columns = [desc[0] for desc in cursor.description]

                for row in rows:
                    row_dict = dict(zip(columns, row))
                    action = CorporateAction(
                        symbol=row_dict['symbol'],
                        action_type=CorporateActionType(row_dict['action_type']),
                        ex_date=row_dict['ex_date'],
                        announcement_date=row_dict['announcement_date'],
                        effective_date=row_dict['effective_date'],
                        ratio=row_dict['ratio'],
                        amount=row_dict['amount'],
                        new_symbol=row_dict['new_symbol'],
                        target_symbol=row_dict['target_symbol'],
                        description=row_dict['description'],
                        source=row_dict['source'],
                        confidence=row_dict['confidence'],
                        processed=bool(row_dict['processed']),
                        created_date=row_dict['created_date']
                    )
                    actions.append(action)

                return actions

        except Exception as e:
            logger.error(f"[corp_actions] Failed to get actions for {symbol}: {e}")
            return []

    def adjust_price_series(self, symbol: str, price_data: pd.DataFrame,
                           adjustment_date: Optional[str] = None) -> pd.DataFrame:
        """
        Apply corporate action adjustments to price series.

        Args:
            symbol: Stock symbol
            price_data: DataFrame with OHLCV data
            adjustment_date: Apply adjustments as of this date (default: latest)

        Returns:
            Adjusted price DataFrame
        """
        try:
            if adjustment_date is None:
                adjustment_date = datetime.now().strftime('%Y-%m-%d')

            # Get adjustment factors
            factors = self._get_adjustment_factors(symbol, adjustment_date)

            if not factors:
                logger.debug(f"[corp_actions] No adjustments needed for {symbol}")
                return price_data.copy()

            adjusted_data = price_data.copy()

            # Apply adjustments in chronological order
            for factor in sorted(factors, key=lambda x: x.date):
                factor_date = pd.to_datetime(factor.date)

                # Apply to all data before and including the factor date
                mask = adjusted_data.index <= factor_date

                if mask.any():
                    # Adjust price columns
                    price_columns = ['open', 'high', 'low', 'close']
                    for col in price_columns:
                        if col in adjusted_data.columns:
                            adjusted_data.loc[mask, col] *= factor.price_factor

                    # Adjust volume
                    if 'volume' in adjusted_data.columns:
                        adjusted_data.loc[mask, 'volume'] *= factor.volume_factor

                    logger.debug(f"[corp_actions] Applied {factor.action_type.value} adjustment "
                               f"for {symbol} on {factor.date} (factor: {factor.price_factor:.4f})")

            return adjusted_data

        except Exception as e:
            logger.error(f"[corp_actions] Price adjustment failed for {symbol}: {e}")
            return price_data.copy()

    def _get_adjustment_factors(self, symbol: str, as_of_date: str) -> List[AdjustmentFactor]:
        """Get all adjustment factors for a symbol up to a specific date."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{as_of_date}"
            if cache_key in self.adjustment_factors_cache:
                return self.adjustment_factors_cache[cache_key]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, date, price_factor, volume_factor, action_type, action_id
                    FROM adjustment_factors
                    WHERE symbol = ? AND date <= ?
                    ORDER BY date
                """, (symbol, as_of_date))

                rows = cursor.fetchall()
                factors = []

                for row in rows:
                    factor = AdjustmentFactor(
                        symbol=row[0],
                        date=row[1],
                        price_factor=row[2],
                        volume_factor=row[3],
                        action_type=CorporateActionType(row[4]),
                        action_id=str(row[5]) if row[5] else None
                    )
                    factors.append(factor)

                # Cache the result
                self.adjustment_factors_cache[cache_key] = factors
                return factors

        except Exception as e:
            logger.error(f"[corp_actions] Failed to get adjustment factors: {e}")
            return []

    def _process_single_action(self, action_id: int):
        """Process a single corporate action to generate adjustment factors."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM corporate_actions WHERE id = ?", (action_id,))
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"[corp_actions] Action {action_id} not found")
                    return

                columns = [desc[0] for desc in cursor.description]
                action_data = dict(zip(columns, row))

                action_type = CorporateActionType(action_data['action_type'])
                symbol = action_data['symbol']
                ex_date = action_data['ex_date']

                # Calculate adjustment factors based on action type
                price_factor, volume_factor = self._calculate_adjustment_factors(action_type, action_data)

                if price_factor != 1.0 or volume_factor != 1.0:
                    # Insert adjustment factor
                    cursor.execute("""
                        INSERT OR REPLACE INTO adjustment_factors
                        (symbol, date, price_factor, volume_factor, action_type, action_id, created_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        ex_date,
                        price_factor,
                        volume_factor,
                        action_type.value,
                        action_id,
                        datetime.now().isoformat()
                    ))

                    # Mark action as processed
                    cursor.execute("UPDATE corporate_actions SET processed = 1 WHERE id = ?", (action_id,))

                    conn.commit()

                    logger.info(f"[corp_actions] Processed {action_type.value} for {symbol}: "
                              f"price_factor={price_factor:.4f}, volume_factor={volume_factor:.4f}")

                    # Clear cache for this symbol
                    self._clear_symbol_cache(symbol)

        except Exception as e:
            logger.error(f"[corp_actions] Failed to process action {action_id}: {e}")

    def _calculate_adjustment_factors(self, action_type: CorporateActionType,
                                    action_data: Dict) -> Tuple[float, float]:
        """Calculate price and volume adjustment factors for a corporate action."""
        try:
            price_factor = 1.0
            volume_factor = 1.0

            if action_type in [CorporateActionType.STOCK_SPLIT, CorporateActionType.REVERSE_SPLIT]:
                ratio = action_data.get('ratio', 1.0)
                if ratio and ratio > 0:
                    price_factor = 1.0 / ratio  # For 2:1 split, ratio=2.0, price_factor=0.5
                    volume_factor = ratio  # Volume scales inversely

            elif action_type == CorporateActionType.DIVIDEND:
                # For cash dividends, adjust by dividend amount
                amount = action_data.get('amount', 0.0)
                if amount and amount > 0:
                    # This is a simplified adjustment - in practice, you'd need the stock price
                    # For now, we don't adjust for cash dividends as they don't affect price continuity
                    pass

            elif action_type == CorporateActionType.STOCK_DIVIDEND:
                # Stock dividend is similar to a split
                ratio = action_data.get('ratio', 0.0)
                if ratio and ratio > 0:
                    adjustment_ratio = 1.0 + ratio  # 10% stock dividend means 1.10 ratio
                    price_factor = 1.0 / adjustment_ratio
                    volume_factor = adjustment_ratio

            elif action_type == CorporateActionType.SPINOFF:
                # Spinoffs typically require manual adjustment based on the distribution ratio
                ratio = action_data.get('ratio', 1.0)
                if ratio and ratio > 0:
                    price_factor = 1.0 / (1.0 + ratio)  # Approximate adjustment

            # For other action types (mergers, symbol changes, etc.), no automatic price adjustment
            # These often require manual handling based on specific terms

            return price_factor, volume_factor

        except Exception as e:
            logger.error(f"[corp_actions] Factor calculation failed: {e}")
            return 1.0, 1.0

    def _clear_symbol_cache(self, symbol: str):
        """Clear cached adjustment factors for a symbol."""
        keys_to_remove = [key for key in self.adjustment_factors_cache.keys() if key.startswith(f"{symbol}_")]
        for key in keys_to_remove:
            del self.adjustment_factors_cache[key]

    def get_symbol_mapping(self, old_symbol: str, as_of_date: str) -> Optional[str]:
        """
        Get current symbol for a historical symbol as of a specific date.

        Args:
            old_symbol: Historical symbol
            as_of_date: Date to check mapping

        Returns:
            Current symbol or None if no mapping found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT new_symbol FROM corporate_actions
                    WHERE symbol = ? AND action_type = ? AND ex_date <= ?
                    ORDER BY ex_date DESC
                    LIMIT 1
                """, (old_symbol, CorporateActionType.SYMBOL_CHANGE.value, as_of_date))

                result = cursor.fetchone()
                return result[0] if result else old_symbol

        except Exception as e:
            logger.error(f"[corp_actions] Symbol mapping lookup failed: {e}")
            return old_symbol

    def validate_adjustments(self, symbol: str, original_data: pd.DataFrame,
                           adjusted_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that adjustments were applied correctly.

        Args:
            symbol: Stock symbol
            original_data: Original price data
            adjusted_data: Adjusted price data

        Returns:
            Validation report
        """
        try:
            report = {
                'symbol': symbol,
                'validation_date': datetime.now().isoformat(),
                'checks_passed': 0,
                'checks_failed': 0,
                'issues': [],
                'summary': {}
            }

            # Check data length consistency
            if len(original_data) != len(adjusted_data):
                report['issues'].append("Data length mismatch")
                report['checks_failed'] += 1
            else:
                report['checks_passed'] += 1

            # Check for negative prices
            if 'close' in adjusted_data.columns:
                negative_prices = (adjusted_data['close'] <= 0).sum()
                if negative_prices > 0:
                    report['issues'].append(f"Found {negative_prices} negative prices")
                    report['checks_failed'] += 1
                else:
                    report['checks_passed'] += 1

            # Check price continuity (no excessive gaps)
            if 'close' in adjusted_data.columns and len(adjusted_data) > 1:
                price_changes = adjusted_data['close'].pct_change().abs()
                excessive_changes = (price_changes > 0.5).sum()  # >50% changes
                if excessive_changes > 0:
                    report['issues'].append(f"Found {excessive_changes} excessive price changes (>50%)")
                    report['checks_failed'] += 1
                else:
                    report['checks_passed'] += 1

            # Summary statistics
            if 'close' in adjusted_data.columns:
                report['summary'] = {
                    'original_price_range': [original_data['close'].min(), original_data['close'].max()],
                    'adjusted_price_range': [adjusted_data['close'].min(), adjusted_data['close'].max()],
                    'total_adjustment_factor': (adjusted_data['close'].iloc[-1] / original_data['close'].iloc[-1]
                                              if len(original_data) > 0 and original_data['close'].iloc[-1] != 0 else 1.0)
                }

            report['overall_status'] = 'PASS' if report['checks_failed'] == 0 else 'FAIL'

            return report

        except Exception as e:
            logger.error(f"[corp_actions] Validation failed: {e}")
            return {'error': str(e)}

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of corporate actions processing status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total actions count
                cursor.execute("SELECT COUNT(*) FROM corporate_actions")
                total_actions = cursor.fetchone()[0]

                # Processed actions count
                cursor.execute("SELECT COUNT(*) FROM corporate_actions WHERE processed = 1")
                processed_actions = cursor.fetchone()[0]

                # Actions by type
                cursor.execute("""
                    SELECT action_type, COUNT(*)
                    FROM corporate_actions
                    GROUP BY action_type
                """)
                actions_by_type = dict(cursor.fetchall())

                # Recent actions (last 30 days)
                thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT COUNT(*) FROM corporate_actions
                    WHERE created_date >= ?
                """, (thirty_days_ago,))
                recent_actions = cursor.fetchone()[0]

                # Unique symbols with actions
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM corporate_actions")
                symbols_with_actions = cursor.fetchone()[0]

                return {
                    'total_actions': total_actions,
                    'processed_actions': processed_actions,
                    'pending_actions': total_actions - processed_actions,
                    'processing_rate': (processed_actions / total_actions * 100) if total_actions > 0 else 0,
                    'actions_by_type': actions_by_type,
                    'recent_actions_30d': recent_actions,
                    'symbols_with_actions': symbols_with_actions,
                    'cache_size': len(self.adjustment_factors_cache),
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"[corp_actions] Failed to get processing summary: {e}")
            return {'error': str(e)}


def create_corporate_actions_processor(data_dir: Optional[str] = None) -> CorporateActionsProcessor:
    """
    Create and configure a corporate actions processor.

    Args:
        data_dir: Custom data directory

    Returns:
        Configured CorporateActionsProcessor instance
    """
    return CorporateActionsProcessor(data_dir or "data_cache/corporate_actions")


if __name__ == "__main__":
    # Test corporate actions processing
    print("=== Corporate Actions Processor Test ===")

    # Create processor
    processor = create_corporate_actions_processor("test_corporate_actions")

    # Add sample stock split
    split_action = CorporateAction(
        symbol="AAPL",
        action_type=CorporateActionType.STOCK_SPLIT,
        ex_date="2024-06-10",
        ratio=4.0,  # 4:1 split
        description="4-for-1 stock split",
        source="test"
    )

    processor.add_corporate_action(split_action)

    # Add sample dividend
    dividend_action = CorporateAction(
        symbol="AAPL",
        action_type=CorporateActionType.DIVIDEND,
        ex_date="2024-05-10",
        amount=0.95,
        description="Quarterly dividend",
        source="test"
    )

    processor.add_corporate_action(dividend_action)

    # Create sample price data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'open': 150.0 + np.random.randn(len(dates)) * 2,
        'high': 155.0 + np.random.randn(len(dates)) * 2,
        'low': 145.0 + np.random.randn(len(dates)) * 2,
        'close': 150.0 + np.random.randn(len(dates)) * 2,
        'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
    }, index=dates)

    # Apply adjustments
    adjusted_data = processor.adjust_price_series("AAPL", sample_data)

    print(f"Original price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    print(f"Adjusted price range: ${adjusted_data['close'].min():.2f} - ${adjusted_data['close'].max():.2f}")

    # Get actions
    actions = processor.get_actions_for_symbol("AAPL")
    print(f"Found {len(actions)} corporate actions for AAPL")

    # Validation
    validation = processor.validate_adjustments("AAPL", sample_data, adjusted_data)
    print(f"Validation status: {validation.get('overall_status', 'UNKNOWN')}")

    # Processing summary
    summary = processor.get_processing_summary()
    print(f"Processing summary: {summary['processed_actions']}/{summary['total_actions']} actions processed")