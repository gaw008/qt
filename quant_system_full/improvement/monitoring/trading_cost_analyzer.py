"""
Trading Cost Analysis (TCA) System

This module provides comprehensive transaction cost analysis for trading operations,
including slippage analysis, execution quality measurement, and cost attribution.

Key Features:
- Real-time and post-trade TCA analysis
- Slippage measurement and attribution
- Market impact analysis
- Execution quality scoring
- Benchmark comparison (VWAP, TWAP, Arrival Price)
- Cost breakdown by trading session
- Performance vs. ADV analysis
- Latency and fill rate monitoring
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


class BenchmarkType(Enum):
    """TCA benchmark types."""
    ARRIVAL_PRICE = "arrival_price"
    VWAP = "vwap"
    TWAP = "twap"
    CLOSE_PRICE = "close_price"
    OPEN_PRICE = "open_price"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"


class ExecutionQuality(Enum):
    """Execution quality ratings."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class TradeExecution:
    """Individual trade execution record."""
    trade_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    timestamp: str

    # Order details
    order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    order_type: str = "market"

    # Execution context
    session_type: str = "regular"  # pre_market, regular, post_market
    market_phase: str = "open"
    venue: str = "primary"

    # Market data at execution
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    spread_bps: Optional[float] = None
    volume_at_time: Optional[int] = None


@dataclass
class TCAAnalysis:
    """TCA analysis results for a trade or order."""
    trade_id: str
    symbol: str
    analysis_type: str  # trade, order, basket
    benchmark_type: BenchmarkType

    # Core TCA metrics
    slippage_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    total_cost_bps: float

    # Execution quality
    execution_quality: ExecutionQuality
    quality_score: float  # 0-100

    # Benchmark comparisons
    benchmark_price: float
    execution_price: float
    price_improvement_bps: float

    # Volume analysis
    participation_rate: float  # % of ADV
    fill_rate: float  # % of order filled

    # Timing analysis
    submission_to_fill_seconds: float
    market_impact_duration_minutes: float

    # Attribution
    cost_attribution: Dict[str, float]  # Breakdown of costs

    # Analysis metadata
    analysis_timestamp: str = ""
    confidence_level: float = 1.0

    def __post_init__(self):
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now().isoformat()


@dataclass
class TCAConfig:
    """Configuration for TCA analysis."""
    # Benchmark settings
    default_benchmark: BenchmarkType = BenchmarkType.VWAP
    vwap_window_minutes: int = 60
    twap_window_minutes: int = 30

    # Market impact parameters
    impact_decay_half_life_minutes: float = 10.0
    min_impact_threshold_bps: float = 1.0

    # ADV calculation
    adv_lookback_days: int = 20
    min_adv_dollars: float = 100000  # Minimum ADV for reliable analysis

    # Quality thresholds (basis points)
    excellent_threshold_bps: float = 5.0
    good_threshold_bps: float = 15.0
    fair_threshold_bps: float = 30.0
    poor_threshold_bps: float = 50.0

    # Analysis parameters
    outlier_threshold_std: float = 3.0
    min_trades_for_analysis: int = 5
    confidence_level: float = 0.95


class TradingCostAnalyzer:
    """
    Comprehensive trading cost analysis system.
    """

    def __init__(self, config: Optional[TCAConfig] = None,
                 data_dir: str = "data_cache/tca"):
        """
        Initialize TCA analyzer.

        Args:
            config: TCA configuration
            data_dir: Directory for storing TCA data
        """
        self.config = config or TCAConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Database for storing TCA data
        self.db_path = self.data_dir / "tca_analysis.db"

        # Analysis cache
        self.analysis_cache: Dict[str, TCAAnalysis] = {}
        self.market_data_cache: Dict[str, pd.DataFrame] = {}

        # Initialize database
        self._init_database()

        logger.info(f"[tca] Analyzer initialized with data dir: {self.data_dir}")

    def _init_database(self):
        """Initialize SQLite database for TCA data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Trade executions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        order_id TEXT,
                        parent_order_id TEXT,
                        order_type TEXT,
                        session_type TEXT,
                        market_phase TEXT,
                        venue TEXT,
                        bid_price REAL,
                        ask_price REAL,
                        spread_bps REAL,
                        volume_at_time INTEGER,
                        created_at TEXT NOT NULL,
                        UNIQUE(trade_id)
                    )
                """)

                # TCA analysis table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tca_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        benchmark_type TEXT NOT NULL,
                        slippage_bps REAL,
                        market_impact_bps REAL,
                        timing_cost_bps REAL,
                        total_cost_bps REAL,
                        execution_quality TEXT,
                        quality_score REAL,
                        benchmark_price REAL,
                        execution_price REAL,
                        price_improvement_bps REAL,
                        participation_rate REAL,
                        fill_rate REAL,
                        submission_to_fill_seconds REAL,
                        market_impact_duration_minutes REAL,
                        cost_attribution TEXT,
                        analysis_timestamp TEXT NOT NULL,
                        confidence_level REAL DEFAULT 1.0,
                        UNIQUE(trade_id, benchmark_type)
                    )
                """)

                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_executions_symbol_time ON trade_executions(symbol, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_symbol_time ON tca_analysis(symbol, analysis_timestamp)")

                conn.commit()

            logger.info("[tca] Database initialized successfully")

        except Exception as e:
            logger.error(f"[tca] Database initialization failed: {e}")
            raise

    def add_trade_execution(self, execution: TradeExecution) -> bool:
        """
        Add a trade execution for TCA analysis.

        Args:
            execution: Trade execution record

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trade_executions
                    (trade_id, symbol, side, quantity, price, timestamp,
                     order_id, parent_order_id, order_type, session_type, market_phase, venue,
                     bid_price, ask_price, spread_bps, volume_at_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution.trade_id, execution.symbol, execution.side,
                    execution.quantity, execution.price, execution.timestamp,
                    execution.order_id, execution.parent_order_id, execution.order_type,
                    execution.session_type, execution.market_phase, execution.venue,
                    execution.bid_price, execution.ask_price, execution.spread_bps,
                    execution.volume_at_time, datetime.now().isoformat()
                ))

                conn.commit()

            logger.info(f"[tca] Added trade execution: {execution.trade_id}")
            return True

        except Exception as e:
            logger.error(f"[tca] Failed to add trade execution: {e}")
            return False

    def analyze_trade(self, trade_id: str,
                     benchmark_type: Optional[BenchmarkType] = None,
                     market_data: Optional[pd.DataFrame] = None) -> TCAAnalysis:
        """
        Perform TCA analysis for a specific trade.

        Args:
            trade_id: Trade ID to analyze
            benchmark_type: Benchmark for comparison
            market_data: Market data for analysis

        Returns:
            TCA analysis results
        """
        try:
            benchmark_type = benchmark_type or self.config.default_benchmark

            # Get trade execution data
            execution = self._get_trade_execution(trade_id)
            if not execution:
                raise ValueError(f"Trade {trade_id} not found")

            # Get market data
            if market_data is None:
                market_data = self._get_market_data(execution.symbol, execution.timestamp)

            # Calculate benchmark price
            benchmark_price = self._calculate_benchmark_price(
                execution, benchmark_type, market_data
            )

            # Calculate slippage
            slippage_bps = self._calculate_slippage(execution, benchmark_price)

            # Calculate market impact
            market_impact_bps = self._calculate_market_impact(execution, market_data)

            # Calculate timing cost
            timing_cost_bps = self._calculate_timing_cost(execution, market_data)

            # Total cost
            total_cost_bps = slippage_bps + market_impact_bps + timing_cost_bps

            # Execution quality assessment
            quality_score, execution_quality = self._assess_execution_quality(total_cost_bps)

            # Price improvement
            price_improvement_bps = -slippage_bps  # Negative slippage = improvement

            # Volume analysis
            participation_rate = self._calculate_participation_rate(execution, market_data)
            fill_rate = 1.0  # Assume full fill for individual trades

            # Timing analysis
            submission_to_fill_seconds = 0.0  # Would need order submission time
            market_impact_duration_minutes = self.config.impact_decay_half_life_minutes

            # Cost attribution
            cost_attribution = {
                'slippage': slippage_bps,
                'market_impact': market_impact_bps,
                'timing_cost': timing_cost_bps,
                'spread_cost': execution.spread_bps or 0.0
            }

            # Create analysis result
            analysis = TCAAnalysis(
                trade_id=trade_id,
                symbol=execution.symbol,
                analysis_type="trade",
                benchmark_type=benchmark_type,
                slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps,
                timing_cost_bps=timing_cost_bps,
                total_cost_bps=total_cost_bps,
                execution_quality=execution_quality,
                quality_score=quality_score,
                benchmark_price=benchmark_price,
                execution_price=execution.price,
                price_improvement_bps=price_improvement_bps,
                participation_rate=participation_rate,
                fill_rate=fill_rate,
                submission_to_fill_seconds=submission_to_fill_seconds,
                market_impact_duration_minutes=market_impact_duration_minutes,
                cost_attribution=cost_attribution
            )

            # Store analysis
            self._store_tca_analysis(analysis)

            # Cache result
            self.analysis_cache[f"{trade_id}_{benchmark_type.value}"] = analysis

            logger.info(f"[tca] Trade analysis complete: {trade_id} - "
                       f"{execution_quality.value} ({total_cost_bps:.1f}bps)")

            return analysis

        except Exception as e:
            logger.error(f"[tca] Trade analysis failed for {trade_id}: {e}")
            raise

    def analyze_basket(self, trade_ids: List[str],
                      benchmark_type: Optional[BenchmarkType] = None) -> Dict[str, Any]:
        """
        Perform TCA analysis for a basket of trades.

        Args:
            trade_ids: List of trade IDs
            benchmark_type: Benchmark for comparison

        Returns:
            Basket TCA analysis results
        """
        try:
            benchmark_type = benchmark_type or self.config.default_benchmark

            # Analyze individual trades
            trade_analyses = []
            for trade_id in trade_ids:
                try:
                    analysis = self.analyze_trade(trade_id, benchmark_type)
                    trade_analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"[tca] Failed to analyze trade {trade_id}: {e}")
                    continue

            if not trade_analyses:
                raise ValueError("No valid trade analyses found")

            # Calculate basket-level metrics
            total_notional = sum(
                abs(analysis.execution_price * self._get_trade_quantity(analysis.trade_id))
                for analysis in trade_analyses
            )

            # Weighted average metrics
            weighted_slippage = sum(
                analysis.slippage_bps * abs(analysis.execution_price * self._get_trade_quantity(analysis.trade_id))
                for analysis in trade_analyses
            ) / total_notional

            weighted_market_impact = sum(
                analysis.market_impact_bps * abs(analysis.execution_price * self._get_trade_quantity(analysis.trade_id))
                for analysis in trade_analyses
            ) / total_notional

            weighted_total_cost = sum(
                analysis.total_cost_bps * abs(analysis.execution_price * self._get_trade_quantity(analysis.trade_id))
                for analysis in trade_analyses
            ) / total_notional

            # Quality distribution
            quality_distribution = {}
            for quality in ExecutionQuality:
                count = len([a for a in trade_analyses if a.execution_quality == quality])
                quality_distribution[quality.value] = count / len(trade_analyses)

            # Summary statistics
            basket_summary = {
                'basket_id': f"basket_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'trade_count': len(trade_analyses),
                'total_notional': total_notional,
                'benchmark_type': benchmark_type.value,
                'weighted_slippage_bps': weighted_slippage,
                'weighted_market_impact_bps': weighted_market_impact,
                'weighted_total_cost_bps': weighted_total_cost,
                'average_quality_score': np.mean([a.quality_score for a in trade_analyses]),
                'quality_distribution': quality_distribution,
                'cost_range_bps': {
                    'min': min(a.total_cost_bps for a in trade_analyses),
                    'max': max(a.total_cost_bps for a in trade_analyses),
                    'std': np.std([a.total_cost_bps for a in trade_analyses])
                },
                'participation_stats': {
                    'average': np.mean([a.participation_rate for a in trade_analyses]),
                    'max': max(a.participation_rate for a in trade_analyses)
                },
                'individual_analyses': [asdict(analysis) for analysis in trade_analyses],
                'analysis_timestamp': datetime.now().isoformat()
            }

            return basket_summary

        except Exception as e:
            logger.error(f"[tca] Basket analysis failed: {e}")
            raise

    def _get_trade_execution(self, trade_id: str) -> Optional[TradeExecution]:
        """Get trade execution from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trade_executions WHERE trade_id = ?", (trade_id,))
                row = cursor.fetchone()

                if row:
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, row))

                    return TradeExecution(
                        trade_id=row_dict['trade_id'],
                        symbol=row_dict['symbol'],
                        side=row_dict['side'],
                        quantity=row_dict['quantity'],
                        price=row_dict['price'],
                        timestamp=row_dict['timestamp'],
                        order_id=row_dict['order_id'],
                        parent_order_id=row_dict['parent_order_id'],
                        order_type=row_dict['order_type'] or "market",
                        session_type=row_dict['session_type'] or "regular",
                        market_phase=row_dict['market_phase'] or "open",
                        venue=row_dict['venue'] or "primary",
                        bid_price=row_dict['bid_price'],
                        ask_price=row_dict['ask_price'],
                        spread_bps=row_dict['spread_bps'],
                        volume_at_time=row_dict['volume_at_time']
                    )

                return None

        except Exception as e:
            logger.error(f"[tca] Failed to get trade execution: {e}")
            return None

    def _get_trade_quantity(self, trade_id: str) -> float:
        """Get trade quantity from database."""
        try:
            execution = self._get_trade_execution(trade_id)
            return execution.quantity if execution else 0.0
        except Exception:
            return 0.0

    def _get_market_data(self, symbol: str, timestamp: str) -> pd.DataFrame:
        """Get market data around trade time."""
        try:
            # In a real implementation, this would fetch market data from your data source
            # For now, return a mock DataFrame
            trade_time = pd.to_datetime(timestamp)
            start_time = trade_time - timedelta(hours=2)
            end_time = trade_time + timedelta(hours=2)

            # Mock market data
            dates = pd.date_range(start=start_time, end=end_time, freq='1min')
            mock_data = pd.DataFrame({
                'close': 100.0 + np.random.randn(len(dates)) * 0.5,
                'volume': 10000 + np.random.randint(-1000, 1000, len(dates)),
                'vwap': 100.0 + np.random.randn(len(dates)) * 0.3
            }, index=dates)

            return mock_data

        except Exception as e:
            logger.error(f"[tca] Failed to get market data: {e}")
            return pd.DataFrame()

    def _calculate_benchmark_price(self, execution: TradeExecution,
                                 benchmark_type: BenchmarkType,
                                 market_data: pd.DataFrame) -> float:
        """Calculate benchmark price based on type."""
        try:
            trade_time = pd.to_datetime(execution.timestamp)

            if benchmark_type == BenchmarkType.ARRIVAL_PRICE:
                # Use the price at order arrival (approximate with execution price)
                return execution.price

            elif benchmark_type == BenchmarkType.VWAP:
                # Calculate VWAP around trade time
                window_start = trade_time - timedelta(minutes=self.config.vwap_window_minutes // 2)
                window_end = trade_time + timedelta(minutes=self.config.vwap_window_minutes // 2)

                window_data = market_data.loc[window_start:window_end]
                if 'vwap' in window_data.columns and not window_data.empty:
                    return window_data['vwap'].mean()
                elif 'close' in window_data.columns and not window_data.empty:
                    # Approximate VWAP with volume-weighted average of closes
                    if 'volume' in window_data.columns:
                        total_value = (window_data['close'] * window_data['volume']).sum()
                        total_volume = window_data['volume'].sum()
                        return total_value / total_volume if total_volume > 0 else window_data['close'].mean()
                    else:
                        return window_data['close'].mean()

            elif benchmark_type == BenchmarkType.TWAP:
                # Calculate TWAP around trade time
                window_start = trade_time - timedelta(minutes=self.config.twap_window_minutes // 2)
                window_end = trade_time + timedelta(minutes=self.config.twap_window_minutes // 2)

                window_data = market_data.loc[window_start:window_end]
                if 'close' in window_data.columns and not window_data.empty:
                    return window_data['close'].mean()

            elif benchmark_type == BenchmarkType.CLOSE_PRICE:
                # Use close price of the trading day
                trade_date = trade_time.date()
                day_data = market_data[market_data.index.date == trade_date]
                if 'close' in day_data.columns and not day_data.empty:
                    return day_data['close'].iloc[-1]

            # Fallback to execution price
            return execution.price

        except Exception as e:
            logger.error(f"[tca] Benchmark calculation failed: {e}")
            return execution.price

    def _calculate_slippage(self, execution: TradeExecution, benchmark_price: float) -> float:
        """Calculate slippage in basis points."""
        try:
            if benchmark_price <= 0:
                return 0.0

            # Slippage = (execution_price - benchmark_price) / benchmark_price * 10000
            # Positive for buy orders means paying more (worse)
            # Negative for sell orders means receiving less (worse)
            price_diff = execution.price - benchmark_price

            if execution.side.lower() == 'sell':
                price_diff = -price_diff  # Flip sign for sell orders

            slippage_bps = (price_diff / benchmark_price) * 10000
            return slippage_bps

        except Exception as e:
            logger.error(f"[tca] Slippage calculation failed: {e}")
            return 0.0

    def _calculate_market_impact(self, execution: TradeExecution,
                               market_data: pd.DataFrame) -> float:
        """Calculate market impact in basis points."""
        try:
            # Simplified market impact calculation
            # In practice, this would use more sophisticated models

            trade_time = pd.to_datetime(execution.timestamp)

            # Get price before and after trade
            pre_trade_window = market_data.loc[:trade_time - timedelta(minutes=1)]
            post_trade_window = market_data.loc[trade_time + timedelta(minutes=1):]

            if pre_trade_window.empty or post_trade_window.empty:
                return 0.0

            pre_price = pre_trade_window['close'].iloc[-1] if 'close' in pre_trade_window.columns else execution.price

            # Look at price movement in the impact decay period
            impact_end_time = trade_time + timedelta(minutes=self.config.impact_decay_half_life_minutes)
            impact_window = market_data.loc[trade_time:impact_end_time]

            if impact_window.empty:
                return 0.0

            # Calculate impact as the price movement in the direction of the trade
            post_price = impact_window['close'].mean() if 'close' in impact_window.columns else execution.price

            price_move = (post_price - pre_price) / pre_price * 10000

            # Impact is positive if price moved in unfavorable direction
            if execution.side.lower() == 'buy' and price_move > 0:
                return min(price_move, 50.0)  # Cap at 50bps
            elif execution.side.lower() == 'sell' and price_move < 0:
                return min(-price_move, 50.0)  # Cap at 50bps

            return max(0.0, abs(price_move) * 0.1)  # Minimal impact if favorable

        except Exception as e:
            logger.error(f"[tca] Market impact calculation failed: {e}")
            return 0.0

    def _calculate_timing_cost(self, execution: TradeExecution,
                             market_data: pd.DataFrame) -> float:
        """Calculate timing cost in basis points."""
        try:
            # Timing cost represents the cost of delayed execution
            # This is a simplified implementation

            if execution.session_type != "regular":
                # Higher timing cost for off-hours trading
                return 5.0

            # Check if execution was during high volatility period
            trade_time = pd.to_datetime(execution.timestamp)
            volatility_window = market_data.loc[trade_time - timedelta(minutes=30):trade_time + timedelta(minutes=30)]

            if 'close' in volatility_window.columns and len(volatility_window) > 1:
                returns = volatility_window['close'].pct_change()
                volatility = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized volatility

                # Higher timing cost during high volatility
                if volatility > 0.5:  # 50% annualized volatility
                    return 10.0
                elif volatility > 0.3:  # 30% annualized volatility
                    return 5.0

            return 1.0  # Base timing cost

        except Exception as e:
            logger.error(f"[tca] Timing cost calculation failed: {e}")
            return 0.0

    def _calculate_participation_rate(self, execution: TradeExecution,
                                    market_data: pd.DataFrame) -> float:
        """Calculate participation rate as % of ADV."""
        try:
            # Calculate ADV (Average Daily Volume)
            trade_date = pd.to_datetime(execution.timestamp).date()

            # Get historical volume data (would be from your data source)
            # For now, use mock calculation
            adv_dollars = 1000000  # Mock ADV of $1M

            trade_value = abs(execution.quantity * execution.price)
            participation_rate = trade_value / adv_dollars

            return min(participation_rate, 1.0)  # Cap at 100%

        except Exception as e:
            logger.error(f"[tca] Participation rate calculation failed: {e}")
            return 0.0

    def _assess_execution_quality(self, total_cost_bps: float) -> Tuple[float, ExecutionQuality]:
        """Assess execution quality based on total cost."""
        try:
            # Convert cost to quality score (0-100)
            if total_cost_bps <= self.config.excellent_threshold_bps:
                quality = ExecutionQuality.EXCELLENT
                score = 100 - (total_cost_bps / self.config.excellent_threshold_bps) * 10
            elif total_cost_bps <= self.config.good_threshold_bps:
                quality = ExecutionQuality.GOOD
                score = 90 - ((total_cost_bps - self.config.excellent_threshold_bps) /
                             (self.config.good_threshold_bps - self.config.excellent_threshold_bps)) * 20
            elif total_cost_bps <= self.config.fair_threshold_bps:
                quality = ExecutionQuality.FAIR
                score = 70 - ((total_cost_bps - self.config.good_threshold_bps) /
                             (self.config.fair_threshold_bps - self.config.good_threshold_bps)) * 20
            elif total_cost_bps <= self.config.poor_threshold_bps:
                quality = ExecutionQuality.POOR
                score = 50 - ((total_cost_bps - self.config.fair_threshold_bps) /
                             (self.config.poor_threshold_bps - self.config.fair_threshold_bps)) * 30
            else:
                quality = ExecutionQuality.VERY_POOR
                score = max(0, 20 - (total_cost_bps - self.config.poor_threshold_bps))

            return max(0.0, min(100.0, score)), quality

        except Exception as e:
            logger.error(f"[tca] Quality assessment failed: {e}")
            return 0.0, ExecutionQuality.VERY_POOR

    def _store_tca_analysis(self, analysis: TCAAnalysis):
        """Store TCA analysis in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tca_analysis
                    (trade_id, symbol, analysis_type, benchmark_type,
                     slippage_bps, market_impact_bps, timing_cost_bps, total_cost_bps,
                     execution_quality, quality_score, benchmark_price, execution_price,
                     price_improvement_bps, participation_rate, fill_rate,
                     submission_to_fill_seconds, market_impact_duration_minutes,
                     cost_attribution, analysis_timestamp, confidence_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.trade_id, analysis.symbol, analysis.analysis_type, analysis.benchmark_type.value,
                    analysis.slippage_bps, analysis.market_impact_bps, analysis.timing_cost_bps, analysis.total_cost_bps,
                    analysis.execution_quality.value, analysis.quality_score,
                    analysis.benchmark_price, analysis.execution_price, analysis.price_improvement_bps,
                    analysis.participation_rate, analysis.fill_rate,
                    analysis.submission_to_fill_seconds, analysis.market_impact_duration_minutes,
                    json.dumps(analysis.cost_attribution), analysis.analysis_timestamp, analysis.confidence_level
                ))

        except Exception as e:
            logger.error(f"[tca] Failed to store TCA analysis: {e}")

    def get_tca_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get TCA summary statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

                cursor = conn.cursor()

                # Overall statistics
                cursor.execute("""
                    SELECT COUNT(*) as total_analyses,
                           AVG(total_cost_bps) as avg_cost_bps,
                           AVG(slippage_bps) as avg_slippage_bps,
                           AVG(market_impact_bps) as avg_impact_bps,
                           AVG(quality_score) as avg_quality_score,
                           COUNT(DISTINCT symbol) as unique_symbols
                    FROM tca_analysis
                    WHERE analysis_timestamp >= ?
                """, (cutoff_date,))

                stats = cursor.fetchone()

                # Quality distribution
                cursor.execute("""
                    SELECT execution_quality, COUNT(*) as count
                    FROM tca_analysis
                    WHERE analysis_timestamp >= ?
                    GROUP BY execution_quality
                """, (cutoff_date,))

                quality_dist = dict(cursor.fetchall())

                # Cost breakdown by benchmark
                cursor.execute("""
                    SELECT benchmark_type, AVG(total_cost_bps) as avg_cost
                    FROM tca_analysis
                    WHERE analysis_timestamp >= ?
                    GROUP BY benchmark_type
                """, (cutoff_date,))

                cost_by_benchmark = dict(cursor.fetchall())

                return {
                    'period_days': days_back,
                    'total_analyses': stats[0] if stats else 0,
                    'average_cost_bps': stats[1] if stats and stats[1] else 0,
                    'average_slippage_bps': stats[2] if stats and stats[2] else 0,
                    'average_market_impact_bps': stats[3] if stats and stats[3] else 0,
                    'average_quality_score': stats[4] if stats and stats[4] else 0,
                    'unique_symbols': stats[5] if stats else 0,
                    'quality_distribution': quality_dist,
                    'cost_by_benchmark': cost_by_benchmark,
                    'cache_size': len(self.analysis_cache),
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"[tca] Failed to get TCA summary: {e}")
            return {'error': str(e)}


def create_trading_cost_analyzer(custom_config: Optional[Dict] = None,
                               data_dir: Optional[str] = None) -> TradingCostAnalyzer:
    """
    Create and configure a trading cost analyzer.

    Args:
        custom_config: Custom configuration parameters
        data_dir: Custom data directory

    Returns:
        Configured TradingCostAnalyzer instance
    """
    config = TCAConfig()

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return TradingCostAnalyzer(config, data_dir or "data_cache/tca")


if __name__ == "__main__":
    # Test TCA analyzer
    print("=== Trading Cost Analyzer Test ===")

    # Create analyzer
    analyzer = create_trading_cost_analyzer()

    # Create sample trade execution
    execution = TradeExecution(
        trade_id="TEST_001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.25,
        timestamp=datetime.now().isoformat(),
        order_type="market",
        session_type="regular",
        bid_price=150.20,
        ask_price=150.30,
        spread_bps=6.67
    )

    # Add execution
    analyzer.add_trade_execution(execution)

    # Analyze trade
    analysis = analyzer.analyze_trade("TEST_001", BenchmarkType.VWAP)

    print(f"TCA Analysis for {execution.symbol}:")
    print(f"  Execution Price: ${analysis.execution_price:.2f}")
    print(f"  Benchmark Price: ${analysis.benchmark_price:.2f}")
    print(f"  Total Cost: {analysis.total_cost_bps:.1f}bps")
    print(f"  Slippage: {analysis.slippage_bps:.1f}bps")
    print(f"  Market Impact: {analysis.market_impact_bps:.1f}bps")
    print(f"  Execution Quality: {analysis.execution_quality.value}")
    print(f"  Quality Score: {analysis.quality_score:.1f}")

    # Get summary
    summary = analyzer.get_tca_summary()
    print(f"\nTCA Summary:")
    print(f"  Total Analyses: {summary['total_analyses']}")
    print(f"  Average Cost: {summary['average_cost_bps']:.1f}bps")
    print(f"  Average Quality Score: {summary['average_quality_score']:.1f}")