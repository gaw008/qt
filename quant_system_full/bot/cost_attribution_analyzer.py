"""
Advanced Cost Attribution Analyzer
高级成本归因分析器

Comprehensive transaction cost analysis with detailed attribution:
- Pre-trade cost estimation and optimization
- Real-time execution cost tracking
- Post-trade cost attribution and analysis
- Benchmark comparison and performance measurement
- Capacity impact analysis

Integrates with Adaptive Execution Engine and Transaction Cost Analyzer.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CostComponent(Enum):
    """Transaction cost components"""
    SPREAD = "spread"
    MARKET_IMPACT = "market_impact"
    TIMING = "timing"
    OPPORTUNITY = "opportunity"
    COMMISSION = "commission"
    FEES = "fees"

class BenchmarkType(Enum):
    """Execution benchmarks"""
    ARRIVAL_PRICE = "arrival"
    VWAP = "vwap"
    TWAP = "twap"
    CLOSE = "close"
    OPEN = "open"
    MIDPOINT = "midpoint"

@dataclass
class CostEstimate:
    """Pre-trade cost estimate"""
    symbol: str
    side: str
    quantity: int
    estimated_price: float

    # Cost components (in basis points)
    spread_cost_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    total_cost_bps: float

    # Market condition inputs
    market_regime: str
    volatility: float
    liquidity_score: float

    # Confidence intervals
    cost_lower_bound: float
    cost_upper_bound: float
    confidence_level: float = 0.95

@dataclass
class ExecutionCostBreakdown:
    """Detailed execution cost breakdown"""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    executed_quantity: int

    # Price points
    arrival_price: float
    average_execution_price: float
    vwap_benchmark: float
    twap_benchmark: float

    # Cost attribution (basis points)
    total_cost_bps: float
    spread_cost_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    opportunity_cost_bps: float
    commission_bps: float

    # Performance vs benchmarks
    vs_arrival_bps: float
    vs_vwap_bps: float
    vs_twap_bps: float

    # Execution metrics
    participation_rate: float
    execution_duration: timedelta
    fill_rate: float

    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CapacityImpact:
    """Analysis of capacity impact on costs"""
    current_aum: float
    trade_size: float

    # Impact scaling
    linear_impact: float
    nonlinear_impact: float
    total_impact_bps: float

    # Capacity metrics
    adv_percentage: float  # % of average daily volume
    capacity_utilization: float  # % of estimated capacity

    # Scaling projections
    projected_costs_10m: float
    projected_costs_50m: float
    projected_costs_100m: float

class CostAttributionAnalyzer:
    """
    Advanced Cost Attribution Analyzer

    Provides comprehensive transaction cost analysis including:
    - Pre-trade cost estimation with confidence intervals
    - Real-time execution monitoring and attribution
    - Post-trade performance analysis vs multiple benchmarks
    - Capacity impact modeling and scaling analysis
    """

    def __init__(self, config_path: str = "config/cost_attribution_config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Historical data for modeling
        self.cost_estimates: List[CostEstimate] = []
        self.execution_costs: List[ExecutionCostBreakdown] = []

        # Market data cache
        self.market_data_cache: Dict[str, Dict] = {}

        # Model parameters (would be fitted from historical data)
        self.cost_model_params = {
            "spread_model": {"alpha": 0.5, "beta": 0.3},
            "impact_model": {"gamma": 0.45, "delta": 0.25},
            "timing_model": {"lambda": 0.15, "mu": 0.10}
        }

        # Database for persistence
        self.db_path = "data_cache/cost_attribution.db"
        self._initialize_database()

        self.logger.info("Cost Attribution Analyzer initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load cost attribution configuration"""
        default_config = {
            "cost_models": {
                "spread_model_type": "regime_dependent",
                "impact_model_type": "square_root",
                "timing_model_type": "momentum_adjusted"
            },
            "benchmarks": {
                "primary": "arrival",
                "secondary": ["vwap", "twap"],
                "intraday_window": 390  # minutes in trading day
            },
            "capacity_analysis": {
                "base_adv_limit": 0.02,  # 2% of ADV
                "nonlinear_threshold": 0.05,  # 5% threshold
                "scaling_exponent": 1.5
            },
            "confidence_intervals": {
                "default_level": 0.95,
                "bootstrap_samples": 1000,
                "estimation_window": 252  # days
            },
            "cost_attribution": {
                "spread_weight": 0.30,
                "impact_weight": 0.45,
                "timing_weight": 0.20,
                "opportunity_weight": 0.05
            }
        }

        try:
            import json
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cost attribution analyzer"""
        logger = logging.getLogger('CostAttribution')
        logger.setLevel(logging.INFO)

        # File handler
        log_path = Path("logs/cost_attribution.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _initialize_database(self):
        """Initialize SQLite database for cost attribution data"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                # Cost estimates table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cost_estimates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        estimated_price REAL NOT NULL,
                        spread_cost_bps REAL NOT NULL,
                        market_impact_bps REAL NOT NULL,
                        timing_cost_bps REAL NOT NULL,
                        total_cost_bps REAL NOT NULL,
                        cost_lower_bound REAL NOT NULL,
                        cost_upper_bound REAL NOT NULL,
                        market_regime TEXT NOT NULL,
                        volatility REAL NOT NULL,
                        liquidity_score REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Execution costs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_costs (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        executed_quantity INTEGER NOT NULL,
                        arrival_price REAL NOT NULL,
                        average_execution_price REAL NOT NULL,
                        vwap_benchmark REAL NOT NULL,
                        twap_benchmark REAL NOT NULL,
                        total_cost_bps REAL NOT NULL,
                        spread_cost_bps REAL NOT NULL,
                        market_impact_bps REAL NOT NULL,
                        timing_cost_bps REAL NOT NULL,
                        opportunity_cost_bps REAL NOT NULL,
                        commission_bps REAL NOT NULL,
                        vs_arrival_bps REAL NOT NULL,
                        vs_vwap_bps REAL NOT NULL,
                        vs_twap_bps REAL NOT NULL,
                        participation_rate REAL NOT NULL,
                        execution_duration_minutes REAL NOT NULL,
                        fill_rate REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Capacity analysis table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS capacity_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        current_aum REAL NOT NULL,
                        trade_size REAL NOT NULL,
                        linear_impact REAL NOT NULL,
                        nonlinear_impact REAL NOT NULL,
                        total_impact_bps REAL NOT NULL,
                        adv_percentage REAL NOT NULL,
                        capacity_utilization REAL NOT NULL,
                        projected_costs_10m REAL NOT NULL,
                        projected_costs_50m REAL NOT NULL,
                        projected_costs_100m REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                conn.commit()

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")

    def estimate_pre_trade_costs(self, symbol: str, side: str, quantity: int,
                                current_price: float, market_data: Dict[str, Any]) -> CostEstimate:
        """Generate pre-trade cost estimate with confidence intervals"""
        try:
            # Extract market condition inputs
            volatility = market_data.get("volatility", 0.25)
            bid_ask_spread = market_data.get("bid_ask_spread", 0.001)
            daily_volume = market_data.get("daily_volume", 1000000)
            market_regime = market_data.get("regime", "normal")

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(daily_volume, bid_ask_spread, volatility)

            # Estimate cost components
            spread_cost = self._estimate_spread_cost(bid_ask_spread, market_regime)
            market_impact = self._estimate_market_impact(quantity, daily_volume, volatility, current_price)
            timing_cost = self._estimate_timing_cost(volatility, market_regime)

            total_cost = spread_cost + market_impact + timing_cost

            # Calculate confidence intervals
            cost_std = self._estimate_cost_uncertainty(symbol, total_cost, market_data)
            confidence_level = self.config["confidence_intervals"]["default_level"]
            z_score = 1.96  # 95% confidence

            cost_lower = total_cost - z_score * cost_std
            cost_upper = total_cost + z_score * cost_std

            estimate = CostEstimate(
                symbol=symbol,
                side=side,
                quantity=quantity,
                estimated_price=current_price,
                spread_cost_bps=spread_cost,
                market_impact_bps=market_impact,
                timing_cost_bps=timing_cost,
                total_cost_bps=total_cost,
                cost_lower_bound=cost_lower,
                cost_upper_bound=cost_upper,
                confidence_level=confidence_level,
                market_regime=market_regime,
                volatility=volatility,
                liquidity_score=liquidity_score
            )

            # Store estimate
            self.cost_estimates.append(estimate)
            self._store_cost_estimate(estimate)

            self.logger.info(f"Pre-trade cost estimate for {symbol}: "
                           f"{total_cost:.1f}bps [{cost_lower:.1f}, {cost_upper:.1f}]")

            return estimate

        except Exception as e:
            self.logger.error(f"Pre-trade cost estimation failed: {e}")
            raise

    def _calculate_liquidity_score(self, daily_volume: float, spread: float, volatility: float) -> float:
        """Calculate composite liquidity score"""
        try:
            # Normalize components (higher is more liquid)
            volume_score = min(1.0, daily_volume / 1000000)  # Normalize to 1M shares
            spread_score = max(0.0, 1.0 - spread * 1000)     # Penalize wide spreads
            vol_score = max(0.0, 1.0 - volatility * 2)       # Penalize high volatility

            # Weighted combination
            liquidity_score = 0.5 * volume_score + 0.3 * spread_score + 0.2 * vol_score

            return np.clip(liquidity_score, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Liquidity score calculation failed: {e}")
            return 0.5  # Default moderate liquidity

    def _estimate_spread_cost(self, bid_ask_spread: float, market_regime: str) -> float:
        """Estimate spread cost component"""
        try:
            # Base spread cost (half-spread)
            base_cost = (bid_ask_spread / 2) * 10000  # Convert to bps

            # Regime adjustments
            regime_multipliers = {
                "normal": 1.0,
                "volatile": 1.3,
                "stressed": 1.8
            }

            spread_cost = base_cost * regime_multipliers.get(market_regime, 1.0)

            return spread_cost

        except Exception as e:
            self.logger.error(f"Spread cost estimation failed: {e}")
            return 5.0  # Default 5 bps

    def _estimate_market_impact(self, quantity: int, daily_volume: float,
                               volatility: float, price: float) -> float:
        """Estimate market impact using square-root model"""
        try:
            # Participation rate
            participation_rate = quantity / daily_volume if daily_volume > 0 else 0.01

            # Square-root impact model: Impact = γ * σ * sqrt(Q/V)
            gamma = self.cost_model_params["impact_model"]["gamma"]
            impact = gamma * volatility * np.sqrt(participation_rate)

            # Convert to basis points
            impact_bps = impact * 10000

            # Apply capacity constraints
            capacity_config = self.config["capacity_analysis"]
            if participation_rate > capacity_config["nonlinear_threshold"]:
                # Nonlinear scaling for large orders
                excess_participation = participation_rate - capacity_config["nonlinear_threshold"]
                nonlinear_penalty = excess_participation ** capacity_config["scaling_exponent"]
                impact_bps *= (1 + nonlinear_penalty)

            return impact_bps

        except Exception as e:
            self.logger.error(f"Market impact estimation failed: {e}")
            return 10.0  # Default 10 bps

    def _estimate_timing_cost(self, volatility: float, market_regime: str) -> float:
        """Estimate timing cost based on volatility and momentum"""
        try:
            # Base timing cost proportional to volatility
            lambda_param = self.cost_model_params["timing_model"]["lambda"]
            base_timing = lambda_param * volatility * 10000  # Convert to bps

            # Regime adjustments for timing
            regime_adjustments = {
                "normal": 1.0,
                "volatile": 1.5,
                "stressed": 2.0
            }

            timing_cost = base_timing * regime_adjustments.get(market_regime, 1.0)

            return timing_cost

        except Exception as e:
            self.logger.error(f"Timing cost estimation failed: {e}")
            return 3.0  # Default 3 bps

    def _estimate_cost_uncertainty(self, symbol: str, base_cost: float,
                                  market_data: Dict[str, Any]) -> float:
        """Estimate uncertainty in cost prediction"""
        try:
            # Base uncertainty as percentage of cost
            base_uncertainty = 0.30  # 30% of cost as base uncertainty

            # Adjust for market conditions
            volatility = market_data.get("volatility", 0.25)
            vol_adjustment = min(2.0, volatility / 0.20)  # Scale with volatility

            # Adjust for liquidity
            liquidity_score = self._calculate_liquidity_score(
                market_data.get("daily_volume", 1000000),
                market_data.get("bid_ask_spread", 0.001),
                volatility
            )
            liquidity_adjustment = 2.0 - liquidity_score  # Lower liquidity = higher uncertainty

            # Final uncertainty
            uncertainty = base_cost * base_uncertainty * vol_adjustment * liquidity_adjustment

            return max(1.0, uncertainty)  # Minimum 1 bps uncertainty

        except Exception as e:
            self.logger.error(f"Cost uncertainty estimation failed: {e}")
            return base_cost * 0.5  # Default 50% uncertainty

    def analyze_execution_costs(self, trade_id: str, execution_data: Dict[str, Any],
                               market_data: Dict[str, Any]) -> ExecutionCostBreakdown:
        """Analyze post-trade execution costs with full attribution"""
        try:
            # Extract execution data
            symbol = execution_data["symbol"]
            side = execution_data["side"]
            quantity = execution_data["quantity"]
            executed_quantity = execution_data.get("executed_quantity", quantity)

            # Price points
            arrival_price = execution_data["arrival_price"]
            avg_execution_price = execution_data["average_execution_price"]

            # Calculate benchmarks
            vwap_benchmark = market_data.get("vwap", arrival_price)
            twap_benchmark = market_data.get("twap", arrival_price)

            # Calculate cost components
            cost_breakdown = self._decompose_execution_costs(
                side, quantity, executed_quantity, arrival_price, avg_execution_price,
                execution_data, market_data
            )

            # Performance vs benchmarks
            vs_arrival = self._calculate_benchmark_performance(
                side, avg_execution_price, arrival_price
            )
            vs_vwap = self._calculate_benchmark_performance(
                side, avg_execution_price, vwap_benchmark
            )
            vs_twap = self._calculate_benchmark_performance(
                side, avg_execution_price, twap_benchmark
            )

            # Execution metrics
            participation_rate = execution_data.get("participation_rate", 0.1)
            execution_duration = execution_data.get("duration", timedelta(minutes=60))
            fill_rate = executed_quantity / quantity if quantity > 0 else 0.0

            breakdown = ExecutionCostBreakdown(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                executed_quantity=executed_quantity,
                arrival_price=arrival_price,
                average_execution_price=avg_execution_price,
                vwap_benchmark=vwap_benchmark,
                twap_benchmark=twap_benchmark,
                total_cost_bps=cost_breakdown["total"],
                spread_cost_bps=cost_breakdown["spread"],
                market_impact_bps=cost_breakdown["impact"],
                timing_cost_bps=cost_breakdown["timing"],
                opportunity_cost_bps=cost_breakdown["opportunity"],
                commission_bps=cost_breakdown["commission"],
                vs_arrival_bps=vs_arrival,
                vs_vwap_bps=vs_vwap,
                vs_twap_bps=vs_twap,
                participation_rate=participation_rate,
                execution_duration=execution_duration,
                fill_rate=fill_rate
            )

            # Store breakdown
            self.execution_costs.append(breakdown)
            self._store_execution_cost(breakdown)

            self.logger.info(f"Execution cost analysis for {trade_id}: "
                           f"Total: {breakdown.total_cost_bps:.1f}bps, "
                           f"vs Arrival: {vs_arrival:.1f}bps")

            return breakdown

        except Exception as e:
            self.logger.error(f"Execution cost analysis failed: {e}")
            raise

    def _decompose_execution_costs(self, side: str, quantity: int, executed_quantity: int,
                                  arrival_price: float, avg_execution_price: float,
                                  execution_data: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Dict[str, float]:
        """Decompose total execution cost into components"""
        try:
            # Total cost vs arrival price
            if side == "buy":
                total_cost = avg_execution_price - arrival_price
            else:
                total_cost = arrival_price - avg_execution_price

            total_cost_bps = (total_cost / arrival_price) * 10000

            # Attribution weights
            weights = self.config["cost_attribution"]

            # Spread component
            bid_ask_spread = market_data.get("bid_ask_spread", 0.001)
            spread_cost_bps = (bid_ask_spread / 2 / arrival_price) * 10000

            # Market impact component (estimated from participation rate)
            participation_rate = execution_data.get("participation_rate", 0.1)
            volatility = market_data.get("volatility", 0.25)
            impact_estimate = 0.5 * volatility * np.sqrt(participation_rate) * 10000

            # Timing component (residual after spread and impact)
            timing_estimate = total_cost_bps - spread_cost_bps - impact_estimate

            # Opportunity cost (for unfilled quantity)
            fill_rate = executed_quantity / quantity if quantity > 0 else 1.0
            opportunity_cost_bps = (1 - fill_rate) * abs(total_cost_bps)

            # Commission (if provided)
            commission_bps = execution_data.get("commission_bps", 0.0)

            return {
                "total": total_cost_bps,
                "spread": spread_cost_bps,
                "impact": impact_estimate,
                "timing": timing_estimate,
                "opportunity": opportunity_cost_bps,
                "commission": commission_bps
            }

        except Exception as e:
            self.logger.error(f"Cost decomposition failed: {e}")
            return {
                "total": 0.0, "spread": 0.0, "impact": 0.0,
                "timing": 0.0, "opportunity": 0.0, "commission": 0.0
            }

    def _calculate_benchmark_performance(self, side: str, execution_price: float,
                                        benchmark_price: float) -> float:
        """Calculate performance vs benchmark in basis points"""
        try:
            if side == "buy":
                performance = execution_price - benchmark_price
            else:
                performance = benchmark_price - execution_price

            performance_bps = (performance / benchmark_price) * 10000

            return performance_bps

        except Exception as e:
            self.logger.error(f"Benchmark performance calculation failed: {e}")
            return 0.0

    def analyze_capacity_impact(self, symbol: str, trade_size: float,
                               current_aum: float, market_data: Dict[str, Any]) -> CapacityImpact:
        """Analyze capacity impact and scaling projections"""
        try:
            daily_volume = market_data.get("daily_volume", 1000000)
            price = market_data.get("price", 100.0)
            volatility = market_data.get("volatility", 0.25)

            # Current trade metrics
            trade_value = trade_size * price
            adv_percentage = trade_size / daily_volume

            # Linear impact component
            linear_coeff = 0.5
            linear_impact = linear_coeff * volatility * np.sqrt(adv_percentage) * 10000

            # Nonlinear impact for large trades
            capacity_config = self.config["capacity_analysis"]
            threshold = capacity_config["nonlinear_threshold"]
            scaling_exp = capacity_config["scaling_exponent"]

            if adv_percentage > threshold:
                excess = adv_percentage - threshold
                nonlinear_impact = linear_impact * (excess ** scaling_exp)
            else:
                nonlinear_impact = 0.0

            total_impact_bps = linear_impact + nonlinear_impact

            # Capacity utilization
            estimated_capacity = 100_000_000  # $100M baseline capacity
            capacity_utilization = current_aum / estimated_capacity

            # Scaling projections
            def project_costs(target_aum: float) -> float:
                scaling_factor = (target_aum / current_aum) ** 0.5  # Square-root scaling
                return total_impact_bps * scaling_factor

            projected_10m = project_costs(10_000_000)
            projected_50m = project_costs(50_000_000)
            projected_100m = project_costs(100_000_000)

            capacity_impact = CapacityImpact(
                current_aum=current_aum,
                trade_size=trade_size,
                linear_impact=linear_impact,
                nonlinear_impact=nonlinear_impact,
                total_impact_bps=total_impact_bps,
                adv_percentage=adv_percentage * 100,  # Convert to percentage
                capacity_utilization=capacity_utilization,
                projected_costs_10m=projected_10m,
                projected_costs_50m=projected_50m,
                projected_costs_100m=projected_100m
            )

            # Store analysis
            self._store_capacity_analysis(symbol, capacity_impact)

            self.logger.info(f"Capacity impact analysis for {symbol}: "
                           f"Current: {total_impact_bps:.1f}bps, "
                           f"@ $100M: {projected_100m:.1f}bps")

            return capacity_impact

        except Exception as e:
            self.logger.error(f"Capacity impact analysis failed: {e}")
            raise

    def generate_cost_performance_report(self, start_date: datetime,
                                        end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive cost performance report"""
        try:
            # Filter data by date range
            period_executions = [
                exec for exec in self.execution_costs
                if start_date <= exec.timestamp <= end_date
            ]

            if not period_executions:
                return {"error": "No execution data for specified period"}

            # Calculate aggregate metrics
            total_trades = len(period_executions)
            total_volume = sum(exec.executed_quantity for exec in period_executions)

            # Cost statistics
            avg_total_cost = np.mean([exec.total_cost_bps for exec in period_executions])
            avg_spread_cost = np.mean([exec.spread_cost_bps for exec in period_executions])
            avg_impact_cost = np.mean([exec.market_impact_bps for exec in period_executions])
            avg_timing_cost = np.mean([exec.timing_cost_bps for exec in period_executions])

            # Performance vs benchmarks
            avg_vs_arrival = np.mean([exec.vs_arrival_bps for exec in period_executions])
            avg_vs_vwap = np.mean([exec.vs_vwap_bps for exec in period_executions])
            avg_vs_twap = np.mean([exec.vs_twap_bps for exec in period_executions])

            # Execution quality metrics
            avg_fill_rate = np.mean([exec.fill_rate for exec in period_executions])
            avg_participation = np.mean([exec.participation_rate for exec in period_executions])

            # Cost distribution analysis
            cost_percentiles = np.percentile(
                [exec.total_cost_bps for exec in period_executions],
                [10, 25, 50, 75, 90]
            )

            report = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_trades": total_trades,
                    "total_volume": total_volume
                },
                "cost_breakdown": {
                    "total_cost_bps": avg_total_cost,
                    "spread_cost_bps": avg_spread_cost,
                    "market_impact_bps": avg_impact_cost,
                    "timing_cost_bps": avg_timing_cost
                },
                "benchmark_performance": {
                    "vs_arrival_bps": avg_vs_arrival,
                    "vs_vwap_bps": avg_vs_vwap,
                    "vs_twap_bps": avg_vs_twap
                },
                "execution_quality": {
                    "average_fill_rate": avg_fill_rate,
                    "average_participation_rate": avg_participation
                },
                "cost_distribution": {
                    "p10": cost_percentiles[0],
                    "p25": cost_percentiles[1],
                    "median": cost_percentiles[2],
                    "p75": cost_percentiles[3],
                    "p90": cost_percentiles[4]
                },
                "generated_at": datetime.now().isoformat()
            }

            self.logger.info(f"Cost performance report generated for period "
                           f"{start_date.date()} to {end_date.date()}")

            return report

        except Exception as e:
            self.logger.error(f"Cost performance report generation failed: {e}")
            return {"error": str(e)}

    def _store_cost_estimate(self, estimate: CostEstimate):
        """Store cost estimate in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cost_estimates
                    (symbol, side, quantity, estimated_price, spread_cost_bps,
                     market_impact_bps, timing_cost_bps, total_cost_bps,
                     cost_lower_bound, cost_upper_bound, market_regime,
                     volatility, liquidity_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    estimate.symbol, estimate.side, estimate.quantity,
                    estimate.estimated_price, estimate.spread_cost_bps,
                    estimate.market_impact_bps, estimate.timing_cost_bps,
                    estimate.total_cost_bps, estimate.cost_lower_bound,
                    estimate.cost_upper_bound, estimate.market_regime,
                    estimate.volatility, estimate.liquidity_score,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Cost estimate storage failed: {e}")

    def _store_execution_cost(self, breakdown: ExecutionCostBreakdown):
        """Store execution cost breakdown in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO execution_costs
                    (trade_id, symbol, side, quantity, executed_quantity,
                     arrival_price, average_execution_price, vwap_benchmark,
                     twap_benchmark, total_cost_bps, spread_cost_bps,
                     market_impact_bps, timing_cost_bps, opportunity_cost_bps,
                     commission_bps, vs_arrival_bps, vs_vwap_bps, vs_twap_bps,
                     participation_rate, execution_duration_minutes, fill_rate, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    breakdown.trade_id, breakdown.symbol, breakdown.side,
                    breakdown.quantity, breakdown.executed_quantity,
                    breakdown.arrival_price, breakdown.average_execution_price,
                    breakdown.vwap_benchmark, breakdown.twap_benchmark,
                    breakdown.total_cost_bps, breakdown.spread_cost_bps,
                    breakdown.market_impact_bps, breakdown.timing_cost_bps,
                    breakdown.opportunity_cost_bps, breakdown.commission_bps,
                    breakdown.vs_arrival_bps, breakdown.vs_vwap_bps,
                    breakdown.vs_twap_bps, breakdown.participation_rate,
                    breakdown.execution_duration.total_seconds() / 60,
                    breakdown.fill_rate, breakdown.timestamp.isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Execution cost storage failed: {e}")

    def _store_capacity_analysis(self, symbol: str, analysis: CapacityImpact):
        """Store capacity analysis in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO capacity_analysis
                    (symbol, current_aum, trade_size, linear_impact, nonlinear_impact,
                     total_impact_bps, adv_percentage, capacity_utilization,
                     projected_costs_10m, projected_costs_50m, projected_costs_100m, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, analysis.current_aum, analysis.trade_size,
                    analysis.linear_impact, analysis.nonlinear_impact,
                    analysis.total_impact_bps, analysis.adv_percentage,
                    analysis.capacity_utilization, analysis.projected_costs_10m,
                    analysis.projected_costs_50m, analysis.projected_costs_100m,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Capacity analysis storage failed: {e}")

def main():
    """Test the cost attribution analyzer"""
    analyzer = CostAttributionAnalyzer()

    print("Testing Cost Attribution Analyzer...")

    # Test pre-trade cost estimate
    market_data = {
        "volatility": 0.30,
        "bid_ask_spread": 0.002,
        "daily_volume": 2000000,
        "regime": "volatile"
    }

    estimate = analyzer.estimate_pre_trade_costs(
        "AAPL", "buy", 5000, 150.0, market_data
    )

    print(f"Pre-trade cost estimate: {estimate.total_cost_bps:.1f}bps "
          f"[{estimate.cost_lower_bound:.1f}, {estimate.cost_upper_bound:.1f}]")

    # Test execution cost analysis
    execution_data = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 5000,
        "executed_quantity": 4800,
        "arrival_price": 150.0,
        "average_execution_price": 150.25,
        "participation_rate": 0.15,
        "duration": timedelta(minutes=45)
    }

    market_data.update({"vwap": 150.20, "twap": 150.15})

    breakdown = analyzer.analyze_execution_costs(
        "TEST_TRADE_001", execution_data, market_data
    )

    print(f"Execution cost breakdown: Total: {breakdown.total_cost_bps:.1f}bps, "
          f"vs VWAP: {breakdown.vs_vwap_bps:.1f}bps")

    # Test capacity analysis
    capacity_impact = analyzer.analyze_capacity_impact(
        "AAPL", 5000, 10000000, market_data
    )

    print(f"Capacity impact: Current: {capacity_impact.total_impact_bps:.1f}bps, "
          f"@ $100M: {capacity_impact.projected_costs_100m:.1f}bps")

    print("Cost Attribution Analyzer test completed")

if __name__ == "__main__":
    main()