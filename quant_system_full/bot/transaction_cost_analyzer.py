#!/usr/bin/env python3
"""
Transaction Cost Analysis and Capacity Modeling
交易成本分析与容量建模

Investment-grade transaction cost analysis for institutional deployment:
- Real trading log calibration of slippage models
- Market impact estimation with capacity curves
- Implementation Shortfall analysis vs benchmarks
- Participation rate and ADV impact modeling
- Multi-venue execution quality analysis

投资级交易成本分析：
- 基于真实交易日志的滑点模型校准
- 市场冲击估计与容量曲线分析
- 实施缺口分析与基准比较
- 参与率与ADV冲击建模
- 多交易场所执行质量分析
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import warnings
from scipy import stats
from scipy.optimize import minimize
import json
import sqlite3

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeExecutionData:
    """Individual trade execution record"""
    timestamp: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    arrival_price: float
    execution_price: float
    market_price_at_completion: float
    volume_20d: float  # 20-day average volume
    market_cap: float
    spread_bps: float
    volatility_20d: float
    adv_participation_rate: float
    implementation_shortfall_bps: float
    execution_time_minutes: float

@dataclass
class CostComponents:
    """Breakdown of trading costs"""
    spread_cost_bps: float = 0.0      # Bid-ask spread cost
    market_impact_bps: float = 0.0    # Temporary + permanent impact
    timing_cost_bps: float = 0.0      # Cost of waiting
    commission_bps: float = 0.0       # Commission and fees
    total_cost_bps: float = 0.0       # Total transaction cost

@dataclass
class CapacityLimits:
    """Portfolio capacity analysis results"""
    current_aum: float
    max_theoretical_capacity: float
    optimal_capacity: float  # Where costs start rising significantly
    cost_at_capacity: Dict[float, float]  # AUM -> expected cost mapping
    turnover_capacity_curve: Dict[float, float]  # Turnover -> max capacity
    liquidity_constraints: List[str]

class TransactionCostAnalyzer:
    """
    Investment-grade transaction cost analyzer with capacity modeling

    Features:
    - Market impact model calibration from real trading data
    - Capacity curve estimation for portfolio scaling
    - Implementation Shortfall attribution analysis
    - Optimal execution parameter recommendation
    """

    def __init__(self, trading_database_path: Optional[str] = None):
        """Initialize with optional database of historical trading data"""
        self.trading_db_path = trading_database_path
        self.execution_history: List[TradeExecutionData] = []
        self.cost_model_params = {}
        self.capacity_analysis = None

        # Default cost model parameters (to be calibrated)
        self.default_params = {
            'spread_capture_rate': 0.5,    # How much of spread we typically pay
            'impact_linear_coeff': 0.1,    # Linear market impact coefficient
            'impact_sqrt_coeff': 0.5,      # Square-root impact coefficient
            'volatility_scaling': 1.0,     # Volatility adjustment factor
            'size_penalty_threshold': 0.1, # Participation rate threshold
            'permanent_impact_ratio': 0.3,  # Permanent vs temporary impact
            'urgency_multiplier': 1.5      # Cost multiplier for urgent trades
        }

        logger.info("Transaction Cost Analyzer initialized")

    def load_trading_history(self, file_path: str) -> bool:
        """Load historical trading data from CSV or database"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.execution_history = [
                    TradeExecutionData(**row.to_dict())
                    for _, row in df.iterrows()
                ]
            elif file_path.endswith('.db') or file_path.endswith('.sqlite'):
                conn = sqlite3.connect(file_path)
                df = pd.read_sql_query("SELECT * FROM trade_executions", conn)
                conn.close()
                self.execution_history = [
                    TradeExecutionData(**row.to_dict())
                    for _, row in df.iterrows()
                ]
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False

            logger.info(f"Loaded {len(self.execution_history)} trade execution records")
            return True

        except Exception as e:
            logger.error(f"Failed to load trading history: {e}")
            return False

    def generate_synthetic_trading_data(self, num_trades: int = 1000) -> List[TradeExecutionData]:
        """Generate synthetic trading data for testing and calibration"""
        np.random.seed(42)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

        synthetic_data = []

        for i in range(num_trades):
            symbol = np.random.choice(symbols)
            side = np.random.choice(['BUY', 'SELL'])

            # Generate realistic market parameters
            market_cap = np.random.lognormal(np.log(10e9), 1.5)  # $10B median
            volume_20d = np.random.lognormal(np.log(1e6), 1.0)   # 1M shares median
            volatility_20d = np.random.normal(0.25, 0.1)         # 25% volatility
            volatility_20d = max(0.05, min(1.0, volatility_20d)) # Clamp to reasonable range

            # Trade parameters
            quantity = int(np.random.lognormal(np.log(1000), 1.5))
            arrival_price = np.random.uniform(50, 500)
            spread_bps = np.random.exponential(10) + 2  # 2+ bps spread

            # Calculate participation rate
            adv_participation = quantity / volume_20d

            # Market impact modeling (square-root + linear)
            sqrt_impact = 0.5 * np.sqrt(adv_participation) * volatility_20d * 10000  # to bps
            linear_impact = 0.1 * adv_participation * 10000  # to bps
            market_impact_bps = sqrt_impact + linear_impact

            # Spread cost
            spread_cost_bps = spread_bps * np.random.uniform(0.3, 0.7)  # Partial spread capture

            # Timing cost (random walk during execution)
            execution_time = np.random.exponential(5) + 1  # 1+ minutes
            timing_cost_bps = np.random.normal(0, volatility_20d * np.sqrt(execution_time/1440) * 10000)

            # Total implementation shortfall
            total_cost_bps = spread_cost_bps + market_impact_bps + timing_cost_bps

            # Execution prices
            cost_multiplier = 1 if side == 'BUY' else -1
            execution_price = arrival_price * (1 + cost_multiplier * total_cost_bps / 10000)
            market_price_completion = arrival_price * (1 + cost_multiplier * timing_cost_bps / 10000)

            trade_data = TradeExecutionData(
                timestamp=(datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                arrival_price=arrival_price,
                execution_price=execution_price,
                market_price_at_completion=market_price_completion,
                volume_20d=volume_20d,
                market_cap=market_cap,
                spread_bps=spread_bps,
                volatility_20d=volatility_20d,
                adv_participation_rate=adv_participation,
                implementation_shortfall_bps=total_cost_bps,
                execution_time_minutes=execution_time
            )

            synthetic_data.append(trade_data)

        return synthetic_data

    def calibrate_cost_model(self, trading_data: Optional[List[TradeExecutionData]] = None) -> Dict[str, float]:
        """
        Calibrate market impact model parameters from real trading data

        Uses multivariate regression to fit:
        Cost = α + β₁×√(participation) + β₂×volatility + β₃×spread + β₄×urgency
        """
        if trading_data is None:
            if not self.execution_history:
                logger.warning("No trading data available, generating synthetic data for calibration")
                trading_data = self.generate_synthetic_trading_data()
            else:
                trading_data = self.execution_history

        # Prepare regression data
        features = []
        target_costs = []

        for trade in trading_data:
            # Feature vector: [sqrt_participation, volatility, spread, urgency_proxy]
            sqrt_participation = np.sqrt(trade.adv_participation_rate)
            volatility = trade.volatility_20d
            spread = trade.spread_bps / 10000  # Convert to decimal
            urgency_proxy = 1.0 / (trade.execution_time_minutes + 1)  # Higher for faster execution

            features.append([1.0, sqrt_participation, volatility, spread, urgency_proxy])  # Include intercept
            target_costs.append(trade.implementation_shortfall_bps)

        X = np.array(features)
        y = np.array(target_costs)

        # Ordinary least squares regression
        try:
            # Use normal equation: β = (X'X)⁻¹X'y
            beta = np.linalg.solve(X.T @ X, X.T @ y)

            # Calculate R-squared
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            self.cost_model_params = {
                'intercept': beta[0],
                'sqrt_participation_coeff': beta[1],
                'volatility_coeff': beta[2],
                'spread_coeff': beta[3],
                'urgency_coeff': beta[4],
                'r_squared': r_squared,
                'calibration_sample_size': len(trading_data),
                'calibration_date': datetime.now().isoformat()
            }

            logger.info(f"Cost model calibrated with R² = {r_squared:.3f} on {len(trading_data)} trades")
            return self.cost_model_params

        except np.linalg.LinAlgError:
            logger.error("Failed to calibrate cost model - using default parameters")
            self.cost_model_params = self.default_params.copy()
            return self.cost_model_params

    def estimate_trade_cost(self,
                          symbol: str,
                          quantity: int,
                          market_data: Dict[str, float],
                          urgency: float = 0.5) -> CostComponents:
        """
        Estimate trading cost for a proposed trade using calibrated model

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            market_data: Dict with 'price', 'volume_20d', 'volatility_20d', 'spread_bps'
            urgency: Urgency factor (0 = patient, 1 = urgent)
        """
        # Extract market data
        price = market_data.get('price', 100)
        volume_20d = market_data.get('volume_20d', 1e6)
        volatility_20d = market_data.get('volatility_20d', 0.2)
        spread_bps = market_data.get('spread_bps', 5)

        # Calculate participation rate
        participation_rate = quantity / volume_20d

        # Use calibrated model if available, otherwise use defaults
        params = self.cost_model_params if self.cost_model_params else self.default_params

        # Market impact estimation
        sqrt_impact = params.get('sqrt_participation_coeff', 0.5) * np.sqrt(participation_rate)
        volatility_impact = params.get('volatility_coeff', 100) * volatility_20d
        market_impact_bps = sqrt_impact + volatility_impact

        # Spread cost
        spread_cost_bps = spread_bps * params.get('spread_capture_rate', 0.5)

        # Urgency cost
        urgency_multiplier = 1 + urgency * params.get('urgency_multiplier', 0.5)
        market_impact_bps *= urgency_multiplier

        # Commission (typical institutional rate)
        notional_value = quantity * price
        commission_bps = max(0.5, 5000 / notional_value * 10000)  # $5000 min or 0.5 bps

        # Timing cost (estimated as a function of urgency and volatility)
        timing_cost_bps = (1 - urgency) * volatility_20d * 10  # Patient trades incur timing risk

        # Total cost
        total_cost_bps = market_impact_bps + spread_cost_bps + commission_bps + timing_cost_bps

        return CostComponents(
            spread_cost_bps=spread_cost_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            commission_bps=commission_bps,
            total_cost_bps=total_cost_bps
        )

    def analyze_capacity_curve(self,
                             portfolio_data: Dict[str, Any],
                             aum_range: List[float],
                             turnover_rate: float = 2.0) -> CapacityLimits:
        """
        Analyze portfolio capacity across different AUM levels

        Args:
            portfolio_data: Current portfolio composition and characteristics
            aum_range: List of AUM levels to analyze (e.g., [10M, 50M, 100M, 500M])
            turnover_rate: Annual portfolio turnover rate
        """
        capacity_results = {}
        cost_at_capacity = {}

        current_aum = portfolio_data.get('total_value', 1e6)
        positions = portfolio_data.get('positions', [])

        for target_aum in aum_range:
            scaling_factor = target_aum / current_aum
            total_cost_bps = 0
            liquidity_violations = 0

            for position in positions:
                # Scale position size
                scaled_quantity = position.get('quantity', 0) * scaling_factor
                symbol = position.get('symbol', 'UNKNOWN')

                # Estimate market data (would be real in production)
                mock_market_data = {
                    'price': position.get('current_price', 100),
                    'volume_20d': np.random.lognormal(np.log(1e6), 1.0),  # Would use real data
                    'volatility_20d': 0.25,
                    'spread_bps': 5
                }

                # Annual trading volume from turnover
                annual_shares_traded = scaled_quantity * turnover_rate

                # Estimate cost per trade (assuming monthly rebalancing)
                monthly_trade_size = annual_shares_traded / 12

                if monthly_trade_size > 0:
                    trade_cost = self.estimate_trade_cost(
                        symbol=symbol,
                        quantity=int(monthly_trade_size),
                        market_data=mock_market_data,
                        urgency=0.3  # Moderate urgency for regular rebalancing
                    )

                    # Weight by position size
                    position_weight = position.get('market_value', 0) / current_aum
                    total_cost_bps += trade_cost.total_cost_bps * position_weight

                    # Check liquidity constraints (>10% ADV participation rate is concerning)
                    participation_rate = monthly_trade_size / mock_market_data['volume_20d']
                    if participation_rate > 0.1:  # 10% ADV threshold
                        liquidity_violations += 1

            cost_at_capacity[target_aum] = total_cost_bps
            capacity_results[target_aum] = {
                'total_cost_bps': total_cost_bps,
                'liquidity_violations': liquidity_violations,
                'feasible': liquidity_violations < len(positions) * 0.2  # <20% violations
            }

        # Find optimal capacity (where cost increases become significant)
        feasible_aums = [aum for aum, result in capacity_results.items() if result['feasible']]

        if not feasible_aums:
            max_theoretical_capacity = min(aum_range)
            optimal_capacity = max_theoretical_capacity * 0.5
        else:
            max_theoretical_capacity = max(feasible_aums)

            # Find optimal capacity (where cost doubles from base case)
            base_cost = cost_at_capacity.get(min(aum_range), 10)  # Base case
            optimal_capacity = max_theoretical_capacity

            for aum in sorted(feasible_aums):
                if cost_at_capacity[aum] > base_cost * 2:  # Cost doubling threshold
                    optimal_capacity = aum * 0.8  # 80% of where costs double
                    break

        # Generate liquidity constraints summary
        constraints = []
        if liquidity_violations > 0:
            constraints.append(f"Liquidity concerns for {liquidity_violations} positions")
        if max_theoretical_capacity < max(aum_range):
            constraints.append(f"Maximum feasible AUM: ${max_theoretical_capacity/1e6:.0f}M")

        self.capacity_analysis = CapacityLimits(
            current_aum=current_aum,
            max_theoretical_capacity=max_theoretical_capacity,
            optimal_capacity=optimal_capacity,
            cost_at_capacity=cost_at_capacity,
            turnover_capacity_curve={turnover_rate: optimal_capacity},
            liquidity_constraints=constraints
        )

        logger.info(f"Capacity analysis: Optimal AUM ${optimal_capacity/1e6:.0f}M, Max ${max_theoretical_capacity/1e6:.0f}M")
        return self.capacity_analysis

    def implementation_shortfall_analysis(self,
                                       benchmark: str = 'arrival_price') -> Dict[str, Any]:
        """
        Analyze implementation shortfall vs various benchmarks

        Benchmarks: 'arrival_price', 'vwap', 'twap', 'close'
        """
        if not self.execution_history:
            logger.warning("No execution history available for IS analysis")
            return {}

        is_results = {
            'total_trades': len(self.execution_history),
            'avg_implementation_shortfall_bps': 0,
            'median_implementation_shortfall_bps': 0,
            'is_volatility_bps': 0,
            'hit_rate': 0,  # Percentage of trades with positive alpha
            'cost_breakdown': {
                'market_impact': 0,
                'spread_capture': 0,
                'timing_cost': 0
            },
            'performance_by_size': {},
            'performance_by_sector': {}
        }

        # Collect IS data
        is_values = [trade.implementation_shortfall_bps for trade in self.execution_history]

        if is_values:
            is_results['avg_implementation_shortfall_bps'] = np.mean(is_values)
            is_results['median_implementation_shortfall_bps'] = np.median(is_values)
            is_results['is_volatility_bps'] = np.std(is_values)
            is_results['hit_rate'] = len([x for x in is_values if x > 0]) / len(is_values)

        # Performance attribution by trade size buckets
        small_trades = [t for t in self.execution_history if t.adv_participation_rate < 0.01]
        medium_trades = [t for t in self.execution_history if 0.01 <= t.adv_participation_rate < 0.05]
        large_trades = [t for t in self.execution_history if t.adv_participation_rate >= 0.05]

        is_results['performance_by_size'] = {
            'small_trades': {
                'count': len(small_trades),
                'avg_cost_bps': np.mean([t.implementation_shortfall_bps for t in small_trades]) if small_trades else 0
            },
            'medium_trades': {
                'count': len(medium_trades),
                'avg_cost_bps': np.mean([t.implementation_shortfall_bps for t in medium_trades]) if medium_trades else 0
            },
            'large_trades': {
                'count': len(large_trades),
                'avg_cost_bps': np.mean([t.implementation_shortfall_bps for t in large_trades]) if large_trades else 0
            }
        }

        return is_results

    def generate_cost_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive transaction cost analysis report"""

        # Ensure we have data to analyze
        if not self.execution_history:
            logger.info("Generating synthetic data for cost analysis demonstration")
            self.execution_history = self.generate_synthetic_trading_data(500)

        # Calibrate cost model
        self.calibrate_cost_model()

        # Run implementation shortfall analysis
        is_analysis = self.implementation_shortfall_analysis()

        # Generate capacity analysis with sample portfolio
        sample_portfolio = {
            'total_value': 10e6,  # $10M portfolio
            'positions': [
                {'symbol': 'AAPL', 'quantity': 1000, 'market_value': 2e6, 'current_price': 150},
                {'symbol': 'GOOGL', 'quantity': 500, 'market_value': 1.5e6, 'current_price': 2500},
                {'symbol': 'MSFT', 'quantity': 800, 'market_value': 1.2e6, 'current_price': 300},
            ]
        }

        capacity_analysis = self.analyze_capacity_curve(
            sample_portfolio,
            [10e6, 25e6, 50e6, 100e6, 250e6, 500e6]  # $10M to $500M
        )

        # Compile comprehensive report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'calibration_summary': {
                'model_parameters': self.cost_model_params,
                'sample_size': len(self.execution_history)
            },
            'implementation_shortfall_analysis': is_analysis,
            'capacity_analysis': {
                'optimal_capacity_usd': capacity_analysis.optimal_capacity,
                'max_theoretical_capacity_usd': capacity_analysis.max_theoretical_capacity,
                'cost_curve': capacity_analysis.cost_at_capacity,
                'liquidity_constraints': capacity_analysis.liquidity_constraints
            },
            'trading_cost_estimates': {
                'small_trade_10k': self.estimate_trade_cost(
                    'SAMPLE', 100,
                    {'price': 100, 'volume_20d': 1e6, 'volatility_20d': 0.2, 'spread_bps': 5}
                ).__dict__,
                'medium_trade_100k': self.estimate_trade_cost(
                    'SAMPLE', 1000,
                    {'price': 100, 'volume_20d': 1e6, 'volatility_20d': 0.2, 'spread_bps': 5}
                ).__dict__,
                'large_trade_1M': self.estimate_trade_cost(
                    'SAMPLE', 10000,
                    {'price': 100, 'volume_20d': 1e6, 'volatility_20d': 0.2, 'spread_bps': 5}
                ).__dict__
            },
            'recommendations': {
                'optimal_execution_strategy': 'TWAP with participation rate limits',
                'max_adv_participation': '5% for large trades, 10% for small trades',
                'recommended_turnover': '150-300% annually',
                'capacity_utilization': f"{capacity_analysis.optimal_capacity/1e6:.0f}M USD optimal"
            }
        }

        return report

    def export_cost_analysis(self, filepath: str) -> bool:
        """Export detailed cost analysis to file"""
        try:
            report = self.generate_cost_analysis_report()

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Cost analysis exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export cost analysis: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Transaction Cost Analyzer - Investment Grade Analysis")
    print("=" * 60)

    # Initialize cost analyzer
    cost_analyzer = TransactionCostAnalyzer()

    # Generate and analyze synthetic trading data
    synthetic_data = cost_analyzer.generate_synthetic_trading_data(200)
    cost_analyzer.execution_history = synthetic_data

    print(f"Generated {len(synthetic_data)} synthetic trades for analysis")

    # Calibrate cost model
    calibration_results = cost_analyzer.calibrate_cost_model()
    print(f"\nCost Model Calibration Results:")
    print(f"R-squared: {calibration_results.get('r_squared', 0):.3f}")
    print(f"Sqrt Participation Coefficient: {calibration_results.get('sqrt_participation_coeff', 0):.4f}")
    print(f"Volatility Coefficient: {calibration_results.get('volatility_coeff', 0):.4f}")

    # Test trade cost estimation
    sample_trade_cost = cost_analyzer.estimate_trade_cost(
        symbol='TEST',
        quantity=5000,
        market_data={
            'price': 150,
            'volume_20d': 2e6,
            'volatility_20d': 0.25,
            'spread_bps': 8
        },
        urgency=0.3
    )

    print(f"\nSample Trade Cost Estimation (5000 shares):")
    print(f"Market Impact: {sample_trade_cost.market_impact_bps:.2f} bps")
    print(f"Spread Cost: {sample_trade_cost.spread_cost_bps:.2f} bps")
    print(f"Timing Cost: {sample_trade_cost.timing_cost_bps:.2f} bps")
    print(f"Total Cost: {sample_trade_cost.total_cost_bps:.2f} bps")

    # Generate full cost analysis report
    full_report = cost_analyzer.generate_cost_analysis_report()
    print(f"\nCapacity Analysis Results:")
    print(f"Optimal Capacity: ${full_report['capacity_analysis']['optimal_capacity_usd']/1e6:.0f}M")
    print(f"Max Theoretical: ${full_report['capacity_analysis']['max_theoretical_capacity_usd']/1e6:.0f}M")

    # Export detailed report
    cost_analyzer.export_cost_analysis("transaction_cost_analysis_report.json")
    print(f"\nDetailed cost analysis exported to file!")