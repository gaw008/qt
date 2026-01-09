#!/usr/bin/env python3
"""
Advanced Trading Cost Model

This module implements sophisticated trading cost estimation including:
- Bid-ask spread costs
- Market impact modeling (temporary and permanent)
- ADV-based participation rate limits
- Time-dependent cost models
- Volatility-adjusted slippage

Based on industry best practices and academic research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order execution types"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


@dataclass
class MarketData:
    """Market data for cost calculation"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    avg_daily_volume: float
    volatility: float
    market_cap: float
    timestamp: datetime


@dataclass
class TradingCostComponents:
    """Breakdown of trading cost components"""
    spread_cost: float
    market_impact_temporary: float
    market_impact_permanent: float
    timing_cost: float
    opportunity_cost: float
    total_cost: float
    cost_basis_points: float


@dataclass
class OrderExecutionPlan:
    """Execution plan with cost estimates"""
    symbol: str
    target_quantity: int
    slices: List[Dict[str, Any]]
    total_estimated_cost: float
    execution_time_minutes: int
    max_participation_rate: float
    recommended_strategy: OrderType


class AdvancedTradingCostModel:
    """
    Advanced trading cost model implementing state-of-the-art cost estimation
    """

    def __init__(self,
                 default_spread_factor: float = 0.5,
                 impact_decay_rate: float = 0.1,
                 max_participation_rate: float = 0.15,
                 volatility_adjustment: bool = True):
        """
        Initialize trading cost model

        Args:
            default_spread_factor: Fraction of spread captured (0.5 = mid-point)
            impact_decay_rate: Rate at which temporary impact decays
            max_participation_rate: Maximum ADV participation allowed
            volatility_adjustment: Whether to adjust costs for volatility
        """
        self.default_spread_factor = default_spread_factor
        self.impact_decay_rate = impact_decay_rate
        self.max_participation_rate = max_participation_rate
        self.volatility_adjustment = volatility_adjustment

        # Model parameters (calibrated from research)
        self.impact_params = {
            'large_cap': {'alpha': 0.6, 'beta': 0.6, 'gamma': 0.2},
            'mid_cap': {'alpha': 0.8, 'beta': 0.7, 'gamma': 0.3},
            'small_cap': {'alpha': 1.0, 'beta': 0.8, 'gamma': 0.4}
        }

        # Timing cost parameters
        self.timing_risk_factor = 0.01  # 1% per day of delay risk

        logger.info("Advanced Trading Cost Model initialized")

    def classify_stock_size(self, market_cap: float) -> str:
        """Classify stock by market cap for cost modeling"""
        if market_cap > 10e9:  # > $10B
            return 'large_cap'
        elif market_cap > 2e9:   # $2B - $10B
            return 'mid_cap'
        else:
            return 'small_cap'

    def calculate_spread_cost(self, market_data: MarketData,
                            quantity: int, order_type: OrderType) -> float:
        """
        Calculate bid-ask spread cost

        Args:
            market_data: Current market data
            quantity: Order quantity
            order_type: Type of order execution

        Returns:
            Spread cost in dollars
        """
        spread = market_data.ask - market_data.bid

        # Adjust spread capture based on order type
        if order_type == OrderType.MARKET:
            spread_factor = 1.0  # Full spread for market orders
        elif order_type == OrderType.AGGRESSIVE:
            spread_factor = 0.8
        elif order_type == OrderType.LIMIT:
            spread_factor = self.default_spread_factor
        elif order_type == OrderType.PASSIVE:
            spread_factor = 0.0  # May receive rebate
        else:  # TWAP, VWAP
            spread_factor = self.default_spread_factor

        spread_cost = abs(quantity) * spread * spread_factor

        logger.debug(f"Spread cost for {market_data.symbol}: ${spread_cost:.2f}")
        return spread_cost

    def calculate_market_impact(self, market_data: MarketData,
                              quantity: int, participation_rate: float,
                              execution_time_minutes: int) -> Tuple[float, float]:
        """
        Calculate temporary and permanent market impact

        Uses the square-root impact model with refinements

        Args:
            market_data: Current market data
            quantity: Order quantity
            participation_rate: Rate of ADV participation
            execution_time_minutes: Expected execution time

        Returns:
            Tuple of (temporary_impact, permanent_impact) in dollars
        """
        # Get parameters based on stock size
        stock_class = self.classify_stock_size(market_data.market_cap)
        params = self.impact_params[stock_class]

        # Normalize quantities
        order_value = abs(quantity) * market_data.price
        adv_value = market_data.avg_daily_volume * market_data.price

        if adv_value == 0:
            return 0.0, 0.0

        # Participation rate impact
        participation_impact = params['alpha'] * (participation_rate ** params['beta'])

        # Size impact (square root of order size)
        size_ratio = order_value / adv_value
        size_impact = params['gamma'] * np.sqrt(size_ratio)

        # Volatility adjustment
        vol_adjustment = 1.0
        if self.volatility_adjustment and market_data.volatility > 0:
            # Higher volatility increases impact
            vol_adjustment = 1.0 + (market_data.volatility - 0.20) * 0.5
            vol_adjustment = max(0.5, min(2.0, vol_adjustment))

        # Base impact in basis points
        base_impact_bps = (participation_impact + size_impact) * vol_adjustment * 100

        # Temporary impact (decays over time)
        temporary_impact_bps = base_impact_bps * 0.7  # 70% is temporary
        temporary_decay = np.exp(-self.impact_decay_rate * execution_time_minutes / 60)
        temporary_impact_bps *= temporary_decay

        # Permanent impact
        permanent_impact_bps = base_impact_bps * 0.3  # 30% is permanent

        # Convert to dollars
        temporary_impact = order_value * temporary_impact_bps / 10000
        permanent_impact = order_value * permanent_impact_bps / 10000

        logger.debug(f"Market impact for {market_data.symbol}: "
                    f"Temp=${temporary_impact:.2f}, Perm=${permanent_impact:.2f}")

        return temporary_impact, permanent_impact

    def calculate_timing_cost(self, market_data: MarketData,
                            quantity: int, delay_hours: float) -> float:
        """
        Calculate opportunity cost of execution delay

        Args:
            market_data: Current market data
            quantity: Order quantity
            delay_hours: Expected delay in hours

        Returns:
            Timing cost in dollars
        """
        if delay_hours <= 0:
            return 0.0

        order_value = abs(quantity) * market_data.price

        # Risk of adverse price movement during delay
        # Based on volatility and time
        daily_vol = market_data.volatility
        hour_vol = daily_vol / np.sqrt(24)  # Convert to hourly volatility

        # Expected cost from random walk + momentum risk
        timing_risk = hour_vol * np.sqrt(delay_hours) * self.timing_risk_factor
        timing_cost = order_value * timing_risk

        logger.debug(f"Timing cost for {market_data.symbol}: ${timing_cost:.2f}")
        return timing_cost

    def calculate_total_cost(self, market_data: MarketData,
                           quantity: int, order_type: OrderType,
                           execution_plan: Optional[Dict] = None) -> TradingCostComponents:
        """
        Calculate comprehensive trading costs

        Args:
            market_data: Current market data
            quantity: Order quantity
            order_type: Type of order execution
            execution_plan: Optional execution parameters

        Returns:
            Complete cost breakdown
        """
        if execution_plan is None:
            execution_plan = {
                'participation_rate': 0.1,
                'execution_time_minutes': 30,
                'delay_hours': 0
            }

        # Calculate each cost component
        spread_cost = self.calculate_spread_cost(market_data, quantity, order_type)

        temp_impact, perm_impact = self.calculate_market_impact(
            market_data, quantity,
            execution_plan['participation_rate'],
            execution_plan['execution_time_minutes']
        )

        timing_cost = self.calculate_timing_cost(
            market_data, quantity, execution_plan['delay_hours']
        )

        # Opportunity cost (simplified)
        opportunity_cost = 0.0  # Could add alpha decay modeling

        # Total cost
        total_cost = spread_cost + temp_impact + perm_impact + timing_cost + opportunity_cost

        # Cost in basis points
        order_value = abs(quantity) * market_data.price
        cost_bps = (total_cost / order_value) * 10000 if order_value > 0 else 0

        return TradingCostComponents(
            spread_cost=spread_cost,
            market_impact_temporary=temp_impact,
            market_impact_permanent=perm_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            cost_basis_points=cost_bps
        )

    def optimize_execution_strategy(self, market_data: MarketData,
                                  target_quantity: int,
                                  max_execution_time_hours: float = 2.0,
                                  cost_tolerance_bps: float = 50.0) -> OrderExecutionPlan:
        """
        Optimize execution strategy to minimize costs

        Args:
            market_data: Current market data
            target_quantity: Total quantity to execute
            max_execution_time_hours: Maximum time allowed
            cost_tolerance_bps: Maximum acceptable cost in basis points

        Returns:
            Optimized execution plan
        """
        best_plan = None
        best_cost = float('inf')

        # Test different execution strategies
        strategies = [
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.TWAP,
            OrderType.VWAP,
            OrderType.AGGRESSIVE
        ]

        for strategy in strategies:
            # Determine optimal parameters for this strategy
            if strategy == OrderType.MARKET:
                participation_rate = 1.0  # Immediate execution
                execution_time = 1  # 1 minute
                slices = [{'quantity': target_quantity, 'time_minutes': 0}]
            else:
                # Optimize slicing for other strategies
                plan = self._optimize_slicing(
                    market_data, target_quantity, strategy,
                    max_execution_time_hours * 60
                )
                participation_rate = plan['participation_rate']
                execution_time = plan['execution_time']
                slices = plan['slices']

            # Calculate cost for this strategy
            execution_plan = {
                'participation_rate': participation_rate,
                'execution_time_minutes': execution_time,
                'delay_hours': 0
            }

            cost_components = self.calculate_total_cost(
                market_data, target_quantity, strategy, execution_plan
            )

            if cost_components.total_cost < best_cost:
                best_cost = cost_components.total_cost
                best_plan = OrderExecutionPlan(
                    symbol=market_data.symbol,
                    target_quantity=target_quantity,
                    slices=slices,
                    total_estimated_cost=cost_components.total_cost,
                    execution_time_minutes=execution_time,
                    max_participation_rate=participation_rate,
                    recommended_strategy=strategy
                )

        logger.info(f"Optimized execution for {market_data.symbol}: "
                   f"{best_plan.recommended_strategy.value} strategy, "
                   f"cost: ${best_cost:.2f}")

        return best_plan

    def _optimize_slicing(self, market_data: MarketData,
                         target_quantity: int, strategy: OrderType,
                         max_time_minutes: float) -> Dict[str, Any]:
        """
        Optimize order slicing for given strategy

        Returns:
            Dictionary with optimal slicing parameters
        """
        # Calculate maximum sustainable participation rate
        daily_volume = market_data.avg_daily_volume
        max_rate = min(self.max_participation_rate,
                      target_quantity / (daily_volume * 0.5))  # 50% of daily volume

        # For most strategies, use moderate participation
        if strategy in [OrderType.TWAP, OrderType.VWAP]:
            participation_rate = min(0.1, max_rate)  # 10% or less
            execution_time = min(max_time_minutes, 60)  # Up to 1 hour
        elif strategy == OrderType.LIMIT:
            participation_rate = min(0.05, max_rate)  # 5% or less
            execution_time = min(max_time_minutes, 120)  # Up to 2 hours
        else:  # AGGRESSIVE
            participation_rate = min(0.2, max_rate)  # 20% or less
            execution_time = min(max_time_minutes, 30)  # Up to 30 minutes

        # Create slicing plan
        num_slices = max(1, int(execution_time / 5))  # 5-minute slices
        slice_size = target_quantity // num_slices
        remainder = target_quantity % num_slices

        slices = []
        for i in range(num_slices):
            slice_qty = slice_size + (1 if i < remainder else 0)
            slices.append({
                'quantity': slice_qty,
                'time_minutes': i * 5,
                'participation_rate': participation_rate
            })

        return {
            'participation_rate': participation_rate,
            'execution_time': execution_time,
            'slices': slices
        }

    def get_market_data_from_symbol(self, symbol: str) -> Optional[MarketData]:
        """
        Fetch market data for cost calculation

        This is a placeholder - in practice, this would integrate with
        your market data provider
        """
        # Placeholder implementation
        # In reality, this would fetch from Tiger API or other data source

        # Default values for testing
        return MarketData(
            symbol=symbol,
            price=100.0,
            bid=99.95,
            ask=100.05,
            volume=1000000,
            avg_daily_volume=2000000,
            volatility=0.25,
            market_cap=5e9,
            timestamp=datetime.now()
        )


def create_cost_model(config: Optional[Dict[str, Any]] = None) -> AdvancedTradingCostModel:
    """
    Factory function to create trading cost model

    Args:
        config: Optional configuration parameters

    Returns:
        Configured trading cost model
    """
    if config is None:
        config = {}

    return AdvancedTradingCostModel(
        default_spread_factor=config.get('spread_factor', 0.5),
        impact_decay_rate=config.get('impact_decay_rate', 0.1),
        max_participation_rate=config.get('max_participation_rate', 0.15),
        volatility_adjustment=config.get('volatility_adjustment', True)
    )


# Example usage and testing
if __name__ == "__main__":
    # Create cost model
    cost_model = create_cost_model()

    # Example market data
    market_data = MarketData(
        symbol="AAPL",
        price=150.0,
        bid=149.95,
        ask=150.05,
        volume=50000000,
        avg_daily_volume=80000000,
        volatility=0.22,
        market_cap=2.5e12,
        timestamp=datetime.now()
    )

    # Calculate costs for different order sizes
    quantities = [100, 1000, 10000]

    for qty in quantities:
        print(f"\nOrder: {qty} shares of {market_data.symbol}")

        # Calculate cost for market order
        cost_components = cost_model.calculate_total_cost(
            market_data, qty, OrderType.MARKET
        )

        print(f"Total cost: ${cost_components.total_cost:.2f} "
              f"({cost_components.cost_basis_points:.1f} bps)")
        print(f"  Spread: ${cost_components.spread_cost:.2f}")
        print(f"  Temp Impact: ${cost_components.market_impact_temporary:.2f}")
        print(f"  Perm Impact: ${cost_components.market_impact_permanent:.2f}")

        # Get optimal execution strategy
        execution_plan = cost_model.optimize_execution_strategy(
            market_data, qty
        )

        print(f"Optimal strategy: {execution_plan.recommended_strategy.value}")
        print(f"Estimated cost: ${execution_plan.total_estimated_cost:.2f}")