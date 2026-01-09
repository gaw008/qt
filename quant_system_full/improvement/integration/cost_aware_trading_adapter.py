#!/usr/bin/env python3
"""
Cost-Aware Trading Adapter
Wraps the existing AutoTradingEngine with cost modeling capabilities
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add paths for imports
improvement_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "worker"))
sys.path.extend([improvement_path, worker_path])

# Import cost modeling system
try:
    from cost_models.trading_cost_model import AdvancedTradingCostModel, MarketData
    COST_MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Cost model not available: {e}")
    COST_MODEL_AVAILABLE = False

# Import existing trading engine
try:
    from auto_trading_engine import AutoTradingEngine
    AUTO_TRADING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Auto trading engine not available: {e}")
    AUTO_TRADING_AVAILABLE = False

logger = logging.getLogger(__name__)

class CostAwareTradingAdapter:
    """
    Cost-aware wrapper for AutoTradingEngine
    Adds trading cost optimization to existing trading decisions
    """

    def __init__(self, dry_run=True, max_position_value=10000, max_daily_trades=10, enable_cost_optimization=True):
        """
        Initialize cost-aware trading adapter

        Args:
            dry_run: Whether to run in simulation mode
            max_position_value: Maximum position value per stock
            max_daily_trades: Maximum daily trades
            enable_cost_optimization: Whether to enable cost optimization
        """
        self.enable_cost_optimization = enable_cost_optimization and COST_MODEL_AVAILABLE

        # Initialize base trading engine
        if AUTO_TRADING_AVAILABLE:
            self.base_engine = AutoTradingEngine(
                dry_run=dry_run,
                max_position_value=max_position_value,
                max_daily_trades=max_daily_trades
            )
        else:
            self.base_engine = None
            logger.error("AutoTradingEngine not available")

        # Initialize cost model
        if self.enable_cost_optimization:
            self.cost_model = AdvancedTradingCostModel()
            logger.info("Cost optimization enabled")
        else:
            self.cost_model = None
            logger.info("Cost optimization disabled")

    def calculate_execution_cost(self, symbol: str, quantity: int, price: float) -> Dict[str, float]:
        """
        Calculate execution cost for a trade

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Current price

        Returns:
            Dictionary with cost breakdown
        """
        if not self.enable_cost_optimization:
            return {"total_cost": 0.0, "cost_basis_points": 0.0}

        try:
            # Create market data for cost calculation
            spread = 0.01  # Default 1 cent spread
            market_data = MarketData(
                symbol=symbol,
                price=price,
                bid=price - spread/2,
                ask=price + spread/2,
                volume=1000000,      # Default current volume
                avg_daily_volume=5000000,  # Default ADV
                volatility=0.20,     # Default 20% volatility
                market_cap=1e9,      # Default $1B market cap
                timestamp=datetime.now()
            )

            # Calculate costs using available methods (default to LIMIT order)
            from cost_models.trading_cost_model import OrderType
            cost_components = self.cost_model.calculate_total_cost(market_data, quantity, OrderType.LIMIT)

            # Extract total cost from components
            if hasattr(cost_components, 'total_cost'):
                total_cost = cost_components.total_cost
            else:
                # Calculate manually from components
                total_cost = (getattr(cost_components, 'spread_cost', 0) +
                             getattr(cost_components, 'market_impact_temporary', 0) +
                             getattr(cost_components, 'market_impact_permanent', 0) +
                             getattr(cost_components, 'timing_cost', 0))

            cost_bps = (total_cost / (price * quantity)) * 10000

            return {
                "total_cost": total_cost,
                "cost_basis_points": cost_bps,
                "recommended_order_type": "limit_order",
                "market_data": {
                    "symbol": symbol,
                    "price": price,
                    "spread": spread,
                    "volume": market_data.volume
                }
            }

        except Exception as e:
            logger.error(f"Error calculating cost for {symbol}: {e}")
            return {"total_cost": 0.0, "cost_basis_points": 0.0}

    def optimize_trading_decision(self, symbol: str, recommended_action: str,
                                 quantity: int, price: float, score: float) -> Dict[str, any]:
        """
        Optimize trading decision based on cost analysis

        Args:
            symbol: Stock symbol
            recommended_action: Original recommended action
            quantity: Recommended quantity
            price: Current price
            score: AI confidence score

        Returns:
            Optimized trading decision
        """
        if not self.enable_cost_optimization:
            return {
                "action": recommended_action,
                "quantity": quantity,
                "price": price,
                "optimization_applied": False
            }

        # Calculate execution costs
        cost_analysis = self.calculate_execution_cost(symbol, quantity, price)
        cost_bps = cost_analysis.get("cost_basis_points", 0.0)

        # Cost-based decision logic
        optimized_action = recommended_action
        optimized_quantity = quantity

        # High cost threshold (> 30 bps) - reduce quantity or skip
        if cost_bps > 30:
            if score < 70:  # Low confidence + high cost = skip
                optimized_action = "skip"
                logger.info(f"Skipping {symbol} due to high cost ({cost_bps:.1f} bps) and low score ({score})")
            else:  # High confidence + high cost = reduce quantity
                optimized_quantity = int(quantity * 0.7)  # Reduce by 30%
                logger.info(f"Reducing {symbol} quantity from {quantity} to {optimized_quantity} due to high cost")

        # Medium cost threshold (15-30 bps) - optimize order type
        elif cost_bps > 15:
            logger.info(f"Medium cost detected for {symbol} ({cost_bps:.1f} bps) - using limit order")

        return {
            "action": optimized_action,
            "quantity": optimized_quantity,
            "price": price,
            "cost_analysis": cost_analysis,
            "optimization_applied": True,
            "original_action": recommended_action,
            "original_quantity": quantity
        }

    def analyze_trading_opportunities(self, current_positions: List[Dict],
                                    recommended_positions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Enhanced trading opportunity analysis with cost optimization

        Args:
            current_positions: Current real positions
            recommended_positions: AI recommended positions

        Returns:
            Trading signals with cost optimization applied
        """
        if not self.base_engine:
            logger.error("Base trading engine not available")
            return {"buy": [], "sell": [], "hold": []}

        # Get base trading signals
        base_signals = self.base_engine.analyze_trading_opportunities(
            current_positions, recommended_positions
        )

        if not self.enable_cost_optimization:
            return base_signals

        # Apply cost optimization to buy signals
        optimized_signals = {
            "buy": [],
            "sell": base_signals.get("sell", []),  # Keep sell signals as-is for now
            "hold": base_signals.get("hold", [])
        }

        for buy_signal in base_signals.get("buy", []):
            symbol = buy_signal.get("symbol")
            quantity = buy_signal.get("qty", 0)
            price = buy_signal.get("price", 0.0)
            score = buy_signal.get("score", 50.0)

            # Apply cost optimization
            optimized_decision = self.optimize_trading_decision(
                symbol, "buy", quantity, price, score
            )

            if optimized_decision["action"] == "buy":
                # Update signal with optimized parameters
                optimized_signal = buy_signal.copy()
                optimized_signal.update({
                    "qty": optimized_decision["quantity"],
                    "cost_analysis": optimized_decision.get("cost_analysis", {}),
                    "optimization_note": f"Cost-optimized: {optimized_decision.get('cost_analysis', {}).get('cost_basis_points', 0):.1f} bps"
                })
                optimized_signals["buy"].append(optimized_signal)
            else:
                # Move to hold if skipped due to cost
                hold_signal = buy_signal.copy()
                hold_signal["action"] = "HOLD"
                hold_signal["reason"] = f"Cost-optimized skip: {optimized_decision.get('cost_analysis', {}).get('cost_basis_points', 0):.1f} bps cost"
                optimized_signals["hold"].append(hold_signal)

        logger.info(f"Cost optimization applied: {len(base_signals.get('buy', []))} -> {len(optimized_signals['buy'])} buy signals")

        return optimized_signals

    def execute_trades(self, trading_signals: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Execute trades with cost optimization

        Args:
            trading_signals: Trading signals to execute

        Returns:
            Execution results
        """
        if not self.base_engine:
            logger.error("Base trading engine not available")
            return []

        # Use base engine's execution method if available
        if hasattr(self.base_engine, 'execute_trades'):
            return self.base_engine.execute_trades(trading_signals)
        else:
            logger.info("Trade execution would be handled by base engine")
            return []

    def get_cost_statistics(self) -> Dict[str, any]:
        """
        Get cost optimization statistics

        Returns:
            Dictionary with cost statistics
        """
        if not self.enable_cost_optimization:
            return {"cost_optimization_enabled": False}

        return {
            "cost_optimization_enabled": True,
            "cost_model_available": COST_MODEL_AVAILABLE,
            "supported_order_types": ["market_order", "limit_order", "twap_order"],
            "cost_thresholds": {
                "high_cost_bps": 30,
                "medium_cost_bps": 15,
                "skip_threshold": "high_cost + low_confidence"
            }
        }

# Factory function for easy integration
def create_cost_aware_trading_engine(dry_run=True, max_position_value=10000,
                                   max_daily_trades=10, enable_cost_optimization=True):
    """
    Factory function to create cost-aware trading engine

    Args:
        dry_run: Whether to run in simulation mode
        max_position_value: Maximum position value per stock
        max_daily_trades: Maximum daily trades
        enable_cost_optimization: Whether to enable cost optimization

    Returns:
        CostAwareTradingAdapter instance
    """
    return CostAwareTradingAdapter(
        dry_run=dry_run,
        max_position_value=max_position_value,
        max_daily_trades=max_daily_trades,
        enable_cost_optimization=enable_cost_optimization
    )

if __name__ == "__main__":
    # Test the adapter
    import json

    adapter = create_cost_aware_trading_engine(dry_run=True, enable_cost_optimization=True)

    # Test cost calculation
    cost_analysis = adapter.calculate_execution_cost("AAPL", 100, 150.0)
    print("Cost Analysis:")
    print(json.dumps(cost_analysis, indent=2))

    # Test optimization
    optimization = adapter.optimize_trading_decision("AAPL", "buy", 100, 150.0, 75.0)
    print("\nOptimization Result:")
    print(json.dumps(optimization, indent=2, default=str))

    # Test statistics
    stats = adapter.get_cost_statistics()
    print("\nCost Statistics:")
    print(json.dumps(stats, indent=2))