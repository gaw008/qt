#!/usr/bin/env python3
"""
Tiger API Cost Model Integration

This module integrates the advanced trading cost model with Tiger API,
providing real-time cost estimation for actual trading decisions.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from bot.tradeup_client import build_clients
from bot.execution_tiger import create_tiger_execution_engine
from .trading_cost_model import (
    AdvancedTradingCostModel, MarketData, OrderType,
    TradingCostComponents, OrderExecutionPlan, create_cost_model
)

logger = logging.getLogger(__name__)


class TigerCostAnalyzer:
    """
    Tiger API integrated cost analyzer

    Combines real Tiger market data with advanced cost modeling
    for accurate pre-trade cost estimation
    """

    def __init__(self, cost_model_config: Optional[Dict] = None):
        """Initialize Tiger cost analyzer"""

        # Initialize Tiger clients
        try:
            self.quote_client, self.trade_client = build_clients()
            if self.trade_client:
                self.execution_engine = create_tiger_execution_engine(
                    self.quote_client, self.trade_client
                )
            else:
                self.execution_engine = None
                logger.warning("Tiger clients not available - using simulation mode")
        except Exception as e:
            logger.error(f"Failed to initialize Tiger clients: {e}")
            self.quote_client = None
            self.trade_client = None
            self.execution_engine = None

        # Initialize cost model
        self.cost_model = create_cost_model(cost_model_config)

        # Cache for market data
        self._market_data_cache = {}
        self._cache_expiry_minutes = 5

        logger.info("Tiger Cost Analyzer initialized")

    def get_real_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Fetch real market data from Tiger API

        Args:
            symbol: Stock symbol

        Returns:
            MarketData object or None if failed
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self._market_data_cache:
                cached_data, cache_time = self._market_data_cache[cache_key]
                if (datetime.now() - cache_time).seconds < self._cache_expiry_minutes * 60:
                    return cached_data

            if not self.quote_client:
                logger.warning(f"Quote client not available, using default data for {symbol}")
                return self._get_default_market_data(symbol)

            # Get quote data from Tiger
            quotes = self.quote_client.get_stock_briefs([symbol])
            if not quotes or len(quotes) == 0:
                logger.warning(f"No quote data for {symbol}")
                return self._get_default_market_data(symbol)

            quote = quotes[0]

            # Get additional market data
            market_data = MarketData(
                symbol=symbol,
                price=float(getattr(quote, 'latest_price', getattr(quote, 'close', 100.0))),
                bid=float(getattr(quote, 'bid', getattr(quote, 'latest_price', 100.0)) - 0.01),
                ask=float(getattr(quote, 'ask', getattr(quote, 'latest_price', 100.0)) + 0.01),
                volume=float(getattr(quote, 'volume', 1000000)),
                avg_daily_volume=float(getattr(quote, 'avg_volume', 2000000)),
                volatility=self._estimate_volatility(quote),
                market_cap=self._estimate_market_cap(quote),
                timestamp=datetime.now()
            )

            # Cache the data
            self._market_data_cache[cache_key] = (market_data, datetime.now())

            logger.debug(f"Retrieved market data for {symbol}: ${market_data.price:.2f}")
            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._get_default_market_data(symbol)

    def _estimate_volatility(self, quote) -> float:
        """Estimate volatility from quote data"""
        try:
            # Try to get volatility from different fields
            if hasattr(quote, 'volatility'):
                return float(quote.volatility)

            # Estimate from price range
            high = getattr(quote, 'high', None)
            low = getattr(quote, 'low', None)
            close = getattr(quote, 'close', None)

            if high and low and close and close > 0:
                # Garman-Klass volatility estimator (simplified)
                daily_range = (high - low) / close
                return min(1.0, max(0.1, daily_range * 5))  # Rough approximation

            return 0.25  # Default 25% annual volatility

        except Exception:
            return 0.25

    def _estimate_market_cap(self, quote) -> float:
        """Estimate market cap from quote data"""
        try:
            if hasattr(quote, 'market_cap'):
                return float(quote.market_cap)

            # Estimate based on price (very rough)
            price = getattr(quote, 'latest_price', getattr(quote, 'close', 100.0))
            if price > 200:
                return 50e9  # Large cap
            elif price > 50:
                return 5e9   # Mid cap
            else:
                return 1e9   # Small cap

        except Exception:
            return 5e9  # Default mid-cap

    def _get_default_market_data(self, symbol: str) -> MarketData:
        """Get default market data when Tiger API is unavailable"""

        # Use some realistic defaults based on common stocks
        defaults = {
            'AAPL': {'price': 175.0, 'vol': 0.25, 'mcap': 2.8e12, 'adv': 50e6},
            'MSFT': {'price': 350.0, 'vol': 0.22, 'mcap': 2.6e12, 'adv': 25e6},
            'GOOGL': {'price': 125.0, 'vol': 0.28, 'mcap': 1.6e12, 'adv': 20e6},
            'TSLA': {'price': 250.0, 'vol': 0.45, 'mcap': 800e9, 'adv': 80e6},
        }

        if symbol in defaults:
            data = defaults[symbol]
        else:
            # Generic defaults
            data = {'price': 100.0, 'vol': 0.30, 'mcap': 5e9, 'adv': 2e6}

        return MarketData(
            symbol=symbol,
            price=data['price'],
            bid=data['price'] - 0.05,
            ask=data['price'] + 0.05,
            volume=data['adv'] / 2,  # Half of average daily volume
            avg_daily_volume=data['adv'],
            volatility=data['vol'],
            market_cap=data['mcap'],
            timestamp=datetime.now()
        )

    def analyze_trade_cost(self, symbol: str, quantity: int,
                          order_type: OrderType = OrderType.MARKET,
                          execution_params: Optional[Dict] = None) -> TradingCostComponents:
        """
        Analyze trading cost for a specific trade

        Args:
            symbol: Stock symbol
            quantity: Number of shares (positive for buy, negative for sell)
            order_type: Type of order execution
            execution_params: Optional execution parameters

        Returns:
            Detailed cost breakdown
        """
        try:
            # Get real market data
            market_data = self.get_real_market_data(symbol)
            if not market_data:
                raise ValueError(f"Could not get market data for {symbol}")

            # Calculate costs
            cost_components = self.cost_model.calculate_total_cost(
                market_data, quantity, order_type, execution_params
            )

            logger.info(f"Cost analysis for {symbol}: {quantity} shares = "
                       f"${cost_components.total_cost:.2f} "
                       f"({cost_components.cost_basis_points:.1f} bps)")

            return cost_components

        except Exception as e:
            logger.error(f"Error analyzing trade cost for {symbol}: {e}")
            raise

    def optimize_execution_for_symbol(self, symbol: str, target_quantity: int,
                                    max_execution_time_hours: float = 2.0,
                                    cost_tolerance_bps: float = 50.0) -> OrderExecutionPlan:
        """
        Optimize execution strategy for a specific symbol

        Args:
            symbol: Stock symbol
            target_quantity: Total quantity to trade
            max_execution_time_hours: Maximum execution time allowed
            cost_tolerance_bps: Maximum acceptable cost in basis points

        Returns:
            Optimized execution plan
        """
        try:
            # Get real market data
            market_data = self.get_real_market_data(symbol)
            if not market_data:
                raise ValueError(f"Could not get market data for {symbol}")

            # Optimize execution
            execution_plan = self.cost_model.optimize_execution_strategy(
                market_data, target_quantity, max_execution_time_hours, cost_tolerance_bps
            )

            logger.info(f"Optimized execution for {symbol}: "
                       f"{execution_plan.recommended_strategy.value} strategy, "
                       f"cost: ${execution_plan.total_estimated_cost:.2f}")

            return execution_plan

        except Exception as e:
            logger.error(f"Error optimizing execution for {symbol}: {e}")
            raise

    def analyze_portfolio_rebalance_costs(self, current_positions: List[Dict],
                                        target_positions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze costs for complete portfolio rebalancing

        Args:
            current_positions: List of current position dicts
            target_positions: List of target position dicts

        Returns:
            Complete rebalancing cost analysis
        """
        try:
            total_cost = 0.0
            trade_analyses = []

            # Create symbol maps
            current_map = {pos['symbol']: pos.get('quantity', 0) for pos in current_positions}
            target_map = {pos['symbol']: pos.get('quantity', 0) for pos in target_positions}

            # Find all symbols that need trading
            all_symbols = set(current_map.keys()) | set(target_map.keys())

            for symbol in all_symbols:
                current_qty = current_map.get(symbol, 0)
                target_qty = target_map.get(symbol, 0)
                trade_qty = target_qty - current_qty

                if abs(trade_qty) > 0:  # Need to trade this symbol
                    try:
                        cost_analysis = self.analyze_trade_cost(symbol, trade_qty)
                        total_cost += cost_analysis.total_cost

                        trade_analyses.append({
                            'symbol': symbol,
                            'trade_quantity': trade_qty,
                            'cost_analysis': cost_analysis,
                            'current_position': current_qty,
                            'target_position': target_qty
                        })

                    except Exception as e:
                        logger.warning(f"Could not analyze cost for {symbol}: {e}")
                        continue

            # Calculate portfolio-level metrics
            total_trade_value = sum(
                abs(trade['trade_quantity']) * self.get_real_market_data(trade['symbol']).price
                for trade in trade_analyses
                if self.get_real_market_data(trade['symbol'])
            )

            avg_cost_bps = (total_cost / total_trade_value * 10000) if total_trade_value > 0 else 0

            result = {
                'total_rebalancing_cost': total_cost,
                'average_cost_bps': avg_cost_bps,
                'total_trade_value': total_trade_value,
                'number_of_trades': len(trade_analyses),
                'trade_analyses': trade_analyses,
                'analysis_timestamp': datetime.now().isoformat()
            }

            logger.info(f"Portfolio rebalance analysis: ${total_cost:.2f} total cost, "
                       f"{avg_cost_bps:.1f} bps average, {len(trade_analyses)} trades")

            return result

        except Exception as e:
            logger.error(f"Error analyzing portfolio rebalance costs: {e}")
            raise

    def get_current_positions_from_tiger(self) -> List[Dict]:
        """Get current positions from Tiger API"""
        try:
            if not self.execution_engine:
                logger.warning("Tiger execution engine not available")
                return []

            positions = self.execution_engine.get_account_positions()

            # Convert to standard format
            formatted_positions = []
            for pos in positions:
                formatted_positions.append({
                    'symbol': pos.get('symbol', ''),
                    'quantity': pos.get('quantity', 0),
                    'market_value': pos.get('market_value', 0.0),
                    'average_cost': pos.get('average_cost', 0.0)
                })

            logger.info(f"Retrieved {len(formatted_positions)} positions from Tiger API")
            return formatted_positions

        except Exception as e:
            logger.error(f"Error getting positions from Tiger: {e}")
            return []


def create_tiger_cost_analyzer(config: Optional[Dict] = None) -> TigerCostAnalyzer:
    """
    Factory function to create Tiger cost analyzer

    Args:
        config: Optional configuration

    Returns:
        TigerCostAnalyzer instance
    """
    return TigerCostAnalyzer(config)


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = create_tiger_cost_analyzer()

    # Example: Analyze cost for buying 1000 shares of AAPL
    try:
        cost_analysis = analyzer.analyze_trade_cost("AAPL", 1000, OrderType.MARKET)
        print(f"AAPL trade cost: ${cost_analysis.total_cost:.2f} "
              f"({cost_analysis.cost_basis_points:.1f} bps)")

        # Optimize execution strategy
        execution_plan = analyzer.optimize_execution_for_symbol("AAPL", 5000)
        print(f"Optimal strategy for AAPL: {execution_plan.recommended_strategy.value}")
        print(f"Estimated cost: ${execution_plan.total_estimated_cost:.2f}")

    except Exception as e:
        print(f"Error in cost analysis: {e}")