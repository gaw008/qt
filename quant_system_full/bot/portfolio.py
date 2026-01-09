"""
Advanced Multi-Stock Portfolio Management System

This module provides comprehensive portfolio management capabilities for the quantitative trading system,
including multi-stock position management, dynamic capital allocation, risk monitoring, and performance tracking.

Features:
- Multi-stock portfolio management with real-time position tracking
- Dynamic capital allocation algorithms (equal weight, market cap weight, risk parity, volatility weight)
- Real-time portfolio risk exposure monitoring and rebalancing
- Position sizing optimization based on volatility and correlation
- Portfolio-level P&L tracking and performance attribution
- Integration with selection results for automatic position management
- Risk management with portfolio-level stop losses and exposure limits
- Performance analytics and attribution analysis
"""

import pandas as pd
import numpy as np
import logging

# Handle pandas_ta import gracefully
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    ta = None
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path

try:
    from .config import SETTINGS
    from .market_time import get_market_manager, MarketPhase
except ImportError:
    # Fallback for standalone imports
    try:
        from config import SETTINGS
        from market_time import get_market_manager, MarketPhase
    except ImportError:
        # Create minimal fallback settings
        class DefaultSettings:
            primary_market = "US"
        SETTINGS = DefaultSettings()

        # Minimal market manager fallback
        def get_market_manager(market):
            class DummyManager:
                pass
            return DummyManager()

        class MarketPhase:
            TRADING = "trading"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    VOLATILITY_WEIGHT = "volatility_weight"
    SCORE_WEIGHT = "score_weight"
    CUSTOM = "custom"


class PositionType(Enum):
    """Position types."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


@dataclass
class Position:
    """Individual position in the portfolio."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float = 0.0
    position_type: PositionType = PositionType.LONG
    entry_time: str = ""
    target_weight: float = 0.0
    actual_weight: float = 0.0
    
    # Risk metrics
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Position metadata
    score: float = 0.0
    sector: str = "Unknown"
    market_cap: float = 0.0
    
    def __post_init__(self):
        if not self.entry_time:
            self.entry_time = datetime.now().isoformat()
        self.update_pnl()
    
    def update_price(self, current_price: float):
        """Update current price and recalculate P&L."""
        self.current_price = current_price
        self.update_pnl()
    
    def update_pnl(self):
        """Update P&L calculations."""
        if self.current_price > 0 and self.entry_price > 0:
            price_change = self.current_price - self.entry_price
            if self.position_type == PositionType.LONG:
                self.unrealized_pnl = price_change * self.quantity
            elif self.position_type == PositionType.SHORT:
                self.unrealized_pnl = -price_change * self.quantity
            else:
                self.unrealized_pnl = 0.0
        
        self.total_pnl = self.unrealized_pnl + self.realized_pnl
    
    def get_market_value(self) -> float:
        """Get current market value of position."""
        return abs(self.quantity) * self.current_price
    
    def get_position_size_dollars(self) -> float:
        """Get position size in dollars."""
        return self.quantity * self.current_price
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.position_type == PositionType.LONG and self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.position_type == PositionType.SHORT and self.quantity < 0


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics."""
    total_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Risk metrics
    portfolio_volatility: float = 0.0
    max_position_weight: float = 0.0
    concentration_risk: float = 0.0
    sector_concentration: Dict[str, float] = field(default_factory=dict)
    
    # Correlation and diversification
    avg_correlation: float = 0.0
    diversification_ratio: float = 0.0
    
    # Value at Risk
    var_1d_95: float = 0.0
    var_1d_99: float = 0.0
    expected_shortfall: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Attribution
    sector_attribution: Dict[str, float] = field(default_factory=dict)
    position_attribution: Dict[str, float] = field(default_factory=dict)
    
    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    
    # Win/Loss statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


class MultiStockPortfolio:
    """
    Advanced multi-stock portfolio management system.
    
    This class provides comprehensive portfolio management capabilities:
    - Multi-stock position tracking and management
    - Dynamic capital allocation strategies
    - Real-time risk monitoring and exposure management
    - Position sizing optimization
    - Performance tracking and attribution
    - Automated rebalancing
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
        max_position_weight: float = 0.1,  # 10% max per position
        max_positions: int = 20,
        rebalance_threshold: float = 0.05,  # 5% deviation trigger
        risk_free_rate: float = 0.02
    ):
        """
        Initialize multi-stock portfolio.
        
        Args:
            initial_capital: Starting portfolio value
            allocation_method: Method for allocating capital across positions
            max_position_weight: Maximum weight per position (0.1 = 10%)
            max_positions: Maximum number of positions
            rebalance_threshold: Threshold for triggering rebalancing
            risk_free_rate: Risk-free rate for performance metrics
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.allocation_method = allocation_method
        self.max_position_weight = max_position_weight
        self.max_positions = max_positions
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.pending_orders: Dict[str, Dict] = {}
        
        # Risk management
        self.portfolio_stop_loss: Optional[float] = None
        self.daily_var_limit: Optional[float] = None
        self.max_sector_exposure: float = 0.3  # 30% max per sector
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # Risk metrics and performance
        self.risk_metrics = PortfolioRiskMetrics()
        self.performance_metrics = PortfolioPerformance()
        
        # Rebalancing
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_frequency = RebalanceFrequency.WEEKLY
        
        # Market time manager
        self.market_manager = get_market_manager(SETTINGS.primary_market)
        
        # Stop loss and take profit triggered positions (for execution)
        self.triggered_positions: List[Dict[str, Any]] = []
        
        logger.info(f"[portfolio] Initialized multi-stock portfolio with ${initial_capital:,.2f}")
        logger.info(f"[portfolio] Allocation method: {allocation_method.value}, Max positions: {max_positions}")
    
    def add_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        position_type: PositionType = PositionType.LONG,
        target_weight: Optional[float] = None,
        score: float = 0.0,
        sector: str = "Unknown",
        market_cap: float = 0.0
    ) -> bool:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (positive for long, negative for short)
            entry_price: Entry price per share
            position_type: Position type (LONG/SHORT)
            target_weight: Target portfolio weight
            score: Selection score for the position
            sector: Stock sector
            market_cap: Market capitalization
            
        Returns:
            True if position added successfully
        """
        try:
            # Check if we already have this position
            if symbol in self.positions:
                logger.warning(f"[portfolio] Position {symbol} already exists, updating instead")
                return self.update_position(symbol, quantity, entry_price)
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.warning(f"[portfolio] Maximum positions ({self.max_positions}) reached")
                return False
            
            # Calculate position value and check weight limits
            position_value = abs(quantity) * entry_price
            current_portfolio_value = self.get_total_value()
            position_weight = position_value / current_portfolio_value if current_portfolio_value > 0 else 0
            
            if position_weight > self.max_position_weight:
                logger.warning(f"[portfolio] Position weight {position_weight:.2%} exceeds limit {self.max_position_weight:.2%}")
                return False
            
            # Check available cash
            if position_value > self.cash:
                logger.warning(f"[portfolio] Insufficient cash: need ${position_value:,.2f}, have ${self.cash:,.2f}")
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,  # Initialize with entry price
                position_type=position_type,
                target_weight=target_weight or position_weight,
                score=score,
                sector=sector,
                market_cap=market_cap
            )
            
            # Calculate stop loss and take profit
            position.atr = self._calculate_atr_for_symbol(symbol, entry_price)
            position.stop_loss_price = self._calculate_stop_loss(entry_price, position.atr, position_type)
            position.take_profit_price = self._calculate_take_profit(entry_price, position.atr, position_type)
            
            # Add to portfolio
            self.positions[symbol] = position
            self.cash -= position_value
            
            # Update portfolio metrics
            self.update_portfolio_metrics()
            
            logger.info(f"[portfolio] Added position: {symbol} {quantity} shares @ ${entry_price:.2f}")
            logger.info(f"[portfolio] Position value: ${position_value:,.2f}, Weight: {position_weight:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to add position {symbol}: {e}")
            return False
    
    def update_position(self, symbol: str, quantity: int, current_price: float) -> bool:
        """
        Update an existing position.
        
        Args:
            symbol: Stock symbol
            quantity: New quantity (can be different from original)
            current_price: Current market price
            
        Returns:
            True if updated successfully
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"[portfolio] Position {symbol} not found for update")
                return False
            
            position = self.positions[symbol]
            old_quantity = position.quantity
            quantity_change = quantity - old_quantity
            
            # Update position
            position.quantity = quantity
            position.update_price(current_price)
            
            # Adjust cash based on quantity change
            cash_change = -quantity_change * current_price
            self.cash += cash_change
            
            # Remove position if quantity is zero
            if quantity == 0:
                return self.close_position(symbol, current_price)
            
            # Update portfolio metrics
            self.update_portfolio_metrics()
            
            logger.info(f"[portfolio] Updated position: {symbol} {old_quantity} -> {quantity} shares @ ${current_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to update position {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float) -> bool:
        """
        Close a position and realize P&L.
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price per share
            
        Returns:
            True if closed successfully
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"[portfolio] Position {symbol} not found for closing")
                return False
            
            position = self.positions[symbol]
            
            # Calculate realized P&L
            if position.position_type == PositionType.LONG:
                realized_pnl = (exit_price - position.entry_price) * position.quantity
            elif position.position_type == PositionType.SHORT:
                realized_pnl = (position.entry_price - exit_price) * abs(position.quantity)
            else:
                realized_pnl = 0.0
            
            # Return cash to portfolio
            cash_return = abs(position.quantity) * exit_price
            self.cash += cash_return
            
            # Update realized P&L
            position.realized_pnl = realized_pnl
            position.update_pnl()
            
            # Log performance
            logger.info(f"[portfolio] Closed position: {symbol} {position.quantity} shares @ ${exit_price:.2f}")
            logger.info(f"[portfolio] Realized P&L: ${realized_pnl:,.2f}")
            
            # Remove from positions
            del self.positions[symbol]
            
            # Update portfolio metrics
            self.update_portfolio_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to close position {symbol}: {e}")
            return False
    
    def update_all_prices(self, price_data: Dict[str, float]):
        """
        Update prices for all positions.
        
        Args:
            price_data: Dictionary of symbol -> current price
        """
        try:
            updated_count = 0
            
            for symbol, position in self.positions.items():
                if symbol in price_data:
                    old_price = position.current_price
                    new_price = price_data[symbol]
                    
                    position.update_price(new_price)
                    updated_count += 1
                    
                    # Check stop loss and take profit triggers
                    self._check_risk_triggers(symbol, position)
            
            if updated_count > 0:
                # Update portfolio metrics after all price updates
                self.update_portfolio_metrics()
                
                logger.debug(f"[portfolio] Updated prices for {updated_count} positions")
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to update prices: {e}")
    
    def calculate_target_allocation(
        self,
        selection_results: List[Dict[str, Any]],
        available_capital: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate target allocation weights based on selection results.
        
        Args:
            selection_results: List of stock selection results with scores
            available_capital: Available capital for allocation
            
        Returns:
            Dictionary of symbol -> target weight
        """
        try:
            if not selection_results:
                return {}
            
            available_capital = available_capital or self.get_available_capital()
            allocation = {}
            
            if self.allocation_method == AllocationMethod.EQUAL_WEIGHT:
                # Equal weight allocation
                n_positions = min(len(selection_results), self.max_positions)
                weight_per_position = 1.0 / n_positions
                
                for i, result in enumerate(selection_results[:n_positions]):
                    symbol = result.get('symbol', result.get('Symbol', ''))
                    allocation[symbol] = weight_per_position
            
            elif self.allocation_method == AllocationMethod.SCORE_WEIGHT:
                # Score-based weighting
                scores = []
                symbols = []
                
                for result in selection_results[:self.max_positions]:
                    symbol = result.get('symbol', result.get('Symbol', ''))
                    score = result.get('score', result.get('avg_score', 0))
                    
                    if score > 0:
                        symbols.append(symbol)
                        scores.append(score)
                
                if scores:
                    total_score = sum(scores)
                    for symbol, score in zip(symbols, scores):
                        allocation[symbol] = (score / total_score) * 0.95  # Use 95% of capital
            
            elif self.allocation_method == AllocationMethod.RISK_PARITY:
                # Risk parity allocation (simplified)
                volatilities = {}
                
                for result in selection_results[:self.max_positions]:
                    symbol = result.get('symbol', result.get('Symbol', ''))
                    # Use score as inverse volatility proxy (higher score = lower vol)
                    volatility = max(0.1, 1.0 / max(0.1, result.get('score', result.get('avg_score', 1.0))))
                    volatilities[symbol] = volatility
                
                if volatilities:
                    # Inverse volatility weighting
                    inv_vol_weights = {s: 1.0 / v for s, v in volatilities.items()}
                    total_inv_vol = sum(inv_vol_weights.values())
                    
                    for symbol, inv_vol in inv_vol_weights.items():
                        allocation[symbol] = (inv_vol / total_inv_vol) * 0.95
            
            # Apply maximum position weight constraint
            for symbol in allocation:
                allocation[symbol] = min(allocation[symbol], self.max_position_weight)
            
            # Normalize if total exceeds 100%
            total_weight = sum(allocation.values())
            if total_weight > 0.95:  # Keep some cash
                for symbol in allocation:
                    allocation[symbol] = allocation[symbol] / total_weight * 0.95
            
            logger.info(f"[portfolio] Calculated target allocation for {len(allocation)} positions")
            
            return allocation
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to calculate target allocation: {e}")
            return {}
    
    def rebalance_portfolio(
        self,
        target_allocation: Dict[str, float],
        current_prices: Dict[str, float],
        force_rebalance: bool = False
    ) -> Dict[str, Tuple[str, int]]:
        """
        Rebalance portfolio to target allocation.
        
        Args:
            target_allocation: Target weights by symbol
            current_prices: Current prices by symbol
            force_rebalance: Force rebalancing even if within threshold
            
        Returns:
            Dictionary of rebalancing orders (symbol -> (action, quantity))
        """
        try:
            if not self._should_rebalance() and not force_rebalance:
                return {}
            
            current_value = self.get_total_value()
            rebalancing_orders = {}
            
            # Calculate current weights
            current_weights = {}
            for symbol, position in self.positions.items():
                current_weights[symbol] = position.get_market_value() / current_value
            
            # Calculate required trades
            for symbol, target_weight in target_allocation.items():
                if symbol not in current_prices:
                    continue
                
                current_weight = current_weights.get(symbol, 0.0)
                weight_diff = target_weight - current_weight
                
                # Only rebalance if difference exceeds threshold
                if abs(weight_diff) > self.rebalance_threshold:
                    target_value = target_weight * current_value
                    current_value_position = current_weights.get(symbol, 0) * current_value
                    value_change = target_value - current_value_position
                    
                    price = current_prices[symbol]
                    quantity_change = int(value_change / price)
                    
                    if abs(quantity_change) > 0:
                        action = "BUY" if quantity_change > 0 else "SELL"
                        rebalancing_orders[symbol] = (action, abs(quantity_change))
            
            # Handle positions not in target allocation (close them)
            for symbol in self.positions:
                if symbol not in target_allocation:
                    quantity = self.positions[symbol].quantity
                    if quantity != 0:
                        action = "SELL" if quantity > 0 else "BUY"
                        rebalancing_orders[symbol] = (action, abs(quantity))
            
            if rebalancing_orders:
                self.last_rebalance = datetime.now()
                logger.info(f"[portfolio] Generated {len(rebalancing_orders)} rebalancing orders")
                
                for symbol, (action, qty) in rebalancing_orders.items():
                    logger.info(f"[portfolio] Rebalance order: {action} {qty} {symbol}")
            
            return rebalancing_orders
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to rebalance portfolio: {e}")
            return {}
    
    def update_portfolio_metrics(self):
        """
        Update comprehensive portfolio risk and performance metrics.
        """
        try:
            current_time = datetime.now()
            total_value = self.get_total_value()
            
            # Calculate exposures
            long_exposure = sum(
                pos.get_market_value() for pos in self.positions.values() if pos.is_long()
            )
            short_exposure = sum(
                pos.get_market_value() for pos in self.positions.values() if pos.is_short()
            )
            
            net_exposure = long_exposure - short_exposure
            gross_exposure = long_exposure + short_exposure
            
            # Calculate position weights and concentration
            position_weights = []
            sector_exposure = {}
            
            for position in self.positions.values():
                weight = position.get_market_value() / total_value if total_value > 0 else 0
                position.actual_weight = weight
                position_weights.append(weight)
                
                # Sector exposure
                sector = position.sector
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0.0
                sector_exposure[sector] += weight
            
            max_position_weight = max(position_weights) if position_weights else 0.0
            concentration_risk = sum(w**2 for w in position_weights)  # Herfindahl index
            
            # Update risk metrics
            self.risk_metrics = PortfolioRiskMetrics(
                total_exposure=total_value,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                max_position_weight=max_position_weight,
                concentration_risk=concentration_risk,
                sector_concentration=sector_exposure,
                last_updated=current_time.isoformat()
            )
            
            # Update performance metrics
            self._calculate_performance_metrics()
            
            # Record performance history
            self.performance_history.append({
                'timestamp': current_time.isoformat(),
                'total_value': total_value,
                'cash': self.cash,
                'positions_count': len(self.positions),
                'total_pnl': self.get_total_pnl(),
                'day_pnl': self.get_daily_pnl()
            })
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            logger.debug(f"[portfolio] Updated portfolio metrics")
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to update portfolio metrics: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio summary information
        """
        try:
            total_value = self.get_total_value()
            total_pnl = self.get_total_pnl()
            
            # Position summaries
            position_summaries = []
            for symbol, position in self.positions.items():
                position_summaries.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.get_market_value(),
                    'weight': position.actual_weight,
                    'unrealized_pnl': position.unrealized_pnl,
                    'total_pnl': position.total_pnl,
                    'sector': position.sector,
                    'score': position.score
                })
            
            # Sort by market value
            position_summaries.sort(key=lambda x: abs(x['market_value']), reverse=True)
            
            return {
                'portfolio_value': total_value,
                'cash': self.cash,
                'invested_capital': total_value - self.cash,
                'total_pnl': total_pnl,
                'total_return_pct': (total_pnl / self.initial_capital) * 100,
                'positions_count': len(self.positions),
                'positions': position_summaries,
                'risk_metrics': asdict(self.risk_metrics),
                'performance_metrics': asdict(self.performance_metrics),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to generate portfolio summary: {e}")
            return {'error': str(e)}
    
    def save_portfolio_state(self, filepath: str) -> bool:
        """
        Save portfolio state to file.
        
        Args:
            filepath: Path to save portfolio state
            
        Returns:
            True if saved successfully
        """
        try:
            portfolio_state = {
                'timestamp': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'cash': self.cash,
                'allocation_method': self.allocation_method.value,
                'max_position_weight': self.max_position_weight,
                'max_positions': self.max_positions,
                'positions': {symbol: asdict(position) for symbol, position in self.positions.items()},
                'risk_metrics': asdict(self.risk_metrics),
                'performance_metrics': asdict(self.performance_metrics),
                'performance_history': self.performance_history[-100:]  # Last 100 records
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(portfolio_state, f, indent=2)
            
            logger.info(f"[portfolio] Portfolio state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to save portfolio state: {e}")
            return False
    
    def load_portfolio_state(self, filepath: str) -> bool:
        """
        Load portfolio state from file.
        
        Args:
            filepath: Path to load portfolio state from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore basic parameters
            self.initial_capital = state.get('initial_capital', self.initial_capital)
            self.current_capital = state.get('current_capital', self.current_capital)
            self.cash = state.get('cash', self.cash)
            
            # Restore positions
            self.positions = {}
            for symbol, pos_data in state.get('positions', {}).items():
                position = Position(**pos_data)
                self.positions[symbol] = position
            
            # Restore performance history
            self.performance_history = state.get('performance_history', [])
            
            # Update metrics
            self.update_portfolio_metrics()
            
            logger.info(f"[portfolio] Portfolio state loaded from {filepath}")
            logger.info(f"[portfolio] Loaded {len(self.positions)} positions")
            
            return True
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to load portfolio state: {e}")
            return False
    
    # Helper methods
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions."""
        return self.cash
    
    def get_total_pnl(self) -> float:
        """Get total portfolio P&L."""
        return self.get_total_value() - self.initial_capital
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L (simplified)."""
        # This would need more sophisticated tracking in production
        if len(self.performance_history) >= 2:
            today_value = self.performance_history[-1]['total_value']
            yesterday_value = self.performance_history[-2]['total_value']
            return today_value - yesterday_value
        return 0.0
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol."""
        return symbol in self.positions
    
    def get_position_count(self) -> int:
        """Get number of positions."""
        return len(self.positions)
    
    def _should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        if self.last_rebalance is None:
            return True
        
        time_since_last = datetime.now() - self.last_rebalance
        
        if self.rebalance_frequency == RebalanceFrequency.DAILY:
            return time_since_last >= timedelta(days=1)
        elif self.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return time_since_last >= timedelta(weeks=1)
        elif self.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return time_since_last >= timedelta(days=30)
        
        return False
    
    def _check_risk_triggers(self, symbol: str, position: Position):
        """Check stop loss and take profit triggers and queue positions for execution."""
        try:
            stop_loss_triggered = False
            take_profit_triggered = False

            # Check stop loss
            if position.stop_loss_price > 0:
                if position.is_long() and position.current_price <= position.stop_loss_price:
                    logger.warning(f"[portfolio] STOP LOSS TRIGGERED for {symbol}: ${position.current_price:.2f} <= ${position.stop_loss_price:.2f}")
                    stop_loss_triggered = True
                elif position.is_short() and position.current_price >= position.stop_loss_price:
                    logger.warning(f"[portfolio] STOP LOSS TRIGGERED for {symbol}: ${position.current_price:.2f} >= ${position.stop_loss_price:.2f}")
                    stop_loss_triggered = True

            # Check take profit
            if position.take_profit_price > 0:
                if position.is_long() and position.current_price >= position.take_profit_price:
                    logger.info(f"[portfolio] TAKE PROFIT TRIGGERED for {symbol}: ${position.current_price:.2f} >= ${position.take_profit_price:.2f}")
                    take_profit_triggered = True
                elif position.is_short() and position.current_price <= position.take_profit_price:
                    logger.info(f"[portfolio] TAKE PROFIT TRIGGERED for {symbol}: ${position.current_price:.2f} <= ${position.take_profit_price:.2f}")
                    take_profit_triggered = True

            # Add to triggered positions list for execution
            if stop_loss_triggered or take_profit_triggered:
                trigger_type = "stop_loss" if stop_loss_triggered else "take_profit"
                trigger_info = {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "current_price": position.current_price,
                    "entry_price": position.entry_price,
                    "trigger_type": trigger_type,
                    "trigger_price": position.stop_loss_price if stop_loss_triggered else position.take_profit_price,
                    "position_type": position.position_type.value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "timestamp": datetime.now().isoformat()
                }
                # Avoid duplicates
                existing_symbols = [t["symbol"] for t in self.triggered_positions]
                if symbol not in existing_symbols:
                    self.triggered_positions.append(trigger_info)
                    logger.warning(f"[portfolio] Position {symbol} queued for {trigger_type} execution: {position.quantity} shares @ ${position.current_price:.2f}")

        except Exception as e:
            logger.error(f"[portfolio] Error checking risk triggers for {symbol}: {e}")

    def get_triggered_positions(self) -> List[Dict[str, Any]]:
        """
        Get and clear the list of positions that triggered stop loss or take profit.

        Returns:
            List of triggered position information for execution.
            Each entry contains: symbol, quantity, current_price, trigger_type, etc.
        """
        triggered = self.triggered_positions.copy()
        self.triggered_positions = []
        if triggered:
            logger.info(f"[portfolio] Returning {len(triggered)} triggered positions for execution")
        return triggered

    def has_triggered_positions(self) -> bool:
        """Check if there are any positions pending execution due to triggers."""
        return len(self.triggered_positions) > 0

    def _calculate_atr_for_symbol(self, symbol: str, current_price: float) -> float:
        """Calculate ATR for a symbol (simplified)."""
        # This is a simplified implementation
        # In production, you would fetch historical data and calculate actual ATR
        return current_price * 0.02  # Assume 2% ATR
    
    def _calculate_stop_loss(self, entry_price: float, atr: float, position_type: PositionType) -> float:
        """Calculate stop loss price."""
        atr_multiplier = 2.0
        
        if position_type == PositionType.LONG:
            return entry_price - (atr * atr_multiplier)
        elif position_type == PositionType.SHORT:
            return entry_price + (atr * atr_multiplier)
        return 0.0
    
    def _calculate_take_profit(self, entry_price: float, atr: float, position_type: PositionType) -> float:
        """Calculate take profit price."""
        atr_multiplier = 3.0
        
        if position_type == PositionType.LONG:
            return entry_price + (atr * atr_multiplier)
        elif position_type == PositionType.SHORT:
            return entry_price - (atr * atr_multiplier)
        return 0.0
    
    def _calculate_performance_metrics(self):
        """Calculate portfolio performance metrics."""
        try:
            if len(self.performance_history) < 2:
                return
            
            # Calculate returns
            values = [record['total_value'] for record in self.performance_history]
            returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
            
            if not returns:
                return
            
            # Basic performance metrics
            total_return = (values[-1] / values[0] - 1) if values[0] > 0 else 0
            avg_return = np.mean(returns) if returns else 0
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
            
            # Risk-adjusted metrics
            excess_returns = [r - self.risk_free_rate/252 for r in returns]
            sharpe_ratio = (np.mean(excess_returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            
            # Downside deviation for Sortino ratio
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = (avg_return / downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0
            
            # Win/Loss statistics
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]
            
            win_rate = len(winning_trades) / len(returns) if returns else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Update performance metrics
            self.performance_metrics = PortfolioPerformance(
                total_return=total_return,
                annualized_return=avg_return * 252,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"[portfolio] Failed to calculate performance metrics: {e}")


# Compatibility alias for legacy code that imports Portfolio class
Portfolio = MultiStockPortfolio


# Legacy functions for backward compatibility

def calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    """
    Legacy function: Calculates the latest ATR value from a given OHLCV DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns ['high', 'low', 'close'].
        length (int): The lookback period for ATR. PDF suggests 14.

    Returns:
        float: The most recent ATR value.
    """
    if df is None or len(df) < length:
        return 0.0
    
    if not HAS_PANDAS_TA or ta is None:
        # Simplified ATR calculation without pandas_ta
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=length).mean()
        
        return float(atr.iloc[-1]) if not atr.empty else 0.0
    
    df = df.copy()
    df.ta.atr(length=length, append=True)
    
    atr_col = f'ATRr_{length}'
    if atr_col in df.columns and not df[atr_col].empty:
        return df[atr_col].iloc[-1]
    return 0.0


def get_position_size(equity: float, price: float, atr: float, 
                        risk_per_trade: float = 0.01, atr_multiplier: float = 2.0) -> int:
    """
    Legacy function: Calculates position size based on the inverse volatility method using ATR.

    Args:
        equity (float): Total current equity of the portfolio.
        price (float): The current price of the asset.
        atr (float): The Average True Range of the asset.
        risk_per_trade (float): The fraction of equity to risk on a single trade (e.g., 0.01 for 1%).
        atr_multiplier (float): The multiplier for ATR to determine the stop loss distance. 
                                (e.g., 2.0 for a 2x ATR stop).

    Returns:
        int: The number of shares to trade. Returns 0 if ATR is zero or price is zero.
    """
    if atr <= 0 or price <= 0:
        return 0

    # 1. How much money are we risking on this trade?
    dollar_risk = equity * risk_per_trade
    
    # 2. How much risk are we taking per share, based on volatility?
    stop_loss_per_share = atr * atr_multiplier
    
    # 3. How many shares can we buy with that risk?
    shares = int(dollar_risk / stop_loss_per_share)
    
    return shares