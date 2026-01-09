"""
Base Selection Strategy Interface

This module defines the base interface and data structures for stock selection strategies.
All selection strategies must inherit from BaseSelectionStrategy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SelectionAction(Enum):
    """Actions that can be recommended for selected stocks."""
    BUY = "buy"
    STRONG_BUY = "strong_buy"
    HOLD = "hold"
    WATCH = "watch"
    AVOID = "avoid"


@dataclass
class SelectionCriteria:
    """Configuration parameters for stock selection strategies."""
    max_stocks: int = 20
    min_market_cap: float = 1e9  # $1B minimum market cap
    max_market_cap: float = 1e12  # $1T maximum market cap
    min_volume: int = 100000  # Minimum daily volume
    min_price: float = 5.0  # Minimum stock price
    max_price: float = 1000.0  # Maximum stock price
    exclude_sectors: List[str] = field(default_factory=list)
    include_sectors: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)
    min_score_threshold: float = 0.0  # Minimum strategy score to include
    
    # Strategy-specific parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Result from a stock selection strategy run."""
    symbol: str
    score: float
    action: SelectionAction
    reasoning: str
    metrics: Dict[str, Any]  # Strategy-specific metrics
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # Confidence level 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'score': self.score,
            'action': self.action.value,
            'reasoning': self.reasoning,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence
        }


@dataclass
class StrategyResults:
    """Complete results from a selection strategy execution."""
    strategy_name: str
    selected_stocks: List[SelectionResult]
    total_candidates: int
    execution_time: float  # seconds
    timestamp: datetime = field(default_factory=datetime.now)
    criteria_used: Optional[SelectionCriteria] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_selections(self, n: int = 10, min_score: float = 0.0) -> List[SelectionResult]:
        """Get top N selections above minimum score threshold."""
        filtered = [s for s in self.selected_stocks if s.score >= min_score]
        return sorted(filtered, key=lambda x: x.score, reverse=True)[:n]
    
    def get_symbols(self, action: Optional[SelectionAction] = None) -> List[str]:
        """Get list of symbols, optionally filtered by action."""
        if action is None:
            return [s.symbol for s in self.selected_stocks]
        return [s.symbol for s in self.selected_stocks if s.action == action]
    
    def to_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.selected_stocks:
            return {
                'strategy': self.strategy_name,
                'total_selected': 0,
                'execution_time': self.execution_time,
                'timestamp': self.timestamp.isoformat()
            }
        
        scores = [s.score for s in self.selected_stocks]
        actions = {}
        for result in self.selected_stocks:
            action = result.action.value
            actions[action] = actions.get(action, 0) + 1
        
        return {
            'strategy': self.strategy_name,
            'total_candidates': self.total_candidates,
            'total_selected': len(self.selected_stocks),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'actions_breakdown': actions,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'errors': len(self.errors)
        }


class BaseSelectionStrategy(ABC):
    """
    Abstract base class for stock selection strategies.
    
    All selection strategies must implement the select_stocks method
    and provide strategy-specific logic for evaluating stocks.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the selection strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def select_stocks(
        self, 
        universe: List[str], 
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """
        Select stocks from the given universe based on strategy logic.
        
        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria and parameters
            
        Returns:
            StrategyResults containing selected stocks and metadata
        """
        pass
    
    @abstractmethod
    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate strategy-specific score for a stock.
        
        Args:
            symbol: Stock symbol
            data: Stock data (price, fundamentals, etc.)
            
        Returns:
            Numerical score (higher is better)
        """
        pass
    
    def get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive stock data for analysis.
        
        This method should be implemented by strategies to gather
        the specific data they need (price, fundamentals, technicals, etc.).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock data or None if unavailable
        """
        try:
            # Default implementation - strategies should override
            # Import here to avoid circular imports
            from bot.data import fetch_history, fetch_ticker_info

            # Get price history (last 100 days)
            df = fetch_history(None, symbol, period='day', limit=100, dry_run=False)

            # Get fundamental data - now respects DATA_SOURCE config
            info = fetch_ticker_info(symbol)

            if df is None or info is None:
                return None

            # Basic data structure
            return {
                'symbol': symbol,
                'price_history': df,
                'fundamentals': info,
                'current_price': df['close'].iloc[-1] if not df.empty else 0,
                'volume': df['volume'].iloc[-1] if not df.empty else 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to get data for {symbol}: {e}")
            return None
    
    def filter_universe(self, universe: List[str], criteria: SelectionCriteria) -> List[str]:
        """
        Filter stock universe based on basic criteria.
        
        Args:
            universe: List of symbols to filter
            criteria: Filtering criteria
            
        Returns:
            Filtered list of symbols
        """
        filtered = []
        
        for symbol in universe:
            # Skip explicitly excluded symbols
            if symbol in criteria.exclude_symbols:
                continue
            
            try:
                data = self.get_stock_data(symbol)
                if data is None:
                    continue
                
                fundamentals = data.get('fundamentals', {})
                current_price = data.get('current_price', 0)
                volume = data.get('volume', 0)
                
                # Price filters
                if current_price < criteria.min_price or current_price > criteria.max_price:
                    continue
                
                # Volume filter
                if volume < criteria.min_volume:
                    continue
                
                # Market cap filters
                market_cap = fundamentals.get('market_cap', 0)
                if market_cap < criteria.min_market_cap or market_cap > criteria.max_market_cap:
                    continue
                
                # Sector filters
                sector = fundamentals.get('sector', '')
                if criteria.exclude_sectors and sector in criteria.exclude_sectors:
                    continue
                if criteria.include_sectors and sector not in criteria.include_sectors:
                    continue
                
                filtered.append(symbol)
                
            except Exception as e:
                self.logger.warning(f"Error filtering {symbol}: {e}")
                continue
        
        self.logger.info(f"Filtered universe: {len(universe)} -> {len(filtered)} stocks")
        return filtered
    
    def validate_criteria(self, criteria: Optional[SelectionCriteria]) -> SelectionCriteria:
        """
        Validate and set default selection criteria.
        
        Args:
            criteria: Input criteria or None
            
        Returns:
            Validated criteria with defaults
        """
        if criteria is None:
            criteria = SelectionCriteria()
        
        # Validate ranges
        if criteria.max_stocks <= 0:
            criteria.max_stocks = 20
        if criteria.min_market_cap < 0:
            criteria.min_market_cap = 1e9
        if criteria.min_price <= 0:
            criteria.min_price = 5.0
        
        return criteria
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy."""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__,
            'module': self.__class__.__module__
        }
    
    def __str__(self) -> str:
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"