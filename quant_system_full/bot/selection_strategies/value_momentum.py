"""
Value Momentum Selection Strategy

This strategy combines value investing principles with momentum indicators
to identify stocks that are both undervalued and showing positive momentum.

Key Metrics:
- P/E Ratio (lower is better for value)
- P/B Ratio (lower is better for value) 
- Price momentum (higher is better)
- Volume momentum (confirming price moves)
- Earnings growth (fundamental momentum)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time

from .base_strategy import (
    BaseSelectionStrategy, SelectionResult, SelectionCriteria, 
    StrategyResults, SelectionAction
)


class ValueMomentumStrategy(BaseSelectionStrategy):
    """
    Value Momentum selection strategy implementation.
    
    Combines value metrics (P/E, P/B) with momentum indicators
    to find undervalued stocks with positive price momentum.
    """
    
    def __init__(self, 
                 value_weight: float = 0.4,
                 momentum_weight: float = 0.6,
                 lookback_days: int = 60):
        """
        Initialize Value Momentum strategy.
        
        Args:
            value_weight: Weight for value component (0-1)
            momentum_weight: Weight for momentum component (0-1)  
            lookback_days: Days to look back for momentum calculation
        """
        super().__init__(
            name="ValueMomentum",
            description="Combines value metrics with momentum indicators"
        )
        
        # Normalize weights
        total_weight = value_weight + momentum_weight
        self.value_weight = value_weight / total_weight
        self.momentum_weight = momentum_weight / total_weight
        self.lookback_days = lookback_days
        
        self.logger.info(f"Initialized with value_weight={self.value_weight:.2f}, "
                        f"momentum_weight={self.momentum_weight:.2f}")
    
    def select_stocks(
        self, 
        universe: List[str], 
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """
        Select stocks using value momentum strategy.
        
        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria
            
        Returns:
            Strategy results with selected stocks
        """
        start_time = time.time()
        criteria = self.validate_criteria(criteria)
        
        self.logger.info(f"Running value momentum selection on {len(universe)} stocks")
        
        # Filter universe based on basic criteria
        filtered_universe = self.filter_universe(universe, criteria)
        
        selected_stocks = []
        errors = []
        
        for symbol in filtered_universe:
            try:
                # Get stock data
                data = self.get_stock_data(symbol)
                if data is None:
                    continue
                
                # Calculate strategy score
                score = self.calculate_score(symbol, data)
                if score < criteria.min_score_threshold:
                    continue
                
                # Determine action and reasoning
                action, reasoning = self._determine_action(score, data)
                
                # Extract key metrics for result
                metrics = self._extract_metrics(data)
                
                result = SelectionResult(
                    symbol=symbol,
                    score=score,
                    action=action,
                    reasoning=reasoning,
                    metrics=metrics,
                    confidence=self._calculate_confidence(score, data)
                )
                
                selected_stocks.append(result)
                
            except Exception as e:
                error_msg = f"Error processing {symbol}: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
        
        # Sort by score and limit results
        selected_stocks.sort(key=lambda x: x.score, reverse=True)
        selected_stocks = selected_stocks[:criteria.max_stocks]
        
        execution_time = time.time() - start_time
        
        results = StrategyResults(
            strategy_name=self.name,
            selected_stocks=selected_stocks,
            total_candidates=len(filtered_universe),
            execution_time=execution_time,
            criteria_used=criteria,
            errors=errors
        )
        
        self.logger.info(f"Selected {len(selected_stocks)} stocks in {execution_time:.2f}s")
        return results
    
    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate value momentum score for a stock.
        
        Score components:
        1. Value Score: Based on P/E, P/B ratios (inverted - lower is better)
        2. Momentum Score: Based on price and volume momentum
        
        Args:
            symbol: Stock symbol
            data: Stock data dictionary
            
        Returns:
            Combined score (0-100, higher is better)
        """
        try:
            fundamentals = data.get('fundamentals', {})
            price_history = data.get('price_history')
            
            if price_history is None or len(price_history) < 20:
                return 0.0
            
            # Calculate value score
            value_score = self._calculate_value_score(fundamentals)
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(price_history)
            
            # Combine scores with weights
            final_score = (
                self.value_weight * value_score +
                self.momentum_weight * momentum_score
            )
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating score for {symbol}: {e}")
            return 0.0
    
    def _calculate_value_score(self, fundamentals: Dict[str, Any]) -> float:
        """Calculate value component of the score."""
        try:
            pe_ratio = fundamentals.get('pe_ratio', 0)
            pb_ratio = fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0)
            
            # Handle missing or invalid ratios
            if pe_ratio <= 0 or pe_ratio > 100:
                pe_score = 0
            else:
                # Invert P/E ratio (lower is better), normalize to 0-50 range
                pe_score = max(0, 50 - (pe_ratio * 50 / 30))  # Assume 30 P/E = 0 score
            
            if pb_ratio <= 0 or pb_ratio > 10:
                pb_score = 0
            else:
                # Invert P/B ratio (lower is better), normalize to 0-50 range
                pb_score = max(0, 50 - (pb_ratio * 50 / 5))  # Assume 5 P/B = 0 score
            
            # Average the scores
            value_score = (pe_score + pb_score) / 2
            
            return value_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating value score: {e}")
            return 0.0
    
    def _calculate_momentum_score(self, price_history: pd.DataFrame) -> float:
        """Calculate momentum component of the score."""
        try:
            if len(price_history) < 20:
                return 0.0
            
            # Calculate price returns over different periods
            current_price = price_history['close'].iloc[-1]
            
            # 1-month return (20 trading days)
            if len(price_history) >= 20:
                price_20d = price_history['close'].iloc[-20]
                return_20d = (current_price - price_20d) / price_20d
            else:
                return_20d = 0
            
            # 2-month return (40 trading days)
            if len(price_history) >= 40:
                price_40d = price_history['close'].iloc[-40]
                return_40d = (current_price - price_40d) / price_40d
            else:
                return_40d = return_20d
            
            # Volume momentum (increasing volume with price)
            volume_momentum = self._calculate_volume_momentum(price_history)
            
            # Combine momentum indicators
            price_momentum = (return_20d * 0.6 + return_40d * 0.4) * 100  # Scale to percentage
            
            # Normalize momentum score to 0-100
            momentum_score = max(0, min(100, 50 + price_momentum * 2))  # Center around 50
            
            # Adjust for volume momentum
            momentum_score = momentum_score * (0.7 + 0.3 * volume_momentum)
            
            return momentum_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _calculate_volume_momentum(self, price_history: pd.DataFrame) -> float:
        """Calculate volume momentum factor (0-1)."""
        try:
            if len(price_history) < 10:
                return 0.5  # Neutral
            
            # Compare recent volume to historical average
            recent_volume = price_history['volume'].iloc[-5:].mean()
            historical_volume = price_history['volume'].iloc[:-5].mean()
            
            if historical_volume == 0:
                return 0.5
            
            volume_ratio = recent_volume / historical_volume
            
            # Normalize to 0-1 range
            volume_momentum = min(1.0, max(0.0, (volume_ratio - 0.5) / 2 + 0.5))
            
            return volume_momentum
            
        except Exception as e:
            self.logger.warning(f"Error calculating volume momentum: {e}")
            return 0.5
    
    def _determine_action(self, score: float, data: Dict[str, Any]) -> tuple[SelectionAction, str]:
        """Determine recommended action based on score and data."""
        fundamentals = data.get('fundamentals', {})
        
        # Action thresholds
        if score >= 80:
            action = SelectionAction.STRONG_BUY
            reasoning = f"Excellent value-momentum combination (score: {score:.1f})"
        elif score >= 60:
            action = SelectionAction.BUY
            reasoning = f"Good value-momentum characteristics (score: {score:.1f})"
        elif score >= 40:
            action = SelectionAction.WATCH
            reasoning = f"Moderate value-momentum potential (score: {score:.1f})"
        else:
            action = SelectionAction.AVOID
            reasoning = f"Weak value-momentum profile (score: {score:.1f})"
        
        # Add specific reasoning based on metrics
        pe_ratio = fundamentals.get('pe_ratio', 0)
        if pe_ratio > 0 and pe_ratio < 15:
            reasoning += "; attractive valuation"
        elif pe_ratio > 30:
            reasoning += "; high valuation concern"
        
        return action, reasoning
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for the selection result."""
        fundamentals = data.get('fundamentals', {})
        price_history = data.get('price_history')
        
        metrics = {
            'pe_ratio': fundamentals.get('pe_ratio', 0),
            'pb_ratio': fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0),
            'market_cap': fundamentals.get('market_cap', 0),
            'current_price': data.get('current_price', 0),
        }
        
        if price_history is not None and len(price_history) >= 20:
            current_price = price_history['close'].iloc[-1]
            price_20d = price_history['close'].iloc[-20] if len(price_history) >= 20 else current_price
            
            metrics.update({
                'return_20d': (current_price - price_20d) / price_20d if price_20d > 0 else 0,
                'avg_volume': price_history['volume'].mean(),
                'price_volatility': price_history['close'].pct_change().std() * np.sqrt(252)  # Annualized
            })
        
        return metrics
    
    def _calculate_confidence(self, score: float, data: Dict[str, Any]) -> float:
        """Calculate confidence level for the selection."""
        fundamentals = data.get('fundamentals', {})
        price_history = data.get('price_history')
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for complete data
        if fundamentals.get('pe_ratio', 0) > 0:
            confidence += 0.2
        if price_history is not None and len(price_history) >= 60:
            confidence += 0.2
        if fundamentals.get('market_cap', 0) > 1e10:  # Large cap = more confidence
            confidence += 0.1
        
        return min(1.0, confidence)