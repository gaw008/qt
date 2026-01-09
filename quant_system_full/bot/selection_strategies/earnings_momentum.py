"""
Earnings Momentum Selection Strategy

This strategy focuses on stocks with strong earnings momentum,
earnings surprises, and positive earnings revisions.

Key Metrics:
- Earnings surprise history
- Earnings growth rates (QoQ, YoY)
- Revenue growth consistency
- Analyst revision trends
- Upcoming earnings catalysts
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


class EarningsMomentumStrategy(BaseSelectionStrategy):
    """
    Earnings Momentum selection strategy implementation.
    
    Identifies stocks with strong earnings momentum and
    positive earnings-related catalysts.
    """
    
    def __init__(self, 
                 earnings_growth_weight: float = 0.4,
                 surprise_weight: float = 0.3,
                 revision_weight: float = 0.3,
                 min_earnings_growth: float = 0.1):
        """
        Initialize Earnings Momentum strategy.
        
        Args:
            earnings_growth_weight: Weight for earnings growth metrics
            surprise_weight: Weight for earnings surprise history
            revision_weight: Weight for analyst revisions
            min_earnings_growth: Minimum YoY earnings growth rate
        """
        super().__init__(
            name="EarningsMomentum",
            description="Focuses on stocks with strong earnings momentum"
        )
        
        # Normalize weights
        total_weight = earnings_growth_weight + surprise_weight + revision_weight
        self.earnings_growth_weight = earnings_growth_weight / total_weight
        self.surprise_weight = surprise_weight / total_weight
        self.revision_weight = revision_weight / total_weight
        self.min_earnings_growth = min_earnings_growth
        
        self.logger.info(f"Initialized with earnings_weight={self.earnings_growth_weight:.2f}, "
                        f"surprise_weight={self.surprise_weight:.2f}, "
                        f"revision_weight={self.revision_weight:.2f}")
    
    def select_stocks(
        self, 
        universe: List[str], 
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """
        Select stocks using earnings momentum strategy.
        
        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria
            
        Returns:
            Strategy results with selected stocks
        """
        start_time = time.time()
        criteria = self.validate_criteria(criteria)
        
        self.logger.info(f"Running earnings momentum selection on {len(universe)} stocks")
        
        # Filter universe based on basic criteria
        filtered_universe = self.filter_universe(universe, criteria)
        
        selected_stocks = []
        errors = []
        
        for symbol in filtered_universe:
            try:
                # Get stock data with earnings information
                data = self.get_earnings_stock_data(symbol)
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
    
    def get_earnings_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stock data with earnings information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with earnings-focused stock data or None if unavailable
        """
        try:
            # Import here to avoid circular imports
            from bot.data import fetch_history
            from bot.yahoo_data import fetch_yahoo_ticker_info
            
            # Get price history
            df = fetch_history(None, symbol, period='day', limit=90, dry_run=False)
            
            # Get fundamental data with earnings information
            info = fetch_yahoo_ticker_info(symbol)
            
            if df is None or info is None:
                return None
            
            # Try to get additional earnings data using MCP if available
            earnings_data = self._get_mcp_earnings_data(symbol)
            
            return {
                'symbol': symbol,
                'price_history': df,
                'fundamentals': info,
                'earnings_data': earnings_data,
                'current_price': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get earnings data for {symbol}: {e}")
            return None
    
    def _get_mcp_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """
        Try to get additional earnings data via MCP tools.
        
        Returns earnings-specific data or empty dict if unavailable.
        """
        try:
            from ..config import SETTINGS
            
            if not SETTINGS.use_mcp_tools:
                return {}
            
            # This would use MCP Yahoo Finance earnings endpoint
            # For now, return empty dict as placeholder
            # TODO: Implement MCP earnings data integration
            
            return {}
            
        except Exception as e:
            self.logger.debug(f"MCP earnings data not available for {symbol}: {e}")
            return {}
    
    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate earnings momentum score for a stock.
        
        Score components:
        1. Earnings Growth Score: QoQ and YoY growth rates
        2. Earnings Surprise Score: Historical surprise track record
        3. Revision Score: Analyst estimate revisions trend
        
        Args:
            symbol: Stock symbol
            data: Stock data dictionary
            
        Returns:
            Combined score (0-100, higher is better)
        """
        try:
            fundamentals = data.get('fundamentals', {})
            earnings_data = data.get('earnings_data', {})
            
            # Calculate component scores
            growth_score = self._calculate_earnings_growth_score(fundamentals, earnings_data)
            surprise_score = self._calculate_earnings_surprise_score(fundamentals, earnings_data)
            revision_score = self._calculate_revision_score(fundamentals, earnings_data)
            
            # Combine scores with weights
            final_score = (
                self.earnings_growth_weight * growth_score +
                self.surprise_weight * surprise_score +
                self.revision_weight * revision_score
            )
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating earnings score for {symbol}: {e}")
            return 0.0
    
    def _calculate_earnings_growth_score(self, fundamentals: Dict[str, Any], earnings_data: Dict[str, Any]) -> float:
        """Calculate earnings growth component score."""
        try:
            score = 0.0
            
            # Year-over-year earnings growth (0-50 points)
            earnings_growth = fundamentals.get('earningsQuarterlyGrowth', 0) or fundamentals.get('earningsGrowth', 0)
            if earnings_growth is not None and earnings_growth != 0:
                if earnings_growth > 0.5:  # >50% growth
                    score += 50
                elif earnings_growth > 0.25:  # >25% growth
                    score += 40
                elif earnings_growth > self.min_earnings_growth:  # >minimum growth
                    score += 30
                elif earnings_growth > 0:  # Positive growth
                    score += 20
                else:  # Negative growth
                    score += 5
            
            # Revenue growth consistency (0-30 points)
            revenue_growth = fundamentals.get('revenueQuarterlyGrowth', 0) or fundamentals.get('revenueGrowth', 0)
            if revenue_growth is not None and revenue_growth != 0:
                if revenue_growth > 0.2:  # >20% revenue growth
                    score += 30
                elif revenue_growth > 0.1:  # >10% revenue growth
                    score += 25
                elif revenue_growth > 0.05:  # >5% revenue growth
                    score += 20
                elif revenue_growth > 0:  # Positive revenue growth
                    score += 15
                else:
                    score += 5
            
            # Profit margins trend (0-20 points)
            profit_margin = fundamentals.get('profitMargins', 0)
            if profit_margin and profit_margin > 0:
                if profit_margin > 0.2:  # >20% profit margin
                    score += 20
                elif profit_margin > 0.15:  # >15% profit margin
                    score += 15
                elif profit_margin > 0.1:  # >10% profit margin
                    score += 10
                else:
                    score += 5
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating earnings growth score: {e}")
            return 0.0
    
    def _calculate_earnings_surprise_score(self, fundamentals: Dict[str, Any], earnings_data: Dict[str, Any]) -> float:
        """Calculate earnings surprise component score."""
        try:
            # This is a simplified version since detailed surprise history
            # requires specialized data feeds
            
            score = 50.0  # Neutral baseline
            
            # Forward P/E vs Trailing P/E comparison (indicates expectations)
            trailing_pe = fundamentals.get('trailingPE', 0) or fundamentals.get('pe_ratio', 0)
            forward_pe = fundamentals.get('forwardPE', 0) or fundamentals.get('forward_pe', 0)
            
            if trailing_pe > 0 and forward_pe > 0:
                pe_improvement = (trailing_pe - forward_pe) / trailing_pe
                if pe_improvement > 0.2:  # Expected 20%+ earnings improvement
                    score += 30
                elif pe_improvement > 0.1:  # Expected 10%+ improvement
                    score += 20
                elif pe_improvement > 0.05:  # Expected 5%+ improvement
                    score += 10
                elif pe_improvement < -0.1:  # Expected deterioration
                    score -= 20
            
            # Earnings date proximity bonus (upcoming catalyst)
            earnings_dates = fundamentals.get('earningsDate', [])
            if earnings_dates:
                try:
                    # Simple check for upcoming earnings (this would need more sophisticated date parsing)
                    score += 10  # Small bonus for having upcoming earnings
                except Exception:
                    pass
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating surprise score: {e}")
            return 50.0
    
    def _calculate_revision_score(self, fundamentals: Dict[str, Any], earnings_data: Dict[str, Any]) -> float:
        """Calculate analyst revision score."""
        try:
            score = 50.0  # Neutral baseline
            
            # Analyst recommendation (if available)
            recommendation = fundamentals.get('recommendationMean', 0)
            if recommendation > 0:
                if recommendation <= 1.5:  # Strong Buy average
                    score += 30
                elif recommendation <= 2.0:  # Buy average
                    score += 20
                elif recommendation <= 2.5:  # Hold average
                    score += 10
                elif recommendation <= 3.5:  # Moderate Hold/Sell
                    score -= 10
                else:  # Sell average
                    score -= 20
            
            # Target price vs current price
            target_price = fundamentals.get('targetMeanPrice', 0) or fundamentals.get('targetHighPrice', 0)
            current_price = fundamentals.get('currentPrice', 0) or fundamentals.get('regularMarketPrice', 0)
            
            if target_price > 0 and current_price > 0:
                upside = (target_price - current_price) / current_price
                if upside > 0.3:  # >30% upside
                    score += 25
                elif upside > 0.2:  # >20% upside
                    score += 20
                elif upside > 0.1:  # >10% upside
                    score += 15
                elif upside > 0.05:  # >5% upside
                    score += 10
                elif upside < -0.1:  # Negative upside (overvalued)
                    score -= 15
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating revision score: {e}")
            return 50.0
    
    def _determine_action(self, score: float, data: Dict[str, Any]) -> tuple[SelectionAction, str]:
        """Determine recommended action based on earnings momentum."""
        fundamentals = data.get('fundamentals', {})
        
        # Base action on score
        if score >= 85:
            action = SelectionAction.STRONG_BUY
            reasoning = f"Exceptional earnings momentum (score: {score:.1f})"
        elif score >= 70:
            action = SelectionAction.BUY  
            reasoning = f"Strong earnings momentum (score: {score:.1f})"
        elif score >= 55:
            action = SelectionAction.WATCH
            reasoning = f"Moderate earnings potential (score: {score:.1f})"
        else:
            action = SelectionAction.AVOID
            reasoning = f"Weak earnings momentum (score: {score:.1f})"
        
        # Add specific earnings details
        earnings_growth = fundamentals.get('earningsQuarterlyGrowth', 0) or fundamentals.get('earningsGrowth', 0)
        if earnings_growth and earnings_growth > 0.25:
            reasoning += f"; strong earnings growth ({earnings_growth:.1%})"
        
        revenue_growth = fundamentals.get('revenueQuarterlyGrowth', 0) or fundamentals.get('revenueGrowth', 0)
        if revenue_growth and revenue_growth > 0.15:
            reasoning += f"; solid revenue growth ({revenue_growth:.1%})"
        
        return action, reasoning
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key earnings metrics."""
        fundamentals = data.get('fundamentals', {})
        
        metrics = {
            'current_price': data.get('current_price', 0),
            'earnings_growth': fundamentals.get('earningsQuarterlyGrowth', 0) or fundamentals.get('earningsGrowth', 0),
            'revenue_growth': fundamentals.get('revenueQuarterlyGrowth', 0) or fundamentals.get('revenueGrowth', 0),
            'profit_margins': fundamentals.get('profitMargins', 0),
            'trailing_pe': fundamentals.get('trailingPE', 0) or fundamentals.get('pe_ratio', 0),
            'forward_pe': fundamentals.get('forwardPE', 0) or fundamentals.get('forward_pe', 0),
            'recommendation_mean': fundamentals.get('recommendationMean', 0),
            'target_price': fundamentals.get('targetMeanPrice', 0) or fundamentals.get('targetHighPrice', 0),
        }
        
        # Calculate potential upside
        current_price = data.get('current_price', 0)
        target_price = metrics['target_price']
        if current_price > 0 and target_price > 0:
            metrics['potential_upside'] = (target_price - current_price) / current_price
        
        return metrics
    
    def _calculate_confidence(self, score: float, data: Dict[str, Any]) -> float:
        """Calculate confidence level for earnings momentum analysis."""
        fundamentals = data.get('fundamentals', {})
        
        confidence = 0.5  # Base confidence
        
        # More data points = higher confidence
        if fundamentals.get('earningsGrowth') is not None:
            confidence += 0.15
        if fundamentals.get('revenueGrowth') is not None:
            confidence += 0.15
        if fundamentals.get('recommendationMean', 0) > 0:
            confidence += 0.1
        if fundamentals.get('targetMeanPrice', 0) > 0:
            confidence += 0.1
        
        # Large companies have more reliable earnings data
        market_cap = fundamentals.get('market_cap', 0)
        if market_cap > 10e9:  # >$10B market cap
            confidence += 0.1
        
        return min(1.0, confidence)