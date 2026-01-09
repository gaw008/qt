"""
Streaming Value Momentum Strategy

This module implements a streaming version of the Value Momentum strategy,
providing real-time results and memory efficiency while maintaining the same
high-quality stock selection logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import logging

try:
    from .streaming_strategy import StreamingSelectionStrategy
except ImportError:
    from streaming_strategy import StreamingSelectionStrategy

logger = logging.getLogger(__name__)


class StreamingValueMomentumStrategy(StreamingSelectionStrategy):
    """
    Streaming implementation of Value Momentum strategy.
    
    This strategy combines value investing principles with momentum indicators
    in a streaming fashion, processing stocks incrementally for better performance.
    """
    
    def __init__(self, 
                 value_weight: float = 0.4,
                 momentum_weight: float = 0.6,
                 candidate_pool_size: int = 100,
                 intermediate_save_interval: int = 5,
                 early_stop_patience: int = 10,
                 min_score_threshold: float = 35.0,
                 progress_callback = None):
        """
        Initialize streaming value momentum strategy.
        
        Args:
            value_weight: Weight for value component (0-1)
            momentum_weight: Weight for momentum component (0-1)
            candidate_pool_size: Maximum candidates to keep in memory
            intermediate_save_interval: Save progress every N batches
            early_stop_patience: Stop if no improvement for N batches
            min_score_threshold: Minimum score threshold for selection
        """
        super().__init__(
            name="StreamingValueMomentum",
            description="Streaming value momentum selection with real-time results",
            candidate_pool_size=candidate_pool_size,
            intermediate_save_interval=intermediate_save_interval,
            early_stop_patience=early_stop_patience,
            min_score_threshold=min_score_threshold,
            progress_callback=progress_callback
        )
        
        # Validate weights
        if not (0 <= value_weight <= 1 and 0 <= momentum_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        total_weight = value_weight + momentum_weight
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights if they don't sum to 1
            self.value_weight = value_weight / total_weight
            self.momentum_weight = momentum_weight / total_weight
            logger.warning(f"Normalized weights: value={self.value_weight:.2f}, "
                          f"momentum={self.momentum_weight:.2f}")
        else:
            self.value_weight = value_weight
            self.momentum_weight = momentum_weight
        
        logger.info(f"Initialized with value_weight={self.value_weight:.2f}, "
                   f"momentum_weight={self.momentum_weight:.2f}")
    
    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate value momentum score for a stock using streaming approach.
        
        This method combines value metrics (P/E, P/B ratios) with momentum
        indicators (price and volume momentum) to generate a comprehensive score.
        
        Args:
            symbol: Stock symbol
            data: Stock data dictionary containing fundamentals and price history
            
        Returns:
            Combined score (0-100, higher is better)
        """
        try:
            fundamentals = data.get('fundamentals', {})
            price_history = data.get('price_history')
            
            if price_history is None or len(price_history) < 20:
                return 0.0
            
            # Calculate value score component
            value_score = self._calculate_value_score(fundamentals)
            
            # Calculate momentum score component
            momentum_score = self._calculate_momentum_score(price_history)
            
            # Combine scores with weights
            final_score = (
                self.value_weight * value_score +
                self.momentum_weight * momentum_score
            )
            
            # Apply quality bonus for high-quality stocks
            quality_bonus = self._calculate_quality_bonus(fundamentals, price_history)
            final_score = min(100.0, final_score + quality_bonus)
            
            return max(0.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}")
            return 0.0
    
    def _calculate_value_score(self, fundamentals: Dict[str, Any]) -> float:
        """Calculate value component of the score with improved fallback logic."""
        try:
            # Get fundamental ratios with multiple key fallbacks
            pe_ratio = self._safe_float(fundamentals.get('pe_ratio') or 
                                      fundamentals.get('trailingPE') or 
                                      fundamentals.get('peRatio'))
            pb_ratio = self._safe_float(fundamentals.get('pb_ratio') or 
                                      fundamentals.get('priceToBook') or 
                                      fundamentals.get('pbRatio'))
            ps_ratio = self._safe_float(fundamentals.get('ps_ratio') or 
                                      fundamentals.get('priceToSalesTrailing12Months') or 
                                      fundamentals.get('psRatio'))
            
            value_score = 0.0
            valid_metrics = 0
            base_score = 20.0  # Minimum score for stocks with price data
            
            # P/E ratio scoring (lower is better) - More lenient thresholds
            if pe_ratio and 0 < pe_ratio <= 100:  # Increased upper bound
                if pe_ratio <= 15:
                    pe_score = 50  # Excellent value
                elif pe_ratio <= 25:
                    pe_score = 35 + (25 - pe_ratio) * 1.5  # Gentler slope
                elif pe_ratio <= 40:
                    pe_score = 20 + (40 - pe_ratio) * 1.0  # Fair value
                else:
                    pe_score = 10  # Still get some points for having PE data
                value_score += pe_score
                valid_metrics += 1
            else:
                # Fallback for stocks without PE (might be growing companies)
                value_score += base_score
            
            # P/B ratio scoring (lower is better) - More lenient
            if pb_ratio and 0 < pb_ratio <= 15:  # Increased upper bound
                if pb_ratio <= 1.5:
                    pb_score = 40  # Excellent value
                elif pb_ratio <= 3:
                    pb_score = 25 + (3 - pb_ratio) * 10  # Good to fair value
                elif pb_ratio <= 5:
                    pb_score = 15 + (5 - pb_ratio) * 5  # Moderate value
                else:
                    pb_score = 10  # Basic score for having data
                value_score += pb_score
                valid_metrics += 1
            else:
                # Fallback for stocks without PB
                value_score += base_score * 0.8
            
            # P/S ratio scoring (lower is better) - More generous
            if ps_ratio and 0 < ps_ratio <= 50:  # Much higher upper bound
                if ps_ratio <= 3:
                    ps_score = 20  # Bonus for low P/S
                elif ps_ratio <= 8:
                    ps_score = 15 + (8 - ps_ratio) * 1.0  # Moderate bonus
                elif ps_ratio <= 15:
                    ps_score = 10  # Small bonus
                else:
                    ps_score = 5  # Still some credit
                value_score += ps_score
                valid_metrics += 1
            else:
                # Fallback for missing PS ratio
                value_score += base_score * 0.5
            
            # Ensure minimum viable metrics or provide fallback score
            if valid_metrics > 0:
                # Normalize but be less harsh on missing metrics
                normalized_score = value_score / max(1, valid_metrics)
                # Apply less severe penalty for missing metrics
                metric_coverage = min(1.0, valid_metrics / 3.0)  # 3 ideal metrics
                coverage_bonus = 0.7 + (metric_coverage * 0.3)  # 70%-100% of score
                return min(100.0, normalized_score * coverage_bonus)
            else:
                # Fallback score for stocks with price data but no fundamentals
                logger.debug("No fundamental metrics available, using fallback score")
                return base_score
            
        except Exception as e:
            logger.warning(f"Error calculating value score: {e}")
            return 20.0  # Return a reasonable fallback score instead of 0
    
    def _calculate_momentum_score(self, price_history: pd.DataFrame) -> float:
        """Calculate momentum component of the score."""
        try:
            if len(price_history) < 20:
                return 0.0
            
            current_price = price_history['close'].iloc[-1]
            momentum_score = 0.0
            
            # Short-term momentum (1-month return, 20 trading days)
            if len(price_history) >= 20:
                price_20d = price_history['close'].iloc[-20]
                return_20d = (current_price - price_20d) / price_20d
                
                # Score based on return magnitude
                if return_20d > 0.20:  # >20% gain
                    short_momentum = 30
                elif return_20d > 0.10:  # >10% gain
                    short_momentum = 20 + (return_20d - 0.10) * 100
                elif return_20d > 0.05:  # >5% gain
                    short_momentum = 15 + (return_20d - 0.05) * 100
                elif return_20d > 0:  # Positive gain
                    short_momentum = 10 + return_20d * 100
                elif return_20d > -0.05:  # Small loss
                    short_momentum = 5 + (return_20d + 0.05) * 100
                else:  # Significant loss
                    short_momentum = max(0, 5 + return_20d * 50)
                
                momentum_score += short_momentum
            
            # Medium-term momentum (2-month return, 40 trading days)
            if len(price_history) >= 40:
                price_40d = price_history['close'].iloc[-40]
                return_40d = (current_price - price_40d) / price_40d
                
                # Weight medium-term less than short-term
                medium_momentum = max(0, min(20, 10 + return_40d * 50))
                momentum_score += medium_momentum * 0.6
            
            # Volume momentum (increasing volume trend)
            volume_momentum = self._calculate_volume_momentum(price_history)
            momentum_score += volume_momentum * 20  # Up to 20 points for volume
            
            # Technical momentum (price relative to moving averages)
            technical_momentum = self._calculate_technical_momentum(price_history)
            momentum_score += technical_momentum * 10  # Up to 10 points for technicals
            
            return min(100.0, momentum_score)
            
        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _calculate_volume_momentum(self, price_history: pd.DataFrame) -> float:
        """Calculate volume momentum factor (0-1)."""
        try:
            if len(price_history) < 10:
                return 0.5  # Neutral
            
            # Compare recent volume to historical average
            recent_volume = price_history['volume'].iloc[-5:].mean()
            historical_volume = price_history['volume'].iloc[:-5].mean()
            
            if historical_volume <= 0:
                return 0.5
            
            volume_ratio = recent_volume / historical_volume
            
            # Score volume momentum
            if volume_ratio > 1.5:  # Significantly higher volume
                return 1.0
            elif volume_ratio > 1.2:  # Moderately higher volume
                return 0.7 + (volume_ratio - 1.2) * 1.0
            elif volume_ratio > 0.8:  # Normal volume
                return 0.5 + (volume_ratio - 0.8) * 0.5
            else:  # Lower volume
                return max(0.0, volume_ratio * 0.625)  # 0.8 * 0.625 = 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating volume momentum: {e}")
            return 0.5
    
    def _calculate_technical_momentum(self, price_history: pd.DataFrame) -> float:
        """Calculate technical momentum based on moving averages."""
        try:
            if len(price_history) < 50:
                return 0.5  # Neutral
            
            current_price = price_history['close'].iloc[-1]
            
            # Calculate moving averages
            ma_20 = price_history['close'].iloc[-20:].mean()
            ma_50 = price_history['close'].iloc[-50:].mean()
            
            technical_score = 0.5  # Start neutral
            
            # Price above/below moving averages
            if current_price > ma_20:
                technical_score += 0.25
            if current_price > ma_50:
                technical_score += 0.15
            if ma_20 > ma_50:  # Upward trend
                technical_score += 0.1
            
            return min(1.0, max(0.0, technical_score))
            
        except Exception as e:
            logger.warning(f"Error calculating technical momentum: {e}")
            return 0.5
    
    def _calculate_quality_bonus(self, fundamentals: Dict[str, Any], price_history: pd.DataFrame) -> float:
        """Calculate quality bonus for high-quality stocks."""
        try:
            bonus = 0.0
            
            # Market cap bonus (prefer larger, more stable companies)
            market_cap = self._safe_float(fundamentals.get('market_cap') or fundamentals.get('marketCap'))
            if market_cap:
                if market_cap > 10e9:  # >$10B
                    bonus += 5
                elif market_cap > 1e9:  # >$1B
                    bonus += 2
            
            # ROE bonus
            roe = self._safe_float(fundamentals.get('return_on_equity') or fundamentals.get('returnOnEquity'))
            if roe and roe > 0.15:  # >15% ROE
                bonus += 3
            elif roe and roe > 0.10:  # >10% ROE
                bonus += 1
            
            # Profit margin bonus
            profit_margin = self._safe_float(fundamentals.get('profit_margin') or fundamentals.get('profitMargins'))
            if profit_margin and profit_margin > 0.20:  # >20% margin
                bonus += 2
            elif profit_margin and profit_margin > 0.10:  # >10% margin
                bonus += 1
            
            # Price stability bonus (lower volatility)
            if len(price_history) >= 30:
                returns = price_history['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                if volatility < 0.20:  # Low volatility
                    bonus += 2
                elif volatility < 0.35:  # Moderate volatility
                    bonus += 1
            
            return bonus
            
        except Exception as e:
            logger.warning(f"Error calculating quality bonus: {e}")
            return 0.0
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0