"""
Technical Breakout Selection Strategy

This strategy identifies stocks showing technical breakout patterns
that often precede significant price movements.

Key Indicators:
- Bollinger Band breakouts
- Moving average crossovers
- Volume confirmation
- Support/Resistance level breaks
- Relative Strength Index (RSI) patterns
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


class TechnicalBreakoutStrategy(BaseSelectionStrategy):
    """
    Technical Breakout selection strategy implementation.
    
    Identifies stocks with strong technical breakout patterns
    confirmed by volume and momentum indicators.
    """
    
    def __init__(self, 
                 sma_short: int = 20,
                 sma_long: int = 50,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 volume_threshold: float = 1.5):
        """
        Initialize Technical Breakout strategy.
        
        Args:
            sma_short: Short-term moving average period
            sma_long: Long-term moving average period  
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            volume_threshold: Minimum volume increase for confirmation
        """
        super().__init__(
            name="TechnicalBreakout",
            description="Identifies stocks with technical breakout patterns"
        )
        
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_threshold = volume_threshold
        
        self.logger.info(f"Initialized with SMA({sma_short},{sma_long}), "
                        f"BB({bb_period},{bb_std}), volume_threshold={volume_threshold}")
    
    def select_stocks(
        self, 
        universe: List[str], 
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """
        Select stocks using technical breakout strategy.
        
        Args:
            universe: List of stock symbols to evaluate
            criteria: Selection criteria
            
        Returns:
            Strategy results with selected stocks
        """
        start_time = time.time()
        criteria = self.validate_criteria(criteria)
        
        self.logger.info(f"Running technical breakout selection on {len(universe)} stocks")
        
        # Filter universe based on basic criteria
        filtered_universe = self.filter_universe(universe, criteria)
        
        selected_stocks = []
        errors = []
        
        for symbol in filtered_universe:
            try:
                # Get stock data with enough history for technical analysis
                data = self.get_extended_stock_data(symbol)
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
    
    def get_extended_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get extended stock data needed for technical analysis.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with extended stock data or None if unavailable
        """
        try:
            # Import here to avoid circular imports
            from bot.data import fetch_history
            from bot.yahoo_data import fetch_yahoo_ticker_info
            
            # Get longer price history for technical indicators (6 months)
            df = fetch_history(None, symbol, period='day', limit=120, dry_run=False)
            
            # Get fundamental data
            info = fetch_yahoo_ticker_info(symbol)
            
            if df is None or len(df) < 60:  # Need at least 60 days
                return None
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            return {
                'symbol': symbol,
                'price_history': df,
                'fundamentals': info or {},
                'current_price': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get extended data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price dataframe."""
        df = df.copy()
        
        # Simple Moving Averages
        df[f'sma_{self.sma_short}'] = df['close'].rolling(self.sma_short).mean()
        df[f'sma_{self.sma_long}'] = df['close'].rolling(self.sma_long).mean()
        
        # Bollinger Bands
        sma_bb = df['close'].rolling(self.bb_period).mean()
        std_bb = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = sma_bb + (std_bb * self.bb_std)
        df['bb_lower'] = sma_bb - (std_bb * self.bb_std)
        df['bb_middle'] = sma_bb
        
        # RSI (Relative Strength Index)
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position within Bollinger Bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving average signals
        df['sma_signal'] = np.where(df[f'sma_{self.sma_short}'] > df[f'sma_{self.sma_long}'], 1, -1)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate technical breakout score for a stock.
        
        Score components:
        1. Breakout Pattern Score (40%): Bollinger Band and MA breakouts
        2. Volume Confirmation Score (30%): Volume supporting the breakout
        3. Momentum Score (20%): RSI and price momentum
        4. Trend Strength Score (10%): Overall trend direction
        
        Args:
            symbol: Stock symbol
            data: Stock data dictionary
            
        Returns:
            Combined score (0-100, higher is better)
        """
        try:
            price_history = data.get('price_history')
            
            if price_history is None or len(price_history) < 60:
                return 0.0
            
            # Get latest data point
            latest = price_history.iloc[-1]
            prev_data = price_history.iloc[-5:]  # Last 5 days for analysis
            
            # 1. Breakout Pattern Score (40%)
            breakout_score = self._calculate_breakout_score(latest, price_history)
            
            # 2. Volume Confirmation Score (30%)
            volume_score = self._calculate_volume_score(latest, prev_data)
            
            # 3. Momentum Score (20%)
            momentum_score = self._calculate_technical_momentum_score(latest, price_history)
            
            # 4. Trend Strength Score (10%)
            trend_score = self._calculate_trend_score(latest, price_history)
            
            # Combine scores with weights
            final_score = (
                0.4 * breakout_score +
                0.3 * volume_score +
                0.2 * momentum_score +
                0.1 * trend_score
            )
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score for {symbol}: {e}")
            return 0.0
    
    def _calculate_breakout_score(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """Calculate breakout pattern score."""
        try:
            score = 0.0
            
            # Bollinger Band breakout (0-40 points)
            if pd.notna(latest['bb_position']):
                if latest['bb_position'] > 1.0:  # Above upper band
                    score += 40
                elif latest['bb_position'] > 0.9:  # Near upper band
                    score += 30
                elif latest['bb_position'] > 0.8:  # Upper portion
                    score += 20
                elif latest['bb_position'] < 0.0:  # Below lower band (oversold)
                    score += 15  # Potential bounce
                
            # Moving Average crossover (0-30 points)
            if pd.notna(latest['sma_signal']):
                if latest['sma_signal'] == 1:  # Short MA > Long MA
                    score += 20
                    
                    # Recent crossover bonus
                    recent_signals = df['sma_signal'].iloc[-10:]
                    if len(recent_signals) > 1 and recent_signals.iloc[-1] == 1 and recent_signals.iloc[-2] == -1:
                        score += 10  # Recent bullish crossover
            
            # Price above key moving averages (0-30 points)
            current_price = latest['close']
            if pd.notna(latest[f'sma_{self.sma_short}']):
                if current_price > latest[f'sma_{self.sma_short}']:
                    score += 15
            if pd.notna(latest[f'sma_{self.sma_long}']):
                if current_price > latest[f'sma_{self.sma_long}']:
                    score += 15
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating breakout score: {e}")
            return 0.0
    
    def _calculate_volume_score(self, latest: pd.Series, prev_data: pd.DataFrame) -> float:
        """Calculate volume confirmation score."""
        try:
            if pd.isna(latest['volume_ratio']):
                return 0.0
            
            volume_ratio = latest['volume_ratio']
            
            # Volume surge scoring
            if volume_ratio > 3.0:
                return 100.0  # Exceptional volume
            elif volume_ratio > 2.0:
                return 80.0   # Strong volume
            elif volume_ratio > self.volume_threshold:
                return 60.0   # Good volume
            elif volume_ratio > 1.0:
                return 40.0   # Above average volume
            else:
                return 20.0   # Below average volume
                
        except Exception as e:
            self.logger.warning(f"Error calculating volume score: {e}")
            return 0.0
    
    def _calculate_technical_momentum_score(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """Calculate technical momentum score."""
        try:
            score = 0.0
            
            # RSI scoring (0-50 points)
            rsi = latest.get('rsi', 50)
            if pd.notna(rsi):
                if 30 <= rsi <= 70:  # Healthy range
                    score += 30
                elif 70 < rsi <= 80:  # Overbought but strong
                    score += 20
                elif 20 <= rsi < 30:  # Oversold, potential bounce
                    score += 25
                else:  # Extreme levels
                    score += 10
            
            # Price momentum (0-50 points)
            if len(df) >= 20:
                current_price = latest['close']
                price_20d = df['close'].iloc[-20]
                
                if price_20d > 0:
                    momentum = (current_price - price_20d) / price_20d
                    if momentum > 0.1:  # >10% gain
                        score += 50
                    elif momentum > 0.05:  # >5% gain
                        score += 35
                    elif momentum > 0.02:  # >2% gain
                        score += 25
                    elif momentum > 0:  # Positive momentum
                        score += 15
                    else:  # Negative momentum
                        score += 5
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _calculate_trend_score(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """Calculate trend strength score."""
        try:
            # Trend consistency over last 20 days
            if len(df) < 20:
                return 50.0
            
            recent_signals = df['sma_signal'].iloc[-20:]
            if len(recent_signals) == 0:
                return 50.0
            
            # Count bullish signals
            bullish_ratio = (recent_signals == 1).sum() / len(recent_signals)
            
            return bullish_ratio * 100
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend score: {e}")
            return 50.0
    
    def _determine_action(self, score: float, data: Dict[str, Any]) -> tuple[SelectionAction, str]:
        """Determine recommended action based on score and technical patterns."""
        price_history = data.get('price_history')
        latest = price_history.iloc[-1] if price_history is not None else None
        
        # Base action on score
        if score >= 85:
            action = SelectionAction.STRONG_BUY
            reasoning = f"Strong technical breakout pattern (score: {score:.1f})"
        elif score >= 70:
            action = SelectionAction.BUY
            reasoning = f"Good breakout potential (score: {score:.1f})"
        elif score >= 50:
            action = SelectionAction.WATCH
            reasoning = f"Developing breakout pattern (score: {score:.1f})"
        else:
            action = SelectionAction.AVOID
            reasoning = f"Weak technical setup (score: {score:.1f})"
        
        # Add specific technical details
        if latest is not None:
            if pd.notna(latest.get('volume_ratio')) and latest['volume_ratio'] > 2.0:
                reasoning += "; strong volume confirmation"
            if pd.notna(latest.get('bb_position')) and latest['bb_position'] > 1.0:
                reasoning += "; Bollinger Band breakout"
            if latest.get('sma_signal') == 1:
                reasoning += "; bullish MA setup"
        
        return action, reasoning
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key technical metrics."""
        price_history = data.get('price_history')
        latest = price_history.iloc[-1] if price_history is not None else {}
        
        metrics = {
            'current_price': data.get('current_price', 0),
            'volume_ratio': latest.get('volume_ratio', 0),
            'rsi': latest.get('rsi', 0),
            'bb_position': latest.get('bb_position', 0),
            'sma_signal': latest.get('sma_signal', 0),
        }
        
        # Add moving averages
        if f'sma_{self.sma_short}' in latest:
            metrics[f'sma_{self.sma_short}'] = latest[f'sma_{self.sma_short}']
        if f'sma_{self.sma_long}' in latest:
            metrics[f'sma_{self.sma_long}'] = latest[f'sma_{self.sma_long}']
        
        return metrics
    
    def _calculate_confidence(self, score: float, data: Dict[str, Any]) -> float:
        """Calculate confidence level for the technical analysis."""
        price_history = data.get('price_history')
        
        confidence = 0.5  # Base confidence
        
        # More data = higher confidence
        if price_history is not None:
            if len(price_history) >= 120:  # 6 months data
                confidence += 0.2
            elif len(price_history) >= 60:  # 3 months data
                confidence += 0.1
        
        # Volume confirmation increases confidence
        latest = price_history.iloc[-1] if price_history is not None else {}
        volume_ratio = latest.get('volume_ratio', 0)
        if volume_ratio > 2.0:
            confidence += 0.2
        elif volume_ratio > 1.5:
            confidence += 0.1
        
        # High score with volume = high confidence
        if score > 80 and volume_ratio > 1.5:
            confidence += 0.1
        
        return min(1.0, confidence)