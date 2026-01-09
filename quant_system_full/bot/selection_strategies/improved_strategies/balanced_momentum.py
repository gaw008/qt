"""
Balanced Momentum Strategy

Medium to long-term momentum strategy with sustained volume confirmation.
Avoids short-term noise by emphasizing 6-12 month trends.

Key features:
- 6-month momentum: 50%
- 12-month momentum: 30%
- 3-month momentum: 20%
- Volume consistency requirement
- Avoids false breakouts through sustained volume analysis

This strategy complements:
- Value strategies (provides growth component)
- Short-term momentum (reduces whipsaw risk)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import logging

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from bot.selection_strategies.base_strategy import (
    BaseSelectionStrategy, SelectionResult, SelectionCriteria,
    StrategyResults, SelectionAction
)

logger = logging.getLogger(__name__)


class BalancedMomentum(BaseSelectionStrategy):
    """
    Balanced Momentum Strategy - medium to long-term trends.

    Momentum weights:
    - 6-month: 50%
    - 12-month: 30%
    - 3-month: 20%

    Volume consistency: Required for high scores
    """

    def __init__(self,
                 momentum_6m_weight: float = 0.5,
                 momentum_12m_weight: float = 0.3,
                 momentum_3m_weight: float = 0.2,
                 require_sustained_volume: bool = True,
                 volume_consistency_window: int = 60):
        """
        Initialize Balanced Momentum strategy.

        Args:
            momentum_6m_weight: Weight for 6-month momentum
            momentum_12m_weight: Weight for 12-month momentum
            momentum_3m_weight: Weight for 3-month momentum
            require_sustained_volume: Require consistent volume
            volume_consistency_window: Days to check volume consistency
        """
        super().__init__(
            name="BalancedMomentum",
            description="Medium-long term momentum with volume confirmation"
        )

        # Normalize weights
        total = momentum_6m_weight + momentum_12m_weight + momentum_3m_weight
        self.momentum_6m_weight = momentum_6m_weight / total
        self.momentum_12m_weight = momentum_12m_weight / total
        self.momentum_3m_weight = momentum_3m_weight / total

        self.require_sustained_volume = require_sustained_volume
        self.volume_consistency_window = volume_consistency_window

        logger.info(f"Initialized BalancedMomentum with 6m:{self.momentum_6m_weight:.1%}, "
                    f"12m:{self.momentum_12m_weight:.1%}, 3m:{self.momentum_3m_weight:.1%}")

    def select_stocks(
        self,
        universe: List[str],
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """Select balanced momentum stocks."""
        start_time = time.time()
        criteria = self.validate_criteria(criteria)

        logger.info(f"Running balanced momentum selection on {len(universe)} stocks")

        filtered_universe = self.filter_universe(universe, criteria)

        selected_stocks = []
        errors = []

        for symbol in filtered_universe:
            try:
                # Get extended data for long-term momentum
                data = self.get_extended_stock_data(symbol)
                if data is None:
                    continue

                score = self.calculate_score(symbol, data)
                if score < criteria.min_score_threshold:
                    continue

                action, reasoning = self._determine_action(score, data)
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
                logger.warning(error_msg)
                errors.append(error_msg)

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

        logger.info(f"Selected {len(selected_stocks)} balanced momentum stocks in {execution_time:.2f}s")
        return results

    def get_extended_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get extended data for momentum calculation."""
        try:
            from bot.data import fetch_history
            from bot.yahoo_data import fetch_yahoo_ticker_info

            # Need 252+ days for 12-month momentum
            df = fetch_history(None, symbol, period='day', limit=300, dry_run=False)

            info = fetch_yahoo_ticker_info(symbol)

            if df is None or len(df) < 100:
                return None

            return {
                'symbol': symbol,
                'price_history': df,
                'fundamentals': info or {},
                'current_price': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
            }

        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            return None

    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate balanced momentum score.

        Score = 50% (6m) + 30% (12m) + 20% (3m) × Volume Consistency
        """
        try:
            price_history = data.get('price_history')

            if price_history is None or len(price_history) < 60:
                return 0.0

            # Calculate momentum components
            momentum_3m = self._calculate_momentum_return(price_history, 63)   # 3 months
            momentum_6m = self._calculate_momentum_return(price_history, 126)  # 6 months
            momentum_12m = self._calculate_momentum_return(price_history, 252) # 12 months

            # Weight and combine
            base_score = (
                self.momentum_6m_weight * momentum_6m +
                self.momentum_12m_weight * momentum_12m +
                self.momentum_3m_weight * momentum_3m
            )

            # Volume consistency adjustment
            if self.require_sustained_volume:
                volume_quality = self._calculate_volume_quality(price_history)
                final_score = base_score * volume_quality
            else:
                final_score = base_score

            return min(100.0, max(0.0, final_score))

        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}")
            return 0.0

    def _calculate_momentum_return(self, price_history: pd.DataFrame, period: int) -> float:
        """Calculate momentum return score for given period."""
        try:
            if len(price_history) < period:
                # Use shorter period as fallback
                period = min(len(price_history) - 1, 63)  # At least 3 months

            if period < 20:
                return 50.0  # Neutral

            current_price = price_history['close'].iloc[-1]
            past_price = price_history['close'].iloc[-period]

            if past_price <= 0:
                return 50.0

            momentum_return = (current_price - past_price) / past_price

            # Map -50% to +50% return to 0-100 score
            # Neutral (0% return) = 50 score
            # +25% return = 100 score
            # -25% return = 0 score
            score = max(0, min(100, 50 + momentum_return * 200))

            return score

        except Exception as e:
            logger.warning(f"Error calculating momentum return: {e}")
            return 50.0

    def _calculate_volume_quality(self, price_history: pd.DataFrame) -> float:
        """
        Calculate volume quality score (0.5-1.0).

        Assesses:
        - Volume consistency (not erratic)
        - Volume trend (increasing preferred)
        - Volume relative to historical average

        Returns:
            Quality multiplier (0.5-1.0)
        """
        try:
            if len(price_history) < self.volume_consistency_window:
                return 0.8  # Neutral penalty

            volume = price_history['volume'].iloc[-self.volume_consistency_window:]

            # 1. Consistency score (lower std dev = higher consistency)
            volume_mean = volume.mean()
            volume_std = volume.std()

            if volume_mean == 0:
                return 0.6

            coefficient_of_variation = volume_std / volume_mean

            # CV < 0.5 = very consistent (0.95-1.0)
            # CV 0.5-1.0 = moderate (0.8-0.95)
            # CV > 1.0 = erratic (0.5-0.8)
            if coefficient_of_variation < 0.5:
                consistency_score = 1.0
            elif coefficient_of_variation < 1.0:
                consistency_score = 0.95 - (coefficient_of_variation - 0.5) * 0.3
            else:
                consistency_score = max(0.5, 0.8 - (coefficient_of_variation - 1.0) * 0.15)

            # 2. Trend score (is volume increasing?)
            recent_volume = volume.iloc[-20:].mean()  # Recent 20 days
            older_volume = volume.iloc[:-20].mean()   # Earlier period

            if older_volume > 0:
                volume_trend_ratio = recent_volume / older_volume
                # Increasing volume = good (up to 1.0)
                # Stable volume = acceptable (0.9)
                # Decreasing volume = concern (0.7-0.9)
                if volume_trend_ratio > 1.2:
                    trend_score = 1.0
                elif volume_trend_ratio > 1.0:
                    trend_score = 0.95
                elif volume_trend_ratio > 0.8:
                    trend_score = 0.9
                else:
                    trend_score = 0.7 + (volume_trend_ratio - 0.5) * 0.4
            else:
                trend_score = 0.8

            # 3. Relative volume score
            current_volume = price_history['volume'].iloc[-1]
            avg_volume = volume_mean

            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

            if relative_volume > 1.5:  # Strong interest
                relative_score = 1.0
            elif relative_volume > 1.0:  # Above average
                relative_score = 0.95
            elif relative_volume > 0.7:  # Acceptable
                relative_score = 0.9
            else:  # Low volume
                relative_score = 0.7

            # Combine scores
            volume_quality = (
                consistency_score * 0.4 +
                trend_score * 0.3 +
                relative_score * 0.3
            )

            return max(0.5, min(1.0, volume_quality))

        except Exception as e:
            logger.warning(f"Error calculating volume quality: {e}")
            return 0.8

    def _determine_action(self, score: float, data: Dict[str, Any]) -> tuple:
        """Determine action for balanced momentum stock."""
        price_history = data.get('price_history')

        if score >= 80:
            action = SelectionAction.STRONG_BUY
            reasoning = f"Strong sustained momentum (score: {score:.1f})"
        elif score >= 65:
            action = SelectionAction.BUY
            reasoning = f"Good momentum trend (score: {score:.1f})"
        elif score >= 50:
            action = SelectionAction.WATCH
            reasoning = f"Developing momentum (score: {score:.1f})"
        else:
            action = SelectionAction.AVOID
            reasoning = f"Weak momentum profile (score: {score:.1f})"

        # Add volume context
        if price_history is not None and len(price_history) >= 60:
            volume_quality = self._calculate_volume_quality(price_history)
            if volume_quality > 0.9:
                reasoning += "; strong volume confirmation"
            elif volume_quality < 0.7:
                reasoning += "; weak volume"

        return action, reasoning

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract momentum metrics."""
        price_history = data.get('price_history')

        metrics = {
            'current_price': data.get('current_price', 0),
            'current_volume': data.get('volume', 0),
        }

        if price_history is not None:
            current_price = price_history['close'].iloc[-1]

            # Calculate returns
            if len(price_history) >= 63:
                price_3m = price_history['close'].iloc[-63]
                metrics['return_3m'] = (current_price - price_3m) / price_3m if price_3m > 0 else 0

            if len(price_history) >= 126:
                price_6m = price_history['close'].iloc[-126]
                metrics['return_6m'] = (current_price - price_6m) / price_6m if price_6m > 0 else 0

            if len(price_history) >= 252:
                price_12m = price_history['close'].iloc[-252]
                metrics['return_12m'] = (current_price - price_12m) / price_12m if price_12m > 0 else 0

            # Volume quality
            if len(price_history) >= 60:
                metrics['volume_quality'] = self._calculate_volume_quality(price_history)

        return metrics

    def _calculate_confidence(self, score: float, data: Dict[str, Any]) -> float:
        """Calculate confidence for balanced momentum."""
        price_history = data.get('price_history')

        confidence = 0.5

        # Data completeness
        if price_history is not None:
            if len(price_history) >= 252:
                confidence += 0.3  # Full year data
            elif len(price_history) >= 126:
                confidence += 0.2  # 6 months data
            else:
                confidence += 0.1  # Basic data

        # Volume quality adds confidence
        if price_history is not None and len(price_history) >= 60:
            volume_quality = self._calculate_volume_quality(price_history)
            confidence += volume_quality * 0.2

        return min(1.0, confidence)
