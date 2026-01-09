"""
Improved Value Momentum Strategy V2

Key improvements over original:
1. 12-1 month momentum (skips recent 21 days to avoid buying at peaks)
2. Enhanced value weighting (60% value vs 40% momentum, reduced from 40%/60%)
3. Overbought filtering (RSI > 75 reduces score)
4. Stricter valuation constraints (P/E > 25 severely penalized)

References:
- Jegadeesh & Titman (1993): 12-1 month momentum
- Asness et al. (2013): Value-momentum combination benefits
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


class ImprovedValueMomentumV2(BaseSelectionStrategy):
    """
    Improved Value Momentum strategy - addresses momentum-chasing issues.

    Key parameters:
    - value_weight: 0.6 (increased from 0.4)
    - momentum_weight: 0.4 (decreased from 0.6)
    - momentum_period_long: 252 days (12 months)
    - momentum_period_skip: 21 days (1 month to skip)
    - max_rsi_threshold: 75 (reduce score if exceeded)
    - max_acceptable_pe: 25 (strict P/E limit)
    """

    def __init__(self,
                 value_weight: float = 0.6,
                 momentum_weight: float = 0.4,
                 momentum_period_long: int = 252,
                 momentum_period_skip: int = 21,
                 max_rsi_threshold: float = 75,
                 max_acceptable_pe: float = 25):
        """
        Initialize Improved Value Momentum V2 strategy.

        Args:
            value_weight: Weight for value component (0-1)
            momentum_weight: Weight for momentum component (0-1)
            momentum_period_long: Long-term momentum period (days)
            momentum_period_skip: Recent period to skip (days)
            max_rsi_threshold: Max RSI before penalty
            max_acceptable_pe: Max acceptable P/E ratio
        """
        super().__init__(
            name="ImprovedValueMomentumV2",
            description="Improved value-momentum with 12-1 month momentum and overbought filter"
        )

        # Normalize weights
        total_weight = value_weight + momentum_weight
        self.value_weight = value_weight / total_weight
        self.momentum_weight = momentum_weight / total_weight

        self.momentum_period_long = momentum_period_long
        self.momentum_period_skip = momentum_period_skip
        self.max_rsi_threshold = max_rsi_threshold
        self.max_acceptable_pe = max_acceptable_pe

        logger.info(f"Initialized ImprovedValueMomentumV2 with value_weight={self.value_weight:.2f}, "
                    f"momentum_weight={self.momentum_weight:.2f}, "
                    f"momentum_period={momentum_period_long}-{momentum_period_skip}")

    def select_stocks(
        self,
        universe: List[str],
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """Select stocks using improved value momentum strategy."""
        start_time = time.time()
        criteria = self.validate_criteria(criteria)

        logger.info(f"[ImprovedValueMomentumV2] Running improved value momentum V2 on {len(universe)} stocks")
        logger.info(f"[ImprovedValueMomentumV2] Criteria - max_pe: {self.max_acceptable_pe}, max_rsi: {self.max_rsi_threshold}, min_score: {criteria.min_score_threshold}")
        logger.info(f"[ImprovedValueMomentumV2] Weights - value: {self.value_weight:.1%}, momentum: {self.momentum_weight:.1%}")

        # Filter universe
        filtered_universe = self.filter_universe(universe, criteria)
        logger.info(f"[ImprovedValueMomentumV2] After base filtering: {len(filtered_universe)} stocks (filtered out {len(universe) - len(filtered_universe)})")

        selected_stocks = []
        errors = []

        # Diagnostic counters
        no_data_count = 0
        insufficient_history_count = 0
        below_threshold_count = 0
        pe_penalties = 0
        rsi_penalties = 0
        score_distribution = []

        for symbol in filtered_universe:
            try:
                # Get extended data for long-term momentum
                data = self.get_extended_stock_data(symbol)
                if data is None:
                    no_data_count += 1
                    continue

                price_history = data.get('price_history')
                if price_history is None or len(price_history) < 100:
                    insufficient_history_count += 1
                    continue

                # Extract fundamentals for diagnostics
                fundamentals = data.get('fundamentals', {})
                pe_ratio = fundamentals.get('pe_ratio', 0)
                pb_ratio = fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0)

                # Calculate RSI for diagnostics
                rsi = self._calculate_rsi(price_history['close'])
                latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

                # Calculate strategy score
                score = self.calculate_score(symbol, data)
                score_distribution.append((symbol, score, pe_ratio, pb_ratio, latest_rsi))

                # Track penalties
                if pe_ratio > self.max_acceptable_pe:
                    pe_penalties += 1
                if latest_rsi > self.max_rsi_threshold:
                    rsi_penalties += 1

                # Log rejection reasons for diagnostic
                if score < criteria.min_score_threshold:
                    below_threshold_count += 1
                    # Log first 5 rejections in detail
                    if below_threshold_count <= 5:
                        logger.debug(f"[ImprovedValueMomentumV2] REJECTED {symbol}: score={score:.1f} (threshold={criteria.min_score_threshold}), "
                                   f"PE={pe_ratio:.1f} (max={self.max_acceptable_pe}), RSI={latest_rsi:.1f} (max={self.max_rsi_threshold})")
                    continue

                # Determine action and reasoning
                action, reasoning = self._determine_action(score, data)

                # Extract metrics
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
                logger.debug(f"[ImprovedValueMomentumV2] ACCEPTED {symbol}: score={score:.1f}, PE={pe_ratio:.1f}, RSI={latest_rsi:.1f}")

            except Exception as e:
                error_msg = f"Error processing {symbol}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

        # Log comprehensive diagnostics
        logger.info(f"[ImprovedValueMomentumV2] Candidates processed: {len(filtered_universe)}")
        logger.info(f"[ImprovedValueMomentumV2] No data available: {no_data_count}")
        logger.info(f"[ImprovedValueMomentumV2] Insufficient history (< 100 days): {insufficient_history_count}")
        logger.info(f"[ImprovedValueMomentumV2] Below score threshold: {below_threshold_count}")
        logger.info(f"[ImprovedValueMomentumV2] PE penalties applied: {pe_penalties}, RSI penalties applied: {rsi_penalties}")
        logger.info(f"[ImprovedValueMomentumV2] Final selections: {len(selected_stocks)}")

        # Log score distribution (top 10 and bottom 10)
        if score_distribution:
            score_distribution.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"[ImprovedValueMomentumV2] Score range: {score_distribution[0][1]:.1f} (best) to {score_distribution[-1][1]:.1f} (worst)")

            # Log top 5 scores
            logger.info(f"[ImprovedValueMomentumV2] Top 5 scores:")
            for i, (sym, scr, pe, pb, rsi) in enumerate(score_distribution[:5], 1):
                logger.info(f"  {i}. {sym}: score={scr:.1f}, PE={pe:.1f}, PB={pb:.1f}, RSI={rsi:.1f}")

            # Log threshold area (around min_score_threshold)
            threshold_area = [x for x in score_distribution if abs(x[1] - criteria.min_score_threshold) < 10]
            if threshold_area:
                logger.info(f"[ImprovedValueMomentumV2] Near threshold ({criteria.min_score_threshold}):")
                for sym, scr, pe, pb, rsi in threshold_area[:3]:
                    logger.info(f"  {sym}: score={scr:.1f}, PE={pe:.1f}, PB={pb:.1f}, RSI={rsi:.1f}")

        # Sort by score
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

        logger.info(f"[ImprovedValueMomentumV2] Selected {len(selected_stocks)} stocks in {execution_time:.2f}s")
        return results

    def get_extended_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get extended stock data for long-term momentum calculation."""
        try:
            from bot.data import fetch_history
            from bot.yahoo_data import fetch_yahoo_ticker_info

            # Get extended price history (need 252 + 21 days minimum)
            df = fetch_history(None, symbol, period='day', limit=300, dry_run=False)

            # Get fundamental data
            info = fetch_yahoo_ticker_info(symbol)

            if df is None or len(df) < 100:  # Need at least 100 days
                return None

            return {
                'symbol': symbol,
                'price_history': df,
                'fundamentals': info or {},
                'current_price': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
            }

        except Exception as e:
            logger.error(f"Failed to get extended data for {symbol}: {e}")
            return None

    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate improved value momentum score.

        Score = 60% Value + 40% Momentum (adjusted from 40/60)

        Enhancements:
        - Value component stricter on high P/E
        - Momentum uses 12-1 month period
        - Overbought penalty applied
        """
        try:
            fundamentals = data.get('fundamentals', {})
            price_history = data.get('price_history')

            if price_history is None or len(price_history) < 50:
                return 0.0

            # Calculate value score (enhanced)
            value_score = self._calculate_enhanced_value_score(fundamentals)

            # Calculate momentum score (12-1 month)
            momentum_score = self._calculate_long_term_momentum_score(price_history)

            # Apply overbought penalty
            overbought_penalty = self._calculate_overbought_penalty(price_history)

            # Combine scores
            final_score = (
                self.value_weight * value_score +
                self.momentum_weight * momentum_score
            ) * overbought_penalty  # Multiply by penalty (0.5-1.0)

            return min(100.0, max(0.0, final_score))

        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}")
            return 0.0

    def _calculate_enhanced_value_score(self, fundamentals: Dict[str, Any]) -> float:
        """Enhanced value scoring with stricter P/E constraints."""
        try:
            pe_ratio = fundamentals.get('pe_ratio', 0)
            pb_ratio = fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0)

            # P/E scoring (0-50 points) - STRICTER
            if pe_ratio <= 0 or pe_ratio > 100:
                pe_score = 0
            elif pe_ratio > self.max_acceptable_pe:  # > 25 严重扣分
                pe_score = max(0, 20 - (pe_ratio - self.max_acceptable_pe) * 2)
            elif pe_ratio < 10:  # Very attractive
                pe_score = 50
            elif pe_ratio < 15:  # Attractive
                pe_score = 45
            elif pe_ratio < 20:  # Reasonable
                pe_score = 35
            else:  # pe_ratio 20-25
                pe_score = 25

            # P/B scoring (0-50 points)
            if pb_ratio <= 0 or pb_ratio > 10:
                pb_score = 0
            elif pb_ratio < 1.0:  # Below book value
                pb_score = 50
            elif pb_ratio < 1.5:  # Reasonable
                pb_score = 45
            elif pb_ratio < 2.5:  # Moderate
                pb_score = 35
            elif pb_ratio < 4.0:  # High but acceptable
                pb_score = 25
            else:  # pb_ratio 4-10
                pb_score = 10

            value_score = (pe_score + pb_score) / 2
            return value_score

        except Exception as e:
            logger.warning(f"Error calculating enhanced value score: {e}")
            return 0.0

    def _calculate_long_term_momentum_score(self, price_history: pd.DataFrame) -> float:
        """Calculate 12-1 month momentum (252-21 days)."""
        try:
            if len(price_history) < (self.momentum_period_long + self.momentum_period_skip):
                # Fallback to shorter period
                if len(price_history) < 126:  # 6 months
                    return 50.0  # Neutral score

                # Use 6-1 month momentum
                current_price = price_history['close'].iloc[-1]
                skip_price = price_history['close'].iloc[-min(21, len(price_history)//2)]
                long_price = price_history['close'].iloc[-min(126, len(price_history)-1)]

                momentum_6m = (skip_price - long_price) / long_price if long_price > 0 else 0
            else:
                # Full 12-1 month momentum
                current_price = price_history['close'].iloc[-1]
                skip_price = price_history['close'].iloc[-self.momentum_period_skip]
                long_price = price_history['close'].iloc[-self.momentum_period_long]

                # Calculate return from 12 months ago to 1 month ago
                momentum_12_1 = (skip_price - long_price) / long_price if long_price > 0 else 0
                momentum_6m = momentum_12_1  # Use same var name

            # Volume momentum (confirmation)
            volume_momentum = self._calculate_volume_momentum(price_history)

            # Score momentum (-50% to +50% maps to 0-100)
            momentum_score = max(0, min(100, 50 + momentum_6m * 100))

            # Adjust for volume (70%-100% weight)
            momentum_score = momentum_score * (0.7 + 0.3 * volume_momentum)

            return momentum_score

        except Exception as e:
            logger.warning(f"Error calculating long-term momentum: {e}")
            return 50.0

    def _calculate_volume_momentum(self, price_history: pd.DataFrame) -> float:
        """Calculate volume momentum (0-1)."""
        try:
            if len(price_history) < 20:
                return 0.5

            recent_volume = price_history['volume'].iloc[-10:].mean()
            historical_volume = price_history['volume'].iloc[:-10].mean()

            if historical_volume == 0:
                return 0.5

            volume_ratio = recent_volume / historical_volume
            volume_momentum = min(1.0, max(0.0, (volume_ratio - 0.5) / 2 + 0.5))

            return volume_momentum

        except Exception as e:
            logger.warning(f"Error calculating volume momentum: {e}")
            return 0.5

    def _calculate_overbought_penalty(self, price_history: pd.DataFrame) -> float:
        """
        Calculate overbought penalty based on RSI and Bollinger Bands.

        Returns:
            Penalty multiplier (0.5-1.0)
            - 1.0 = no penalty
            - 0.5 = severe overbought, 50% score reduction
        """
        try:
            if len(price_history) < 50:
                return 1.0  # No penalty for insufficient data

            # Calculate RSI (14-period)
            rsi = self._calculate_rsi(price_history['close'])
            latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # Calculate Bollinger Band position
            bb_position = self._calculate_bb_position(price_history)

            penalty = 1.0

            # RSI penalty
            if latest_rsi > self.max_rsi_threshold:  # > 75
                excess = latest_rsi - self.max_rsi_threshold
                # Reduce by 2% for each point above 75
                penalty *= max(0.5, 1.0 - excess * 0.02)

            # Bollinger Band penalty (if above upper band)
            if bb_position > 1.2:  # 20% above upper band
                penalty *= 0.7  # 30% reduction
            elif bb_position > 1.1:  # 10% above upper band
                penalty *= 0.85  # 15% reduction

            return max(0.5, penalty)  # Floor at 50% of original score

        except Exception as e:
            logger.warning(f"Error calculating overbought penalty: {e}")
            return 1.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bb_position(self, price_history: pd.DataFrame, period: int = 20, std: float = 2.0) -> float:
        """Calculate position within Bollinger Bands (0-2, >1 = above upper band)."""
        try:
            close = price_history['close']
            sma = close.rolling(period).mean()
            std_dev = close.rolling(period).std()

            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)

            latest_close = close.iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]

            if pd.isna(latest_upper) or pd.isna(latest_lower) or latest_upper == latest_lower:
                return 0.5  # Neutral

            bb_position = (latest_close - latest_lower) / (latest_upper - latest_lower)
            return bb_position

        except Exception as e:
            logger.warning(f"Error calculating BB position: {e}")
            return 0.5

    def _determine_action(self, score: float, data: Dict[str, Any]) -> tuple:
        """Determine action based on score."""
        fundamentals = data.get('fundamentals', {})
        pe_ratio = fundamentals.get('pe_ratio', 0)

        if score >= 75:
            action = SelectionAction.STRONG_BUY
            reasoning = f"Excellent value-momentum (score: {score:.1f})"
        elif score >= 60:
            action = SelectionAction.BUY
            reasoning = f"Good value-momentum (score: {score:.1f})"
        elif score >= 45:
            action = SelectionAction.WATCH
            reasoning = f"Moderate potential (score: {score:.1f})"
        else:
            action = SelectionAction.AVOID
            reasoning = f"Weak profile (score: {score:.1f})"

        # Add valuation context
        if pe_ratio > 0 and pe_ratio < 15:
            reasoning += "; attractive valuation"
        elif pe_ratio > self.max_acceptable_pe:
            reasoning += f"; high P/E ({pe_ratio:.1f})"

        return action, reasoning

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics."""
        fundamentals = data.get('fundamentals', {})
        price_history = data.get('price_history')

        metrics = {
            'current_price': data.get('current_price', 0),
            'pe_ratio': fundamentals.get('pe_ratio', 0),
            'pb_ratio': fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0),
            'market_cap': fundamentals.get('market_cap', 0),
        }

        # Add momentum metrics
        if price_history is not None and len(price_history) >= 50:
            current_price = price_history['close'].iloc[-1]

            if len(price_history) >= 21:
                price_1m_ago = price_history['close'].iloc[-21]
                metrics['return_1m'] = (current_price - price_1m_ago) / price_1m_ago if price_1m_ago > 0 else 0

            if len(price_history) >= 126:
                price_6m_ago = price_history['close'].iloc[-126]
                metrics['return_6m'] = (current_price - price_6m_ago) / price_6m_ago if price_6m_ago > 0 else 0

            # RSI
            rsi = self._calculate_rsi(price_history['close'])
            metrics['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        return metrics

    def _calculate_confidence(self, score: float, data: Dict[str, Any]) -> float:
        """Calculate confidence level."""
        fundamentals = data.get('fundamentals', {})
        price_history = data.get('price_history')

        confidence = 0.5

        # Data completeness
        if fundamentals.get('pe_ratio', 0) > 0:
            confidence += 0.15
        if price_history is not None and len(price_history) >= 252:
            confidence += 0.2
        if fundamentals.get('market_cap', 0) > 10e9:
            confidence += 0.15

        return min(1.0, confidence)
