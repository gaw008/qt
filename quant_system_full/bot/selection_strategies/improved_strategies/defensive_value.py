"""
Defensive Value Strategy

Pure value strategy for style diversification and counter-cyclical holdings.
Focuses on undervalued stocks with strong fundamentals, ignoring momentum.

Key features:
- No momentum component (pure value)
- Focus on low P/E, P/B, high dividend yield
- Debt-to-equity ratio consideration
- Targets overlooked/underpriced stocks

This strategy complements momentum strategies by providing:
- Downside protection in momentum crashes
- Style diversification
- Counter-cyclical opportunities
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


class DefensiveValue(BaseSelectionStrategy):
    """
    Defensive Value Strategy - pure value without momentum.

    Scoring weights:
    - P/E Ratio: 40%
    - P/B Ratio: 30%
    - Dividend Yield: 20%
    - Debt-to-Equity: 10%

    Target: Overlooked value stocks that may lag momentum but provide
    downside protection and eventual mean reversion.
    """

    def __init__(self,
                 pe_weight: float = 0.4,
                 pb_weight: float = 0.3,
                 dividend_weight: float = 0.2,
                 debt_weight: float = 0.1,
                 max_pe: float = 15,
                 min_dividend_yield: float = 0.02):
        """
        Initialize Defensive Value strategy.

        Args:
            pe_weight: Weight for P/E component
            pb_weight: Weight for P/B component
            dividend_weight: Weight for dividend yield
            debt_weight: Weight for debt-to-equity
            max_pe: Maximum acceptable P/E
            min_dividend_yield: Minimum dividend yield (2%)
        """
        super().__init__(
            name="DefensiveValue",
            description="Pure value strategy for style diversification"
        )

        # Normalize weights
        total = pe_weight + pb_weight + dividend_weight + debt_weight
        self.pe_weight = pe_weight / total
        self.pb_weight = pb_weight / total
        self.dividend_weight = dividend_weight / total
        self.debt_weight = debt_weight / total

        self.max_pe = max_pe
        self.min_dividend_yield = min_dividend_yield

        logger.info(f"Initialized DefensiveValue with max_pe={max_pe}, "
                    f"min_div_yield={min_dividend_yield:.2%}")

    def select_stocks(
        self,
        universe: List[str],
        criteria: Optional[SelectionCriteria] = None
    ) -> StrategyResults:
        """Select defensive value stocks."""
        start_time = time.time()
        criteria = self.validate_criteria(criteria)

        logger.info(f"[DefensiveValue] Running defensive value selection on {len(universe)} stocks")
        logger.info(f"[DefensiveValue] Criteria - max_pe: {self.max_pe}, min_dividend: {self.min_dividend_yield:.2%}, min_score: {criteria.min_score_threshold}")

        filtered_universe = self.filter_universe(universe, criteria)
        logger.info(f"[DefensiveValue] After base filtering: {len(filtered_universe)} stocks (filtered out {len(universe) - len(filtered_universe)})")

        selected_stocks = []
        errors = []

        # Diagnostic counters
        no_data_count = 0
        below_threshold_count = 0
        pe_failures = 0
        dividend_failures = 0
        score_distribution = []

        for symbol in filtered_universe:
            try:
                data = self.get_stock_data(symbol)
                if data is None:
                    no_data_count += 1
                    continue

                # Extract fundamentals for diagnostics
                fundamentals = data.get('fundamentals', {})
                pe_ratio = fundamentals.get('pe_ratio', 0) or fundamentals.get('trailingPE', 0)
                pb_ratio = fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0)
                dividend_yield = fundamentals.get('dividend_yield', 0) or fundamentals.get('dividendYield', 0)

                score = self.calculate_score(symbol, data)
                score_distribution.append((symbol, score, pe_ratio, pb_ratio, dividend_yield))

                # Log rejection reasons for diagnostic
                if score < criteria.min_score_threshold:
                    below_threshold_count += 1
                    if pe_ratio > self.max_pe or pe_ratio <= 0:
                        pe_failures += 1
                    if dividend_yield < self.min_dividend_yield:
                        dividend_failures += 1
                    # Log first 5 rejections in detail
                    if below_threshold_count <= 5:
                        logger.debug(f"[DefensiveValue] REJECTED {symbol}: score={score:.1f} (threshold={criteria.min_score_threshold}), "
                                   f"PE={pe_ratio:.1f} (max={self.max_pe}), div={dividend_yield:.2%} (min={self.min_dividend_yield:.2%})")
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
                logger.debug(f"[DefensiveValue] ACCEPTED {symbol}: score={score:.1f}, PE={pe_ratio:.1f}, div={dividend_yield:.2%}")

            except Exception as e:
                error_msg = f"Error processing {symbol}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

        # Log comprehensive diagnostics
        logger.info(f"[DefensiveValue] Candidates processed: {len(filtered_universe)}")
        logger.info(f"[DefensiveValue] No data available: {no_data_count}")
        logger.info(f"[DefensiveValue] Below score threshold: {below_threshold_count} (PE failures: {pe_failures}, Dividend failures: {dividend_failures})")
        logger.info(f"[DefensiveValue] Final selections: {len(selected_stocks)}")

        # Log score distribution (top 10 and bottom 10)
        if score_distribution:
            score_distribution.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"[DefensiveValue] Score range: {score_distribution[0][1]:.1f} (best) to {score_distribution[-1][1]:.1f} (worst)")

            # Log top 5 scores
            logger.info(f"[DefensiveValue] Top 5 scores:")
            for i, (sym, scr, pe, pb, div) in enumerate(score_distribution[:5], 1):
                logger.info(f"  {i}. {sym}: score={scr:.1f}, PE={pe:.1f}, PB={pb:.1f}, div={div:.2%}")

            # Log threshold area (around min_score_threshold)
            threshold_area = [x for x in score_distribution if abs(x[1] - criteria.min_score_threshold) < 10]
            if threshold_area:
                logger.info(f"[DefensiveValue] Near threshold ({criteria.min_score_threshold}):")
                for sym, scr, pe, pb, div in threshold_area[:3]:
                    logger.info(f"  {sym}: score={scr:.1f}, PE={pe:.1f}, PB={pb:.1f}, div={div:.2%}")

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

        logger.info(f"[DefensiveValue] Selected {len(selected_stocks)} defensive value stocks in {execution_time:.2f}s")
        return results

    def calculate_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Calculate pure value score (no momentum).

        Score = 40% P/E + 30% P/B + 20% Dividend + 10% Debt
        """
        try:
            fundamentals = data.get('fundamentals', {})

            pe_score = self._calculate_pe_score(fundamentals)
            pb_score = self._calculate_pb_score(fundamentals)
            dividend_score = self._calculate_dividend_score(fundamentals)
            debt_score = self._calculate_debt_score(fundamentals)

            final_score = (
                self.pe_weight * pe_score +
                self.pb_weight * pb_score +
                self.dividend_weight * dividend_score +
                self.debt_weight * debt_score
            )

            return min(100.0, max(0.0, final_score))

        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}")
            return 0.0

    def _calculate_pe_score(self, fundamentals: Dict[str, Any]) -> float:
        """P/E score: lower is better."""
        pe_ratio = fundamentals.get('pe_ratio', 0) or fundamentals.get('trailingPE', 0)

        if pe_ratio <= 0 or pe_ratio > 50:
            return 0

        # Excellent value: P/E < 8
        if pe_ratio < 8:
            return 100
        # Very good: P/E 8-12
        elif pe_ratio < 12:
            return 90
        # Good: P/E 12-15
        elif pe_ratio <= self.max_pe:
            return 75
        # Acceptable: P/E 15-20
        elif pe_ratio < 20:
            return 50
        # Moderate: P/E 20-30
        elif pe_ratio < 30:
            return 25
        # High: P/E 30-50
        else:
            return 10

    def _calculate_pb_score(self, fundamentals: Dict[str, Any]) -> float:
        """P/B score: lower is better."""
        pb_ratio = fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0)

        if pb_ratio <= 0 or pb_ratio > 20:
            return 0

        # Excellent: P/B < 0.8 (below book value)
        if pb_ratio < 0.8:
            return 100
        # Very good: P/B 0.8-1.2
        elif pb_ratio < 1.2:
            return 90
        # Good: P/B 1.2-2.0
        elif pb_ratio < 2.0:
            return 75
        # Acceptable: P/B 2.0-3.0
        elif pb_ratio < 3.0:
            return 50
        # Moderate: P/B 3.0-5.0
        elif pb_ratio < 5.0:
            return 30
        # High: P/B 5.0-20
        else:
            return 10

    def _calculate_dividend_score(self, fundamentals: Dict[str, Any]) -> float:
        """Dividend yield score: higher is better."""
        dividend_yield = fundamentals.get('dividend_yield', 0) or fundamentals.get('dividendYield', 0)

        if dividend_yield <= 0:
            return 20  # Base score for non-dividend stocks

        # Excellent: >5% yield
        if dividend_yield > 0.05:
            return 100
        # Very good: 4-5%
        elif dividend_yield > 0.04:
            return 90
        # Good: 3-4%
        elif dividend_yield > 0.03:
            return 75
        # Acceptable: 2-3%
        elif dividend_yield >= self.min_dividend_yield:
            return 60
        # Low: 1-2%
        elif dividend_yield > 0.01:
            return 40
        # Very low: <1%
        else:
            return 25

    def _calculate_debt_score(self, fundamentals: Dict[str, Any]) -> float:
        """Debt-to-equity score: lower is better."""
        debt_to_equity = fundamentals.get('debtToEquity', 0) or fundamentals.get('debt_to_equity', 0)

        if debt_to_equity < 0:
            return 50  # Unknown, neutral

        # Excellent: < 0.3 (very low debt)
        if debt_to_equity < 0.3:
            return 100
        # Very good: 0.3-0.5
        elif debt_to_equity < 0.5:
            return 90
        # Good: 0.5-1.0
        elif debt_to_equity < 1.0:
            return 75
        # Acceptable: 1.0-2.0
        elif debt_to_equity < 2.0:
            return 50
        # High: 2.0-3.0
        elif debt_to_equity < 3.0:
            return 25
        # Very high: > 3.0
        else:
            return 10

    def _determine_action(self, score: float, data: Dict[str, Any]) -> tuple:
        """Determine action for defensive value stock."""
        fundamentals = data.get('fundamentals', {})
        pe_ratio = fundamentals.get('pe_ratio', 0)
        dividend_yield = fundamentals.get('dividend_yield', 0)

        if score >= 80:
            action = SelectionAction.STRONG_BUY
            reasoning = f"Exceptional value opportunity (score: {score:.1f})"
        elif score >= 65:
            action = SelectionAction.BUY
            reasoning = f"Strong value characteristics (score: {score:.1f})"
        elif score >= 50:
            action = SelectionAction.WATCH
            reasoning = f"Moderate value potential (score: {score:.1f})"
        else:
            action = SelectionAction.AVOID
            reasoning = f"Insufficient value appeal (score: {score:.1f})"

        # Add specific details
        if pe_ratio > 0 and pe_ratio < 10:
            reasoning += f"; very low P/E ({pe_ratio:.1f})"
        if dividend_yield > 0.03:
            reasoning += f"; attractive dividend ({dividend_yield:.2%})"

        return action, reasoning

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract value metrics."""
        fundamentals = data.get('fundamentals', {})

        return {
            'current_price': data.get('current_price', 0),
            'pe_ratio': fundamentals.get('pe_ratio', 0),
            'pb_ratio': fundamentals.get('pb_ratio', 0) or fundamentals.get('priceToBook', 0),
            'dividend_yield': fundamentals.get('dividend_yield', 0) or fundamentals.get('dividendYield', 0),
            'debt_to_equity': fundamentals.get('debtToEquity', 0) or fundamentals.get('debt_to_equity', 0),
            'market_cap': fundamentals.get('market_cap', 0),
        }

    def _calculate_confidence(self, score: float, data: Dict[str, Any]) -> float:
        """Calculate confidence for defensive value."""
        fundamentals = data.get('fundamentals', {})

        confidence = 0.5

        # Data completeness
        if fundamentals.get('pe_ratio', 0) > 0:
            confidence += 0.2
        if fundamentals.get('pb_ratio', 0) > 0:
            confidence += 0.15
        if fundamentals.get('dividend_yield', 0) > 0:
            confidence += 0.1
        if fundamentals.get('market_cap', 0) > 5e9:  # Large cap
            confidence += 0.05

        return min(1.0, confidence)
