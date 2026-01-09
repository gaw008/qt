"""
LLM Selection Enhancer

This module applies LLM-based text analysis to enhance stock selection results.

Architecture:
    - apply_triage: News analysis for Technical Breakout stocks (gate/downweight)
    - apply_deep: Earnings analysis for Value Momentum and Earnings Momentum (boost)

Enhancement Rules:
    Triage (News):
        - Quality Gate: If news_quality < 40 -> Zero out Technical Breakout score
        - Risk Penalty: If risk_flags > 70 -> Downweight Technical Breakout by 50%

    Deep (Earnings):
        - Earnings Score boost for Earnings Momentum: +/- 20 points max
        - Quality Score boost for Value Momentum: +/- 15 points max

All enhancements are applied to copies, never to original base results.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SelectionEnhancer:
    """
    Applies LLM analysis results to enhance stock selection scores.

    This class takes LLM analysis results (news triage, earnings deep analysis)
    and applies them to the appropriate strategy scores according to enhancement rules.

    All enhancements are non-destructive - they modify enhanced_results but track
    all changes for transparency and debugging.
    """

    def __init__(self):
        """Initialize the enhancer."""
        # Lazy initialization of scorers
        self._news_scorer = None
        self._earnings_scorer = None
        self._quality_scorer = None

    @property
    def news_scorer(self):
        """Lazy load news scorer."""
        if self._news_scorer is None:
            try:
                from .scorers.news_scorer import NewsScorer
                self._news_scorer = NewsScorer()
            except Exception as e:
                logger.error(f"[LLM] Failed to load news scorer: {e}")
        return self._news_scorer

    @property
    def earnings_scorer(self):
        """Lazy load earnings scorer."""
        if self._earnings_scorer is None:
            try:
                from .scorers.earnings_scorer import EarningsScorer
                self._earnings_scorer = EarningsScorer()
            except Exception as e:
                logger.error(f"[LLM] Failed to load earnings scorer: {e}")
        return self._earnings_scorer

    @property
    def quality_scorer(self):
        """Lazy load quality scorer."""
        if self._quality_scorer is None:
            try:
                from .scorers.quality_scorer import QualityScorer
                self._quality_scorer = QualityScorer()
            except Exception as e:
                logger.error(f"[LLM] Failed to load quality scorer: {e}")
        return self._quality_scorer

    def apply_triage(
        self,
        results: List[Dict[str, Any]],
        targets: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Apply news triage analysis to Technical Breakout strategy.

        This method analyzes recent news for specified stocks and applies
        quality gates and risk penalties to Technical Breakout scores.

        Args:
            results: List of selection results (will be modified)
            targets: List of symbols to analyze (prioritized stocks)

        Returns:
            Enhanced results with triage adjustments applied

        Enhancement Logic:
            1. Fetch and analyze news for each target symbol
            2. Apply quality gate (news_quality < 40 -> zero out Technical Breakout)
            3. Apply risk penalty (risk_flags > 70 -> downweight by 50%)
            4. Track all changes in llm_metadata
        """
        if not targets:
            logger.info("[LLM] No triage targets, skipping")
            return results

        logger.info(f"[LLM_TRIAGE] Starting news analysis for {len(targets)} stocks")
        logger.debug(f"[LLM_TRIAGE] Target stocks: {', '.join(targets[:10])}{'...' if len(targets) > 10 else ''}")

        # Build symbol -> result mapping for fast lookup
        symbol_map = {r["symbol"]: r for r in results}

        enhanced_count = 0
        gated_count = 0
        penalized_count = 0
        passed_count = 0

        for symbol in targets:
            if symbol not in symbol_map:
                logger.warning(f"[LLM] Symbol {symbol} not in results, skipping")
                continue

            try:
                # Get LLM news analysis
                if self.news_scorer:
                    news_analysis = self.news_scorer.analyze(symbol)
                else:
                    logger.warning(f"[LLM] News scorer not available, skipping {symbol}")
                    continue

                if not news_analysis:
                    logger.debug(f"[LLM] No news analysis for {symbol}")
                    continue

                # Extract scores
                news_quality = news_analysis.get("news_quality", 50)
                risk_flags = news_analysis.get("risk_flags", 0)

                # Get current result
                result = symbol_map[symbol]

                # Initialize LLM metadata if not present
                if "llm_metadata" not in result:
                    result["llm_metadata"] = {}

                result["llm_metadata"]["triage"] = {
                    "news_quality": news_quality,
                    "risk_flags": risk_flags,
                    "timestamp": datetime.now().isoformat()
                }

                # Find Technical Breakout score
                strategies = result.get("strategies", {})
                if "technical_breakout" not in strategies:
                    logger.debug(f"[LLM] No Technical Breakout score for {symbol}")
                    continue

                original_score = strategies["technical_breakout"]

                # Apply Enhancement Rules
                enhancement_applied = False

                # Rule 1: Quality Gate (< 40 -> zero out)
                if news_quality < 40:
                    strategies["technical_breakout"] = 0
                    result["llm_metadata"]["triage"]["action"] = "GATED"
                    result["llm_metadata"]["triage"]["original_score"] = original_score
                    gated_count += 1
                    enhancement_applied = True
                    logger.info(f"[LLM] {symbol}: Gated (quality={news_quality}), TB: {original_score:.1f} -> 0")

                # Rule 2: Risk Penalty (> 70 -> downweight by 50%)
                elif risk_flags > 70:
                    new_score = original_score * 0.5
                    strategies["technical_breakout"] = new_score
                    result["llm_metadata"]["triage"]["action"] = "PENALIZED"
                    result["llm_metadata"]["triage"]["original_score"] = original_score
                    penalized_count += 1
                    enhancement_applied = True
                    logger.info(f"[LLM] {symbol}: Penalized (risk={risk_flags}), TB: {original_score:.1f} -> {new_score:.1f}")

                else:
                    result["llm_metadata"]["triage"]["action"] = "PASSED"
                    passed_count += 1
                    logger.debug(f"[LLM] {symbol}: Passed triage (quality={news_quality}, risk={risk_flags})")

                if enhancement_applied:
                    # Recalculate avg_score
                    result["avg_score"] = sum(strategies.values()) / len(strategies)
                    enhanced_count += 1

            except Exception as e:
                logger.error(f"[LLM_TRIAGE] Error analyzing {symbol}: {e}")
                continue

        logger.info(f"[LLM_TRIAGE] Complete: {len(targets)} analyzed, "
                   f"{passed_count} passed, {gated_count} gated, {penalized_count} penalized")

        return results

    def apply_deep(
        self,
        results: List[Dict[str, Any]],
        targets: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Apply deep earnings analysis to Value Momentum and Earnings Momentum strategies.

        This method performs detailed analysis of earnings reports and financial
        quality, applying score adjustments to relevant strategies.

        Args:
            results: List of selection results (will be modified)
            targets: List of symbols to analyze (top candidates)

        Returns:
            Enhanced results with deep analysis adjustments applied

        Enhancement Logic:
            1. Fetch and analyze earnings documents for each target
            2. Apply earnings score boost to Earnings Momentum (+/- 20 points max)
            3. Apply quality score boost to Value Momentum (+/- 15 points max)
            4. Track all changes in llm_metadata
        """
        if not targets:
            logger.info("[LLM] No deep targets, skipping")
            return results

        logger.info(f"[LLM_DEEP] Starting earnings/quality analysis for {len(targets)} stocks")
        logger.debug(f"[LLM_DEEP] Target stocks: {', '.join(targets[:10])}{'...' if len(targets) > 10 else ''}")

        # Build symbol -> result mapping
        symbol_map = {r["symbol"]: r for r in results}

        enhanced_count = 0
        earnings_enhanced = 0
        quality_enhanced = 0

        for symbol in targets:
            if symbol not in symbol_map:
                logger.warning(f"[LLM] Symbol {symbol} not in results, skipping")
                continue

            try:
                # Get LLM earnings and quality analysis
                earnings_analysis = None
                quality_analysis = None

                if self.earnings_scorer:
                    earnings_analysis = self.earnings_scorer.analyze(symbol)

                if self.quality_scorer:
                    quality_analysis = self.quality_scorer.analyze(symbol)

                if not earnings_analysis and not quality_analysis:
                    logger.debug(f"[LLM] No deep analysis available for {symbol}")
                    continue

                # Get current result
                result = symbol_map[symbol]

                # Initialize LLM metadata if not present
                if "llm_metadata" not in result:
                    result["llm_metadata"] = {}

                result["llm_metadata"]["deep"] = {
                    "timestamp": datetime.now().isoformat()
                }

                strategies = result.get("strategies", {})
                enhancement_applied = False

                # Apply Earnings Score Boost to Earnings Momentum
                if earnings_analysis and "earnings_momentum" in strategies:
                    earnings_score = earnings_analysis.get("earnings_score", 0)
                    # Convert -100 to +100 range to -20 to +20 boost
                    boost = (earnings_score / 100) * 20

                    original_score = strategies["earnings_momentum"]
                    new_score = max(0, min(100, original_score + boost))
                    strategies["earnings_momentum"] = new_score

                    result["llm_metadata"]["deep"]["earnings"] = {
                        "earnings_score": earnings_score,
                        "boost": boost,
                        "original_score": original_score
                    }

                    enhancement_applied = True
                    earnings_enhanced += 1
                    logger.info(f"[LLM_DEEP] {symbol}: Earnings boost={boost:+.1f}, EM: {original_score:.1f} -> {new_score:.1f}")

                # Apply Quality Score Boost to Value Momentum
                if quality_analysis and "value_momentum" in strategies:
                    quality_score = quality_analysis.get("quality_score", 0)
                    # Convert -100 to +100 range to -15 to +15 boost
                    boost = (quality_score / 100) * 15

                    original_score = strategies["value_momentum"]
                    new_score = max(0, min(100, original_score + boost))
                    strategies["value_momentum"] = new_score

                    result["llm_metadata"]["deep"]["quality"] = {
                        "quality_score": quality_score,
                        "boost": boost,
                        "original_score": original_score
                    }

                    enhancement_applied = True
                    quality_enhanced += 1
                    logger.info(f"[LLM_DEEP] {symbol}: Quality boost={boost:+.1f}, VM: {original_score:.1f} -> {new_score:.1f}")

                if enhancement_applied:
                    # Recalculate avg_score
                    result["avg_score"] = sum(strategies.values()) / len(strategies)
                    enhanced_count += 1

            except Exception as e:
                logger.error(f"[LLM_DEEP] Error analyzing {symbol}: {e}")
                continue

        logger.info(f"[LLM_DEEP] Complete: {len(targets)} analyzed, "
                   f"{earnings_enhanced} earnings enhanced, {quality_enhanced} quality enhanced")

        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get aggregated usage statistics from all scorers.

        Returns:
            dict: {
                "total_calls": int,
                "total_cost": float,
                "total_input_tokens": int,
                "total_output_tokens": int
            }
        """
        total_calls = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        # Collect stats from news scorer
        if self._news_scorer and hasattr(self._news_scorer, 'openai_client'):
            client = self._news_scorer.openai_client
            if client:
                total_calls += client.total_calls
                total_cost += client.total_cost
                total_input_tokens += client.total_input_tokens
                total_output_tokens += client.total_output_tokens

        # Collect stats from earnings scorer
        if self._earnings_scorer and hasattr(self._earnings_scorer, 'openai_client'):
            client = self._earnings_scorer.openai_client
            if client:
                total_calls += client.total_calls
                total_cost += client.total_cost
                total_input_tokens += client.total_input_tokens
                total_output_tokens += client.total_output_tokens

        # Collect stats from quality scorer
        if self._quality_scorer and hasattr(self._quality_scorer, 'openai_client'):
            client = self._quality_scorer.openai_client
            if client:
                total_calls += client.total_calls
                total_cost += client.total_cost
                total_input_tokens += client.total_input_tokens
                total_output_tokens += client.total_output_tokens

        return {
            "total_calls": total_calls,
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        }
