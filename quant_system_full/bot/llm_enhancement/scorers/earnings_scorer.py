"""
Earnings Scorer

Analyzes earnings documents using LLM and returns structured scores.
"""

import logging
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class EarningsScorer:
    """
    Scores stocks based on earnings analysis using LLM.

    Pipeline:
    1. Fetch SEC documents for symbol
    2. Check cache for recent analysis
    3. If not cached, call LLM with earnings deep prompt
    4. Validate and return structured scores
    """

    def __init__(self):
        """Initialize earnings scorer with lazy loading."""
        self._config = None
        self._sec_fetcher = None
        self._openai_client = None
        self._cache = None

    @property
    def config(self):
        """Lazy load config."""
        if self._config is None:
            from ..config import LLM_CONFIG
            self._config = LLM_CONFIG
        return self._config

    @property
    def sec_fetcher(self):
        """Lazy load SEC fetcher."""
        if self._sec_fetcher is None:
            from ..data_sources.sec_edgar import SECEdgarFetcher
            self._sec_fetcher = SECEdgarFetcher(self.config)
        return self._sec_fetcher

    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            from ..clients.openai_client import OpenAIClient
            self._openai_client = OpenAIClient(self.config)
        return self._openai_client

    @property
    def cache(self):
        """Lazy load cache."""
        if self._cache is None:
            from ..cache.doc_cache import DocumentCache
            self._cache = DocumentCache(self.config)
        return self._cache

    def analyze(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analyze earnings documents for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            dict: {
                "earnings_score": int (-100 to +100),
                "summary": str,
                "key_insights": List[str],
                "confidence": int (0-100)
            } or None if analysis fails
        """
        try:
            # Check cache first
            cache_key = f"earnings_deep:{symbol}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"[LLM] Using cached earnings analysis for {symbol}")
                return cached

            # Fetch SEC documents
            documents = self.sec_fetcher.fetch(symbol)
            if not documents:
                logger.warning(f"[LLM] No earnings documents found for {symbol}")
                return None

            # Build prompt
            from ..prompts.earnings_deep import build_earnings_deep_prompt
            prompt = build_earnings_deep_prompt(symbol, documents)

            # Call LLM with GPT-5 reasoning effort (deep analysis)
            response = self.openai_client.complete(
                prompt=prompt,
                model=self.config.model_deep,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
                reasoning_effort=self.config.reasoning_effort_deep
            )

            # Parse and validate response
            analysis = json.loads(response["content"])

            from ..models.schemas import validate_earnings_response
            validated = validate_earnings_response(analysis)

            if not validated:
                logger.error(f"[LLM] Failed to validate earnings response for {symbol}")
                return None

            result = validated.dict()

            # Cache result
            self.cache.set(cache_key, result)

            logger.info(
                f"[LLM] Earnings analysis for {symbol}: "
                f"score={result['earnings_score']}, confidence={result['confidence']}"
            )

            return result

        except Exception as e:
            logger.error(f"[LLM] Error analyzing earnings for {symbol}: {e}")
            return None
