"""
Quality Scorer

Analyzes financial quality from documents using LLM and returns structured scores.
"""

import logging
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Scores stocks based on quality analysis using LLM.

    Pipeline:
    1. Fetch SEC documents for symbol
    2. Check cache for recent analysis
    3. If not cached, call LLM with quality deep prompt
    4. Validate and return structured scores
    """

    def __init__(self):
        """Initialize quality scorer with lazy loading."""
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
        Analyze quality from financial documents for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            dict: {
                "quality_score": int (-100 to +100),
                "summary": str,
                "key_insights": List[str],
                "confidence": int (0-100)
            } or None if analysis fails
        """
        try:
            # Check cache first
            cache_key = f"quality_deep:{symbol}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"[LLM] Using cached quality analysis for {symbol}")
                return cached

            # Fetch SEC documents
            documents = self.sec_fetcher.fetch(symbol)
            if not documents:
                logger.warning(f"[LLM] No quality documents found for {symbol}")
                return None

            # Build prompt
            from ..prompts.quality_deep import build_quality_deep_prompt
            prompt = build_quality_deep_prompt(symbol, documents)

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

            from ..models.schemas import validate_quality_response
            validated = validate_quality_response(analysis)

            if not validated:
                logger.error(f"[LLM] Failed to validate quality response for {symbol}")
                return None

            result = validated.dict()

            # Cache result
            self.cache.set(cache_key, result)

            logger.info(
                f"[LLM] Quality analysis for {symbol}: "
                f"score={result['quality_score']}, confidence={result['confidence']}"
            )

            return result

        except Exception as e:
            logger.error(f"[LLM] Error analyzing quality for {symbol}: {e}")
            return None
