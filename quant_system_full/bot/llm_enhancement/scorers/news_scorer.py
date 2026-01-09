"""
News Scorer

Analyzes news articles using LLM and returns structured scores.
"""

import logging
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class NewsScorer:
    """
    Scores stocks based on news analysis using LLM.

    Pipeline:
    1. Fetch news for symbol
    2. Check cache for recent analysis
    3. If not cached, call LLM with news triage prompt
    4. Validate and return structured scores
    """

    def __init__(self):
        """Initialize news scorer with lazy loading."""
        self._config = None
        self._news_fetcher = None
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
    def news_fetcher(self):
        """Lazy load news fetcher."""
        if self._news_fetcher is None:
            from ..data_sources.news_fetcher import NewsFetcher
            self._news_fetcher = NewsFetcher(self.config)
        return self._news_fetcher

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
        Analyze news for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            dict: {
                "news_quality": int (0-100),
                "risk_flags": int (0-100),
                "summary": str,
                "key_insights": List[str],
                "confidence": int (0-100)
            } or None if analysis fails
        """
        try:
            # Check cache first
            cache_key = f"news_triage:{symbol}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"[LLM] Using cached news analysis for {symbol}")
                return cached

            # Fetch news
            news_items = self.news_fetcher.fetch(symbol)
            if not news_items:
                logger.warning(f"[LLM] No news found for {symbol}")
                return None

            # Build prompt
            from ..prompts.news_triage import build_news_triage_prompt
            prompt = build_news_triage_prompt(symbol, news_items)

            # Call LLM with GPT-5 reasoning effort
            response = self.openai_client.complete(
                prompt=prompt,
                model=self.config.model_triage,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
                reasoning_effort=self.config.reasoning_effort_triage
            )

            # Parse and validate response
            analysis = json.loads(response["content"])

            from ..models.schemas import validate_news_response
            validated = validate_news_response(analysis)

            if not validated:
                logger.error(f"[LLM] Failed to validate news response for {symbol}")
                return None

            result = validated.dict()

            # Cache result
            self.cache.set(cache_key, result)

            logger.info(
                f"[LLM] News analysis for {symbol}: "
                f"quality={result['news_quality']}, risk={result['risk_flags']}, "
                f"confidence={result['confidence']}"
            )

            return result

        except Exception as e:
            logger.error(f"[LLM] Error analyzing news for {symbol}: {e}")
            return None
