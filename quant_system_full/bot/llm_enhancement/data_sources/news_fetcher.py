"""
News Fetcher

Fetches recent news articles for stocks from various sources.
Supports Yahoo Finance API, Alpha Vantage, and Finnhub.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Fetches news articles for stocks.

    Supports multiple data sources with automatic fallback:
    1. Yahoo Finance API (default)
    2. Alpha Vantage (if API key provided)
    3. Finnhub (if API key provided)
    """

    def __init__(self, config):
        """
        Initialize news fetcher.

        Args:
            config: LLMEnhancementConfig instance
        """
        self.config = config
        self.source = config.news_data_source

    def fetch(self, symbol: str, max_age_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch news for a stock symbol.

        Args:
            symbol: Stock ticker
            max_age_days: Maximum age of news in days (defaults to config.news_fetch_days)

        Returns:
            List of news items, each with:
                - title: str
                - description: str
                - source: str
                - date: str (ISO format)
                - url: str
        """
        max_age_days = max_age_days or self.config.news_fetch_days

        try:
            if self.source == "yahoo_finance_api":
                return self._fetch_yahoo(symbol, max_age_days)
            elif self.source == "alphavantage":
                return self._fetch_alphavantage(symbol, max_age_days)
            elif self.source == "finnhub":
                return self._fetch_finnhub(symbol, max_age_days)
            else:
                logger.warning(f"[LLM] Unknown news source: {self.source}, falling back to Yahoo")
                return self._fetch_yahoo(symbol, max_age_days)

        except Exception as e:
            logger.error(f"[LLM] Failed to fetch news for {symbol}: {e}")
            return []

    def _fetch_yahoo(self, symbol: str, max_age_days: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Yahoo Finance API.

        Args:
            symbol: Stock ticker
            max_age_days: Maximum age of news in days

        Returns:
            List of news items
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                logger.debug(f"[LLM] No news found for {symbol}")
                return []

            # Filter by date and format
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            results = []

            for item in news[:self.config.news_max_items]:
                try:
                    # Yahoo Finance API uses nested 'content' structure
                    content = item.get("content", {})

                    # Parse ISO date string from pubDate
                    pub_date_str = content.get("pubDate", "")
                    if not pub_date_str:
                        continue

                    # Parse ISO format date (e.g., "2025-10-13T02:02:40Z")
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))

                    # Remove timezone info for comparison with naive cutoff_date
                    pub_date = pub_date.replace(tzinfo=None)

                    if pub_date < cutoff_date:
                        continue

                    # Extract nested provider info
                    provider = content.get("provider", {})
                    provider_name = provider.get("displayName", "Unknown")

                    # Extract nested URL
                    click_through = content.get("clickThroughUrl", {})
                    url = click_through.get("url", "")

                    results.append({
                        "title": content.get("title", ""),
                        "description": content.get("summary", ""),
                        "source": provider_name,
                        "date": pub_date.isoformat(),
                        "url": url
                    })
                except Exception as e:
                    logger.debug(f"[LLM] Error parsing news item: {e}")
                    continue

            logger.info(f"[LLM] Fetched {len(results)} news items for {symbol} from Yahoo Finance")
            return results

        except Exception as e:
            logger.error(f"[LLM] Yahoo Finance news fetch failed for {symbol}: {e}")
            return []

    def _fetch_alphavantage(self, symbol: str, max_age_days: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Alpha Vantage.

        Args:
            symbol: Stock ticker
            max_age_days: Maximum age of news in days

        Returns:
            List of news items
        """
        # Placeholder for Alpha Vantage implementation
        logger.warning(f"[LLM] Alpha Vantage news source not yet implemented")
        return []

    def _fetch_finnhub(self, symbol: str, max_age_days: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Finnhub.

        Args:
            symbol: Stock ticker
            max_age_days: Maximum age of news in days

        Returns:
            List of news items
        """
        # Placeholder for Finnhub implementation
        logger.warning(f"[LLM] Finnhub news source not yet implemented")
        return []
