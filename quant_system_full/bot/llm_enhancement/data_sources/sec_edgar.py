"""
SEC EDGAR Document Fetcher

Fetches earnings documents (8-K, 10-Q, 10-K) from SEC EDGAR database.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SECEdgarFetcher:
    """
    Fetches earnings documents from SEC EDGAR.

    Focuses on:
    - 8-K: Current reports (earnings announcements)
    - 10-Q: Quarterly reports
    - 10-K: Annual reports
    """

    BASE_URL = "https://www.sec.gov"

    def __init__(self, config):
        """
        Initialize SEC EDGAR fetcher.

        Args:
            config: LLMEnhancementConfig instance
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.sec_edgar_user_agent
        })

    def fetch(self, symbol: str, max_age_days: int = 90) -> List[Dict[str, Any]]:
        """
        Fetch recent SEC filings for a stock.

        Args:
            symbol: Stock ticker
            max_age_days: Maximum age of documents in days

        Returns:
            List of document excerpts, each with:
                - type: str (8-K, 10-Q, 10-K)
                - date: str (ISO format)
                - text: str (relevant excerpt)
                - url: str
        """
        try:
            # Get CIK for symbol
            cik = self._get_cik(symbol)
            if not cik:
                logger.warning(f"[LLM] Could not find CIK for {symbol}")
                return []

            # Fetch recent filings
            filings = self._fetch_recent_filings(cik, max_age_days)

            # Extract relevant content from each filing
            documents = []
            for filing in filings[:5]:  # Limit to 5 most recent
                try:
                    content = self._extract_filing_content(filing)
                    if content:
                        documents.append(content)
                except Exception as e:
                    logger.debug(f"[LLM] Error extracting filing content: {e}")
                    continue

            logger.info(f"[LLM] Fetched {len(documents)} SEC documents for {symbol}")
            return documents

        except Exception as e:
            logger.error(f"[LLM] Failed to fetch SEC documents for {symbol}: {e}")
            return []

    def _get_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a stock symbol.

        Args:
            symbol: Stock ticker

        Returns:
            CIK string or None if not found
        """
        try:
            # Use SEC company tickers JSON
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Search for symbol
            for item in data.values():
                if item.get("ticker", "").upper() == symbol.upper():
                    cik = str(item.get("cik_str", "")).zfill(10)
                    logger.debug(f"[LLM] Found CIK {cik} for {symbol}")
                    return cik

            return None

        except Exception as e:
            logger.error(f"[LLM] Error fetching CIK for {symbol}: {e}")
            return None

    def _fetch_recent_filings(self, cik: str, max_age_days: int) -> List[Dict[str, Any]]:
        """
        Fetch recent filings for a CIK using new SEC EDGAR Data API.

        Args:
            cik: Company CIK
            max_age_days: Maximum age of filings

        Returns:
            List of filing metadata
        """
        try:
            # Use new SEC EDGAR Data API
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            # Extract recent filings
            if 'filings' not in data or 'recent' not in data['filings']:
                logger.warning(f"[LLM] No filings data found for CIK {cik}")
                return []

            recent = data['filings']['recent']
            forms = recent.get('form', [])
            filing_dates = recent.get('filingDate', [])
            accession_numbers = recent.get('accessionNumber', [])

            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            filings = []

            # Target form types
            target_forms = ['8-K', '10-Q', '10-K']

            for i, form in enumerate(forms):
                try:
                    # Filter by form type
                    if form not in target_forms:
                        continue

                    # Parse and filter by date
                    filing_date = filing_dates[i]
                    file_date = datetime.strptime(filing_date, "%Y-%m-%d")
                    if file_date < cutoff_date:
                        continue

                    # Build filing URL from accession number
                    accession = accession_numbers[i].replace('-', '')
                    filing_url = f"{self.BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession_numbers[i]}-index.htm"

                    filings.append({
                        "type": form,
                        "date": file_date.isoformat(),
                        "url": filing_url
                    })

                except Exception as e:
                    logger.debug(f"[LLM] Error parsing filing entry: {e}")
                    continue

            logger.debug(f"[LLM] Found {len(filings)} recent filings for CIK {cik}")
            return filings

        except Exception as e:
            logger.error(f"[LLM] Error fetching filings for CIK {cik}: {e}")
            return []

    def _extract_filing_content(self, filing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract relevant content from a filing.

        Args:
            filing: Filing metadata

        Returns:
            Document with extracted text or None
        """
        try:
            # Fetch filing document
            response = self.session.get(filing["url"], timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract text from filing
            # For simplicity, we'll extract from the first few thousand characters
            text = soup.get_text(separator="\n", strip=True)

            # Limit text length
            max_length = 3000
            if len(text) > max_length:
                text = text[:max_length] + "..."

            return {
                "type": filing["type"],
                "date": filing["date"],
                "text": text,
                "url": filing["url"]
            }

        except Exception as e:
            logger.error(f"[LLM] Error extracting filing content: {e}")
            return None
