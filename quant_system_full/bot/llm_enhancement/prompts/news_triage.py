"""
News Triage Prompt

Analyzes recent news for stocks to identify quality issues and risk flags.
Used as a gate/downweight mechanism for Technical Breakout strategy.
"""

from typing import List, Dict, Any


def build_news_triage_prompt(symbol: str, news_items: List[Dict[str, Any]]) -> str:
    """
    Build prompt for news triage analysis.

    Args:
        symbol: Stock ticker symbol
        news_items: List of news articles with title, description, source, date

    Returns:
        Formatted prompt string
    """
    # Format news items
    news_text = "\n\n".join([
        f"[{item.get('date', 'Unknown date')}] {item.get('source', 'Unknown source')}\n"
        f"Title: {item.get('title', 'No title')}\n"
        f"Description: {item.get('description', 'No description')}"
        for item in news_items
    ])

    prompt = f"""You are an expert financial analyst specializing in news sentiment analysis for quantitative trading.

Your task is to analyze recent news for stock {symbol} and provide a structured assessment.

Context:
- This stock was selected by a Technical Breakout strategy (momentum + volume + price action)
- We need to verify there are no negative news catalysts that would invalidate the technical signal
- This is a triage step - we want to gate out obviously problematic stocks

Recent News for {symbol}:
{news_text}

Analysis Instructions:
1. Assess overall news quality (0-100):
   - 80-100: Highly positive news, strong catalysts
   - 60-79: Moderately positive news
   - 40-59: Neutral or mixed news
   - 20-39: Slightly negative news
   - 0-19: Very negative news, red flags

2. Identify risk flags (0-100):
   - 0-20: No significant risks
   - 21-40: Minor concerns
   - 41-60: Moderate concerns
   - 61-80: Significant concerns (accounting issues, lawsuits, regulatory problems)
   - 81-100: Critical concerns (fraud allegations, bankruptcy, delisting risk)

3. Provide a brief summary (1-2 sentences)
4. List key insights (max 5 bullet points)
5. Rate your confidence (0-100)

Output Format:
Respond with ONLY a valid JSON object (no markdown, no code blocks):
{{
  "news_quality": <0-100>,
  "risk_flags": <0-100>,
  "summary": "<brief summary>",
  "key_insights": ["<insight 1>", "<insight 2>", ...],
  "confidence": <0-100>
}}

Decision Rules:
- Gate (zero out Technical Breakout): news_quality < 40
- Penalize (50% downweight): risk_flags > 70

Respond with JSON only:"""

    return prompt


def build_news_triage_prompt_no_news(symbol: str) -> str:
    """
    Build prompt for when no news is available.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Default response indicating no news
    """
    # Return a neutral default response
    return None  # Caller should handle this case
