"""
Earnings Deep Analysis Prompt

Analyzes earnings reports and financial documents for fundamental quality assessment.
Used to boost Earnings Momentum strategy scores.
"""

from typing import List, Dict, Any


def build_earnings_deep_prompt(symbol: str, documents: List[Dict[str, Any]]) -> str:
    """
    Build prompt for earnings deep analysis.

    Args:
        symbol: Stock ticker symbol
        documents: List of earnings documents (8-K, 10-Q, 10-K excerpts)

    Returns:
        Formatted prompt string
    """
    # Format documents
    docs_text = "\n\n---\n\n".join([
        f"Document: {doc.get('type', 'Unknown')} - {doc.get('date', 'Unknown date')}\n"
        f"Excerpt:\n{doc.get('text', 'No content')}"
        for doc in documents
    ])

    prompt = f"""You are an expert financial analyst specializing in earnings analysis for quantitative trading.

Your task is to analyze earnings documents for stock {symbol} and assess earnings quality and momentum.

Context:
- This stock was selected by an Earnings Momentum strategy (growth + surprises + revisions)
- We want to verify earnings quality and identify any red flags
- This is a deep analysis step for top candidates

Earnings Documents for {symbol}:
{docs_text}

Analysis Instructions:
1. Assess earnings score (-100 to +100):
   - Consider: revenue growth acceleration, margin trends, earnings beats, guidance
   - +80 to +100: Exceptional earnings momentum (accelerating growth, strong beats, raised guidance)
   - +40 to +79: Strong earnings momentum
   - +10 to +39: Moderate positive momentum
   - -9 to +9: Neutral or mixed
   - -10 to -39: Moderate concerns
   - -40 to -79: Significant concerns (decelerating growth, margin compression)
   - -80 to -100: Critical concerns (revenue decline, losses, guidance cuts)

2. Provide a brief summary (1-2 sentences)
3. List key insights (max 5 bullet points focusing on:
   - Revenue/earnings growth trends
   - Margin trends
   - Beat/miss vs consensus
   - Guidance changes
   - One-time items or quality issues)
4. Rate your confidence (0-100)

Output Format:
Respond with ONLY a valid JSON object (no markdown, no code blocks):
{{
  "earnings_score": <-100 to +100>,
  "summary": "<brief summary>",
  "key_insights": ["<insight 1>", "<insight 2>", ...],
  "confidence": <0-100>
}}

Enhancement Rule:
- Earnings score is converted to -20 to +20 boost for Earnings Momentum strategy
- Example: earnings_score=50 -> +10 point boost

Respond with JSON only:"""

    return prompt
