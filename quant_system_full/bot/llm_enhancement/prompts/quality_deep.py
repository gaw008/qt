"""
Quality Deep Analysis Prompt

Analyzes financial quality from earnings documents and balance sheet data.
Used to boost Value Momentum strategy scores.
"""

from typing import List, Dict, Any


def build_quality_deep_prompt(symbol: str, documents: List[Dict[str, Any]]) -> str:
    """
    Build prompt for quality deep analysis.

    Args:
        symbol: Stock ticker symbol
        documents: List of financial documents (10-Q, 10-K excerpts with financials)

    Returns:
        Formatted prompt string
    """
    # Format documents
    docs_text = "\n\n---\n\n".join([
        f"Document: {doc.get('type', 'Unknown')} - {doc.get('date', 'Unknown date')}\n"
        f"Financial Data:\n{doc.get('text', 'No content')}"
        for doc in documents
    ])

    prompt = f"""You are an expert financial analyst specializing in fundamental quality analysis for quantitative trading.

Your task is to analyze financial documents for stock {symbol} and assess fundamental quality.

Context:
- This stock was selected by a Value Momentum strategy (valuation + momentum)
- We want to verify financial quality and identify any balance sheet risks
- This is a deep analysis step for top candidates

Financial Documents for {symbol}:
{docs_text}

Analysis Instructions:
1. Assess quality score (-100 to +100):
   - Consider: balance sheet strength, cash flow quality, debt levels, working capital trends
   - +80 to +100: Exceptional quality (fortress balance sheet, strong cash flow, low/declining debt)
   - +40 to +79: Strong quality
   - +10 to +39: Moderate quality
   - -9 to +9: Neutral or mixed
   - -10 to -39: Moderate concerns
   - -40 to -79: Significant concerns (high debt, weak cash flow, deteriorating working capital)
   - -80 to -100: Critical concerns (distress, covenant violations, going concern issues)

2. Provide a brief summary (1-2 sentences)
3. List key insights (max 5 bullet points focusing on:
   - Balance sheet strength (debt, liquidity ratios)
   - Cash flow quality and trends
   - Working capital trends
   - Asset quality
   - Off-balance sheet items or accounting concerns)
4. Rate your confidence (0-100)

Output Format:
Respond with ONLY a valid JSON object (no markdown, no code blocks):
{{
  "quality_score": <-100 to +100>,
  "summary": "<brief summary>",
  "key_insights": ["<insight 1>", "<insight 2>", ...],
  "confidence": <0-100>
}}

Enhancement Rule:
- Quality score is converted to -15 to +15 boost for Value Momentum strategy
- Example: quality_score=60 -> +9 point boost

Respond with JSON only:"""

    return prompt
