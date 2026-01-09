"""
Pydantic Schemas for LLM Responses

Validates JSON outputs from LLM to ensure correct structure and types.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class NewsAnalysisSchema(BaseModel):
    """
    Schema for news triage analysis output.

    This validates the JSON response from the news triage prompt.
    """

    news_quality: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall news quality score (0-100)"
    )

    risk_flags: int = Field(
        ...,
        ge=0,
        le=100,
        description="Risk flags severity (0-100)"
    )

    summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief summary of news analysis"
    )

    key_insights: List[str] = Field(
        ...,
        min_items=0,
        max_items=5,
        description="Key insights from news (max 5)"
    )

    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in analysis (0-100)"
    )

    @validator("key_insights")
    def validate_insights(cls, v):
        """Ensure insights are non-empty strings."""
        return [insight.strip() for insight in v if insight.strip()]

    class Config:
        schema_extra = {
            "example": {
                "news_quality": 75,
                "risk_flags": 30,
                "summary": "Positive earnings beat with strong guidance, no significant risks identified",
                "key_insights": [
                    "Q3 earnings beat by 15%",
                    "Raised full-year guidance",
                    "Strong demand in cloud segment"
                ],
                "confidence": 85
            }
        }


class EarningsAnalysisSchema(BaseModel):
    """
    Schema for earnings deep analysis output.

    This validates the JSON response from the earnings deep analysis prompt.
    """

    earnings_score: int = Field(
        ...,
        ge=-100,
        le=100,
        description="Earnings quality score (-100 to +100)"
    )

    summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief summary of earnings analysis"
    )

    key_insights: List[str] = Field(
        ...,
        min_items=0,
        max_items=5,
        description="Key insights from earnings (max 5)"
    )

    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in analysis (0-100)"
    )

    @validator("key_insights")
    def validate_insights(cls, v):
        """Ensure insights are non-empty strings."""
        return [insight.strip() for insight in v if insight.strip()]

    class Config:
        schema_extra = {
            "example": {
                "earnings_score": 65,
                "summary": "Strong quarter with accelerating revenue growth and margin expansion",
                "key_insights": [
                    "Revenue growth +25% YoY vs +18% previous quarter",
                    "Operating margin expanded to 32% from 28%",
                    "Beat consensus EPS by $0.15"
                ],
                "confidence": 80
            }
        }


class QualityAnalysisSchema(BaseModel):
    """
    Schema for quality analysis output.

    This validates the JSON response from the quality deep analysis prompt.
    """

    quality_score: int = Field(
        ...,
        ge=-100,
        le=100,
        description="Overall quality score (-100 to +100)"
    )

    summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief summary of quality analysis"
    )

    key_insights: List[str] = Field(
        ...,
        min_items=0,
        max_items=5,
        description="Key insights from quality analysis (max 5)"
    )

    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in analysis (0-100)"
    )

    @validator("key_insights")
    def validate_insights(cls, v):
        """Ensure insights are non-empty strings."""
        return [insight.strip() for insight in v if insight.strip()]

    class Config:
        schema_extra = {
            "example": {
                "quality_score": 70,
                "summary": "Strong balance sheet with improving cash flow and declining debt",
                "key_insights": [
                    "Debt-to-equity ratio improved to 0.3 from 0.5",
                    "Free cash flow +40% YoY",
                    "Current ratio healthy at 2.1"
                ],
                "confidence": 75
            }
        }


def validate_news_response(response: dict) -> Optional[NewsAnalysisSchema]:
    """
    Validate news triage response.

    Args:
        response: Dict from LLM

    Returns:
        Validated NewsAnalysisSchema or None if invalid
    """
    try:
        return NewsAnalysisSchema(**response)
    except Exception as e:
        from logging import getLogger
        logger = getLogger(__name__)
        logger.error(f"[LLM] Failed to validate news response: {e}")
        return None


def validate_earnings_response(response: dict) -> Optional[EarningsAnalysisSchema]:
    """
    Validate earnings analysis response.

    Args:
        response: Dict from LLM

    Returns:
        Validated EarningsAnalysisSchema or None if invalid
    """
    try:
        return EarningsAnalysisSchema(**response)
    except Exception as e:
        from logging import getLogger
        logger = getLogger(__name__)
        logger.error(f"[LLM] Failed to validate earnings response: {e}")
        return None


def validate_quality_response(response: dict) -> Optional[QualityAnalysisSchema]:
    """
    Validate quality analysis response.

    Args:
        response: Dict from LLM

    Returns:
        Validated QualityAnalysisSchema or None if invalid
    """
    try:
        return QualityAnalysisSchema(**response)
    except Exception as e:
        from logging import getLogger
        logger = getLogger(__name__)
        logger.error(f"[LLM] Failed to validate quality response: {e}")
        return None
