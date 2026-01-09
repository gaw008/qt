"""
LLM Enhancement Module for Stock Selection

This is a completely independent module that enhances stock selection results
using Large Language Models (LLMs) for text analysis of news and earnings reports.

Features:
- Zero impact on base system (fully independent)
- Can be completely disabled via ENABLE_LLM_ENHANCEMENT=false
- Automatic fallback to base results on failure
- Independent logging, caching, and monitoring
- Cost controls and budget limits

Architecture:
    Base System: universe -> strategies -> combiner -> Top 10 -> status.json
    LLM Layer:   Base results -> LLM Enhancement -> llm_enhanced_status.json

The LLM layer is a decorator that never modifies the base results.
"""

__version__ = "1.0.0"
__author__ = "Quant Trading System"

from .config import LLM_CONFIG
from .pipeline import get_llm_pipeline

__all__ = ["LLM_CONFIG", "get_llm_pipeline"]
