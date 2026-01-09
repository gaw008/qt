"""
Independent LLM Enhancement Configuration

This configuration system is completely separate from bot/config.py
to ensure zero impact on the base trading system.

All settings are controlled via environment variables with safe defaults.
Default state: DISABLED (ENABLE_LLM_ENHANCEMENT=false)
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class LLMEnhancementConfig:
    """
    Complete LLM Enhancement configuration.

    All parameters are loaded from environment variables.
    Default behavior: DISABLED to ensure zero impact on existing system.
    """

    # === Master Control ===
    # Setting this to false completely disables all LLM functionality
    enabled: bool = os.getenv("ENABLE_LLM_ENHANCEMENT", "false").lower() == "true"

    # === Operation Mode ===
    # full: Complete pipeline (triage + deep)
    # triage_only: Only news triage (cheaper, faster)
    # deep_only: Only deep earnings analysis (skip triage)
    mode: Literal["full", "triage_only", "deep_only"] = os.getenv("LLM_ENHANCEMENT_MODE", "full")

    # === OpenAI Configuration ===
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_triage: str = os.getenv("LLM_MODEL_TRIAGE", "gpt-5-nano")
    model_deep: str = os.getenv("LLM_MODEL_DEEP", "gpt-5")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # GPT-5 specific parameters
    reasoning_effort_triage: str = os.getenv("LLM_REASONING_EFFORT_TRIAGE", "minimal")
    reasoning_effort_deep: str = os.getenv("LLM_REASONING_EFFORT_DEEP", "medium")

    # === Funnel Parameters ===
    # Three-layer funnel: 5000 -> K_PRE -> M_TRIAGE -> M_FINAL -> 10
    k_pre: int = int(os.getenv("LLM_K_PRE", "300"))        # Base system output
    m_triage: int = int(os.getenv("LLM_M_TRIAGE", "60"))   # Triage targets
    m_final: int = int(os.getenv("LLM_M_FINAL", "15"))     # Deep analysis targets

    # === Budget Controls ===
    daily_budget: int = int(os.getenv("LLM_DAILY_BUDGET", "60"))                    # Total calls per day
    cost_limit_per_run: float = float(os.getenv("LLM_COST_LIMIT", "0.20"))         # Hard limit per selection run

    # === Cache Configuration ===
    cache_enabled: bool = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_days: int = int(os.getenv("LLM_CACHE_TTL_DAYS", "7"))
    cache_dir: str = os.getenv("LLM_CACHE_DIR", "data_cache/llm")

    # === Data Source Configuration ===
    # News fetching
    news_fetch_days: int = int(os.getenv("NEWS_FETCH_DAYS", "3"))                   # Lookback window
    news_max_items: int = int(os.getenv("NEWS_MAX_ITEMS", "8"))                     # Max items per stock
    news_data_source: str = os.getenv("NEWS_DATA_SOURCE", "yahoo_finance_api")      # yahoo_finance_api | alphavantage | finnhub

    # SEC EDGAR for earnings documents
    sec_edgar_user_agent: str = os.getenv("SEC_EDGAR_USER_AGENT", "QuantSystem info@example.com")

    # === Threshold Configuration ===
    news_quality_gate: int = int(os.getenv("NEWS_QUALITY_GATE", "40"))              # Gate threshold for news quality
    risk_flags_penalty: int = int(os.getenv("RISK_FLAGS_PENALTY", "70"))            # Penalty threshold for risk flags

    # === Independent Logging ===
    log_file: str = os.getenv("LLM_LOG_FILE", "logs/llm_enhancement.log")
    log_level: str = os.getenv("LLM_LOG_LEVEL", "INFO")

    # === Independent State File ===
    status_file: str = os.getenv("LLM_STATUS_FILE", "dashboard/state/llm_enhanced_status.json")

    # === Failure Handling ===
    fallback_on_error: bool = os.getenv("LLM_FALLBACK_ON_ERROR", "true").lower() == "true"

    # === Monitoring Configuration ===
    enable_metrics: bool = os.getenv("LLM_ENABLE_METRICS", "true").lower() == "true"
    metrics_log_file: str = os.getenv("LLM_METRICS_LOG_FILE", "logs/llm_metrics.log")

    def is_available(self) -> bool:
        """
        Check if LLM enhancement is available and properly configured.

        Returns:
            bool: True if enabled and has valid API key
        """
        return (
            self.enabled and
            self.openai_api_key and
            len(self.openai_api_key) > 0
        )

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration for common issues.

        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.enabled:
            return True, None  # Disabled is valid state

        if not self.openai_api_key:
            return False, "OPENAI_API_KEY is required when LLM enhancement is enabled"

        if self.daily_budget <= 0:
            return False, "LLM_DAILY_BUDGET must be positive"

        if self.cost_limit_per_run <= 0:
            return False, "LLM_COST_LIMIT must be positive"

        if self.m_triage > self.k_pre:
            return False, "LLM_M_TRIAGE cannot exceed LLM_K_PRE"

        if self.m_final > self.m_triage:
            return False, "LLM_M_FINAL cannot exceed LLM_M_TRIAGE"

        if self.mode not in ["full", "triage_only", "deep_only"]:
            return False, f"Invalid LLM_ENHANCEMENT_MODE: {self.mode}"

        return True, None

    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "is_available": self.is_available(),
            "model_triage": self.model_triage,
            "model_deep": self.model_deep,
            "funnel": {
                "k_pre": self.k_pre,
                "m_triage": self.m_triage,
                "m_final": self.m_final
            },
            "budget": {
                "daily": self.daily_budget,
                "cost_limit": self.cost_limit_per_run
            },
            "cache_enabled": self.cache_enabled,
            "fallback_on_error": self.fallback_on_error
        }


# Global configuration instance
LLM_CONFIG = LLMEnhancementConfig()

# Validate on import (warnings only, don't block)
_is_valid, _error_msg = LLM_CONFIG.validate()
if not _is_valid:
    import warnings
    warnings.warn(f"LLM Enhancement configuration invalid: {_error_msg}. Module will be disabled.")
