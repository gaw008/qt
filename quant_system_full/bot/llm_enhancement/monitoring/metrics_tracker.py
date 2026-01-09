"""
Metrics Tracker

Tracks performance metrics for LLM enhancement system.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks performance metrics for LLM enhancement.

    Metrics:
    - Stage execution times
    - Cache hit rates
    - LLM call counts
    - Enhancement statistics
    """

    def __init__(self, config=None):
        """
        Initialize metrics tracker.

        Args:
            config: LLMEnhancementConfig instance (optional)
        """
        if config is None:
            from ..config import LLM_CONFIG
            config = LLM_CONFIG

        self.config = config

        # Metrics for current run
        self.stage_metrics = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_calls = 0
        self.errors = []

        # Performance metrics
        self.start_time = None
        self.end_time = None

    def start_run(self):
        """Start tracking a new run."""
        self.start_time = datetime.now()
        self.stage_metrics = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_calls = 0
        self.errors = []

        logger.debug("[LLM] Started metrics tracking")

    def record_stage(self, stage: str, count: int):
        """
        Record stage execution.

        Args:
            stage: Stage name (e.g., "triage", "deep")
            count: Number of stocks processed
        """
        if stage not in self.stage_metrics:
            self.stage_metrics[stage] = {
                "count": 0,
                "timestamp": datetime.now().isoformat()
            }

        self.stage_metrics[stage]["count"] += count

        logger.debug(f"[LLM] Recorded stage: {stage} (count={count})")

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def record_llm_call(self):
        """Record an LLM API call."""
        self.llm_calls += 1

    def record_error(self, error: str):
        """
        Record an error.

        Args:
            error: Error message
        """
        self.errors.append({
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def end_run(self):
        """End tracking for current run."""
        self.end_time = datetime.now()

        logger.debug("[LLM] Ended metrics tracking")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.

        Returns:
            dict: Complete metrics summary
        """
        execution_time = 0.0
        if self.start_time and self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()

        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0.0

        return {
            "execution_time": execution_time,
            "stages": self.stage_metrics,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "errors": len(self.errors),
            "error_details": self.errors
        }

    def save_to_file(self, file_path: str):
        """
        Save metrics to file.

        Args:
            file_path: Path to save metrics
        """
        try:
            summary = self.get_summary()

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"[LLM] Metrics saved to {file_path}")

        except Exception as e:
            logger.error(f"[LLM] Error saving metrics: {e}")
