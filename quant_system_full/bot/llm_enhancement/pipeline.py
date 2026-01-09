"""
LLM Enhancement Main Pipeline

This is the main entry point for the LLM enhancement system.
It acts as a decorator over base selection results, never modifying the input.

Architecture:
    Input: Base selection results (from runner.py)
    Process: Triage (news) -> Deep (earnings) -> Enhance scores
    Output: Enhanced results + metrics (written to independent status file)

Failure Handling:
    - Any error -> log and fallback to base results
    - Cost exceeded -> skip LLM, use base results
    - Budget exhausted -> skip LLM, use base results
"""

import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from .config import LLM_CONFIG

# Configure logger
logger = logging.getLogger(__name__)

class LLMEnhancementPipeline:
    """
    LLM Enhancement Pipeline (Decorator Pattern).

    This class decorates base selection results with LLM-based enhancements
    without ever modifying the original results.

    Features:
    - Complete independence from base system
    - Automatic fallback on any error
    - Budget and cost controls
    - Independent logging and status files
    """

    def __init__(self):
        """Initialize the LLM enhancement pipeline."""
        self.config = LLM_CONFIG

        # Lazy initialization of components
        self._enhancer = None
        self._metrics_tracker = None
        self._cost_monitor = None

        if not self.config.is_available():
            logger.warning("[LLM] Enhancement disabled or not properly configured")
            logger.info(f"[LLM] Config: enabled={self.config.enabled}, has_api_key={bool(self.config.openai_api_key)}")

    @property
    def enhancer(self):
        """Lazy load enhancer."""
        if self._enhancer is None and self.config.is_available():
            try:
                from .enhancer import SelectionEnhancer
                self._enhancer = SelectionEnhancer()
            except Exception as e:
                logger.error(f"[LLM] Failed to initialize enhancer: {e}")
        return self._enhancer

    @property
    def metrics_tracker(self):
        """Lazy load metrics tracker."""
        if self._metrics_tracker is None:
            try:
                from .monitoring.metrics_tracker import MetricsTracker
                self._metrics_tracker = MetricsTracker()
            except Exception as e:
                logger.warning(f"[LLM] Metrics tracker not available: {e}")
        return self._metrics_tracker

    @property
    def cost_monitor(self):
        """Lazy load cost monitor."""
        if self._cost_monitor is None:
            try:
                from .monitoring.cost_monitor import CostMonitor
                self._cost_monitor = CostMonitor()
            except Exception as e:
                logger.warning(f"[LLM] Cost monitor not available: {e}")
        return self._cost_monitor

    def enhance(self, base_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance base selection results using LLM analysis.

        This is the main entry point. It takes base results and optionally
        enhances them using LLM-based text analysis of news and earnings.

        Args:
            base_results: List of selection results from base system
                         Each item should have: symbol, avg_score, strategies, etc.

        Returns:
            dict: {
                "enabled": bool,                    # Whether LLM was actually used
                "mode": str,                        # Operation mode
                "base_results": List[Dict],         # Original results (unchanged)
                "enhanced_results": List[Dict],     # LLM-enhanced results
                "metrics": Dict,                    # Execution metrics
                "errors": List[str],                # Any errors encountered
                "timestamp": str                    # ISO format timestamp
            }

        Note:
            - base_results is NEVER modified (defensive copy made)
            - On any error, returns base_results as enhanced_results
            - Writes results to independent status file
        """
        start_time = datetime.now()

        # If not enabled/available, return base results immediately
        if not self.config.is_available():
            result = {
                "enabled": False,
                "mode": self.config.mode,
                "base_results": base_results,
                "enhanced_results": base_results,  # Same as base
                "metrics": {
                    "execution_time": 0.0,
                    "base_count": len(base_results),
                    "enhanced_count": len(base_results),
                    "cost_usd": 0.0,
                    "llm_calls": 0,
                    "cache_hits": 0
                },
                "errors": ["LLM enhancement is disabled or not configured"],
                "timestamp": datetime.now().isoformat()
            }
            self._write_status(result)
            return result

        logger.info("=" * 60)
        logger.info(f"[LLM_PIPELINE] Starting LLM Enhancement Pipeline")
        logger.info(f"[LLM_PIPELINE] Mode: {self.config.mode}")
        logger.info(f"[LLM_PIPELINE] Input: {len(base_results)} base selections")
        logger.info(f"[LLM_PIPELINE] Models: triage={self.config.model_triage}, deep={self.config.model_deep}")
        logger.info("=" * 60)

        errors = []
        enhanced_results = self._deep_copy_results(base_results)

        try:
            # === Pre-flight Checks ===

            # Check daily budget
            if self.cost_monitor and not self.cost_monitor.check_daily_budget():
                msg = "Daily LLM budget exhausted"
                errors.append(msg)
                logger.warning(f"[LLM] {msg}, falling back to base results")
                raise RuntimeError(msg)

            # Estimate cost for this run
            estimated_cost = self._estimate_cost(len(base_results))
            if estimated_cost > self.config.cost_limit_per_run:
                msg = f"Estimated cost ${estimated_cost:.4f} exceeds limit ${self.config.cost_limit_per_run:.4f}"
                errors.append(msg)
                logger.warning(f"[LLM] {msg}, falling back to base results")
                raise RuntimeError(msg)

            logger.info(f"[LLM] Estimated cost: ${estimated_cost:.4f}")

            # === Execute Enhancement Pipeline ===

            # Step 1: Build event queue (prioritize which stocks to analyze)
            logger.info("[LLM] Step 1: Building event queue")
            from .event_queue.queue_builder import build_event_queue, prioritize

            events = build_event_queue(enhanced_results)
            logger.info(f"[LLM] Found {len(events)} triggered events")

            # Step 2: Triage (News analysis for gate/downweight)
            if self.config.mode in ["full", "triage_only"]:
                triage_targets = prioritize(events)[:self.config.m_triage]
                logger.info(f"[LLM_PIPELINE] Step 2/3: Triage (News Analysis)")
                logger.info(f"[LLM_PIPELINE] Analyzing {len(triage_targets)} stocks for news sentiment and risk flags")

                if triage_targets and self.enhancer:
                    enhanced_results = self.enhancer.apply_triage(
                        enhanced_results,
                        triage_targets
                    )
                    logger.info(f"[LLM_PIPELINE] Triage stage completed")

                    if self.metrics_tracker:
                        self.metrics_tracker.record_stage("triage", len(triage_targets))
                else:
                    logger.warning(f"[LLM_PIPELINE] Triage skipped: no targets or enhancer unavailable")

            # Step 3: Re-sort and select deep analysis targets
            if self.config.mode in ["full", "deep_only"]:
                # Sort by current scores
                enhanced_results.sort(
                    key=lambda x: x.get("avg_score", 0),
                    reverse=True
                )

                deep_targets = [
                    r["symbol"] for r in enhanced_results[:self.config.m_final]
                ]
                logger.info(f"[LLM_PIPELINE] Step 3/3: Deep Analysis (Earnings & Quality)")
                logger.info(f"[LLM_PIPELINE] Analyzing top {len(deep_targets)} stocks for earnings and financial quality")

                if deep_targets and self.enhancer:
                    enhanced_results = self.enhancer.apply_deep(
                        enhanced_results,
                        deep_targets
                    )
                    logger.info(f"[LLM_PIPELINE] Deep analysis stage completed")

                    if self.metrics_tracker:
                        self.metrics_tracker.record_stage("deep", len(deep_targets))
                else:
                    logger.warning(f"[LLM_PIPELINE] Deep analysis skipped: no targets or enhancer unavailable")

            # Step 4: Final sort
            enhanced_results.sort(
                key=lambda x: x.get("avg_score", 0),
                reverse=True
            )

            logger.info("[LLM] Enhancement pipeline completed successfully")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[LLM] Enhancement failed: {error_msg}")
            logger.debug(traceback.format_exc())
            errors.append(error_msg)

            # Fallback to base results
            if self.config.fallback_on_error:
                logger.info("[LLM] Falling back to base results due to error")
                enhanced_results = base_results
            else:
                logger.warning("[LLM] Fallback disabled, using partially enhanced results")

        # === Build Result Package ===

        execution_time = (datetime.now() - start_time).total_seconds()

        # Gather metrics from enhancer's OpenAI clients
        llm_stats = {"total_calls": 0, "total_cost": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}
        if self.enhancer:
            try:
                llm_stats = self.enhancer.get_usage_stats()
            except Exception as e:
                logger.warning(f"[LLM] Failed to get usage stats from enhancer: {e}")

        # Gather metrics
        metrics = {
            "execution_time": execution_time,
            "base_count": len(base_results),
            "enhanced_count": len(enhanced_results),
            "cost_usd": llm_stats.get("total_cost", 0.0),
            "llm_calls": llm_stats.get("total_calls", 0),
            "total_input_tokens": llm_stats.get("total_input_tokens", 0),
            "total_output_tokens": llm_stats.get("total_output_tokens", 0)
        }

        if self.metrics_tracker:
            tracker_metrics = self.metrics_tracker.get_summary()
            # Add cache hits from tracker
            metrics["cache_hits"] = tracker_metrics.get("cache_hits", 0)
            metrics["cache_misses"] = tracker_metrics.get("cache_misses", 0)

        result = {
            "enabled": True,
            "mode": self.config.mode,
            "base_results": base_results,
            "enhanced_results": enhanced_results,
            "metrics": metrics,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }

        # Write to independent status file
        self._write_status(result)

        logger.info("=" * 60)
        if errors:
            logger.warning(f"[LLM_PIPELINE] Completed with {len(errors)} error(s) in {execution_time:.2f}s")
            for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
                logger.warning(f"[LLM_PIPELINE] Error {i}: {error}")
        else:
            logger.info(f"[LLM_PIPELINE] Successfully enhanced {len(enhanced_results)} selections in {execution_time:.2f}s")

        logger.info(f"[LLM_PIPELINE] API calls: {metrics.get('llm_calls', 0)}, "
                   f"cache hits: {metrics.get('cache_hits', 0)}, "
                   f"total cost: ${metrics.get('cost_usd', 0):.4f}")
        logger.info("=" * 60)

        return result

    def _deep_copy_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make a deep copy of results to avoid modifying originals.

        Args:
            results: Original results list

        Returns:
            Deep copy of results
        """
        import copy
        return copy.deepcopy(results)

    def _estimate_cost(self, num_candidates: int) -> float:
        """
        Estimate LLM cost for this run.

        Cost breakdown (using gpt-4o-mini):
        - Triage: ~800 input + 80 output tokens per stock
          ~$0.0001 per stock
        - Deep: ~3000 input + 200 output tokens per stock
          ~$0.001 per stock

        Args:
            num_candidates: Number of candidate stocks

        Returns:
            Estimated cost in USD
        """
        triage_cost = 0.0
        deep_cost = 0.0

        if self.config.mode in ["full", "triage_only"]:
            num_triage = min(num_candidates, self.config.m_triage)
            triage_cost = num_triage * 0.0001

        if self.config.mode in ["full", "deep_only"]:
            num_deep = min(num_candidates, self.config.m_final)
            deep_cost = num_deep * 0.001

        total = triage_cost + deep_cost
        logger.debug(f"[LLM] Cost estimate: triage=${triage_cost:.4f}, deep=${deep_cost:.4f}, total=${total:.4f}")

        return total

    def _write_status(self, result: Dict[str, Any]):
        """
        Write enhancement result to independent status file.

        Args:
            result: Complete result dict to write

        Note:
            - Creates directory if needed
            - Overwrites previous status
            - Logs any write errors but doesn't raise
        """
        try:
            status_file = Path(self.config.status_file)
            status_file.parent.mkdir(parents=True, exist_ok=True)

            with open(status_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            logger.debug(f"[LLM] Status written to {status_file}")

        except Exception as e:
            logger.error(f"[LLM] Failed to write status file: {e}")


# === Global Singleton ===

_pipeline_instance = None

def get_llm_pipeline() -> LLMEnhancementPipeline:
    """
    Get the global LLM enhancement pipeline instance.

    Returns:
        LLMEnhancementPipeline: Singleton instance

    Note:
        - Created on first call
        - Thread-safe (Python GIL)
        - Safe to call multiple times
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = LLMEnhancementPipeline()

    return _pipeline_instance
