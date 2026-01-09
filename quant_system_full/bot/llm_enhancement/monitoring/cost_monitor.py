"""
Cost Monitor

Tracks LLM API costs and enforces budget limits.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CostMonitor:
    """
    Monitors LLM API costs and enforces budget limits.

    Features:
    - Daily budget tracking
    - Per-run cost limits
    - Cost history persistence
    - Budget alerts
    """

    def __init__(self, config=None):
        """
        Initialize cost monitor.

        Args:
            config: LLMEnhancementConfig instance (optional)
        """
        if config is None:
            from ..config import LLM_CONFIG
            config = LLM_CONFIG

        self.config = config
        self.state_file = Path(config.cache_dir) / "cost_monitor_state.json"

        # Current run tracking
        self.current_run_cost = 0.0
        self.current_run_calls = 0

        # Load state
        self._load_state()

    def _load_state(self):
        """Load cost tracking state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.daily_cost = state.get("daily_cost", 0.0)
                self.daily_calls = state.get("daily_calls", 0)
                self.last_reset = datetime.fromisoformat(state.get("last_reset", datetime.now().isoformat()))

                # Reset if new day
                if self.last_reset.date() < datetime.now().date():
                    logger.info("[LLM] New day detected, resetting daily budget")
                    self._reset_daily()
            else:
                self._reset_daily()

        except Exception as e:
            logger.error(f"[LLM] Error loading cost monitor state: {e}")
            self._reset_daily()

    def _save_state(self):
        """Save cost tracking state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "daily_cost": self.daily_cost,
                "daily_calls": self.daily_calls,
                "last_reset": self.last_reset.isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"[LLM] Error saving cost monitor state: {e}")

    def _reset_daily(self):
        """Reset daily counters."""
        self.daily_cost = 0.0
        self.daily_calls = 0
        self.last_reset = datetime.now()
        self._save_state()

    def check_daily_budget(self) -> bool:
        """
        Check if daily budget allows more calls.

        Returns:
            True if budget allows more calls, False otherwise
        """
        if self.daily_calls >= self.config.daily_budget:
            logger.warning(
                f"[LLM] Daily budget exhausted: {self.daily_calls}/{self.config.daily_budget} calls used"
            )
            return False

        return True

    def check_run_cost_limit(self, estimated_cost: float) -> bool:
        """
        Check if estimated cost exceeds per-run limit.

        Args:
            estimated_cost: Estimated cost for this run

        Returns:
            True if cost is within limit, False otherwise
        """
        if estimated_cost > self.config.cost_limit_per_run:
            logger.warning(
                f"[LLM] Estimated cost ${estimated_cost:.4f} exceeds limit ${self.config.cost_limit_per_run:.4f}"
            )
            return False

        return True

    def record_call(self, cost: float):
        """
        Record an API call and its cost.

        Args:
            cost: Cost of the call in USD
        """
        self.current_run_cost += cost
        self.current_run_calls += 1

        self.daily_cost += cost
        self.daily_calls += 1

        self._save_state()

        logger.debug(
            f"[LLM] Recorded call: ${cost:.6f} "
            f"(run: ${self.current_run_cost:.4f}, daily: ${self.daily_cost:.4f})"
        )

    def reset_run(self):
        """Reset current run counters."""
        self.current_run_cost = 0.0
        self.current_run_calls = 0

    def get_current_run_cost(self) -> float:
        """Get current run cost."""
        return self.current_run_cost

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cost statistics.

        Returns:
            dict: {
                "daily_cost": float,
                "daily_calls": int,
                "daily_budget": int,
                "daily_budget_remaining": int,
                "current_run_cost": float,
                "current_run_calls": int,
                "last_reset": str
            }
        """
        return {
            "daily_cost": self.daily_cost,
            "daily_calls": self.daily_calls,
            "daily_budget": self.config.daily_budget,
            "daily_budget_remaining": self.config.daily_budget - self.daily_calls,
            "current_run_cost": self.current_run_cost,
            "current_run_calls": self.current_run_calls,
            "last_reset": self.last_reset.isoformat()
        }
