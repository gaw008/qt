"""
AdaptiveLearner - Machine learning module for optimal parameter discovery.

This module implements adaptive learning for the Intelligent Trading Decision System:
1. Optimal score thresholds (Bayesian optimization)
2. Per-stock best strategies (Thompson Sampling)
3. Factor weights (Online gradient descent)
4. Time-of-day patterns (Time series analysis)

Learning Schedule:
- Update frequency: Daily (not hourly) - aligned with Layer 1
- Persistence: Save to JSON file

Thompson Sampling:
- Track successes/failures per strategy per stock
- Sample from Beta distribution for selection
"""

import json
import logging
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Default parameters file location
DEFAULT_PARAMS_FILE = "data_cache/adaptive_learner_params.json"


class AdaptiveLearner:
    """
    Adaptive learning engine for trading parameter optimization.

    Implements:
    - Bayesian optimization for score thresholds
    - Thompson Sampling for strategy selection
    - Online gradient descent for factor weights
    - Time-of-day pattern analysis
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        min_trades_for_learning: int = 20,
        update_frequency: str = "daily"
    ):
        """
        Initialize AdaptiveLearner.

        Args:
            params_file: Path to JSON file for parameter persistence
            min_trades_for_learning: Minimum trades before applying learned parameters
            update_frequency: Learning update frequency ('daily', 'weekly')
        """
        self.params_file = params_file or DEFAULT_PARAMS_FILE
        self.min_trades_for_learning = min_trades_for_learning
        self.update_frequency = update_frequency

        # Bayesian optimization for thresholds
        # Store observations as (threshold, win_rate) pairs
        self._threshold_observations: Dict[str, List[Tuple[float, float]]] = {}
        self._optimal_thresholds: Dict[str, float] = {}

        # Thompson Sampling for strategy selection
        # Structure: {symbol: {strategy_name: {"alpha": successes+1, "beta": failures+1}}}
        self._strategy_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Online gradient descent for factor weights
        # Structure: {"stability": 0.25, "volume": 0.45, "price_action": 0.30}
        self._factor_weights: Dict[str, float] = {
            "stability": 0.25,
            "volume": 0.45,
            "price_action": 0.30
        }
        self._factor_gradients: Dict[str, float] = {}
        self._factor_learning_rate: float = 0.01

        # Time-of-day patterns
        # Structure: {hour: {"trades": count, "wins": count, "avg_return": float}}
        self._time_patterns: Dict[int, Dict[str, float]] = {}

        # Trade history for learning
        self._trade_history: List[Dict[str, Any]] = []

        # Last update timestamp
        self._last_update: Optional[datetime] = None

        # Load existing parameters
        self.load_learned_parameters()

    def optimize_threshold(
        self,
        symbol: str,
        trade_history: List[Dict[str, Any]]
    ) -> float:
        """
        Optimize score threshold for a specific symbol using Bayesian optimization.

        Uses a simplified Bayesian approach:
        - Prior: Normal distribution around base threshold (65)
        - Likelihood: Based on observed win rates at different thresholds
        - Posterior: Updated threshold estimate

        Args:
            symbol: Stock symbol
            trade_history: List of trade dicts with 'score', 'was_profitable', 'pnl'

        Returns:
            Optimal threshold for the symbol (or default if insufficient data)
        """
        if len(trade_history) < self.min_trades_for_learning:
            # Not enough data - return stored or default
            return self._optimal_thresholds.get(symbol, 65.0)

        # Group trades by score buckets (5-point intervals)
        score_buckets: Dict[int, List[bool]] = {}
        for trade in trade_history:
            score = trade.get('score', 65)
            was_profitable = trade.get('was_profitable', False)

            bucket = int(score // 5) * 5  # Round down to nearest 5
            if bucket not in score_buckets:
                score_buckets[bucket] = []
            score_buckets[bucket].append(was_profitable)

        # Calculate win rate for each bucket
        bucket_win_rates: Dict[int, float] = {}
        for bucket, results in score_buckets.items():
            if len(results) >= 3:  # Need at least 3 trades per bucket
                win_rate = sum(1 for r in results if r) / len(results)
                bucket_win_rates[bucket] = win_rate

        if not bucket_win_rates:
            return self._optimal_thresholds.get(symbol, 65.0)

        # Find threshold where win rate exceeds 50%
        # Start from lowest score bucket and find inflection point
        sorted_buckets = sorted(bucket_win_rates.keys())
        optimal_threshold = 65.0  # Default

        for bucket in sorted_buckets:
            win_rate = bucket_win_rates[bucket]
            if win_rate >= 0.50:
                # This bucket has positive expectation
                optimal_threshold = float(bucket)
                break
            else:
                # Need higher score
                optimal_threshold = float(bucket + 5)

        # Clamp to reasonable range
        optimal_threshold = max(50.0, min(85.0, optimal_threshold))

        # Store observation
        if symbol not in self._threshold_observations:
            self._threshold_observations[symbol] = []

        overall_win_rate = sum(
            1 for t in trade_history if t.get('was_profitable', False)
        ) / len(trade_history)
        self._threshold_observations[symbol].append((optimal_threshold, overall_win_rate))

        # Apply exponential smoothing to stored optimal
        stored = self._optimal_thresholds.get(symbol, 65.0)
        alpha = 0.3  # Smoothing factor
        self._optimal_thresholds[symbol] = alpha * optimal_threshold + (1 - alpha) * stored

        logger.info(
            f"AdaptiveLearner: {symbol} threshold optimized to "
            f"{self._optimal_thresholds[symbol]:.1f} (raw: {optimal_threshold:.1f})"
        )

        return self._optimal_thresholds[symbol]

    def optimize_factor_weights(
        self,
        performance_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Optimize factor weights using online gradient descent.

        Updates weights based on which factors contributed most to profitable trades.

        Args:
            performance_data: Dict with:
                - 'trades': List of trade dicts with 'score_components' and 'pnl'
                - 'period': Time period for analysis ('daily', 'weekly')

        Returns:
            Updated factor weights dict
        """
        trades = performance_data.get('trades', [])

        if len(trades) < self.min_trades_for_learning:
            return self._factor_weights.copy()

        # Calculate gradient for each factor
        # Gradient = correlation between factor score and PnL
        factor_pnl_correlation: Dict[str, float] = {}

        for factor in self._factor_weights.keys():
            factor_scores = []
            pnls = []

            for trade in trades:
                components = trade.get('score_components', {})
                factor_score = components.get(factor, 0)
                pnl = trade.get('pnl', 0)

                if factor_score is not None:
                    factor_scores.append(factor_score)
                    pnls.append(pnl)

            if len(factor_scores) >= 5:
                # Simple correlation calculation
                correlation = self._calculate_correlation(factor_scores, pnls)
                factor_pnl_correlation[factor] = correlation

        # Update weights using gradient descent
        if factor_pnl_correlation:
            # Normalize correlations to gradient direction
            total_abs_corr = sum(abs(c) for c in factor_pnl_correlation.values()) or 1.0

            for factor, corr in factor_pnl_correlation.items():
                gradient = corr / total_abs_corr

                # Update weight
                old_weight = self._factor_weights[factor]
                new_weight = old_weight + self._factor_learning_rate * gradient

                # Clamp to valid range
                new_weight = max(0.05, min(0.60, new_weight))
                self._factor_weights[factor] = new_weight

            # Normalize weights to sum to 1.0
            total_weight = sum(self._factor_weights.values())
            if total_weight > 0:
                for factor in self._factor_weights:
                    self._factor_weights[factor] /= total_weight

        logger.info(
            f"AdaptiveLearner: Factor weights updated - "
            f"stability={self._factor_weights['stability']:.2f}, "
            f"volume={self._factor_weights['volume']:.2f}, "
            f"price_action={self._factor_weights['price_action']:.2f}"
        )

        return self._factor_weights.copy()

    def get_time_of_day_adjustment(self, hour: int) -> float:
        """
        Get trading adjustment factor based on time of day.

        Analyzes historical performance by hour to identify
        optimal and suboptimal trading times.

        Args:
            hour: Hour of day (0-23, in market timezone)

        Returns:
            Adjustment factor (0.5-1.5, where 1.0 is neutral)
        """
        pattern = self._time_patterns.get(hour, {})

        if pattern.get('trades', 0) < 10:
            # Not enough data - return neutral
            return 1.0

        win_rate = pattern.get('wins', 0) / pattern.get('trades', 1)
        avg_return = pattern.get('avg_return', 0)

        # Calculate adjustment based on win rate and return
        # Win rate contribution: 50% win rate = 1.0
        win_adj = 0.5 + win_rate  # 0.5 to 1.5

        # Return contribution: normalize to similar scale
        # Assume +/- 2% avg return maps to +/- 0.5 adjustment
        return_adj = 1.0 + (avg_return / 0.04)  # 4% swing = 0.5 adjustment
        return_adj = max(0.5, min(1.5, return_adj))

        # Combined adjustment (weighted average)
        adjustment = 0.6 * win_adj + 0.4 * return_adj

        # Clamp to reasonable range
        adjustment = max(0.5, min(1.5, adjustment))

        return adjustment

    def update_from_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Update learning models from a completed trade.

        Args:
            trade_result: Dict containing:
                - symbol: Stock symbol
                - action: 'BUY' or 'SELL'
                - score: Signal score at entry
                - score_components: Dict of factor scores
                - pnl: Profit/loss amount
                - was_profitable: Boolean
                - strategy: Strategy name used
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp
        """
        # Store in trade history
        self._trade_history.append(trade_result)

        # Limit history size
        max_history = 1000
        if len(self._trade_history) > max_history:
            self._trade_history = self._trade_history[-max_history:]

        symbol = trade_result.get('symbol', 'UNKNOWN')
        strategy = trade_result.get('strategy', 'default')
        was_profitable = trade_result.get('was_profitable', False)
        entry_time = trade_result.get('entry_time')

        # Update Thompson Sampling stats
        self._update_strategy_stats(symbol, strategy, was_profitable)

        # Update time-of-day patterns
        if entry_time:
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except ValueError:
                    entry_time = None

            if entry_time:
                hour = entry_time.hour
                self._update_time_pattern(hour, trade_result)

        logger.debug(
            f"AdaptiveLearner: Updated from trade - {symbol} {strategy} "
            f"{'WIN' if was_profitable else 'LOSS'}"
        )

    def _update_strategy_stats(
        self,
        symbol: str,
        strategy: str,
        was_profitable: bool
    ) -> None:
        """Update Thompson Sampling statistics for strategy selection."""
        if symbol not in self._strategy_stats:
            self._strategy_stats[symbol] = {}

        if strategy not in self._strategy_stats[symbol]:
            # Initialize with Beta(1, 1) prior (uniform)
            self._strategy_stats[symbol][strategy] = {"alpha": 1.0, "beta": 1.0}

        stats = self._strategy_stats[symbol][strategy]
        if was_profitable:
            stats["alpha"] += 1.0
        else:
            stats["beta"] += 1.0

    def _update_time_pattern(self, hour: int, trade_result: Dict[str, Any]) -> None:
        """Update time-of-day pattern statistics."""
        if hour not in self._time_patterns:
            self._time_patterns[hour] = {
                "trades": 0,
                "wins": 0,
                "total_return": 0.0,
                "avg_return": 0.0
            }

        pattern = self._time_patterns[hour]
        pattern["trades"] += 1

        if trade_result.get('was_profitable', False):
            pattern["wins"] += 1

        # Calculate return percentage
        entry_price = trade_result.get('entry_price', 0)
        pnl = trade_result.get('pnl', 0)
        quantity = trade_result.get('quantity', 1)

        if entry_price > 0 and quantity > 0:
            return_pct = pnl / (entry_price * quantity)
            pattern["total_return"] += return_pct
            pattern["avg_return"] = pattern["total_return"] / pattern["trades"]

    def get_optimal_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Get all optimal parameters for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with optimal parameters:
                - threshold: Optimal score threshold
                - factor_weights: Optimal factor weights
                - best_strategy: Thompson Sampling selected strategy
                - time_adjustments: Hour -> adjustment factor
        """
        # Get optimal threshold
        threshold = self._optimal_thresholds.get(symbol, 65.0)

        # Get best strategy via Thompson Sampling
        best_strategy = self._select_strategy_thompson(symbol)

        # Get time adjustments for trading hours
        time_adjustments = {}
        for hour in range(9, 17):  # Market hours
            time_adjustments[hour] = self.get_time_of_day_adjustment(hour)

        return {
            "symbol": symbol,
            "threshold": threshold,
            "factor_weights": self._factor_weights.copy(),
            "best_strategy": best_strategy,
            "time_adjustments": time_adjustments,
            "trades_used": len([t for t in self._trade_history if t.get('symbol') == symbol])
        }

    def _select_strategy_thompson(self, symbol: str) -> Optional[str]:
        """
        Select best strategy for symbol using Thompson Sampling.

        Samples from Beta distribution for each strategy and selects highest.

        Args:
            symbol: Stock symbol

        Returns:
            Selected strategy name or None
        """
        if symbol not in self._strategy_stats:
            return None

        strategies = self._strategy_stats[symbol]
        if not strategies:
            return None

        # Sample from each strategy's Beta distribution
        samples: Dict[str, float] = {}
        for strategy, stats in strategies.items():
            alpha = stats.get("alpha", 1.0)
            beta = stats.get("beta", 1.0)

            # Sample from Beta(alpha, beta)
            sample = self._sample_beta(alpha, beta)
            samples[strategy] = sample

        # Select strategy with highest sample
        best_strategy = max(samples, key=samples.get)

        return best_strategy

    def _sample_beta(self, alpha: float, beta: float) -> float:
        """
        Sample from Beta distribution.

        Uses the ratio of Gamma variates method.
        """
        if alpha <= 0 or beta <= 0:
            return 0.5

        try:
            x = random.gammavariate(alpha, 1.0)
            y = random.gammavariate(beta, 1.0)
            return x / (x + y) if (x + y) > 0 else 0.5
        except (ValueError, ZeroDivisionError):
            return 0.5

    def _calculate_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2 or n != len(y):
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Calculate covariance and standard deviations
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def save_learned_parameters(self) -> None:
        """
        Save learned parameters to JSON file.

        Persists:
        - Optimal thresholds per symbol
        - Strategy statistics (Thompson Sampling)
        - Factor weights
        - Time patterns
        """
        params = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "optimal_thresholds": self._optimal_thresholds,
            "strategy_stats": self._strategy_stats,
            "factor_weights": self._factor_weights,
            "time_patterns": {str(k): v for k, v in self._time_patterns.items()},
            "threshold_observations": {
                k: [(t, w) for t, w in v]
                for k, v in self._threshold_observations.items()
            },
            "meta": {
                "min_trades_for_learning": self.min_trades_for_learning,
                "update_frequency": self.update_frequency,
                "total_trades_learned": len(self._trade_history)
            }
        }

        # Ensure directory exists
        params_path = Path(self.params_file)
        params_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)

            logger.info(f"AdaptiveLearner: Parameters saved to {self.params_file}")

        except Exception as e:
            logger.error(f"AdaptiveLearner: Failed to save parameters - {e}")

    def load_learned_parameters(self) -> None:
        """
        Load learned parameters from JSON file.

        Restores all persisted learning state.
        """
        params_path = Path(self.params_file)

        if not params_path.exists():
            logger.info(f"AdaptiveLearner: No saved parameters at {self.params_file}")
            return

        try:
            with open(params_path, 'r') as f:
                params = json.load(f)

            # Restore state
            self._optimal_thresholds = params.get("optimal_thresholds", {})
            self._strategy_stats = params.get("strategy_stats", {})
            self._factor_weights = params.get("factor_weights", self._factor_weights)

            # Restore time patterns (convert string keys back to int)
            time_patterns = params.get("time_patterns", {})
            self._time_patterns = {int(k): v for k, v in time_patterns.items()}

            # Restore threshold observations
            obs = params.get("threshold_observations", {})
            self._threshold_observations = {
                k: [(t, w) for t, w in v]
                for k, v in obs.items()
            }

            logger.info(
                f"AdaptiveLearner: Loaded parameters from {self.params_file} - "
                f"{len(self._optimal_thresholds)} symbol thresholds, "
                f"{len(self._strategy_stats)} symbol strategies"
            )

        except Exception as e:
            logger.error(f"AdaptiveLearner: Failed to load parameters - {e}")

    def should_update(self) -> bool:
        """
        Check if learning update should run based on schedule.

        Returns:
            True if update should run
        """
        now = datetime.now()

        if self._last_update is None:
            return True

        if self.update_frequency == "daily":
            # Update if last update was more than 1 day ago
            return (now - self._last_update) >= timedelta(days=1)

        elif self.update_frequency == "weekly":
            # Update if last update was more than 7 days ago
            return (now - self._last_update) >= timedelta(days=7)

        return False

    def run_daily_update(self) -> Dict[str, Any]:
        """
        Run daily learning update.

        Processes trade history to update all learning models.

        Returns:
            Dict with update summary
        """
        if not self.should_update() and len(self._trade_history) < self.min_trades_for_learning:
            return {"status": "skipped", "reason": "Not enough trades or not time for update"}

        # Group trades by symbol
        trades_by_symbol: Dict[str, List[Dict]] = {}
        for trade in self._trade_history:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)

        # Optimize thresholds for each symbol
        updated_thresholds = 0
        for symbol, trades in trades_by_symbol.items():
            if len(trades) >= self.min_trades_for_learning:
                self.optimize_threshold(symbol, trades)
                updated_thresholds += 1

        # Optimize factor weights using all recent trades
        self.optimize_factor_weights({"trades": self._trade_history})

        # Record update time
        self._last_update = datetime.now()

        # Save parameters
        self.save_learned_parameters()

        summary = {
            "status": "completed",
            "timestamp": self._last_update.isoformat(),
            "trades_processed": len(self._trade_history),
            "symbols_updated": updated_thresholds,
            "factor_weights": self._factor_weights.copy()
        }

        logger.info(f"AdaptiveLearner: Daily update completed - {summary}")

        return summary

    def get_strategy_win_rates(self, symbol: str) -> Dict[str, float]:
        """
        Get win rates for all strategies for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict of strategy name -> win rate
        """
        if symbol not in self._strategy_stats:
            return {}

        win_rates = {}
        for strategy, stats in self._strategy_stats[symbol].items():
            alpha = stats.get("alpha", 1.0)
            beta = stats.get("beta", 1.0)
            # Expected value of Beta distribution
            win_rate = alpha / (alpha + beta)
            win_rates[strategy] = win_rate

        return win_rates

    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get current learning status for monitoring.

        Returns:
            Dict with learning statistics
        """
        return {
            "total_trades_learned": len(self._trade_history),
            "symbols_with_thresholds": len(self._optimal_thresholds),
            "symbols_with_strategies": len(self._strategy_stats),
            "hours_with_patterns": len(self._time_patterns),
            "current_factor_weights": self._factor_weights.copy(),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "update_frequency": self.update_frequency,
            "min_trades_for_learning": self.min_trades_for_learning
        }


# Global singleton instance
_adaptive_learner: Optional[AdaptiveLearner] = None


def get_adaptive_learner() -> AdaptiveLearner:
    """Get the global AdaptiveLearner singleton."""
    global _adaptive_learner
    if _adaptive_learner is None:
        _adaptive_learner = AdaptiveLearner()
    return _adaptive_learner


def set_adaptive_learner(learner: AdaptiveLearner) -> None:
    """Set the global AdaptiveLearner singleton."""
    global _adaptive_learner
    _adaptive_learner = learner
