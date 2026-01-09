#!/usr/bin/env python3
"""
Risk Management Performance Optimization System
风险管理性能优化系统

High-performance risk system calibration for live trading:
- Optimized ES@97.5% calculations with sub-second response
- Real-time drawdown monitoring with 30-second intervals
- Factor crowding detection with minimal computational overhead
- Performance-tuned risk calculations for 4000+ stock universe
- Memory-efficient risk metric storage and retrieval

Live Trading Performance Features:
- Sub-100ms risk metric updates
- Concurrent risk calculations for portfolio components
- Intelligent caching for repeated calculations
- Real-time risk threshold monitoring
- Optimized memory usage for continuous operation
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from scipy import stats
import numba
from numba import jit, njit
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json
import pickle
from functools import lru_cache
import weakref
import gc

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class PerformanceLevel(Enum):
    """Performance optimization levels for different trading conditions"""
    PRODUCTION = "PRODUCTION"      # Maximum performance, minimal logging
    DEVELOPMENT = "DEVELOPMENT"    # Balanced performance with monitoring
    TESTING = "TESTING"           # Full monitoring with debug features
    BENCHMARK = "BENCHMARK"       # Performance measurement mode

@dataclass
class OptimizedRiskLimits:
    """Performance-optimized risk limits with fast lookup"""
    # Core ES limits
    es_97_5_daily_limit: float = 0.03      # 3% daily ES limit
    es_97_5_weekly_limit: float = 0.08     # 8% weekly ES limit
    es_99_daily_limit: float = 0.05        # 5% daily ES@99% limit

    # Fast drawdown thresholds
    tier_1_drawdown: float = 0.05          # 5% warning level
    tier_2_drawdown: float = 0.08          # 8% action level
    tier_3_drawdown: float = 0.12          # 12% emergency level

    # Position concentration (optimized for fast checks)
    max_single_position: float = 0.08      # 8% maximum single position
    max_sector_concentration: float = 0.20  # 20% maximum sector allocation
    max_correlation_threshold: float = 0.75 # 75% correlation warning

    # Performance-critical limits
    daily_var_95_limit: float = 0.025      # 2.5% daily VaR limit
    portfolio_beta_limit: float = 1.5      # 1.5 maximum portfolio beta
    tracking_error_limit: float = 0.06     # 6% tracking error limit

    # Factor crowding thresholds (fast computation)
    hhi_crowding_threshold: float = 0.25   # HHI concentration threshold
    gini_crowding_threshold: float = 0.60  # Gini coefficient threshold
    effective_breadth_min: float = 8.0     # Minimum effective breadth

@dataclass
class PerformanceMetrics:
    """Real-time performance tracking for risk calculations"""
    calculation_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_calculations: int = 0
    last_update_timestamp: str = ""

@njit(cache=True, fastmath=True)
def fast_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.975) -> float:
    """
    Numba-optimized Expected Shortfall calculation
    Ultra-fast ES computation for real-time risk monitoring
    """
    if len(returns) == 0:
        return 0.0

    # Sort returns (faster than np.sort for small arrays)
    sorted_returns = np.sort(returns)

    # Calculate VaR cutoff index
    var_index = int((1 - confidence_level) * len(sorted_returns))
    if var_index == 0:
        var_index = 1

    # ES is mean of tail beyond VaR
    if var_index >= len(sorted_returns):
        return abs(sorted_returns[0])

    tail_mean = np.mean(sorted_returns[:var_index])
    return abs(tail_mean)

@njit(cache=True, fastmath=True)
def fast_drawdown_calculation(cumulative_returns: np.ndarray) -> Tuple[float, float]:
    """
    Optimized drawdown calculation with current and maximum drawdown
    """
    if len(cumulative_returns) == 0:
        return 0.0, 0.0

    running_max = cumulative_returns[0]
    max_drawdown = 0.0
    current_drawdown = 0.0

    for i in range(len(cumulative_returns)):
        value = cumulative_returns[i]
        if value > running_max:
            running_max = value

        drawdown = (running_max - value) / running_max if running_max > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # Current drawdown is the drawdown at the end
        if i == len(cumulative_returns) - 1:
            current_drawdown = drawdown

    return current_drawdown, max_drawdown

@njit(cache=True, fastmath=True)
def fast_herfindahl_index(weights: np.ndarray) -> float:
    """Fast HHI calculation for concentration risk"""
    return np.sum(weights ** 2)

@njit(cache=True, fastmath=True)
def fast_gini_coefficient(values: np.ndarray) -> float:
    """Fast Gini coefficient for factor crowding detection"""
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)

    return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

class PerformanceCache:
    """Memory-efficient caching system for risk calculations"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 30):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            current_time = time.time()

            if key in self.cache:
                if current_time - self.timestamps[key] < self.ttl_seconds:
                    self.hit_count += 1
                    return self.cache[key]
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.timestamps[key]

            self.miss_count += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            current_time = time.time()

            # Clean expired entries if cache is full
            if len(self.cache) >= self.max_size:
                self._cleanup_expired(current_time)

            # Remove oldest if still full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]

            self.cache[key] = value
            self.timestamps[key] = current_time

    def _cleanup_expired(self, current_time: float) -> None:
        """Remove expired entries"""
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]

    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hit_count = 0
            self.miss_count = 0

class RiskPerformanceOptimizer:
    """
    High-performance risk management system optimized for live trading

    Key Performance Features:
    - Sub-100ms risk calculation updates
    - Concurrent processing for portfolio components
    - Intelligent caching with 30-second TTL
    - Memory-efficient storage for continuous operation
    - Real-time performance monitoring
    """

    def __init__(self,
                 performance_level: PerformanceLevel = PerformanceLevel.PRODUCTION,
                 max_workers: int = 4,
                 cache_size: int = 1000):

        self.performance_level = performance_level
        self.max_workers = max_workers
        self.limits = OptimizedRiskLimits()

        # Performance components
        self.cache = PerformanceCache(max_size=cache_size, ttl_seconds=30)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = PerformanceMetrics()

        # State tracking
        self.active_calculations = 0
        self.calculation_queue = queue.Queue()
        self.performance_history = []

        # Memory management
        self.memory_threshold_mb = 500  # Alert if memory usage exceeds 500MB
        self.gc_interval = 100  # Run garbage collection every 100 calculations
        self.calculation_count = 0

        logger.info(f"Risk Performance Optimizer initialized - Level: {performance_level.value}")

    def start_performance_monitoring(self) -> None:
        """Start background performance monitoring thread"""
        def monitor():
            while True:
                try:
                    # Update performance metrics
                    process = psutil.Process()
                    self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                    self.metrics.cpu_usage_percent = process.cpu_percent()
                    self.metrics.cache_hit_rate = self.cache.get_hit_rate()
                    self.metrics.concurrent_calculations = self.active_calculations
                    self.metrics.last_update_timestamp = datetime.now().isoformat()

                    # Memory management
                    if self.metrics.memory_usage_mb > self.memory_threshold_mb:
                        logger.warning(f"High memory usage: {self.metrics.memory_usage_mb:.1f}MB")
                        gc.collect()

                    time.sleep(1)  # Monitor every second

                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(5)

        if self.performance_level != PerformanceLevel.PRODUCTION:
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()

    def calculate_portfolio_es_optimized(self,
                                       portfolio_returns: np.ndarray,
                                       confidence_levels: List[float] = [0.95, 0.975, 0.99]) -> Dict[str, float]:
        """
        Optimized ES calculation for multiple confidence levels
        Uses caching and vectorized operations for maximum performance
        """
        start_time = time.perf_counter()

        # Create cache key
        cache_key = f"es_{hash(portfolio_returns.tobytes())}_{len(confidence_levels)}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Calculate ES for all confidence levels efficiently
        results = {}
        for confidence_level in confidence_levels:
            es_value = fast_expected_shortfall(portfolio_returns, confidence_level)
            results[f"es_{int(confidence_level*100)}"] = es_value

        # Cache result
        self.cache.put(cache_key, results)

        # Update performance metrics
        calculation_time = (time.perf_counter() - start_time) * 1000
        self.metrics.calculation_time_ms = calculation_time

        return results

    def real_time_drawdown_monitor(self,
                                 portfolio_values: np.ndarray,
                                 update_frequency_seconds: int = 30) -> Dict[str, Any]:
        """
        Real-time drawdown monitoring with configurable update frequency
        Optimized for continuous operation with minimal overhead
        """
        start_time = time.perf_counter()

        # Fast drawdown calculation
        current_dd, max_dd = fast_drawdown_calculation(portfolio_values)

        # Determine risk tier based on current drawdown
        risk_tier = 0
        tier_actions = []

        if current_dd >= self.limits.tier_3_drawdown:
            risk_tier = 3
            tier_actions = ["emergency_deleverage", "close_high_risk_positions", "increase_cash"]
        elif current_dd >= self.limits.tier_2_drawdown:
            risk_tier = 2
            tier_actions = ["reduce_position_sizes", "tighten_stop_losses", "reduce_correlation"]
        elif current_dd >= self.limits.tier_1_drawdown:
            risk_tier = 1
            tier_actions = ["monitor_closely", "prepare_contingencies", "review_positions"]

        calculation_time = (time.perf_counter() - start_time) * 1000

        return {
            "current_drawdown": current_dd,
            "maximum_drawdown": max_dd,
            "risk_tier": risk_tier,
            "tier_actions": tier_actions,
            "calculation_time_ms": calculation_time,
            "timestamp": datetime.now().isoformat(),
            "breach_status": {
                "tier_1": current_dd >= self.limits.tier_1_drawdown,
                "tier_2": current_dd >= self.limits.tier_2_drawdown,
                "tier_3": current_dd >= self.limits.tier_3_drawdown
            }
        }

    def optimized_factor_crowding_check(self,
                                      factor_exposures: Dict[str, np.ndarray],
                                      portfolio_weights: np.ndarray) -> Dict[str, Any]:
        """
        High-performance factor crowding detection
        Optimized for real-time monitoring of large portfolios
        """
        start_time = time.perf_counter()

        crowding_results = {}

        for factor_name, exposures in factor_exposures.items():
            # Fast HHI calculation
            weighted_exposures = exposures * portfolio_weights
            normalized_weights = weighted_exposures / np.sum(np.abs(weighted_exposures))
            hhi = fast_herfindahl_index(np.abs(normalized_weights))

            # Fast Gini coefficient
            gini = fast_gini_coefficient(np.abs(normalized_weights))

            # Effective breadth (inverse of HHI)
            effective_breadth = 1 / hhi if hhi > 0 else len(exposures)

            # Crowding assessment
            is_crowded = (hhi > self.limits.hhi_crowding_threshold or
                         gini > self.limits.gini_crowding_threshold or
                         effective_breadth < self.limits.effective_breadth_min)

            # Crowding score (0-100)
            hhi_score = min(100, (hhi / self.limits.hhi_crowding_threshold) * 50)
            gini_score = min(100, (gini / self.limits.gini_crowding_threshold) * 50)
            crowding_score = max(hhi_score, gini_score)

            crowding_results[factor_name] = {
                "hhi": hhi,
                "gini": gini,
                "effective_breadth": effective_breadth,
                "is_crowded": is_crowded,
                "crowding_score": crowding_score,
                "breach_thresholds": {
                    "hhi_breach": hhi > self.limits.hhi_crowding_threshold,
                    "gini_breach": gini > self.limits.gini_crowding_threshold,
                    "breadth_breach": effective_breadth < self.limits.effective_breadth_min
                }
            }

        calculation_time = (time.perf_counter() - start_time) * 1000

        # Overall portfolio crowding assessment
        avg_crowding_score = np.mean([r["crowding_score"] for r in crowding_results.values()])
        crowded_factors = [f for f, r in crowding_results.items() if r["is_crowded"]]

        return {
            "factor_results": crowding_results,
            "overall_crowding_score": avg_crowding_score,
            "crowded_factors": crowded_factors,
            "crowding_risk_level": "HIGH" if len(crowded_factors) > 2 else "MEDIUM" if len(crowded_factors) > 0 else "LOW",
            "calculation_time_ms": calculation_time,
            "timestamp": datetime.now().isoformat()
        }

    def concurrent_risk_assessment(self,
                                 portfolio_data: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Concurrent risk assessment for maximum performance
        Executes multiple risk calculations in parallel
        """
        start_time = time.perf_counter()
        self.active_calculations += 1

        try:
            # Prepare data
            returns = portfolio_data.get("returns", np.array([]))
            values = portfolio_data.get("values", np.array([]))
            factor_exposures = portfolio_data.get("factor_exposures", {})
            weights = portfolio_data.get("weights", np.array([]))

            # Submit concurrent calculations
            futures = {}

            # ES calculation
            if len(returns) > 0:
                futures["es_metrics"] = self.executor.submit(
                    self.calculate_portfolio_es_optimized, returns
                )

            # Drawdown monitoring
            if len(values) > 0:
                futures["drawdown_metrics"] = self.executor.submit(
                    self.real_time_drawdown_monitor, values
                )

            # Factor crowding
            if factor_exposures and len(weights) > 0:
                futures["crowding_metrics"] = self.executor.submit(
                    self.optimized_factor_crowding_check, factor_exposures, weights
                )

            # Collect results
            results = {}
            for metric_name, future in futures.items():
                try:
                    results[metric_name] = future.result(timeout=1.0)  # 1 second timeout
                except Exception as e:
                    logger.error(f"Error in {metric_name}: {e}")
                    results[metric_name] = {"error": str(e)}

            # Overall risk assessment
            total_calculation_time = (time.perf_counter() - start_time) * 1000

            # Determine overall risk level
            risk_indicators = []
            if "es_metrics" in results:
                es_97_5 = results["es_metrics"].get("es_97", 0)
                if es_97_5 > self.limits.es_97_5_daily_limit:
                    risk_indicators.append("HIGH_ES")

            if "drawdown_metrics" in results:
                tier = results["drawdown_metrics"].get("risk_tier", 0)
                if tier >= 2:
                    risk_indicators.append("HIGH_DRAWDOWN")

            if "crowding_metrics" in results:
                crowding_level = results["crowding_metrics"].get("crowding_risk_level", "LOW")
                if crowding_level in ["HIGH", "MEDIUM"]:
                    risk_indicators.append("FACTOR_CROWDING")

            # Overall risk level
            if "HIGH_ES" in risk_indicators or "HIGH_DRAWDOWN" in risk_indicators:
                overall_risk = "HIGH"
            elif len(risk_indicators) > 0:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"

            # Compile final assessment
            assessment = {
                "timestamp": datetime.now().isoformat(),
                "overall_risk_level": overall_risk,
                "risk_indicators": risk_indicators,
                "detailed_metrics": results,
                "performance": {
                    "total_calculation_time_ms": total_calculation_time,
                    "concurrent_calculations": len(futures),
                    "cache_hit_rate": self.cache.get_hit_rate(),
                    "memory_usage_mb": self.metrics.memory_usage_mb
                },
                "alerts": self._generate_risk_alerts(results, risk_indicators)
            }

            # Performance tracking
            self.calculation_count += 1
            if self.calculation_count % self.gc_interval == 0:
                gc.collect()

            return assessment

        finally:
            self.active_calculations -= 1

    def _generate_risk_alerts(self, results: Dict[str, Any], indicators: List[str]) -> List[Dict[str, Any]]:
        """Generate actionable risk alerts based on assessment results"""
        alerts = []

        for indicator in indicators:
            if indicator == "HIGH_ES":
                alerts.append({
                    "level": "HIGH",
                    "category": "Tail Risk",
                    "message": "Expected Shortfall exceeds daily limit",
                    "actions": ["reduce_position_sizes", "implement_hedging", "increase_diversification"]
                })
            elif indicator == "HIGH_DRAWDOWN":
                tier = results.get("drawdown_metrics", {}).get("risk_tier", 0)
                alerts.append({
                    "level": "CRITICAL" if tier >= 3 else "HIGH",
                    "category": "Drawdown Risk",
                    "message": f"Drawdown tier {tier} activated",
                    "actions": results.get("drawdown_metrics", {}).get("tier_actions", [])
                })
            elif indicator == "FACTOR_CROWDING":
                crowded_factors = results.get("crowding_metrics", {}).get("crowded_factors", [])
                alerts.append({
                    "level": "MEDIUM",
                    "category": "Factor Crowding",
                    "message": f"Crowded factors detected: {', '.join(crowded_factors[:3])}",
                    "actions": ["diversify_factors", "reduce_concentration", "rebalance_exposures"]
                })

        return alerts

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get real-time performance dashboard data"""
        return {
            "performance_level": self.performance_level.value,
            "real_time_metrics": {
                "calculation_time_ms": self.metrics.calculation_time_ms,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "concurrent_calculations": self.metrics.concurrent_calculations
            },
            "optimization_stats": {
                "total_calculations": self.calculation_count,
                "active_calculations": self.active_calculations,
                "cache_size": len(self.cache.cache),
                "max_workers": self.max_workers
            },
            "risk_limits": {
                "es_97_5_daily": self.limits.es_97_5_daily_limit,
                "tier_1_drawdown": self.limits.tier_1_drawdown,
                "tier_2_drawdown": self.limits.tier_2_drawdown,
                "tier_3_drawdown": self.limits.tier_3_drawdown,
                "hhi_threshold": self.limits.hhi_crowding_threshold
            },
            "timestamp": datetime.now().isoformat()
        }

    def export_performance_report(self, filepath: str) -> bool:
        """Export comprehensive performance and calibration report"""
        try:
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "optimizer_config": {
                    "performance_level": self.performance_level.value,
                    "max_workers": self.max_workers,
                    "cache_size": len(self.cache.cache),
                    "memory_threshold_mb": self.memory_threshold_mb
                },
                "calibrated_limits": {
                    "es_97_5_daily_limit": self.limits.es_97_5_daily_limit,
                    "es_97_5_weekly_limit": self.limits.es_97_5_weekly_limit,
                    "tier_1_drawdown": self.limits.tier_1_drawdown,
                    "tier_2_drawdown": self.limits.tier_2_drawdown,
                    "tier_3_drawdown": self.limits.tier_3_drawdown,
                    "max_single_position": self.limits.max_single_position,
                    "hhi_crowding_threshold": self.limits.hhi_crowding_threshold,
                    "gini_crowding_threshold": self.limits.gini_crowding_threshold
                },
                "performance_metrics": self.get_performance_dashboard(),
                "optimization_recommendations": {
                    "current_performance": "OPTIMIZED" if self.metrics.calculation_time_ms < 100 else "NEEDS_OPTIMIZATION",
                    "memory_efficiency": "GOOD" if self.metrics.memory_usage_mb < 300 else "REVIEW_NEEDED",
                    "cache_effectiveness": "EXCELLENT" if self.cache.get_hit_rate() > 0.8 else "GOOD" if self.cache.get_hit_rate() > 0.6 else "POOR"
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Performance optimization report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.cache.clear()
        gc.collect()

# Testing and calibration functions
def run_performance_calibration_test():
    """Run comprehensive performance calibration test"""
    print("Risk Management Performance Calibration Test")
    print("=" * 50)

    # Initialize optimizer
    optimizer = RiskPerformanceOptimizer(
        performance_level=PerformanceLevel.DEVELOPMENT,
        max_workers=4,
        cache_size=500
    )

    # Start monitoring
    optimizer.start_performance_monitoring()

    # Generate test data (simulating real portfolio)
    np.random.seed(42)
    n_periods = 252  # 1 year daily data
    n_stocks = 100   # Portfolio size

    # Simulate portfolio returns and values
    returns = np.random.normal(0.0008, 0.015, n_periods)  # Daily returns ~20% annual vol
    returns[200:210] = np.random.normal(-0.03, 0.01, 10)  # Crisis period

    values = np.cumprod(1 + returns) * 1000000  # Starting with $1M

    # Simulate factor exposures
    factor_exposures = {
        "momentum": np.random.normal(0, 1, n_stocks),
        "value": np.random.normal(0, 1, n_stocks),
        "quality": np.random.normal(0, 1, n_stocks),
        "low_vol": np.random.normal(0, 1, n_stocks)
    }

    # Add some factor crowding
    factor_exposures["momentum"][:20] = 2.5  # Concentrated momentum exposure

    # Portfolio weights (concentrated)
    weights = np.random.dirichlet(np.ones(n_stocks) * 0.3)  # Concentrated allocation

    # Test data
    portfolio_data = {
        "returns": returns,
        "values": values,
        "factor_exposures": factor_exposures,
        "weights": weights
    }

    market_data = {
        "vix": 22.0,
        "market_correlation": 0.6
    }

    # Run performance tests
    print("Running concurrent risk assessment...")
    start_time = time.perf_counter()

    # Multiple concurrent assessments to test performance
    results = []
    for i in range(10):
        result = optimizer.concurrent_risk_assessment(portfolio_data, market_data)
        results.append(result)
        print(f"Assessment {i+1}: {result['performance']['total_calculation_time_ms']:.1f}ms")

    total_time = time.perf_counter() - start_time
    avg_calculation_time = np.mean([r['performance']['total_calculation_time_ms'] for r in results])

    print(f"\nPerformance Results:")
    print(f"Total test time: {total_time:.2f} seconds")
    print(f"Average calculation time: {avg_calculation_time:.1f}ms")
    print(f"Cache hit rate: {optimizer.cache.get_hit_rate():.2f}")

    # Display latest assessment
    latest = results[-1]
    print(f"\nLatest Risk Assessment:")
    print(f"Overall Risk Level: {latest['overall_risk_level']}")
    print(f"Risk Indicators: {latest['risk_indicators']}")

    if "es_metrics" in latest["detailed_metrics"]:
        es_data = latest["detailed_metrics"]["es_metrics"]
        print(f"ES@97.5%: {es_data.get('es_97', 0):.4f}")

    if "drawdown_metrics" in latest["detailed_metrics"]:
        dd_data = latest["detailed_metrics"]["drawdown_metrics"]
        print(f"Current Drawdown: {dd_data.get('current_drawdown', 0):.4f}")
        print(f"Risk Tier: {dd_data.get('risk_tier', 0)}")

    if "crowding_metrics" in latest["detailed_metrics"]:
        crowding_data = latest["detailed_metrics"]["crowding_metrics"]
        print(f"Crowded Factors: {crowding_data.get('crowded_factors', [])}")

    # Export performance report
    optimizer.export_performance_report("risk_performance_calibration_report.json")

    # Cleanup
    optimizer.cleanup()

    print(f"\nCalibration test completed successfully!")
    print(f"Performance report exported: risk_performance_calibration_report.json")

    return True

if __name__ == "__main__":
    # Run calibration test
    run_performance_calibration_test()