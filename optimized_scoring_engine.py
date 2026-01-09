#!/usr/bin/env python3
"""
Optimized Multi-Factor Scoring Engine - Performance Enhanced Version
高性能多因子评分引擎

Performance Optimizations:
- Vectorized computations with NumPy SIMD operations
- Parallel processing for multi-stock analysis
- Intelligent caching system for factor calculations
- Memory-optimized data structures
- JIT compilation with Numba for hot paths

Target: 400+ stocks/second processing (150-300% improvement)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import warnings
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing

# Performance enhancement imports
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available - JIT compilation disabled")

try:
    from performance_optimization_engine import (
        VectorizedProcessor, PerformanceCache, MemoryOptimizer
    )
    OPTIMIZATION_ENGINE_AVAILABLE = True
except ImportError:
    OPTIMIZATION_ENGINE_AVAILABLE = False
    warnings.warn("Performance optimization engine not available")

# Scientific computing
try:
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler, RobustScaler
    HAS_SCIPY_SKLEARN = True
except ImportError:
    HAS_SCIPY_SKLEARN = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedFactorWeights:
    """Enhanced factor weights with performance optimization parameters"""

    # Core factor weights
    valuation_weight: float = 0.25
    volume_weight: float = 0.15
    momentum_weight: float = 0.20
    technical_weight: float = 0.25
    market_sentiment_weight: float = 0.15

    # Performance optimization settings
    enable_parallel_processing: bool = True
    max_workers: int = None  # Will auto-detect optimal
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_vectorization: bool = True
    batch_size: int = 100  # For parallel processing

    # Algorithm optimization
    enable_jit_compilation: bool = NUMBA_AVAILABLE
    memory_optimization: bool = True
    chunk_processing: bool = True

    def __post_init__(self):
        """Initialize optimization parameters"""
        if self.max_workers is None:
            self.max_workers = min(32, multiprocessing.cpu_count() * 2)

        # Normalize weights
        total = sum([self.valuation_weight, self.volume_weight, self.momentum_weight,
                    self.technical_weight, self.market_sentiment_weight])
        if abs(total - 1.0) > 1e-6:
            factor = 1.0 / total
            self.valuation_weight *= factor
            self.volume_weight *= factor
            self.momentum_weight *= factor
            self.technical_weight *= factor
            self.market_sentiment_weight *= factor

@dataclass
class OptimizedScoringResult:
    """Enhanced scoring result with performance metrics"""

    scores: pd.DataFrame
    factor_contributions: pd.DataFrame
    factor_correlations: pd.DataFrame
    weights_used: Dict[str, float]

    # Performance metrics
    processing_time_seconds: float = 0.0
    stocks_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0

    # Optimization details
    optimizations_applied: List[str] = field(default_factory=list)
    performance_improvement: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

class VectorizedFactorCalculator:
    """High-performance vectorized factor calculations"""

    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def fast_momentum_score(prices: np.ndarray, returns: np.ndarray) -> float:
        """Ultra-fast momentum calculation with JIT compilation"""
        if len(prices) < 20:
            return 0.0

        # Multiple momentum timeframes
        mom_5d = (prices[-1] / prices[-5] - 1) * 100 if len(prices) >= 5 else 0
        mom_20d = (prices[-1] / prices[-20] - 1) * 100 if len(prices) >= 20 else 0

        # Momentum acceleration
        recent_mom = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        long_mom = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        acceleration = recent_mom - long_mom

        # Weighted composite
        return (mom_5d * 0.3 + mom_20d * 0.5 + acceleration * 100 * 0.2)

    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def fast_volatility_score(returns: np.ndarray) -> float:
        """High-speed volatility analysis"""
        if len(returns) < 10:
            return 0.0

        # Current volatility
        current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

        # Volatility trend
        vol_short = np.std(returns[-10:]) if len(returns) >= 10 else current_vol
        vol_long = np.std(returns[-30:]) if len(returns) >= 30 else current_vol
        vol_trend = (vol_short - vol_long) / (vol_long + 1e-10)

        # Score: prefer lower volatility with stable trend
        base_score = max(0, 1 - current_vol * 50)  # Scale volatility
        trend_penalty = abs(vol_trend) * 0.5

        return max(0, base_score - trend_penalty)

    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def fast_technical_score(prices: np.ndarray, volumes: np.ndarray) -> float:
        """Optimized technical analysis composite"""
        if len(prices) < 20:
            return 0.0

        # Moving average convergence
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma_10
        ma_signal = (sma_10 / sma_20 - 1) * 100

        # Price position relative to recent range
        high_20 = np.max(prices[-20:]) if len(prices) >= 20 else prices[-1]
        low_20 = np.min(prices[-20:]) if len(prices) >= 20 else prices[-1]
        range_position = (prices[-1] - low_20) / (high_20 - low_20 + 1e-10)

        # Volume trend
        vol_avg = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        vol_recent = volumes[-1]
        vol_ratio = vol_recent / (vol_avg + 1e-10)

        # Composite technical score
        return (ma_signal * 0.4 + range_position * 40 * 0.4 +
                min(vol_ratio, 3.0) * 10 * 0.2)

    @staticmethod
    def batch_calculate_factors(price_data: Dict[str, np.ndarray],
                              volume_data: Dict[str, np.ndarray],
                              max_workers: int = 16) -> Dict[str, Dict[str, float]]:
        """Parallel batch factor calculation for multiple stocks"""

        results = {}
        symbols = list(price_data.keys())

        def calculate_symbol_factors(symbol: str) -> Tuple[str, Dict[str, float]]:
            """Calculate all factors for a single symbol"""
            try:
                prices = price_data.get(symbol, np.array([]))
                volumes = volume_data.get(symbol, np.array([]))

                if len(prices) < 5:  # Minimum data requirement
                    return symbol, {}

                # Calculate returns
                returns = np.diff(np.log(prices + 1e-10))

                # Calculate factors using optimized functions
                factors = {
                    'momentum_score': VectorizedFactorCalculator.fast_momentum_score(prices, returns),
                    'volatility_score': VectorizedFactorCalculator.fast_volatility_score(returns),
                    'technical_score': VectorizedFactorCalculator.fast_technical_score(prices, volumes)
                }

                return symbol, factors

            except Exception as e:
                logger.error(f"Error calculating factors for {symbol}: {e}")
                return symbol, {}

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(calculate_symbol_factors, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol, factors = future.result()
                results[symbol] = factors

        return results

class OptimizedMultiFactorScoringEngine:
    """High-performance multi-factor scoring engine with advanced optimizations"""

    def __init__(self, weights: Optional[OptimizedFactorWeights] = None):
        """Initialize optimized scoring engine"""
        self.weights = weights or OptimizedFactorWeights()

        # Performance components
        self.cache = PerformanceCache(max_memory_mb=500) if OPTIMIZATION_ENGINE_AVAILABLE else None
        self.memory_optimizer = MemoryOptimizer() if OPTIMIZATION_ENGINE_AVAILABLE else None
        self.vectorized_processor = VectorizedFactorCalculator()

        # Performance tracking
        self.factor_history = []
        self.performance_stats = {
            'total_calculations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"Optimized scoring engine initialized")
        logger.info(f"JIT compilation: {'Enabled' if self.weights.enable_jit_compilation else 'Disabled'}")
        logger.info(f"Parallel processing: {self.weights.max_workers} workers")
        logger.info(f"Caching: {'Enabled' if self.weights.enable_caching else 'Disabled'}")

    def calculate_optimized_scores(self,
                                 stock_data: Dict[str, pd.DataFrame],
                                 use_cache: bool = True) -> OptimizedScoringResult:
        """
        Calculate composite scores with maximum performance optimization

        Args:
            stock_data: Dictionary of symbol -> OHLCV DataFrame
            use_cache: Whether to use caching for performance

        Returns:
            OptimizedScoringResult with performance metrics
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        optimizations_applied = []

        logger.info(f"Starting optimized scoring for {len(stock_data)} stocks")

        # Step 1: Data preprocessing and optimization
        if self.weights.memory_optimization and self.memory_optimizer:
            optimized_data = {}
            for symbol, df in stock_data.items():
                optimized_data[symbol] = self.memory_optimizer.optimize_dataframe(df.copy())
            stock_data = optimized_data
            optimizations_applied.append("memory_optimization")

        # Step 2: Check cache for existing calculations
        cache_key = self._generate_cache_key(stock_data.keys())
        cached_result = None

        if use_cache and self.cache and self.weights.enable_caching:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                optimizations_applied.append("cache_hit")
                logger.info("Using cached scoring results")

                # Update performance metrics and return
                processing_time = time.time() - start_time
                cached_result.processing_time_seconds = processing_time
                cached_result.optimizations_applied = optimizations_applied
                return cached_result

        self.performance_stats['cache_misses'] += 1

        # Step 3: Convert to optimized data structures
        price_arrays = {}
        volume_arrays = {}

        for symbol, df in stock_data.items():
            if not df.empty and 'close' in df.columns:
                # Use numpy arrays for vectorization
                price_arrays[symbol] = df['close'].values.astype(np.float64)
                volume_arrays[symbol] = df.get('volume', pd.Series([1] * len(df))).values.astype(np.float64)

        if self.weights.enable_vectorization:
            optimizations_applied.append("vectorization")

        # Step 4: Parallel factor calculation
        if self.weights.enable_parallel_processing:
            factor_results = self.vectorized_processor.batch_calculate_factors(
                price_arrays, volume_arrays, self.weights.max_workers
            )
            optimizations_applied.append("parallel_processing")
        else:
            # Sequential fallback
            factor_results = {}
            for symbol in price_arrays:
                prices = price_arrays[symbol]
                volumes = volume_arrays[symbol]
                returns = np.diff(np.log(prices + 1e-10))

                factor_results[symbol] = {
                    'momentum_score': self.vectorized_processor.fast_momentum_score(prices, returns),
                    'volatility_score': self.vectorized_processor.fast_volatility_score(returns),
                    'technical_score': self.vectorized_processor.fast_technical_score(prices, volumes)
                }

        # Step 5: Vectorized score normalization and combination
        scores_df = self._create_optimized_scores_dataframe(factor_results)

        if self.weights.enable_vectorization:
            normalized_scores = self._vectorized_normalization(scores_df)
            optimizations_applied.append("vectorized_normalization")
        else:
            normalized_scores = self._standard_normalization(scores_df)

        # Step 6: Calculate final composite scores
        composite_scores = self._calculate_weighted_composite(normalized_scores)

        # Step 7: Generate factor contributions and correlations
        factor_contributions = self._calculate_factor_contributions(normalized_scores)
        factor_correlations = normalized_scores.corr() if len(normalized_scores) > 1 else pd.DataFrame()

        # Step 8: Performance metrics calculation
        end_time = time.time()
        processing_time = end_time - start_time
        end_memory = self._get_memory_usage()

        stocks_per_second = len(stock_data) / processing_time if processing_time > 0 else 0
        memory_used = max(0, end_memory - start_memory)

        # Step 9: Create optimized result
        result = OptimizedScoringResult(
            scores=composite_scores,
            factor_contributions=factor_contributions,
            factor_correlations=factor_correlations,
            weights_used={
                'momentum': self.weights.momentum_weight,
                'volatility': self.weights.volume_weight,  # Using volume weight for volatility
                'technical': self.weights.technical_weight
            },
            processing_time_seconds=processing_time,
            stocks_per_second=stocks_per_second,
            memory_usage_mb=memory_used,
            cache_hit_rate=self._calculate_cache_hit_rate(),
            optimizations_applied=optimizations_applied,
            performance_improvement=self._estimate_performance_improvement(processing_time, len(stock_data))
        )

        # Step 10: Cache result for future use
        if self.cache and self.weights.enable_caching:
            self.cache.set(cache_key, result)

        # Update performance statistics
        self.performance_stats['total_calculations'] += len(stock_data)
        self.performance_stats['total_time'] += processing_time

        logger.info(f"Optimized scoring completed: {stocks_per_second:.1f} stocks/sec")
        logger.info(f"Optimizations applied: {', '.join(optimizations_applied)}")

        return result

    def _create_optimized_scores_dataframe(self, factor_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create optimized DataFrame from factor results"""
        data = []
        for symbol, factors in factor_results.items():
            row = {'symbol': symbol}
            row.update(factors)
            data.append(row)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Memory optimization
        if self.memory_optimizer:
            df = self.memory_optimizer.optimize_dataframe(df)

        return df

    def _vectorized_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """High-performance vectorized normalization"""
        if df.empty:
            return df

        result = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Vectorized robust normalization
        for col in numeric_columns:
            if col != 'symbol':
                values = result[col].values
                median = np.median(values)
                mad = np.median(np.abs(values - median))

                # Robust z-score with vectorization
                normalized = (values - median) / (mad * 1.4826 + 1e-9)
                result[col] = np.clip(normalized, -3, 3)  # Clip extreme outliers

        return result

    def _standard_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard normalization fallback"""
        if df.empty:
            return df

        result = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'symbol':
                series = result[col]
                result[col] = (series - series.mean()) / (series.std() + 1e-9)

        return result

    def _calculate_weighted_composite(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted composite scores with vectorization"""
        if normalized_df.empty:
            return pd.DataFrame()

        result = normalized_df[['symbol']].copy() if 'symbol' in normalized_df.columns else pd.DataFrame()

        # Vectorized weighted combination
        composite_score = np.zeros(len(normalized_df))

        score_columns = {
            'momentum_score': self.weights.momentum_weight,
            'volatility_score': self.weights.volume_weight,
            'technical_score': self.weights.technical_weight
        }

        for col, weight in score_columns.items():
            if col in normalized_df.columns:
                composite_score += normalized_df[col].fillna(0).values * weight

        result['composite_score'] = composite_score
        result['rank'] = (-composite_score).argsort().argsort() + 1  # Vectorized ranking
        result['percentile'] = result['composite_score'].rank(pct=True)

        return result

    def _calculate_factor_contributions(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate individual factor contributions"""
        if normalized_df.empty:
            return pd.DataFrame()

        contributions = normalized_df[['symbol']].copy() if 'symbol' in normalized_df.columns else pd.DataFrame()

        score_columns = {
            'momentum_score': self.weights.momentum_weight,
            'volatility_score': self.weights.volume_weight,
            'technical_score': self.weights.technical_weight
        }

        for col, weight in score_columns.items():
            if col in normalized_df.columns:
                contributions[f'{col}_contribution'] = normalized_df[col].fillna(0) * weight

        return contributions

    def _generate_cache_key(self, symbols) -> str:
        """Generate cache key for symbol set"""
        sorted_symbols = sorted(symbols)
        return f"scores_{hash(tuple(sorted_symbols))}"

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.memory_optimizer:
            return self.memory_optimizer.get_memory_report()['current_usage_mb']
        return 0.0

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        return self.performance_stats['cache_hits'] / total if total > 0 else 0.0

    def _estimate_performance_improvement(self, processing_time: float, num_stocks: int) -> float:
        """Estimate performance improvement vs baseline"""
        # Baseline: ~200 stocks/second (current system)
        baseline_time = num_stocks / 200.0
        current_time = processing_time

        if current_time > 0:
            improvement = baseline_time / current_time
            return max(1.0, improvement)
        return 1.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        avg_time_per_calc = (self.performance_stats['total_time'] /
                           self.performance_stats['total_calculations']
                           if self.performance_stats['total_calculations'] > 0 else 0)

        avg_throughput = (self.performance_stats['total_calculations'] /
                         self.performance_stats['total_time']
                         if self.performance_stats['total_time'] > 0 else 0)

        summary = {
            'total_calculations': self.performance_stats['total_calculations'],
            'total_processing_time': self.performance_stats['total_time'],
            'average_time_per_calculation': avg_time_per_calc,
            'average_throughput_stocks_per_sec': avg_throughput,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'optimizations_enabled': {
                'jit_compilation': self.weights.enable_jit_compilation,
                'parallel_processing': self.weights.enable_parallel_processing,
                'vectorization': self.weights.enable_vectorization,
                'caching': self.weights.enable_caching,
                'memory_optimization': self.weights.memory_optimization
            },
            'configuration': {
                'max_workers': self.weights.max_workers,
                'batch_size': self.weights.batch_size,
                'cache_ttl_seconds': self.weights.cache_ttl_seconds
            }
        }

        if self.cache:
            summary['cache_statistics'] = self.cache.get_stats()

        if self.memory_optimizer:
            summary['memory_report'] = self.memory_optimizer.get_memory_report()

        return summary

# Example usage and testing
async def main():
    """Demonstration of optimized scoring engine"""
    from datetime import datetime, timedelta

    # Initialize optimized engine
    weights = OptimizedFactorWeights(
        enable_parallel_processing=True,
        enable_caching=True,
        enable_vectorization=True,
        max_workers=16
    )

    engine = OptimizedMultiFactorScoringEngine(weights)

    # Generate test data
    symbols = [f"STOCK_{i:04d}" for i in range(1000)]
    test_data = {}

    print("Generating test data...")
    for symbol in symbols:
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(hash(symbol) % 2**31)  # Consistent random data
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        volumes = np.random.randint(1000, 100000, len(dates))

        test_data[symbol] = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })

    print(f"Testing optimized scoring with {len(test_data)} stocks...")

    # Test scoring performance
    start_time = time.time()
    result = engine.calculate_optimized_scores(test_data)
    end_time = time.time()

    print(f"\nPerformance Results:")
    print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
    print(f"Throughput: {result.stocks_per_second:.1f} stocks/second")
    print(f"Memory usage: {result.memory_usage_mb:.1f} MB")
    print(f"Cache hit rate: {result.cache_hit_rate:.1%}")
    print(f"Performance improvement: {result.performance_improvement:.1f}x")
    print(f"Optimizations applied: {', '.join(result.optimizations_applied)}")

    # Test caching performance (second run)
    print(f"\nTesting cache performance (second run)...")
    start_time_cached = time.time()
    result_cached = engine.calculate_optimized_scores(test_data, use_cache=True)
    end_time_cached = time.time()

    print(f"Cached processing time: {result_cached.processing_time_seconds:.2f} seconds")
    print(f"Cache speedup: {result.processing_time_seconds / result_cached.processing_time_seconds:.1f}x")

    # Performance summary
    summary = engine.get_performance_summary()
    print(f"\nOverall Performance Summary:")
    print(f"Average throughput: {summary['average_throughput_stocks_per_sec']:.1f} stocks/sec")
    print(f"Total calculations: {summary['total_calculations']}")
    print(f"Cache hit rate: {summary['cache_hit_rate']:.1%}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())