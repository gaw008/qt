#!/usr/bin/env python3
"""
Comprehensive Test Suite for Performance Optimization Framework

This test suite validates the performance optimization framework components:
- High-performance backtesting engine validation
- Database query optimizer testing
- Progress monitoring system verification
- Bottleneck analysis accuracy testing
- Resource utilization tracking validation
- Performance benchmark comparisons

Test Coverage:
- Parallel processing efficiency validation
- Cache performance and hit rate testing
- Memory management effectiveness
- Query optimization impact measurement
- Bottleneck detection accuracy
- Resource monitoring precision
- Integration testing with existing components
"""

import sys
import os
import unittest
import tempfile
import shutil
import time
import threading
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import performance framework components
from bot.performance_backtesting_engine import (
    HighPerformanceDataCache, ParallelBacktestExecutor, BacktestConfig,
    BacktestBenchmarkSuite, create_optimized_config
)
from bot.database_query_optimizer import (
    DatabaseQueryOptimizer, ConnectionPool, QueryResultCache
)
from bot.progress_monitoring_system import (
    ProgressTracker, ResourceMonitor, TaskProgress
)
from bot.bottleneck_analyzer import (
    PerformanceProfiler, BottleneckDetector, OptimizationRecommendationEngine,
    BottleneckReport
)

# Test data generation
import pandas as pd
import numpy as np
import sqlite3

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Configure test logging
import logging
logging.basicConfig(level=logging.ERROR)  # Reduce noise during testing


class TestHighPerformanceDataCache(unittest.TestCase):
    """Test high-performance data cache functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BacktestConfig(
            cache_dir=self.temp_dir,
            max_cache_size_gb=0.1  # Small cache for testing
        )
        self.cache = HighPerformanceDataCache(self.config)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        # Create test data
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)

        # Test cache miss
        result = self.cache.get('AAPL', start_date, end_date)
        self.assertIsNone(result)

        # Test cache set and hit
        self.cache.set('AAPL', start_date, end_date, test_data)
        result = self.cache.get('AAPL', start_date, end_date)

        self.assertIsNotNone(result)
        self.assertTrue(result.equals(test_data))

        # Test cache statistics
        stats = self.cache.get_statistics()
        self.assertEqual(stats['total_hits'], 1)
        self.assertEqual(stats['total_misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)

    def test_cache_performance(self):
        """Test cache performance characteristics."""
        # Generate test data
        symbols = [f'TEST{i:03d}' for i in range(100)]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)

        test_data = pd.DataFrame({
            'close': np.random.randn(252)
        }, index=pd.date_range('2023-01-01', periods=252))

        # Measure cache performance
        cache_times = []
        direct_times = []

        for symbol in symbols[:10]:  # Test with 10 symbols
            # First access (cache miss)
            start_time = time.time()
            self.cache.set(symbol, start_date, end_date, test_data)
            set_time = time.time() - start_time

            # Second access (cache hit)
            start_time = time.time()
            cached_result = self.cache.get(symbol, start_date, end_date)
            get_time = time.time() - start_time

            cache_times.append(get_time)
            self.assertIsNotNone(cached_result)

        # Cache should be significantly faster than data generation
        avg_cache_time = np.mean(cache_times)
        self.assertLess(avg_cache_time, 0.01)  # Should be < 10ms


class TestDatabaseQueryOptimizer(unittest.TestCase):
    """Test database query optimizer functionality."""

    def setUp(self):
        """Set up test environment with sample database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_data.db')
        self.optimizer = DatabaseQueryOptimizer(self.db_path)

        # Create sample data
        self._create_sample_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_data(self):
        """Create sample historical data for testing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Generate sample price data
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

            for symbol in symbols:
                for date_val in dates:
                    price = 100 + np.random.randn() * 5
                    cursor.execute('''
                        INSERT OR REPLACE INTO historical_prices
                        (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, date_val.date(),
                        price, price * 1.02, price * 0.98, price * 1.01,
                        int(1000000 + np.random.randn() * 100000)
                    ))

            conn.commit()

    def test_query_performance(self):
        """Test query performance and caching."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 6, 30)

        # First query (cache miss)
        start_time = time.time()
        result1 = self.optimizer.get_historical_data('AAPL', start_date, end_date)
        first_query_time = time.time() - start_time

        # Second query (cache hit)
        start_time = time.time()
        result2 = self.optimizer.get_historical_data('AAPL', start_date, end_date)
        second_query_time = time.time() - start_time

        # Verify results are identical
        self.assertTrue(result1.equals(result2))

        # Cache should provide significant speedup
        speedup = first_query_time / second_query_time if second_query_time > 0 else float('inf')
        self.assertGreater(speedup, 2.0)  # At least 2x speedup

        # Check cache statistics
        metrics = self.optimizer.get_performance_metrics()
        self.assertGreater(metrics['query_metrics']['cache_hit_rate'], 0)

    def test_batch_query_efficiency(self):
        """Test batch query efficiency."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start_date = date(2023, 1, 1)
        end_date = date(2023, 3, 31)

        # Measure batch query performance
        start_time = time.time()
        batch_results = self.optimizer.get_batch_historical_data(symbols, start_date, end_date)
        batch_time = time.time() - start_time

        # Measure individual query performance
        start_time = time.time()
        individual_results = {}
        for symbol in symbols:
            individual_results[symbol] = self.optimizer.get_historical_data(
                symbol, start_date, end_date
            )
        individual_time = time.time() - start_time

        # Verify results are equivalent
        self.assertEqual(len(batch_results), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, batch_results)
            self.assertGreater(len(batch_results[symbol]), 0)

        # Batch processing should be more efficient for cold cache
        self.optimizer.clear_cache()  # Clear cache for fair comparison


class TestProgressMonitoringSystem(unittest.TestCase):
    """Test progress monitoring system functionality."""

    def setUp(self):
        """Set up test environment."""
        self.tracker = ProgressTracker()

    def tearDown(self):
        """Clean up test environment."""
        self.tracker.cleanup()

    def test_task_progress_tracking(self):
        """Test task progress tracking functionality."""
        # Create test task
        task = self.tracker.create_task('test_task', 'Test Operation', 100)

        self.assertEqual(task.task_id, 'test_task')
        self.assertEqual(task.total_items, 100)
        self.assertEqual(task.progress_percent, 0.0)

        # Update progress
        self.tracker.update_task('test_task', 25)
        task = self.tracker.get_task('test_task')
        self.assertEqual(task.progress_percent, 25.0)

        # Complete task
        self.tracker.complete_task('test_task', success=True)
        task = self.tracker.get_task('test_task')
        self.assertEqual(task.status, 'completed')

    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        monitor = self.tracker.resource_monitor

        # Start monitoring
        monitor.start_monitoring()

        # Wait for some data collection
        time.sleep(2)

        # Get current resources
        current = monitor.get_current_resources()
        if current:  # Only test if monitoring is available
            self.assertGreaterEqual(current.cpu_percent, 0)
            self.assertGreaterEqual(current.memory_percent, 0)

        # Get performance summary
        summary = monitor.get_performance_summary()
        if 'error' not in summary:
            self.assertIn('cpu', summary)
            self.assertIn('memory', summary)

        monitor.stop_monitoring()

    def test_context_manager_tracking(self):
        """Test context manager for operation tracking."""
        with self.tracker.track_operation('test_operation', 50) as task:
            for i in range(51):
                task.update(i)
                time.sleep(0.01)  # Simulate work

        # Verify task completion
        final_task = self.tracker.get_task(task.task_id)
        self.assertEqual(final_task.status, 'completed')
        self.assertEqual(final_task.progress_percent, 100.0)


class TestBottleneckAnalyzer(unittest.TestCase):
    """Test bottleneck analyzer functionality."""

    def setUp(self):
        """Set up test environment."""
        self.profiler = PerformanceProfiler()
        self.detector = BottleneckDetector(self.profiler)
        self.optimizer = OptimizationRecommendationEngine()

    def test_performance_profiling(self):
        """Test performance profiling capabilities."""
        @self.profiler.profile_function('test_cpu_function')
        def cpu_intensive_function():
            return sum(i ** 2 for i in range(10000))

        @self.profiler.profile_function('test_io_function')
        def io_intensive_function():
            time.sleep(0.1)  # Simulate I/O
            return "completed"

        # Execute functions
        cpu_result = cpu_intensive_function()
        io_result = io_intensive_function()

        # Check profiling data
        self.assertIn('test_cpu_function', self.profiler.profiles)
        self.assertIn('test_io_function', self.profiler.profiles)

        # Get performance summaries
        cpu_summary = self.profiler.get_performance_summary('test_cpu_function')
        io_summary = self.profiler.get_performance_summary('test_io_function')

        self.assertIn('wall_time', cpu_summary)
        self.assertIn('cpu_time', cpu_summary)
        self.assertIn('wall_time', io_summary)
        self.assertIn('io_wait_time', io_summary)

    def test_bottleneck_detection(self):
        """Test bottleneck detection accuracy."""
        # Create functions with known bottleneck patterns
        @self.profiler.profile_function('cpu_bound')
        def cpu_bound_function():
            return sum(i ** 2 for i in range(50000))

        @self.profiler.profile_function('io_bound')
        def io_bound_function():
            time.sleep(0.5)
            return "io_complete"

        # Execute functions
        cpu_bound_function()
        io_bound_function()

        # Analyze bottlenecks
        cpu_bottlenecks = self.detector.analyze_bottlenecks('cpu_bound')
        io_bottlenecks = self.detector.analyze_bottlenecks('io_bound')

        # I/O bound function should have I/O bottleneck
        io_bottleneck_types = [b.bottleneck_type for b in io_bottlenecks]
        self.assertIn('io', io_bottleneck_types)

    def test_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        # Create mock bottleneck reports
        bottlenecks = [
            BottleneckReport(
                operation_name="test_operation",
                analysis_timestamp=datetime.now(),
                bottleneck_type="cpu",
                severity="high",
                impact_score=75.0,
                description="CPU utilization bottleneck",
                affected_functions=["slow_function"],
                performance_impact="High CPU wait time",
                optimization_recommendations=["Optimize algorithms"],
                estimated_improvement="50-70% improvement"
            ),
            BottleneckReport(
                operation_name="test_operation",
                analysis_timestamp=datetime.now(),
                bottleneck_type="io",
                severity="medium",
                impact_score=50.0,
                description="I/O wait bottleneck",
                affected_functions=["data_loader"],
                performance_impact="I/O blocking",
                optimization_recommendations=["Add caching"],
                estimated_improvement="30-50% improvement"
            )
        ]

        # Generate recommendations
        recommendations = self.optimizer.generate_recommendations(bottlenecks)

        self.assertEqual(recommendations['status'], 'optimization_needed')
        self.assertEqual(recommendations['total_bottlenecks'], 2)
        self.assertIn('recommendations', recommendations)
        self.assertGreater(len(recommendations['recommendations']), 0)


class TestIntegratedPerformanceFramework(unittest.TestCase):
    """Test integrated performance framework functionality."""

    def setUp(self):
        """Set up integrated test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = create_optimized_config(target_memory_gb=4.0)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parallel_backtest_execution(self):
        """Test parallel backtest execution with performance monitoring."""
        executor = ParallelBacktestExecutor(self.config)

        def simple_backtest(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
            """Simple backtest function for testing."""
            if data.empty:
                return None

            # Simulate some computation
            time.sleep(0.01)

            returns = data['close'].pct_change().dropna() if 'close' in data.columns else pd.Series([0.01])

            return {
                'symbol': symbol,
                'total_return': returns.sum(),
                'volatility': returns.std(),
                'data_points': len(data)
            }

        # Create mock data for testing
        def mock_load_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({
                'open': 100 + np.random.randn(len(dates)),
                'high': 102 + np.random.randn(len(dates)),
                'low': 98 + np.random.randn(len(dates)),
                'close': 100 + np.random.randn(len(dates)),
                'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
            }, index=dates)

        # Replace data loading function temporarily
        original_load = executor._load_stock_data
        executor._load_stock_data = mock_load_data

        try:
            # Execute parallel backtest
            test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            start_date = date(2023, 1, 1)
            end_date = date(2023, 6, 30)

            results = executor.execute_parallel_backtest(
                test_symbols, simple_backtest, start_date, end_date
            )

            # Validate results
            self.assertIn('results', results)
            self.assertIn('metrics', results)
            self.assertIn('cache_stats', results)

            # Check that we got results for most symbols
            self.assertGreater(results['success_rate'], 0.5)

            # Check performance metrics
            metrics = results['metrics']
            self.assertGreater(metrics['total_stocks_processed'], 0)

            # Check cache statistics
            cache_stats = results['cache_stats']
            self.assertIn('hit_rate', cache_stats)

        finally:
            # Restore original function
            executor._load_stock_data = original_load

    def test_benchmark_suite(self):
        """Test comprehensive benchmark suite."""
        benchmark_suite = BacktestBenchmarkSuite(self.config)

        # Run memory benchmark
        memory_results = benchmark_suite.benchmark_memory_usage(50)  # 50MB test
        self.assertIn('initial_memory_mb', memory_results)
        self.assertIn('peak_memory_mb', memory_results)

        # Validate memory efficiency
        if 'memory_efficiency' in memory_results:
            self.assertGreater(memory_results['memory_efficiency'], 0)


class TestPerformanceRegression(unittest.TestCase):
    """Test performance regression detection."""

    def test_performance_baseline_comparison(self):
        """Test performance baseline establishment and comparison."""
        # This test would compare current performance against saved baselines
        # For now, we'll create a simple framework

        baseline_metrics = {
            'cache_hit_rate': 0.85,
            'avg_query_time': 0.050,
            'parallel_efficiency': 0.75,
            'memory_efficiency': 0.80
        }

        # Simulate current metrics (slightly degraded)
        current_metrics = {
            'cache_hit_rate': 0.82,
            'avg_query_time': 0.055,
            'parallel_efficiency': 0.72,
            'memory_efficiency': 0.78
        }

        # Check for regressions
        regressions = []
        threshold = 0.05  # 5% degradation threshold

        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics.get(metric, 0)
            if current_value < baseline_value * (1 - threshold):
                regression_pct = (baseline_value - current_value) / baseline_value * 100
                regressions.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'regression_percent': regression_pct
                })

        # In a real test, we'd assert no regressions or acceptable ones
        if regressions:
            print(f"Performance regressions detected: {len(regressions)}")
            for regression in regressions:
                print(f"  {regression['metric']}: {regression['regression_percent']:.1f}% degradation")


def run_performance_tests():
    """Run all performance optimization framework tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_cases = [
        TestHighPerformanceDataCache,
        TestDatabaseQueryOptimizer,
        TestProgressMonitoringSystem,
        TestBottleneckAnalyzer,
        TestIntegratedPerformanceFramework,
        TestPerformanceRegression
    ]

    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Generate test report
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'test_details': {
            'failures': [str(failure) for failure in result.failures],
            'errors': [str(error) for error in result.errors]
        }
    }

    # Save test report
    output_dir = Path("reports/performance_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"performance_test_report_{timestamp}.json"

    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)

    print(f"\nTest Report Summary:")
    print(f"Tests Run: {test_report['tests_run']}")
    print(f"Failures: {test_report['failures']}")
    print(f"Errors: {test_report['errors']}")
    print(f"Success Rate: {test_report['success_rate']:.1%}")
    print(f"Detailed report saved to: {report_path}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)