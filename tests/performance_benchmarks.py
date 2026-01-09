#!/usr/bin/env python3
"""
Performance Benchmarks for Refactored Quantitative Trading System

This module provides comprehensive performance benchmarking for:
- Risk calculation services
- Scoring engine components
- Memory usage patterns
- System throughput under load

Target Performance Metrics:
- Risk assessment: <2 seconds for 1000 stocks
- Scoring: <5 seconds for 4000 stocks
- Memory usage: <2GB for full system operation
"""

import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import tracemalloc
import sys
import os
from dataclasses import dataclass
from contextlib import contextmanager

# Add the bot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quant_system_full', 'bot'))

from risk_calculation_services import (
    TailRiskCalculator,
    RegimeDetectionService,
    DrawdownManager,
    CorrelationAnalyzer
)

from risk_assessment_orchestrator import RiskAssessmentOrchestrator

from scoring_services import (
    FactorCalculationService,
    FactorNormalizationService,
    CorrelationAnalysisService,
    WeightOptimizationService
)

from scoring_orchestrator import ScoringOrchestrator


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    execution_time: float
    memory_peak_mb: float
    memory_current_mb: float
    throughput_per_second: float
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = None


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmark suite for the refactored system.
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }

    @contextmanager
    def _measure_performance(self, test_name: str, item_count: int = 1):
        """Context manager for measuring performance metrics."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()

        # Record initial state
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
            success = True
            error_message = ""
        except Exception as e:
            success = False
            error_message = str(e)

        # Record final state
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Get peak memory from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024

        tracemalloc.stop()

        # Calculate metrics
        execution_time = end_time - start_time
        throughput = item_count / execution_time if execution_time > 0 else 0

        # Store result
        result = BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            memory_peak_mb=peak_memory_mb,
            memory_current_mb=end_memory,
            throughput_per_second=throughput,
            success=success,
            error_message=error_message,
            metadata={
                'item_count': item_count,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory
            }
        )

        self.results.append(result)

    def _generate_test_data(self, num_symbols: int, num_days: int) -> Dict[str, pd.DataFrame]:
        """Generate test market data for benchmarking."""
        test_data = {}
        dates = pd.date_range('2023-01-01', periods=num_days, freq='D')

        np.random.seed(42)  # For reproducible results

        for i in range(num_symbols):
            symbol = f'STOCK{i:04d}'

            # Generate realistic price data
            initial_price = 50 + np.random.uniform(0, 200)
            price_drift = np.random.normal(0, 0.001, num_days)
            price_volatility = np.random.uniform(0.01, 0.03)

            prices = [initial_price]
            for j in range(1, num_days):
                price_change = np.random.normal(price_drift[j], price_volatility)
                new_price = prices[-1] * (1 + price_change)
                prices.append(max(new_price, 0.01))  # Prevent negative prices

            # Generate OHLCV data
            test_data[symbol] = pd.DataFrame({
                'date': dates,
                'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
                'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
                'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
                'close': prices,
                'volume': np.random.randint(100000, 10000000, num_days)
            })

        return test_data

    def _generate_return_data(self, num_symbols: int, num_days: int) -> Dict[str, np.ndarray]:
        """Generate return data for correlation analysis."""
        np.random.seed(42)

        return_data = {}
        for i in range(num_symbols):
            symbol = f'STOCK{i:04d}'
            returns = np.random.normal(0.001, 0.02, num_days)
            return_data[symbol] = returns

        return return_data

    # Risk Calculation Benchmarks

    def benchmark_tail_risk_calculation(self):
        """Benchmark tail risk calculation performance."""
        calculator = TailRiskCalculator()

        # Test with different dataset sizes
        test_sizes = [100, 500, 1000, 2500]

        for size in test_sizes:
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, size)

            with self._measure_performance(f"tail_risk_calculation_{size}_returns", size):
                metrics = calculator.calculate_comprehensive_tail_metrics(returns)
                assert metrics.es_97_5 >= 0

    def benchmark_regime_detection(self):
        """Benchmark market regime detection performance."""
        detector = RegimeDetectionService()

        # Generate market data scenarios
        market_scenarios = []
        for i in range(1000):
            scenario = {
                'vix': np.random.uniform(10, 50),
                'market_correlation': np.random.uniform(0, 1),
                'momentum_strength': np.random.uniform(-1, 1)
            }
            market_scenarios.append(scenario)

        with self._measure_performance("regime_detection_1000_scenarios", 1000):
            for scenario in market_scenarios:
                regime = detector.detect_market_regime(scenario)
                assert regime is not None

    def benchmark_correlation_analysis(self):
        """Benchmark portfolio correlation analysis."""
        analyzer = CorrelationAnalyzer()

        # Test with different portfolio sizes
        portfolio_sizes = [10, 50, 100, 500]

        for size in portfolio_sizes:
            return_data = self._generate_return_data(size, 252)

            with self._measure_performance(f"correlation_analysis_{size}_assets", size):
                corr_matrix = analyzer.calculate_portfolio_correlation_matrix(return_data)
                high_corr_pairs = analyzer.identify_high_correlation_pairs(corr_matrix, 0.7)
                assert isinstance(corr_matrix, pd.DataFrame)

    def benchmark_risk_assessment_orchestrator(self):
        """Benchmark complete risk assessment workflow."""
        orchestrator = RiskAssessmentOrchestrator()

        # Test with realistic portfolio sizes
        portfolio_sizes = [10, 50, 100, 200]

        for size in portfolio_sizes:
            # Create mock portfolio
            portfolio = {
                'total_value': 10000000,
                'positions': []
            }

            for i in range(size):
                position = {
                    'symbol': f'STOCK{i:04d}',
                    'market_value': 10000000 / size,
                    'sector': f'Sector{i % 10}'
                }
                portfolio['positions'].append(position)

            # Create market data
            market_data = {
                'vix': 25.0,
                'market_correlation': 0.6,
                'momentum_strength': 0.3
            }

            # Generate return history
            np.random.seed(42)
            returns_history = np.random.normal(0.001, 0.02, 252)

            with self._measure_performance(f"risk_assessment_{size}_positions", size):
                assessment = orchestrator.assess_portfolio_risk(
                    portfolio, market_data, returns_history
                )
                assert assessment.overall_risk_score >= 0

    # Scoring Engine Benchmarks

    def benchmark_factor_calculation(self):
        """Benchmark factor calculation performance."""
        service = FactorCalculationService()

        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 4000]

        for size in dataset_sizes:
            test_data = self._generate_test_data(size, 50)

            with self._measure_performance(f"factor_calculation_{size}_symbols", size):
                factor_df = service.calculate_all_factors(test_data)
                assert len(factor_df) <= size  # Some may fail but that's ok

    def benchmark_factor_normalization(self):
        """Benchmark factor normalization performance."""
        from scoring_services import NormalizationConfig

        config = NormalizationConfig(method="robust")
        service = FactorNormalizationService(config)

        # Test with different factor dataset sizes
        sizes = [100, 500, 1000, 5000]

        for size in sizes:
            # Generate factor data
            np.random.seed(42)
            factor_data = pd.DataFrame({
                'symbol': [f'STOCK{i:04d}' for i in range(size)],
                'valuation_score': np.random.normal(1.0, 0.5, size),
                'momentum_score': np.random.normal(0.0, 1.0, size),
                'technical_score': np.random.normal(0.5, 0.8, size),
                'volume_score': np.random.normal(0.2, 0.6, size),
                'market_sentiment_score': np.random.normal(0.0, 0.7, size)
            })

            with self._measure_performance(f"factor_normalization_{size}_symbols", size):
                normalized = service.normalize_factors(factor_data)
                assert len(normalized) == size

    def benchmark_scoring_orchestrator(self):
        """Benchmark complete scoring orchestration."""
        orchestrator = ScoringOrchestrator()

        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]  # Up to 2000 for memory constraints

        for size in dataset_sizes:
            test_data = self._generate_test_data(size, 30)  # Reduced days for performance

            with self._measure_performance(f"scoring_orchestrator_{size}_symbols", size):
                result = orchestrator.calculate_composite_scores(test_data)
                assert isinstance(result.scores, pd.DataFrame)

    # System Integration Benchmarks

    def benchmark_complete_system_workflow(self):
        """Benchmark complete system workflow with both risk and scoring."""
        risk_orchestrator = RiskAssessmentOrchestrator()
        scoring_orchestrator = ScoringOrchestrator()

        # Test with realistic system load
        num_symbols = 500
        test_data = self._generate_test_data(num_symbols, 100)

        with self._measure_performance("complete_system_workflow", num_symbols):
            # Step 1: Calculate scores
            scoring_result = scoring_orchestrator.calculate_composite_scores(test_data)

            # Step 2: Create portfolio from top scores
            top_stocks = scoring_result.scores.nlargest(20, 'composite_score')

            portfolio = {
                'total_value': 1000000,
                'positions': []
            }

            for _, stock in top_stocks.iterrows():
                position = {
                    'symbol': stock['symbol'],
                    'market_value': 50000,  # Equal weight
                    'sector': f'Sector{hash(stock["symbol"]) % 10}'
                }
                portfolio['positions'].append(position)

            # Step 3: Assess risk
            market_data = {'vix': 20.0, 'market_correlation': 0.5, 'momentum_strength': 0.2}
            returns_history = np.random.normal(0.001, 0.02, 252)

            risk_assessment = risk_orchestrator.assess_portfolio_risk(
                portfolio, market_data, returns_history
            )

            assert risk_assessment.overall_risk_score >= 0

    def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency under sustained load."""
        orchestrator = ScoringOrchestrator()

        # Run multiple iterations to test memory leaks
        num_iterations = 10
        symbols_per_iteration = 100

        with self._measure_performance("memory_efficiency_test", num_iterations):
            for i in range(num_iterations):
                test_data = self._generate_test_data(symbols_per_iteration, 30)
                result = orchestrator.calculate_composite_scores(test_data)

                # Force garbage collection
                import gc
                gc.collect()

    # Performance Analysis Methods

    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("Starting Performance Benchmark Suite")
        print("=" * 50)
        print(f"System Info: {self.system_info}")
        print("=" * 50)

        benchmark_methods = [
            self.benchmark_tail_risk_calculation,
            self.benchmark_regime_detection,
            self.benchmark_correlation_analysis,
            self.benchmark_risk_assessment_orchestrator,
            self.benchmark_factor_calculation,
            self.benchmark_factor_normalization,
            self.benchmark_scoring_orchestrator,
            self.benchmark_complete_system_workflow,
            self.benchmark_memory_efficiency
        ]

        for i, benchmark_method in enumerate(benchmark_methods, 1):
            print(f"\n[{i}/{len(benchmark_methods)}] Running {benchmark_method.__name__}...")

            try:
                benchmark_method()
                print(f"✓ {benchmark_method.__name__} completed")
            except Exception as e:
                print(f"✗ {benchmark_method.__name__} failed: {e}")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights."""
        if not self.results:
            return {"error": "No benchmark results available"}

        analysis = {
            'summary': {
                'total_tests': len(self.results),
                'successful_tests': len([r for r in self.results if r.success]),
                'failed_tests': len([r for r in self.results if not r.success])
            },
            'performance_metrics': {},
            'memory_metrics': {},
            'throughput_metrics': {},
            'target_compliance': {}
        }

        # Performance analysis
        execution_times = [r.execution_time for r in self.results if r.success]
        if execution_times:
            analysis['performance_metrics'] = {
                'avg_execution_time': np.mean(execution_times),
                'max_execution_time': np.max(execution_times),
                'min_execution_time': np.min(execution_times),
                'std_execution_time': np.std(execution_times)
            }

        # Memory analysis
        memory_peaks = [r.memory_peak_mb for r in self.results if r.success]
        if memory_peaks:
            analysis['memory_metrics'] = {
                'avg_memory_peak_mb': np.mean(memory_peaks),
                'max_memory_peak_mb': np.max(memory_peaks),
                'total_memory_available_gb': self.system_info['memory_total_gb']
            }

        # Throughput analysis
        throughputs = [r.throughput_per_second for r in self.results if r.success and r.throughput_per_second > 0]
        if throughputs:
            analysis['throughput_metrics'] = {
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'min_throughput': np.min(throughputs)
            }

        # Target compliance
        analysis['target_compliance'] = self._check_target_compliance()

        return analysis

    def _check_target_compliance(self) -> Dict[str, Any]:
        """Check compliance with performance targets."""
        compliance = {
            'risk_assessment_target': {'target': '2s for 1000 stocks', 'status': 'unknown'},
            'scoring_target': {'target': '5s for 4000 stocks', 'status': 'unknown'},
            'memory_target': {'target': '2GB system', 'status': 'unknown'}
        }

        # Check risk assessment targets
        risk_results = [r for r in self.results if 'risk_assessment' in r.test_name and r.success]
        for result in risk_results:
            if result.metadata and result.metadata.get('item_count', 0) >= 100:  # Scaled check
                # Scale to 1000 stocks
                scaled_time = result.execution_time * (1000 / result.metadata['item_count'])
                compliance['risk_assessment_target']['status'] = 'PASS' if scaled_time <= 2.0 else 'FAIL'
                compliance['risk_assessment_target']['actual'] = f'{scaled_time:.2f}s estimated for 1000 stocks'

        # Check scoring targets
        scoring_results = [r for r in self.results if 'scoring_orchestrator' in r.test_name and r.success]
        for result in scoring_results:
            if result.metadata and result.metadata.get('item_count', 0) >= 500:  # Scaled check
                # Scale to 4000 stocks
                scaled_time = result.execution_time * (4000 / result.metadata['item_count'])
                compliance['scoring_target']['status'] = 'PASS' if scaled_time <= 5.0 else 'FAIL'
                compliance['scoring_target']['actual'] = f'{scaled_time:.2f}s estimated for 4000 stocks'

        # Check memory targets
        max_memory = max([r.memory_peak_mb for r in self.results if r.success], default=0)
        max_memory_gb = max_memory / 1024
        compliance['memory_target']['status'] = 'PASS' if max_memory_gb <= 2.0 else 'FAIL'
        compliance['memory_target']['actual'] = f'{max_memory_gb:.2f}GB peak usage'

        return compliance

    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        analysis = self.analyze_results()

        report = []
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # System information
        report.append("SYSTEM INFORMATION")
        report.append("-" * 20)
        for key, value in self.system_info.items():
            report.append(f"{key}: {value}")
        report.append("")

        # Summary
        summary = analysis['summary']
        report.append("TEST SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Successful: {summary['successful_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append(f"Success Rate: {(summary['successful_tests']/summary['total_tests']*100):.1f}%")
        report.append("")

        # Target compliance
        if 'target_compliance' in analysis:
            report.append("TARGET COMPLIANCE")
            report.append("-" * 20)
            for target, details in analysis['target_compliance'].items():
                status_symbol = "✓" if details['status'] == 'PASS' else "✗" if details['status'] == 'FAIL' else "?"
                report.append(f"{status_symbol} {target}: {details['target']}")
                if 'actual' in details:
                    report.append(f"    Actual: {details['actual']}")
            report.append("")

        # Performance metrics
        if 'performance_metrics' in analysis and analysis['performance_metrics']:
            metrics = analysis['performance_metrics']
            report.append("PERFORMANCE METRICS")
            report.append("-" * 20)
            report.append(f"Average Execution Time: {metrics['avg_execution_time']:.3f}s")
            report.append(f"Maximum Execution Time: {metrics['max_execution_time']:.3f}s")
            report.append(f"Minimum Execution Time: {metrics['min_execution_time']:.3f}s")
            report.append("")

        # Memory metrics
        if 'memory_metrics' in analysis and analysis['memory_metrics']:
            metrics = analysis['memory_metrics']
            report.append("MEMORY METRICS")
            report.append("-" * 20)
            report.append(f"Average Memory Peak: {metrics['avg_memory_peak_mb']:.1f} MB")
            report.append(f"Maximum Memory Peak: {metrics['max_memory_peak_mb']:.1f} MB")
            report.append(f"Total Available Memory: {metrics['total_memory_available_gb']:.1f} GB")
            report.append("")

        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 20)
        for result in self.results:
            status = "✓" if result.success else "✗"
            report.append(f"{status} {result.test_name}")
            report.append(f"    Execution Time: {result.execution_time:.3f}s")
            report.append(f"    Memory Peak: {result.memory_peak_mb:.1f} MB")
            if result.throughput_per_second > 0:
                report.append(f"    Throughput: {result.throughput_per_second:.1f} items/sec")
            if not result.success:
                report.append(f"    Error: {result.error_message}")
            report.append("")

        return "\n".join(report)

    def save_report(self, filepath: str):
        """Save performance report to file."""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Performance report saved to: {filepath}")


def main():
    """Main benchmark execution function."""
    print("Investment-Grade Quantitative Trading System")
    print("Performance Benchmark Suite")
    print("=" * 50)

    # Create benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite()

    # Run all benchmarks
    benchmark_suite.run_all_benchmarks()

    # Generate and save report
    report_filename = f"performance_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    benchmark_suite.save_report(report_filename)

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    analysis = benchmark_suite.analyze_results()

    # Print key metrics
    if 'target_compliance' in analysis:
        print("\nTarget Compliance:")
        for target, details in analysis['target_compliance'].items():
            status = details['status']
            symbol = "✓" if status == 'PASS' else "✗" if status == 'FAIL' else "?"
            print(f"  {symbol} {target}: {details['target']}")

    print(f"\nTotal Tests: {analysis['summary']['total_tests']}")
    print(f"Success Rate: {(analysis['summary']['successful_tests']/analysis['summary']['total_tests']*100):.1f}%")

    if 'performance_metrics' in analysis and analysis['performance_metrics']:
        print(f"Average Execution Time: {analysis['performance_metrics']['avg_execution_time']:.3f}s")

    if 'memory_metrics' in analysis and analysis['memory_metrics']:
        print(f"Peak Memory Usage: {analysis['memory_metrics']['max_memory_peak_mb']:.1f} MB")

    print(f"\nDetailed report saved to: {report_filename}")


if __name__ == '__main__':
    main()