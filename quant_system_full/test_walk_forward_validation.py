#!/usr/bin/env python3
"""
Comprehensive Test Suite for Walk-Forward Validation Framework

This test suite validates the Walk-Forward Validation Framework with:
- Edge case handling and robustness testing
- Statistical significance validation
- Performance degradation detection
- Crisis period analysis
- Quality assurance validation
- Integration testing with existing components
- Benchmark comparison accuracy

Test Coverage:
- Statistical Testing Engine validation
- Walk-Forward Window Generation
- Multi-phase validation consistency
- Risk-based validation criteria
- Overfitting detection mechanisms
- Multiple testing correction accuracy
- Real-world scenario simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Tuple
import tempfile
import shutil

# Import validation framework components
from bot.walk_forward_validator import (
    WalkForwardValidator, WalkForwardConfig, WalkForwardResults,
    ValidationPhase, WindowType, StatisticalTest, StatisticalTestingEngine
)
from bot.validation_integration_adapter import (
    ValidationIntegrationAdapter, ValidationPipeline, IntegratedValidationResults,
    validate_strategy_comprehensive
)

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')


class TestStatisticalTestingEngine(unittest.TestCase):
    """Test the statistical testing engine"""

    def setUp(self):
        self.config = WalkForwardConfig()
        self.engine = StatisticalTestingEngine(self.config)
        np.random.seed(42)

    def test_t_test_significance(self):
        """Test t-test statistical significance testing"""
        # Test with significantly positive returns
        positive_returns = np.random.normal(0.001, 0.01, 1000)  # 0.1% daily mean
        result = self.engine.t_test_significance(positive_returns)

        self.assertIn('test_type', result)
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIsInstance(result['significant'], bool)

        # Test with zero-mean returns
        zero_returns = np.random.normal(0, 0.01, 100)
        result = self.engine.t_test_significance(zero_returns)
        self.assertIsInstance(result['p_value'], float)
        self.assertGreaterEqual(result['p_value'], 0.0)
        self.assertLessEqual(result['p_value'], 1.0)

        # Test with benchmark comparison
        benchmark_returns = np.random.normal(0.0005, 0.01, 1000)
        result = self.engine.t_test_significance(positive_returns, benchmark_returns)
        self.assertEqual(result['test_type'], 'paired_t_test')

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation"""
        returns = np.random.normal(0.001, 0.02, 500)
        result = self.engine.bootstrap_confidence_interval(returns, n_bootstrap=1000)

        self.assertIn('original_statistic', result)
        self.assertIn('confidence_interval', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)

        # Check confidence interval structure
        ci = result['confidence_interval']
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])  # Lower bound <= Upper bound

        # Test with empty data
        result = self.engine.bootstrap_confidence_interval(np.array([]))
        self.assertTrue(np.isnan(result['confidence_interval'][0]))

    def test_reality_check_test(self):
        """Test White's Reality Check for multiple strategies"""
        # Create multiple strategy returns
        strategy_returns = {}
        benchmark = np.random.normal(0.0002, 0.015, 1000)

        # Strategy 1: Slightly better than benchmark
        strategy_returns['strategy_1'] = benchmark + np.random.normal(0.0001, 0.005, 1000)

        # Strategy 2: Similar to benchmark
        strategy_returns['strategy_2'] = benchmark + np.random.normal(0, 0.008, 1000)

        # Strategy 3: Worse than benchmark
        strategy_returns['strategy_3'] = benchmark + np.random.normal(-0.0001, 0.01, 1000)

        result = self.engine.reality_check_test(strategy_returns, benchmark)

        self.assertIn('test_type', result)
        self.assertIn('best_strategy', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIn(result['best_strategy'], strategy_returns.keys())

    def test_superior_predictive_ability_test(self):
        """Test Hansen's Superior Predictive Ability test"""
        strategy_returns = np.random.normal(0.0008, 0.02, 500)
        benchmark_returns = np.random.normal(0.0005, 0.018, 500)

        result = self.engine.superior_predictive_ability_test(strategy_returns, benchmark_returns)

        self.assertIn('test_type', result)
        self.assertIn('test_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertEqual(result['test_type'], 'spa_test')

        # Test with mismatched lengths
        short_benchmark = benchmark_returns[:100]
        result = self.engine.superior_predictive_ability_test(strategy_returns, short_benchmark)
        self.assertEqual(result['p_value'], 1.0)
        self.assertFalse(result['significant'])

    def test_multiple_testing_correction(self):
        """Test multiple testing correction methods"""
        # Create p-values with some significant and some not
        p_values = [0.001, 0.03, 0.08, 0.15, 0.4, 0.7, 0.9]

        # Test Bonferroni correction
        result = self.engine.multiple_testing_correction(p_values, "bonferroni")
        self.assertEqual(result['method'], 'bonferroni')
        self.assertEqual(len(result['rejected']), len(p_values))
        self.assertLessEqual(result['n_rejected'], len(p_values))

        # Test FDR BH correction
        result = self.engine.multiple_testing_correction(p_values, "fdr_bh")
        self.assertEqual(result['method'], 'fdr_bh')
        self.assertIsInstance(result['family_wise_error_rate'], float)

        # Test FDR BY correction
        result = self.engine.multiple_testing_correction(p_values, "fdr_by")
        self.assertEqual(result['method'], 'fdr_by')

        # Test with invalid method
        with self.assertRaises(ValueError):
            self.engine.multiple_testing_correction(p_values, "invalid_method")


class TestWalkForwardValidator(unittest.TestCase):
    """Test the main walk-forward validator"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = WalkForwardConfig(
            min_train_months=12,
            test_window_months=3,
            step_months=3,
            bootstrap_samples=100,  # Reduced for faster testing
            results_directory=self.temp_dir,
            save_detailed_results=False,
            export_diagnostics=False
        )
        self.validator = WalkForwardValidator(self.config)
        np.random.seed(42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_data(self, years: int = 5) -> pd.DataFrame:
        """Create realistic test data with regime changes"""
        dates = pd.date_range('2018-01-01', periods=years*252, freq='D')

        # Generate returns with regime changes
        returns = np.random.randn(len(dates)) * 0.015
        returns[:252] += 0.0003  # Bull market first year
        returns[252:504] -= 0.0005  # Bear market second year
        returns[504:] += 0.0002  # Recovery

        # Generate price data
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)

        return data

    def create_test_strategy(self):
        """Create a simple test strategy"""
        def momentum_strategy(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.01):
            if len(data) < lookback + 1:
                return pd.Series(index=data.index, data=0.0)

            returns = data['close'].pct_change()
            momentum = returns.rolling(lookback).mean()
            signals = np.where(momentum > threshold, 1, np.where(momentum < -threshold, -1, 0))
            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        return momentum_strategy

    def test_walk_forward_window_generation(self):
        """Test walk-forward window generation"""
        data = self.create_test_data(3)  # 3 years of data
        phase = ValidationPhase.PHASE_3

        windows = self.validator._generate_walk_forward_windows(data, phase)

        # Check that windows were generated
        self.assertGreater(len(windows), 0)

        # Check window structure
        for window in windows:
            self.assertIn('window_id', window)
            self.assertIn('train_start', window)
            self.assertIn('train_end', window)
            self.assertIn('test_start', window)
            self.assertIn('test_end', window)
            self.assertIn('train_observations', window)
            self.assertIn('test_observations', window)

            # Validate temporal order
            self.assertLess(window['train_start'], window['train_end'])
            self.assertLess(window['train_end'], window['test_start'])
            self.assertLess(window['test_start'], window['test_end'])

    def test_single_window_processing(self):
        """Test processing of a single walk-forward window"""
        data = self.create_test_data(2)
        strategy = self.create_test_strategy()

        # Create a test window
        window = {
            'window_id': 0,
            'train_start': data.index[0],
            'train_end': data.index[200],
            'test_start': data.index[200],
            'test_end': data.index[300],
            'train_observations': 200,
            'test_observations': 100
        }

        result = self.validator._process_single_window(
            window, strategy, data, None, {'lookback': 15, 'threshold': 0.015}, None
        )

        # Check result structure
        self.assertIn('window_id', result)
        self.assertIn('optimal_params', result)
        self.assertIn('in_sample_metrics', result)
        self.assertIn('out_of_sample_metrics', result)
        self.assertIn('returns', result)
        self.assertIn('market_regime', result)

        # Check metrics structure
        oos_metrics = result['out_of_sample_metrics']
        expected_metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        for metric in expected_metrics:
            self.assertIn(metric, oos_metrics)

    def test_phase_validation(self):
        """Test validation of a single phase"""
        data = self.create_test_data(3)
        strategy = self.create_test_strategy()

        # Create benchmark data
        benchmark_data = pd.DataFrame({
            'close': data['close'] * 0.95 + np.random.randn(len(data)) * 0.01
        }, index=data.index)

        phase_result = self.validator._validate_phase(
            ValidationPhase.PHASE_3, strategy, data, benchmark_data,
            {'lookback': 20, 'threshold': 0.02}, None
        )

        # Check phase result structure
        self.assertEqual(phase_result.phase, ValidationPhase.PHASE_3)
        self.assertGreater(phase_result.successful_windows, 0)
        self.assertIsInstance(phase_result.sharpe_ratio, float)
        self.assertIsInstance(phase_result.max_drawdown, float)
        self.assertIsInstance(phase_result.significance_tests, dict)

    def test_complete_validation(self):
        """Test complete walk-forward validation process"""
        data = self.create_test_data(4)
        strategy = self.create_test_strategy()

        # Create benchmark
        benchmark_data = pd.DataFrame({
            'close': data['close'] * 0.9
        }, index=data.index)

        # Configure with single phase for faster testing
        self.config.phases = {
            ValidationPhase.PHASE_3: ("2018-01-01", "2021-12-31")
        }

        results = self.validator.validate_strategy(
            strategy_func=strategy,
            data=data,
            benchmark_data=benchmark_data,
            strategy_params={'lookback': 20, 'threshold': 0.02}
        )

        # Check results structure
        self.assertIsInstance(results, WalkForwardResults)
        self.assertEqual(len(results.phase_results), 1)
        self.assertIsInstance(results.validation_passed, bool)
        self.assertIsInstance(results.overall_significance, dict)

    def test_edge_case_handling(self):
        """Test edge case handling"""
        # Test with insufficient data
        small_data = self.create_test_data(1)  # Only 1 year
        strategy = self.create_test_strategy()

        # Should handle gracefully
        try:
            results = self.validator.validate_strategy(
                strategy_func=strategy,
                data=small_data,
                strategy_params={'lookback': 10}
            )
            # Should complete without error, may have warnings
        except Exception as e:
            self.fail(f"Validation failed with insufficient data: {e}")

        # Test with empty strategy returns
        def empty_strategy(data: pd.DataFrame, **params):
            return pd.Series(index=data.index, data=0.0)

        try:
            results = self.validator.validate_strategy(
                strategy_func=empty_strategy,
                data=self.create_test_data(2)
            )
        except Exception as e:
            self.fail(f"Validation failed with empty strategy: {e}")

    def test_performance_degradation_detection(self):
        """Test performance degradation detection"""
        # Create strategy with intentional overfitting
        def overfitted_strategy(data: pd.DataFrame, **params):
            # Strategy that works well in-sample but poorly out-of-sample
            returns = data['close'].pct_change()

            # Use future information (intentional overfitting)
            future_returns = returns.shift(-1)
            signals = np.where(future_returns > 0.01, 1,
                              np.where(future_returns < -0.01, -1, 0))

            strategy_returns = pd.Series(signals, index=data.index) * returns
            return strategy_returns.fillna(0)

        data = self.create_test_data(3)

        # This should detect overfitting
        results = self.validator.validate_strategy(
            strategy_func=overfitted_strategy,
            data=data
        )

        # Check overfitting assessment
        self.assertIn('overfitting_assessment', results.__dict__)


class TestValidationIntegrationAdapter(unittest.TestCase):
    """Test the validation integration adapter"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = ValidationIntegrationAdapter()
        np.random.seed(42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_pipeline(self) -> ValidationPipeline:
        """Create a test validation pipeline"""
        def test_strategy(data: pd.DataFrame, lookback: int = 15) -> pd.Series:
            if len(data) < lookback + 1:
                return pd.Series(index=data.index, data=0.0)

            returns = data['close'].pct_change()
            momentum = returns.rolling(lookback).mean()
            signals = np.where(momentum > 0.005, 1, np.where(momentum < -0.005, -1, 0))
            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        return ValidationPipeline(
            strategy_name="Test_Strategy",
            strategy_function=test_strategy,
            symbols=["TEST"],
            strategy_parameters={'lookback': 15},
            start_date="2020-01-01",
            end_date="2023-12-31",
            enable_walk_forward=True,
            enable_purged_kfold=True,
            enable_risk_assessment=True,
            results_directory=self.temp_dir,
            generate_comprehensive_report=False,
            export_data=False
        )

    def test_data_preparation(self):
        """Test data preparation functionality"""
        pipeline = self.create_test_pipeline()

        data, benchmark_data = self.adapter._prepare_validation_data(pipeline)

        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(benchmark_data, pd.DataFrame)
        self.assertIn('close', data.columns)
        self.assertGreater(len(data), 0)
        self.assertGreater(len(benchmark_data), 0)

    def test_integrated_validation(self):
        """Test complete integrated validation"""
        pipeline = self.create_test_pipeline()

        results = self.adapter.validate_strategy(pipeline)

        # Check results structure
        self.assertIsInstance(results, IntegratedValidationResults)
        self.assertEqual(results.pipeline_config.strategy_name, "Test_Strategy")
        self.assertIsInstance(results.overall_validation_passed, bool)
        self.assertIsInstance(results.validation_confidence, float)
        self.assertIn(results.recommendation,
                     ['ACCEPT_HIGH_CONFIDENCE', 'ACCEPT_MEDIUM_CONFIDENCE',
                      'ACCEPT_LOW_CONFIDENCE', 'CONDITIONAL_ACCEPT', 'REJECT'])

    def test_validation_consistency_analysis(self):
        """Test validation consistency analysis"""
        pipeline = self.create_test_pipeline()
        results = self.adapter.validate_strategy(pipeline)

        consistency = results.validation_consistency
        self.assertIn('methods_used', consistency)
        self.assertIn('overall_consistency_score', consistency)
        self.assertIsInstance(consistency['overall_consistency_score'], float)
        self.assertGreaterEqual(consistency['overall_consistency_score'], 0.0)
        self.assertLessEqual(consistency['overall_consistency_score'], 1.0)

    def test_convenience_function(self):
        """Test convenience function for validation"""
        def simple_strategy(data: pd.DataFrame, period: int = 10) -> pd.Series:
            if len(data) < period + 1:
                return pd.Series(index=data.index, data=0.0)

            returns = data['close'].pct_change()
            ma = data['close'].rolling(period).mean()
            signals = np.where(data['close'] > ma, 1, -1)
            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        results = validate_strategy_comprehensive(
            strategy_name="Simple_MA_Strategy",
            strategy_function=simple_strategy,
            symbols=["TEST"],
            strategy_params={'period': 10},
            start_date="2020-01-01",
            end_date="2022-12-31"
        )

        self.assertIsInstance(results, IntegratedValidationResults)
        self.assertEqual(results.pipeline_config.strategy_name, "Simple_MA_Strategy")


class TestCrisisPerformanceAnalysis(unittest.TestCase):
    """Test crisis period performance analysis"""

    def setUp(self):
        self.validator = WalkForwardValidator()
        np.random.seed(42)

    def create_crisis_data(self) -> pd.DataFrame:
        """Create data with a simulated crisis period"""
        dates = pd.date_range('2019-01-01', '2022-12-31', freq='D')

        # Normal returns
        returns = np.random.randn(len(dates)) * 0.01

        # Crisis period (simulate COVID-like crash)
        crisis_start = pd.to_datetime('2020-03-01')
        crisis_end = pd.to_datetime('2020-05-01')
        crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)

        # High volatility and negative returns during crisis
        returns[crisis_mask] = np.random.randn(np.sum(crisis_mask)) * 0.05 - 0.01

        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(2000000, 8000000, len(dates))
        }, index=dates)

    def test_crisis_detection_and_analysis(self):
        """Test crisis period detection and performance analysis"""
        data = self.create_crisis_data()

        # Create mock window results with varying volatility
        window_results = []
        for i in range(10):
            # Some windows with high volatility (crisis)
            volatility = 0.05 if i in [3, 4, 5] else 0.02

            window_results.append({
                'window_id': i,
                'out_of_sample_metrics': {
                    'volatility': volatility,
                    'sharpe_ratio': np.random.normal(0.5, 0.3),
                    'max_drawdown': np.random.uniform(-0.2, -0.05)
                }
            })

        crisis_analysis = self.validator._analyze_crisis_performance(window_results, data)

        # Check crisis analysis structure
        self.assertIn('crisis_window_count', crisis_analysis)
        self.assertIn('normal_window_count', crisis_analysis)
        self.assertIn('crisis_ratio', crisis_analysis)

        # Should detect crisis windows
        self.assertGreater(crisis_analysis['crisis_window_count'], 0)


class TestRealWorldScenarios(unittest.TestCase):
    """Test with real-world scenarios and edge cases"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_market_regime_transitions(self):
        """Test validation across different market regimes"""
        # Create data with clear regime transitions
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='D')

        # Different regime periods
        n_days = len(dates)
        returns = np.random.randn(n_days) * 0.01

        # Bull market 2015-2017
        returns[:int(0.3 * n_days)] += 0.0004

        # Volatile period 2018-2019
        returns[int(0.3 * n_days):int(0.55 * n_days)] *= 1.5

        # Crisis 2020
        crisis_start = int(0.55 * n_days)
        crisis_end = int(0.65 * n_days)
        returns[crisis_start:crisis_end] = np.random.randn(crisis_end - crisis_start) * 0.03 - 0.008

        # Recovery 2021-2023
        returns[crisis_end:] += 0.0003

        prices = 100 * np.exp(np.cumsum(returns))
        data = pd.DataFrame({'close': prices}, index=dates)

        # Test strategy that adapts to regimes
        def adaptive_strategy(data: pd.DataFrame, short_ma: int = 10, long_ma: int = 30) -> pd.Series:
            if len(data) < long_ma + 1:
                return pd.Series(index=data.index, data=0.0)

            # Moving average crossover
            short_ma_vals = data['close'].rolling(short_ma).mean()
            long_ma_vals = data['close'].rolling(long_ma).mean()

            signals = np.where(short_ma_vals > long_ma_vals, 1, -1)
            returns = data['close'].pct_change()
            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        # Validate strategy
        config = WalkForwardConfig(
            phases={ValidationPhase.PHASE_3: ("2015-01-01", "2023-12-31")},
            min_train_months=24,
            test_window_months=6,
            step_months=6,
            bootstrap_samples=500,
            results_directory=self.temp_dir
        )

        validator = WalkForwardValidator(config)

        results = validator.validate_strategy(
            strategy_func=adaptive_strategy,
            data=data,
            strategy_params={'short_ma': 10, 'long_ma': 30}
        )

        # Should complete successfully despite regime changes
        self.assertIsInstance(results, WalkForwardResults)
        self.assertGreater(len(results.phase_results), 0)

    def test_low_frequency_data_handling(self):
        """Test handling of low-frequency data (weekly/monthly)"""
        # Create weekly data
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='W')
        returns = np.random.randn(len(dates)) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({'close': prices}, index=dates)

        def weekly_strategy(data: pd.DataFrame, lookback: int = 4) -> pd.Series:
            if len(data) < lookback + 1:
                return pd.Series(index=data.index, data=0.0)

            returns = data['close'].pct_change()
            momentum = returns.rolling(lookback).mean()
            signals = np.where(momentum > 0.002, 1, np.where(momentum < -0.002, -1, 0))
            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        # Adjusted config for weekly data
        config = WalkForwardConfig(
            min_train_months=12,
            test_window_months=3,
            min_observations_per_window=12,  # Reduced for weekly data
            bootstrap_samples=200
        )

        validator = WalkForwardValidator(config)

        # Should handle weekly data appropriately
        results = validator.validate_strategy(
            strategy_func=weekly_strategy,
            data=data,
            strategy_params={'lookback': 4}
        )

        self.assertIsInstance(results, WalkForwardResults)

    def test_parameter_sensitivity_analysis(self):
        """Test parameter sensitivity across validation windows"""
        data_generator = lambda: pd.DataFrame({
            'close': 100 * np.exp(np.cumsum(np.random.randn(1000) * 0.015)),
            'volume': np.random.randint(1000000, 5000000, 1000)
        }, index=pd.date_range('2020-01-01', periods=1000, freq='D'))

        def parameterized_strategy(data: pd.DataFrame, lookback: int = 20, threshold: float = 0.01) -> pd.Series:
            if len(data) < lookback + 1:
                return pd.Series(index=data.index, data=0.0)

            returns = data['close'].pct_change()
            momentum = returns.rolling(lookback).mean()
            signals = np.where(momentum > threshold, 1, np.where(momentum < -threshold, -1, 0))
            strategy_returns = pd.Series(signals, index=data.index).shift(1) * returns
            return strategy_returns.fillna(0)

        # Test different parameter combinations
        param_combinations = [
            {'lookback': 15, 'threshold': 0.008},
            {'lookback': 20, 'threshold': 0.01},
            {'lookback': 25, 'threshold': 0.012}
        ]

        results_list = []
        for params in param_combinations:
            data = data_generator()
            validator = WalkForwardValidator()

            results = validator.validate_strategy(
                strategy_func=parameterized_strategy,
                data=data,
                strategy_params=params
            )
            results_list.append((params, results))

        # All parameter combinations should produce valid results
        for params, results in results_list:
            self.assertIsInstance(results, WalkForwardResults)
            self.assertIsInstance(results.validation_passed, bool)


def run_performance_benchmarks():
    """Run performance benchmarks for the validation framework"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)

    import time

    # Test data sizes
    data_sizes = [252, 504, 1008, 2016]  # 1, 2, 4, 8 years of daily data

    for size in data_sizes:
        print(f"\nTesting with {size} data points ({size/252:.1f} years)...")

        # Generate test data
        dates = pd.date_range('2020-01-01', periods=size, freq='D')
        returns = np.random.randn(size) * 0.015
        data = pd.DataFrame({
            'close': 100 * np.exp(np.cumsum(returns)),
            'volume': np.random.randint(1000000, 5000000, size)
        }, index=dates)

        def test_strategy(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
            if len(data) < lookback + 1:
                return pd.Series(index=data.index, data=0.0)
            returns = data['close'].pct_change()
            momentum = returns.rolling(lookback).mean()
            signals = np.where(momentum > 0.01, 1, np.where(momentum < -0.01, -1, 0))
            return pd.Series(signals, index=data.index).shift(1) * returns

        # Configure for performance testing
        config = WalkForwardConfig(
            min_train_months=6,
            test_window_months=3,
            step_months=3,
            bootstrap_samples=1000,
            save_detailed_results=False,
            export_diagnostics=False
        )

        validator = WalkForwardValidator(config)

        # Time the validation
        start_time = time.time()
        results = validator.validate_strategy(
            strategy_func=test_strategy,
            data=data,
            strategy_params={'lookback': 20}
        )
        end_time = time.time()

        validation_time = end_time - start_time
        windows_count = sum(len(phase_result.window_results)
                           for phase_result in results.phase_results.values())

        print(f"  Validation time: {validation_time:.2f} seconds")
        print(f"  Windows processed: {windows_count}")
        print(f"  Time per window: {validation_time/max(windows_count, 1):.3f} seconds")
        print(f"  Validation passed: {results.validation_passed}")


if __name__ == '__main__':
    print("Walk-Forward Validation Framework - Comprehensive Test Suite")
    print("=" * 80)

    # Run unit tests
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestStatisticalTestingEngine))
    test_suite.addTest(unittest.makeSuite(TestWalkForwardValidator))
    test_suite.addTest(unittest.makeSuite(TestValidationIntegrationAdapter))
    test_suite.addTest(unittest.makeSuite(TestCrisisPerformanceAnalysis))
    test_suite.addTest(unittest.makeSuite(TestRealWorldScenarios))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)

    # Print summary
    print(f"\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100):.1f}%")

    if test_result.failures:
        print(f"\nFAILURES:")
        for test, traceback in test_result.failures:
            print(f"  - {test}: {traceback}")

    if test_result.errors:
        print(f"\nERRORS:")
        for test, traceback in test_result.errors:
            print(f"  - {test}: {traceback}")

    # Run performance benchmarks if tests passed
    if len(test_result.failures) == 0 and len(test_result.errors) == 0:
        run_performance_benchmarks()

    print(f"\nWalk-Forward Validation Framework testing completed!")
    print(f"Framework is {'READY FOR PRODUCTION' if len(test_result.failures) == 0 and len(test_result.errors) == 0 else 'REQUIRES FIXES'}")