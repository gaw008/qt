#!/usr/bin/env python3
"""
Unit Tests for Scoring Services

Tests for the refactored scoring engine components including:
- FactorCalculationService
- FactorNormalizationService
- CorrelationAnalysisService
- WeightOptimizationService
- ScoringOrchestrator
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys
import os

# Add the bot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quant_system_full', 'bot'))

from scoring_services import (
    FactorCalculationService,
    FactorNormalizationService,
    CorrelationAnalysisService,
    WeightOptimizationService,
    NormalizationConfig,
    FactorScore,
    FactorCalculationStrategy,
    FallbackFactorStrategy
)

from scoring_orchestrator import (
    ScoringOrchestrator,
    FactorWeights,
    ScoringResult
)


class MockFactorStrategy(FactorCalculationStrategy):
    """Mock factor strategy for testing."""

    def __init__(self, factor_name: str, base_score: float = 1.0):
        self.factor_name = factor_name
        self.base_score = base_score

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        if data.empty:
            return 0.0
        # Simple calculation based on price change
        if 'close' in data.columns and len(data) > 1:
            return_rate = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
            return float(self.base_score * (1 + return_rate))
        return self.base_score

    def get_factor_name(self) -> str:
        return self.factor_name


class TestFactorCalculationService(unittest.TestCase):
    """Test cases for FactorCalculationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = FactorCalculationService()

        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        self.test_data = {
            'AAPL': pd.DataFrame({
                'date': dates,
                'open': 150 + np.random.randn(100) * 2,
                'high': 155 + np.random.randn(100) * 2,
                'low': 145 + np.random.randn(100) * 2,
                'close': 150 + np.random.randn(100) * 2,
                'volume': 1000000 + np.random.randint(-100000, 100000, 100)
            }),
            'GOOGL': pd.DataFrame({
                'date': dates,
                'open': 2500 + np.random.randn(100) * 50,
                'high': 2550 + np.random.randn(100) * 50,
                'low': 2450 + np.random.randn(100) * 50,
                'close': 2500 + np.random.randn(100) * 50,
                'volume': 500000 + np.random.randint(-50000, 50000, 100)
            })
        }

        # Register mock strategies
        self.service.strategies = {
            'valuation': MockFactorStrategy('valuation', 1.2),
            'momentum': MockFactorStrategy('momentum', 0.8),
            'technical': MockFactorStrategy('technical', 1.0)
        }

    def test_calculate_single_factor(self):
        """Test single factor calculation."""
        factor_score = self.service.calculate_factor(
            'valuation', 'AAPL', self.test_data['AAPL']
        )

        self.assertIsInstance(factor_score, FactorScore)
        self.assertEqual(factor_score.symbol, 'AAPL')
        self.assertEqual(factor_score.factor_name, 'valuation')
        self.assertIsInstance(factor_score.raw_score, float)

    def test_calculate_unknown_factor(self):
        """Test calculation with unknown factor."""
        factor_score = self.service.calculate_factor(
            'unknown_factor', 'AAPL', self.test_data['AAPL']
        )

        self.assertEqual(factor_score.raw_score, 0.0)
        self.assertEqual(factor_score.normalized_score, 0.0)

    def test_calculate_all_factors(self):
        """Test calculating all factors for all symbols."""
        result_df = self.service.calculate_all_factors(self.test_data)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        self.assertIn('symbol', result_df.columns)
        self.assertIn('valuation_score', result_df.columns)
        self.assertIn('momentum_score', result_df.columns)
        self.assertIn('technical_score', result_df.columns)

        # Check symbols are present
        symbols = result_df['symbol'].tolist()
        self.assertIn('AAPL', symbols)
        self.assertIn('GOOGL', symbols)

    def test_calculate_with_empty_data(self):
        """Test calculation with empty data."""
        empty_data = {}
        result_df = self.service.calculate_all_factors(empty_data)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 0)

    def test_calculate_with_invalid_data(self):
        """Test calculation with invalid data."""
        invalid_data = {
            'INVALID': pd.DataFrame()  # Empty DataFrame
        }
        result_df = self.service.calculate_all_factors(invalid_data)

        # Should handle gracefully
        self.assertIsInstance(result_df, pd.DataFrame)

    def test_register_custom_strategy(self):
        """Test registering custom factor strategy."""
        custom_strategy = MockFactorStrategy('custom', 2.0)
        self.service.register_strategy('custom', custom_strategy)

        # Test the custom strategy works
        factor_score = self.service.calculate_factor(
            'custom', 'AAPL', self.test_data['AAPL']
        )

        self.assertEqual(factor_score.factor_name, 'custom')
        self.assertNotEqual(factor_score.raw_score, 0.0)


class TestFactorNormalizationService(unittest.TestCase):
    """Test cases for FactorNormalizationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = NormalizationConfig(method="robust")
        self.service = FactorNormalizationService(self.config)

        # Create test factor data
        np.random.seed(42)
        self.factor_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
            'valuation_score': [1.2, 0.8, 1.5, 0.5, 2.0],
            'momentum_score': [0.3, 0.7, -0.2, 0.9, 1.1],
            'technical_score': [-0.5, 0.3, 0.8, -0.1, 0.4]
        })

    def test_robust_normalization(self):
        """Test robust normalization method."""
        normalized = self.service.normalize_factors(self.factor_data)

        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertEqual(len(normalized), len(self.factor_data))

        # Check that normalization was applied
        for col in ['valuation_score', 'momentum_score', 'technical_score']:
            original_std = self.factor_data[col].std()
            normalized_mad = (normalized[col] - normalized[col].median()).abs().median()
            # Normalized data should have different scale
            self.assertNotEqual(original_std, normalized_mad)

    def test_standard_normalization(self):
        """Test standard normalization method."""
        config = NormalizationConfig(method="standard")
        service = FactorNormalizationService(config)

        normalized = service.normalize_factors(self.factor_data)

        # Check z-score properties (approximately)
        for col in ['valuation_score', 'momentum_score', 'technical_score']:
            mean = normalized[col].mean()
            std = normalized[col].std()
            self.assertAlmostEqual(mean, 0.0, places=10)
            self.assertAlmostEqual(std, 1.0, places=10)

    def test_winsorize_normalization(self):
        """Test winsorize normalization method."""
        config = NormalizationConfig(method="winsorize", winsorize_percentile=0.1)
        service = FactorNormalizationService(config)

        # Add outliers
        outlier_data = self.factor_data.copy()
        outlier_data.loc[0, 'valuation_score'] = 100.0  # Extreme outlier

        normalized = service.normalize_factors(outlier_data)

        # Outlier should be handled
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertLess(abs(normalized['valuation_score'].max()), 10)  # Should be bounded

    def test_sector_neutrality(self):
        """Test sector neutrality application."""
        config = NormalizationConfig(enable_sector_neutrality=True)
        service = FactorNormalizationService(config)

        sector_mapping = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology',
            'MSFT': 'Technology',
            'AMZN': 'Consumer',
            'TSLA': 'Consumer'
        }

        neutral_scores = service.apply_sector_neutrality(self.factor_data, sector_mapping)

        self.assertIsInstance(neutral_scores, pd.DataFrame)
        self.assertEqual(len(neutral_scores), len(self.factor_data))

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        normalized = self.service.normalize_factors(empty_df)

        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertTrue(normalized.empty)

    def test_non_numeric_columns(self):
        """Test handling of non-numeric columns."""
        mixed_data = self.factor_data.copy()
        mixed_data['text_column'] = ['A', 'B', 'C', 'D', 'E']

        normalized = self.service.normalize_factors(mixed_data)

        # Text column should be preserved
        self.assertIn('text_column', normalized.columns)
        self.assertEqual(normalized['text_column'].tolist(), ['A', 'B', 'C', 'D', 'E'])


class TestCorrelationAnalysisService(unittest.TestCase):
    """Test cases for CorrelationAnalysisService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = CorrelationAnalysisService(high_correlation_threshold=0.7)

        # Create correlated factor data
        np.random.seed(42)
        base_factor = np.random.randn(100)

        self.factor_data = pd.DataFrame({
            'symbol': [f'STOCK{i}' for i in range(100)],
            'factor_a': base_factor + np.random.randn(100) * 0.1,  # Highly correlated
            'factor_b': base_factor + np.random.randn(100) * 0.1,  # Highly correlated
            'factor_c': np.random.randn(100),  # Independent
            'factor_d': np.random.randn(100)   # Independent
        })

    def test_correlation_matrix_calculation(self):
        """Test factor correlation matrix calculation."""
        corr_matrix = self.service.calculate_factor_correlations(self.factor_data)

        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (4, 4))

        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix.values), [1.0, 1.0, 1.0, 1.0])

        # Check high correlation between factor_a and factor_b
        corr_ab = abs(corr_matrix.loc['factor_a', 'factor_b'])
        self.assertGreater(corr_ab, 0.5)  # Should be highly correlated

    def test_redundant_factor_detection(self):
        """Test detection of redundant factors."""
        corr_matrix = self.service.calculate_factor_correlations(self.factor_data)
        redundant_pairs = self.service.detect_redundant_factors(corr_matrix)

        self.assertIsInstance(redundant_pairs, list)
        # Should find high correlation between factor_a and factor_b
        self.assertGreater(len(redundant_pairs), 0)

        # Check format of redundant pairs
        for factor1, factor2, correlation in redundant_pairs:
            self.assertIsInstance(factor1, str)
            self.assertIsInstance(factor2, str)
            self.assertIsInstance(correlation, float)
            self.assertGreaterEqual(correlation, self.service.high_correlation_threshold)

    def test_factor_loadings_calculation(self):
        """Test factor loading statistics."""
        loadings = self.service.calculate_factor_loadings(self.factor_data)

        self.assertIsInstance(loadings, dict)
        self.assertIn('factor_a', loadings)

        # Check loading statistics structure
        for factor, stats in loadings.items():
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('skewness', stats)
            self.assertIn('kurtosis', stats)
            self.assertIn('non_zero_ratio', stats)

    def test_empty_data_handling(self):
        """Test handling of empty factor data."""
        empty_df = pd.DataFrame()

        corr_matrix = self.service.calculate_factor_correlations(empty_df)
        self.assertTrue(corr_matrix.empty)

        redundant_pairs = self.service.detect_redundant_factors(corr_matrix)
        self.assertEqual(len(redundant_pairs), 0)

        loadings = self.service.calculate_factor_loadings(empty_df)
        self.assertEqual(len(loadings), 0)


class TestWeightOptimizationService(unittest.TestCase):
    """Test cases for WeightOptimizationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = WeightOptimizationService(
            min_weight=0.05,
            max_weight=0.50,
            redundancy_penalty=0.1
        )

        self.base_weights = {
            'valuation': 0.25,
            'momentum': 0.25,
            'technical': 0.25,
            'volume': 0.25
        }

        # Create correlation data
        self.correlation_matrix = pd.DataFrame({
            'valuation_score': [1.0, 0.1, 0.2, 0.0],
            'momentum_score': [0.1, 1.0, 0.8, 0.1],  # High correlation with technical
            'technical_score': [0.2, 0.8, 1.0, 0.1],  # High correlation with momentum
            'volume_score': [0.0, 0.1, 0.1, 1.0]
        }, index=['valuation_score', 'momentum_score', 'technical_score', 'volume_score'])

        self.redundant_pairs = [('momentum_score', 'technical_score', 0.8)]

    def test_weight_optimization_basic(self):
        """Test basic weight optimization."""
        optimized = self.service.optimize_weights(
            self.base_weights,
            self.correlation_matrix,
            self.redundant_pairs
        )

        self.assertIsInstance(optimized, dict)

        # Weights should sum to 1
        total_weight = sum(optimized.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)

        # All weights should be within bounds
        for weight in optimized.values():
            self.assertGreaterEqual(weight, self.service.min_weight)
            self.assertLessEqual(weight, self.service.max_weight)

    def test_correlation_penalty_application(self):
        """Test correlation penalty reduces weights of correlated factors."""
        optimized = self.service.optimize_weights(
            self.base_weights,
            self.correlation_matrix,
            self.redundant_pairs
        )

        # Momentum and technical should have reduced weights due to correlation
        original_momentum = self.base_weights['momentum']
        original_technical = self.base_weights['technical']

        optimized_momentum = optimized.get('momentum', 0)
        optimized_technical = optimized.get('technical', 0)

        # At least one should be reduced (may not be both due to renormalization)
        total_original = original_momentum + original_technical
        total_optimized = optimized_momentum + optimized_technical

        self.assertLess(total_optimized, total_original)

    def test_performance_adjustment(self):
        """Test performance-based weight adjustment."""
        performance_data = {
            'valuation': 1.5,  # Good performance
            'momentum': 0.5,   # Poor performance
            'technical': 1.0,  # Average performance
            'volume': 1.2      # Good performance
        }

        optimized = self.service.optimize_weights(
            self.base_weights,
            self.correlation_matrix,
            self.redundant_pairs,
            performance_data
        )

        # Valuation should get higher weight due to good performance
        # Momentum should get lower weight due to poor performance
        self.assertGreater(optimized.get('valuation', 0), optimized.get('momentum', 0))

    def test_weight_constraints_enforcement(self):
        """Test that weight constraints are properly enforced."""
        # Create extreme weights
        extreme_weights = {
            'valuation': 0.01,  # Below minimum
            'momentum': 0.99,   # Above maximum
            'technical': 0.0,
            'volume': 0.0
        }

        optimized = self.service.optimize_weights(
            extreme_weights,
            pd.DataFrame(),
            []
        )

        # All weights should be within bounds
        for weight in optimized.values():
            self.assertGreaterEqual(weight, self.service.min_weight)
            self.assertLessEqual(weight, self.service.max_weight)

    def test_weight_key_finding(self):
        """Test finding weight keys for factor names."""
        # Test direct match
        key = self.service._find_weight_key('valuation', self.base_weights)
        self.assertEqual(key, 'valuation')

        # Test with score suffix
        key = self.service._find_weight_key('valuation_score', self.base_weights)
        self.assertEqual(key, 'valuation')

        # Test with weight suffix
        weight_dict = {'valuation_weight': 0.25}
        key = self.service._find_weight_key('valuation', weight_dict)
        self.assertEqual(key, 'valuation_weight')

        # Test non-existent
        key = self.service._find_weight_key('nonexistent', self.base_weights)
        self.assertIsNone(key)


class TestScoringOrchestrator(unittest.TestCase):
    """Test cases for ScoringOrchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.factor_weights = FactorWeights(
            valuation=0.3, volume=0.1, momentum=0.2, technical=0.2, market_sentiment=0.2
        )
        self.orchestrator = ScoringOrchestrator(self.factor_weights)

        # Create test market data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)

        self.test_data = {}
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        for symbol in symbols:
            self.test_data[symbol] = pd.DataFrame({
                'date': dates,
                'open': 100 + np.random.randn(50) * 2,
                'high': 105 + np.random.randn(50) * 2,
                'low': 95 + np.random.randn(50) * 2,
                'close': 100 + np.random.randn(50) * 2,
                'volume': 1000000 + np.random.randint(-100000, 100000, 50)
            })

        # Mock the factor calculation service
        self.orchestrator.factor_calculator.strategies = {
            'valuation': MockFactorStrategy('valuation', 1.0),
            'volume': MockFactorStrategy('volume', 0.5),
            'momentum': MockFactorStrategy('momentum', 0.8),
            'technical': MockFactorStrategy('technical', 1.2),
            'market_sentiment': MockFactorStrategy('market_sentiment', 0.6)
        }

    def test_composite_score_calculation(self):
        """Test complete composite score calculation."""
        result = self.orchestrator.calculate_composite_scores(self.test_data)

        self.assertIsInstance(result, ScoringResult)
        self.assertIsInstance(result.scores, pd.DataFrame)
        self.assertGreater(len(result.scores), 0)

        # Check required columns
        self.assertIn('symbol', result.scores.columns)
        self.assertIn('composite_score', result.scores.columns)
        self.assertIn('rank', result.scores.columns)
        self.assertIn('percentile', result.scores.columns)

        # Check factor contributions
        self.assertIsInstance(result.factor_contributions, pd.DataFrame)
        self.assertIn('symbol', result.factor_contributions.columns)

        # Check weights were used
        self.assertIsInstance(result.weights_used, dict)
        self.assertGreater(len(result.weights_used), 0)

    def test_empty_data_handling(self):
        """Test handling of empty input data."""
        result = self.orchestrator.calculate_composite_scores({})

        self.assertIsInstance(result, ScoringResult)
        self.assertTrue(result.scores.empty)
        self.assertTrue(result.factor_contributions.empty)

    def test_custom_weights(self):
        """Test calculation with custom weights."""
        custom_weights = {
            'valuation': 0.5,
            'volume': 0.1,
            'momentum': 0.1,
            'technical': 0.1,
            'market_sentiment': 0.2
        }

        result = self.orchestrator.calculate_composite_scores(
            self.test_data, custom_weights=custom_weights
        )

        # Should use custom weights
        self.assertEqual(result.weights_used['valuation'], 0.5)

    def test_trading_signal_generation(self):
        """Test trading signal generation."""
        result = self.orchestrator.calculate_composite_scores(self.test_data)
        signals = self.orchestrator.generate_trading_signals(
            result, buy_threshold=0.6, sell_threshold=0.4, max_positions=2
        )

        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('signal_strength', signals.columns)

        # Check signal values are valid
        unique_signals = signals['signal'].unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])

    def test_score_explanation(self):
        """Test score explanation functionality."""
        result = self.orchestrator.calculate_composite_scores(self.test_data)
        explanation = self.orchestrator.explain_scores(result, top_n=2)

        self.assertIsInstance(explanation, dict)
        self.assertIn('top_stocks', explanation)
        self.assertIn('bottom_stocks', explanation)
        self.assertIn('factor_weights', explanation)
        self.assertIn('performance_metrics', explanation)

        # Check top stocks format
        if explanation['top_stocks']:
            top_stock = explanation['top_stocks'][0]
            self.assertIn('symbol', top_stock)
            self.assertIn('composite_score', top_stock)

    def test_configuration_save_load(self):
        """Test configuration save and load."""
        config_path = 'test_config.json'

        try:
            # Save configuration
            success = self.orchestrator.save_configuration(config_path)
            self.assertTrue(success)

            # Load configuration
            loaded_orchestrator = ScoringOrchestrator.load_configuration(config_path)
            self.assertIsInstance(loaded_orchestrator, ScoringOrchestrator)

            # Check weights are preserved
            self.assertAlmostEqual(
                loaded_orchestrator.factor_weights.valuation,
                self.factor_weights.valuation,
                places=6
            )

        finally:
            # Clean up
            if os.path.exists(config_path):
                os.remove(config_path)

    def test_history_tracking(self):
        """Test scoring history tracking."""
        # Initial state
        self.assertEqual(len(self.orchestrator.scoring_history), 0)

        # Calculate scores
        result = self.orchestrator.calculate_composite_scores(self.test_data)

        # History should be updated
        self.assertEqual(len(self.orchestrator.scoring_history), 1)

        # Calculate again
        result2 = self.orchestrator.calculate_composite_scores(self.test_data)

        # History should grow
        self.assertEqual(len(self.orchestrator.scoring_history), 2)


class TestPerformanceTests(unittest.TestCase):
    """Performance and stress tests for scoring services."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = ScoringOrchestrator()

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        symbols = [f'STOCK{i:04d}' for i in range(100)]
        dates = pd.date_range('2023-01-01', periods=252, freq='D')

        large_dataset = {}
        np.random.seed(42)

        for symbol in symbols:
            large_dataset[symbol] = pd.DataFrame({
                'date': dates,
                'open': 100 + np.random.randn(252) * 5,
                'high': 105 + np.random.randn(252) * 5,
                'low': 95 + np.random.randn(252) * 5,
                'close': 100 + np.random.randn(252) * 5,
                'volume': 1000000 + np.random.randint(-100000, 100000, 252)
            })

        # Mock strategies for performance test
        self.orchestrator.factor_calculator.strategies = {
            'valuation': MockFactorStrategy('valuation'),
            'volume': MockFactorStrategy('volume'),
            'momentum': MockFactorStrategy('momentum'),
            'technical': MockFactorStrategy('technical'),
            'market_sentiment': MockFactorStrategy('market_sentiment')
        }

        # Measure execution time
        start_time = datetime.now()

        result = self.orchestrator.calculate_composite_scores(large_dataset)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(duration, 30.0)  # 30 seconds for 100 stocks
        self.assertEqual(len(result.scores), 100)

    def test_memory_efficiency(self):
        """Test memory efficiency with repeated calculations."""
        # Small dataset for repeated calculations
        test_data = {
            'TEST': pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [100, 101, 102],
                'volume': [1000000, 1100000, 1200000]
            })
        }

        # Mock strategies
        self.orchestrator.factor_calculator.strategies = {
            'valuation': MockFactorStrategy('valuation'),
            'momentum': MockFactorStrategy('momentum')
        }

        # Run multiple calculations
        for i in range(10):
            result = self.orchestrator.calculate_composite_scores(test_data)
            self.assertIsInstance(result, ScoringResult)

        # History should be managed (not grow indefinitely)
        self.assertLessEqual(len(self.orchestrator.scoring_history), 10)


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFactorCalculationService))
    test_suite.addTest(unittest.makeSuite(TestFactorNormalizationService))
    test_suite.addTest(unittest.makeSuite(TestCorrelationAnalysisService))
    test_suite.addTest(unittest.makeSuite(TestWeightOptimizationService))
    test_suite.addTest(unittest.makeSuite(TestScoringOrchestrator))
    test_suite.addTest(unittest.makeSuite(TestPerformanceTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)