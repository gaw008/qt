#!/usr/bin/env python3
"""
Unit Tests for Risk Calculation Services

Tests for the refactored risk management components including:
- TailRiskCalculator
- RegimeDetectionService
- DrawdownManager
- CorrelationAnalyzer
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

from risk_calculation_services import (
    TailRiskCalculator,
    RegimeDetectionService,
    DrawdownManager,
    CorrelationAnalyzer,
    MarketRegime,
    TailRiskMetrics,
    DrawdownTier
)


class TestTailRiskCalculator(unittest.TestCase):
    """Test cases for TailRiskCalculator service."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = TailRiskCalculator()
        # Generate test return data
        np.random.seed(42)
        self.normal_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        self.crisis_returns = np.concatenate([
            np.random.normal(0.001, 0.02, 200),
            np.random.normal(-0.05, 0.05, 52)  # Crisis period
        ])

    def test_expected_shortfall_calculation(self):
        """Test Expected Shortfall calculation."""
        # Test with normal returns
        es_975 = self.calculator.calculate_expected_shortfall(self.normal_returns, 0.975)
        self.assertIsInstance(es_975, float)
        self.assertGreaterEqual(es_975, 0)  # ES should be positive

        # Test with crisis returns
        es_crisis = self.calculator.calculate_expected_shortfall(self.crisis_returns, 0.975)
        self.assertGreater(es_crisis, es_975)  # Crisis should have higher ES

        # Test different confidence levels
        es_99 = self.calculator.calculate_expected_shortfall(self.normal_returns, 0.99)
        self.assertGreater(es_99, es_975)  # Higher confidence should give higher ES

    def test_expected_shortfall_edge_cases(self):
        """Test ES calculation edge cases."""
        # Empty array
        es_empty = self.calculator.calculate_expected_shortfall(np.array([]))
        self.assertEqual(es_empty, 0.0)

        # Single value
        es_single = self.calculator.calculate_expected_shortfall(np.array([0.01]))
        self.assertGreaterEqual(es_single, 0)

        # All positive returns
        es_positive = self.calculator.calculate_expected_shortfall(np.array([0.01, 0.02, 0.03]))
        self.assertGreaterEqual(es_positive, 0)

    def test_tail_dependence_calculation(self):
        """Test tail dependence calculation."""
        market_returns = np.random.normal(0.001, 0.015, 252)

        # Correlated portfolio
        correlated_portfolio = 0.7 * market_returns + 0.3 * np.random.normal(0.001, 0.01, 252)

        tail_dep = self.calculator.calculate_tail_dependence(
            correlated_portfolio, market_returns, 0.95
        )

        self.assertIsInstance(tail_dep, float)
        self.assertGreaterEqual(tail_dep, -1.0)
        self.assertLessEqual(tail_dep, 1.0)

    def test_tail_dependence_edge_cases(self):
        """Test tail dependence edge cases."""
        # Different length arrays
        tail_dep = self.calculator.calculate_tail_dependence(
            np.array([1, 2, 3]), np.array([1, 2]), 0.95
        )
        self.assertEqual(tail_dep, 0.0)

        # Insufficient data
        tail_dep = self.calculator.calculate_tail_dependence(
            np.array([1, 2]), np.array([1, 2]), 0.95
        )
        self.assertEqual(tail_dep, 0.0)

    def test_comprehensive_tail_metrics(self):
        """Test comprehensive tail metrics calculation."""
        metrics = self.calculator.calculate_comprehensive_tail_metrics(self.normal_returns)

        self.assertIsInstance(metrics, TailRiskMetrics)
        self.assertGreaterEqual(metrics.es_97_5, 0)
        self.assertGreaterEqual(metrics.es_99, 0)
        self.assertGreaterEqual(metrics.max_drawdown, 0)
        self.assertIsInstance(metrics.skewness, float)
        self.assertIsInstance(metrics.kurtosis, float)
        self.assertIsInstance(metrics.calmar_ratio, float)

    def test_comprehensive_tail_metrics_insufficient_data(self):
        """Test tail metrics with insufficient data."""
        short_returns = np.array([0.01, 0.02])
        metrics = self.calculator.calculate_comprehensive_tail_metrics(short_returns)

        # Should return default metrics
        self.assertEqual(metrics.es_97_5, 0.0)
        self.assertEqual(metrics.es_99, 0.0)


class TestRegimeDetectionService(unittest.TestCase):
    """Test cases for RegimeDetectionService."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = RegimeDetectionService()

    def test_regime_detection_normal(self):
        """Test normal market regime detection."""
        market_data = {
            'vix': 15.0,
            'market_correlation': 0.3,
            'momentum_strength': 0.2
        }

        regime = self.detector.detect_market_regime(market_data)
        self.assertEqual(regime, MarketRegime.NORMAL)

    def test_regime_detection_volatile(self):
        """Test volatile market regime detection."""
        market_data = {
            'vix': 25.0,
            'market_correlation': 0.6,
            'momentum_strength': 0.2
        }

        regime = self.detector.detect_market_regime(market_data)
        self.assertEqual(regime, MarketRegime.VOLATILE)

    def test_regime_detection_trending(self):
        """Test trending market regime detection."""
        market_data = {
            'vix': 18.0,
            'market_correlation': 0.4,
            'momentum_strength': 0.7
        }

        regime = self.detector.detect_market_regime(market_data)
        self.assertEqual(regime, MarketRegime.TRENDING)

    def test_regime_detection_crisis(self):
        """Test crisis market regime detection."""
        market_data = {
            'vix': 35.0,
            'market_correlation': 0.8,
            'momentum_strength': 0.1
        }

        regime = self.detector.detect_market_regime(market_data)
        self.assertEqual(regime, MarketRegime.CRISIS)

    def test_regime_multipliers(self):
        """Test regime multiplier retrieval."""
        # Test all regimes
        for regime in MarketRegime:
            multipliers = self.detector.get_regime_multipliers(regime)
            self.assertIsInstance(multipliers, dict)
            self.assertIn('var', multipliers)
            self.assertIn('position', multipliers)
            self.assertIn('es', multipliers)

    def test_regime_adjustments(self):
        """Test regime-based limit adjustments."""
        base_limits = {
            'max_portfolio_var': 0.20,
            'max_single_position': 0.10,
            'es_97_5_limit': 0.05
        }

        # Test crisis adjustment
        adjusted = self.detector.apply_regime_adjustments(base_limits, MarketRegime.CRISIS)

        # Crisis should reduce limits
        self.assertLess(adjusted['max_portfolio_var'], base_limits['max_portfolio_var'])
        self.assertLess(adjusted['max_single_position'], base_limits['max_single_position'])
        self.assertGreater(adjusted['es_97_5_limit'], base_limits['es_97_5_limit'])

    def test_missing_market_data(self):
        """Test regime detection with missing data."""
        # Empty market data
        regime = self.detector.detect_market_regime({})
        self.assertEqual(regime, MarketRegime.NORMAL)  # Should default to normal

        # Partial market data
        regime = self.detector.detect_market_regime({'vix': 25.0})
        self.assertIsInstance(regime, MarketRegime)


class TestDrawdownManager(unittest.TestCase):
    """Test cases for DrawdownManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = DrawdownManager()

    def test_tier_detection_normal(self):
        """Test normal conditions (no tier activation)."""
        tier, actions, severity = self.manager.check_drawdown_tier(0.05)  # 5% drawdown

        self.assertEqual(tier, 0)
        self.assertEqual(actions, [])
        self.assertEqual(severity, "NORMAL")

    def test_tier_1_activation(self):
        """Test Tier 1 activation."""
        tier, actions, severity = self.manager.check_drawdown_tier(0.10)  # 10% drawdown

        self.assertEqual(tier, 1)
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)
        self.assertEqual(severity, "TIER_1")

    def test_tier_2_activation(self):
        """Test Tier 2 activation."""
        tier, actions, severity = self.manager.check_drawdown_tier(0.13)  # 13% drawdown

        self.assertEqual(tier, 2)
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)
        self.assertEqual(severity, "TIER_2")

    def test_tier_3_activation(self):
        """Test Tier 3 activation."""
        tier, actions, severity = self.manager.check_drawdown_tier(0.20)  # 20% drawdown

        self.assertEqual(tier, 3)
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)
        self.assertEqual(severity, "TIER_3")

    def test_tier_configuration(self):
        """Test tier configuration retrieval."""
        config = self.manager.get_tier_configuration()

        self.assertIsInstance(config, list)
        self.assertEqual(len(config), 3)  # Should have 3 tiers

        for tier_config in config:
            self.assertIn('tier', tier_config)
            self.assertIn('threshold', tier_config)
            self.assertIn('actions', tier_config)
            self.assertIn('severity', tier_config)

    def test_tier_threshold_update(self):
        """Test updating tier thresholds."""
        # Valid update
        result = self.manager.update_tier_threshold(1, 0.10)
        self.assertTrue(result)

        # Check update took effect
        tier, _, _ = self.manager.check_drawdown_tier(0.10)
        self.assertEqual(tier, 1)

        # Invalid tier
        result = self.manager.update_tier_threshold(5, 0.10)
        self.assertFalse(result)


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for CorrelationAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer()

        # Generate test data
        np.random.seed(42)
        self.returns_data = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.025, 100),
            'MSFT': np.random.normal(0.001, 0.018, 100)
        }

        # Create correlated data
        base_returns = np.random.normal(0.001, 0.02, 100)
        self.correlated_data = {
            'AAPL': base_returns + np.random.normal(0, 0.005, 100),
            'GOOGL': base_returns + np.random.normal(0, 0.005, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)  # Independent
        }

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation."""
        corr_matrix = self.analyzer.calculate_portfolio_correlation_matrix(self.returns_data)

        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (3, 3))

        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix.values), [1.0, 1.0, 1.0])

    def test_correlation_matrix_empty_data(self):
        """Test correlation matrix with empty data."""
        corr_matrix = self.analyzer.calculate_portfolio_correlation_matrix({})

        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertTrue(corr_matrix.empty)

    def test_high_correlation_detection(self):
        """Test high correlation pair detection."""
        corr_matrix = self.analyzer.calculate_portfolio_correlation_matrix(self.correlated_data)
        high_corr_pairs = self.analyzer.identify_high_correlation_pairs(corr_matrix, 0.5)

        self.assertIsInstance(high_corr_pairs, list)
        # Should find high correlation between AAPL and GOOGL
        self.assertGreater(len(high_corr_pairs), 0)

    def test_diversification_ratio(self):
        """Test diversification ratio calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        volatilities = np.array([0.20, 0.25, 0.18])
        correlation_matrix = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ])

        div_ratio = self.analyzer.calculate_diversification_ratio(
            weights, volatilities, correlation_matrix
        )

        self.assertIsInstance(div_ratio, float)
        self.assertGreaterEqual(div_ratio, 1.0)  # Should be >= 1 for diversified portfolio

    def test_diversification_ratio_edge_cases(self):
        """Test diversification ratio edge cases."""
        # Single asset
        weights = np.array([1.0])
        volatilities = np.array([0.20])
        correlation_matrix = np.array([[1.0]])

        div_ratio = self.analyzer.calculate_diversification_ratio(
            weights, volatilities, correlation_matrix
        )

        self.assertAlmostEqual(div_ratio, 1.0, places=6)

    def test_concentration_risk_analysis(self):
        """Test concentration risk analysis."""
        weights = np.array([0.5, 0.3, 0.2])
        sectors = ['Tech', 'Tech', 'Finance']

        metrics = self.analyzer.analyze_concentration_risk(weights, sectors)

        self.assertIsInstance(metrics, dict)
        self.assertIn('herfindahl_index', metrics)
        self.assertIn('max_weight', metrics)
        self.assertIn('effective_positions', metrics)
        self.assertIn('sector_concentration', metrics)

        # Check values
        self.assertEqual(metrics['max_weight'], 0.5)
        self.assertIn('Tech', metrics['sector_concentration'])

    def test_concentration_risk_no_sectors(self):
        """Test concentration risk without sector data."""
        weights = np.array([0.4, 0.3, 0.3])

        metrics = self.analyzer.analyze_concentration_risk(weights)

        self.assertIsInstance(metrics, dict)
        self.assertIn('herfindahl_index', metrics)
        self.assertNotIn('sector_concentration', metrics)


class TestIntegrationTests(unittest.TestCase):
    """Integration tests for risk services working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.tail_calculator = TailRiskCalculator()
        self.regime_detector = RegimeDetectionService()
        self.drawdown_manager = DrawdownManager()
        self.correlation_analyzer = CorrelationAnalyzer()

    def test_complete_risk_assessment_workflow(self):
        """Test complete risk assessment workflow."""
        # Generate test data
        np.random.seed(42)
        portfolio_returns = np.random.normal(0.001, 0.02, 252)
        market_returns = np.random.normal(0.001, 0.015, 252)

        market_data = {
            'vix': 25.0,
            'market_correlation': 0.6,
            'momentum_strength': 0.3
        }

        # Step 1: Calculate tail metrics
        tail_metrics = self.tail_calculator.calculate_comprehensive_tail_metrics(portfolio_returns)
        self.assertIsInstance(tail_metrics, TailRiskMetrics)

        # Step 2: Detect regime
        regime = self.regime_detector.detect_market_regime(market_data)
        self.assertEqual(regime, MarketRegime.VOLATILE)

        # Step 3: Check drawdown tier
        tier, actions, severity = self.drawdown_manager.check_drawdown_tier(tail_metrics.max_drawdown)
        self.assertIsInstance(tier, int)

        # Step 4: Calculate tail dependence
        tail_dependence = self.tail_calculator.calculate_tail_dependence(
            portfolio_returns, market_returns
        )
        self.assertIsInstance(tail_dependence, float)

    def test_performance_under_stress(self):
        """Test performance with large datasets."""
        # Large dataset
        large_returns = np.random.normal(0.001, 0.02, 10000)

        # Should complete in reasonable time
        start_time = datetime.now()

        tail_metrics = self.tail_calculator.calculate_comprehensive_tail_metrics(large_returns)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within 5 seconds
        self.assertLess(duration, 5.0)
        self.assertIsInstance(tail_metrics, TailRiskMetrics)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestTailRiskCalculator))
    test_suite.addTest(unittest.makeSuite(TestRegimeDetectionService))
    test_suite.addTest(unittest.makeSuite(TestDrawdownManager))
    test_suite.addTest(unittest.makeSuite(TestCorrelationAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestIntegrationTests))

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