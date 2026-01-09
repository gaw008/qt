#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Market Regime Classification System

This module provides extensive testing for all components of the market regime
classification system including HMM models, threshold detection, ML classifiers,
and ensemble methods.

Test Categories:
- Unit tests for individual components
- Integration tests with existing systems
- Performance and accuracy validation
- Crisis period detection validation
- Data handling and error recovery
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Import the regime classifier components
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from market_regime_classifier import (
        MarketRegimeClassifier,
        HiddenMarkovRegimeModel,
        ThresholdRegimeDetector,
        MLRegimeClassifier,
        MarketRegime,
        RegimeIndicators,
        RegimePrediction,
        create_regime_classifier,
        get_regime_for_risk_manager
    )
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import regime classifier: {e}")
    CLASSIFIER_AVAILABLE = False


class TestRegimeIndicators(unittest.TestCase):
    """Test RegimeIndicators dataclass"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

    def test_regime_indicators_creation(self):
        """Test creation of RegimeIndicators"""
        indicators = RegimeIndicators()

        # Test default values
        self.assertEqual(indicators.vix_level, 0.0)
        self.assertEqual(indicators.correlation_level, 0.0)
        self.assertIsInstance(indicators.timestamp, datetime)

    def test_regime_indicators_with_values(self):
        """Test RegimeIndicators with custom values"""
        test_time = datetime.now()
        indicators = RegimeIndicators(
            vix_level=25.5,
            volatility_percentile=75.0,
            correlation_level=0.65,
            timestamp=test_time
        )

        self.assertEqual(indicators.vix_level, 25.5)
        self.assertEqual(indicators.volatility_percentile, 75.0)
        self.assertEqual(indicators.correlation_level, 0.65)
        self.assertEqual(indicators.timestamp, test_time)


class TestThresholdRegimeDetector(unittest.TestCase):
    """Test threshold-based regime detection"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

        self.detector = ThresholdRegimeDetector()

    def test_detector_initialization(self):
        """Test detector initialization with default config"""
        self.assertIsInstance(self.detector.config, dict)
        self.assertIn('vix_normal_max', self.detector.config)
        self.assertEqual(self.detector.config['vix_normal_max'], 20.0)

    def test_detector_custom_config(self):
        """Test detector with custom configuration"""
        custom_config = {
            'vix_normal_max': 18.0,
            'vix_crisis_min': 40.0
        }
        detector = ThresholdRegimeDetector(custom_config)

        self.assertEqual(detector.config['vix_normal_max'], 18.0)
        self.assertEqual(detector.config['vix_crisis_min'], 40.0)

    def test_calculate_indicators_empty_data(self):
        """Test indicator calculation with empty data"""
        empty_data = {}
        indicators = self.detector.calculate_indicators(empty_data)

        self.assertIsInstance(indicators, RegimeIndicators)
        self.assertEqual(indicators.vix_level, 0.0)
        self.assertEqual(indicators.correlation_level, 0.0)

    def test_calculate_indicators_vix_data(self):
        """Test indicator calculation with VIX data"""
        # Create mock VIX data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        vix_data = pd.DataFrame({
            'close': np.random.uniform(15, 35, 30),
            'open': np.random.uniform(15, 35, 30),
            'high': np.random.uniform(20, 40, 30),
            'low': np.random.uniform(10, 30, 30)
        }, index=dates)

        data = {'vix_data': vix_data}
        indicators = self.detector.calculate_indicators(data)

        self.assertGreater(indicators.vix_level, 0)
        self.assertNotEqual(indicators.vix_change, 0)

    def test_calculate_indicators_market_data(self):
        """Test indicator calculation with market data"""
        # Create mock market data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')

        market_data = {}
        for symbol in ['SPY', 'QQQ', 'IWM']:
            prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 50))
            market_data[symbol] = pd.DataFrame({
                'close': prices,
                'open': prices * (1 + np.random.normal(0, 0.01, 50)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.02, 50))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.02, 50))),
                'volume': np.random.randint(1000000, 10000000, 50)
            }, index=dates)

        data = {'market_data': market_data}
        indicators = self.detector.calculate_indicators(data)

        self.assertGreaterEqual(indicators.volatility_percentile, 0)
        # Correlation might be 0 if calculation fails, which is acceptable

    def test_classify_regime_normal(self):
        """Test regime classification for normal conditions"""
        indicators = RegimeIndicators(
            vix_level=18.0,
            volatility_percentile=45.0,
            correlation_level=0.4,
            credit_spread=80.0,
            breadth_deterioration=0.3
        )

        regime, confidence = self.detector.classify_regime(indicators)

        self.assertEqual(regime, MarketRegime.NORMAL)
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_classify_regime_volatile(self):
        """Test regime classification for volatile conditions"""
        indicators = RegimeIndicators(
            vix_level=28.0,
            volatility_percentile=85.0,
            correlation_level=0.65,
            credit_spread=180.0,
            breadth_deterioration=0.6
        )

        regime, confidence = self.detector.classify_regime(indicators)

        self.assertEqual(regime, MarketRegime.VOLATILE)
        self.assertGreater(confidence, 0)

    def test_classify_regime_crisis(self):
        """Test regime classification for crisis conditions"""
        indicators = RegimeIndicators(
            vix_level=45.0,
            volatility_percentile=95.0,
            correlation_level=0.85,
            credit_spread=350.0,
            breadth_deterioration=0.8
        )

        regime, confidence = self.detector.classify_regime(indicators)

        self.assertEqual(regime, MarketRegime.CRISIS)
        self.assertGreater(confidence, 0)

    def test_classify_regime_insufficient_data(self):
        """Test regime classification with insufficient indicators"""
        indicators = RegimeIndicators(
            vix_level=25.0  # Only one indicator
        )

        regime, confidence = self.detector.classify_regime(indicators)

        # Should default to NORMAL with low confidence
        self.assertEqual(regime, MarketRegime.NORMAL)
        self.assertLess(confidence, 0.5)


class TestHiddenMarkovRegimeModel(unittest.TestCase):
    """Test Hidden Markov Model for regime detection"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

        self.hmm_model = HiddenMarkovRegimeModel()

    def test_hmm_initialization(self):
        """Test HMM model initialization"""
        self.assertEqual(self.hmm_model.n_regimes, 3)
        self.assertEqual(self.hmm_model.random_state, 42)
        self.assertIsNone(self.hmm_model.model)

    def test_prepare_features_empty_data(self):
        """Test feature preparation with empty data"""
        empty_data = {}
        features = self.hmm_model.prepare_features(empty_data)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features.shape), 2)  # Should be 2D array

    def test_prepare_features_with_data(self):
        """Test feature preparation with market data"""
        # Create comprehensive test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

        # VIX data
        vix_data = pd.DataFrame({
            'close': 20 + 10 * np.sin(np.arange(100) / 10) + np.random.normal(0, 2, 100)
        }, index=dates)

        # Market data
        market_data = {}
        for symbol in ['SPY', 'QQQ', 'IWM']:
            prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100))
            market_data[symbol] = pd.DataFrame({
                'close': prices
            }, index=dates)

        data = {
            'vix_data': vix_data,
            'market_data': market_data
        }

        features = self.hmm_model.prepare_features(data)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features.shape), 2)
        self.assertGreater(features.shape[0], 0)
        self.assertGreater(features.shape[1], 0)

    def test_fit_model(self):
        """Test HMM model fitting"""
        # Create training data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

        # Simulate regime-switching VIX
        vix_values = []
        current_regime = 0
        regime_duration = 0

        for i in range(252):
            if regime_duration > 30 and np.random.random() < 0.1:
                current_regime = np.random.choice([0, 1, 2])
                regime_duration = 0

            if current_regime == 0:  # Normal
                vix_val = np.random.normal(18, 3)
            elif current_regime == 1:  # Volatile
                vix_val = np.random.normal(28, 5)
            else:  # Crisis
                vix_val = np.random.normal(40, 8)

            vix_values.append(max(5, vix_val))
            regime_duration += 1

        vix_data = pd.DataFrame({'close': vix_values}, index=dates)

        # Market data
        market_data = {}
        for symbol in ['SPY', 'QQQ']:
            returns = np.random.normal(0, 0.02, 252)
            prices = 100 * np.cumprod(1 + returns)
            market_data[symbol] = pd.DataFrame({'close': prices}, index=dates)

        training_data = {
            'vix_data': vix_data,
            'market_data': market_data
        }

        # Fit model
        self.hmm_model.fit(training_data)

        self.assertIsNotNone(self.hmm_model.model)
        self.assertTrue(len(self.hmm_model.feature_names) > 0)

    def test_predict_regime_without_fitting(self):
        """Test prediction without fitted model raises error"""
        test_data = {'vix_data': pd.DataFrame({'close': [20]})}

        with self.assertRaises(ValueError):
            self.hmm_model.predict_regime(test_data)


class TestMLRegimeClassifier(unittest.TestCase):
    """Test Machine Learning regime classifier"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

        self.ml_classifier = MLRegimeClassifier()

    def test_ml_initialization(self):
        """Test ML classifier initialization"""
        self.assertEqual(self.ml_classifier.random_state, 42)
        self.assertFalse(self.ml_classifier.is_fitted)
        self.assertTrue(len(self.ml_classifier.crisis_periods) > 0)

    def test_create_labels(self):
        """Test label creation for training"""
        # Create date range covering known crisis periods
        dates = pd.date_range(start='2008-01-01', end='2010-01-01', freq='D')
        labels = self.ml_classifier.create_labels(dates)

        self.assertEqual(len(labels), len(dates))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2])))

        # Should have some crisis labels (2008 financial crisis)
        self.assertTrue(np.any(labels == 2))

    def test_prepare_features_for_ml(self):
        """Test feature preparation for ML training"""
        # Create realistic training data
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

        # VIX with crisis period simulation
        vix_values = np.random.normal(20, 5, 200)
        vix_values[40:80] = np.random.normal(35, 10, 40)  # Crisis period
        vix_data = pd.DataFrame({'close': vix_values}, index=dates)

        # Market data with multiple symbols
        market_data = {}
        for symbol in ['SPY', 'QQQ', 'IWM', 'XLF']:
            returns = np.random.normal(0, 0.02, 200)
            returns[40:80] *= 2  # Higher volatility during crisis
            prices = 100 * np.cumprod(1 + returns)
            market_data[symbol] = pd.DataFrame({'close': prices}, index=dates)

        data = {
            'vix_data': vix_data,
            'market_data': market_data
        }

        features, feature_names = self.ml_classifier.prepare_features(data)

        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(feature_names, list)
        self.assertEqual(len(features.shape), 2)
        self.assertGreater(len(feature_names), 0)

    def test_fit_ml_model(self):
        """Test ML model fitting"""
        # Create training data with clear regime patterns
        dates = pd.date_range(start='2008-01-01', end='2022-12-31', freq='D')
        n_days = len(dates)

        # Simulate VIX with known crisis periods
        vix_values = np.random.normal(20, 3, n_days)

        # Add crisis spikes during known periods
        for start_str, end_str in self.ml_classifier.crisis_periods:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)

            start_idx = max(0, (start_date - dates[0]).days)
            end_idx = min(n_days, (end_date - dates[0]).days)

            if start_idx < end_idx:
                vix_values[start_idx:end_idx] = np.random.normal(40, 10, end_idx - start_idx)

        vix_data = pd.DataFrame({'close': vix_values}, index=dates)

        # Market data
        market_data = {}
        for symbol in ['SPY', 'QQQ']:
            returns = np.random.normal(0, 0.015, n_days)
            prices = 100 * np.cumprod(1 + returns)
            market_data[symbol] = pd.DataFrame({'close': prices}, index=dates)

        training_data = {
            'vix_data': vix_data,
            'market_data': market_data
        }

        # Fit model
        self.ml_classifier.fit(training_data)

        self.assertTrue(self.ml_classifier.is_fitted)
        self.assertTrue(len(self.ml_classifier.feature_names) > 0)


class TestMarketRegimeClassifier(unittest.TestCase):
    """Test main MarketRegimeClassifier class"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        self.classifier = MarketRegimeClassifier(
            cache_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_classifier_initialization(self):
        """Test classifier initialization"""
        self.assertIsInstance(self.classifier.hmm_model, HiddenMarkovRegimeModel)
        self.assertIsInstance(self.classifier.threshold_detector, ThresholdRegimeDetector)
        self.assertIsInstance(self.classifier.ml_classifier, MLRegimeClassifier)
        self.assertEqual(self.classifier.current_regime, MarketRegime.NORMAL)

    def test_load_market_data_fallback(self):
        """Test market data loading with fallback to dummy data"""
        # Mock the data loading to force fallback
        with patch('market_regime_classifier.HAS_DEPENDENCIES', False):
            data = self.classifier.load_market_data(
                symbols=['SPY', 'QQQ'],
                limit=100,
                include_vix=True
            )

        self.assertIn('market_data', data)
        self.assertIn('vix_data', data)
        self.assertIsInstance(data['market_data'], dict)
        self.assertIsInstance(data['vix_data'], pd.DataFrame)

    def test_dummy_data_generation(self):
        """Test dummy data generation"""
        symbols = ['SPY', 'QQQ', 'IWM']
        dummy_data = self.classifier._generate_dummy_market_data(symbols, 100)

        self.assertEqual(len(dummy_data), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, dummy_data)
            df = dummy_data[symbol]
            self.assertEqual(len(df), 100)
            self.assertIn('close', df.columns)
            self.assertIn('volume', df.columns)

    def test_dummy_vix_generation(self):
        """Test dummy VIX data generation"""
        vix_data = self.classifier._generate_dummy_vix_data(50)

        self.assertEqual(len(vix_data), 50)
        self.assertIn('close', vix_data.columns)

        # VIX should be in reasonable range
        vix_values = vix_data['close']
        self.assertTrue(np.all(vix_values >= 5))
        self.assertTrue(np.all(vix_values <= 80))

    def test_predict_regime_ensemble(self):
        """Test ensemble regime prediction"""
        # Create test data
        test_data = {
            'market_data': self.classifier._generate_dummy_market_data(['SPY', 'QQQ'], 100),
            'vix_data': self.classifier._generate_dummy_vix_data(100)
        }

        # Make prediction
        prediction = self.classifier.predict_regime(test_data, method='ensemble')

        self.assertIsInstance(prediction, RegimePrediction)
        self.assertIn(prediction.regime, [MarketRegime.NORMAL, MarketRegime.VOLATILE, MarketRegime.CRISIS])
        self.assertGreaterEqual(prediction.confidence, 0)
        self.assertLessEqual(prediction.confidence, 1)

        # Probabilities should sum to approximately 1
        prob_sum = (prediction.probability_normal +
                   prediction.probability_volatile +
                   prediction.probability_crisis)
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predict_regime_threshold_method(self):
        """Test threshold-only regime prediction"""
        test_data = {
            'market_data': self.classifier._generate_dummy_market_data(['SPY'], 50),
            'vix_data': self.classifier._generate_dummy_vix_data(50)
        }

        prediction = self.classifier.predict_regime(test_data, method='threshold')

        self.assertIsInstance(prediction, RegimePrediction)
        self.assertEqual(prediction.method, 'threshold')

    def test_regime_transition_recording(self):
        """Test regime transition recording"""
        # Initial regime
        initial_regime = self.classifier.current_regime
        initial_count = len(self.classifier.transition_history)

        # Force a regime change by directly setting it
        self.classifier._record_regime_transition(
            initial_regime,
            MarketRegime.VOLATILE,
            0.8
        )

        # Check transition was recorded
        self.assertEqual(len(self.classifier.transition_history), initial_count + 1)

        transition = self.classifier.transition_history[-1]
        self.assertEqual(transition.from_regime, initial_regime)
        self.assertEqual(transition.to_regime, MarketRegime.VOLATILE)
        self.assertEqual(transition.confidence, 0.8)

    def test_save_and_load_models(self):
        """Test model saving and loading"""
        # Create and fit models with dummy data
        test_data = {
            'market_data': self.classifier._generate_dummy_market_data(['SPY', 'QQQ'], 200),
            'vix_data': self.classifier._generate_dummy_vix_data(200)
        }

        # Fit models
        self.classifier.fit_models(test_data)

        # Save models
        save_success = self.classifier.save_models()
        self.assertTrue(save_success)

        # Create new classifier and load models
        new_classifier = MarketRegimeClassifier(cache_dir=self.temp_dir)
        load_success = new_classifier.load_models()
        self.assertTrue(load_success)

        # Verify models were loaded
        self.assertIsNotNone(new_classifier.hmm_model.model)
        self.assertTrue(new_classifier.ml_classifier.is_fitted)

    def test_get_regime_summary(self):
        """Test regime summary generation"""
        # Add some fake history
        test_data = {
            'market_data': self.classifier._generate_dummy_market_data(['SPY'], 50),
            'vix_data': self.classifier._generate_dummy_vix_data(50)
        }

        # Make a few predictions to build history
        for _ in range(5):
            self.classifier.predict_regime(test_data)

        summary = self.classifier.get_regime_summary()

        self.assertIn('current_regime', summary)
        self.assertIn('confidence', summary)
        self.assertIn('probabilities', summary)
        self.assertIn('indicators', summary)
        self.assertIn('recent_distribution', summary)
        self.assertIn('models_fitted', summary)

    def test_validate_crisis_periods_no_data(self):
        """Test crisis period validation with no historical data"""
        # Mock empty historical data
        with patch.object(self.classifier, 'get_historical_regimes') as mock_history:
            mock_history.return_value = pd.DataFrame()

            validation = self.classifier.validate_crisis_periods()

            self.assertIn('error', validation)

    def test_factory_function(self):
        """Test factory function"""
        classifier = create_regime_classifier()

        self.assertIsInstance(classifier, MarketRegimeClassifier)

    def test_risk_manager_integration(self):
        """Test integration function for existing risk manager"""
        # Mock the classifier to avoid data loading
        with patch('market_regime_classifier.MarketRegimeClassifier') as mock_cls:
            mock_instance = Mock()
            mock_prediction = Mock()
            mock_prediction.regime = MarketRegime.VOLATILE
            mock_instance.predict_regime.return_value = mock_prediction
            mock_instance.load_models.return_value = True
            mock_cls.return_value = mock_instance

            result = get_regime_for_risk_manager()

            # Should return an ExistingMarketRegime enum value
            # (This will import the existing enum if available)
            self.assertIsNotNone(result)


class TestIntegrationAndPerformance(unittest.TestCase):
    """Integration and performance tests"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create classifier
        temp_dir = tempfile.mkdtemp()

        try:
            classifier = MarketRegimeClassifier(cache_dir=temp_dir)

            # Load data (will use dummy data)
            data = classifier.load_market_data(
                symbols=['SPY', 'QQQ'],
                limit=100
            )

            # Fit models
            classifier.fit_models(data)

            # Make prediction
            prediction = classifier.predict_regime(data)

            # Validate results
            self.assertIsInstance(prediction, RegimePrediction)
            self.assertIn(prediction.regime, [MarketRegime.NORMAL, MarketRegime.VOLATILE, MarketRegime.CRISIS])

            # Get summary
            summary = classifier.get_regime_summary()
            self.assertIn('current_regime', summary)

            # Save models
            save_success = classifier.save_models()
            self.assertTrue(save_success)

        finally:
            # Clean up
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_performance_with_large_data(self):
        """Test performance with larger datasets"""
        classifier = MarketRegimeClassifier()

        # Create larger dummy dataset
        large_data = {
            'market_data': classifier._generate_dummy_market_data(
                ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK'], 500
            ),
            'vix_data': classifier._generate_dummy_vix_data(500)
        }

        # Time the fitting process
        import time
        start_time = time.time()

        classifier.fit_models(large_data)

        fit_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(fit_time, 60)  # 60 seconds max

        # Time the prediction process
        start_time = time.time()

        prediction = classifier.predict_regime(large_data)

        predict_time = time.time() - start_time

        # Prediction should be fast
        self.assertLess(predict_time, 5)  # 5 seconds max

        # Verify prediction quality
        self.assertIsInstance(prediction, RegimePrediction)
        self.assertGreater(prediction.confidence, 0)

    def test_error_handling_and_recovery(self):
        """Test error handling and graceful degradation"""
        classifier = MarketRegimeClassifier()

        # Test with malformed data
        bad_data = {
            'market_data': {'BAD': pd.DataFrame({'wrong_columns': [1, 2, 3]})},
            'vix_data': pd.DataFrame({'also_wrong': [1, 2, 3]})
        }

        # Should not crash, should return reasonable defaults
        prediction = classifier.predict_regime(bad_data)

        self.assertIsInstance(prediction, RegimePrediction)
        # Should default to NORMAL regime with low confidence
        self.assertLessEqual(prediction.confidence, 0.6)

    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        classifier = MarketRegimeClassifier()

        test_data = {
            'market_data': classifier._generate_dummy_market_data(['SPY'], 100),
            'vix_data': classifier._generate_dummy_vix_data(100)
        }

        # Make many predictions to test memory management
        initial_history_length = len(classifier.regime_history)

        for _ in range(1200):  # More than the 1000 limit
            classifier.predict_regime(test_data)

        # History should be limited to prevent memory growth
        self.assertLessEqual(len(classifier.regime_history), 1000)

        # Should have more than initial but capped
        self.assertGreater(len(classifier.regime_history), initial_history_length)


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test edge cases and robustness"""

    def setUp(self):
        """Set up test fixtures"""
        if not CLASSIFIER_AVAILABLE:
            self.skipTest("Regime classifier not available")

    def test_single_data_point(self):
        """Test behavior with minimal data"""
        classifier = MarketRegimeClassifier()

        # Single data point
        minimal_data = {
            'market_data': {
                'SPY': pd.DataFrame({
                    'close': [100],
                    'open': [99],
                    'high': [101],
                    'low': [98],
                    'volume': [1000000]
                })
            },
            'vix_data': pd.DataFrame({
                'close': [20],
                'open': [19],
                'high': [21],
                'low': [18],
                'volume': [100000]
            })
        }

        # Should handle gracefully
        prediction = classifier.predict_regime(minimal_data)

        self.assertIsInstance(prediction, RegimePrediction)

    def test_missing_vix_data(self):
        """Test behavior when VIX data is missing"""
        classifier = MarketRegimeClassifier()

        # Data without VIX
        no_vix_data = {
            'market_data': classifier._generate_dummy_market_data(['SPY'], 50)
        }

        prediction = classifier.predict_regime(no_vix_data)

        self.assertIsInstance(prediction, RegimePrediction)
        # VIX level should be 0 when missing
        self.assertEqual(prediction.indicators.vix_level, 0.0)

    def test_all_nan_data(self):
        """Test behavior with NaN data"""
        classifier = MarketRegimeClassifier()

        # Data with NaN values
        nan_data = {
            'market_data': {
                'SPY': pd.DataFrame({
                    'close': [np.nan] * 50,
                    'open': [np.nan] * 50,
                    'high': [np.nan] * 50,
                    'low': [np.nan] * 50,
                    'volume': [np.nan] * 50
                })
            },
            'vix_data': pd.DataFrame({
                'close': [np.nan] * 50
            })
        }

        prediction = classifier.predict_regime(nan_data)

        self.assertIsInstance(prediction, RegimePrediction)
        # Should default to reasonable values

    def test_extreme_values(self):
        """Test behavior with extreme market values"""
        classifier = MarketRegimeClassifier()

        # Extreme VIX values
        extreme_data = {
            'market_data': classifier._generate_dummy_market_data(['SPY'], 50),
            'vix_data': pd.DataFrame({
                'close': [100] * 50  # Extremely high VIX
            })
        }

        prediction = classifier.predict_regime(extreme_data)

        self.assertIsInstance(prediction, RegimePrediction)
        # Should likely classify as crisis
        self.assertIn(prediction.regime, [MarketRegime.VOLATILE, MarketRegime.CRISIS])


# Test runner
if __name__ == '__main__':
    # Configure test runner
    import sys

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestRegimeIndicators,
        TestThresholdRegimeDetector,
        TestHiddenMarkovRegimeModel,
        TestMLRegimeClassifier,
        TestMarketRegimeClassifier,
        TestIntegrationAndPerformance,
        TestEdgeCasesAndRobustness
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.splitlines()[-1]}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.splitlines()[-1]}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)