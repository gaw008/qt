#!/usr/bin/env python3
"""
Risk Management Calibration Test Suite
风险管理校准测试套件

Comprehensive testing and validation for live trading risk management:
- ES@97.5% calculation accuracy and performance
- Drawdown monitoring under various market conditions
- Factor crowding detection effectiveness
- Real-time performance benchmarks
- Integration with execution systems

Test Categories:
1. Performance Tests - Sub-100ms response validation
2. Accuracy Tests - Risk metric calculation validation
3. Stress Tests - High volatility and crisis scenarios
4. Integration Tests - Live trading workflow validation
5. Calibration Tests - Parameter optimization validation
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import warnings
from typing import Dict, List, Any
import json

# Add bot directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot'))

# Import risk management components
try:
    from bot.enhanced_risk_manager import EnhancedRiskManager, RiskLevel, MarketRegime
    from bot.factor_crowding_monitor import FactorCrowdingMonitor, CrowdingLevel
    from bot.risk_performance_optimizer import RiskPerformanceOptimizer, PerformanceLevel
    from bot.live_risk_integration import LiveRiskIntegrator, RiskCheckResult, TradingAction
except ImportError as e:
    print(f"Warning: Could not import risk components: {e}")
    print("Some tests may be skipped")

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress warnings
warnings.filterwarnings('ignore')

class RiskPerformanceTests(unittest.TestCase):
    """Performance validation tests for risk calculations"""

    def setUp(self):
        """Set up test environment"""
        self.optimizer = RiskPerformanceOptimizer(
            performance_level=PerformanceLevel.TESTING,
            max_workers=2,
            cache_size=100
        )

        # Generate test data
        np.random.seed(42)
        self.test_returns = np.random.normal(0.001, 0.02, 252)  # 1 year daily
        self.test_returns[200:210] = np.random.normal(-0.04, 0.01, 10)  # Crisis

        # Portfolio test data
        self.portfolio_values = np.cumprod(1 + self.test_returns) * 1000000
        self.factor_exposures = {
            "momentum": np.random.normal(0, 1, 100),
            "value": np.random.normal(0, 1, 100),
            "quality": np.random.normal(0, 1, 100)
        }
        self.portfolio_weights = np.random.dirichlet(np.ones(100) * 0.5)

    def test_es_calculation_performance(self):
        """Test ES calculation meets sub-100ms requirement"""
        print("\nTesting ES calculation performance...")

        # Warm up
        for _ in range(5):
            self.optimizer.calculate_portfolio_es_optimized(self.test_returns)

        # Performance test
        times = []
        for i in range(20):
            start_time = time.perf_counter()
            result = self.optimizer.calculate_portfolio_es_optimized(self.test_returns)
            end_time = time.perf_counter()

            calculation_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(calculation_time)

            # Verify results are reasonable
            self.assertIn('es_97', result)
            self.assertIn('es_99', result)
            self.assertGreater(result['es_97'], 0)
            self.assertGreater(result['es_99'], result['es_97'])

        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)

        print(f"ES Calculation Performance:")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Maximum time: {max_time:.2f}ms")
        print(f"  Minimum time: {min_time:.2f}ms")

        # Performance requirements
        self.assertLess(avg_time, 50.0, "Average ES calculation should be under 50ms")
        self.assertLess(max_time, 100.0, "Maximum ES calculation should be under 100ms")

    def test_drawdown_monitoring_performance(self):
        """Test drawdown monitoring performance"""
        print("\nTesting drawdown monitoring performance...")

        times = []
        for i in range(15):
            start_time = time.perf_counter()
            result = self.optimizer.real_time_drawdown_monitor(self.portfolio_values)
            end_time = time.perf_counter()

            calculation_time = (end_time - start_time) * 1000
            times.append(calculation_time)

            # Verify results
            self.assertIn('current_drawdown', result)
            self.assertIn('maximum_drawdown', result)
            self.assertIn('risk_tier', result)
            self.assertGreaterEqual(result['current_drawdown'], 0)

        avg_time = np.mean(times)
        print(f"Drawdown Monitoring Performance:")
        print(f"  Average time: {avg_time:.2f}ms")

        self.assertLess(avg_time, 30.0, "Drawdown calculation should be under 30ms")

    def test_factor_crowding_performance(self):
        """Test factor crowding detection performance"""
        print("\nTesting factor crowding performance...")

        times = []
        for i in range(10):
            start_time = time.perf_counter()
            result = self.optimizer.optimized_factor_crowding_check(
                self.factor_exposures, self.portfolio_weights
            )
            end_time = time.perf_counter()

            calculation_time = (end_time - start_time) * 1000
            times.append(calculation_time)

            # Verify results
            self.assertIn('factor_results', result)
            self.assertIn('overall_crowding_score', result)

        avg_time = np.mean(times)
        print(f"Factor Crowding Performance:")
        print(f"  Average time: {avg_time:.2f}ms")

        self.assertLess(avg_time, 75.0, "Factor crowding check should be under 75ms")

    def test_concurrent_risk_assessment_performance(self):
        """Test concurrent risk assessment performance"""
        print("\nTesting concurrent risk assessment performance...")

        portfolio_data = {
            "returns": self.test_returns,
            "values": self.portfolio_values,
            "factor_exposures": self.factor_exposures,
            "weights": self.portfolio_weights
        }

        market_data = {"vix": 25.0, "market_correlation": 0.6}

        times = []
        for i in range(8):
            start_time = time.perf_counter()
            result = self.optimizer.concurrent_risk_assessment(portfolio_data, market_data)
            end_time = time.perf_counter()

            calculation_time = (end_time - start_time) * 1000
            times.append(calculation_time)

            # Verify comprehensive results
            self.assertIn('overall_risk_level', result)
            self.assertIn('detailed_metrics', result)
            self.assertIn('performance', result)

        avg_time = np.mean(times)
        print(f"Concurrent Assessment Performance:")
        print(f"  Average time: {avg_time:.2f}ms")

        self.assertLess(avg_time, 150.0, "Concurrent assessment should be under 150ms")

class RiskAccuracyTests(unittest.TestCase):
    """Accuracy validation tests for risk calculations"""

    def setUp(self):
        """Set up test environment"""
        self.risk_manager = EnhancedRiskManager()
        np.random.seed(123)

    def test_es_calculation_accuracy(self):
        """Test ES calculation accuracy against known values"""
        print("\nTesting ES calculation accuracy...")

        # Create known distribution
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 95)

        # Calculate ES@97.5% (should be mean of worst 2.5% = top 2-3 worst returns)
        es_975 = self.risk_manager.calculate_expected_shortfall(returns, 0.975)

        # For 100 samples, 2.5% = 2.5 samples, so we take worst 2-3 returns
        expected_es = np.mean([-0.05, -0.04])  # Mean of worst 2 returns

        print(f"ES@97.5%: {es_975:.4f}, Expected: {expected_es:.4f}")

        # Allow small tolerance for numerical precision
        self.assertAlmostEqual(es_975, abs(expected_es), places=3)

    def test_drawdown_calculation_accuracy(self):
        """Test drawdown calculation accuracy"""
        print("\nTesting drawdown calculation accuracy...")

        # Create portfolio with known drawdown
        values = np.array([100, 110, 105, 120, 90, 95, 115])  # Max DD should be 25%
        returns = np.diff(values) / values[:-1]

        tail_metrics = self.risk_manager.calculate_tail_risk_metrics(returns)

        # Maximum drawdown should be (120-90)/120 = 25%
        expected_max_dd = (120 - 90) / 120

        print(f"Calculated Max DD: {tail_metrics.max_drawdown:.4f}")
        print(f"Expected Max DD: {expected_max_dd:.4f}")

        self.assertAlmostEqual(tail_metrics.max_drawdown, expected_max_dd, places=3)

    def test_regime_detection_accuracy(self):
        """Test market regime detection accuracy"""
        print("\nTesting market regime detection...")

        # Test normal market conditions
        normal_market = {"vix": 18, "market_correlation": 0.4, "momentum_strength": 0.2}
        regime = self.risk_manager.detect_market_regime(normal_market)
        self.assertEqual(regime, MarketRegime.NORMAL)

        # Test crisis conditions
        crisis_market = {"vix": 35, "market_correlation": 0.8, "momentum_strength": 0.1}
        regime = self.risk_manager.detect_market_regime(crisis_market)
        self.assertEqual(regime, MarketRegime.CRISIS)

        # Test volatile conditions
        volatile_market = {"vix": 25, "market_correlation": 0.6, "momentum_strength": 0.2}
        regime = self.risk_manager.detect_market_regime(volatile_market)
        self.assertEqual(regime, MarketRegime.VOLATILE)

        print("Market regime detection: PASSED")

class RiskStressTests(unittest.TestCase):
    """Stress testing under extreme market conditions"""

    def setUp(self):
        """Set up stress test environment"""
        self.integrator = LiveRiskIntegrator()
        np.random.seed(456)

    def test_crisis_scenario_handling(self):
        """Test risk system under crisis scenario"""
        print("\nTesting crisis scenario handling...")

        # Create crisis scenario (2008-style crash)
        crisis_returns = np.concatenate([
            np.random.normal(0.001, 0.015, 200),  # Normal period
            np.random.normal(-0.06, 0.04, 30),    # Crisis period
            np.random.normal(0.002, 0.025, 50)    # Recovery
        ])

        portfolio_data = {
            'total_value': 1000000,
            'returns': crisis_returns,
            'current_drawdown': 0.25,  # 25% drawdown
            'positions': [
                {'symbol': 'AAPL', 'market_value': 200000, 'sector': 'Technology'},
                {'symbol': 'GOOGL', 'market_value': 150000, 'sector': 'Technology'}
            ]
        }

        market_data = {'vix': 45, 'market_correlation': 0.9}

        self.integrator.update_portfolio_state(portfolio_data)
        self.integrator.update_market_data(market_data)

        # Test emergency stop triggers
        emergency_stop, reasons = self.integrator.emergency_stop_check()

        print(f"Emergency stop triggered: {emergency_stop}")
        print(f"Reasons: {reasons}")

        # Should trigger emergency stop due to extreme drawdown
        self.assertTrue(emergency_stop, "Emergency stop should trigger in crisis")
        self.assertGreater(len(reasons), 0, "Should have specific emergency reasons")

    def test_high_volatility_performance(self):
        """Test performance under high volatility"""
        print("\nTesting high volatility performance...")

        # High volatility scenario
        high_vol_returns = np.random.normal(0, 0.05, 1000)  # 80% annual volatility

        optimizer = RiskPerformanceOptimizer(PerformanceLevel.PRODUCTION)

        # Test performance doesn't degrade significantly
        times = []
        for i in range(10):
            start = time.perf_counter()
            result = optimizer.calculate_portfolio_es_optimized(high_vol_returns)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        print(f"High volatility ES calculation: {avg_time:.2f}ms")

        # Performance should still be reasonable
        self.assertLess(avg_time, 100.0, "Performance should remain good under high volatility")

    def test_concentrated_portfolio_stress(self):
        """Test highly concentrated portfolio scenarios"""
        print("\nTesting concentrated portfolio stress...")

        # Highly concentrated portfolio
        portfolio_data = {
            'total_value': 1000000,
            'positions': [
                {'symbol': 'AAPL', 'market_value': 400000, 'sector': 'Technology'},  # 40%
                {'symbol': 'GOOGL', 'market_value': 300000, 'sector': 'Technology'}, # 30%
                {'symbol': 'MSFT', 'market_value': 200000, 'sector': 'Technology'}   # 20%
            ],
            'returns': np.random.normal(-0.01, 0.03, 100)  # Negative trend
        }

        self.integrator.update_portfolio_state(portfolio_data)

        # Test large trade in already concentrated position
        risk_check = self.integrator.pre_trade_risk_check(
            symbol="AAPL",
            quantity=1000,  # Large additional position
            order_type="MARKET",
            current_price=150.0
        )

        print(f"Concentrated portfolio risk check: {risk_check.check_result.value}")
        print(f"Violations: {risk_check.violations}")

        # Should flag concentration risk
        self.assertIn("POSITION_SIZE_EXCEEDED", risk_check.violations)
        self.assertEqual(risk_check.check_result, RiskCheckResult.FAIL)

class RiskIntegrationTests(unittest.TestCase):
    """Integration testing with live trading workflows"""

    def setUp(self):
        """Set up integration test environment"""
        self.integrator = LiveRiskIntegrator()

    def test_pre_trade_workflow(self):
        """Test complete pre-trade risk workflow"""
        print("\nTesting pre-trade workflow...")

        # Setup portfolio
        portfolio_data = {
            'total_value': 1000000,
            'positions': [
                {'symbol': 'AAPL', 'market_value': 100000, 'sector': 'Technology'},
                {'symbol': 'JPM', 'market_value': 80000, 'sector': 'Financial'}
            ],
            'returns': np.random.normal(0.001, 0.015, 100)
        }

        self.integrator.update_portfolio_state(portfolio_data)

        # Test normal trade
        risk_check = self.integrator.pre_trade_risk_check(
            symbol="MSFT",
            quantity=200,
            order_type="MARKET",
            current_price=250.0
        )

        self.assertEqual(risk_check.check_result, RiskCheckResult.PASS)
        self.assertEqual(risk_check.allowed_quantity, 200)

        # Test oversized trade
        large_risk_check = self.integrator.pre_trade_risk_check(
            symbol="NVDA",
            quantity=2000,  # Very large position
            order_type="MARKET",
            current_price=400.0
        )

        self.assertIn(large_risk_check.check_result, [RiskCheckResult.FAIL, RiskCheckResult.WARNING])
        print(f"Large trade check: {large_risk_check.check_result.value}")

    def test_post_trade_workflow(self):
        """Test post-trade risk update workflow"""
        print("\nTesting post-trade workflow...")

        portfolio_data = {
            'total_value': 1000000,
            'positions': [{'symbol': 'AAPL', 'market_value': 100000, 'sector': 'Technology'}]
        }

        self.integrator.update_portfolio_state(portfolio_data)

        # Execute trade and update
        post_trade = self.integrator.post_trade_risk_update(
            trade_id="TRADE_001",
            symbol="MSFT",
            quantity=100,
            execution_price=250.0
        )

        self.assertEqual(post_trade.trade_id, "TRADE_001")
        self.assertGreater(post_trade.risk_contribution, 0)
        self.assertIsInstance(post_trade.new_risk_metrics, dict)

    def test_emergency_controls(self):
        """Test emergency control mechanisms"""
        print("\nTesting emergency controls...")

        # Disable trading
        self.integrator.disable_trading()

        risk_check = self.integrator.pre_trade_risk_check("AAPL", 100, "MARKET", 150.0)
        self.assertEqual(risk_check.check_result, RiskCheckResult.FAIL)
        self.assertIn("TRADING_DISABLED", risk_check.violations)

        # Re-enable trading
        self.integrator.enable_trading()

        risk_check = self.integrator.pre_trade_risk_check("AAPL", 100, "MARKET", 150.0)
        self.assertNotEqual(risk_check.check_result, RiskCheckResult.FAIL)

class RiskCalibrationTests(unittest.TestCase):
    """Calibration validation tests"""

    def test_limit_calibration(self):
        """Test risk limit calibration"""
        print("\nTesting risk limit calibration...")

        integrator = LiveRiskIntegrator()

        # Verify limits are within reasonable ranges
        limits = integrator.limits

        # Position sizing limits
        self.assertLessEqual(limits.max_position_size_pct, 0.15)  # Max 15% single position
        self.assertGreaterEqual(limits.max_position_size_pct, 0.05)  # Min 5% makes sense

        # Risk metric limits
        self.assertLessEqual(limits.max_es_97_5_daily, 0.05)  # Max 5% daily ES
        self.assertGreaterEqual(limits.max_es_97_5_daily, 0.02)  # Min 2% for active strategy

        # Drawdown limits should be progressive
        self.assertLess(limits.warning_drawdown, limits.action_drawdown)
        self.assertLess(limits.action_drawdown, limits.emergency_drawdown)

        print(f"Position size limit: {limits.max_position_size_pct:.1%}")
        print(f"ES@97.5% limit: {limits.max_es_97_5_daily:.1%}")
        print(f"Emergency drawdown: {limits.emergency_drawdown:.1%}")

    def test_performance_optimization(self):
        """Test performance optimization effectiveness"""
        print("\nTesting performance optimization...")

        optimizer = RiskPerformanceOptimizer(PerformanceLevel.PRODUCTION)

        # Test caching effectiveness
        test_data = np.random.normal(0, 0.02, 252)

        # First calculation (cache miss)
        start = time.perf_counter()
        result1 = optimizer.calculate_portfolio_es_optimized(test_data)
        time1 = time.perf_counter() - start

        # Second calculation (cache hit)
        start = time.perf_counter()
        result2 = optimizer.calculate_portfolio_es_optimized(test_data)
        time2 = time.perf_counter() - start

        # Results should be identical
        self.assertEqual(result1, result2)

        # Second call should be faster (cache hit)
        print(f"First call: {time1*1000:.2f}ms, Second call: {time2*1000:.2f}ms")
        self.assertLess(time2, time1 * 0.5, "Cache should provide significant speedup")

def run_comprehensive_calibration_test():
    """Run comprehensive risk management calibration test"""
    print("Risk Management System Calibration Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add performance tests
    test_suite.addTest(unittest.makeSuite(RiskPerformanceTests))
    test_suite.addTest(unittest.makeSuite(RiskAccuracyTests))
    test_suite.addTest(unittest.makeSuite(RiskStressTests))
    test_suite.addTest(unittest.makeSuite(RiskIntegrationTests))
    test_suite.addTest(unittest.makeSuite(RiskCalibrationTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Export test report
    test_report = {
        "test_timestamp": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        "calibration_status": "PASSED" if len(result.failures) == 0 and len(result.errors) == 0 else "FAILED",
        "performance_requirements": {
            "es_calculation_ms": "< 50ms average",
            "drawdown_monitoring_ms": "< 30ms average",
            "factor_crowding_ms": "< 75ms average",
            "concurrent_assessment_ms": "< 150ms average"
        },
        "risk_limits_validated": {
            "max_position_size": "5-15%",
            "max_es_97_5_daily": "2-5%",
            "emergency_drawdown": "Progressive tiers"
        }
    }

    with open("risk_calibration_test_report.json", 'w', encoding='utf-8') as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)

    print(f"\nTest report exported: risk_calibration_test_report.json")
    print(f"Calibration Status: {test_report['calibration_status']}")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_calibration_test()
    sys.exit(0 if success else 1)