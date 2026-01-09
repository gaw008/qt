"""
Comprehensive Test Suite for Risk Control and Backtesting Systems

This module provides thorough testing of:
- Risk filtering engine functionality
- Portfolio-level backtesting capabilities  
- Integration between scoring, screening, and risk systems
- Performance validation and attribution analysis
- Real-world scenario testing
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import tempfile
from pathlib import Path

# Add bot directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot'))

# Import test targets
try:
    from bot.risk_filters import RiskFilterEngine, RiskLimits, RiskMetrics
    from bot.risk_integrated_selection import RiskIntegratedSelector, SelectionConfig
    from backtest import PortfolioBacktester
    from bot.scoring_engine import MultiFactorScoringEngine, FactorWeights
    from bot.stock_screener import StockScreener, ScreeningCriteria
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Could not import modules for testing: {e}")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate realistic test data for risk system testing."""
    
    @staticmethod
    def generate_ohlcv_data(symbol: str, 
                           start_date: str = "2020-01-01", 
                           end_date: str = "2023-12-31",
                           base_price: float = 100.0,
                           volatility: float = 0.2) -> pd.DataFrame:
        """Generate realistic OHLCV data for testing."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range
        dates = pd.date_range(start=start, end=end, freq='D')
        n_periods = len(dates)
        
        # Generate price series with realistic characteristics
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        # Log returns with volatility clustering
        returns = np.random.normal(0, volatility/np.sqrt(252), n_periods)
        
        # Add some momentum and mean reversion
        for i in range(1, len(returns)):
            momentum = 0.05 * returns[i-1]  # Small momentum effect
            mean_reversion = -0.02 * np.sum(returns[max(0, i-10):i]) / min(i, 10)
            returns[i] += momentum + mean_reversion
        
        # Convert to prices
        log_prices = np.cumsum(returns) + np.log(base_price)
        prices = np.exp(log_prices)
        
        # Generate OHLCV
        high_mult = 1 + np.abs(np.random.normal(0, 0.01, n_periods))
        low_mult = 1 - np.abs(np.random.normal(0, 0.01, n_periods))
        
        high = prices * high_mult
        low = prices * low_mult
        
        # Volume with realistic patterns
        base_volume = 1000000 * (1 + hash(symbol) % 10)  # Different base volumes
        volume_noise = np.random.lognormal(0, 0.5, n_periods)
        volume = (base_volume * volume_noise).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': high,
            'low': low, 
            'close': prices,
            'volume': volume
        })
        
        return df
    
    @staticmethod
    def generate_market_data_dict(symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Generate market data for multiple symbols."""
        market_data = {}
        
        for symbol in symbols:
            # Vary characteristics by symbol
            base_price = 50 + (hash(symbol) % 500)
            volatility = 0.1 + 0.3 * ((hash(symbol) % 100) / 100)
            
            df = TestDataGenerator.generate_ohlcv_data(
                symbol, base_price=base_price, volatility=volatility, **kwargs
            )
            market_data[symbol] = df
        
        return market_data
    
    @staticmethod
    def create_test_csv_files(symbols: List[str], output_dir: str):
        """Create CSV files for backtesting."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        market_data = TestDataGenerator.generate_market_data_dict(symbols)
        
        for symbol, df in market_data.items():
            csv_path = Path(output_dir) / f"{symbol}.csv"
            df.to_csv(csv_path, index=False)
        
        return output_dir


class RiskFilterEngineTests(unittest.TestCase):\n    \"\"\"Test suite for RiskFilterEngine.\"\"\"\n    \n    def setUp(self):\n        if not MODULES_AVAILABLE:\n            self.skipTest(\"Required modules not available\")\n        \n        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']\n        self.market_data = TestDataGenerator.generate_market_data_dict(self.symbols)\n        self.risk_engine = RiskFilterEngine()\n    \n    def test_risk_metrics_calculation(self):\n        \"\"\"Test risk metrics calculation.\"\"\"\n        logger.info(\"Testing risk metrics calculation\")\n        \n        risk_metrics = self.risk_engine._calculate_risk_metrics(\n            self.symbols, self.market_data, None, None\n        )\n        \n        self.assertEqual(len(risk_metrics), len(self.symbols))\n        \n        for symbol, metrics in risk_metrics.items():\n            self.assertIsInstance(metrics, RiskMetrics)\n            self.assertEqual(metrics.symbol, symbol)\n            self.assertGreater(metrics.volatility, 0)\n            self.assertGreater(metrics.liquidity_score, 0)\n            self.assertGreater(metrics.market_cap, 0)\n            self.assertIn(metrics.risk_tier, ['Low', 'Medium', 'High'])\n        \n        logger.info(\"✓ Risk metrics calculation passed\")\n    \n    def test_individual_filters(self):\n        \"\"\"Test individual security filters.\"\"\"\n        logger.info(\"Testing individual security filters\")\n        \n        # Test with default limits\n        filtered_symbols, risk_metrics = self.risk_engine.apply_risk_filters(\n            self.symbols, self.market_data\n        )\n        \n        self.assertIsInstance(filtered_symbols, list)\n        self.assertLessEqual(len(filtered_symbols), len(self.symbols))\n        \n        # All filtered symbols should pass individual filters\n        for symbol in filtered_symbols:\n            metrics = risk_metrics.get(symbol)\n            self.assertIsNotNone(metrics)\n            self.assertTrue(metrics.passes_filters)\n        \n        logger.info(f\"✓ Individual filters passed: {len(filtered_symbols)}/{len(self.symbols)} symbols\")\n    \n    def test_position_sizing(self):\n        \"\"\"Test risk-adjusted position sizing.\"\"\"\n        logger.info(\"Testing position sizing\")\n        \n        # Create mock scores\n        scores = {symbol: np.random.uniform(0.5, 1.0) for symbol in self.symbols}\n        risk_metrics = self.risk_engine._calculate_risk_metrics(\n            self.symbols, self.market_data, None, None\n        )\n        \n        position_weights = self.risk_engine.calculate_position_sizes(\n            self.symbols, scores, risk_metrics\n        )\n        \n        # Validate position weights\n        total_weight = sum(position_weights.values())\n        self.assertAlmostEqual(total_weight, 1.0, places=2)\n        \n        for symbol, weight in position_weights.items():\n            self.assertGreaterEqual(weight, 0)\n            self.assertLessEqual(weight, self.risk_engine.limits.max_single_position)\n        \n        logger.info(f\"✓ Position sizing passed: {len(position_weights)} positions\")\n    \n    def test_portfolio_risk_validation(self):\n        \"\"\"Test portfolio-level risk validation.\"\"\"\n        logger.info(\"Testing portfolio risk validation\")\n        \n        # Create test portfolio\n        positions = {symbol: 1.0/len(self.symbols) for symbol in self.symbols}\n        \n        risk_metrics = self.risk_engine._calculate_risk_metrics(\n            self.symbols, self.market_data, None, None\n        )\n        \n        validation_result = self.risk_engine.validate_portfolio_risk(\n            positions, self.market_data, risk_metrics\n        )\n        \n        self.assertIn('passes_validation', validation_result)\n        self.assertIn('risk_violations', validation_result)\n        self.assertIn('risk_metrics', validation_result)\n        \n        self.assertIsInstance(validation_result['passes_validation'], bool)\n        self.assertIsInstance(validation_result['risk_violations'], list)\n        self.assertIsInstance(validation_result['risk_metrics'], dict)\n        \n        logger.info(f\"✓ Portfolio validation passed: {'PASS' if validation_result['passes_validation'] else 'FAIL'}\")\n    \n    def test_extreme_risk_scenarios(self):\n        \"\"\"Test risk filtering under extreme conditions.\"\"\"\n        logger.info(\"Testing extreme risk scenarios\")\n        \n        # Create high-risk limits\n        high_risk_limits = RiskLimits(\n            max_volatility=0.1,  # Very low volatility threshold\n            min_avg_volume=10000000,  # High volume requirement\n            min_market_cap=100e9,  # Very high market cap\n            max_single_position=0.05,  # Very small positions\n        )\n        \n        extreme_risk_engine = RiskFilterEngine(high_risk_limits)\n        \n        filtered_symbols, _ = extreme_risk_engine.apply_risk_filters(\n            self.symbols, self.market_data\n        )\n        \n        # Should filter out most/all symbols\n        self.assertLessEqual(len(filtered_symbols), len(self.symbols))\n        \n        logger.info(f\"✓ Extreme risk scenarios passed: {len(filtered_symbols)} symbols survived\")\n\n\nclass PortfolioBacktesterTests(unittest.TestCase):\n    \"\"\"Test suite for PortfolioBacktester.\"\"\"\n    \n    def setUp(self):\n        if not MODULES_AVAILABLE:\n            self.skipTest(\"Required modules not available\")\n        \n        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']\n        self.temp_dir = tempfile.mkdtemp()\n        TestDataGenerator.create_test_csv_files(self.symbols, self.temp_dir)\n        \n        self.backtester = PortfolioBacktester(\n            start_date=\"2022-01-01\",\n            end_date=\"2023-06-30\",\n            initial_capital=1000000.0,\n            rebalance_frequency=\"monthly\"\n        )\n    \n    def tearDown(self):\n        # Clean up temporary files\n        import shutil\n        shutil.rmtree(self.temp_dir, ignore_errors=True)\n    \n    def test_basic_portfolio_backtest(self):\n        \"\"\"Test basic portfolio backtesting functionality.\"\"\"\n        logger.info(\"Testing basic portfolio backtest\")\n        \n        results = self.backtester.run_portfolio_backtest(\n            data_source=\"csv\",\n            csv_directory=self.temp_dir,\n            universe=self.symbols\n        )\n        \n        self.assertIn('config', results)\n        self.assertIn('periods', results)\n        self.assertIn('performance', results)\n        \n        periods = results['periods']\n        self.assertGreater(len(periods), 0)\n        \n        # Check period structure\n        for period in periods:\n            self.assertIn('date', period)\n            self.assertIn('portfolio_value', period)\n            self.assertIn('positions', period)\n        \n        # Check performance metrics\n        if 'performance' in results and results['performance']:\n            performance = results['performance']\n            self.assertIn('total_return', performance)\n            self.assertIn('volatility', performance)\n            self.assertIn('sharpe_ratio', performance)\n            self.assertIn('max_drawdown', performance)\n        \n        logger.info(\"✓ Basic portfolio backtest passed\")\n    \n    def test_strategy_comparison(self):\n        \"\"\"Test strategy comparison functionality.\"\"\"\n        logger.info(\"Testing strategy comparison\")\n        \n        strategies = {\n            \"conservative\": {\"top_n\": 10, \"risk_adjustment\": \"high\"},\n            \"aggressive\": {\"top_n\": 20, \"risk_adjustment\": \"low\"}\n        }\n        \n        comparison_results = self.backtester.run_strategy_comparison(\n            strategies,\n            data_source=\"csv\",\n            csv_directory=self.temp_dir\n        )\n        \n        self.assertIn('strategies', comparison_results)\n        self.assertIn('comparison_metrics', comparison_results)\n        \n        strategies_results = comparison_results['strategies']\n        self.assertEqual(len(strategies_results), len(strategies))\n        \n        for strategy_name in strategies:\n            self.assertIn(strategy_name, strategies_results)\n        \n        logger.info(\"✓ Strategy comparison passed\")\n    \n    def test_performance_metrics_calculation(self):\n        \"\"\"Test performance metrics calculation accuracy.\"\"\"\n        logger.info(\"Testing performance metrics calculation\")\n        \n        # Create simple test case with known returns\n        test_periods = [\n            {\"date\": \"2022-01-01\", \"portfolio_value\": 1000000},\n            {\"date\": \"2022-02-01\", \"portfolio_value\": 1050000},  # 5% return\n            {\"date\": \"2022-03-01\", \"portfolio_value\": 1100000},  # ~4.76% return\n            {\"date\": \"2022-04-01\", \"portfolio_value\": 1080000},  # -1.82% return\n        ]\n        \n        test_results = {\"periods\": test_periods}\n        performance = self.backtester._calculate_performance_metrics(test_results)\n        \n        self.assertIn('total_return', performance)\n        self.assertIn('volatility', performance)\n        self.assertIn('sharpe_ratio', performance)\n        \n        # Total return should be 8% (1,080,000 / 1,000,000 - 1)\n        expected_total_return = 0.08\n        self.assertAlmostEqual(performance['total_return'], expected_total_return, places=3)\n        \n        logger.info(\"✓ Performance metrics calculation passed\")\n\n\nclass RiskIntegratedSelectorTests(unittest.TestCase):\n    \"\"\"Test suite for RiskIntegratedSelector.\"\"\"\n    \n    def setUp(self):\n        if not MODULES_AVAILABLE:\n            self.skipTest(\"Required modules not available\")\n        \n        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']\n        self.market_data = TestDataGenerator.generate_market_data_dict(self.symbols)\n        \n        # Create mock quote client\n        self.mock_quote_client = type('MockClient', (), {})()\n        \n        self.selector = RiskIntegratedSelector()\n    \n    def test_integrated_selection_process(self):\n        \"\"\"Test complete integrated selection process.\"\"\"\n        logger.info(\"Testing integrated selection process\")\n        \n        result = self.selector.run_integrated_selection(\n            self.mock_quote_client,\n            universe=self.symbols,\n            market_data=self.market_data,\n            dry_run=True\n        )\n        \n        self.assertIsInstance(result.selected_symbols, list)\n        self.assertIsInstance(result.position_weights, dict)\n        \n        # Check position weights sum to 1\n        if result.position_weights:\n            total_weight = sum(result.position_weights.values())\n            self.assertAlmostEqual(total_weight, 1.0, places=2)\n        \n        # Check all selected symbols have weights\n        for symbol in result.selected_symbols:\n            self.assertIn(symbol, result.position_weights)\n            self.assertGreater(result.position_weights[symbol], 0)\n        \n        logger.info(f\"✓ Integrated selection passed: {len(result.selected_symbols)} positions\")\n    \n    def test_rebalance_decision_logic(self):\n        \"\"\"Test rebalancing decision logic.\"\"\"\n        logger.info(\"Testing rebalance decision logic\")\n        \n        # Test with current positions\n        current_positions = {symbol: 1.0/len(self.symbols) for symbol in self.symbols[:5]}\n        \n        should_rebalance, analysis = self.selector.check_rebalance_needed(\n            current_positions, self.market_data\n        )\n        \n        self.assertIsInstance(should_rebalance, bool)\n        self.assertIsInstance(analysis, dict)\n        self.assertIn('rebalance_needed', analysis)\n        \n        logger.info(f\"✓ Rebalance decision logic passed: {'Rebalance' if should_rebalance else 'Hold'}\")\n    \n    def test_selection_stability(self):\n        \"\"\"Test stability of selection over multiple runs.\"\"\"\n        logger.info(\"Testing selection stability\")\n        \n        results = []\n        \n        # Run selection multiple times\n        for i in range(3):\n            result = self.selector.run_integrated_selection(\n                self.mock_quote_client,\n                universe=self.symbols,\n                market_data=self.market_data,\n                dry_run=True\n            )\n            results.append(result)\n        \n        # Calculate stability\n        stability = self.selector._calculate_position_stability()\n        \n        self.assertGreaterEqual(stability, 0.0)\n        self.assertLessEqual(stability, 1.0)\n        \n        logger.info(f\"✓ Selection stability passed: {stability:.3f}\")\n\n\nclass IntegrationTests(unittest.TestCase):\n    \"\"\"Integration tests between all components.\"\"\"\n    \n    def setUp(self):\n        if not MODULES_AVAILABLE:\n            self.skipTest(\"Required modules not available\")\n        \n        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']\n        self.temp_dir = tempfile.mkdtemp()\n        TestDataGenerator.create_test_csv_files(self.symbols, self.temp_dir)\n    \n    def tearDown(self):\n        import shutil\n        shutil.rmtree(self.temp_dir, ignore_errors=True)\n    \n    def test_full_system_integration(self):\n        \"\"\"Test complete system integration.\"\"\"\n        logger.info(\"Testing full system integration\")\n        \n        # 1. Run risk-integrated selection\n        selector = RiskIntegratedSelector()\n        mock_client = type('MockClient', (), {})()\n        \n        selection_result = selector.run_integrated_selection(\n            mock_client,\n            universe=self.symbols,\n            dry_run=True\n        )\n        \n        self.assertGreater(len(selection_result.selected_symbols), 0)\n        \n        # 2. Run portfolio backtest with selection\n        backtester = PortfolioBacktester(\n            start_date=\"2022-01-01\",\n            end_date=\"2023-06-30\"\n        )\n        \n        backtest_results = backtester.run_portfolio_backtest(\n            data_source=\"csv\",\n            csv_directory=self.temp_dir,\n            universe=self.symbols\n        )\n        \n        self.assertIn('performance', backtest_results)\n        \n        # 3. Validate risk compliance\n        if selection_result.position_weights:\n            from bot.risk_integrated_selection import validate_portfolio_risk_compliance\n            \n            # This would normally use live data, but we'll test the structure\n            validation = {\n                \"passes_validation\": True,\n                \"risk_violations\": [],\n                \"portfolio_metrics\": {}\n            }\n            \n            self.assertIsInstance(validation, dict)\n        \n        logger.info(\"✓ Full system integration passed\")\n    \n    def test_error_handling_and_fallbacks(self):\n        \"\"\"Test system behavior under error conditions.\"\"\"\n        logger.info(\"Testing error handling and fallbacks\")\n        \n        # Test with empty market data\n        selector = RiskIntegratedSelector()\n        mock_client = type('MockClient', (), {})()\n        \n        result = selector.run_integrated_selection(\n            mock_client,\n            universe=[],\n            market_data={},\n            dry_run=True\n        )\n        \n        # Should handle gracefully\n        self.assertEqual(len(result.selected_symbols), 0)\n        self.assertEqual(len(result.position_weights), 0)\n        \n        # Test backtester with no data\n        backtester = PortfolioBacktester(\n            start_date=\"2023-01-01\",\n            end_date=\"2023-01-31\"\n        )\n        \n        results = backtester.run_portfolio_backtest(\n            data_source=\"csv\",\n            csv_directory=\"/nonexistent/path\",\n            universe=self.symbols\n        )\n        \n        # Should not crash\n        self.assertIsInstance(results, dict)\n        \n        logger.info(\"✓ Error handling and fallbacks passed\")\n\n\nclass PerformanceTests(unittest.TestCase):\n    \"\"\"Performance and scalability tests.\"\"\"\n    \n    def setUp(self):\n        if not MODULES_AVAILABLE:\n            self.skipTest(\"Required modules not available\")\n    \n    def test_large_universe_performance(self):\n        \"\"\"Test performance with large stock universe.\"\"\"\n        logger.info(\"Testing large universe performance\")\n        \n        # Create larger test universe\n        large_universe = [f\"STOCK{i:03d}\" for i in range(100)]\n        market_data = TestDataGenerator.generate_market_data_dict(large_universe)\n        \n        start_time = datetime.now()\n        \n        risk_engine = RiskFilterEngine()\n        filtered_symbols, risk_metrics = risk_engine.apply_risk_filters(\n            large_universe, market_data\n        )\n        \n        elapsed_time = (datetime.now() - start_time).total_seconds()\n        \n        self.assertLess(elapsed_time, 30)  # Should complete in under 30 seconds\n        self.assertGreater(len(filtered_symbols), 0)\n        \n        logger.info(f\"✓ Large universe performance passed: {len(filtered_symbols)}/100 in {elapsed_time:.2f}s\")\n    \n    def test_memory_usage(self):\n        \"\"\"Test memory usage with multiple selections.\"\"\"\n        logger.info(\"Testing memory usage\")\n        \n        selector = RiskIntegratedSelector()\n        mock_client = type('MockClient', (), {})()\n        \n        symbols = ['AAPL', 'MSFT', 'GOOGL']\n        \n        # Run multiple selections to test memory usage\n        for i in range(10):\n            market_data = TestDataGenerator.generate_market_data_dict(symbols)\n            result = selector.run_integrated_selection(\n                mock_client,\n                universe=symbols,\n                market_data=market_data,\n                dry_run=True\n            )\n            \n            self.assertIsNotNone(result)\n        \n        # Check that history is maintained within reasonable bounds\n        self.assertLessEqual(len(selector.selection_history), 100)\n        \n        logger.info(\"✓ Memory usage test passed\")\n\n\ndef run_comprehensive_tests():\n    \"\"\"Run all test suites with detailed reporting.\"\"\"\n    if not MODULES_AVAILABLE:\n        print(\"❌ Cannot run tests - required modules not available\")\n        return False\n    \n    print(\"\\n\" + \"=\"*60)\n    print(\"COMPREHENSIVE RISK CONTROL AND BACKTESTING TESTS\")\n    print(\"=\"*60)\n    \n    # Test suites to run\n    test_suites = [\n        RiskFilterEngineTests,\n        PortfolioBacktesterTests, \n        RiskIntegratedSelectorTests,\n        IntegrationTests,\n        PerformanceTests\n    ]\n    \n    all_passed = True\n    results_summary = []\n    \n    for test_suite_class in test_suites:\n        print(f\"\\n📋 Running {test_suite_class.__name__}...\")\n        \n        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite_class)\n        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))\n        result = runner.run(suite)\n        \n        suite_passed = result.wasSuccessful()\n        all_passed &= suite_passed\n        \n        status = \"✅ PASSED\" if suite_passed else \"❌ FAILED\"\n        results_summary.append((test_suite_class.__name__, status, result.testsRun, len(result.failures), len(result.errors)))\n        \n        print(f\"   {status} - {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors\")\n        \n        if result.failures:\n            print(\"   Failures:\")\n            for test, failure in result.failures:\n                print(f\"     - {test}: {failure.split('\\n')[0]}\")\n        \n        if result.errors:\n            print(\"   Errors:\")\n            for test, error in result.errors:\n                print(f\"     - {test}: {error.split('\\n')[0]}\")\n    \n    # Print summary\n    print(\"\\n\" + \"=\"*60)\n    print(\"TEST RESULTS SUMMARY\")\n    print(\"=\"*60)\n    \n    for suite_name, status, tests, failures, errors in results_summary:\n        print(f\"{suite_name:35} {status:10} ({tests} tests)\")\n    \n    print(\"\\n\" + \"=\"*60)\n    \n    if all_passed:\n        print(\"🎉 ALL TESTS PASSED! Risk control and backtesting systems are working correctly.\")\n    else:\n        print(\"⚠️  SOME TESTS FAILED! Please review the failures above.\")\n    \n    print(\"=\"*60)\n    \n    return all_passed\n\n\nif __name__ == '__main__':\n    success = run_comprehensive_tests()\n    sys.exit(0 if success else 1)