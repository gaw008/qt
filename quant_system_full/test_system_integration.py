#!/usr/bin/env python3
"""
Comprehensive System Integration Test for Intelligent Stock Selection Trading System

This test validates the end-to-end functionality of the complete quantitative trading system,
including intelligent stock selection, risk management, portfolio optimization, and real-time monitoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
from typing import Dict, List, Any, Optional

# Import all system components
try:
    from bot.sector_manager import SectorManager
    from bot.stock_screener import screen_stocks, screen_sector_stocks
    from bot.scoring_engine import MultiFactorScoringEngine
    from bot.selection_strategies.base_strategy import SelectionStrategy
    from bot.selection_strategies.value_momentum import ValueMomentumStrategy
    from bot.selection_strategies.technical_breakout import TechnicalBreakoutStrategy
    from bot.market_time import MarketTimeManager, MarketPhase
    from bot.portfolio import MultiStockPortfolio
    from bot.execution import ExecutionEngine
    from bot.realtime_monitor import RealTimeMonitor
    from bot.report_generator import ReportGenerator
    from bot.performance_optimizer import PerformanceOptimizer, cached, rate_limit
    SYSTEM_IMPORTS = True
except ImportError as e:
    print(f"Warning: Could not import all system components: {e}")
    SYSTEM_IMPORTS = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class SystemIntegrationTest:
    """Comprehensive system integration test suite."""
    
    def __init__(self):
        self.test_results = []
        self.sector_manager = None
        self.scoring_engine = None
        self.portfolio = None
        self.execution_engine = None
        self.monitor = None
        self.report_generator = None
        self.optimizer = None
        
        # Test configuration
        self.test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        self.test_cash = 100000.0
        
        print("🚀 Initializing Comprehensive System Integration Test")
        print("=" * 70)
    
    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        if not SYSTEM_IMPORTS:
            print("❌ Cannot run tests: System imports failed")
            return False
        
        test_methods = [
            self.test_system_initialization,
            self.test_sector_management,
            self.test_stock_selection_pipeline,
            self.test_multi_factor_scoring,
            self.test_selection_strategies,
            self.test_market_time_detection,
            self.test_portfolio_management,
            self.test_risk_management,
            self.test_execution_engine,
            self.test_real_time_monitoring,
            self.test_performance_optimization,
            self.test_report_generation,
            self.test_end_to_end_workflow
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                print(f"\n📋 Running {test_method.__name__}...")
                success = test_method()
                if success:
                    print(f"✅ {test_method.__name__} PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_method.__name__} FAILED")
                self.test_results.append({
                    'test': test_method.__name__,
                    'status': 'PASSED' if success else 'FAILED',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"💥 {test_method.__name__} ERROR: {e}")
                self.test_results.append({
                    'test': test_method.__name__,
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Print summary
        print("\n" + "=" * 70)
        print("🏁 TEST SUMMARY")
        print("=" * 70)
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        print(f"📊 Success Rate: {passed/total*100:.1f}%")
        
        # Save test results
        self.save_test_results()
        
        return passed == total
    
    def test_system_initialization(self) -> bool:
        """Test system component initialization."""
        try:
            # Initialize core components
            self.sector_manager = SectorManager()
            self.scoring_engine = MultiFactorScoringEngine()
            self.portfolio = MultiStockPortfolio(initial_cash=self.test_cash)
            self.execution_engine = ExecutionEngine(self.portfolio)
            self.monitor = RealTimeMonitor()
            self.report_generator = ReportGenerator()
            self.optimizer = PerformanceOptimizer()
            
            print("   📦 All system components initialized successfully")
            return True
        except Exception as e:
            print(f"   💥 Initialization failed: {e}")
            return False
    
    def test_sector_management(self) -> bool:
        """Test sector management functionality."""
        try:
            # Test sector listing
            sectors = self.sector_manager.get_available_sectors()
            assert len(sectors) > 0, "No sectors available"
            print(f"   📊 Found {len(sectors)} sectors: {list(sectors.keys())}")
            
            # Test getting stocks from a sector
            tech_stocks = self.sector_manager.get_sector_stocks('Technology')
            assert len(tech_stocks) > 0, "No technology stocks found"
            print(f"   💻 Technology sector has {len(tech_stocks)} stocks")
            
            # Test stock validation
            valid_symbols = self.sector_manager.validate_symbols(self.test_symbols[:3])
            print(f"   ✅ Validated {len(valid_symbols)} symbols")
            
            return True
        except Exception as e:
            print(f"   💥 Sector management test failed: {e}")
            return False
    
    def test_stock_selection_pipeline(self) -> bool:
        """Test the complete stock selection pipeline."""
        try:
            # Test basic stock screening
            results = screen_stocks(self.test_symbols, top_n=5)
            assert 'selected_stocks' in results, "No selection results"
            assert len(results['selected_stocks']) > 0, "No stocks selected"
            print(f"   🎯 Selected {len(results['selected_stocks'])} stocks from {len(self.test_symbols)}")
            
            # Test sector-based screening
            sector_results = screen_sector_stocks('Technology', top_n=10)
            assert 'selected_stocks' in sector_results, "No sector selection results"
            print(f"   🏭 Selected {len(sector_results['selected_stocks'])} technology stocks")
            
            return True
        except Exception as e:
            print(f"   💥 Stock selection pipeline test failed: {e}")
            return False
    
    def test_multi_factor_scoring(self) -> bool:
        """Test multi-factor scoring system."""
        try:
            # Generate sample data for scoring
            sample_data = self.generate_sample_data(self.test_symbols[:5])
            
            # Test factor scoring
            scores = self.scoring_engine.calculate_factor_scores(sample_data)
            assert not scores.empty, "No factor scores calculated"
            print(f"   📊 Calculated factor scores for {len(scores)} stocks")
            
            # Test composite scoring
            composite_scores = self.scoring_engine.calculate_composite_scores(sample_data)
            assert not composite_scores.empty, "No composite scores calculated"
            print(f"   🎯 Calculated composite scores: avg={composite_scores['composite_score'].mean():.1f}")
            
            return True
        except Exception as e:
            print(f"   💥 Multi-factor scoring test failed: {e}")
            return False
    
    def test_selection_strategies(self) -> bool:
        """Test selection strategies."""
        try:
            sample_data = self.generate_sample_data(self.test_symbols)
            
            # Test Value Momentum Strategy
            value_momentum = ValueMomentumStrategy()
            vm_results = value_momentum.select_stocks(sample_data, top_n=5)
            assert len(vm_results['selected_stocks']) > 0, "Value momentum strategy failed"
            print(f"   📈 Value momentum selected {len(vm_results['selected_stocks'])} stocks")
            
            # Test Technical Breakout Strategy  
            tech_breakout = TechnicalBreakoutStrategy()
            tb_results = tech_breakout.select_stocks(sample_data, top_n=5)
            assert len(tb_results['selected_stocks']) > 0, "Technical breakout strategy failed"
            print(f"   📊 Technical breakout selected {len(tb_results['selected_stocks'])} stocks")
            
            return True
        except Exception as e:
            print(f"   💥 Selection strategies test failed: {e}")
            return False
    
    def test_market_time_detection(self) -> bool:
        """Test market time detection."""
        try:
            market_manager = MarketTimeManager()
            
            # Test current market phase
            phase = market_manager.get_current_phase()
            print(f"   🕐 Current market phase: {phase.value}")
            
            # Test market schedule
            is_trading_day = market_manager.is_trading_day()
            print(f"   📅 Is trading day: {is_trading_day}")
            
            # Test hours until next phase
            hours_until_next = market_manager.hours_until_next_phase()
            print(f"   ⏰ Hours until next phase: {hours_until_next:.1f}")
            
            return True
        except Exception as e:
            print(f"   💥 Market time detection test failed: {e}")
            return False
    
    def test_portfolio_management(self) -> bool:
        """Test portfolio management functionality."""
        try:
            # Test adding positions
            for i, symbol in enumerate(self.test_symbols[:3]):
                shares = 10 + i * 5
                price = 100 + i * 10
                self.portfolio.add_position(symbol, shares, price)
            
            positions = self.portfolio.get_positions()
            assert len(positions) == 3, "Portfolio positions not added correctly"
            print(f"   💼 Portfolio has {len(positions)} positions")
            
            # Test portfolio value calculation
            total_value = self.portfolio.get_total_value()
            print(f"   💰 Total portfolio value: ${total_value:,.2f}")
            
            # Test position summary
            summary = self.portfolio.get_portfolio_summary()
            print(f"   📊 Portfolio P&L: ${summary['unrealized_pnl']:,.2f}")
            
            return True
        except Exception as e:
            print(f"   💥 Portfolio management test failed: {e}")
            return False
    
    def test_risk_management(self) -> bool:
        """Test risk management features."""
        try:
            # Test portfolio risk metrics
            if hasattr(self.portfolio, 'calculate_portfolio_risk'):
                risk_metrics = self.portfolio.calculate_portfolio_risk()
                print(f"   ⚖️ Portfolio risk calculated: keys={list(risk_metrics.keys())}")
            
            # Test position sizing constraints
            max_position_value = self.portfolio.cash * 0.1  # 10% max position
            assert max_position_value > 0, "Position sizing constraint failed"
            print(f"   📏 Max position size: ${max_position_value:,.2f}")
            
            return True
        except Exception as e:
            print(f"   💥 Risk management test failed: {e}")
            return False
    
    def test_execution_engine(self) -> bool:
        """Test trade execution functionality."""
        try:
            # Test order creation
            order_id = self.execution_engine.place_order('AAPL', 'buy', 5, 150.0)
            assert order_id is not None, "Order placement failed"
            print(f"   📋 Order placed with ID: {order_id}")
            
            # Test order status
            orders = self.execution_engine.get_orders()
            assert len(orders) > 0, "No orders found"
            print(f"   📊 Found {len(orders)} orders")
            
            return True
        except Exception as e:
            print(f"   💥 Execution engine test failed: {e}")
            return False
    
    def test_real_time_monitoring(self) -> bool:
        """Test real-time monitoring system."""
        try:
            # Test monitor initialization
            assert self.monitor is not None, "Monitor not initialized"
            print("   📺 Real-time monitor initialized")
            
            # Test adding alert
            if hasattr(self.monitor, 'add_alert'):
                alert_id = self.monitor.add_alert(
                    'PORTFOLIO_DRAWDOWN',
                    'Portfolio drawdown exceeds 5%',
                    'medium'
                )
                print(f"   🚨 Alert added: {alert_id}")
            
            return True
        except Exception as e:
            print(f"   💥 Real-time monitoring test failed: {e}")
            return False
    
    def test_performance_optimization(self) -> bool:
        """Test performance optimization features."""
        try:
            # Test caching
            @cached(ttl=60)
            def expensive_calc(x):
                time.sleep(0.01)  # Simulate work
                return x * x
            
            # First call (cache miss)
            start = time.time()
            result1 = expensive_calc(10)
            time1 = time.time() - start
            
            # Second call (cache hit)
            start = time.time()
            result2 = expensive_calc(10)
            time2 = time.time() - start
            
            assert result1 == result2, "Cached function returned different results"
            assert time2 < time1, "Cache didn't improve performance"
            print(f"   💨 Caching improved performance by {time1/time2:.1f}x")
            
            # Test rate limiting
            @rate_limit(calls_per_second=5.0)
            def rate_limited_func():
                return "success"
            
            start = time.time()
            for _ in range(3):
                rate_limited_func()
            elapsed = time.time() - start
            print(f"   ⏱️ Rate limiting: 3 calls took {elapsed:.2f}s")
            
            return True
        except Exception as e:
            print(f"   💥 Performance optimization test failed: {e}")
            return False
    
    def test_report_generation(self) -> bool:
        """Test report generation functionality."""
        try:
            # Test daily selection report
            sample_selection = {
                'total_analyzed': 50,
                'selected_stocks': [
                    {'symbol': 'AAPL', 'score': 85.5, 'sector': 'Technology', 'price': 175.0},
                    {'symbol': 'GOOGL', 'score': 82.1, 'sector': 'Technology', 'price': 135.0}
                ]
            }
            
            report_path = self.report_generator.generate_daily_selection_report(sample_selection)
            assert os.path.exists(report_path), "Report file not created"
            print(f"   📄 Daily selection report generated: {os.path.basename(report_path)}")
            
            return True
        except Exception as e:
            print(f"   💥 Report generation test failed: {e}")
            return False
    
    def test_end_to_end_workflow(self) -> bool:
        """Test complete end-to-end workflow."""
        try:
            print("   🔄 Running end-to-end workflow simulation...")
            
            # Step 1: Market time check
            market_manager = MarketTimeManager()
            phase = market_manager.get_current_phase()
            print(f"      1. Market phase: {phase.value}")
            
            # Step 2: Stock selection
            selection_results = screen_stocks(self.test_symbols[:5], top_n=3)
            selected = selection_results['selected_stocks']
            print(f"      2. Selected {len(selected)} stocks")
            
            # Step 3: Portfolio update
            for stock in selected[:2]:  # Add top 2 stocks
                symbol = stock['symbol']
                shares = int(1000 / stock['price'])  # $1000 position
                self.portfolio.add_position(symbol, shares, stock['price'])
            
            positions = len(self.portfolio.get_positions())
            print(f"      3. Portfolio updated: {positions} positions")
            
            # Step 4: Risk check
            total_value = self.portfolio.get_total_value()
            cash_ratio = self.portfolio.cash / total_value if total_value > 0 else 1.0
            print(f"      4. Risk check: {cash_ratio:.1%} cash ratio")
            
            # Step 5: Performance summary
            perf_summary = self.optimizer.get_performance_summary()
            cache_hit_rate = perf_summary['metrics']['cache_hit_rate']
            print(f"      5. Performance: {cache_hit_rate:.1%} cache hit rate")
            
            print("   ✅ End-to-end workflow completed successfully")
            return True
            
        except Exception as e:
            print(f"   💥 End-to-end workflow test failed: {e}")
            return False
    
    def generate_sample_data(self, symbols: List[str], days: int = 100) -> Dict[str, pd.DataFrame]:
        """Generate sample OHLCV data for testing."""
        data = {}
        
        for symbol in symbols:
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.001, 0.02, days)  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))  # Price series
            
            # Create OHLCV data
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = prices * (1 + np.random.normal(0, 0.005, days))
            df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.01, days)))
            df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.01, days)))
            df['volume'] = np.random.lognormal(15, 0.5, days).astype(int)
            
            # Ensure positive prices
            df = df.abs()
            df.loc[df['low'] <= 0, 'low'] = 0.01
            
            data[symbol] = df
        
        return data
    
    def save_test_results(self):
        """Save test results to file."""
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'test_run': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['status'] == 'PASSED']),
                'failed': len([r for r in self.test_results if r['status'] == 'FAILED']),
                'errors': len([r for r in self.test_results if r['status'] == 'ERROR'])
            },
            'results': self.test_results,
            'system_info': {
                'python_version': sys.version,
                'test_symbols': self.test_symbols,
                'test_cash': self.test_cash
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Test results saved to: {results_file}")


def main():
    """Run the comprehensive system integration test."""
    print("🎯 INTELLIGENT STOCK SELECTION TRADING SYSTEM")
    print("🧪 COMPREHENSIVE INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run tests
    test_suite = SystemIntegrationTest()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the results above.")
        return 1


if __name__ == "__main__":
    exit(main())