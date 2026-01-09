#!/usr/bin/env python3
"""
Portfolio Management Integration Test Suite

This script validates the integration between the multi-stock portfolio management system,
execution engine, and real-time monitoring components.

Test Coverage:
- Multi-stock portfolio creation and management
- Dynamic capital allocation and rebalancing
- Order execution and position tracking
- Real-time monitoring and alerting
- Risk management and position limits
- Performance tracking and metrics
- Integration with data feeds and market timing
"""

import sys
import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Add bot directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

# Import the modules we're testing
from bot.portfolio import (
    MultiStockPortfolio, 
    AllocationMethod, 
    PositionType,
    Position
)
from bot.execution import (
    ExecutionEngine, 
    OrderSide, 
    OrderType, 
    ExecutionStrategy,
    create_execution_engine,
    create_batch_orders_from_rebalancing
)
from bot.realtime_monitor import (
    RealTimeMonitor,
    create_realtime_monitor,
    get_market_data_summary
)
from bot.config import SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioIntegrationTester:
    """Comprehensive integration testing for portfolio management system."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.test_results: List[Dict[str, Any]] = []
        self.portfolio: MultiStockPortfolio = None
        self.execution_engine: ExecutionEngine = None
        self.monitor: RealTimeMonitor = None
        
        # Test data
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.test_prices = {
            'AAPL': 150.0,
            'MSFT': 280.0,
            'GOOGL': 2500.0,
            'AMZN': 3200.0,
            'TSLA': 800.0
        }
        
        logger.info(f"[test] Initialized portfolio integration tester with ${initial_capital:,.2f}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("[test] Starting comprehensive portfolio integration tests...")
        
        test_results = {
            'start_time': datetime.now().isoformat(),
            'tests': [],
            'summary': {}
        }
        
        try:
            # Test 1: Portfolio Creation and Basic Operations
            result1 = self.test_portfolio_creation()
            test_results['tests'].append(result1)
            
            # Test 2: Multi-Stock Position Management
            result2 = self.test_multi_stock_positions()
            test_results['tests'].append(result2)
            
            # Test 3: Dynamic Capital Allocation
            result3 = self.test_capital_allocation()
            test_results['tests'].append(result3)
            
            # Test 4: Execution Engine Integration
            result4 = self.test_execution_integration()
            test_results['tests'].append(result4)
            
            # Test 5: Portfolio Rebalancing
            result5 = self.test_portfolio_rebalancing()
            test_results['tests'].append(result5)
            
            # Test 6: Real-Time Monitoring
            result6 = self.test_realtime_monitoring()
            test_results['tests'].append(result6)
            
            # Test 7: Risk Management
            result7 = self.test_risk_management()
            test_results['tests'].append(result7)
            
            # Test 8: Performance Tracking
            result8 = self.test_performance_tracking()
            test_results['tests'].append(result8)
            
            # Test 9: State Persistence
            result9 = self.test_state_persistence()
            test_results['tests'].append(result9)
            
            # Generate summary
            test_results['summary'] = self._generate_test_summary(test_results['tests'])
            test_results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"[test] Integration tests completed: {test_results['summary']}")
            
        except Exception as e:
            logger.error(f"[test] Integration test suite failed: {e}")
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.now().isoformat()
        
        return test_results
    
    def test_portfolio_creation(self) -> Dict[str, Any]:
        """Test portfolio creation and basic functionality."""
        test_name = "Portfolio Creation and Basic Operations"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            # Create portfolio with different allocation methods
            portfolios = {}
            
            for method in [AllocationMethod.EQUAL_WEIGHT, AllocationMethod.SCORE_WEIGHT, AllocationMethod.RISK_PARITY]:
                portfolio = MultiStockPortfolio(
                    initial_capital=self.initial_capital,
                    allocation_method=method,
                    max_position_weight=0.2,  # 20% max per position
                    max_positions=10
                )
                portfolios[method.value] = portfolio
            
            # Use equal weight portfolio for further tests
            self.portfolio = portfolios['equal_weight']
            
            # Test basic properties
            assert self.portfolio.get_total_value() == self.initial_capital
            assert self.portfolio.get_available_capital() == self.initial_capital
            assert self.portfolio.get_position_count() == 0
            assert self.portfolio.get_total_pnl() == 0.0
            
            logger.info(f"[test] ✓ Portfolio created successfully with ${self.initial_capital:,.2f}")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'portfolios_created': len(portfolios),
                    'initial_capital': self.initial_capital,
                    'total_value': self.portfolio.get_total_value(),
                    'available_capital': self.portfolio.get_available_capital()
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_multi_stock_positions(self) -> Dict[str, Any]:
        """Test adding and managing multiple stock positions."""
        test_name = "Multi-Stock Position Management"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            added_positions = []
            
            # Add positions for all test symbols
            for symbol, price in self.test_prices.items():
                quantity = int(5000 / price)  # $5k per position
                
                success = self.portfolio.add_position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    position_type=PositionType.LONG,
                    score=85.0,
                    sector="Technology" if symbol != "AMZN" else "Consumer"
                )
                
                if success:
                    added_positions.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'entry_price': price,
                        'market_value': quantity * price
                    })
            
            # Verify positions
            assert len(added_positions) == len(self.test_symbols)
            assert self.portfolio.get_position_count() == len(self.test_symbols)
            
            # Test position retrieval
            for symbol in self.test_symbols:
                position = self.portfolio.get_position(symbol)
                assert position is not None
                assert position.symbol == symbol
                assert position.current_price > 0
            
            # Update prices and check P&L
            updated_prices = {symbol: price * 1.05 for symbol, price in self.test_prices.items()}
            self.portfolio.update_all_prices(updated_prices)
            
            total_pnl = self.portfolio.get_total_pnl()
            assert total_pnl > 0  # Should be positive with 5% price increase
            
            logger.info(f"[test] ✓ Added {len(added_positions)} positions, P&L: ${total_pnl:,.2f}")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'positions_added': len(added_positions),
                    'total_portfolio_value': self.portfolio.get_total_value(),
                    'total_pnl': total_pnl,
                    'positions': added_positions
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_capital_allocation(self) -> Dict[str, Any]:
        """Test dynamic capital allocation strategies."""
        test_name = "Dynamic Capital Allocation"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Create mock selection results
            selection_results = [
                {'symbol': 'AAPL', 'score': 95.0, 'avg_score': 95.0},
                {'symbol': 'MSFT', 'score': 88.0, 'avg_score': 88.0},
                {'symbol': 'GOOGL', 'score': 82.0, 'avg_score': 82.0},
                {'symbol': 'TSLA', 'score': 75.0, 'avg_score': 75.0}
            ]
            
            allocation_results = {}
            
            # Test different allocation methods
            for method in [AllocationMethod.EQUAL_WEIGHT, AllocationMethod.SCORE_WEIGHT, AllocationMethod.RISK_PARITY]:
                test_portfolio = MultiStockPortfolio(
                    initial_capital=50000.0,
                    allocation_method=method,
                    max_position_weight=0.3
                )
                
                allocation = test_portfolio.calculate_target_allocation(
                    selection_results=selection_results,
                    available_capital=50000.0
                )
                
                allocation_results[method.value] = allocation
                
                # Verify allocation constraints
                total_weight = sum(allocation.values())
                max_weight = max(allocation.values()) if allocation else 0
                
                assert total_weight <= 1.0  # Should not exceed 100%
                assert max_weight <= test_portfolio.max_position_weight
                assert len(allocation) <= len(selection_results)
            
            logger.info(f"[test] ✓ Tested {len(allocation_results)} allocation methods")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'allocation_methods_tested': len(allocation_results),
                    'selection_results_count': len(selection_results),
                    'allocations': allocation_results
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_execution_integration(self) -> Dict[str, Any]:
        """Test execution engine integration with portfolio."""
        test_name = "Execution Engine Integration"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Create execution engine
            self.execution_engine = create_execution_engine(portfolio=self.portfolio)
            
            # Start execution engine
            self.execution_engine.start(num_workers=2)
            
            # Wait for startup
            time.sleep(1)
            
            # Test order submission
            order_ids = []
            
            for symbol, price in list(self.test_prices.items())[:3]:  # Test with 3 symbols
                order_id = self.execution_engine.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=10,  # Small quantity for testing
                    order_type=OrderType.MARKET,
                    limit_price=price,
                    execution_strategy=ExecutionStrategy.IMMEDIATE
                )
                
                if order_id:
                    order_ids.append(order_id)
            
            # Wait for order processing
            time.sleep(2)
            
            # Check order statuses
            filled_orders = 0
            for order_id in order_ids:
                order_state = self.execution_engine.get_order_status(order_id)
                if order_state and order_state.status.value in ['filled', 'partially_filled']:
                    filled_orders += 1
            
            # Get execution summary
            exec_summary = self.execution_engine.get_execution_summary()
            
            # Stop execution engine
            self.execution_engine.stop()
            
            logger.info(f"[test] ✓ Submitted {len(order_ids)} orders, {filled_orders} processed")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'orders_submitted': len(order_ids),
                    'orders_processed': filled_orders,
                    'execution_summary': exec_summary
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_portfolio_rebalancing(self) -> Dict[str, Any]:
        """Test portfolio rebalancing functionality."""
        test_name = "Portfolio Rebalancing"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Create target allocation
            target_allocation = {
                'AAPL': 0.25,
                'MSFT': 0.20,
                'GOOGL': 0.15,
                'TSLA': 0.10
            }
            
            current_prices = self.test_prices.copy()
            
            # Generate rebalancing orders
            rebalancing_orders = self.portfolio.rebalance_portfolio(
                target_allocation=target_allocation,
                current_prices=current_prices,
                force_rebalance=True
            )
            
            # Create batch orders for execution
            if rebalancing_orders:
                batch_orders = create_batch_orders_from_rebalancing(
                    rebalancing_orders=rebalancing_orders,
                    current_prices=current_prices,
                    execution_strategy=ExecutionStrategy.TWAP
                )
            else:
                batch_orders = []
            
            # Test rebalancing logic
            assert isinstance(rebalancing_orders, dict)
            
            total_target_weight = sum(target_allocation.values())
            assert total_target_weight <= 1.0
            
            logger.info(f"[test] ✓ Generated {len(rebalancing_orders)} rebalancing orders")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'target_allocation': target_allocation,
                    'rebalancing_orders_count': len(rebalancing_orders),
                    'batch_orders_count': len(batch_orders),
                    'total_target_weight': total_target_weight
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_realtime_monitoring(self) -> Dict[str, Any]:
        """Test real-time monitoring integration."""
        test_name = "Real-Time Monitoring"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Create real-time monitor
            self.monitor = create_realtime_monitor(
                portfolio=self.portfolio,
                execution_engine=self.execution_engine
            )
            
            # Start monitoring
            self.monitor.start()
            
            # Wait for startup
            time.sleep(2)
            
            # Check monitoring status
            status = self.monitor.get_monitoring_status()
            
            assert status['running'] is True
            assert status['state'] == 'running'
            assert 'subscribed_symbols' in status
            
            # Test market data summary
            market_summary = get_market_data_summary(self.monitor)
            
            # Test alert system (simulate an alert)
            from bot.realtime_monitor import Alert, AlertType, AlertSeverity
            
            test_alert = Alert(
                alert_id="TEST_ALERT_001",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.LOW,
                symbol="TEST",
                title="Test alert",
                message="This is a test alert"
            )
            
            self.monitor._add_alert(test_alert)
            
            # Get recent alerts
            recent_alerts = self.monitor.get_recent_alerts(hours=1)
            assert len(recent_alerts) >= 1
            
            # Stop monitoring
            self.monitor.stop()
            
            logger.info(f"[test] ✓ Real-time monitoring tested successfully")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'monitoring_status': status,
                    'market_summary': market_summary,
                    'alerts_generated': len(recent_alerts)
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_risk_management(self) -> Dict[str, Any]:
        """Test risk management features."""
        test_name = "Risk Management"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Update portfolio metrics
            self.portfolio.update_portfolio_metrics()
            
            # Get portfolio summary
            summary = self.portfolio.get_portfolio_summary()
            
            # Test risk metrics
            risk_metrics = summary.get('risk_metrics', {})
            
            assert 'total_exposure' in risk_metrics
            assert 'long_exposure' in risk_metrics
            assert 'net_exposure' in risk_metrics
            assert 'max_position_weight' in risk_metrics
            assert 'concentration_risk' in risk_metrics
            assert 'sector_concentration' in risk_metrics
            
            # Test position limits
            max_position_weight = risk_metrics.get('max_position_weight', 0)
            assert max_position_weight <= self.portfolio.max_position_weight
            
            # Test sector concentration
            sector_concentration = risk_metrics.get('sector_concentration', {})
            for sector, weight in sector_concentration.items():
                assert weight <= self.portfolio.max_sector_exposure
            
            # Simulate price drops to test risk alerts
            if self.monitor and self.monitor.running:
                # This would be tested in real market conditions
                pass
            
            logger.info(f"[test] ✓ Risk management features validated")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'risk_metrics': risk_metrics,
                    'max_position_weight': max_position_weight,
                    'sector_concentration': sector_concentration,
                    'portfolio_summary': {k: v for k, v in summary.items() if k != 'positions'}
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_performance_tracking(self) -> Dict[str, Any]:
        """Test performance tracking and metrics calculation."""
        test_name = "Performance Tracking"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Simulate some time passing and price changes
            for i in range(5):
                # Simulate price changes
                price_multiplier = 1.0 + (i * 0.01)  # 1% increase each iteration
                updated_prices = {symbol: price * price_multiplier for symbol, price in self.test_prices.items()}
                
                self.portfolio.update_all_prices(updated_prices)
                
                # Let the portfolio record performance history
                time.sleep(0.1)
            
            # Get performance metrics
            performance_metrics = self.portfolio.performance_metrics
            
            # Verify performance attributes
            assert hasattr(performance_metrics, 'total_return')
            assert hasattr(performance_metrics, 'volatility')
            assert hasattr(performance_metrics, 'sharpe_ratio')
            assert hasattr(performance_metrics, 'win_rate')
            
            # Test portfolio summary
            summary = self.portfolio.get_portfolio_summary()
            performance_data = summary.get('performance_metrics', {})
            
            assert 'total_return' in performance_data
            assert 'volatility' in performance_data
            assert 'last_updated' in performance_data
            
            # Test performance history
            history_count = len(self.portfolio.performance_history)
            assert history_count > 0
            
            logger.info(f"[test] ✓ Performance tracking with {history_count} history points")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'performance_metrics': performance_data,
                    'history_points': history_count,
                    'total_return': performance_metrics.total_return,
                    'volatility': performance_metrics.volatility,
                    'sharpe_ratio': performance_metrics.sharpe_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def test_state_persistence(self) -> Dict[str, Any]:
        """Test portfolio state saving and loading."""
        test_name = "State Persistence"
        logger.info(f"[test] Running: {test_name}")
        
        try:
            if not self.portfolio:
                raise ValueError("Portfolio not initialized")
            
            # Save portfolio state
            test_filepath = os.path.join(os.path.dirname(__file__), 'test_portfolio_state.json')
            
            save_success = self.portfolio.save_portfolio_state(test_filepath)
            assert save_success is True
            assert os.path.exists(test_filepath)
            
            # Create new portfolio and load state
            new_portfolio = MultiStockPortfolio(initial_capital=50000.0)  # Different initial capital
            
            load_success = new_portfolio.load_portfolio_state(test_filepath)
            assert load_success is True
            
            # Compare portfolios
            original_summary = self.portfolio.get_portfolio_summary()
            loaded_summary = new_portfolio.get_portfolio_summary()
            
            assert loaded_summary['portfolio_value'] == original_summary['portfolio_value']
            assert loaded_summary['positions_count'] == original_summary['positions_count']
            assert len(new_portfolio.positions) == len(self.portfolio.positions)
            
            # Verify positions match
            for symbol, position in self.portfolio.positions.items():
                loaded_position = new_portfolio.get_position(symbol)
                assert loaded_position is not None
                assert loaded_position.symbol == position.symbol
                assert loaded_position.quantity == position.quantity
                assert loaded_position.entry_price == position.entry_price
            
            # Clean up test file
            try:
                os.remove(test_filepath)
            except:
                pass
            
            logger.info(f"[test] ✓ State persistence with {len(self.portfolio.positions)} positions")
            
            return {
                'test_name': test_name,
                'status': 'passed',
                'details': {
                    'save_success': save_success,
                    'load_success': load_success,
                    'positions_saved': len(self.portfolio.positions),
                    'positions_loaded': len(new_portfolio.positions),
                    'test_filepath': test_filepath
                }
            }
            
        except Exception as e:
            logger.error(f"[test] ✗ {test_name} failed: {e}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_test_summary(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        total_tests = len(test_results)
        passed_tests = len([t for t in test_results if t.get('status') == 'passed'])
        failed_tests = len([t for t in test_results if t.get('status') == 'failed'])
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests) if total_tests > 0 else 0.0,
            'overall_status': 'passed' if failed_tests == 0 else 'failed'
        }


def main():
    """Run the portfolio integration tests."""
    print("=" * 80)
    print("Portfolio Management Integration Test Suite")
    print("=" * 80)
    
    # Create tester
    tester = PortfolioIntegrationTester(initial_capital=100000.0)
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Display results
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        summary = results.get('summary', {})
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed_tests', 0)}")
        print(f"Failed: {summary.get('failed_tests', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"Overall Status: {summary.get('overall_status', 'unknown').upper()}")
        
        # Detailed results
        print("\n" + "-" * 80)
        print("DETAILED RESULTS")
        print("-" * 80)
        
        for i, test in enumerate(results.get('tests', []), 1):
            status_symbol = "✓" if test.get('status') == 'passed' else "✗"
            print(f"{i:2d}. {status_symbol} {test.get('test_name', 'Unknown Test')}")
            
            if test.get('status') == 'failed':
                print(f"     Error: {test.get('error', 'Unknown error')}")
        
        # Save results to file
        results_file = os.path.join(os.path.dirname(__file__), 'portfolio_integration_test_results.json')
        
        try:
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📄 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to save results file: {e}")
        
        print("\n" + "=" * 80)
        
        # Return appropriate exit code
        return 0 if summary.get('overall_status') == 'passed' else 1
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        logger.error(f"Test suite error: {e}")
        return 1
    
    finally:
        # Cleanup
        try:
            if tester.execution_engine:
                tester.execution_engine.stop()
            if tester.monitor:
                tester.monitor.stop()
        except:
            pass


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)