#!/usr/bin/env python3
"""
Robust Execution Trading Adapter

Integrates execution robustness framework into the trading system
to provide reliable order execution with intelligent retry mechanisms,
circuit breakers, and error handling.

This adapter wraps trading operations with robustness features.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bot"))
sys.path.insert(0, str(project_root / "improvement"))

from execution_robustness.execution_robustness_framework import (
    ExecutionRobustnessFramework,
    CircuitBreakerConfig,
    robust_execution,
    get_global_framework
)

logger = logging.getLogger(__name__)


@dataclass
class RobustExecutionConfig:
    """Configuration for robust execution"""
    max_retries_trading: int = 3  # Maximum retries for trading operations
    max_retries_data: int = 2     # Maximum retries for data operations
    trading_timeout: float = 30.0  # Timeout for trading operations
    data_timeout: float = 15.0     # Timeout for data operations
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


class RobustExecutionAdapter:
    """
    Robust execution adapter for trading operations

    Provides:
    - Reliable order execution with retries
    - Circuit breaker protection for external services
    - Performance monitoring and error tracking
    - Graceful degradation for non-critical operations
    """

    def __init__(self,
                 auto_trading_engine=None,
                 config: Optional[RobustExecutionConfig] = None,
                 enable_monitoring: bool = True):
        """
        Initialize robust execution adapter

        Args:
            auto_trading_engine: Existing AutoTradingEngine instance
            config: Robustness configuration
            enable_monitoring: Enable performance monitoring
        """
        self.auto_trading_engine = auto_trading_engine
        self.config = config or RobustExecutionConfig()
        self.enable_monitoring = enable_monitoring

        # Get or create global framework
        self.framework = get_global_framework()

        # Setup circuit breakers
        if self.config.enable_circuit_breakers:
            self._setup_circuit_breakers()

        logger.info("Robust Execution Adapter initialized")

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for different services"""
        # Tiger API circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            timeout_duration=self.config.circuit_breaker_timeout
        )
        self.framework._get_circuit_breaker("tiger_api").config = cb_config

        # Data source circuit breaker
        self.framework._get_circuit_breaker("data_source").config = cb_config

        # Market data circuit breaker
        self.framework._get_circuit_breaker("market_data").config = cb_config

    def execute_trade_robust(self,
                           trade_function: Callable,
                           *args,
                           order_type: str = "market",
                           priority: int = 3,  # High priority for trades
                           **kwargs) -> Any:
        """
        Execute trade with robustness features

        Args:
            trade_function: Trading function to execute
            *args: Function arguments
            order_type: Type of order (market, limit, stop)
            priority: Execution priority (1-5, higher = more important)
            **kwargs: Function keyword arguments

        Returns:
            Trade execution result
        """
        try:
            return self.framework.execute_robust(
                trade_function,
                *args,
                max_retries=self.config.max_retries_trading,
                timeout=self.config.trading_timeout,
                circuit_breaker="tiger_api",
                priority=priority,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Robust trade execution failed: {e}")
            # Log trade failure for analysis
            self._log_trade_failure(trade_function.__name__ if hasattr(trade_function, '__name__') else str(trade_function), e)
            raise

    def fetch_data_robust(self,
                         data_function: Callable,
                         *args,
                         data_type: str = "market_data",
                         priority: int = 2,  # Medium priority for data
                         **kwargs) -> Any:
        """
        Fetch data with robustness features

        Args:
            data_function: Data fetching function
            *args: Function arguments
            data_type: Type of data being fetched
            priority: Execution priority
            **kwargs: Function keyword arguments

        Returns:
            Data fetch result
        """
        try:
            return self.framework.execute_robust(
                data_function,
                *args,
                max_retries=self.config.max_retries_data,
                timeout=self.config.data_timeout,
                circuit_breaker="data_source",
                priority=priority,
                **kwargs
            )

        except Exception as e:
            logger.warning(f"Robust data fetch failed: {e}")
            # For data operations, we might return cached data or default values
            return self._handle_data_failure(data_function, data_type, e)

    def execute_trading_operation_async(self,
                                      operation: Callable,
                                      callback: Optional[Callable] = None,
                                      **kwargs) -> str:
        """
        Execute trading operation asynchronously

        Args:
            operation: Trading operation to execute
            callback: Optional completion callback
            **kwargs: Operation parameters

        Returns:
            Task ID for tracking
        """
        return self.framework.execute_async(
            operation,
            callback=callback,
            max_retries=self.config.max_retries_trading,
            timeout=self.config.trading_timeout,
            **kwargs
        )

    def place_order_robust(self,
                          symbol: str,
                          quantity: int,
                          order_type: str = "market",
                          price: Optional[float] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Place order with robust execution

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            order_type: Order type (market, limit, stop)
            price: Order price (for limit orders)
            **kwargs: Additional order parameters

        Returns:
            Order execution result
        """
        def _place_order():
            if self.auto_trading_engine and hasattr(self.auto_trading_engine, 'place_order'):
                return self.auto_trading_engine.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                    **kwargs
                )
            else:
                # Simulate order placement for testing
                return {
                    'order_id': f"test_{symbol}_{datetime.now().timestamp()}",
                    'symbol': symbol,
                    'quantity': quantity,
                    'order_type': order_type,
                    'price': price,
                    'status': 'filled',
                    'timestamp': datetime.now()
                }

        return self.execute_trade_robust(
            _place_order,
            order_type=order_type,
            priority=5  # Highest priority for actual trades
        )

    def get_market_data_robust(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market data with robust fetching

        Args:
            symbols: List of symbols to fetch

        Returns:
            Market data dictionary
        """
        def _get_market_data():
            if self.auto_trading_engine and hasattr(self.auto_trading_engine, 'get_market_data'):
                return self.auto_trading_engine.get_market_data(symbols)
            else:
                # Simulate market data for testing
                import random
                data = {}
                for symbol in symbols:
                    base_price = hash(symbol) % 1000 + 50  # Reproducible "random" price
                    data[symbol] = {
                        'price': base_price + random.uniform(-5, 5),
                        'bid': base_price - 0.5,
                        'ask': base_price + 0.5,
                        'volume': random.randint(10000, 1000000),
                        'timestamp': datetime.now()
                    }
                return data

        return self.fetch_data_robust(
            _get_market_data,
            data_type="market_data",
            priority=2
        )

    def get_account_info_robust(self) -> Dict[str, Any]:
        """
        Get account information with robust fetching

        Returns:
            Account information
        """
        def _get_account_info():
            if self.auto_trading_engine and hasattr(self.auto_trading_engine, 'get_account_info'):
                return self.auto_trading_engine.get_account_info()
            else:
                # Simulate account info for testing
                return {
                    'account_id': 'test_account',
                    'buying_power': 50000.0,
                    'total_equity': 75000.0,
                    'day_trade_buying_power': 200000.0,
                    'timestamp': datetime.now()
                }

        return self.fetch_data_robust(
            _get_account_info,
            data_type="account_info",
            priority=2
        )

    def analyze_trading_opportunities_robust(self,
                                           current_positions: List[Dict],
                                           recommended_positions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze trading opportunities with robust execution

        Args:
            current_positions: Current portfolio positions
            recommended_positions: AI-recommended positions

        Returns:
            Trading analysis results
        """
        def _analyze_opportunities():
            if self.auto_trading_engine and hasattr(self.auto_trading_engine, 'analyze_trading_opportunities'):
                return self.auto_trading_engine.analyze_trading_opportunities(
                    current_positions, recommended_positions
                )
            else:
                # Default analysis logic
                return self._default_trading_analysis(current_positions, recommended_positions)

        try:
            return self.framework.execute_robust(
                _analyze_opportunities,
                max_retries=2,  # Lower retries for analysis
                timeout=45.0,   # Longer timeout for complex analysis
                priority=2
            )
        except Exception as e:
            logger.error(f"Trading analysis failed: {e}")
            # Return safe default analysis
            return {
                'trading_signals': {'buy': [], 'sell': [], 'hold': current_positions},
                'execution_results': [],
                'trading_summary': {
                    'robust_execution': True,
                    'analysis_failed': True,
                    'error': str(e)
                }
            }

    def _handle_data_failure(self, data_function: Callable, data_type: str, error: Exception) -> Any:
        """Handle data fetching failures with graceful degradation"""
        logger.warning(f"Data failure for {data_type}: {error}")

        # Return appropriate default/cached data based on type
        if data_type == "market_data":
            return {}  # Empty market data
        elif data_type == "account_info":
            return {
                'account_id': 'unknown',
                'buying_power': 0.0,
                'total_equity': 0.0,
                'error': 'Data unavailable'
            }
        else:
            return None

    def _log_trade_failure(self, operation_name: str, error: Exception):
        """Log trade failure for analysis"""
        failure_log = {
            'timestamp': datetime.now(),
            'operation': operation_name,
            'error': str(error),
            'error_type': type(error).__name__
        }
        logger.error(f"Trade failure logged: {failure_log}")

    def _default_trading_analysis(self, current_positions: List[Dict], recommended_positions: List[Dict]) -> Dict[str, Any]:
        """Default trading analysis when engine is unavailable"""
        return {
            'trading_signals': {
                'buy': [pos for pos in recommended_positions if pos.get('action') in ['buy', 'strong_buy']],
                'sell': [],
                'hold': current_positions
            },
            'execution_results': [],
            'trading_summary': {
                'robust_execution': True,
                'default_analysis': True,
                'total_signals': len(recommended_positions)
            }
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        framework_stats = self.framework.get_performance_stats()

        # Add adapter-specific stats
        adapter_stats = {
            'config': {
                'max_retries_trading': self.config.max_retries_trading,
                'max_retries_data': self.config.max_retries_data,
                'trading_timeout': self.config.trading_timeout,
                'data_timeout': self.config.data_timeout,
                'circuit_breakers_enabled': self.config.enable_circuit_breakers
            },
            'adapter_info': {
                'monitoring_enabled': self.enable_monitoring,
                'framework_running': self.framework._running
            }
        }

        return {
            **framework_stats,
            'adapter_stats': adapter_stats
        }

    def reset_circuit_breakers(self):
        """Reset all circuit breakers to closed state"""
        for breaker in self.framework.circuit_breakers.values():
            breaker.state = breaker.state.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
        logger.info("All circuit breakers reset")

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        stats = self.get_execution_stats()

        # Analyze health
        circuit_breakers = stats.get('circuit_breakers', {})
        healthy_breakers = sum(1 for cb in circuit_breakers.values() if cb['state'] == 'closed')
        total_breakers = len(circuit_breakers)

        error_stats = stats.get('error_stats', {})
        total_errors = sum(error_stats.values())

        health_score = 100.0
        if total_breakers > 0:
            health_score *= (healthy_breakers / total_breakers)

        # Reduce score based on recent errors
        if total_errors > 10:
            health_score *= 0.8
        elif total_errors > 5:
            health_score *= 0.9

        status = "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy"

        return {
            'overall_status': status,
            'health_score': health_score,
            'circuit_breakers': {
                'healthy': healthy_breakers,
                'total': total_breakers
            },
            'error_summary': {
                'total_errors': total_errors,
                'error_breakdown': error_stats
            },
            'framework_status': {
                'running': self.framework._running,
                'active_tasks': len([t for t in self.framework.tasks.values() if t.status.value == 'pending'])
            }
        }


def create_robust_execution_engine(auto_trading_engine=None,
                                 max_retries_trading: int = 3,
                                 max_retries_data: int = 2,
                                 enable_circuit_breakers: bool = True) -> RobustExecutionAdapter:
    """
    Factory function to create robust execution adapter

    Args:
        auto_trading_engine: Optional existing AutoTradingEngine
        max_retries_trading: Maximum retries for trading operations
        max_retries_data: Maximum retries for data operations
        enable_circuit_breakers: Enable circuit breaker protection

    Returns:
        Configured RobustExecutionAdapter
    """
    config = RobustExecutionConfig(
        max_retries_trading=max_retries_trading,
        max_retries_data=max_retries_data,
        enable_circuit_breakers=enable_circuit_breakers
    )

    return RobustExecutionAdapter(
        auto_trading_engine=auto_trading_engine,
        config=config,
        enable_monitoring=True
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    import random

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Robust Execution Adapter")
    print("=" * 40)

    # Create adapter
    adapter = create_robust_execution_engine()

    try:
        # Test market data fetching
        print("Testing market data fetching...")
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        market_data = adapter.get_market_data_robust(symbols)
        print(f"Market data fetched for {len(market_data)} symbols")

        # Test account info
        print("\nTesting account info...")
        account_info = adapter.get_account_info_robust()
        print(f"Account equity: ${account_info['total_equity']:,.2f}")

        # Test order placement
        print("\nTesting order placement...")
        order_result = adapter.place_order_robust(
            symbol='AAPL',
            quantity=100,
            order_type='market'
        )
        print(f"Order placed: {order_result['order_id']}")

        # Test trading analysis
        print("\nTesting trading analysis...")
        current_positions = []
        recommended_positions = [
            {'symbol': 'AAPL', 'qty': 100, 'action': 'buy', 'price': 150.0},
            {'symbol': 'GOOGL', 'qty': 10, 'action': 'buy', 'price': 2500.0}
        ]

        analysis = adapter.analyze_trading_opportunities_robust(
            current_positions, recommended_positions
        )
        print(f"Analysis completed - Buy signals: {len(analysis['trading_signals']['buy'])}")

        # Test async execution
        print("\nTesting async execution...")
        def slow_operation():
            time.sleep(1)
            return f"Async result at {datetime.now()}"

        task_id = adapter.execute_trading_operation_async(
            slow_operation,
            callback=lambda tid, result, error: print(f"Async completed: {result or error}")
        )
        print(f"Async task started: {task_id}")

        # Wait for async completion
        time.sleep(2)

        # Get performance stats
        print("\nExecution Statistics:")
        stats = adapter.get_execution_stats()
        print(f"Framework running: {stats['framework_info']['running']}")

        if stats['circuit_breakers']:
            print("Circuit Breaker Status:")
            for name, cb_stats in stats['circuit_breakers'].items():
                print(f"  {name}: {cb_stats['state']}")

        # Get health status
        health = adapter.get_health_status()
        print(f"\nSystem Health: {health['overall_status']} ({health['health_score']:.1f}%)")

        print("\nRobust Execution Test Completed Successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        adapter.framework.stop()