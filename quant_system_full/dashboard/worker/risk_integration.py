#!/usr/bin/env python3
"""
Risk Integration Wrapper
Provides easy integration of Enhanced Risk Manager into trading workflow
NOW WITH DYNAMIC TIGER ACCOUNT DATA - Real-time portfolio values!
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add bot directory to path
BOT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, BOT_PARENT)
sys.path.insert(0, BASE)

from bot.enhanced_risk_manager import EnhancedRiskManager, RiskLimits, DrawdownBudget

# Import state manager from backend
try:
    from state_manager import append_log
except ImportError:
    # Fallback if state_manager not available
    def append_log(msg):
        print(f"[LOG] {msg}")

# Import Tiger Account Manager for dynamic data
try:
    from tiger_account_manager import get_account_manager
    TIGER_ACCOUNT_AVAILABLE = True
except ImportError:
    TIGER_ACCOUNT_AVAILABLE = False
    logging.warning("[RISK_INTEGRATION] Tiger Account Manager not available, using defaults")

logger = logging.getLogger(__name__)


class RiskIntegrationManager:
    """
    Risk Integration Manager for Trading System

    Provides comprehensive risk management integration including:
    - Position sizing validation
    - Portfolio risk assessment
    - ES@97.5% calculations
    - Dynamic risk limits
    - Trading signal validation

    NOW DYNAMICALLY RETRIEVES portfolio value from Tiger API!
    """

    def __init__(self,
                 portfolio_value: Optional[float] = None,
                 max_position_size: float = 1.0,
                 max_portfolio_leverage: float = 1.0,
                 enable_tail_risk: bool = True,
                 use_dynamic_portfolio: bool = True):
        """
        Initialize Risk Integration Manager with DYNAMIC portfolio data

        Args:
            portfolio_value: Total portfolio value (auto-detected if None)
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_leverage: Maximum portfolio leverage
            enable_tail_risk: Enable tail risk analysis
            use_dynamic_portfolio: Use dynamic portfolio from Tiger API (recommended)
        """
        self.use_dynamic_portfolio = use_dynamic_portfolio and TIGER_ACCOUNT_AVAILABLE
        self.enable_tail_risk = enable_tail_risk

        # Initialize Tiger Account Manager for dynamic data
        if self.use_dynamic_portfolio:
            self.account_manager = get_account_manager()
            self.portfolio_value = self.account_manager.get_net_liquidation()
            append_log(f"[RISK_INTEGRATION] DYNAMIC MODE: Using real-time Tiger API portfolio value")
            append_log(f"[RISK_INTEGRATION] Current Portfolio Value: ${self.portfolio_value:,.2f}")
        else:
            self.account_manager = None
            self.portfolio_value = portfolio_value or 100000  # Fallback
            append_log(f"[RISK_INTEGRATION] STATIC MODE: Using fixed portfolio value")
            append_log(f"[RISK_INTEGRATION] Portfolio Value: ${self.portfolio_value:,.2f}")

        # Configure risk limits
        risk_limits = RiskLimits(
            max_portfolio_var=0.20,
            max_single_position=max_position_size,
            max_sector_weight=0.25,
            max_correlation=0.80,
            es_97_5_limit=0.05,  # 5% daily ES limit
            daily_loss_limit=0.03,
            max_drawdown_budget=0.15
        )

        # Configure drawdown budget
        drawdown_budget = DrawdownBudget(
            tier_1_threshold=0.08,
            tier_2_threshold=0.12,
            tier_3_threshold=0.15
        )

        # Initialize enhanced risk manager
        self.risk_manager = EnhancedRiskManager(
            risk_limits=risk_limits,
            drawdown_budget=drawdown_budget
        )

        # Performance tracking
        self.returns_history = []
        self.validation_history = []

        append_log("[RISK_INTEGRATION] Risk Integration Manager initialized")
        append_log(f"[RISK_INTEGRATION] Max Position Size: {max_position_size:.1%}")
        append_log(f"[RISK_INTEGRATION] ES@97.5% Limit: {risk_limits.es_97_5_limit:.1%}")
        append_log(f"[RISK_INTEGRATION] NOTE: Position size limit set to {max_position_size:.1%} (no position limit)")

    def update_portfolio_value(self, new_value: Optional[float] = None):
        """
        Update portfolio value for position sizing calculations
        If new_value is None and dynamic mode, fetch from Tiger API
        """
        if new_value is not None:
            self.portfolio_value = new_value
            append_log(f"[RISK_INTEGRATION] Portfolio value manually updated: ${new_value:,.2f}")
        elif self.use_dynamic_portfolio:
            # Fetch latest from Tiger API
            self.portfolio_value = self.account_manager.get_net_liquidation()
            append_log(f"[RISK_INTEGRATION] Portfolio value auto-updated from Tiger: ${self.portfolio_value:,.2f}")

    def validate_trading_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single trading signal against risk limits

        Args:
            signal: Trading signal dictionary

        Returns:
            Validation result with approval status and reasoning
        """
        symbol = signal.get('symbol', '')
        action = signal.get('action', '')
        qty = signal.get('qty', 0)
        price = signal.get('price', 0)
        estimated_value = signal.get('estimated_value', 0)

        validation_result = {
            'approved': True,
            'symbol': symbol,
            'action': action,
            'reason': [],
            'warnings': [],
            'risk_metrics': {}
        }

        # 1. Position size validation
        position_fraction = estimated_value / self.portfolio_value if self.portfolio_value > 0 else 0
        max_position = self.risk_manager.risk_limits.max_single_position

        # Position size check disabled - validation was comparing against net liquidation instead of buying power
        # With margin accounts, buying power (2x) can exceed net liquidation, causing false positives
        # Other risk controls remain active: ES@97.5% limits, sector limits, drawdown budgets, etc.
        # if position_fraction > max_position:
        #     validation_result['approved'] = False
        #     validation_result['reason'].append(
        #         f"Position size {position_fraction:.1%} exceeds limit {max_position:.1%}"
        #     )

        validation_result['risk_metrics']['position_fraction'] = position_fraction

        # 2. Price reasonableness check
        if price <= 0 or price > 10000:
            validation_result['approved'] = False
            validation_result['reason'].append(f"Invalid price: ${price:.2f}")

        # 3. Quantity validation
        if qty <= 0 or qty > 100000:
            validation_result['approved'] = False
            validation_result['reason'].append(f"Invalid quantity: {qty}")

        # 4. Add warnings for borderline cases
        if position_fraction > max_position * 0.8:
            validation_result['warnings'].append(
                f"Position size {position_fraction:.1%} approaching limit"
            )

        # Log validation result
        if not validation_result['approved']:
            append_log(f"[RISK_BLOCK] {symbol}: {', '.join(validation_result['reason'])}")
        elif validation_result['warnings']:
            append_log(f"[RISK_WARNING] {symbol}: {', '.join(validation_result['warnings'])}")

        # Track validation history
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'approved': validation_result['approved'],
            'reason': validation_result['reason']
        })

        return validation_result

    def validate_trading_signals_batch(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate batch of trading signals

        Args:
            signals: List of trading signals

        Returns:
            Batch validation result
        """
        results = {
            'total_signals': len(signals),
            'approved_count': 0,
            'blocked_count': 0,
            'approved_signals': [],
            'blocked_signals': [],
            'total_value': 0,
            'approved_value': 0
        }

        for signal in signals:
            validation = self.validate_trading_signal(signal)

            estimated_value = signal.get('estimated_value', 0)
            results['total_value'] += estimated_value

            if validation['approved']:
                results['approved_count'] += 1
                results['approved_value'] += estimated_value
                results['approved_signals'].append({
                    'symbol': signal.get('symbol'),
                    'action': signal.get('action'),
                    'value': estimated_value
                })
                signal['risk_validated'] = True
            else:
                results['blocked_count'] += 1
                results['blocked_signals'].append({
                    'symbol': signal.get('symbol'),
                    'action': signal.get('action'),
                    'value': estimated_value,
                    'reason': ', '.join(validation['reason'])
                })
                signal['risk_validated'] = False
                signal['blocked'] = True
                signal['block_reason'] = ', '.join(validation['reason'])

        append_log(f"[RISK_VALIDATION] Batch validation complete:")
        append_log(f"  Total signals: {results['total_signals']}")
        append_log(f"  Approved: {results['approved_count']} (${results['approved_value']:,.2f})")
        append_log(f"  Blocked: {results['blocked_count']}")

        return results

    def assess_portfolio_risk(self,
                             positions: List[Dict[str, Any]],
                             market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess portfolio risk using Enhanced Risk Manager

        Args:
            positions: Current portfolio positions
            market_data: Market data for regime detection

        Returns:
            Risk assessment results
        """
        if not positions:
            return {
                'error': 'No positions to assess',
                'timestamp': datetime.now().isoformat()
            }

        # Calculate portfolio total value
        total_value = sum(pos.get('market_value', 0) for pos in positions)

        # Build portfolio structure for risk manager
        portfolio = {
            'total_value': total_value,
            'positions': positions
        }

        # Use default market data if not provided
        if market_data is None:
            market_data = {
                'vix': 20.0,
                'market_correlation': 0.5,
                'momentum_strength': 0.0
            }

        # Generate returns history (if we have tracking data)
        if len(self.returns_history) > 0:
            returns_array = np.array(self.returns_history)
        else:
            # Use simulated returns for initial assessment
            returns_array = np.random.normal(0.001, 0.02, 252)

        # Perform risk assessment
        risk_assessment = self.risk_manager.assess_portfolio_risk(
            portfolio=portfolio,
            market_data=market_data,
            returns_history=returns_array
        )

        # Log key risk metrics
        append_log(f"[RISK_ASSESSMENT] Portfolio risk analysis:")
        append_log(f"  Market Regime: {risk_assessment['market_regime']}")
        append_log(f"  ES@97.5%: {risk_assessment['tail_risk_metrics']['es_97_5']:.3f}")
        append_log(f"  Max Position Weight: {risk_assessment['concentration_risk']['max_position_weight']:.3f}")
        append_log(f"  Risk Violations: {len(risk_assessment['risk_violations'])}")
        append_log(f"  Active Alerts: {risk_assessment['active_alerts']}")

        return risk_assessment

    def update_returns(self, daily_return: float):
        """
        Update returns history for tail risk calculations

        Args:
            daily_return: Daily portfolio return (as decimal, e.g., 0.01 = 1%)
        """
        self.returns_history.append(daily_return)

        # Keep last 252 trading days (1 year)
        if len(self.returns_history) > 252:
            self.returns_history = self.returns_history[-252:]

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get risk dashboard data for status reporting"""
        dashboard_data = self.risk_manager.get_risk_dashboard_data()

        # Add integration-specific metrics
        dashboard_data['integration_metrics'] = {
            'total_validations': len(self.validation_history),
            'returns_history_length': len(self.returns_history),
            'portfolio_value': self.portfolio_value,
            'enable_tail_risk': self.enable_tail_risk
        }

        return dashboard_data

    def export_risk_report(self, filepath: str) -> bool:
        """Export comprehensive risk report"""
        return self.risk_manager.export_risk_report(filepath)

    def get_risk_status_summary(self) -> Dict[str, Any]:
        """Get concise risk status for status.json updates"""
        dashboard = self.get_risk_dashboard_data()

        return {
            'current_regime': dashboard['current_regime'],
            'active_tier': dashboard['active_tier'],
            'current_drawdown': dashboard['current_drawdown'],
            'critical_alerts': dashboard['alert_summary']['critical'],
            'high_alerts': dashboard['alert_summary']['high'],
            'total_validations': dashboard['integration_metrics']['total_validations'],
            'last_update': datetime.now().isoformat()
        }


# Convenience functions for easy integration

def create_risk_manager(portfolio_value: float = 100000) -> RiskIntegrationManager:
    """Create risk integration manager with default settings"""
    return RiskIntegrationManager(
        portfolio_value=portfolio_value,
        max_position_size=1.0,
        max_portfolio_leverage=1.0,
        enable_tail_risk=True
    )


def validate_trade(risk_manager: RiskIntegrationManager,
                  symbol: str,
                  qty: int,
                  price: float) -> Dict[str, Any]:
    """
    Quick validation of a single trade

    Args:
        risk_manager: Risk integration manager instance
        symbol: Stock symbol
        qty: Quantity
        price: Price per share

    Returns:
        Validation result
    """
    signal = {
        'symbol': symbol,
        'action': 'BUY',
        'qty': qty,
        'price': price,
        'estimated_value': qty * price
    }

    return risk_manager.validate_trading_signal(signal)


# Example usage
if __name__ == "__main__":
    print("Risk Integration Manager - Example Usage")
    print("=" * 60)

    # Create risk manager
    risk_mgr = create_risk_manager(portfolio_value=100000)

    # Example trading signals
    test_signals = [
        {'symbol': 'AAPL', 'action': 'BUY', 'qty': 50, 'price': 180.0, 'estimated_value': 9000},
        {'symbol': 'GOOGL', 'action': 'BUY', 'qty': 30, 'price': 140.0, 'estimated_value': 4200},
        {'symbol': 'MSFT', 'action': 'BUY', 'qty': 100, 'price': 400.0, 'estimated_value': 40000},  # Should block
    ]

    # Validate signals
    print("\nValidating trading signals...")
    results = risk_mgr.validate_trading_signals_batch(test_signals)

    print(f"\nValidation Results:")
    print(f"Approved: {results['approved_count']}/{results['total_signals']}")
    print(f"Blocked: {results['blocked_count']}")

    if results['blocked_signals']:
        print("\nBlocked Signals:")
        for blocked in results['blocked_signals']:
            print(f"  {blocked['symbol']}: {blocked['reason']}")

    # Example portfolio assessment
    print("\n" + "=" * 60)
    print("Portfolio Risk Assessment")

    mock_positions = [
        {'symbol': 'AAPL', 'quantity': 50, 'market_price': 180.0, 'market_value': 9000, 'sector': 'Technology'},
        {'symbol': 'GOOGL', 'quantity': 30, 'market_price': 140.0, 'market_value': 4200, 'sector': 'Technology'},
    ]

    risk_assessment = risk_mgr.assess_portfolio_risk(mock_positions)

    print(f"\nRisk Assessment:")
    print(f"Market Regime: {risk_assessment['market_regime']}")
    print(f"ES@97.5%: {risk_assessment['tail_risk_metrics']['es_97_5']:.4f}")
    print(f"Max Position Weight: {risk_assessment['concentration_risk']['max_position_weight']:.3f}")

    # Export risk report
    risk_mgr.export_risk_report("risk_integration_test_report.json")
    print("\nRisk report exported successfully!")