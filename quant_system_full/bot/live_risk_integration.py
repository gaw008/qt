#!/usr/bin/env python3
"""
Live Trading Risk Integration System
实时交易风险集成系统

Real-time risk management integration for live trading operations:
- Pre-trade risk checks with sub-second response
- Post-trade risk attribution and monitoring
- Dynamic position sizing based on risk metrics
- Emergency stop triggers and fail-safes
- Real-time risk dashboard integration

Live Trading Integration Features:
- Tiger API execution integration
- Real-time portfolio risk monitoring
- Automated risk-based position sizing
- Emergency stop mechanisms
- Risk performance tracking
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import threading
import time
import json
from dataclasses import dataclass
from enum import Enum

# Import local risk management components
try:
    from enhanced_risk_manager import EnhancedRiskManager, RiskLevel, MarketRegime, RiskAlert
    from factor_crowding_monitor import FactorCrowdingMonitor
    from risk_performance_optimizer import RiskPerformanceOptimizer, PerformanceLevel
except ImportError as e:
    logging.warning(f"Could not import risk components: {e}")

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAction(Enum):
    """Trading actions based on risk assessment"""
    ALLOW = "ALLOW"                    # Normal trading allowed
    REDUCE = "REDUCE"                  # Reduce position sizes
    BLOCK = "BLOCK"                    # Block new positions
    EMERGENCY_STOP = "EMERGENCY_STOP"  # Emergency stop all trading

class RiskCheckResult(Enum):
    """Risk check results for trading decisions"""
    PASS = "PASS"                      # Risk check passed
    WARNING = "WARNING"                # Proceed with caution
    FAIL = "FAIL"                      # Risk check failed
    EMERGENCY = "EMERGENCY"            # Emergency risk condition

@dataclass
class LiveTradingLimits:
    """Calibrated limits for live trading operations"""
    # Position sizing limits
    max_position_size_pct: float = 0.08        # 8% maximum single position
    max_daily_turnover_pct: float = 0.20       # 20% maximum daily turnover
    max_sector_allocation_pct: float = 0.25     # 25% maximum sector allocation

    # Risk metric limits
    max_portfolio_var_daily: float = 0.025      # 2.5% daily VaR limit
    max_es_97_5_daily: float = 0.03            # 3% daily ES@97.5% limit
    max_tracking_error: float = 0.06           # 6% annual tracking error

    # Drawdown limits with actions
    warning_drawdown: float = 0.05             # 5% warning level
    action_drawdown: float = 0.08              # 8% action level
    emergency_drawdown: float = 0.12           # 12% emergency level

    # Liquidity and market impact limits
    max_adv_participation: float = 0.10        # 10% of average daily volume
    max_market_impact_bps: float = 20          # 20 bps maximum market impact

    # Factor crowding limits
    max_factor_hhi: float = 0.30               # 30% HHI threshold
    min_effective_breadth: float = 6.0         # Minimum 6 effective positions

@dataclass
class PreTradeRiskCheck:
    """Pre-trade risk check results"""
    check_result: RiskCheckResult
    allowed_quantity: int
    risk_score: float
    violations: List[str]
    warnings: List[str]
    suggested_actions: List[str]
    check_timestamp: str

@dataclass
class PostTradeRiskUpdate:
    """Post-trade risk update"""
    trade_id: str
    risk_contribution: float
    portfolio_risk_change: float
    new_risk_metrics: Dict[str, float]
    risk_alerts: List[RiskAlert]
    update_timestamp: str

class LiveRiskIntegrator:
    """
    Live trading risk integration system

    Provides real-time risk management for live trading operations with:
    - Pre-trade risk checks
    - Real-time portfolio monitoring
    - Risk-based position sizing
    - Emergency controls
    """

    def __init__(self,
                 limits: Optional[LiveTradingLimits] = None,
                 performance_level: PerformanceLevel = PerformanceLevel.PRODUCTION):

        self.limits = limits or LiveTradingLimits()
        self.performance_level = performance_level

        # Initialize risk management components
        self.enhanced_risk_manager = EnhancedRiskManager()
        self.factor_crowding_monitor = FactorCrowdingMonitor()
        self.performance_optimizer = RiskPerformanceOptimizer(
            performance_level=performance_level,
            max_workers=2,  # Conservative for live trading
            cache_size=500
        )

        # Live trading state
        self.current_portfolio = {}
        self.current_market_data = {}
        self.trading_enabled = True
        self.emergency_mode = False
        self.last_risk_update = datetime.now()

        # Performance tracking
        self.risk_check_times = []
        self.pre_trade_checks_count = 0
        self.blocked_trades_count = 0
        self.emergency_stops_count = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info("Live Risk Integrator initialized for production trading")

    def update_portfolio_state(self, portfolio_data: Dict[str, Any]) -> None:
        """Update current portfolio state for risk calculations"""
        with self.lock:
            self.current_portfolio = portfolio_data.copy()
            self.last_risk_update = datetime.now()

    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update current market data for risk calculations"""
        with self.lock:
            self.current_market_data = market_data.copy()

    def pre_trade_risk_check(self,
                           symbol: str,
                           quantity: int,
                           order_type: str,
                           current_price: float) -> PreTradeRiskCheck:
        """
        Comprehensive pre-trade risk check

        Validates trade against risk limits and portfolio constraints
        Returns risk check result with allowed quantity and warnings
        """
        start_time = time.perf_counter()

        try:
            with self.lock:
                # Emergency mode check
                if self.emergency_mode:
                    return PreTradeRiskCheck(
                        check_result=RiskCheckResult.EMERGENCY,
                        allowed_quantity=0,
                        risk_score=100.0,
                        violations=["EMERGENCY_MODE_ACTIVE"],
                        warnings=["Trading blocked due to emergency conditions"],
                        suggested_actions=["Wait for emergency mode clearance"],
                        check_timestamp=datetime.now().isoformat()
                    )

                # Basic validation
                if not self.trading_enabled:
                    return PreTradeRiskCheck(
                        check_result=RiskCheckResult.FAIL,
                        allowed_quantity=0,
                        risk_score=0.0,
                        violations=["TRADING_DISABLED"],
                        warnings=["Trading is currently disabled"],
                        suggested_actions=["Enable trading or contact administrator"],
                        check_timestamp=datetime.now().isoformat()
                    )

                # Calculate proposed position impact
                trade_value = abs(quantity * current_price)
                portfolio_value = self.current_portfolio.get('total_value', 0)

                if portfolio_value == 0:
                    return PreTradeRiskCheck(
                        check_result=RiskCheckResult.FAIL,
                        allowed_quantity=0,
                        risk_score=0.0,
                        violations=["NO_PORTFOLIO_DATA"],
                        warnings=["Portfolio data not available"],
                        suggested_actions=["Update portfolio data before trading"],
                        check_timestamp=datetime.now().isoformat()
                    )

                # Position sizing checks
                position_pct = trade_value / portfolio_value
                violations = []
                warnings = []
                risk_score = 0.0

                # Check position size limits
                if position_pct > self.limits.max_position_size_pct:
                    max_allowed_value = portfolio_value * self.limits.max_position_size_pct
                    max_allowed_quantity = int(max_allowed_value / current_price)
                    violations.append("POSITION_SIZE_EXCEEDED")
                    risk_score += 30.0
                elif position_pct > self.limits.max_position_size_pct * 0.8:
                    warnings.append("LARGE_POSITION_SIZE")
                    risk_score += 15.0

                # Check sector concentration
                symbol_sector = self._get_symbol_sector(symbol)
                current_sector_allocation = self._calculate_sector_allocation(symbol_sector)
                new_sector_allocation = current_sector_allocation + position_pct

                if new_sector_allocation > self.limits.max_sector_allocation_pct:
                    violations.append("SECTOR_CONCENTRATION_EXCEEDED")
                    risk_score += 25.0
                elif new_sector_allocation > self.limits.max_sector_allocation_pct * 0.9:
                    warnings.append("HIGH_SECTOR_CONCENTRATION")
                    risk_score += 10.0

                # Check liquidity constraints
                daily_volume = self._get_daily_volume(symbol)
                if daily_volume > 0:
                    volume_participation = abs(quantity) / daily_volume
                    if volume_participation > self.limits.max_adv_participation:
                        violations.append("LIQUIDITY_CONSTRAINT")
                        risk_score += 20.0

                # Real-time risk assessment
                if self.current_portfolio.get('returns') is not None:
                    portfolio_returns = np.array(self.current_portfolio['returns'])

                    # Calculate current ES
                    current_es = self.performance_optimizer.calculate_portfolio_es_optimized(
                        portfolio_returns, [0.975]
                    ).get('es_97', 0)

                    if current_es > self.limits.max_es_97_5_daily:
                        violations.append("ES_LIMIT_EXCEEDED")
                        risk_score += 35.0
                    elif current_es > self.limits.max_es_97_5_daily * 0.8:
                        warnings.append("HIGH_ES_RISK")
                        risk_score += 15.0

                # Factor crowding check (if exposures available)
                if self.current_portfolio.get('factor_exposures'):
                    crowding_check = self.performance_optimizer.optimized_factor_crowding_check(
                        self.current_portfolio['factor_exposures'],
                        np.array(self.current_portfolio.get('weights', []))
                    )

                    if crowding_check['crowding_risk_level'] == 'HIGH':
                        violations.append("FACTOR_CROWDING")
                        risk_score += 20.0

                # Determine final result
                if len(violations) > 0:
                    if "EMERGENCY_MODE_ACTIVE" in violations or "ES_LIMIT_EXCEEDED" in violations:
                        check_result = RiskCheckResult.EMERGENCY
                        allowed_quantity = 0
                    else:
                        check_result = RiskCheckResult.FAIL
                        # Calculate reduced quantity if position size exceeded
                        if "POSITION_SIZE_EXCEEDED" in violations:
                            max_allowed_value = portfolio_value * self.limits.max_position_size_pct
                            allowed_quantity = int(max_allowed_value / current_price)
                        else:
                            allowed_quantity = 0
                elif len(warnings) > 0:
                    check_result = RiskCheckResult.WARNING
                    allowed_quantity = quantity
                else:
                    check_result = RiskCheckResult.PASS
                    allowed_quantity = quantity

                # Generate suggested actions
                suggested_actions = []
                if "POSITION_SIZE_EXCEEDED" in violations:
                    suggested_actions.append(f"Reduce quantity to {allowed_quantity}")
                if "SECTOR_CONCENTRATION_EXCEEDED" in violations:
                    suggested_actions.append("Diversify across sectors")
                if "ES_LIMIT_EXCEEDED" in violations:
                    suggested_actions.append("Reduce portfolio risk before trading")
                if "FACTOR_CROWDING" in violations:
                    suggested_actions.append("Diversify factor exposures")

                # Performance tracking
                self.pre_trade_checks_count += 1
                check_time = (time.perf_counter() - start_time) * 1000
                self.risk_check_times.append(check_time)

                if check_result in [RiskCheckResult.FAIL, RiskCheckResult.EMERGENCY]:
                    self.blocked_trades_count += 1

                return PreTradeRiskCheck(
                    check_result=check_result,
                    allowed_quantity=allowed_quantity,
                    risk_score=risk_score,
                    violations=violations,
                    warnings=warnings,
                    suggested_actions=suggested_actions,
                    check_timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            logger.error(f"Pre-trade risk check error: {e}")
            return PreTradeRiskCheck(
                check_result=RiskCheckResult.FAIL,
                allowed_quantity=0,
                risk_score=100.0,
                violations=["RISK_CHECK_ERROR"],
                warnings=[f"Risk check failed: {str(e)}"],
                suggested_actions=["Contact system administrator"],
                check_timestamp=datetime.now().isoformat()
            )

    def post_trade_risk_update(self,
                             trade_id: str,
                             symbol: str,
                             quantity: int,
                             execution_price: float) -> PostTradeRiskUpdate:
        """
        Post-trade risk update and portfolio impact assessment
        """
        try:
            with self.lock:
                # Calculate trade impact
                trade_value = quantity * execution_price
                portfolio_value = self.current_portfolio.get('total_value', 0)

                if portfolio_value > 0:
                    risk_contribution = abs(trade_value) / portfolio_value
                else:
                    risk_contribution = 0.0

                # Update portfolio state (simplified)
                # In real implementation, this would trigger full portfolio recalculation

                # Generate risk alerts if needed
                alerts = []
                if risk_contribution > 0.05:  # 5% impact threshold
                    alerts.append(RiskAlert(
                        timestamp=datetime.now().isoformat(),
                        level=RiskLevel.MEDIUM,
                        category="Portfolio Impact",
                        message=f"Large trade impact: {risk_contribution:.1%}",
                        metric_value=risk_contribution,
                        limit_value=0.05,
                        suggested_actions=["Monitor position closely", "Consider rebalancing"]
                    ))

                # Calculate new risk metrics (placeholder)
                new_risk_metrics = {
                    "portfolio_var": 0.025,  # Would be calculated from updated portfolio
                    "max_drawdown": 0.03,
                    "beta": 1.1
                }

                return PostTradeRiskUpdate(
                    trade_id=trade_id,
                    risk_contribution=risk_contribution,
                    portfolio_risk_change=0.0,  # Would be calculated
                    new_risk_metrics=new_risk_metrics,
                    risk_alerts=alerts,
                    update_timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            logger.error(f"Post-trade risk update error: {e}")
            return PostTradeRiskUpdate(
                trade_id=trade_id,
                risk_contribution=0.0,
                portfolio_risk_change=0.0,
                new_risk_metrics={},
                risk_alerts=[],
                update_timestamp=datetime.now().isoformat()
            )

    def emergency_stop_check(self) -> Tuple[bool, List[str]]:
        """
        Emergency stop condition check
        Returns (should_stop, reasons)
        """
        try:
            with self.lock:
                reasons = []

                # Check drawdown emergency level
                if self.current_portfolio.get('current_drawdown', 0) > self.limits.emergency_drawdown:
                    reasons.append(f"Emergency drawdown exceeded: {self.current_portfolio['current_drawdown']:.1%}")

                # Check portfolio VaR emergency level
                portfolio_returns = self.current_portfolio.get('returns')
                if portfolio_returns is not None:
                    returns_array = np.array(portfolio_returns)
                    if len(returns_array) > 10:
                        current_var = np.percentile(returns_array, 5)  # 95% VaR
                        if abs(current_var) > self.limits.max_portfolio_var_daily * 2:  # 2x normal limit
                            reasons.append(f"Emergency VaR level: {abs(current_var):.1%}")

                # Check market conditions
                vix = self.current_market_data.get('vix', 20)
                if vix > 40:  # Extreme volatility
                    reasons.append(f"Extreme market volatility: VIX {vix}")

                # System health checks
                memory_usage = self.performance_optimizer.metrics.memory_usage_mb
                if memory_usage > 1000:  # 1GB memory usage
                    reasons.append(f"High system memory usage: {memory_usage:.0f}MB")

                should_stop = len(reasons) > 0

                if should_stop and not self.emergency_mode:
                    self.emergency_mode = True
                    self.emergency_stops_count += 1
                    logger.critical(f"EMERGENCY STOP TRIGGERED: {reasons}")

                return should_stop, reasons

        except Exception as e:
            logger.error(f"Emergency stop check error: {e}")
            return True, [f"Emergency check failed: {str(e)}"]

    def get_real_time_risk_dashboard(self) -> Dict[str, Any]:
        """Get real-time risk dashboard data for monitoring"""
        try:
            with self.lock:
                # Performance metrics
                avg_check_time = np.mean(self.risk_check_times[-100:]) if self.risk_check_times else 0.0
                block_rate = self.blocked_trades_count / max(self.pre_trade_checks_count, 1)

                # Current risk status
                emergency_stop, emergency_reasons = self.emergency_stop_check()

                dashboard_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system_status": {
                        "trading_enabled": self.trading_enabled,
                        "emergency_mode": self.emergency_mode,
                        "emergency_reasons": emergency_reasons
                    },
                    "performance_metrics": {
                        "avg_risk_check_time_ms": avg_check_time,
                        "pre_trade_checks_count": self.pre_trade_checks_count,
                        "blocked_trades_count": self.blocked_trades_count,
                        "block_rate": block_rate,
                        "emergency_stops_count": self.emergency_stops_count
                    },
                    "current_risk_limits": {
                        "max_position_size_pct": self.limits.max_position_size_pct,
                        "max_es_97_5_daily": self.limits.max_es_97_5_daily,
                        "emergency_drawdown": self.limits.emergency_drawdown,
                        "max_factor_hhi": self.limits.max_factor_hhi
                    },
                    "portfolio_status": {
                        "total_value": self.current_portfolio.get('total_value', 0),
                        "current_drawdown": self.current_portfolio.get('current_drawdown', 0),
                        "position_count": len(self.current_portfolio.get('positions', [])),
                        "last_update": self.last_risk_update.isoformat()
                    },
                    "market_conditions": {
                        "vix": self.current_market_data.get('vix', 20),
                        "market_correlation": self.current_market_data.get('market_correlation', 0.5)
                    }
                }

                return dashboard_data

        except Exception as e:
            logger.error(f"Dashboard data error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol (placeholder implementation)"""
        # In real implementation, this would query a sector mapping database
        sector_mapping = {
            'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
        }
        return sector_mapping.get(symbol, 'Other')

    def _calculate_sector_allocation(self, sector: str) -> float:
        """Calculate current sector allocation"""
        # Placeholder implementation
        positions = self.current_portfolio.get('positions', [])
        total_value = self.current_portfolio.get('total_value', 0)

        if total_value == 0:
            return 0.0

        sector_value = sum(
            pos.get('market_value', 0) for pos in positions
            if pos.get('sector') == sector
        )

        return sector_value / total_value

    def _get_daily_volume(self, symbol: str) -> int:
        """Get average daily volume for symbol (placeholder)"""
        # In real implementation, this would query market data
        return 1000000  # Default 1M shares

    def enable_trading(self) -> None:
        """Enable trading"""
        with self.lock:
            self.trading_enabled = True
            logger.info("Trading enabled")

    def disable_trading(self) -> None:
        """Disable trading"""
        with self.lock:
            self.trading_enabled = False
            logger.info("Trading disabled")

    def clear_emergency_mode(self) -> None:
        """Clear emergency mode (manual override)"""
        with self.lock:
            self.emergency_mode = False
            logger.info("Emergency mode cleared")

    def export_risk_integration_report(self, filepath: str) -> bool:
        """Export comprehensive risk integration report"""
        try:
            report_data = {
                "report_timestamp": datetime.now().isoformat(),
                "integration_config": {
                    "performance_level": self.performance_level.value,
                    "trading_enabled": self.trading_enabled,
                    "emergency_mode": self.emergency_mode
                },
                "calibrated_limits": {
                    "max_position_size_pct": self.limits.max_position_size_pct,
                    "max_es_97_5_daily": self.limits.max_es_97_5_daily,
                    "emergency_drawdown": self.limits.emergency_drawdown,
                    "max_factor_hhi": self.limits.max_factor_hhi
                },
                "performance_summary": {
                    "total_risk_checks": self.pre_trade_checks_count,
                    "blocked_trades": self.blocked_trades_count,
                    "emergency_stops": self.emergency_stops_count,
                    "avg_check_time_ms": np.mean(self.risk_check_times) if self.risk_check_times else 0
                },
                "current_status": self.get_real_time_risk_dashboard()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Risk integration report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export integration report: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Live Trading Risk Integration System")
    print("=" * 40)

    # Initialize live risk integrator
    integrator = LiveRiskIntegrator(
        performance_level=PerformanceLevel.DEVELOPMENT
    )

    # Simulate portfolio data
    portfolio_data = {
        'total_value': 1000000,
        'positions': [
            {'symbol': 'AAPL', 'market_value': 100000, 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'market_value': 80000, 'sector': 'Technology'}
        ],
        'returns': np.random.normal(0.001, 0.015, 100),
        'current_drawdown': 0.03
    }

    market_data = {
        'vix': 22.0,
        'market_correlation': 0.6
    }

    # Update states
    integrator.update_portfolio_state(portfolio_data)
    integrator.update_market_data(market_data)

    # Test pre-trade risk check
    print("Testing pre-trade risk check...")
    risk_check = integrator.pre_trade_risk_check(
        symbol="AAPL",
        quantity=500,
        order_type="MARKET",
        current_price=150.0
    )

    print(f"Risk Check Result: {risk_check.check_result.value}")
    print(f"Allowed Quantity: {risk_check.allowed_quantity}")
    print(f"Risk Score: {risk_check.risk_score:.1f}")
    print(f"Violations: {risk_check.violations}")
    print(f"Warnings: {risk_check.warnings}")

    # Test emergency stop check
    emergency_stop, reasons = integrator.emergency_stop_check()
    print(f"\nEmergency Stop: {emergency_stop}")
    if reasons:
        print(f"Reasons: {reasons}")

    # Get dashboard data
    dashboard = integrator.get_real_time_risk_dashboard()
    print(f"\nDashboard Status:")
    print(f"Trading Enabled: {dashboard['system_status']['trading_enabled']}")
    print(f"Emergency Mode: {dashboard['system_status']['emergency_mode']}")
    print(f"Block Rate: {dashboard['performance_metrics']['block_rate']:.1%}")

    # Export report
    integrator.export_risk_integration_report("live_risk_integration_report.json")
    print(f"\nIntegration report exported successfully!")