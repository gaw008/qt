#!/usr/bin/env python3
"""
Compliance Integration Wrapper
Simplified interface for integrating ComplianceMonitoringSystem into trading workflow
NOW WITH DYNAMIC TIGER ACCOUNT DATA - No more hardcoded values!
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add bot path for imports
BOT_PATH = Path(__file__).parent.parent.parent / "bot"
sys.path.insert(0, str(BOT_PATH))

from compliance_monitoring_system import (
    ComplianceMonitoringSystem,
    ComplianceViolation,
    ViolationSeverity
)

# Import Tiger Account Manager for dynamic data
try:
    from tiger_account_manager import get_account_manager
    TIGER_ACCOUNT_AVAILABLE = True
except ImportError:
    TIGER_ACCOUNT_AVAILABLE = False
    logging.warning("[COMPLIANCE] Tiger Account Manager not available, using defaults")

logger = logging.getLogger(__name__)


class ComplianceMonitor:
    """
    Simplified compliance monitor wrapper for trading system integration.

    Provides pre-trade validation and continuous compliance monitoring
    with simplified interface for the trading workflow.

    NOW DYNAMICALLY RETRIEVES account data from Tiger API!
    """

    def __init__(self,
                 account_id: Optional[str] = None,
                 max_position_percentage: float = 0.25,
                 max_concentration: float = 0.25,
                 enable_continuous_monitoring: bool = True,
                 use_dynamic_limits: bool = True):
        """
        Initialize compliance monitor with DYNAMIC account data.

        Args:
            account_id: Trading account identifier (auto-detected if None)
            max_position_percentage: Max position as % of account (default 25%)
            max_concentration: Maximum concentration per holding (0-1)
            enable_continuous_monitoring: Enable background monitoring
            use_dynamic_limits: Use dynamic limits from Tiger API (recommended)
        """
        self.use_dynamic_limits = use_dynamic_limits and TIGER_ACCOUNT_AVAILABLE
        self.max_position_percentage = max_position_percentage
        self.max_concentration = max_concentration
        self.enable_continuous_monitoring = enable_continuous_monitoring

        # Initialize Tiger Account Manager for dynamic data
        if self.use_dynamic_limits:
            self.account_manager = get_account_manager()
            self.account_id = self.account_manager.account_id

            # Get initial account data
            self._update_dynamic_limits()

            logger.info(f"[COMPLIANCE] DYNAMIC MODE: Using real-time Tiger API data")
            logger.info(f"[COMPLIANCE] Account {self.account_id}: Net=${self.account_value:,.2f}")
            logger.info(f"[COMPLIANCE] Dynamic max position: ${self.max_position_value:,.2f} "
                       f"({self.max_position_percentage*100:.0f}% of account)")
        else:
            # Fallback to static values
            self.account_manager = None
            self.account_id = account_id or "41169270"
            self.account_value = 100000  # Default fallback
            self.max_position_value = 50000  # Default fallback
            logger.warning(f"[COMPLIANCE] STATIC MODE: Using fallback values")
            logger.info(f"[COMPLIANCE] Max position: ${self.max_position_value:,.0f}, "
                       f"Max concentration: {self.max_concentration:.1%}")

        # Initialize underlying compliance system
        self.compliance_system = ComplianceMonitoringSystem()

        # Integration state
        self.last_check_time = datetime.now()
        self.check_count = 0

    def _update_dynamic_limits(self):
        """Update limits based on current Tiger account data"""
        if not self.use_dynamic_limits:
            return

        try:
            assets = self.account_manager.get_account_assets()
            if assets:
                self.account_value = assets['net_liquidation']
                self.max_position_value = self.account_manager.calculate_max_position_size(
                    self.max_position_percentage
                )
                logger.debug(f"[COMPLIANCE] Updated limits: Account=${self.account_value:,.2f}, "
                            f"MaxPos=${self.max_position_value:,.2f}")
        except Exception as e:
            logger.error(f"[COMPLIANCE] Failed to update dynamic limits: {e}")

    def validate_trade(self,
                      symbol: str,
                      side: str,
                      quantity: int,
                      price: float,
                      current_positions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Pre-trade compliance validation.

        Args:
            symbol: Stock symbol
            side: Trade side ('BUY' or 'SELL')
            quantity: Number of shares
            price: Price per share
            current_positions: Current portfolio positions

        Returns:
            Validation result with compliance status and any violations
        """
        try:
            # Update dynamic limits before validation
            if self.use_dynamic_limits:
                self._update_dynamic_limits()

            trade_value = quantity * price

            # Basic trade validation
            violations = []

            # Check 1: Position size limit
            if side.upper() == 'BUY' and trade_value > self.max_position_value:
                violations.append({
                    'rule_id': 'POS_001',
                    'severity': 'MEDIUM',
                    'message': f'Position value ${trade_value:,.0f} exceeds limit ${self.max_position_value:,.0f}',
                    'actual_value': trade_value,
                    'threshold': self.max_position_value
                })

            # Check 2: Concentration limit (if positions provided)
            if current_positions and side.upper() == 'BUY':
                total_portfolio_value = sum(pos.get('market_value', 0) for pos in current_positions)
                total_portfolio_value += trade_value

                if total_portfolio_value > 0:
                    position_concentration = trade_value / total_portfolio_value

                    if position_concentration > self.max_concentration:
                        violations.append({
                            'rule_id': 'CON_001',
                            'severity': 'MEDIUM',
                            'message': f'Position concentration {position_concentration:.1%} exceeds limit {self.max_concentration:.1%}',
                            'actual_value': position_concentration,
                            'threshold': self.max_concentration
                        })

            # Check 3: Price reasonableness ($1 - $10,000 per share)
            if price < 1.0 or price > 10000:
                violations.append({
                    'rule_id': 'DATA_001',
                    'severity': 'LOW',
                    'message': f'Price ${price:.2f} outside reasonable range ($1-$10,000)',
                    'actual_value': price,
                    'threshold': 10000 if price > 10000 else 1.0
                })

            # Check 4: Quantity reasonableness (1 - 10,000 shares)
            if quantity < 1 or quantity > 10000:
                violations.append({
                    'rule_id': 'DATA_001',
                    'severity': 'LOW',
                    'message': f'Quantity {quantity} outside reasonable range (1-10,000)',
                    'actual_value': quantity,
                    'threshold': 10000 if quantity > 10000 else 1
                })

            # Determine overall compliance
            is_compliant = len(violations) == 0

            # Check for critical violations
            has_critical = any(v['severity'] == 'CRITICAL' for v in violations)
            has_high = any(v['severity'] == 'HIGH' for v in violations)

            result = {
                'compliant': is_compliant,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'trade_value': trade_value,
                'violations': violations,
                'violation_count': len(violations),
                'has_critical_violations': has_critical,
                'has_high_violations': has_high,
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'APPROVE' if is_compliant else ('REJECT' if has_critical else 'REVIEW')
            }

            if not is_compliant:
                logger.warning(f"[COMPLIANCE] Trade validation failed for {symbol}: {len(violations)} violations")
                for v in violations:
                    logger.warning(f"  - {v['rule_id']}: {v['message']}")
            else:
                logger.info(f"[COMPLIANCE] Trade validated: {side} {quantity} {symbol} @ ${price:.2f}")

            return result

        except Exception as e:
            logger.error(f"[COMPLIANCE] Trade validation error: {e}")
            return {
                'compliant': False,
                'error': str(e),
                'violations': [{
                    'rule_id': 'SYS_ERROR',
                    'severity': 'HIGH',
                    'message': f'Validation system error: {e}'
                }],
                'recommendation': 'REJECT'
            }

    def check_all_compliance_rules(self,
                                   portfolio_data: Optional[Dict] = None) -> List[ComplianceViolation]:
        """
        Check all compliance rules against current portfolio.

        Args:
            portfolio_data: Current portfolio metrics (optional)

        Returns:
            List of detected compliance violations
        """
        try:
            self.check_count += 1
            self.last_check_time = datetime.now()

            # Get violations from compliance system
            violations = list(self.compliance_system.active_violations.values())

            logger.info(f"[COMPLIANCE] Check #{self.check_count}: {len(violations)} active violations")

            if violations:
                # Log summary by severity
                severity_counts = {}
                for v in violations:
                    sev = v.severity.value
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1

                for severity, count in severity_counts.items():
                    logger.warning(f"  - {severity.upper()}: {count} violations")

            return violations

        except Exception as e:
            logger.error(f"[COMPLIANCE] Rule check error: {e}")
            return []

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get current compliance monitoring status.

        Returns:
            Status dictionary with compliance metrics
        """
        try:
            system_status = self.compliance_system.get_current_status()

            status = {
                'account_id': self.account_id,
                'monitoring_active': system_status.get('monitoring_active', False),
                'total_rules': system_status.get('total_rules', 0),
                'active_violations': system_status.get('active_violations', 0),
                'last_check': self.last_check_time.isoformat(),
                'check_count': self.check_count,
                'violation_summary': system_status.get('violation_summary', {}),
                'max_position_value': self.max_position_value,
                'max_concentration': self.max_concentration,
                'continuous_monitoring': self.enable_continuous_monitoring
            }

            return status

        except Exception as e:
            logger.error(f"[COMPLIANCE] Status retrieval error: {e}")
            return {
                'error': str(e),
                'account_id': self.account_id
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary format."""
        return {
            'violation_id': getattr(self, 'violation_id', 'unknown'),
            'rule_id': getattr(self, 'rule_id', 'unknown'),
            'severity': getattr(self, 'severity', 'unknown'),
            'message': getattr(self, 'message', 'No message'),
            'timestamp': datetime.now().isoformat()
        }


# Extend ComplianceViolation class with to_dict method
if not hasattr(ComplianceViolation, 'to_dict'):
    def _violation_to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary format."""
        return {
            'violation_id': self.violation_id,
            'rule_id': self.rule_id,
            'violation_type': self.violation_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'actual_value': self.actual_value,
            'threshold_value': self.threshold_value,
            'deviation_amount': self.deviation_amount,
            'affected_entity': self.affected_entity,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'detection_method': self.detection_method
        }

    ComplianceViolation.to_dict = _violation_to_dict