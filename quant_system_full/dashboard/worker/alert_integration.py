#!/usr/bin/env python3
"""
Alert Integration Wrapper
Provides simplified interface for Intelligent Alert System integration

Integrates investment-grade alert system with multi-level prioritization,
context-aware routing, and intelligent deduplication.

Features:
- Simplified API for sending alerts
- Automatic severity mapping
- Fallback to logging when alert system unavailable
- Context-aware alert enhancement
- Thread-safe alert delivery
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Intelligent Alert System
try:
    import sys
    import os
    BOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(BOT_PATH, "bot"))

    from intelligent_alert_system_c1 import IntelligentAlertSystemC1, AlertSeverity, AlertCategory
    ALERT_SYSTEM_AVAILABLE = True
    logger.info("Intelligent Alert System module loaded successfully")
except ImportError as e:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning(f"Intelligent Alert System not available: {e}")
    logger.warning("Falling back to standard logging")

class AlertManager:
    """
    Simplified alert manager for system integration.

    Provides high-level interface for sending alerts with automatic
    severity mapping, context enhancement, and fallback handling.
    """

    def __init__(self):
        """Initialize alert manager with intelligent alert system."""
        self.alert_system = None
        self.alert_count = 0
        self.last_alert_time = {}

        if ALERT_SYSTEM_AVAILABLE:
            try:
                self.alert_system = IntelligentAlertSystemC1()
                self.alert_system.start_alert_system()
                logger.info("Alert Manager initialized with Intelligent Alert System")
            except Exception as e:
                logger.error(f"Failed to initialize Alert Manager: {e}")
                self.alert_system = None
        else:
            logger.warning("Alert Manager running in fallback mode (logging only)")

    def send_alert(self, category: str, message: str, severity: str = 'INFO',
                   details: Optional[Dict[str, Any]] = None,
                   source: str = 'TradingSystem') -> bool:
        """
        Send alert through intelligent alert system.

        Args:
            category: Alert category (RISK, TRADING, SYSTEM, COMPLIANCE, etc.)
            message: Alert message
            severity: Alert severity (LOW, INFO, MEDIUM, HIGH, CRITICAL)
            details: Additional context and details
            source: Source component generating the alert

        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.alert_system:
            # Fallback to simple logging
            self._log_alert(category, message, severity, details)
            return True

        try:
            # Map severity strings to AlertSeverity enum
            severity_map = {
                'LOW': AlertSeverity.LOW,
                'INFO': AlertSeverity.INFO,
                'MEDIUM': AlertSeverity.MEDIUM,
                'HIGH': AlertSeverity.HIGH,
                'CRITICAL': AlertSeverity.CRITICAL
            }

            alert_severity = severity_map.get(severity.upper(), AlertSeverity.INFO)

            # Map category strings to AlertCategory enum
            category_map = {
                'RISK': AlertCategory.RISK_BREACH,
                'TRADING': AlertCategory.TRADING_ANOMALY,
                'SYSTEM': AlertCategory.SYSTEM_FAILURE,
                'COMPLIANCE': AlertCategory.COMPLIANCE_VIOLATION,
                'PERFORMANCE': AlertCategory.PERFORMANCE_DEGRADATION,
                'SECURITY': AlertCategory.SECURITY_THREAT,
                'DATA': AlertCategory.DATA_QUALITY,
                'NETWORK': AlertCategory.NETWORK_CONNECTIVITY,
                'RESOURCE': AlertCategory.RESOURCE_EXHAUSTION,
                'MARKET': AlertCategory.MARKET_EVENT
            }

            alert_category = category_map.get(category.upper(), AlertCategory.SYSTEM_FAILURE)

            # Enhance context with system information
            enhanced_context = details or {}
            enhanced_context['timestamp'] = datetime.now().isoformat()
            enhanced_context['source_component'] = source

            # Create alert with intelligent system
            alert = self.alert_system.create_alert(
                title=f"{category.upper()}: {message[:100]}",
                message=message,
                severity=alert_severity,
                category=alert_category,
                source=source,
                context=enhanced_context
            )

            if alert:
                self.alert_count += 1
                self.last_alert_time[category] = datetime.now()
                logger.debug(f"Alert sent: {category} - {severity} - {message[:50]}")
                return True
            else:
                logger.warning(f"Alert suppressed: {category} - {message[:50]}")
                return False

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            # Fallback to logging
            self._log_alert(category, message, severity, details)
            return False

    def _log_alert(self, category: str, message: str, severity: str,
                   details: Optional[Dict[str, Any]]) -> None:
        """
        Fallback logging when alert system unavailable.

        Args:
            category: Alert category
            message: Alert message
            severity: Alert severity
            details: Additional details
        """
        log_msg = f"[{severity}] [{category}] {message}"

        if details:
            log_msg += f" | Details: {details}"

        # Map to appropriate log level
        if severity in ['CRITICAL', 'HIGH']:
            logger.error(log_msg)
        elif severity == 'MEDIUM':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def send_risk_alert(self, message: str, severity: str = 'HIGH',
                       details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send risk-related alert.

        Args:
            message: Risk alert message
            severity: Alert severity
            details: Risk metrics and context

        Returns:
            True if alert sent successfully
        """
        return self.send_alert('RISK', message, severity, details, source='RiskManager')

    def send_trading_alert(self, message: str, severity: str = 'INFO',
                          details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send trading-related alert.

        Args:
            message: Trading alert message
            severity: Alert severity
            details: Trading context and metrics

        Returns:
            True if alert sent successfully
        """
        return self.send_alert('TRADING', message, severity, details, source='TradingEngine')

    def send_compliance_alert(self, message: str, severity: str = 'CRITICAL',
                             details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send compliance violation alert.

        Args:
            message: Compliance alert message
            severity: Alert severity (defaults to CRITICAL)
            details: Violation details and context

        Returns:
            True if alert sent successfully
        """
        return self.send_alert('COMPLIANCE', message, severity, details, source='ComplianceMonitor')

    def send_system_alert(self, message: str, severity: str = 'HIGH',
                         details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send system-related alert.

        Args:
            message: System alert message
            severity: Alert severity
            details: System context and metrics

        Returns:
            True if alert sent successfully
        """
        return self.send_alert('SYSTEM', message, severity, details, source='SystemMonitor')

    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dictionary with alert statistics
        """
        return {
            'total_alerts_sent': self.alert_count,
            'last_alert_times': {k: v.isoformat() for k, v in self.last_alert_time.items()},
            'alert_system_available': self.alert_system is not None,
            'active_alerts': len(self.alert_system.get_active_alerts()) if self.alert_system else 0
        }

    def shutdown(self) -> None:
        """Shutdown alert manager gracefully."""
        if self.alert_system:
            try:
                self.alert_system.shutdown()
                logger.info("Alert Manager shutdown complete")
            except Exception as e:
                logger.error(f"Error during alert manager shutdown: {e}")

# Singleton instance
_alert_manager = None

def get_alert_manager() -> AlertManager:
    """
    Get or create global alert manager singleton.

    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager

def send_alert(category: str, message: str, severity: str = 'INFO',
              details: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to send alert via global alert manager.

    Args:
        category: Alert category
        message: Alert message
        severity: Alert severity
        details: Additional context

    Returns:
        True if alert sent successfully
    """
    manager = get_alert_manager()
    return manager.send_alert(category, message, severity, details)