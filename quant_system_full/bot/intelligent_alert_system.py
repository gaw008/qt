#!/usr/bin/env python3
"""
Intelligent Alert System for Quantitative Trading

This module provides comprehensive alerting capabilities for the trading system,
including risk-based alerts, performance monitoring, system health alerts,
and market condition warnings.

Alert Types:
- RISK_ALERT: Position risk, portfolio risk, drawdown alerts
- PERFORMANCE_ALERT: Poor performance, unexpected returns
- SYSTEM_ALERT: System failures, data issues, connectivity problems
- MARKET_ALERT: Market regime changes, volatility spikes
- EXECUTION_ALERT: Order failures, slippage issues

Alert Channels:
- Dashboard notifications (WebSocket)
- Log file alerts
- Console warnings
- Future: Email, SMS, Slack integration
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import websockets
import queue
import threading

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert categories"""
    RISK_ALERT = "risk_alert"
    PERFORMANCE_ALERT = "performance_alert"
    SYSTEM_ALERT = "system_alert"
    MARKET_ALERT = "market_alert"
    EXECUTION_ALERT = "execution_alert"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    context: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        data['alert_type'] = self.alert_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['severity'] = AlertSeverity(data['severity'])
        data['alert_type'] = AlertType(data['alert_type'])
        return cls(**data)


class AlertRule:
    """Base class for alert rules"""

    def __init__(self, rule_id: str, name: str, severity: AlertSeverity, alert_type: AlertType):
        self.rule_id = rule_id
        self.name = name
        self.severity = severity
        self.alert_type = alert_type
        self.enabled = True
        self.last_triggered = None
        self.min_interval = timedelta(minutes=5)  # Minimum time between alerts

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """Override this method to implement rule logic"""
        return False

    def can_trigger(self) -> bool:
        """Check if enough time has passed since last trigger"""
        if self.last_triggered is None:
            return True
        return datetime.now() - self.last_triggered >= self.min_interval

    def trigger(self, context: Dict[str, Any]) -> Optional[Alert]:
        """Trigger alert if conditions are met"""
        if not self.enabled or not self.can_trigger():
            return None

        if self.should_trigger(context):
            self.last_triggered = datetime.now()
            return self.create_alert(context)
        return None

    def create_alert(self, context: Dict[str, Any]) -> Alert:
        """Create alert from rule and context - override for custom alerts"""
        return Alert(
            id=f"{self.rule_id}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=self.severity,
            alert_type=self.alert_type,
            title=self.name,
            message=f"Alert triggered for {self.name}",
            context=context
        )


class RiskAlertRule(AlertRule):
    """Risk-based alert rules"""

    def __init__(self, rule_id: str, threshold: float, metric: str):
        super().__init__(
            rule_id=rule_id,
            name=f"Risk Alert: {metric}",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.RISK_ALERT
        )
        self.threshold = threshold
        self.metric = metric

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        value = context.get(self.metric, 0)
        return value > self.threshold

    def create_alert(self, context: Dict[str, Any]) -> Alert:
        value = context.get(self.metric, 0)
        return Alert(
            id=f"{self.rule_id}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=self.severity,
            alert_type=self.alert_type,
            title=f"Risk Alert: {self.metric.replace('_', ' ').title()}",
            message=f"{self.metric.replace('_', ' ').title()} reached {value:.2f}, exceeding threshold of {self.threshold}",
            context=context
        )


class PerformanceAlertRule(AlertRule):
    """Performance monitoring alert rules"""

    def __init__(self, rule_id: str, threshold: float, metric: str, is_minimum: bool = True):
        super().__init__(
            rule_id=rule_id,
            name=f"Performance Alert: {metric}",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.PERFORMANCE_ALERT
        )
        self.threshold = threshold
        self.metric = metric
        self.is_minimum = is_minimum  # True for minimum thresholds, False for maximum

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        value = context.get(self.metric, 0)
        if self.is_minimum:
            return value < self.threshold
        else:
            return value > self.threshold

    def create_alert(self, context: Dict[str, Any]) -> Alert:
        value = context.get(self.metric, 0)
        comparison = "below" if self.is_minimum else "above"
        return Alert(
            id=f"{self.rule_id}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=self.severity,
            alert_type=self.alert_type,
            title=f"Performance Alert: {self.metric.replace('_', ' ').title()}",
            message=f"{self.metric.replace('_', ' ').title()} is {value:.2f}, {comparison} threshold of {self.threshold}",
            context=context
        )


class SystemHealthAlertRule(AlertRule):
    """System health monitoring alert rules"""

    def __init__(self, rule_id: str, check_function: Callable[[], bool], message: str):
        super().__init__(
            rule_id=rule_id,
            name=f"System Health: {rule_id}",
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.SYSTEM_ALERT
        )
        self.check_function = check_function
        self.message = message

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        try:
            return not self.check_function()
        except Exception as e:
            logger.error(f"Error in health check {self.rule_id}: {e}")
            return True

    def create_alert(self, context: Dict[str, Any]) -> Alert:
        return Alert(
            id=f"{self.rule_id}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=self.severity,
            alert_type=self.alert_type,
            title=f"System Health Alert: {self.rule_id}",
            message=self.message,
            context=context
        )


class IntelligentAlertSystem:
    """
    Comprehensive alert management system
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert system

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.rules = []
        self.active_alerts = []
        self.alert_history = []
        self.alert_queue = queue.Queue()
        self.subscribers = []  # WebSocket connections

        # Alert persistence
        self.max_active_alerts = self.config.get('max_active_alerts', 100)
        self.max_history_size = self.config.get('max_history_size', 1000)

        # Initialize default rules
        self._initialize_default_rules()

        # Start background processing
        self._start_background_processor()

        logger.info("Intelligent alert system initialized")

    def _initialize_default_rules(self):
        """Initialize default alert rules"""

        # Risk alerts
        self.add_rule(RiskAlertRule("portfolio_drawdown", 0.05, "max_drawdown"))
        self.add_rule(RiskAlertRule("position_risk", 0.10, "position_risk"))
        self.add_rule(RiskAlertRule("portfolio_volatility", 0.30, "portfolio_volatility"))

        # Performance alerts
        self.add_rule(PerformanceAlertRule("low_sharpe", 0.5, "sharpe_ratio", is_minimum=True))
        self.add_rule(PerformanceAlertRule("negative_returns", -0.02, "daily_return", is_minimum=True))

        # System health alerts (simplified - would need actual health checks)
        self.add_rule(SystemHealthAlertRule(
            "data_connectivity",
            lambda: True,  # Placeholder - implement actual connectivity check
            "Data connectivity issue detected"
        ))

    def _start_background_processor(self):
        """Start background alert processing thread"""
        def process_alerts():
            while True:
                try:
                    # Process any queued alerts
                    while not self.alert_queue.empty():
                        alert = self.alert_queue.get_nowait()
                        self._process_alert(alert)

                    # Clean up old alerts
                    self._cleanup_alerts()

                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in alert processor: {e}")
                    time.sleep(5)

        processor_thread = threading.Thread(target=process_alerts, daemon=True)
        processor_thread.start()

    def add_rule(self, rule: AlertRule):
        """Add alert rule to system"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str):
        """Remove alert rule by ID"""
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        logger.info(f"Removed alert rule: {rule_id}")

    def evaluate_rules(self, context: Dict[str, Any]):
        """Evaluate all active rules against current context"""
        for rule in self.rules:
            if rule.enabled:
                alert = rule.trigger(context)
                if alert:
                    self.alert_queue.put(alert)

    def _process_alert(self, alert: Alert):
        """Process a new alert"""
        # Add to active alerts
        self.active_alerts.append(alert)

        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)

        logger.log(log_level, f"[ALERT] {alert.title}: {alert.message}")

        # Notify subscribers
        self._notify_subscribers(alert)

        # Console output for critical alerts
        if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            print(f"🚨 {alert.severity.value.upper()}: {alert.title}")
            print(f"   {alert.message}")

    def _notify_subscribers(self, alert: Alert):
        """Notify WebSocket subscribers of new alert"""
        if not self.subscribers:
            return

        message = {
            'type': 'alert',
            'data': alert.to_dict()
        }

        # Remove disconnected subscribers
        active_subscribers = []
        for ws in self.subscribers:
            try:
                asyncio.create_task(ws.send(json.dumps(message)))
                active_subscribers.append(ws)
            except Exception:
                pass  # Connection closed

        self.subscribers = active_subscribers

    def _cleanup_alerts(self):
        """Clean up old alerts"""
        # Move resolved/old active alerts to history
        cutoff_time = datetime.now() - timedelta(hours=24)

        for alert in self.active_alerts[:]:
            if alert.resolved or alert.timestamp < cutoff_time:
                self.active_alerts.remove(alert)
                self.alert_history.append(alert)

        # Limit active alerts count
        if len(self.active_alerts) > self.max_active_alerts:
            overflow = self.active_alerts[:-self.max_active_alerts]
            self.active_alerts = self.active_alerts[-self.max_active_alerts:]
            self.alert_history.extend(overflow)

        # Limit history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        alerts = self.active_alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status"""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = len([
                a for a in self.active_alerts if a.severity == severity
            ])

        active_by_type = {}
        for alert_type in AlertType:
            active_by_type[alert_type.value] = len([
                a for a in self.active_alerts if a.alert_type == alert_type
            ])

        return {
            'total_active_alerts': len(self.active_alerts),
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'active_by_severity': active_by_severity,
            'active_by_type': active_by_type,
            'last_alert': self.active_alerts[-1].to_dict() if self.active_alerts else None
        }

    def subscribe_websocket(self, websocket):
        """Subscribe a WebSocket connection for real-time alerts"""
        self.subscribers.append(websocket)
        logger.info("New WebSocket subscriber added")

    def unsubscribe_websocket(self, websocket):
        """Unsubscribe a WebSocket connection"""
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
            logger.info("WebSocket subscriber removed")

    def create_custom_alert(self, severity: AlertSeverity, alert_type: AlertType,
                          title: str, message: str, context: Dict[str, Any] = None) -> Alert:
        """Create and queue a custom alert"""
        alert = Alert(
            id=f"custom_{int(time.time())}",
            timestamp=datetime.now(),
            severity=severity,
            alert_type=alert_type,
            title=title,
            message=message,
            context=context or {}
        )

        self.alert_queue.put(alert)
        return alert

    def emergency_alert(self, message: str, context: Dict[str, Any] = None):
        """Create critical emergency alert"""
        return self.create_custom_alert(
            AlertSeverity.CRITICAL,
            AlertType.SYSTEM_ALERT,
            "EMERGENCY ALERT",
            message,
            context
        )


# Factory function for easy integration
def create_intelligent_alert_system(config: Optional[Dict[str, Any]] = None) -> IntelligentAlertSystem:
    """
    Factory function to create intelligent alert system

    Args:
        config: Optional configuration dictionary

    Returns:
        IntelligentAlertSystem instance
    """
    return IntelligentAlertSystem(config)


# Global alert system instance (singleton pattern)
_global_alert_system = None

def get_alert_system() -> IntelligentAlertSystem:
    """Get global alert system instance (creates if not exists)"""
    global _global_alert_system
    if _global_alert_system is None:
        _global_alert_system = create_intelligent_alert_system()
    return _global_alert_system


if __name__ == "__main__":
    # Test the alert system
    alert_system = create_intelligent_alert_system()

    # Test risk alert
    test_context = {
        'max_drawdown': 0.08,  # 8% drawdown
        'portfolio_volatility': 0.25,  # 25% volatility
        'sharpe_ratio': 0.3  # Low Sharpe ratio
    }

    print("Testing alert system...")
    alert_system.evaluate_rules(test_context)

    # Test custom alert
    alert_system.create_custom_alert(
        AlertSeverity.WARNING,
        AlertType.EXECUTION_ALERT,
        "Order Execution Delay",
        "Order for AAPL took longer than expected to execute"
    )

    # Wait for processing
    time.sleep(2)

    # Show results
    summary = alert_system.get_alert_summary()
    print(f"Alert summary: {json.dumps(summary, indent=2)}")

    active_alerts = alert_system.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  - {alert.severity.value}: {alert.title}")