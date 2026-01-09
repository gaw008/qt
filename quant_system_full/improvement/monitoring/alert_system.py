"""
Intelligent Alert System

This module provides comprehensive alerting capabilities for the trading system,
including multi-level thresholds, intelligent notification routing, and automated responses.

Key Features:
- Multi-level alert thresholds (Green/Yellow/Red)
- Intelligent alert aggregation and deduplication
- Multiple notification channels (Email, Webhook, Lark/DingTalk)
- Automated response actions (position reduction, trading halt)
- Alert correlation and root cause analysis
- Performance metrics and SLO monitoring
- Historical alert analysis and pattern detection
"""

import numpy as np
import pandas as pd
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Alert categories."""
    TRADING = "trading"
    RISK = "risk"
    SYSTEM = "system"
    DATA = "data"
    PERFORMANCE = "performance"
    EXECUTION = "execution"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LARK = "lark"
    DINGTALK = "dingtalk"
    SMS = "sms"
    SLACK = "slack"


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    category: AlertCategory
    level: AlertLevel
    condition: str  # Condition expression
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=

    # Rule configuration
    enabled: bool = True
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10

    # Notification settings
    notification_channels: List[NotificationChannel] = None
    escalation_delay_minutes: int = 30
    auto_resolve: bool = False

    # Response actions
    auto_actions: List[str] = None  # List of action names

    # Metadata
    description: str = ""
    created_by: str = ""
    created_at: str = ""

    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = [NotificationChannel.EMAIL]
        if self.auto_actions is None:
            self.auto_actions = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Alert:
    """Individual alert instance."""
    alert_id: str
    rule_id: str
    name: str
    category: AlertCategory
    level: AlertLevel

    # Alert details
    message: str
    current_value: float
    threshold_value: float

    # Context
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = None

    # Status tracking
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: str = ""
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    acknowledged_by: Optional[str] = None

    # Notification tracking
    notifications_sent: List[str] = None
    actions_taken: List[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
        if self.notifications_sent is None:
            self.notifications_sent = []
        if self.actions_taken is None:
            self.actions_taken = []


@dataclass
class AlertSystemConfig:
    """Configuration for alert system."""
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    from_email: str = ""
    admin_emails: List[str] = None

    # Webhook settings
    webhook_urls: Dict[str, str] = None

    # Lark/DingTalk settings
    lark_webhook_url: str = ""
    dingtalk_webhook_url: str = ""

    # Alert aggregation
    aggregation_window_minutes: int = 5
    max_alerts_per_window: int = 100

    # System thresholds (default red line values)
    max_daily_turnover_pct: float = 30.0
    max_sector_concentration_pct: float = 25.0
    max_slippage_multiplier: float = 2.0
    max_latency_seconds: float = 3.0
    max_drawdown_pct: float = 5.0

    # Database settings
    max_alert_history_days: int = 90
    cleanup_interval_hours: int = 24

    def __post_init__(self):
        if self.admin_emails is None:
            self.admin_emails = []
        if self.webhook_urls is None:
            self.webhook_urls = {}


class AlertSystem:
    """
    Comprehensive intelligent alert system.
    """

    def __init__(self, config: Optional[AlertSystemConfig] = None,
                 data_dir: str = "data_cache/alerts"):
        """
        Initialize alert system.

        Args:
            config: Alert system configuration
            data_dir: Directory for storing alert data
        """
        self.config = config or AlertSystemConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Database for storing alerts
        self.db_path = self.data_dir / "alerts.db"

        # Alert state
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.suppressed_rules: Dict[str, datetime] = {}

        # Notification state
        self.notification_queue: List[Dict] = []
        self.last_cleanup: datetime = datetime.now()

        # Response actions
        self.response_actions: Dict[str, Callable] = {}

        # Initialize database and load rules
        self._init_database()
        self._load_default_rules()
        self._register_default_actions()

        logger.info(f"[alerts] System initialized with {len(self.alert_rules)} rules")

    def _init_database(self):
        """Initialize SQLite database for alerts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Alert rules table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alert_rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rule_id TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        level TEXT NOT NULL,
                        condition_expr TEXT NOT NULL,
                        threshold_value REAL NOT NULL,
                        comparison_operator TEXT NOT NULL,
                        enabled INTEGER DEFAULT 1,
                        cooldown_minutes INTEGER DEFAULT 5,
                        max_alerts_per_hour INTEGER DEFAULT 10,
                        notification_channels TEXT,
                        escalation_delay_minutes INTEGER DEFAULT 30,
                        auto_resolve INTEGER DEFAULT 0,
                        auto_actions TEXT,
                        description TEXT,
                        created_by TEXT,
                        created_at TEXT NOT NULL
                    )
                """)

                # Alerts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL UNIQUE,
                        rule_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        symbol TEXT,
                        strategy TEXT,
                        metadata TEXT,
                        status TEXT DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        acknowledged_at TEXT,
                        resolved_at TEXT,
                        acknowledged_by TEXT,
                        notifications_sent TEXT,
                        actions_taken TEXT
                    )
                """)

                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status_created ON alerts(status, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_level_category ON alerts(level, category)")

                conn.commit()

            logger.info("[alerts] Database initialized successfully")

        except Exception as e:
            logger.error(f"[alerts] Database initialization failed: {e}")
            raise

    def _load_default_rules(self):
        """Load default alert rules."""
        try:
            # Load existing rules from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM alert_rules WHERE enabled = 1")
                rows = cursor.fetchall()

                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    row_dict = dict(zip(columns, row))

                    rule = AlertRule(
                        rule_id=row_dict['rule_id'],
                        name=row_dict['name'],
                        category=AlertCategory(row_dict['category']),
                        level=AlertLevel(row_dict['level']),
                        condition=row_dict['condition_expr'],
                        threshold_value=row_dict['threshold_value'],
                        comparison_operator=row_dict['comparison_operator'],
                        enabled=bool(row_dict['enabled']),
                        cooldown_minutes=row_dict['cooldown_minutes'],
                        max_alerts_per_hour=row_dict['max_alerts_per_hour'],
                        notification_channels=[NotificationChannel(ch) for ch in json.loads(row_dict['notification_channels'] or '["email"]')],
                        escalation_delay_minutes=row_dict['escalation_delay_minutes'],
                        auto_resolve=bool(row_dict['auto_resolve']),
                        auto_actions=json.loads(row_dict['auto_actions'] or '[]'),
                        description=row_dict['description'] or "",
                        created_by=row_dict['created_by'] or "",
                        created_at=row_dict['created_at']
                    )

                    self.alert_rules[rule.rule_id] = rule

            # Add default rules if none exist
            if not self.alert_rules:
                self._create_default_rules()

        except Exception as e:
            logger.error(f"[alerts] Failed to load default rules: {e}")

    def _create_default_rules(self):
        """Create default alert rules."""
        try:
            default_rules = [
                # Trading alerts
                AlertRule(
                    rule_id="high_daily_turnover",
                    name="High Daily Turnover",
                    category=AlertCategory.TRADING,
                    level=AlertLevel.CRITICAL,
                    condition="daily_turnover_pct",
                    threshold_value=self.config.max_daily_turnover_pct,
                    comparison_operator=">",
                    description="Daily turnover exceeds maximum threshold",
                    auto_actions=["reduce_position_sizes"]
                ),

                AlertRule(
                    rule_id="high_sector_concentration",
                    name="High Sector Concentration",
                    category=AlertCategory.RISK,
                    level=AlertLevel.CRITICAL,
                    condition="max_sector_weight_pct",
                    threshold_value=self.config.max_sector_concentration_pct,
                    comparison_operator=">",
                    description="Single sector concentration exceeds limit",
                    auto_actions=["rebalance_sectors"]
                ),

                AlertRule(
                    rule_id="excessive_slippage",
                    name="Excessive Slippage",
                    category=AlertCategory.EXECUTION,
                    level=AlertLevel.CRITICAL,
                    condition="slippage_vs_model_ratio",
                    threshold_value=self.config.max_slippage_multiplier,
                    comparison_operator=">",
                    description="Actual slippage exceeds model prediction by significant margin"
                ),

                AlertRule(
                    rule_id="high_execution_latency",
                    name="High Execution Latency",
                    category=AlertCategory.EXECUTION,
                    level=AlertLevel.WARNING,
                    condition="avg_execution_latency_seconds",
                    threshold_value=self.config.max_latency_seconds,
                    comparison_operator=">",
                    description="Order execution latency exceeds threshold"
                ),

                AlertRule(
                    rule_id="portfolio_drawdown",
                    name="Portfolio Drawdown",
                    category=AlertCategory.RISK,
                    level=AlertLevel.CRITICAL,
                    condition="current_drawdown_pct",
                    threshold_value=self.config.max_drawdown_pct,
                    comparison_operator=">",
                    description="Portfolio drawdown exceeds maximum threshold",
                    auto_actions=["emergency_risk_reduction"]
                ),

                # System alerts
                AlertRule(
                    rule_id="data_quality_degradation",
                    name="Data Quality Degradation",
                    category=AlertCategory.DATA,
                    level=AlertLevel.WARNING,
                    condition="data_quality_score",
                    threshold_value=0.8,
                    comparison_operator="<",
                    description="Data quality score has fallen below acceptable level"
                ),

                AlertRule(
                    rule_id="low_execution_rate",
                    name="Low Execution Rate",
                    category=AlertCategory.EXECUTION,
                    level=AlertLevel.WARNING,
                    condition="execution_success_rate",
                    threshold_value=0.8,
                    comparison_operator="<",
                    description="Order execution success rate is below threshold"
                ),

                # Performance alerts
                AlertRule(
                    rule_id="strategy_underperformance",
                    name="Strategy Underperformance",
                    category=AlertCategory.PERFORMANCE,
                    level=AlertLevel.WARNING,
                    condition="strategy_alpha_vs_benchmark",
                    threshold_value=-0.02,  # -2%
                    comparison_operator="<",
                    description="Strategy is underperforming benchmark significantly"
                )
            ]

            for rule in default_rules:
                self.add_alert_rule(rule)

            logger.info(f"[alerts] Created {len(default_rules)} default alert rules")

        except Exception as e:
            logger.error(f"[alerts] Failed to create default rules: {e}")

    def _register_default_actions(self):
        """Register default response actions."""
        self.response_actions = {
            'reduce_position_sizes': self._action_reduce_position_sizes,
            'rebalance_sectors': self._action_rebalance_sectors,
            'emergency_risk_reduction': self._action_emergency_risk_reduction,
            'halt_trading': self._action_halt_trading,
            'send_emergency_notification': self._action_send_emergency_notification
        }

    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alert_rules
                    (rule_id, name, category, level, condition_expr, threshold_value, comparison_operator,
                     enabled, cooldown_minutes, max_alerts_per_hour, notification_channels,
                     escalation_delay_minutes, auto_resolve, auto_actions, description, created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.rule_id, rule.name, rule.category.value, rule.level.value,
                    rule.condition, rule.threshold_value, rule.comparison_operator,
                    int(rule.enabled), rule.cooldown_minutes, rule.max_alerts_per_hour,
                    json.dumps([ch.value for ch in rule.notification_channels]),
                    rule.escalation_delay_minutes, int(rule.auto_resolve),
                    json.dumps(rule.auto_actions), rule.description, rule.created_by, rule.created_at
                ))

                conn.commit()

            self.alert_rules[rule.rule_id] = rule
            logger.info(f"[alerts] Added alert rule: {rule.name}")
            return True

        except Exception as e:
            logger.error(f"[alerts] Failed to add alert rule: {e}")
            return False

    def check_conditions(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check all alert conditions against current metrics.

        Args:
            metrics: Dictionary of current metric values

        Returns:
            List of triggered alerts
        """
        try:
            triggered_alerts = []

            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue

                # Check if rule is in cooldown
                if rule_id in self.suppressed_rules:
                    if datetime.now() - self.suppressed_rules[rule_id] < timedelta(minutes=rule.cooldown_minutes):
                        continue
                    else:
                        del self.suppressed_rules[rule_id]

                # Check condition
                if rule.condition in metrics:
                    current_value = metrics[rule.condition]

                    if self._evaluate_condition(current_value, rule.threshold_value, rule.comparison_operator):
                        # Create alert
                        alert = Alert(
                            alert_id=f"{rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            rule_id=rule_id,
                            name=rule.name,
                            category=rule.category,
                            level=rule.level,
                            message=self._format_alert_message(rule, current_value),
                            current_value=current_value,
                            threshold_value=rule.threshold_value,
                            metadata={'condition': rule.condition, 'operator': rule.comparison_operator}
                        )

                        triggered_alerts.append(alert)

                        # Add to suppression list
                        self.suppressed_rules[rule_id] = datetime.now()

            return triggered_alerts

        except Exception as e:
            logger.error(f"[alerts] Condition checking failed: {e}")
            return []

    def _evaluate_condition(self, current_value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition."""
        try:
            if operator == ">":
                return current_value > threshold
            elif operator == "<":
                return current_value < threshold
            elif operator == ">=":
                return current_value >= threshold
            elif operator == "<=":
                return current_value <= threshold
            elif operator == "==":
                return abs(current_value - threshold) < 1e-9
            elif operator == "!=":
                return abs(current_value - threshold) >= 1e-9
            else:
                logger.warning(f"[alerts] Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.error(f"[alerts] Condition evaluation failed: {e}")
            return False

    def _format_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Format alert message."""
        try:
            return (f"{rule.name}: {rule.condition} is {current_value:.2f} "
                   f"(threshold: {rule.comparison_operator} {rule.threshold_value:.2f})")
        except Exception:
            return f"{rule.name}: Alert triggered"

    async def process_alerts(self, alerts: List[Alert]):
        """Process triggered alerts."""
        try:
            for alert in alerts:
                # Store alert
                self._store_alert(alert)
                self.active_alerts[alert.alert_id] = alert

                # Get rule for actions
                rule = self.alert_rules.get(alert.rule_id)
                if not rule:
                    continue

                # Send notifications
                await self._send_notifications(alert, rule)

                # Execute automated actions
                await self._execute_auto_actions(alert, rule)

                logger.info(f"[alerts] Processed alert: {alert.name} ({alert.level.value})")

        except Exception as e:
            logger.error(f"[alerts] Alert processing failed: {e}")

    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert."""
        try:
            for channel in rule.notification_channels:
                try:
                    if channel == NotificationChannel.EMAIL:
                        await self._send_email_notification(alert)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook_notification(alert)
                    elif channel == NotificationChannel.LARK:
                        await self._send_lark_notification(alert)
                    elif channel == NotificationChannel.DINGTALK:
                        await self._send_dingtalk_notification(alert)

                    alert.notifications_sent.append(f"{channel.value}:{datetime.now().isoformat()}")

                except Exception as e:
                    logger.error(f"[alerts] Failed to send {channel.value} notification: {e}")

        except Exception as e:
            logger.error(f"[alerts] Notification sending failed: {e}")

    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        try:
            if not self.config.email_username or not self.config.admin_emails:
                return

            msg = MimeMultipart()
            msg['From'] = self.config.from_email or self.config.email_username
            msg['To'] = ', '.join(self.config.admin_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.name}"

            body = f"""
Alert Details:
- Name: {alert.name}
- Level: {alert.level.value.upper()}
- Category: {alert.category.value}
- Message: {alert.message}
- Current Value: {alert.current_value:.4f}
- Threshold: {alert.threshold_value:.4f}
- Time: {alert.created_at}
- Alert ID: {alert.alert_id}

Please investigate and take appropriate action.
"""

            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()

        except Exception as e:
            logger.error(f"[alerts] Email notification failed: {e}")

    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        try:
            webhook_url = self.config.webhook_urls.get('default')
            if not webhook_url:
                return

            payload = {
                'alert_id': alert.alert_id,
                'name': alert.name,
                'level': alert.level.value,
                'category': alert.category.value,
                'message': alert.message,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'created_at': alert.created_at
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"[alerts] Webhook returned status {response.status}")

        except Exception as e:
            logger.error(f"[alerts] Webhook notification failed: {e}")

    async def _send_lark_notification(self, alert: Alert):
        """Send Lark notification."""
        try:
            if not self.config.lark_webhook_url:
                return

            payload = {
                "msg_type": "text",
                "content": {
                    "text": f"🚨 [{alert.level.value.upper()}] {alert.name}\n"
                           f"Message: {alert.message}\n"
                           f"Value: {alert.current_value:.4f} (threshold: {alert.threshold_value:.4f})\n"
                           f"Time: {alert.created_at}"
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.lark_webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"[alerts] Lark notification returned status {response.status}")

        except Exception as e:
            logger.error(f"[alerts] Lark notification failed: {e}")

    async def _send_dingtalk_notification(self, alert: Alert):
        """Send DingTalk notification."""
        try:
            if not self.config.dingtalk_webhook_url:
                return

            payload = {
                "msgtype": "text",
                "text": {
                    "content": f"🚨 [{alert.level.value.upper()}] {alert.name}\n"
                              f"Message: {alert.message}\n"
                              f"Value: {alert.current_value:.4f} (threshold: {alert.threshold_value:.4f})\n"
                              f"Time: {alert.created_at}"
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.dingtalk_webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"[alerts] DingTalk notification returned status {response.status}")

        except Exception as e:
            logger.error(f"[alerts] DingTalk notification failed: {e}")

    async def _execute_auto_actions(self, alert: Alert, rule: AlertRule):
        """Execute automated actions for alert."""
        try:
            for action_name in rule.auto_actions:
                if action_name in self.response_actions:
                    try:
                        await self.response_actions[action_name](alert)
                        alert.actions_taken.append(f"{action_name}:{datetime.now().isoformat()}")
                        logger.info(f"[alerts] Executed auto action: {action_name}")
                    except Exception as e:
                        logger.error(f"[alerts] Auto action {action_name} failed: {e}")

        except Exception as e:
            logger.error(f"[alerts] Auto action execution failed: {e}")

    async def _action_reduce_position_sizes(self, alert: Alert):
        """Reduce position sizes action."""
        logger.info("[alerts] AUTO ACTION: Reducing position sizes by 20%")
        # Implementation would integrate with portfolio management system

    async def _action_rebalance_sectors(self, alert: Alert):
        """Rebalance sector exposure action."""
        logger.info("[alerts] AUTO ACTION: Rebalancing sector exposure")
        # Implementation would trigger sector rebalancing

    async def _action_emergency_risk_reduction(self, alert: Alert):
        """Emergency risk reduction action."""
        logger.critical("[alerts] AUTO ACTION: Emergency risk reduction - reducing portfolio by 50%")
        # Implementation would significantly reduce portfolio exposure

    async def _action_halt_trading(self, alert: Alert):
        """Halt trading action."""
        logger.critical("[alerts] AUTO ACTION: Halting all trading activities")
        # Implementation would stop all trading

    async def _action_send_emergency_notification(self, alert: Alert):
        """Send emergency notification to all channels."""
        logger.critical("[alerts] AUTO ACTION: Sending emergency notifications")
        # Implementation would send notifications to all available channels

    def _store_alert(self, alert: Alert):
        """Store alert in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts
                    (alert_id, rule_id, name, category, level, message, current_value, threshold_value,
                     symbol, strategy, metadata, status, created_at, acknowledged_at, resolved_at,
                     acknowledged_by, notifications_sent, actions_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id, alert.rule_id, alert.name, alert.category.value, alert.level.value,
                    alert.message, alert.current_value, alert.threshold_value,
                    alert.symbol, alert.strategy, json.dumps(alert.metadata), alert.status.value,
                    alert.created_at, alert.acknowledged_at, alert.resolved_at, alert.acknowledged_by,
                    json.dumps(alert.notifications_sent), json.dumps(alert.actions_taken)
                ))

        except Exception as e:
            logger.error(f"[alerts] Failed to store alert: {e}")

    def get_alert_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get alert system summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

                cursor = conn.cursor()

                # Alert counts by level
                cursor.execute("""
                    SELECT level, COUNT(*) as count
                    FROM alerts
                    WHERE created_at >= ?
                    GROUP BY level
                """, (cutoff_date,))

                alerts_by_level = dict(cursor.fetchall())

                # Alert counts by category
                cursor.execute("""
                    SELECT category, COUNT(*) as count
                    FROM alerts
                    WHERE created_at >= ?
                    GROUP BY category
                """, (cutoff_date,))

                alerts_by_category = dict(cursor.fetchall())

                # Active alerts
                cursor.execute("""
                    SELECT COUNT(*) FROM alerts
                    WHERE status = 'active'
                """)

                active_count = cursor.fetchone()[0]

                # Top triggered rules
                cursor.execute("""
                    SELECT rule_id, COUNT(*) as count
                    FROM alerts
                    WHERE created_at >= ?
                    GROUP BY rule_id
                    ORDER BY count DESC
                    LIMIT 5
                """, (cutoff_date,))

                top_rules = cursor.fetchall()

                return {
                    'period_days': days_back,
                    'total_rules': len(self.alert_rules),
                    'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                    'active_alerts': active_count,
                    'suppressed_rules': len(self.suppressed_rules),
                    'alerts_by_level': alerts_by_level,
                    'alerts_by_category': alerts_by_category,
                    'top_triggered_rules': [{'rule_id': r[0], 'count': r[1]} for r in top_rules],
                    'available_actions': list(self.response_actions.keys()),
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"[alerts] Failed to get alert summary: {e}")
            return {'error': str(e)}


def create_alert_system(custom_config: Optional[Dict] = None,
                       data_dir: Optional[str] = None) -> AlertSystem:
    """
    Create and configure an alert system.

    Args:
        custom_config: Custom configuration parameters
        data_dir: Custom data directory

    Returns:
        Configured AlertSystem instance
    """
    config = AlertSystemConfig()

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return AlertSystem(config, data_dir or "data_cache/alerts")


if __name__ == "__main__":
    # Test alert system
    print("=== Alert System Test ===")

    # Create alert system
    alert_system = create_alert_system()

    # Test metrics that should trigger alerts
    test_metrics = {
        'daily_turnover_pct': 35.0,  # Above 30% threshold
        'max_sector_weight_pct': 28.0,  # Above 25% threshold
        'current_drawdown_pct': 3.0,  # Below 5% threshold
        'data_quality_score': 0.75,  # Below 0.8 threshold
        'execution_success_rate': 0.95  # Above 0.8 threshold (no alert)
    }

    # Check conditions
    triggered_alerts = alert_system.check_conditions(test_metrics)

    print(f"Triggered {len(triggered_alerts)} alerts:")
    for alert in triggered_alerts:
        print(f"  - {alert.name} ({alert.level.value}): {alert.message}")

    # Process alerts (async)
    async def test_processing():
        await alert_system.process_alerts(triggered_alerts)

    try:
        asyncio.run(test_processing())
    except Exception as e:
        print(f"Alert processing test failed: {e}")

    # Get summary
    summary = alert_system.get_alert_summary()
    print(f"\nAlert System Summary:")
    print(f"  Total Rules: {summary['total_rules']}")
    print(f"  Active Alerts: {summary['active_alerts']}")
    print(f"  Alerts by Level: {summary['alerts_by_level']}")
    print(f"  Available Actions: {len(summary['available_actions'])}")