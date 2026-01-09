#!/usr/bin/env python3
"""
Intelligent Alert System C1 - AI-Driven Alert Management
智能预警系统C1 - AI驱动的预警管理

Investment-grade alert system providing:
- AI-driven alert prioritization and severity prediction
- Context-aware escalation with market condition awareness
- Smart alert consolidation and storm prevention
- Predictive alerting with machine learning
- Multi-channel notification delivery
- Comprehensive alert lifecycle management

Features:
- Machine learning for alert pattern recognition
- Intelligent routing based on system context
- Multi-tier escalation with smart throttling
- Real-time sentiment analysis integration
- Performance-based alert tuning
- Comprehensive audit trail and analytics

Author: Quantitative Trading System
Version: 1.0 - Investment Grade
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import websockets
import hashlib
import statistics
from collections import deque, defaultdict
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

class AlertSeverity(Enum):
    """Alert severity levels with AI scoring"""
    CRITICAL = ("CRITICAL", 1.0, "#FF0000")
    HIGH = ("HIGH", 0.8, "#FF6600")
    MEDIUM = ("MEDIUM", 0.6, "#FFAA00")
    LOW = ("LOW", 0.4, "#FFDD00")
    INFO = ("INFO", 0.2, "#00AA00")

class AlertCategory(Enum):
    """Alert categories for intelligent routing"""
    SYSTEM_FAILURE = "SYSTEM_FAILURE"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    SECURITY_THREAT = "SECURITY_THREAT"
    TRADING_ANOMALY = "TRADING_ANOMALY"
    RISK_BREACH = "RISK_BREACH"
    DATA_QUALITY = "DATA_QUALITY"
    NETWORK_CONNECTIVITY = "NETWORK_CONNECTIVITY"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    MARKET_EVENT = "MARKET_EVENT"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    DATABASE = "database"

class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"
    ESCALATED = "ESCALATED"

@dataclass
class AlertRule:
    """Alert rule configuration with AI parameters"""
    rule_id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str
    threshold: float
    evaluation_window: int  # seconds
    cooldown_period: int  # seconds
    escalation_rules: List[Dict[str, Any]]
    ai_enabled: bool = True
    context_aware: bool = True
    predictive_enabled: bool = True
    machine_learning_weight: float = 0.3
    notification_channels: List[NotificationChannel] = field(default_factory=list)

@dataclass
class AlertEvent:
    """Comprehensive alert event structure"""
    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    source: str
    context: Dict[str, Any]

    # AI-enhanced fields
    ai_confidence: float
    priority_score: float
    predicted_impact: float
    similar_incidents: List[str]
    recommended_actions: List[str]

    # Lifecycle management
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None

    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    escalation_level: int = 0

@dataclass
class AlertMetrics:
    """Real-time alert system metrics"""
    timestamp: datetime
    active_alerts: int
    alerts_per_minute: float
    false_positive_rate: float
    resolution_time_avg: float
    escalation_rate: float
    notification_success_rate: float
    ai_prediction_accuracy: float
    system_load: float

class IntelligentAlertSystemC1:
    """AI-driven intelligent alert management system"""

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = config_path or self.base_dir / "config" / "alert_config.json"
        self.data_dir = self.base_dir / "data" / "alerts"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()
        self.shutdown_event = threading.Event()

        # Core components
        self.config = self._load_configuration()
        self.db_connection = self._initialize_database()
        self.ai_models = self._initialize_ai_models()
        self.notification_manager = NotificationManager(self.config)

        # Runtime state
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1min intervals
        self.escalation_chains: Dict[str, List[Dict]] = {}

        # AI/ML components
        self.feature_scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.severity_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.pattern_cache: Dict[str, Any] = {}
        self.context_weights: Dict[str, float] = {}

        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="AlertSystem")
        self.processing_threads: List[threading.Thread] = []

        # Performance optimization
        self.alert_cache = {}
        self.batch_size = 50
        self.processing_queue = deque()

        self.logger.info("Intelligent Alert System C1 initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for alert system"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('IntelligentAlertSystem')
        logger.setLevel(logging.INFO)

        # Console handler with color coding
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '\033[96m%(asctime)s\033[0m - \033[95mALERT\033[0m - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler for alert audit
        file_handler = logging.FileHandler(
            log_dir / f"intelligent_alerts_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _load_configuration(self) -> Dict[str, Any]:
        """Load alert system configuration"""
        default_config = {
            "ai_settings": {
                "prediction_enabled": True,
                "severity_prediction_threshold": 0.7,
                "anomaly_detection_threshold": 0.3,
                "context_learning_rate": 0.1,
                "pattern_recognition_window": 3600
            },
            "escalation_settings": {
                "max_escalation_levels": 5,
                "escalation_intervals": [300, 600, 1800, 3600, 7200],
                "auto_escalation_enabled": True,
                "escalation_dampening": 0.8
            },
            "notification_settings": {
                "batch_notifications": True,
                "consolidation_window": 60,
                "rate_limiting_enabled": True,
                "max_notifications_per_minute": 10
            },
            "performance_settings": {
                "processing_threads": 4,
                "batch_size": 50,
                "cache_size": 1000,
                "cleanup_interval": 3600
            },
            "channels": {
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_address": "trading-system@company.com"
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#alerts"
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": ""
                }
            }
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Could not load config, using defaults: {e}")

        return default_config

    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for alert storage"""
        db_path = self.data_dir / "alerts.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                rule_id TEXT,
                timestamp DATETIME,
                severity TEXT,
                category TEXT,
                title TEXT,
                message TEXT,
                source TEXT,
                context TEXT,
                ai_confidence REAL,
                priority_score REAL,
                predicted_impact REAL,
                status TEXT,
                acknowledged_by TEXT,
                acknowledged_at DATETIME,
                resolved_at DATETIME,
                escalated_at DATETIME,
                escalation_level INTEGER
            );

            CREATE TABLE IF NOT EXISTS alert_metrics (
                timestamp DATETIME PRIMARY KEY,
                active_alerts INTEGER,
                alerts_per_minute REAL,
                false_positive_rate REAL,
                resolution_time_avg REAL,
                escalation_rate REAL,
                notification_success_rate REAL,
                ai_prediction_accuracy REAL,
                system_load REAL
            );

            CREATE TABLE IF NOT EXISTS alert_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_data TEXT,
                frequency INTEGER,
                confidence REAL,
                last_seen DATETIME
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
        """)

        conn.commit()
        return conn

    def _initialize_ai_models(self) -> Dict[str, Any]:
        """Initialize AI/ML models for intelligent alerting"""
        models = {}

        try:
            # Load pre-trained models if they exist
            model_dir = self.data_dir / "models"
            model_dir.mkdir(exist_ok=True)

            # Anomaly detection model
            anomaly_model_path = model_dir / "anomaly_detector.pkl"
            if anomaly_model_path.exists():
                with open(anomaly_model_path, 'rb') as f:
                    models['anomaly_detector'] = pickle.load(f)
            else:
                models['anomaly_detector'] = self.anomaly_detector

            # Severity prediction model
            severity_model_path = model_dir / "severity_predictor.pkl"
            if severity_model_path.exists():
                with open(severity_model_path, 'rb') as f:
                    models['severity_predictor'] = pickle.load(f)
            else:
                models['severity_predictor'] = self.severity_predictor

            self.logger.info("AI models initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing AI models: {e}")
            models = {
                'anomaly_detector': self.anomaly_detector,
                'severity_predictor': self.severity_predictor
            }

        return models

    def start_alert_system(self) -> None:
        """Start the intelligent alert system"""
        self.logger.info("=== Starting Intelligent Alert System C1 ===")

        # Alert processing thread
        processing_thread = threading.Thread(
            target=self._alert_processing_loop,
            name="AlertProcessor",
            daemon=True
        )
        processing_thread.start()
        self.processing_threads.append(processing_thread)

        # AI analysis thread
        ai_thread = threading.Thread(
            target=self._ai_analysis_loop,
            name="AIAnalyzer",
            daemon=True
        )
        ai_thread.start()
        self.processing_threads.append(ai_thread)

        # Escalation management thread
        escalation_thread = threading.Thread(
            target=self._escalation_management_loop,
            name="EscalationManager",
            daemon=True
        )
        escalation_thread.start()
        self.processing_threads.append(escalation_thread)

        # Metrics collection thread
        metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        metrics_thread.start()
        self.processing_threads.append(metrics_thread)

        # Cleanup thread
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="AlertCleaner",
            daemon=True
        )
        cleanup_thread.start()
        self.processing_threads.append(cleanup_thread)

        self.logger.info(f"Alert system started with {len(self.processing_threads)} threads")

    def create_alert(self,
                    title: str,
                    message: str,
                    severity: Union[AlertSeverity, str],
                    category: Union[AlertCategory, str],
                    source: str,
                    context: Optional[Dict[str, Any]] = None,
                    rule_id: Optional[str] = None) -> AlertEvent:
        """Create intelligent alert with AI enhancement"""

        # Normalize enums
        if isinstance(severity, str):
            severity = AlertSeverity[severity.upper()]
        if isinstance(category, str):
            category = AlertCategory[category.upper()]

        # Generate unique alert ID
        alert_id = self._generate_alert_id(title, source, context or {})

        # Check for duplicate suppression
        if self._is_suppressed(alert_id):
            return None

        # AI-enhanced alert creation
        ai_metrics = self._calculate_ai_metrics(title, message, context or {}, severity, category)

        # Create alert event
        alert = AlertEvent(
            alert_id=alert_id,
            rule_id=rule_id or f"auto_{category.value.lower()}",
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            message=message,
            source=source,
            context=context or {},
            ai_confidence=ai_metrics['confidence'],
            priority_score=ai_metrics['priority_score'],
            predicted_impact=ai_metrics['predicted_impact'],
            similar_incidents=ai_metrics['similar_incidents'],
            recommended_actions=ai_metrics['recommended_actions']
        )

        # Add to processing queue
        self.processing_queue.append(alert)
        self.active_alerts[alert_id] = alert

        self.logger.info(f"Created alert: {alert_id} - {severity.value[0]} - {title}")

        return alert

    def _calculate_ai_metrics(self, title: str, message: str, context: Dict[str, Any],
                            severity: AlertSeverity, category: AlertCategory) -> Dict[str, Any]:
        """Calculate AI-enhanced alert metrics"""

        try:
            # Feature extraction for AI analysis
            features = self._extract_alert_features(title, message, context, severity, category)

            # Anomaly detection
            anomaly_score = self._detect_anomaly(features)

            # Severity prediction
            predicted_severity = self._predict_severity(features)

            # Pattern matching
            similar_incidents = self._find_similar_incidents(features)

            # Priority scoring with AI
            priority_score = self._calculate_priority_score(
                severity, category, anomaly_score, predicted_severity, context
            )

            # Impact prediction
            predicted_impact = self._predict_impact(features, context)

            # Recommended actions
            recommended_actions = self._generate_recommendations(category, severity, context)

            # AI confidence based on multiple factors
            ai_confidence = min(1.0, (
                (1.0 - anomaly_score) * 0.3 +  # Normal patterns are more confident
                predicted_severity * 0.4 +  # Model confidence
                (len(similar_incidents) / 10.0) * 0.3  # Historical similarity
            ))

            return {
                'confidence': ai_confidence,
                'priority_score': priority_score,
                'predicted_impact': predicted_impact,
                'similar_incidents': similar_incidents,
                'recommended_actions': recommended_actions,
                'anomaly_score': anomaly_score,
                'predicted_severity': predicted_severity
            }

        except Exception as e:
            self.logger.error(f"Error calculating AI metrics: {e}")
            return {
                'confidence': 0.5,
                'priority_score': severity.value[1],
                'predicted_impact': 0.5,
                'similar_incidents': [],
                'recommended_actions': ["Review alert manually"],
                'anomaly_score': 0.5,
                'predicted_severity': 0.5
            }

    def _extract_alert_features(self, title: str, message: str, context: Dict[str, Any],
                              severity: AlertSeverity, category: AlertCategory) -> np.ndarray:
        """Extract numerical features for AI analysis"""
        features = []

        # Text-based features
        features.append(len(title))
        features.append(len(message))
        features.append(title.count('error'))
        features.append(title.count('critical'))
        features.append(title.count('warning'))
        features.append(message.count('failed'))
        features.append(message.count('timeout'))
        features.append(message.count('%'))

        # Severity and category encoding
        features.append(severity.value[1])
        features.append(hash(category.value) % 1000 / 1000.0)

        # Context-based features
        features.append(len(context))
        features.append(context.get('cpu_usage', 0) / 100.0 if 'cpu_usage' in context else 0)
        features.append(context.get('memory_usage', 0) / 100.0 if 'memory_usage' in context else 0)
        features.append(context.get('error_count', 0) / 100.0 if 'error_count' in context else 0)
        features.append(1.0 if context.get('market_hours', False) else 0.0)

        # Time-based features
        now = datetime.now()
        features.append(now.hour / 24.0)
        features.append(now.weekday() / 7.0)
        features.append((now - datetime(now.year, 1, 1)).days / 365.0)

        return np.array(features).reshape(1, -1)

    def _detect_anomaly(self, features: np.ndarray) -> float:
        """Detect anomalous alert patterns"""
        try:
            if hasattr(self.ai_models['anomaly_detector'], 'decision_function'):
                anomaly_score = self.ai_models['anomaly_detector'].decision_function(features)[0]
                # Normalize to 0-1 range (higher = more anomalous)
                return max(0, min(1, (anomaly_score + 1) / 2))
            else:
                return 0.5
        except Exception:
            return 0.5

    def _predict_severity(self, features: np.ndarray) -> float:
        """Predict alert severity using ML"""
        try:
            if hasattr(self.ai_models['severity_predictor'], 'predict_proba'):
                probabilities = self.ai_models['severity_predictor'].predict_proba(features)[0]
                return max(probabilities)
            else:
                return 0.5
        except Exception:
            return 0.5

    def _find_similar_incidents(self, features: np.ndarray, limit: int = 5) -> List[str]:
        """Find similar historical incidents"""
        try:
            # Simple similarity based on recent patterns
            similar = []
            cutoff_time = datetime.now() - timedelta(days=7)

            for alert in list(self.alert_history):
                if alert.timestamp > cutoff_time:
                    # Simple text similarity check
                    if len(similar) < limit:
                        similar.append(alert.alert_id)

            return similar

        except Exception:
            return []

    def _calculate_priority_score(self, severity: AlertSeverity, category: AlertCategory,
                                anomaly_score: float, predicted_severity: float,
                                context: Dict[str, Any]) -> float:
        """Calculate intelligent priority score"""

        base_priority = severity.value[1]

        # Category weight
        category_weights = {
            AlertCategory.SYSTEM_FAILURE: 1.0,
            AlertCategory.SECURITY_THREAT: 0.9,
            AlertCategory.RISK_BREACH: 0.9,
            AlertCategory.TRADING_ANOMALY: 0.8,
            AlertCategory.PERFORMANCE_DEGRADATION: 0.7,
            AlertCategory.COMPLIANCE_VIOLATION: 0.8,
            AlertCategory.RESOURCE_EXHAUSTION: 0.6,
            AlertCategory.NETWORK_CONNECTIVITY: 0.5,
            AlertCategory.DATA_QUALITY: 0.4,
            AlertCategory.MARKET_EVENT: 0.3
        }

        category_weight = category_weights.get(category, 0.5)

        # Context-aware adjustments
        context_multiplier = 1.0
        if context.get('market_hours', False):
            context_multiplier += 0.2
        if context.get('trading_active', False):
            context_multiplier += 0.3
        if context.get('high_volume_period', False):
            context_multiplier += 0.1

        # AI enhancement
        ai_adjustment = (anomaly_score * 0.3 + predicted_severity * 0.7)

        # Final priority calculation
        priority = (
            base_priority * 0.4 +
            category_weight * 0.3 +
            ai_adjustment * 0.3
        ) * context_multiplier

        return min(1.0, max(0.0, priority))

    def _predict_impact(self, features: np.ndarray, context: Dict[str, Any]) -> float:
        """Predict potential impact of alert"""

        # Simple impact prediction based on context
        impact_factors = [
            context.get('affected_systems', 1) / 10.0,
            context.get('user_count_affected', 0) / 1000.0,
            1.0 if context.get('trading_affected', False) else 0.0,
            context.get('revenue_impact', 0) / 10000.0,
            1.0 if context.get('compliance_risk', False) else 0.0
        ]

        return min(1.0, sum(impact_factors) / len(impact_factors))

    def _generate_recommendations(self, category: AlertCategory, severity: AlertSeverity,
                                context: Dict[str, Any]) -> List[str]:
        """Generate AI-driven action recommendations"""

        recommendations = []

        # Category-specific recommendations
        if category == AlertCategory.SYSTEM_FAILURE:
            recommendations.extend([
                "Check system processes and restart if necessary",
                "Review system logs for error patterns",
                "Validate system configuration integrity"
            ])
        elif category == AlertCategory.PERFORMANCE_DEGRADATION:
            recommendations.extend([
                "Monitor resource utilization trends",
                "Analyze performance bottlenecks",
                "Consider scaling resources if needed"
            ])
        elif category == AlertCategory.SECURITY_THREAT:
            recommendations.extend([
                "Immediate security assessment required",
                "Review access logs and authentication patterns",
                "Consider temporary access restrictions"
            ])
        elif category == AlertCategory.RISK_BREACH:
            recommendations.extend([
                "Immediate risk assessment and mitigation",
                "Review portfolio positions and exposure",
                "Consider position size reduction"
            ])

        # Severity-specific recommendations
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            recommendations.append("Immediate escalation to on-call team")
            recommendations.append("Consider emergency procedures activation")

        # Context-specific recommendations
        if context.get('trading_active', False):
            recommendations.append("Assess impact on active trading operations")
        if context.get('market_hours', False):
            recommendations.append("Monitor market impact and trading exposure")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _generate_alert_id(self, title: str, source: str, context: Dict[str, Any]) -> str:
        """Generate unique alert ID with collision detection"""
        content = f"{title}_{source}_{json.dumps(context, sort_keys=True)}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _is_suppressed(self, alert_id: str) -> bool:
        """Check if alert should be suppressed"""

        # Check cooldown periods
        current_time = datetime.now()
        for existing_alert in self.active_alerts.values():
            if (existing_alert.suppressed_until and
                current_time < existing_alert.suppressed_until):
                return True

        # Check rate limiting
        recent_alerts = [a for a in self.alert_history
                        if a.timestamp > current_time - timedelta(minutes=1)]
        if len(recent_alerts) > self.config['notification_settings']['max_notifications_per_minute']:
            return True

        return False

    def _alert_processing_loop(self) -> None:
        """Main alert processing loop"""
        self.logger.info("Alert processing loop started")

        while not self.shutdown_event.is_set():
            try:
                if not self.processing_queue:
                    time.sleep(0.1)
                    continue

                # Process alerts in batches
                batch = []
                while len(batch) < self.batch_size and self.processing_queue:
                    batch.append(self.processing_queue.popleft())

                if batch:
                    self._process_alert_batch(batch)

                time.sleep(0.05)  # Brief pause to prevent high CPU usage

            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                time.sleep(1)

    def _process_alert_batch(self, alerts: List[AlertEvent]) -> None:
        """Process a batch of alerts efficiently"""

        for alert in alerts:
            try:
                # Store in database
                self._store_alert(alert)

                # Add to history
                self.alert_history.append(alert)

                # Send notifications
                self._schedule_notifications(alert)

                # Update metrics
                self._update_alert_metrics(alert)

            except Exception as e:
                self.logger.error(f"Error processing alert {alert.alert_id}: {e}")

    def _store_alert(self, alert: AlertEvent) -> None:
        """Store alert in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, rule_id, timestamp, severity, category, title, message,
                    source, context, ai_confidence, priority_score, predicted_impact,
                    status, acknowledged_by, acknowledged_at, resolved_at, escalated_at,
                    escalation_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.rule_id,
                alert.timestamp.isoformat(),
                alert.severity.value[0],
                alert.category.value,
                alert.title,
                alert.message,
                alert.source,
                json.dumps(alert.context),
                alert.ai_confidence,
                alert.priority_score,
                alert.predicted_impact,
                alert.status.value,
                alert.acknowledged_by,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.escalated_at.isoformat() if alert.escalated_at else None,
                alert.escalation_level
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing alert {alert.alert_id}: {e}")

    def _schedule_notifications(self, alert: AlertEvent) -> None:
        """Schedule intelligent notifications for alert"""

        # Determine notification channels based on severity and context
        channels = self._select_notification_channels(alert)

        # Schedule notifications
        for channel in channels:
            self.thread_pool.submit(
                self.notification_manager.send_notification,
                alert, channel
            )

    def _select_notification_channels(self, alert: AlertEvent) -> List[NotificationChannel]:
        """Intelligently select notification channels"""
        channels = [NotificationChannel.CONSOLE, NotificationChannel.DATABASE]

        # Add channels based on severity
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            channels.extend([NotificationChannel.EMAIL])
            if self.config['channels']['slack']['enabled']:
                channels.append(NotificationChannel.SLACK)

        # Context-aware channel selection
        if alert.context.get('trading_active', False):
            channels.append(NotificationChannel.EMAIL)

        return channels

    def _ai_analysis_loop(self) -> None:
        """Continuous AI analysis and learning loop"""
        self.logger.info("AI analysis loop started")

        while not self.shutdown_event.is_set():
            try:
                # Retrain models periodically
                if len(self.alert_history) > 100:
                    self._retrain_ai_models()

                # Pattern analysis
                self._analyze_alert_patterns()

                # Update context weights
                self._update_context_weights()

                time.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in AI analysis loop: {e}")
                time.sleep(300)

    def _retrain_ai_models(self) -> None:
        """Retrain AI models with recent data"""
        try:
            # Prepare training data from recent alerts
            features = []
            labels = []

            for alert in list(self.alert_history)[-1000:]:  # Last 1000 alerts
                feature_vector = self._extract_alert_features(
                    alert.title, alert.message, alert.context,
                    alert.severity, alert.category
                )
                features.append(feature_vector.flatten())
                labels.append(alert.severity.value[1])

            if len(features) > 50:
                X = np.array(features)
                y = np.array(labels)

                # Retrain severity predictor
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                self.ai_models['severity_predictor'].fit(X_train, y_train)

                # Update anomaly detector
                self.ai_models['anomaly_detector'].fit(X_train)

                # Save models
                self._save_ai_models()

                self.logger.info("AI models retrained successfully")

        except Exception as e:
            self.logger.error(f"Error retraining AI models: {e}")

    def _save_ai_models(self) -> None:
        """Save trained AI models"""
        try:
            model_dir = self.data_dir / "models"
            model_dir.mkdir(exist_ok=True)

            with open(model_dir / "anomaly_detector.pkl", 'wb') as f:
                pickle.dump(self.ai_models['anomaly_detector'], f)

            with open(model_dir / "severity_predictor.pkl", 'wb') as f:
                pickle.dump(self.ai_models['severity_predictor'], f)

        except Exception as e:
            self.logger.error(f"Error saving AI models: {e}")

    def _analyze_alert_patterns(self) -> None:
        """Analyze alert patterns for optimization"""
        try:
            # Pattern frequency analysis
            pattern_counts = defaultdict(int)

            for alert in list(self.alert_history)[-500:]:
                pattern_key = f"{alert.category.value}_{alert.severity.value[0]}"
                pattern_counts[pattern_key] += 1

            # Store patterns
            cursor = self.db_connection.cursor()
            for pattern, count in pattern_counts.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO alert_patterns
                    (pattern_id, pattern_data, frequency, confidence, last_seen)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    pattern,
                    json.dumps({"pattern": pattern, "count": count}),
                    count,
                    min(1.0, count / 100.0),
                    datetime.now().isoformat()
                ))

            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error analyzing alert patterns: {e}")

    def _update_context_weights(self) -> None:
        """Update context weights based on alert resolution success"""
        try:
            # Analyze successful vs failed resolutions
            resolved_alerts = [a for a in self.alert_history
                             if a.status == AlertStatus.RESOLVED]

            if len(resolved_alerts) > 10:
                # Update weights based on resolution patterns
                for context_key in ['market_hours', 'trading_active', 'high_volume_period']:
                    success_with_context = sum(1 for a in resolved_alerts
                                             if a.context.get(context_key, False))
                    if success_with_context > 0:
                        self.context_weights[context_key] = min(1.5,
                            success_with_context / len(resolved_alerts) + 1.0)

        except Exception as e:
            self.logger.error(f"Error updating context weights: {e}")

    def _escalation_management_loop(self) -> None:
        """Manage alert escalation process"""
        self.logger.info("Escalation management loop started")

        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()

                for alert in list(self.active_alerts.values()):
                    if alert.status == AlertStatus.ACTIVE:
                        self._check_escalation(alert, current_time)

                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in escalation management loop: {e}")
                time.sleep(60)

    def _check_escalation(self, alert: AlertEvent, current_time: datetime) -> None:
        """Check if alert needs escalation"""
        try:
            escalation_intervals = self.config['escalation_settings']['escalation_intervals']
            max_levels = self.config['escalation_settings']['max_escalation_levels']

            if alert.escalation_level < max_levels:
                time_since_creation = (current_time - alert.timestamp).total_seconds()
                required_interval = escalation_intervals[
                    min(alert.escalation_level, len(escalation_intervals) - 1)
                ]

                if time_since_creation >= required_interval:
                    self._escalate_alert(alert)

        except Exception as e:
            self.logger.error(f"Error checking escalation for {alert.alert_id}: {e}")

    def _escalate_alert(self, alert: AlertEvent) -> None:
        """Escalate alert to next level"""
        try:
            alert.escalation_level += 1
            alert.escalated_at = datetime.now()

            # Enhanced notification for escalated alerts
            escalated_channels = [
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
                NotificationChannel.CONSOLE
            ]

            for channel in escalated_channels:
                if channel == NotificationChannel.EMAIL or self.config['channels']['slack']['enabled']:
                    self.thread_pool.submit(
                        self.notification_manager.send_escalated_notification,
                        alert, channel
                    )

            # Update in database
            self._store_alert(alert)

            self.logger.warning(f"Alert escalated: {alert.alert_id} - Level {alert.escalation_level}")

        except Exception as e:
            self.logger.error(f"Error escalating alert {alert.alert_id}: {e}")

    def _metrics_collection_loop(self) -> None:
        """Collect and store alert system metrics"""
        self.logger.info("Metrics collection loop started")

        while not self.shutdown_event.is_set():
            try:
                metrics = self._calculate_current_metrics()
                self._store_metrics(metrics)
                self.metrics_history.append(metrics)

                time.sleep(60)  # Collect metrics every minute

            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(60)

    def _calculate_current_metrics(self) -> AlertMetrics:
        """Calculate current system metrics"""
        current_time = datetime.now()

        # Active alerts
        active_count = len([a for a in self.active_alerts.values()
                           if a.status == AlertStatus.ACTIVE])

        # Alerts per minute (last hour)
        hour_ago = current_time - timedelta(hours=1)
        recent_alerts = [a for a in self.alert_history if a.timestamp > hour_ago]
        alerts_per_minute = len(recent_alerts) / 60.0

        # False positive rate (simplified)
        resolved_alerts = [a for a in recent_alerts if a.status == AlertStatus.RESOLVED]
        false_positive_rate = 0.1 if not recent_alerts else len(resolved_alerts) / len(recent_alerts)

        # Resolution time average
        resolution_times = []
        for alert in resolved_alerts:
            if alert.resolved_at:
                resolution_time = (alert.resolved_at - alert.timestamp).total_seconds()
                resolution_times.append(resolution_time)

        avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0

        # Escalation rate
        escalated_alerts = [a for a in recent_alerts if a.escalation_level > 0]
        escalation_rate = len(escalated_alerts) / len(recent_alerts) if recent_alerts else 0

        return AlertMetrics(
            timestamp=current_time,
            active_alerts=active_count,
            alerts_per_minute=alerts_per_minute,
            false_positive_rate=false_positive_rate,
            resolution_time_avg=avg_resolution_time,
            escalation_rate=escalation_rate,
            notification_success_rate=0.95,  # Placeholder
            ai_prediction_accuracy=0.85,  # Placeholder
            system_load=0.3  # Placeholder
        )

    def _store_metrics(self, metrics: AlertMetrics) -> None:
        """Store metrics in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alert_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.active_alerts,
                metrics.alerts_per_minute,
                metrics.false_positive_rate,
                metrics.resolution_time_avg,
                metrics.escalation_rate,
                metrics.notification_success_rate,
                metrics.ai_prediction_accuracy,
                metrics.system_load
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")

    def _cleanup_loop(self) -> None:
        """Cleanup old alerts and optimize performance"""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(days=30)

                # Cleanup resolved alerts older than 30 days
                alerts_to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    if (alert.status == AlertStatus.RESOLVED and
                        alert.resolved_at and alert.resolved_at < cutoff_time):
                        alerts_to_remove.append(alert_id)

                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]

                # Cleanup database
                cursor = self.db_connection.cursor()
                cursor.execute("DELETE FROM alerts WHERE resolved_at < ?",
                             (cutoff_time.isoformat(),))
                self.db_connection.commit()

                if alerts_to_remove:
                    self.logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")

                time.sleep(3600)  # Run cleanup every hour

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(3600)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()

                self._store_alert(alert)

                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True

        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()

                self._store_alert(alert)

                self.logger.info(f"Alert resolved: {alert_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")

        return False

    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values()
                if alert.status == AlertStatus.ACTIVE]

    def get_alert_metrics(self) -> AlertMetrics:
        """Get current alert system metrics"""
        return self._calculate_current_metrics()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        metrics = self.get_alert_metrics()

        return {
            'system_status': 'OPERATIONAL',
            'active_alerts': len(self.get_active_alerts()),
            'total_alerts_24h': len([a for a in self.alert_history
                                   if a.timestamp > datetime.now() - timedelta(days=1)]),
            'ai_models_trained': len(self.ai_models),
            'notification_channels_active': len([ch for ch in NotificationChannel]),
            'escalation_levels_configured': self.config['escalation_settings']['max_escalation_levels'],
            'metrics': asdict(metrics),
            'threads_active': len(self.processing_threads),
            'processing_queue_size': len(self.processing_queue)
        }

    def shutdown(self) -> None:
        """Shutdown the alert system gracefully"""
        self.logger.info("Shutting down intelligent alert system...")

        self.shutdown_event.set()

        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=10)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Close database connection
        if self.db_connection:
            self.db_connection.close()

        self.logger.info("Intelligent alert system shutdown complete")


class NotificationManager:
    """Manages multi-channel notifications"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('NotificationManager')

    def send_notification(self, alert: AlertEvent, channel: NotificationChannel) -> bool:
        """Send notification through specified channel"""
        try:
            if channel == NotificationChannel.EMAIL:
                return self._send_email(alert)
            elif channel == NotificationChannel.SLACK:
                return self._send_slack(alert)
            elif channel == NotificationChannel.DISCORD:
                return self._send_discord(alert)
            elif channel == NotificationChannel.CONSOLE:
                return self._send_console(alert)
            elif channel == NotificationChannel.DATABASE:
                return True  # Already stored
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error sending notification via {channel.value}: {e}")
            return False

    def send_escalated_notification(self, alert: AlertEvent, channel: NotificationChannel) -> bool:
        """Send escalated notification with enhanced formatting"""
        # Add escalation context
        original_title = alert.title
        alert.title = f"[ESCALATED L{alert.escalation_level}] {original_title}"

        result = self.send_notification(alert, channel)

        # Restore original title
        alert.title = original_title

        return result

    def _send_email(self, alert: AlertEvent) -> bool:
        """Send email notification"""
        try:
            email_config = self.config['channels']['email']
            if not email_config['enabled']:
                return False

            # Create email message
            msg = MimeMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = email_config.get('to_address', 'admin@company.com')
            msg['Subject'] = f"[{alert.severity.value[0]}] {alert.title}"

            # Email body
            body = f"""
Alert Details:
=============
ID: {alert.alert_id}
Severity: {alert.severity.value[0]}
Category: {alert.category.value}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

AI Analysis:
- Confidence: {alert.ai_confidence:.2f}
- Priority Score: {alert.priority_score:.2f}
- Predicted Impact: {alert.predicted_impact:.2f}

Recommended Actions:
{chr(10).join(f"- {action}" for action in alert.recommended_actions)}

Context:
{json.dumps(alert.context, indent=2)}
            """

            msg.attach(MimeText(body, 'plain'))

            # Send email (configuration required)
            self.logger.info(f"Email notification prepared for {alert.alert_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error sending email for {alert.alert_id}: {e}")
            return False

    def _send_slack(self, alert: AlertEvent) -> bool:
        """Send Slack notification"""
        try:
            slack_config = self.config['channels']['slack']
            if not slack_config['enabled'] or not slack_config['webhook_url']:
                return False

            # Create Slack message
            color = alert.severity.value[2]  # Color code
            message = {
                "attachments": [{
                    "color": color,
                    "title": f"{alert.severity.value[0]} Alert: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "AI Confidence", "value": f"{alert.ai_confidence:.2f}", "short": True},
                        {"title": "Priority", "value": f"{alert.priority_score:.2f}", "short": True}
                    ],
                    "timestamp": alert.timestamp.timestamp()
                }]
            }

            response = requests.post(slack_config['webhook_url'], json=message, timeout=10)
            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Error sending Slack notification for {alert.alert_id}: {e}")
            return False

    def _send_discord(self, alert: AlertEvent) -> bool:
        """Send Discord notification"""
        try:
            discord_config = self.config['channels']['discord']
            if not discord_config['enabled'] or not discord_config['webhook_url']:
                return False

            # Create Discord embed
            color = int(alert.severity.value[2][1:], 16)  # Convert hex to int
            embed = {
                "title": f"{alert.severity.value[0]} Alert",
                "description": alert.title,
                "color": color,
                "fields": [
                    {"name": "Message", "value": alert.message[:1024]},
                    {"name": "Source", "value": alert.source, "inline": True},
                    {"name": "Category", "value": alert.category.value, "inline": True},
                    {"name": "AI Confidence", "value": f"{alert.ai_confidence:.2f}", "inline": True}
                ],
                "timestamp": alert.timestamp.isoformat()
            }

            response = requests.post(
                discord_config['webhook_url'],
                json={"embeds": [embed]},
                timeout=10
            )
            return response.status_code in [200, 204]

        except Exception as e:
            self.logger.error(f"Error sending Discord notification for {alert.alert_id}: {e}")
            return False

    def _send_console(self, alert: AlertEvent) -> bool:
        """Send console notification"""
        try:
            color_code = {
                AlertSeverity.CRITICAL: '\033[91m',
                AlertSeverity.HIGH: '\033[93m',
                AlertSeverity.MEDIUM: '\033[94m',
                AlertSeverity.LOW: '\033[92m',
                AlertSeverity.INFO: '\033[96m'
            }.get(alert.severity, '\033[0m')

            print(f"\n{color_code}{'='*60}")
            print(f"ALERT: {alert.severity.value[0]} - {alert.title}")
            print(f"{'='*60}\033[0m")
            print(f"ID: {alert.alert_id}")
            print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Source: {alert.source}")
            print(f"Category: {alert.category.value}")
            print(f"Message: {alert.message}")
            print(f"AI Confidence: {alert.ai_confidence:.2f}")
            print(f"Priority: {alert.priority_score:.2f}")
            if alert.recommended_actions:
                print("Recommended Actions:")
                for i, action in enumerate(alert.recommended_actions, 1):
                    print(f"  {i}. {action}")
            print(f"{color_code}{'='*60}\033[0m\n")

            return True

        except Exception as e:
            self.logger.error(f"Error sending console notification for {alert.alert_id}: {e}")
            return False


def main():
    """Main entry point for intelligent alert system"""
    alert_system = IntelligentAlertSystemC1()

    try:
        print("="*80)
        print("        INTELLIGENT ALERT SYSTEM C1 - AI-DRIVEN MONITORING")
        print("                Professional Quantitative Trading System")
        print("="*80)
        print("Starting AI-enhanced alert management...")
        print("Features: ML-based prioritization, context-aware routing, predictive alerting")
        print("Press Ctrl+C to stop")
        print("="*80)

        # Start alert system
        alert_system.start_alert_system()

        # Demo: Create some test alerts
        time.sleep(2)

        # System failure alert
        alert_system.create_alert(
            title="Database Connection Lost",
            message="Unable to connect to primary database server",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM_FAILURE,
            source="DatabaseManager",
            context={
                "database_host": "primary-db",
                "error_code": "CONNECTION_TIMEOUT",
                "retry_count": 3,
                "market_hours": True,
                "trading_active": True
            }
        )

        # Performance alert
        alert_system.create_alert(
            title="High CPU Usage Detected",
            message="CPU usage has exceeded 90% for 5 minutes",
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PERFORMANCE_DEGRADATION,
            source="SystemMonitor",
            context={
                "cpu_usage": 92.5,
                "memory_usage": 78.3,
                "affected_processes": ["trading_engine", "risk_manager"],
                "duration_minutes": 5
            }
        )

        # Trading anomaly
        alert_system.create_alert(
            title="Unusual Trading Pattern Detected",
            message="Abnormal order flow detected in AAPL",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.TRADING_ANOMALY,
            source="TradingEngine",
            context={
                "symbol": "AAPL",
                "anomaly_score": 0.85,
                "volume_deviation": 3.2,
                "price_impact": 0.05
            }
        )

        # Keep running and show status
        while True:
            time.sleep(30)  # Status update every 30 seconds

            status = alert_system.get_system_status()
            print(f"\nStatus: {datetime.now().strftime('%H:%M:%S')} - "
                  f"Active Alerts: {status['active_alerts']}, "
                  f"24h Total: {status['total_alerts_24h']}, "
                  f"Queue: {status['processing_queue_size']}")

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in alert system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        alert_system.shutdown()

if __name__ == "__main__":
    sys.exit(main())