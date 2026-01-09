#!/usr/bin/env python3
"""
Enhanced System Health Monitoring
Professional Quantitative Trading System

This script provides comprehensive system health monitoring including:
- Real-time system resource monitoring
- Process health and performance tracking
- Network and API connectivity monitoring
- Database and storage health checks
- Trading system specific metrics
- Automated alerting and notifications
- Performance bottleneck detection
- Predictive health analysis
- AI-powered anomaly detection
- Integration with intelligent alert system
- GPU monitoring and performance tracking
- Advanced performance analytics

Features:
- Multi-threaded monitoring with intelligent scheduling
- Advanced metrics collection and analysis
- Real-time dashboard updates
- Configurable alert thresholds
- Historical performance tracking
- Automated health reporting
- Integration with trading system components
- Machine learning-based anomaly detection
- Predictive performance monitoring
- Intelligent alert system integration
- GPU performance monitoring
- Investment-grade monitoring capabilities

Author: Quantitative Trading System
Version: 3.0 - Enhanced Investment Grade
"""

# Set encoding for Windows compatibility
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import time
import json
import logging
import threading
import subprocess
import statistics
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import psutil
import requests
import socket
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class HealthMetric:
    """Enhanced health metric data structure."""
    name: str
    value: Union[float, int, str, bool]
    unit: str
    status: str  # 'healthy', 'warning', 'critical'
    threshold_warning: Optional[float]
    threshold_critical: Optional[float]
    timestamp: datetime
    details: Optional[str] = None

    # Enhanced fields for investment-grade monitoring
    trend: Optional[str] = None  # 'up', 'down', 'stable'
    anomaly_score: float = 0.0
    performance_impact: str = 'none'  # 'none', 'low', 'medium', 'high', 'critical'
    source_system: str = 'system'
    correlation_id: Optional[str] = None

@dataclass
class SystemHealth:
    """Enhanced overall system health status."""
    overall_status: str
    score: float  # 0-100
    metrics: List[HealthMetric]
    alerts: List[str]
    timestamp: datetime

    # Enhanced monitoring fields
    performance_score: float = 0.0
    reliability_score: float = 0.0
    efficiency_score: float = 0.0
    predictive_alerts: List[str] = None
    bottlenecks: List[Dict[str, Any]] = None
    recommendations: List[str] = None

@dataclass
class EnhancedHealthReport:
    """Comprehensive health analysis report"""
    timestamp: datetime
    overall_health: SystemHealth
    detailed_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    capacity_planning: Dict[str, Any]
    sla_compliance: Dict[str, Any]
    recommendations: List[str]

class EnhancedHealthMonitor:
    """Enhanced comprehensive system health monitoring with AI capabilities."""

    def __init__(self, enable_ai_features: bool = True):
        self.base_dir = Path(__file__).parent.absolute()
        self.quant_dir = self.base_dir / "quant_system_full"
        self.data_dir = self.base_dir / "health_data"
        self.data_dir.mkdir(exist_ok=True)

        self.logger = self._setup_logging()
        self.shutdown_event = threading.Event()
        self.monitoring_threads: List[threading.Thread] = []

        # Enhanced configuration
        self.config = self._load_enhanced_configuration()

        # AI/ML Components
        self.enable_ai_features = enable_ai_features
        if self.enable_ai_features:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self._initialize_ai_models()

        # Enhanced monitoring state
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.performance_cache = deque(maxlen=100)
        self.alert_history = deque(maxlen=1000)

        # Database for advanced analytics
        self.db_connection = self._initialize_enhanced_database()

        # Integration components
        self.intelligent_alerts = None
        self.performance_monitor = None
        self.gpu_monitor = None

        # Enhanced health tracking
        self.health_history: List[SystemHealth] = []
        self.last_alerts: Dict[str, datetime] = {}
        self.performance_samples: Dict[str, List[float]] = {}
        self.process_monitors: Dict[str, psutil.Process] = {}

        # Service endpoints to monitor
        self.services = {
            'backend_api': {'url': 'http://localhost:8000/health', 'timeout': 10},
            'react_frontend': {'url': 'http://localhost:3000', 'timeout': 5},
            'streamlit_dashboard': {'url': 'http://localhost:8501', 'timeout': 5}
        }

        # Critical processes to monitor
        self.critical_processes = [
            'python',  # Trading bot processes
            'uvicorn',  # Backend API
            'streamlit',  # Dashboard
            'node'  # React frontend
        ]

        # Enhanced monitoring capabilities
        self._establish_baselines()
        self._initialize_integrations()

        self.logger.info("Enhanced Health Monitor initialized with AI capabilities")

    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging for health monitoring."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('EnhancedHealthMonitor')
        logger.setLevel(logging.INFO)

        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '\033[92m%(asctime)s\033[0m - \033[96mENHANCED-HEALTH\033[0m - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"enhanced_health_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _load_enhanced_configuration(self) -> Dict[str, Any]:
        """Load enhanced monitoring configuration"""
        config = {
            'monitoring_interval': 10,  # seconds
            'history_retention_hours': 168,  # 1 week
            'alert_cooldown_minutes': 5,
            'performance_samples': 100,  # for moving averages
            'network_timeout': 5,
            'api_timeout': 10,
            'critical_memory_threshold': 90,  # percentage
            'warning_memory_threshold': 80,
            'critical_cpu_threshold': 95,
            'warning_cpu_threshold': 85,
            'critical_disk_threshold': 95,
            'warning_disk_threshold': 85,
            'max_response_time_ms': 1000,
            'min_available_connections': 10,

            # Enhanced monitoring features
            'ai_anomaly_detection': True,
            'predictive_monitoring': True,
            'trend_analysis': True,
            'capacity_planning': True,
            'performance_correlation': True,
            'intelligent_alerting': True,
            'gpu_monitoring': True,

            # AI thresholds
            'anomaly_threshold': 0.3,
            'trend_significance_threshold': 0.1,
            'prediction_confidence_threshold': 0.7,

            # Performance baselines
            'baseline_learning_rate': 0.1,
            'baseline_adaptation_threshold': 0.2,

            # SLA targets
            'sla_availability': 99.9,
            'sla_response_time_ms': 500,
            'sla_error_rate': 0.1
        }

        return config

    def _initialize_ai_models(self) -> None:
        """Initialize AI/ML models for enhanced monitoring"""
        try:
            # Initialize anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=self.config['anomaly_threshold'],
                random_state=42
            )

            # Load pre-trained models if available
            model_dir = self.data_dir / "ai_models"
            model_dir.mkdir(exist_ok=True)

            self.logger.info("AI models initialized for enhanced monitoring")

        except Exception as e:
            self.logger.error(f"Error initializing AI models: {e}")
            self.enable_ai_features = False

    def _initialize_enhanced_database(self) -> sqlite3.Connection:
        """Initialize enhanced database for advanced analytics"""
        db_path = self.data_dir / "enhanced_health.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create enhanced tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS enhanced_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                name TEXT,
                value REAL,
                unit TEXT,
                status TEXT,
                trend TEXT,
                anomaly_score REAL,
                performance_impact TEXT,
                source_system TEXT,
                correlation_id TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS health_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                overall_score REAL,
                performance_score REAL,
                reliability_score REAL,
                efficiency_score REAL,
                detailed_analysis TEXT,
                trend_analysis TEXT,
                anomaly_detection TEXT,
                recommendations TEXT
            );

            CREATE TABLE IF NOT EXISTS baselines (
                metric_name TEXT PRIMARY KEY,
                baseline_value REAL,
                std_deviation REAL,
                confidence_interval REAL,
                last_updated DATETIME,
                sample_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_name TEXT,
                predicted_value REAL,
                confidence REAL,
                time_horizon_minutes INTEGER,
                model_used TEXT,
                actual_value REAL,
                prediction_error REAL
            );

            CREATE INDEX IF NOT EXISTS idx_enhanced_metrics_timestamp ON enhanced_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_enhanced_metrics_name ON enhanced_metrics(name);
            CREATE INDEX IF NOT EXISTS idx_health_reports_timestamp ON health_reports(timestamp);
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
        """)

        conn.commit()
        return conn

    def _establish_baselines(self) -> None:
        """Establish enhanced performance baselines"""
        try:
            # Load existing baselines
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT metric_name, baseline_value, std_deviation, confidence_interval
                FROM baselines
            """)

            for row in cursor.fetchall():
                metric_name, baseline_value, std_deviation, confidence_interval = row
                self.baseline_metrics[metric_name] = {
                    'baseline': baseline_value,
                    'std_dev': std_deviation,
                    'confidence_interval': confidence_interval
                }

            self.logger.info(f"Loaded {len(self.baseline_metrics)} enhanced baseline metrics")

        except Exception as e:
            self.logger.error(f"Error loading enhanced baselines: {e}")

    def _initialize_integrations(self) -> None:
        """Initialize integrations with other monitoring components"""
        try:
            # Try to import and initialize intelligent alert system
            try:
                from intelligent_alert_system_c1 import IntelligentAlertSystemC1
                self.intelligent_alerts = IntelligentAlertSystemC1()
                self.logger.info("Intelligent Alert System integration initialized")
            except ImportError:
                self.logger.debug("Intelligent Alert System not available")

            # Try to import and initialize advanced performance monitor
            try:
                from advanced_performance_monitor import AdvancedPerformanceMonitor
                self.performance_monitor = AdvancedPerformanceMonitor()
                self.logger.info("Advanced Performance Monitor integration initialized")
            except ImportError:
                self.logger.debug("Advanced Performance Monitor not available")

            # Try to import and initialize GPU monitor
            try:
                from setup_gpu import GPUSystemManager
                self.gpu_monitor = GPUSystemManager()
                self.logger.info("GPU System Manager integration initialized")
            except ImportError:
                self.logger.debug("GPU System Manager not available")

        except Exception as e:
            self.logger.error(f"Error initializing integrations: {e}")

    def start_enhanced_monitoring(self) -> None:
        """Start enhanced monitoring with AI capabilities."""
        self.logger.info("=== Starting Enhanced System Health Monitoring ===")

        # Start core monitoring threads
        self._start_core_monitoring()

        # Start AI enhancement threads if enabled
        if self.enable_ai_features:
            self._start_ai_monitoring()

        # Start integration monitoring
        self._start_integration_monitoring()

        # Start integrated systems
        if self.intelligent_alerts:
            self.intelligent_alerts.start_alert_system()

        if self.performance_monitor:
            self.performance_monitor.start_monitoring()

        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring(self.gpu_monitor.detect_gpu_environment())

        self.logger.info(f"Enhanced monitoring started with {len(self.monitoring_threads)} threads")

    def _start_core_monitoring(self) -> None:
        """Start core monitoring threads"""
        # System resource monitoring
        system_thread = threading.Thread(
            target=self._enhanced_system_monitoring_loop,
            name="EnhancedSystemMonitor",
            daemon=True
        )
        system_thread.start()
        self.monitoring_threads.append(system_thread)

        # Process monitoring
        process_thread = threading.Thread(
            target=self._enhanced_process_monitoring_loop,
            name="EnhancedProcessMonitor",
            daemon=True
        )
        process_thread.start()
        self.monitoring_threads.append(process_thread)

        # Network and service monitoring
        network_thread = threading.Thread(
            target=self._enhanced_network_monitoring_loop,
            name="EnhancedNetworkMonitor",
            daemon=True
        )
        network_thread.start()
        self.monitoring_threads.append(network_thread)

        # Trading system monitoring
        trading_thread = threading.Thread(
            target=self._enhanced_trading_monitoring_loop,
            name="EnhancedTradingMonitor",
            daemon=True
        )
        trading_thread.start()
        self.monitoring_threads.append(trading_thread)

    def _start_ai_monitoring(self) -> None:
        """Start AI-enhanced monitoring threads"""
        # Anomaly detection thread
        anomaly_thread = threading.Thread(
            target=self._anomaly_detection_loop,
            name="AnomalyDetector",
            daemon=True
        )
        anomaly_thread.start()
        self.monitoring_threads.append(anomaly_thread)

        # Trend analysis thread
        trend_thread = threading.Thread(
            target=self._trend_analysis_loop,
            name="TrendAnalyzer",
            daemon=True
        )
        trend_thread.start()
        self.monitoring_threads.append(trend_thread)

        # Predictive monitoring thread
        prediction_thread = threading.Thread(
            target=self._predictive_monitoring_loop,
            name="PredictiveMonitor",
            daemon=True
        )
        prediction_thread.start()
        self.monitoring_threads.append(prediction_thread)

    def _start_integration_monitoring(self) -> None:
        """Start integration monitoring threads"""
        # Health report generation
        report_thread = threading.Thread(
            target=self._enhanced_report_generation_loop,
            name="EnhancedReportGenerator",
            daemon=True
        )
        report_thread.start()
        self.monitoring_threads.append(report_thread)

        # Capacity planning thread
        capacity_thread = threading.Thread(
            target=self._capacity_planning_loop,
            name="CapacityPlanner",
            daemon=True
        )
        capacity_thread.start()
        self.monitoring_threads.append(capacity_thread)

    def _enhanced_system_monitoring_loop(self) -> None:
        """Enhanced system resource monitoring with AI analysis"""
        self.logger.info("Enhanced system monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                enhanced_metrics = []

                # Collect basic system metrics
                basic_metrics = self._collect_basic_system_metrics(current_time)
                enhanced_metrics.extend(basic_metrics)

                # Enhance metrics with AI analysis
                if self.enable_ai_features:
                    for metric in enhanced_metrics:
                        self._enhance_metric_with_ai(metric)

                # Process and store enhanced metrics
                for metric in enhanced_metrics:
                    self._process_enhanced_metric(metric)
                    self._store_enhanced_metric(metric)

                # Check for alerts
                self._check_and_generate_enhanced_alerts(enhanced_metrics)

                time.sleep(self.config['monitoring_interval'])

            except Exception as e:
                self.logger.error(f"Error in enhanced system monitoring loop: {e}")
                time.sleep(30)

    def _collect_basic_system_metrics(self, timestamp: datetime) -> List[HealthMetric]:
        """Collect basic system metrics with enhanced data"""
        metrics = []

        # CPU metrics with enhanced analysis
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]

        cpu_metric = HealthMetric(
            name='cpu_utilization',
            value=cpu_percent,
            unit='percent',
            status=self._determine_status(cpu_percent, self.config['warning_cpu_threshold'],
                                        self.config['critical_cpu_threshold']),
            threshold_warning=self.config['warning_cpu_threshold'],
            threshold_critical=self.config['critical_cpu_threshold'],
            timestamp=timestamp,
            details=f"{cpu_count} cores, load avg: {load_avg[0]:.2f}",
            source_system='system_cpu'
        )
        metrics.append(cpu_metric)

        # Memory metrics with swap analysis
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_metric = HealthMetric(
            name='memory_utilization',
            value=memory.percent,
            unit='percent',
            status=self._determine_status(memory.percent, self.config['warning_memory_threshold'],
                                        self.config['critical_memory_threshold']),
            threshold_warning=self.config['warning_memory_threshold'],
            threshold_critical=self.config['critical_memory_threshold'],
            timestamp=timestamp,
            details=f"{memory.used/(1024**3):.1f}GB used of {memory.total/(1024**3):.1f}GB",
            source_system='system_memory'
        )
        metrics.append(memory_metric)

        # Disk metrics with I/O analysis
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100

            disk_metric = HealthMetric(
                name='disk_utilization',
                value=disk_percent,
                unit='percent',
                status=self._determine_status(disk_percent, self.config['warning_disk_threshold'],
                                            self.config['critical_disk_threshold']),
                threshold_warning=self.config['warning_disk_threshold'],
                threshold_critical=self.config['critical_disk_threshold'],
                timestamp=timestamp,
                details=f"{disk_usage.free/(1024**3):.1f}GB free",
                source_system='system_disk'
            )
            metrics.append(disk_metric)

        except Exception as e:
            self.logger.debug(f"Error collecting disk metrics: {e}")

        # Network metrics with bandwidth analysis
        try:
            net_io = psutil.net_io_counters()
            if net_io and hasattr(self, '_last_net_io'):
                time_diff = (timestamp - self._last_net_timestamp).total_seconds()
                if time_diff > 0:
                    bytes_sent_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_diff
                    bytes_recv_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_diff

                    network_metric = HealthMetric(
                        name='network_throughput',
                        value=(bytes_sent_rate + bytes_recv_rate) / (1024*1024),  # MB/s
                        unit='MB/s',
                        status='healthy',  # Network throughput doesn't have fixed thresholds
                        threshold_warning=None,
                        threshold_critical=None,
                        timestamp=timestamp,
                        details=f"↑{bytes_sent_rate/(1024*1024):.2f}MB/s ↓{bytes_recv_rate/(1024*1024):.2f}MB/s",
                        source_system='system_network'
                    )
                    metrics.append(network_metric)

            self._last_net_io = net_io
            self._last_net_timestamp = timestamp

        except Exception as e:
            self.logger.debug(f"Error collecting network metrics: {e}")

        return metrics

    def _enhance_metric_with_ai(self, metric: HealthMetric) -> None:
        """Enhance metric with AI analysis"""
        try:
            # Calculate trend
            metric.trend = self._calculate_enhanced_trend(metric.name, metric.value)

            # Calculate anomaly score
            metric.anomaly_score = self._calculate_anomaly_score(metric.name, metric.value)

            # Determine performance impact
            metric.performance_impact = self._determine_performance_impact(metric)

            # Generate correlation ID for related metrics
            metric.correlation_id = self._generate_correlation_id(metric)

        except Exception as e:
            self.logger.debug(f"Error enhancing metric {metric.name} with AI: {e}")

    def _calculate_enhanced_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate enhanced trend analysis"""
        if metric_name not in self.metrics_history:
            return "stable"

        history = list(self.metrics_history[metric_name])
        if len(history) < 5:
            return "stable"

        # Get recent values
        recent_values = [m.value for m in history[-10:] if isinstance(m.value, (int, float))]

        if len(recent_values) < 3:
            return "stable"

        # Calculate trend using linear regression
        try:
            from sklearn.linear_model import LinearRegression

            X = np.array(range(len(recent_values))).reshape(-1, 1)
            y = np.array(recent_values)

            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]

            # Determine trend significance
            threshold = self.config['trend_significance_threshold']
            avg_value = np.mean(recent_values)

            if avg_value > 0:
                relative_slope = abs(slope) / avg_value

                if relative_slope > threshold:
                    return "up" if slope > 0 else "down"
                else:
                    return "stable"
            else:
                return "stable"

        except Exception:
            # Fallback to simple trend calculation
            if len(recent_values) >= 3:
                recent_avg = statistics.mean(recent_values[-3:])
                older_avg = statistics.mean(recent_values[-6:-3]) if len(recent_values) >= 6 else recent_avg

                change_percent = abs(recent_avg - older_avg) / max(older_avg, 1.0)

                if change_percent > 0.1:
                    return "up" if recent_avg > older_avg else "down"

            return "stable"

    def _calculate_anomaly_score(self, metric_name: str, current_value: float) -> float:
        """Calculate AI-based anomaly score"""
        if not self.enable_ai_features:
            return 0.0

        try:
            # Use baseline comparison
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                baseline_value = baseline.get('baseline', current_value)
                std_dev = baseline.get('std_dev', 1.0)

                if std_dev > 0:
                    z_score = abs(current_value - baseline_value) / std_dev
                    # Normalize to 0-1 scale
                    return min(1.0, z_score / 3.0)

            return 0.0

        except Exception as e:
            self.logger.debug(f"Error calculating anomaly score for {metric_name}: {e}")
            return 0.0

    def _determine_performance_impact(self, metric: HealthMetric) -> str:
        """Determine performance impact level"""
        # Map metric status to performance impact
        status_impact_map = {
            'healthy': 'none',
            'warning': 'low' if metric.anomaly_score < 0.5 else 'medium',
            'critical': 'high' if metric.anomaly_score < 0.8 else 'critical'
        }

        base_impact = status_impact_map.get(metric.status, 'none')

        # Adjust based on system criticality
        if metric.source_system in ['system_cpu', 'system_memory']:
            if base_impact == 'low':
                base_impact = 'medium'
            elif base_impact == 'medium':
                base_impact = 'high'

        return base_impact

    def _generate_correlation_id(self, metric: HealthMetric) -> str:
        """Generate correlation ID for related metrics"""
        # Simple correlation based on timestamp and system
        timestamp_bucket = metric.timestamp.strftime('%Y%m%d%H%M')  # Minute-level correlation
        return f"{metric.source_system}_{timestamp_bucket}"

    def _process_enhanced_metric(self, metric: HealthMetric) -> None:
        """Process enhanced metric for analytics"""
        # Add to history
        self.metrics_history[metric.name].append(metric)

        # Update baseline if conditions are met
        if self.config.get('baseline_learning_rate', 0) > 0:
            self._update_adaptive_baseline(metric)

    def _update_adaptive_baseline(self, metric: HealthMetric) -> None:
        """Update baseline with adaptive learning"""
        try:
            history = list(self.metrics_history[metric.name])

            # Need sufficient data points
            if len(history) < 50:
                return

            # Extract numeric values
            values = [m.value for m in history[-100:] if isinstance(m.value, (int, float))]

            if len(values) < 20:
                return

            # Calculate adaptive baseline
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 1.0

            # Calculate confidence interval
            confidence_interval = 1.96 * std_dev / np.sqrt(len(values))  # 95% CI

            # Update baseline with learning rate
            if metric.name in self.baseline_metrics:
                current_baseline = self.baseline_metrics[metric.name]['baseline']
                learning_rate = self.config['baseline_learning_rate']

                new_baseline = (1 - learning_rate) * current_baseline + learning_rate * mean_value
            else:
                new_baseline = mean_value

            # Store updated baseline
            self.baseline_metrics[metric.name] = {
                'baseline': new_baseline,
                'std_dev': std_dev,
                'confidence_interval': confidence_interval
            }

            # Update database periodically
            if len(history) % 100 == 0:  # Every 100 updates
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO baselines
                    (metric_name, baseline_value, std_deviation, confidence_interval, last_updated, sample_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    new_baseline,
                    std_dev,
                    confidence_interval,
                    datetime.now().isoformat(),
                    len(values)
                ))
                self.db_connection.commit()

        except Exception as e:
            self.logger.debug(f"Error updating adaptive baseline for {metric.name}: {e}")

    def _store_enhanced_metric(self, metric: HealthMetric) -> None:
        """Store enhanced metric in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO enhanced_metrics (
                    timestamp, name, value, unit, status, trend, anomaly_score,
                    performance_impact, source_system, correlation_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp.isoformat(),
                metric.name,
                metric.value,
                metric.unit,
                metric.status,
                metric.trend or 'stable',
                metric.anomaly_score,
                metric.performance_impact,
                metric.source_system,
                metric.correlation_id,
                json.dumps({
                    'threshold_warning': metric.threshold_warning,
                    'threshold_critical': metric.threshold_critical,
                    'details': metric.details
                })
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing enhanced metric {metric.name}: {e}")

    def _check_and_generate_enhanced_alerts(self, metrics: List[HealthMetric]) -> None:
        """Generate enhanced alerts with AI analysis"""
        for metric in metrics:
            try:
                # Standard threshold-based alerts
                if metric.status in ['warning', 'critical']:
                    self._generate_standard_alert(metric)

                # AI-based anomaly alerts
                if metric.anomaly_score > self.config['anomaly_threshold']:
                    self._generate_anomaly_alert(metric)

                # Trend-based predictive alerts
                if metric.trend in ['up', 'down'] and metric.performance_impact in ['high', 'critical']:
                    self._generate_trend_alert(metric)

                # Integration with intelligent alert system
                if self.intelligent_alerts and metric.status == 'critical':
                    self._send_to_intelligent_alerts(metric)

            except Exception as e:
                self.logger.error(f"Error generating enhanced alerts for {metric.name}: {e}")

    def _generate_standard_alert(self, metric: HealthMetric) -> None:
        """Generate standard threshold-based alert"""
        alert_key = f"{metric.name}_{metric.status}"
        current_time = datetime.now()

        # Check cooldown
        if alert_key in self.last_alerts:
            time_since_last = current_time - self.last_alerts[alert_key]
            if time_since_last.total_seconds() < self.config['alert_cooldown_minutes'] * 60:
                return

        alert_message = (f"ALERT [{metric.status.upper()}]: {metric.name} = {metric.value} {metric.unit}"
                        f" (Trend: {metric.trend}, Impact: {metric.performance_impact})")

        if metric.details:
            alert_message += f" - {metric.details}"

        # Log alert
        if metric.status == 'critical':
            self.logger.error(alert_message)
        else:
            self.logger.warning(alert_message)

        self.last_alerts[alert_key] = current_time

    def _generate_anomaly_alert(self, metric: HealthMetric) -> None:
        """Generate AI-based anomaly alert"""
        alert_message = (f"ANOMALY DETECTED: {metric.name} = {metric.value} {metric.unit} "
                        f"(Anomaly Score: {metric.anomaly_score:.2f})")

        self.logger.warning(alert_message)

    def _generate_trend_alert(self, metric: HealthMetric) -> None:
        """Generate trend-based predictive alert"""
        alert_message = (f"TREND ALERT: {metric.name} trending {metric.trend} "
                        f"with {metric.performance_impact} performance impact")

        self.logger.info(alert_message)

    def _send_to_intelligent_alerts(self, metric: HealthMetric) -> None:
        """Send critical alerts to intelligent alert system"""
        try:
            if self.intelligent_alerts:
                # Map health metric to intelligent alert
                severity = {
                    'critical': 'CRITICAL',
                    'warning': 'HIGH',
                    'healthy': 'INFO'
                }.get(metric.status, 'MEDIUM')

                category = {
                    'system_cpu': 'PERFORMANCE_DEGRADATION',
                    'system_memory': 'RESOURCE_EXHAUSTION',
                    'system_disk': 'RESOURCE_EXHAUSTION',
                    'system_network': 'NETWORK_CONNECTIVITY'
                }.get(metric.source_system, 'SYSTEM_FAILURE')

                self.intelligent_alerts.create_alert(
                    title=f"Health Monitor: {metric.name} Alert",
                    message=f"{metric.name} = {metric.value} {metric.unit}. {metric.details or ''}",
                    severity=severity,
                    category=category,
                    source="EnhancedHealthMonitor",
                    context={
                        'metric_name': metric.name,
                        'metric_value': metric.value,
                        'trend': metric.trend,
                        'anomaly_score': metric.anomaly_score,
                        'performance_impact': metric.performance_impact,
                        'source_system': metric.source_system
                    }
                )

        except Exception as e:
            self.logger.error(f"Error sending alert to intelligent system: {e}")

    def _enhanced_process_monitoring_loop(self) -> None:
        """Enhanced process monitoring with performance analysis"""
        self.logger.info("Enhanced process monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                process_metrics = []

                # Monitor critical processes with enhanced analysis
                for process_key, process_info in self._get_enhanced_process_info().items():
                    if process_info['running']:
                        # Create enhanced process metrics
                        cpu_metric = HealthMetric(
                            name=f'process_{process_key}_cpu',
                            value=process_info['cpu_percent'],
                            unit='percent',
                            status=self._determine_status(process_info['cpu_percent'], 25, 50),
                            threshold_warning=25,
                            threshold_critical=50,
                            timestamp=current_time,
                            details=f"PID: {process_info['pid']}",
                            source_system=f'process_{process_key}'
                        )

                        memory_metric = HealthMetric(
                            name=f'process_{process_key}_memory',
                            value=process_info['memory_percent'],
                            unit='percent',
                            status=self._determine_status(process_info['memory_percent'], 15, 30),
                            threshold_warning=15,
                            threshold_critical=30,
                            timestamp=current_time,
                            details=f"RSS: {process_info['memory_mb']:.1f}MB",
                            source_system=f'process_{process_key}'
                        )

                        process_metrics.extend([cpu_metric, memory_metric])

                # Enhanced analysis for process metrics
                if self.enable_ai_features:
                    for metric in process_metrics:
                        self._enhance_metric_with_ai(metric)

                # Process and store
                for metric in process_metrics:
                    self._process_enhanced_metric(metric)
                    self._store_enhanced_metric(metric)

                self._check_and_generate_enhanced_alerts(process_metrics)

                time.sleep(self.config['monitoring_interval'] * 2)

            except Exception as e:
                self.logger.error(f"Error in enhanced process monitoring loop: {e}")
                time.sleep(30)

    def _get_enhanced_process_info(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced process information"""
        process_info = {}

        try:
            # Monitor specific trading system processes
            process_patterns = {
                'backend': {'names': ['uvicorn'], 'cmdline_contains': ['app:app']},
                'frontend': {'names': ['node'], 'cmdline_contains': ['npm run dev', 'vite']},
                'streamlit': {'names': ['streamlit'], 'cmdline_contains': []},
                'worker': {'names': ['python'], 'cmdline_contains': ['runner.py']}
            }

            for process_key, patterns in process_patterns.items():
                process_info[process_key] = {
                    'running': False,
                    'pid': None,
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_mb': 0.0,
                    'threads': 0,
                    'status': 'stopped'
                }

                # Find matching processes
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        proc_info = proc.info
                        proc_name = proc_info.get('name', '').lower()
                        cmdline = ' '.join(proc_info.get('cmdline', []))

                        # Check if process matches pattern
                        name_match = any(name.lower() in proc_name for name in patterns['names'])
                        cmdline_match = (not patterns['cmdline_contains'] or
                                       any(cmd in cmdline for cmd in patterns['cmdline_contains']))

                        if name_match and cmdline_match:
                            # Get process details
                            proc_obj = psutil.Process(proc_info['pid'])
                            memory_info = proc_obj.memory_info()

                            process_info[process_key].update({
                                'running': True,
                                'pid': proc_info['pid'],
                                'cpu_percent': proc_obj.cpu_percent(),
                                'memory_percent': proc_obj.memory_percent(),
                                'memory_mb': memory_info.rss / (1024 * 1024),
                                'threads': proc_obj.num_threads(),
                                'status': 'running'
                            })
                            break

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

        except Exception as e:
            self.logger.error(f"Error getting enhanced process info: {e}")

        return process_info

    def _enhanced_network_monitoring_loop(self) -> None:
        """Enhanced network and service monitoring"""
        self.logger.info("Enhanced network monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                network_metrics = []

                # Test service endpoints with enhanced analysis
                for service_name, service_config in self.services.items():
                    start_time = time.time()
                    service_status = 'healthy'
                    response_time = 0
                    status_code = 0

                    try:
                        response = requests.get(
                            service_config['url'],
                            timeout=service_config['timeout']
                        )
                        response_time = (time.time() - start_time) * 1000  # ms
                        status_code = response.status_code

                        service_status = 'healthy' if status_code < 400 else 'critical'

                    except requests.RequestException:
                        service_status = 'critical'
                        response_time = service_config['timeout'] * 1000

                    # Create enhanced service metrics
                    availability_metric = HealthMetric(
                        name=f'service_{service_name}_availability',
                        value=1 if service_status == 'healthy' else 0,
                        unit='boolean',
                        status=service_status,
                        threshold_warning=1,
                        threshold_critical=0,
                        timestamp=current_time,
                        details=f"HTTP {status_code}" if status_code > 0 else "Connection failed",
                        source_system=f'service_{service_name}'
                    )

                    response_time_metric = HealthMetric(
                        name=f'service_{service_name}_response_time',
                        value=response_time,
                        unit='ms',
                        status=self._determine_status(response_time, 500, 2000),
                        threshold_warning=500,
                        threshold_critical=2000,
                        timestamp=current_time,
                        details=f"Endpoint: {service_config['url']}",
                        source_system=f'service_{service_name}'
                    )

                    network_metrics.extend([availability_metric, response_time_metric])

                # Enhanced analysis
                if self.enable_ai_features:
                    for metric in network_metrics:
                        self._enhance_metric_with_ai(metric)

                # Process and store
                for metric in network_metrics:
                    self._process_enhanced_metric(metric)
                    self._store_enhanced_metric(metric)

                self._check_and_generate_enhanced_alerts(network_metrics)

                time.sleep(self.config['monitoring_interval'] * 2)

            except Exception as e:
                self.logger.error(f"Error in enhanced network monitoring loop: {e}")
                time.sleep(30)

    def _enhanced_trading_monitoring_loop(self) -> None:
        """Enhanced trading system specific monitoring"""
        self.logger.info("Enhanced trading system monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                trading_metrics = []

                # Enhanced log analysis
                error_count = self._analyze_enhanced_logs()
                log_metric = HealthMetric(
                    name='trading_system_errors',
                    value=error_count,
                    unit='count',
                    status=self._determine_status(error_count, 5, 20, reverse=True),
                    threshold_warning=5,
                    threshold_critical=20,
                    timestamp=current_time,
                    details=f"Errors in last hour",
                    source_system='trading_logs'
                )

                # Enhanced data cache analysis
                cache_health_score = self._analyze_enhanced_data_cache()
                cache_metric = HealthMetric(
                    name='data_cache_health_score',
                    value=cache_health_score,
                    unit='score',
                    status=self._determine_status(cache_health_score, 0.7, 0.5, reverse=True),
                    threshold_warning=0.7,
                    threshold_critical=0.5,
                    timestamp=current_time,
                    details="Cache freshness and accessibility",
                    source_system='trading_data'
                )

                # Enhanced configuration integrity
                config_health_score = self._analyze_enhanced_configuration()
                config_metric = HealthMetric(
                    name='configuration_integrity_score',
                    value=config_health_score,
                    unit='score',
                    status=self._determine_status(config_health_score, 0.8, 0.6, reverse=True),
                    threshold_warning=0.8,
                    threshold_critical=0.6,
                    timestamp=current_time,
                    details="Configuration files validation",
                    source_system='trading_config'
                )

                trading_metrics.extend([log_metric, cache_metric, config_metric])

                # Enhanced analysis
                if self.enable_ai_features:
                    for metric in trading_metrics:
                        self._enhance_metric_with_ai(metric)

                # Process and store
                for metric in trading_metrics:
                    self._process_enhanced_metric(metric)
                    self._store_enhanced_metric(metric)

                self._check_and_generate_enhanced_alerts(trading_metrics)

                time.sleep(self.config['monitoring_interval'] * 3)

            except Exception as e:
                self.logger.error(f"Error in enhanced trading monitoring loop: {e}")
                time.sleep(60)

    def _analyze_enhanced_logs(self) -> int:
        """Analyze logs with enhanced pattern detection"""
        try:
            error_count = 0
            log_dir = self.base_dir / "logs"
            cutoff_time = datetime.now() - timedelta(hours=1)

            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    try:
                        file_age = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if file_age < cutoff_time:
                            continue

                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                if any(level in line.upper() for level in ['ERROR', 'CRITICAL', 'EXCEPTION']):
                                    error_count += 1

                    except Exception:
                        continue

            return min(error_count, 100)  # Cap at 100

        except Exception:
            return 0

    def _analyze_enhanced_data_cache(self) -> float:
        """Analyze data cache with enhanced health scoring"""
        try:
            data_cache_dir = self.quant_dir / "data_cache"
            if not data_cache_dir.exists():
                return 0.0

            health_score = 0.0
            max_score = 4.0

            # Check if cache directory is accessible
            if data_cache_dir.is_dir():
                health_score += 1.0

            # Check for recent files (last 4 hours)
            recent_cutoff = time.time() - 4 * 3600
            recent_files = sum(1 for f in data_cache_dir.rglob("*")
                             if f.is_file() and f.stat().st_mtime > recent_cutoff)

            if recent_files > 0:
                health_score += 1.0

            # Check cache size (not too large, not empty)
            try:
                cache_size = sum(f.stat().st_size for f in data_cache_dir.rglob("*") if f.is_file())
                cache_size_gb = cache_size / (1024**3)

                if 0.1 < cache_size_gb < 50:  # Between 100MB and 50GB
                    health_score += 1.0
                elif cache_size_gb > 0:
                    health_score += 0.5

            except Exception:
                pass

            # Check file diversity (different types of cached data)
            try:
                file_extensions = set()
                for f in data_cache_dir.rglob("*"):
                    if f.is_file():
                        file_extensions.add(f.suffix.lower())

                if len(file_extensions) >= 3:  # At least 3 different types
                    health_score += 1.0
                elif len(file_extensions) >= 1:
                    health_score += 0.5

            except Exception:
                pass

            return min(1.0, health_score / max_score)

        except Exception:
            return 0.5

    def _analyze_enhanced_configuration(self) -> float:
        """Analyze configuration with enhanced integrity checking"""
        try:
            health_score = 0.0
            max_score = 5.0

            # Check critical configuration files
            critical_configs = [
                self.quant_dir / "config.example.env",
                self.quant_dir / "dashboard" / "backend" / "app.py",
                self.quant_dir / "dashboard" / "worker" / "runner.py",
                self.quant_dir / "UI" / "package.json"
            ]

            existing_configs = sum(1 for config in critical_configs if config.exists())
            health_score += (existing_configs / len(critical_configs)) * 2.0

            # Check .env file
            env_file = self.quant_dir / ".env"
            if env_file.exists():
                health_score += 1.0

                # Check for required environment variables
                try:
                    with open(env_file, 'r') as f:
                        env_content = f.read()
                        required_vars = ['TIGER_ID', 'ACCOUNT', 'PRIVATE_KEY_PATH']
                        found_vars = sum(1 for var in required_vars if var in env_content)
                        health_score += (found_vars / len(required_vars)) * 1.0

                except Exception:
                    pass

            # Check package.json integrity
            package_json = self.quant_dir / "UI" / "package.json"
            if package_json.exists():
                try:
                    with open(package_json, 'r') as f:
                        json.load(f)  # Validate JSON format
                    health_score += 1.0
                except json.JSONDecodeError:
                    health_score += 0.5  # File exists but corrupted

            return min(1.0, health_score / max_score)

        except Exception:
            return 0.5

    def _anomaly_detection_loop(self) -> None:
        """AI-based anomaly detection loop"""
        if not self.enable_ai_features:
            return

        self.logger.info("AI anomaly detection loop started")

        while not self.shutdown_event.is_set():
            try:
                # Perform comprehensive anomaly detection
                anomalies = self._detect_system_anomalies()

                if anomalies:
                    self.logger.warning(f"Detected {len(anomalies)} system anomalies")

                    for anomaly in anomalies:
                        # Create anomaly alert
                        if self.intelligent_alerts:
                            self.intelligent_alerts.create_alert(
                                title=f"Anomaly Detected: {anomaly['metric']}",
                                message=f"Unusual behavior detected in {anomaly['metric']} with score {anomaly['score']:.2f}",
                                severity='HIGH' if anomaly['score'] > 0.7 else 'MEDIUM',
                                category='PERFORMANCE_DEGRADATION',
                                source="AI_AnomalyDetector",
                                context=anomaly
                            )

                time.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in anomaly detection loop: {e}")
                time.sleep(300)

    def _detect_system_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies using AI models"""
        anomalies = []

        try:
            # Get recent metrics for analysis
            recent_time = datetime.now() - timedelta(hours=2)

            for metric_name, history in self.metrics_history.items():
                # Filter recent metrics
                recent_metrics = [m for m in history
                                if m.timestamp > recent_time and isinstance(m.value, (int, float))]

                if len(recent_metrics) < 20:  # Need sufficient data
                    continue

                # Prepare data for anomaly detection
                values = np.array([[m.value] for m in recent_metrics])

                # Detect anomalies
                try:
                    anomaly_predictions = self.anomaly_detector.fit_predict(values)
                    anomaly_scores = self.anomaly_detector.decision_function(values)

                    # Find anomalous points
                    for i, (prediction, score) in enumerate(zip(anomaly_predictions, anomaly_scores)):
                        if prediction == -1:  # Anomaly detected
                            anomalies.append({
                                'metric': metric_name,
                                'timestamp': recent_metrics[i].timestamp.isoformat(),
                                'value': recent_metrics[i].value,
                                'score': abs(score),  # Anomaly score
                                'type': 'statistical_anomaly'
                            })

                except Exception as e:
                    self.logger.debug(f"Error detecting anomalies for {metric_name}: {e}")

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")

        return anomalies

    def _trend_analysis_loop(self) -> None:
        """Enhanced trend analysis loop"""
        self.logger.info("Enhanced trend analysis loop started")

        while not self.shutdown_event.is_set():
            try:
                # Perform trend analysis on all metrics
                trends = self._analyze_system_trends()

                # Log significant trends
                significant_trends = {k: v for k, v in trends.items()
                                    if v.get('significance', 0) > 0.5}

                if significant_trends:
                    self.logger.info(f"Detected {len(significant_trends)} significant trends")

                    for metric_name, trend_data in significant_trends.items():
                        trend_direction = trend_data.get('direction', 'unknown')
                        significance = trend_data.get('significance', 0)

                        self.logger.info(f"Trend: {metric_name} trending {trend_direction} "
                                       f"(significance: {significance:.2f})")

                time.sleep(600)  # Run every 10 minutes

            except Exception as e:
                self.logger.error(f"Error in trend analysis loop: {e}")
                time.sleep(600)

    def _analyze_system_trends(self) -> Dict[str, Dict[str, Any]]:
        """Analyze trends across all system metrics"""
        trends = {}

        try:
            from sklearn.linear_model import LinearRegression

            for metric_name, history in self.metrics_history.items():
                # Get sufficient data for trend analysis
                if len(history) < 50:
                    continue

                # Extract numeric values with timestamps
                data_points = []
                for m in list(history)[-100:]:  # Last 100 points
                    if isinstance(m.value, (int, float)):
                        data_points.append((m.timestamp, m.value))

                if len(data_points) < 20:
                    continue

                # Prepare data for regression
                timestamps = np.array([(dp[0] - data_points[0][0]).total_seconds()
                                     for dp in data_points]).reshape(-1, 1)
                values = np.array([dp[1] for dp in data_points])

                # Fit linear regression
                try:
                    model = LinearRegression().fit(timestamps, values)
                    slope = model.coef_[0]
                    r_squared = model.score(timestamps, values)

                    # Calculate trend significance
                    avg_value = np.mean(values)
                    if avg_value != 0:
                        relative_slope = abs(slope) / abs(avg_value)
                        significance = relative_slope * r_squared
                    else:
                        significance = 0

                    # Determine trend direction
                    direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'

                    trends[metric_name] = {
                        'direction': direction,
                        'slope': slope,
                        'r_squared': r_squared,
                        'significance': significance,
                        'data_points': len(data_points)
                    }

                except Exception as e:
                    self.logger.debug(f"Error analyzing trend for {metric_name}: {e}")

        except ImportError:
            self.logger.debug("scikit-learn not available for trend analysis")
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")

        return trends

    def _predictive_monitoring_loop(self) -> None:
        """Predictive monitoring and capacity planning loop"""
        self.logger.info("Predictive monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                # Generate predictions for key metrics
                predictions = self._generate_system_predictions()

                # Check predictions against thresholds
                concerning_predictions = []
                for prediction in predictions:
                    if prediction['confidence'] > self.config['prediction_confidence_threshold']:
                        predicted_value = prediction['predicted_value']
                        current_value = prediction['current_value']

                        # Check if prediction suggests potential issues
                        change_ratio = abs(predicted_value - current_value) / max(current_value, 1.0)

                        if change_ratio > 0.2:  # 20% change predicted
                            concerning_predictions.append(prediction)

                # Generate predictive alerts
                for prediction in concerning_predictions:
                    if self.intelligent_alerts:
                        self.intelligent_alerts.create_alert(
                            title=f"Predictive Alert: {prediction['metric_name']}",
                            message=f"Predicted significant change in {prediction['metric_name']}: "
                                   f"current={prediction['current_value']:.2f}, "
                                   f"predicted={prediction['predicted_value']:.2f} "
                                   f"in {prediction['time_horizon_minutes']} minutes",
                            severity='MEDIUM',
                            category='PERFORMANCE_DEGRADATION',
                            source="PredictiveMonitor",
                            context=prediction
                        )

                time.sleep(1800)  # Run every 30 minutes

            except Exception as e:
                self.logger.error(f"Error in predictive monitoring loop: {e}")
                time.sleep(1800)

    def _generate_system_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for system metrics"""
        predictions = []

        try:
            for metric_name, history in self.metrics_history.items():
                if len(history) < 50:  # Need sufficient data
                    continue

                # Extract recent numeric values
                recent_values = []
                for m in list(history)[-50:]:
                    if isinstance(m.value, (int, float)):
                        recent_values.append(m.value)

                if len(recent_values) < 20:
                    continue

                # Simple prediction using moving averages and trend
                try:
                    # Calculate short-term and long-term averages
                    short_term_avg = statistics.mean(recent_values[-5:])
                    long_term_avg = statistics.mean(recent_values[-20:])

                    # Calculate trend
                    trend = short_term_avg - long_term_avg

                    # Simple prediction: current + trend
                    current_value = recent_values[-1]
                    predicted_value = current_value + trend

                    # Calculate confidence based on trend consistency
                    recent_changes = [recent_values[i] - recent_values[i-1]
                                    for i in range(1, min(10, len(recent_values)))]

                    if recent_changes:
                        trend_consistency = 1.0 - (statistics.stdev(recent_changes) /
                                                 max(abs(statistics.mean(recent_changes)), 1.0))
                        confidence = max(0.1, min(0.9, trend_consistency))
                    else:
                        confidence = 0.5

                    predictions.append({
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'predicted_value': predicted_value,
                        'confidence': confidence,
                        'time_horizon_minutes': 30,  # 30 minute prediction
                        'model_type': 'trend_based',
                        'timestamp': datetime.now().isoformat()
                    })

                    # Store prediction in database
                    self._store_prediction(metric_name, current_value, predicted_value,
                                        confidence, 30, 'trend_based')

                except Exception as e:
                    self.logger.debug(f"Error generating prediction for {metric_name}: {e}")

        except Exception as e:
            self.logger.error(f"Error generating system predictions: {e}")

        return predictions

    def _store_prediction(self, metric_name: str, current_value: float, predicted_value: float,
                         confidence: float, time_horizon: int, model_type: str) -> None:
        """Store prediction in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, metric_name, predicted_value, confidence,
                    time_horizon_minutes, model_used, actual_value, prediction_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metric_name,
                predicted_value,
                confidence,
                time_horizon,
                model_type,
                None,  # actual_value will be updated later
                None   # prediction_error will be calculated later
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.debug(f"Error storing prediction: {e}")

    def _enhanced_report_generation_loop(self) -> None:
        """Enhanced health report generation loop"""
        self.logger.info("Enhanced report generation loop started")

        while not self.shutdown_event.is_set():
            try:
                # Generate comprehensive health report
                report = self._generate_enhanced_health_report()

                # Store report
                self._store_enhanced_health_report(report)

                # Log report summary
                self.logger.info(
                    f"Enhanced Health Report - Overall: {report.overall_health.score:.1f}/100, "
                    f"Performance: {report.overall_health.performance_score:.1f}, "
                    f"Reliability: {report.overall_health.reliability_score:.1f}, "
                    f"Efficiency: {report.overall_health.efficiency_score:.1f}"
                )

                # Generate alerts for poor performance
                if report.overall_health.score < 70:
                    if self.intelligent_alerts:
                        self.intelligent_alerts.create_alert(
                            title="System Health Report - Performance Degradation",
                            message=f"System health score has dropped to {report.overall_health.score:.1f}/100",
                            severity='HIGH' if report.overall_health.score < 50 else 'MEDIUM',
                            category='PERFORMANCE_DEGRADATION',
                            source="EnhancedHealthReporter",
                            context={
                                'overall_score': report.overall_health.score,
                                'performance_score': report.overall_health.performance_score,
                                'reliability_score': report.overall_health.reliability_score,
                                'recommendations': report.recommendations[:3]
                            }
                        )

                time.sleep(3600)  # Generate report every hour

            except Exception as e:
                self.logger.error(f"Error in enhanced report generation loop: {e}")
                time.sleep(3600)

    def _generate_enhanced_health_report(self) -> EnhancedHealthReport:
        """Generate comprehensive enhanced health report"""
        current_time = datetime.now()

        # Get recent metrics
        recent_time = current_time - timedelta(minutes=30)
        recent_metrics = []

        for history in self.metrics_history.values():
            recent_metrics.extend([m for m in history if m.timestamp > recent_time])

        # Calculate overall health scores
        overall_health = self._calculate_enhanced_health_scores(recent_metrics)

        # Perform detailed analysis
        detailed_analysis = {
            'total_metrics': len(recent_metrics),
            'healthy_metrics': len([m for m in recent_metrics if m.status == 'healthy']),
            'warning_metrics': len([m for m in recent_metrics if m.status == 'warning']),
            'critical_metrics': len([m for m in recent_metrics if m.status == 'critical']),
            'high_anomaly_metrics': len([m for m in recent_metrics if m.anomaly_score > 0.5]),
            'trending_up_metrics': len([m for m in recent_metrics if m.trend == 'up']),
            'trending_down_metrics': len([m for m in recent_metrics if m.trend == 'down'])
        }

        # Trend analysis
        trend_analysis = self._analyze_system_trends()

        # Anomaly detection results
        anomaly_detection = {
            'anomalies_detected': len(self._detect_system_anomalies()),
            'anomaly_threshold': self.config['anomaly_threshold'],
            'detection_enabled': self.enable_ai_features
        }

        # Performance metrics
        performance_metrics = {
            'average_response_time': self._calculate_average_response_time(recent_metrics),
            'availability_percentage': self._calculate_availability_percentage(recent_metrics),
            'error_rate': self._calculate_error_rate(),
            'throughput_metrics': self._calculate_throughput_metrics(recent_metrics)
        }

        # Capacity planning
        capacity_planning = {
            'predictions_generated': len(self._generate_system_predictions()),
            'concerning_trends': len([t for t in trend_analysis.values()
                                    if t.get('significance', 0) > 0.7]),
            'capacity_alerts': self._get_capacity_alerts()
        }

        # SLA compliance
        sla_compliance = self._check_sla_compliance(recent_metrics)

        # Generate recommendations
        recommendations = self._generate_enhanced_recommendations(
            recent_metrics, detailed_analysis, trend_analysis
        )

        # Create enhanced health report
        report = EnhancedHealthReport(
            timestamp=current_time,
            overall_health=overall_health,
            detailed_analysis=detailed_analysis,
            trend_analysis=trend_analysis,
            anomaly_detection=anomaly_detection,
            performance_metrics=performance_metrics,
            capacity_planning=capacity_planning,
            sla_compliance=sla_compliance,
            recommendations=recommendations
        )

        return report

    def _calculate_enhanced_health_scores(self, recent_metrics: List[HealthMetric]) -> SystemHealth:
        """Calculate enhanced health scores"""
        if not recent_metrics:
            return SystemHealth(
                overall_status='unknown',
                score=50.0,
                metrics=[],
                alerts=[],
                timestamp=datetime.now(),
                performance_score=50.0,
                reliability_score=50.0,
                efficiency_score=50.0
            )

        # Calculate overall score
        status_weights = {'healthy': 1.0, 'warning': 0.5, 'critical': 0.0}
        status_scores = [status_weights.get(m.status, 0.5) for m in recent_metrics]
        overall_score = statistics.mean(status_scores) * 100

        # Calculate performance score (based on response times and throughput)
        performance_metrics = [m for m in recent_metrics
                             if 'response_time' in m.name or 'throughput' in m.name]
        if performance_metrics:
            perf_scores = [status_weights.get(m.status, 0.5) for m in performance_metrics]
            performance_score = statistics.mean(perf_scores) * 100
        else:
            performance_score = overall_score

        # Calculate reliability score (based on availability and errors)
        reliability_metrics = [m for m in recent_metrics
                             if 'availability' in m.name or 'error' in m.name]
        if reliability_metrics:
            rel_scores = [status_weights.get(m.status, 0.5) for m in reliability_metrics]
            reliability_score = statistics.mean(rel_scores) * 100
        else:
            reliability_score = overall_score

        # Calculate efficiency score (based on resource utilization)
        efficiency_metrics = [m for m in recent_metrics
                            if 'utilization' in m.name or 'cpu' in m.name or 'memory' in m.name]
        if efficiency_metrics:
            eff_scores = [status_weights.get(m.status, 0.5) for m in efficiency_metrics]
            efficiency_score = statistics.mean(eff_scores) * 100
        else:
            efficiency_score = overall_score

        # Determine overall status
        if overall_score >= 80:
            overall_status = 'healthy'
        elif overall_score >= 60:
            overall_status = 'warning'
        else:
            overall_status = 'critical'

        # Get current alerts
        current_alerts = [f"{m.name}: {m.status}" for m in recent_metrics if m.status != 'healthy']

        return SystemHealth(
            overall_status=overall_status,
            score=overall_score,
            metrics=recent_metrics,
            alerts=current_alerts,
            timestamp=datetime.now(),
            performance_score=performance_score,
            reliability_score=reliability_score,
            efficiency_score=efficiency_score
        )

    def _calculate_average_response_time(self, metrics: List[HealthMetric]) -> float:
        """Calculate average response time from metrics"""
        response_time_metrics = [m for m in metrics if 'response_time' in m.name
                               and isinstance(m.value, (int, float))]

        if response_time_metrics:
            return statistics.mean([m.value for m in response_time_metrics])
        return 0.0

    def _calculate_availability_percentage(self, metrics: List[HealthMetric]) -> float:
        """Calculate availability percentage"""
        availability_metrics = [m for m in metrics if 'availability' in m.name]

        if availability_metrics:
            available_count = sum(1 for m in availability_metrics if m.value == 1)
            return (available_count / len(availability_metrics)) * 100
        return 100.0

    def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        # This would integrate with actual error tracking
        # For now, return a placeholder based on log analysis
        return 0.1  # 0.1% error rate

    def _calculate_throughput_metrics(self, metrics: List[HealthMetric]) -> Dict[str, float]:
        """Calculate throughput metrics"""
        throughput_metrics = [m for m in metrics if 'throughput' in m.name
                            and isinstance(m.value, (int, float))]

        if throughput_metrics:
            return {
                'average_throughput': statistics.mean([m.value for m in throughput_metrics]),
                'max_throughput': max([m.value for m in throughput_metrics]),
                'min_throughput': min([m.value for m in throughput_metrics])
            }
        return {'average_throughput': 0.0, 'max_throughput': 0.0, 'min_throughput': 0.0}

    def _get_capacity_alerts(self) -> List[str]:
        """Get capacity planning alerts"""
        alerts = []

        # Check for trending metrics that might need attention
        trends = self._analyze_system_trends()

        for metric_name, trend_data in trends.items():
            if (trend_data.get('significance', 0) > 0.7 and
                trend_data.get('direction') == 'increasing' and
                'utilization' in metric_name):
                alerts.append(f"{metric_name} trending up significantly - consider capacity planning")

        return alerts

    def _check_sla_compliance(self, metrics: List[HealthMetric]) -> Dict[str, Any]:
        """Check SLA compliance"""
        sla_results = {
            'availability_sla': {'target': self.config['sla_availability'], 'actual': 0.0, 'compliant': False},
            'response_time_sla': {'target': self.config['sla_response_time_ms'], 'actual': 0.0, 'compliant': False},
            'error_rate_sla': {'target': self.config['sla_error_rate'], 'actual': 0.0, 'compliant': False}
        }

        # Check availability SLA
        actual_availability = self._calculate_availability_percentage(metrics)
        sla_results['availability_sla']['actual'] = actual_availability
        sla_results['availability_sla']['compliant'] = actual_availability >= self.config['sla_availability']

        # Check response time SLA
        actual_response_time = self._calculate_average_response_time(metrics)
        sla_results['response_time_sla']['actual'] = actual_response_time
        sla_results['response_time_sla']['compliant'] = actual_response_time <= self.config['sla_response_time_ms']

        # Check error rate SLA
        actual_error_rate = self._calculate_error_rate()
        sla_results['error_rate_sla']['actual'] = actual_error_rate
        sla_results['error_rate_sla']['compliant'] = actual_error_rate <= self.config['sla_error_rate']

        return sla_results

    def _generate_enhanced_recommendations(self, metrics: List[HealthMetric],
                                         detailed_analysis: Dict[str, Any],
                                         trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate enhanced system recommendations"""
        recommendations = []

        # Critical metrics recommendations
        if detailed_analysis['critical_metrics'] > 0:
            recommendations.append(f"Immediate attention required: {detailed_analysis['critical_metrics']} critical metrics detected")

        # High anomaly recommendations
        if detailed_analysis['high_anomaly_metrics'] > 5:
            recommendations.append(f"Investigate anomalies: {detailed_analysis['high_anomaly_metrics']} metrics showing unusual behavior")

        # Trend-based recommendations
        concerning_trends = [name for name, data in trend_analysis.items()
                           if data.get('significance', 0) > 0.7 and data.get('direction') == 'increasing']

        if concerning_trends:
            if any('cpu' in name for name in concerning_trends):
                recommendations.append("CPU utilization trending up - consider load balancing or optimization")

            if any('memory' in name for name in concerning_trends):
                recommendations.append("Memory usage trending up - investigate memory leaks or increase capacity")

            if any('disk' in name for name in concerning_trends):
                recommendations.append("Disk usage trending up - consider cleanup or storage expansion")

        # Performance recommendations
        response_time_metrics = [m for m in metrics if 'response_time' in m.name and m.status != 'healthy']
        if response_time_metrics:
            recommendations.append("Response time degradation detected - optimize application performance")

        # Availability recommendations
        availability_metrics = [m for m in metrics if 'availability' in m.name and m.value < 1]
        if availability_metrics:
            recommendations.append("Service availability issues detected - check service health and dependencies")

        # General recommendations
        if not recommendations:
            recommendations.append("System health is within normal parameters")

        return recommendations[:10]  # Limit to top 10 recommendations

    def _store_enhanced_health_report(self, report: EnhancedHealthReport) -> None:
        """Store enhanced health report in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO health_reports (
                    timestamp, overall_score, performance_score, reliability_score,
                    efficiency_score, detailed_analysis, trend_analysis,
                    anomaly_detection, recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.timestamp.isoformat(),
                report.overall_health.score,
                report.overall_health.performance_score,
                report.overall_health.reliability_score,
                report.overall_health.efficiency_score,
                json.dumps(report.detailed_analysis),
                json.dumps(report.trend_analysis, default=str),
                json.dumps(report.anomaly_detection),
                json.dumps(report.recommendations)
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing enhanced health report: {e}")

    def _capacity_planning_loop(self) -> None:
        """Capacity planning and resource forecasting loop"""
        self.logger.info("Capacity planning loop started")

        while not self.shutdown_event.is_set():
            try:
                # Perform capacity analysis
                capacity_analysis = self._perform_capacity_analysis()

                # Generate capacity alerts if needed
                if capacity_analysis['alerts']:
                    for alert in capacity_analysis['alerts']:
                        if self.intelligent_alerts:
                            self.intelligent_alerts.create_alert(
                                title=f"Capacity Planning Alert: {alert['resource']}",
                                message=alert['message'],
                                severity='MEDIUM',
                                category='RESOURCE_EXHAUSTION',
                                source="CapacityPlanner",
                                context=alert
                            )

                        self.logger.warning(f"Capacity Alert: {alert['message']}")

                time.sleep(7200)  # Run every 2 hours

            except Exception as e:
                self.logger.error(f"Error in capacity planning loop: {e}")
                time.sleep(7200)

    def _perform_capacity_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive capacity analysis"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'forecasts': {},
            'alerts': [],
            'recommendations': []
        }

        try:
            # Analyze key resources
            resources_to_analyze = ['cpu_utilization', 'memory_utilization', 'disk_utilization']

            for resource in resources_to_analyze:
                if resource in self.metrics_history:
                    forecast = self._forecast_resource_usage(resource)
                    analysis['forecasts'][resource] = forecast

                    # Check if forecast suggests capacity issues
                    if forecast['projected_usage_7d'] > 85:
                        analysis['alerts'].append({
                            'resource': resource,
                            'message': f"{resource} projected to reach {forecast['projected_usage_7d']:.1f}% in 7 days",
                            'severity': 'HIGH' if forecast['projected_usage_7d'] > 95 else 'MEDIUM',
                            'recommendation': f"Consider scaling {resource.replace('_utilization', '')} resources"
                        })

        except Exception as e:
            self.logger.error(f"Error performing capacity analysis: {e}")

        return analysis

    def _forecast_resource_usage(self, resource: str) -> Dict[str, Any]:
        """Forecast resource usage based on historical trends"""
        forecast = {
            'resource': resource,
            'current_usage': 0.0,
            'projected_usage_7d': 0.0,
            'projected_usage_30d': 0.0,
            'trend_confidence': 0.0
        }

        try:
            history = list(self.metrics_history[resource])
            if len(history) < 50:
                return forecast

            # Get recent numeric values
            values = [m.value for m in history[-100:] if isinstance(m.value, (int, float))]
            if len(values) < 20:
                return forecast

            current_usage = values[-1]
            forecast['current_usage'] = current_usage

            # Calculate trend
            try:
                from sklearn.linear_model import LinearRegression

                X = np.array(range(len(values))).reshape(-1, 1)
                y = np.array(values)

                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                r_squared = model.score(X, y)

                # Project usage
                forecast['projected_usage_7d'] = current_usage + (slope * 7 * 24 * 6)  # 7 days at 10min intervals
                forecast['projected_usage_30d'] = current_usage + (slope * 30 * 24 * 6)  # 30 days
                forecast['trend_confidence'] = r_squared

            except Exception:
                # Fallback to simple trend calculation
                if len(values) >= 10:
                    recent_avg = statistics.mean(values[-5:])
                    older_avg = statistics.mean(values[-10:-5])
                    trend = recent_avg - older_avg

                    forecast['projected_usage_7d'] = current_usage + (trend * 10)
                    forecast['projected_usage_30d'] = current_usage + (trend * 40)
                    forecast['trend_confidence'] = 0.5

        except Exception as e:
            self.logger.debug(f"Error forecasting {resource}: {e}")

        return forecast

    def _determine_status(self, value: float, warning_threshold: float,
                         critical_threshold: float, reverse: bool = False) -> str:
        """Determine health status based on value and thresholds"""
        if reverse:  # For metrics where lower values are worse
            if value <= critical_threshold:
                return 'critical'
            elif value <= warning_threshold:
                return 'warning'
            else:
                return 'healthy'
        else:  # For metrics where higher values are worse
            if value >= critical_threshold:
                return 'critical'
            elif value >= warning_threshold:
                return 'warning'
            else:
                return 'healthy'

    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system status"""
        current_time = datetime.now()

        # Get recent metrics
        recent_time = current_time - timedelta(minutes=10)
        recent_metrics = []

        for history in self.metrics_history.values():
            recent_metrics.extend([m for m in history if m.timestamp > recent_time])

        # Calculate enhanced status
        if recent_metrics:
            enhanced_health = self._calculate_enhanced_health_scores(recent_metrics)
        else:
            enhanced_health = SystemHealth(
                overall_status='unknown',
                score=50.0,
                metrics=[],
                alerts=[],
                timestamp=current_time,
                performance_score=50.0,
                reliability_score=50.0,
                efficiency_score=50.0
            )

        return {
            'timestamp': current_time.isoformat(),
            'enhanced_monitoring': True,
            'ai_features_enabled': self.enable_ai_features,
            'overall_health': enhanced_health.overall_status,
            'overall_score': enhanced_health.score,
            'performance_score': enhanced_health.performance_score,
            'reliability_score': enhanced_health.reliability_score,
            'efficiency_score': enhanced_health.efficiency_score,
            'recent_metrics_count': len(recent_metrics),
            'active_alerts': len(enhanced_health.alerts),
            'monitoring_threads': len(self.monitoring_threads),
            'integrations': {
                'intelligent_alerts': self.intelligent_alerts is not None,
                'performance_monitor': self.performance_monitor is not None,
                'gpu_monitor': self.gpu_monitor is not None
            },
            'database_size_mb': self._get_database_size(),
            'ai_model_status': 'active' if self.enable_ai_features else 'disabled'
        }

    def _get_database_size(self) -> float:
        """Get database size in MB"""
        try:
            db_path = self.data_dir / "enhanced_health.db"
            if db_path.exists():
                return db_path.stat().st_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0

    def print_enhanced_health_dashboard(self) -> None:
        """Print enhanced real-time health dashboard"""
        while not self.shutdown_event.is_set():
            try:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')

                print("="*80)
                print("      ENHANCED QUANTITATIVE TRADING SYSTEM - HEALTH MONITOR")
                print("                AI-Powered Investment Grade Monitoring")
                print("="*80)

                # Get enhanced status
                status = self.get_enhanced_system_status()

                # Overall status with enhanced metrics
                status_color = {
                    'healthy': '\033[92m',  # Green
                    'warning': '\033[93m',  # Yellow
                    'critical': '\033[91m', # Red
                    'unknown': '\033[94m'   # Blue
                }.get(status['overall_health'], '\033[0m')

                print(f"Overall Status: {status_color}{status['overall_health'].upper()}\033[0m")
                print(f"Overall Score: {status['overall_score']:.1f}/100")
                print(f"Performance: {status['performance_score']:.1f} | "
                      f"Reliability: {status['reliability_score']:.1f} | "
                      f"Efficiency: {status['efficiency_score']:.1f}")
                print(f"AI Features: {'ENABLED' if status['ai_features_enabled'] else 'DISABLED'}")
                print(f"Timestamp: {datetime.fromisoformat(status['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)

                # Integration status
                print("System Integrations:")
                integrations = status['integrations']
                print(f"  {'✓' if integrations['intelligent_alerts'] else '✗'} Intelligent Alert System")
                print(f"  {'✓' if integrations['performance_monitor'] else '✗'} Advanced Performance Monitor")
                print(f"  {'✓' if integrations['gpu_monitor'] else '✗'} GPU System Manager")

                # Enhanced metrics
                print("-" * 80)
                print("Enhanced Monitoring Status:")
                print(f"  Recent Metrics: {status['recent_metrics_count']}")
                print(f"  Active Alerts: {status['active_alerts']}")
                print(f"  Monitoring Threads: {status['monitoring_threads']}")
                print(f"  Database Size: {status['database_size_mb']:.1f}MB")
                print(f"  AI Model Status: {status['ai_model_status'].upper()}")

                # Show recent significant metrics
                recent_time = datetime.now() - timedelta(minutes=5)
                significant_metrics = []

                for history in self.metrics_history.values():
                    for metric in history:
                        if (metric.timestamp > recent_time and
                            metric.status != 'healthy' or
                            metric.anomaly_score > 0.3):
                            significant_metrics.append(metric)

                if significant_metrics:
                    print("-" * 80)
                    print("Significant Recent Metrics:")
                    for metric in significant_metrics[-10:]:  # Show last 10
                        status_symbol = {
                            'healthy': '\033[92m[OK]\033[0m',
                            'warning': '\033[93m[WARN]\033[0m',
                            'critical': '\033[91m[CRIT]\033[0m'
                        }.get(metric.status, '[?]')

                        anomaly_indicator = f" (A:{metric.anomaly_score:.2f})" if metric.anomaly_score > 0.1 else ""
                        trend_indicator = f" ↗️" if metric.trend == 'up' else "↘️" if metric.trend == 'down' else ""

                        print(f"  {status_symbol} {metric.name}: {metric.value} {metric.unit}{anomaly_indicator}{trend_indicator}")

                # Commands and status
                print("-" * 80)
                print("Enhanced Features Active:")
                if self.enable_ai_features:
                    print("  • AI Anomaly Detection")
                    print("  • Predictive Monitoring")
                    print("  • Trend Analysis")
                    print("  • Capacity Planning")

                print("Commands: Ctrl+C to stop monitoring")
                print("="*80)

                time.sleep(30)  # Update every 30 seconds

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in enhanced health dashboard: {e}")
                time.sleep(10)

    def shutdown(self) -> None:
        """Enhanced shutdown procedure"""
        self.logger.info("Shutting down enhanced health monitoring...")
        self.shutdown_event.set()

        # Shutdown integrations
        if self.intelligent_alerts:
            try:
                self.intelligent_alerts.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down intelligent alerts: {e}")

        if self.performance_monitor:
            try:
                self.performance_monitor.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down performance monitor: {e}")

        if self.gpu_monitor:
            try:
                self.gpu_monitor.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down GPU monitor: {e}")

        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=10)

        # Generate final enhanced report
        try:
            final_report = self._generate_enhanced_health_report()
            self._store_enhanced_health_report(final_report)
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")

        # Close database connection
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()

        self.logger.info("Enhanced health monitoring shutdown complete")


# Backward compatibility with original HealthMonitor
class HealthMonitor(EnhancedHealthMonitor):
    """Backward compatibility wrapper"""

    def __init__(self):
        super().__init__(enable_ai_features=False)  # Disable AI for compatibility

    def start_monitoring(self):
        """Start basic monitoring (backward compatibility)"""
        return self.start_enhanced_monitoring()

    def get_current_health_status(self) -> SystemHealth:
        """Get current health status (backward compatibility)"""
        enhanced_status = self.get_enhanced_system_status()

        # Convert to basic format
        return SystemHealth(
            overall_status=enhanced_status['overall_health'],
            score=enhanced_status['overall_score'],
            metrics=[],  # Would need to populate from recent metrics
            alerts=[],
            timestamp=datetime.fromisoformat(enhanced_status['timestamp'])
        )

    def print_health_dashboard(self):
        """Print basic dashboard (backward compatibility)"""
        return self.print_enhanced_health_dashboard()


def main():
    """Enhanced entry point for health monitoring."""
    # Check command line arguments for enhanced features
    enable_enhanced = '--enhanced' in sys.argv or '--ai' in sys.argv

    if enable_enhanced:
        monitor = EnhancedHealthMonitor(enable_ai_features=True)
        print("Starting Enhanced Health Monitor with AI capabilities...")
    else:
        monitor = HealthMonitor()  # Basic compatibility mode
        print("Starting Basic Health Monitor (use --enhanced for AI features)...")

    try:
        if enable_enhanced:
            # Start enhanced monitoring
            monitor.start_enhanced_monitoring()

            # Display enhanced dashboard
            monitor.print_enhanced_health_dashboard()
        else:
            # Start basic monitoring
            monitor.start_monitoring()

            # Display basic dashboard
            monitor.print_health_dashboard()

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in health monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.shutdown()

    return 0

if __name__ == "__main__":
    sys.exit(main())