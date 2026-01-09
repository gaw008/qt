#!/usr/bin/env python3
"""
Advanced Performance Monitor - Comprehensive System Performance Analysis
高级性能监控器 - 综合系统性能分析

Investment-grade performance monitoring providing:
- Real-time performance metrics collection and analysis
- Bottleneck identification and root cause analysis
- Capacity planning with predictive analytics
- SLA monitoring and compliance tracking
- Performance optimization recommendations
- Historical trend analysis and reporting

Features:
- Multi-dimensional performance tracking (CPU, Memory, Disk, Network, GPU)
- AI-powered anomaly detection and performance prediction
- Automatic bottleneck detection with impact analysis
- Resource utilization forecasting and capacity planning
- Performance regression detection and alerting
- Comprehensive performance dashboards and reports

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
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import statistics
from collections import deque, defaultdict
import psutil
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

class PerformanceLevel(Enum):
    """Performance level classification"""
    EXCELLENT = ("EXCELLENT", 0.9, "#00FF00")
    GOOD = ("GOOD", 0.7, "#7FFF00")
    ACCEPTABLE = ("ACCEPTABLE", 0.5, "#FFFF00")
    POOR = ("POOR", 0.3, "#FF7F00")
    CRITICAL = ("CRITICAL", 0.1, "#FF0000")

class MetricType(Enum):
    """Performance metric types"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    UTILIZATION = "utilization"
    CAPACITY = "capacity"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"

class BottleneckType(Enum):
    """System bottleneck classifications"""
    CPU_BOUND = "CPU_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    IO_BOUND = "IO_BOUND"
    NETWORK_BOUND = "NETWORK_BOUND"
    GPU_BOUND = "GPU_BOUND"
    DATABASE_BOUND = "DATABASE_BOUND"
    DISK_BOUND = "DISK_BOUND"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime
    source: str

    # Thresholds
    excellent_threshold: float
    good_threshold: float
    acceptable_threshold: float

    # Analysis
    performance_level: PerformanceLevel
    trend_direction: str  # "up", "down", "stable"
    trend_strength: float  # 0-1
    anomaly_score: float  # 0-1

    # Context
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report"""
    timestamp: datetime
    duration_minutes: int

    # Overall performance
    overall_score: float  # 0-1
    performance_level: PerformanceLevel

    # Metrics summary
    metrics: List[PerformanceMetric]
    metric_summary: Dict[str, Dict[str, float]]

    # Analysis
    bottlenecks: List[Dict[str, Any]]
    trends: Dict[str, Dict[str, float]]
    anomalies: List[Dict[str, Any]]

    # Recommendations
    optimization_recommendations: List[str]
    capacity_recommendations: List[str]

    # SLA compliance
    sla_compliance: Dict[str, bool]
    sla_violations: List[Dict[str, Any]]

class AdvancedPerformanceMonitor:
    """Comprehensive performance monitoring and analysis system"""

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.config_path = config_path or self.base_dir / "performance_config.json"
        self.data_dir = self.base_dir / "data" / "performance"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()
        self.shutdown_event = threading.Event()

        # Configuration
        self.config = self._load_configuration()

        # Database for metrics storage
        self.db_connection = self._initialize_database()

        # AI/ML components
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trend_analyzer = LinearRegression()
        self.feature_scaler = StandardScaler()

        # Performance tracking
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.performance_targets: Dict[str, Dict[str, float]] = {}

        # Monitoring state
        self.monitoring_threads: List[threading.Thread] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="PerfMonitor")

        # System baselines
        self._establish_baselines()

        self.logger.info("Advanced Performance Monitor initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for performance monitoring"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('AdvancedPerformanceMonitor')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '\033[92m%(asctime)s\033[0m - \033[94mPERF\033[0m - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"performance_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _load_configuration(self) -> Dict[str, Any]:
        """Load performance monitoring configuration"""
        default_config = {
            "monitoring": {
                "interval_seconds": 10,
                "detailed_interval_seconds": 60,
                "history_retention_hours": 168,  # 1 week
                "batch_size": 100
            },
            "thresholds": {
                "cpu_utilization": {"excellent": 30, "good": 50, "acceptable": 70},
                "memory_utilization": {"excellent": 40, "good": 60, "acceptable": 80},
                "disk_utilization": {"excellent": 50, "good": 70, "acceptable": 85},
                "network_utilization": {"excellent": 30, "good": 50, "acceptable": 70},
                "response_time_ms": {"excellent": 100, "good": 500, "acceptable": 1000},
                "throughput_rps": {"excellent": 1000, "good": 500, "acceptable": 100}
            },
            "sla_targets": {
                "availability_percent": 99.9,
                "response_time_ms": 500,
                "error_rate_percent": 0.1,
                "throughput_min_rps": 100
            },
            "ai_analysis": {
                "anomaly_detection_enabled": True,
                "trend_analysis_enabled": True,
                "predictive_analysis_enabled": True,
                "baseline_learning_enabled": True
            },
            "alerts": {
                "performance_degradation_threshold": 0.2,
                "bottleneck_detection_threshold": 0.8,
                "sla_violation_alerts": True,
                "predictive_alerts_enabled": True
            }
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Could not load performance config, using defaults: {e}")

        return default_config

    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize database for performance metrics storage"""
        db_path = self.data_dir / "performance_metrics.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                name TEXT,
                value REAL,
                unit TEXT,
                metric_type TEXT,
                source TEXT,
                performance_level TEXT,
                trend_direction TEXT,
                trend_strength REAL,
                anomaly_score REAL,
                tags TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS performance_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                duration_minutes INTEGER,
                overall_score REAL,
                performance_level TEXT,
                bottlenecks TEXT,
                trends TEXT,
                anomalies TEXT,
                recommendations TEXT,
                sla_compliance TEXT
            );

            CREATE TABLE IF NOT EXISTS baselines (
                metric_name TEXT PRIMARY KEY,
                baseline_value REAL,
                std_deviation REAL,
                last_updated DATETIME,
                sample_count INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
            CREATE INDEX IF NOT EXISTS idx_reports_timestamp ON performance_reports(timestamp);
        """)

        conn.commit()
        return conn

    def _establish_baselines(self) -> None:
        """Establish performance baselines"""
        try:
            # Load existing baselines from database
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT metric_name, baseline_value, std_deviation FROM baselines")

            for row in cursor.fetchall():
                metric_name, baseline_value, std_deviation = row
                self.baseline_metrics[metric_name] = {
                    'baseline': baseline_value,
                    'std_dev': std_deviation
                }

            self.logger.info(f"Loaded {len(self.baseline_metrics)} baseline metrics")

        except Exception as e:
            self.logger.error(f"Error loading baselines: {e}")

    def start_monitoring(self) -> None:
        """Start comprehensive performance monitoring"""
        self.logger.info("=== Starting Advanced Performance Monitoring ===")

        # Real-time metrics collection
        metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        metrics_thread.start()
        self.monitoring_threads.append(metrics_thread)

        # System resource monitoring
        system_thread = threading.Thread(
            target=self._system_monitoring_loop,
            name="SystemMonitor",
            daemon=True
        )
        system_thread.start()
        self.monitoring_threads.append(system_thread)

        # Application performance monitoring
        app_thread = threading.Thread(
            target=self._application_monitoring_loop,
            name="ApplicationMonitor",
            daemon=True
        )
        app_thread.start()
        self.monitoring_threads.append(app_thread)

        # AI analysis and prediction
        ai_thread = threading.Thread(
            target=self._ai_analysis_loop,
            name="AIAnalyzer",
            daemon=True
        )
        ai_thread.start()
        self.monitoring_threads.append(ai_thread)

        # Report generation
        report_thread = threading.Thread(
            target=self._report_generation_loop,
            name="ReportGenerator",
            daemon=True
        )
        report_thread.start()
        self.monitoring_threads.append(report_thread)

        # SLA monitoring
        sla_thread = threading.Thread(
            target=self._sla_monitoring_loop,
            name="SLAMonitor",
            daemon=True
        )
        sla_thread.start()
        self.monitoring_threads.append(sla_thread)

        self.logger.info(f"Performance monitoring started with {len(self.monitoring_threads)} threads")

    def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop"""
        self.logger.info("Metrics collection loop started")
        interval = self.config['monitoring']['interval_seconds']

        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()

                # Collect system metrics
                system_metrics = self._collect_system_metrics(current_time)

                # Process and store metrics
                for metric in system_metrics:
                    self._process_metric(metric)
                    self._store_metric(metric)

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(30)

    def _collect_system_metrics(self, timestamp: datetime) -> List[PerformanceMetric]:
        """Collect comprehensive system performance metrics"""
        metrics = []

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]

        metrics.append(self._create_metric(
            "cpu_utilization", cpu_percent, "percent", MetricType.UTILIZATION,
            timestamp, "system", self.config['thresholds']['cpu_utilization']
        ))

        metrics.append(self._create_metric(
            "cpu_load_avg_1m", load_avg[0], "load", MetricType.UTILIZATION,
            timestamp, "system", {"excellent": 0.5, "good": 1.0, "acceptable": 2.0}
        ))

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics.append(self._create_metric(
            "memory_utilization", memory.percent, "percent", MetricType.UTILIZATION,
            timestamp, "system", self.config['thresholds']['memory_utilization']
        ))

        metrics.append(self._create_metric(
            "memory_available_gb", memory.available / (1024**3), "GB", MetricType.CAPACITY,
            timestamp, "system", {"excellent": 8.0, "good": 4.0, "acceptable": 2.0}
        ))

        if swap.total > 0:
            metrics.append(self._create_metric(
                "swap_utilization", swap.percent, "percent", MetricType.UTILIZATION,
                timestamp, "system", {"excellent": 0, "good": 10, "acceptable": 25}
            ))

        # Disk metrics
        for disk in psutil.disk_partitions():
            try:
                disk_usage = psutil.disk_usage(disk.mountpoint)
                disk_name = disk.device.replace('/', '_').replace(':', '').replace('\\', '_')

                metrics.append(self._create_metric(
                    f"disk_utilization_{disk_name}",
                    (disk_usage.used / disk_usage.total) * 100, "percent",
                    MetricType.UTILIZATION, timestamp, "system",
                    self.config['thresholds']['disk_utilization']
                ))

            except (PermissionError, OSError):
                continue

        # Disk I/O metrics
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                if hasattr(self, '_last_disk_io'):
                    time_diff = (timestamp - self._last_disk_timestamp).total_seconds()
                    if time_diff > 0:
                        read_rate = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_diff / (1024*1024)
                        write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_diff / (1024*1024)

                        metrics.append(self._create_metric(
                            "disk_read_rate_mbps", read_rate, "MB/s", MetricType.THROUGHPUT,
                            timestamp, "system", {"excellent": 100, "good": 50, "acceptable": 10}
                        ))

                        metrics.append(self._create_metric(
                            "disk_write_rate_mbps", write_rate, "MB/s", MetricType.THROUGHPUT,
                            timestamp, "system", {"excellent": 100, "good": 50, "acceptable": 10}
                        ))

                self._last_disk_io = disk_io
                self._last_disk_timestamp = timestamp

        except Exception:
            pass

        # Network metrics
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                if hasattr(self, '_last_net_io'):
                    time_diff = (timestamp - self._last_net_timestamp).total_seconds()
                    if time_diff > 0:
                        bytes_sent_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_diff / (1024*1024)
                        bytes_recv_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_diff / (1024*1024)

                        metrics.append(self._create_metric(
                            "network_send_rate_mbps", bytes_sent_rate, "MB/s", MetricType.THROUGHPUT,
                            timestamp, "system", {"excellent": 10, "good": 5, "acceptable": 1}
                        ))

                        metrics.append(self._create_metric(
                            "network_recv_rate_mbps", bytes_recv_rate, "MB/s", MetricType.THROUGHPUT,
                            timestamp, "system", {"excellent": 10, "good": 5, "acceptable": 1}
                        ))

                self._last_net_io = net_io
                self._last_net_timestamp = timestamp

        except Exception:
            pass

        # Process metrics for trading system components
        trading_processes = self._get_trading_processes()
        for proc_name, proc_list in trading_processes.items():
            if proc_list:
                total_cpu = sum(p.cpu_percent() for p in proc_list)
                total_memory = sum(p.memory_percent() for p in proc_list)

                metrics.append(self._create_metric(
                    f"process_{proc_name}_cpu", total_cpu, "percent", MetricType.UTILIZATION,
                    timestamp, "application", {"excellent": 10, "good": 25, "acceptable": 50}
                ))

                metrics.append(self._create_metric(
                    f"process_{proc_name}_memory", total_memory, "percent", MetricType.UTILIZATION,
                    timestamp, "application", {"excellent": 5, "good": 15, "acceptable": 30}
                ))

        return metrics

    def _get_trading_processes(self) -> Dict[str, List]:
        """Get trading system processes"""
        trading_processes = {
            'backend': [],
            'frontend': [],
            'worker': [],
            'streamlit': []
        }

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                proc_info = proc.info
                cmdline = ' '.join(proc_info.get('cmdline', []))

                if 'uvicorn' in cmdline and 'app:app' in cmdline:
                    trading_processes['backend'].append(proc)
                elif 'node' in proc_info.get('name', '') and ('npm run dev' in cmdline or 'vite' in cmdline):
                    trading_processes['frontend'].append(proc)
                elif 'runner.py' in cmdline:
                    trading_processes['worker'].append(proc)
                elif 'streamlit' in proc_info.get('name', ''):
                    trading_processes['streamlit'].append(proc)

        except Exception as e:
            self.logger.debug(f"Error getting trading processes: {e}")

        return trading_processes

    def _create_metric(self, name: str, value: float, unit: str, metric_type: MetricType,
                      timestamp: datetime, source: str, thresholds: Dict[str, float]) -> PerformanceMetric:
        """Create a performance metric with analysis"""

        # Determine performance level
        performance_level = self._determine_performance_level(value, thresholds, metric_type)

        # Calculate trend (simplified)
        trend_direction, trend_strength = self._calculate_trend(name, value)

        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(name, value)

        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metric_type=metric_type,
            timestamp=timestamp,
            source=source,
            excellent_threshold=thresholds.get('excellent', 0),
            good_threshold=thresholds.get('good', 0),
            acceptable_threshold=thresholds.get('acceptable', 0),
            performance_level=performance_level,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            anomaly_score=anomaly_score
        )

        return metric

    def _determine_performance_level(self, value: float, thresholds: Dict[str, float],
                                   metric_type: MetricType) -> PerformanceLevel:
        """Determine performance level based on value and thresholds"""

        # For utilization metrics, lower is better
        if metric_type == MetricType.UTILIZATION:
            if value <= thresholds.get('excellent', 0):
                return PerformanceLevel.EXCELLENT
            elif value <= thresholds.get('good', 0):
                return PerformanceLevel.GOOD
            elif value <= thresholds.get('acceptable', 0):
                return PerformanceLevel.ACCEPTABLE
            elif value <= thresholds.get('acceptable', 0) * 1.5:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL

        # For throughput/capacity metrics, higher is better
        elif metric_type in [MetricType.THROUGHPUT, MetricType.CAPACITY]:
            if value >= thresholds.get('excellent', 0):
                return PerformanceLevel.EXCELLENT
            elif value >= thresholds.get('good', 0):
                return PerformanceLevel.GOOD
            elif value >= thresholds.get('acceptable', 0):
                return PerformanceLevel.ACCEPTABLE
            elif value >= thresholds.get('acceptable', 0) * 0.5:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL

        # For latency/error rate, lower is better
        elif metric_type in [MetricType.LATENCY, MetricType.ERROR_RATE]:
            if value <= thresholds.get('excellent', 0):
                return PerformanceLevel.EXCELLENT
            elif value <= thresholds.get('good', 0):
                return PerformanceLevel.GOOD
            elif value <= thresholds.get('acceptable', 0):
                return PerformanceLevel.ACCEPTABLE
            elif value <= thresholds.get('acceptable', 0) * 2:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL

        return PerformanceLevel.ACCEPTABLE

    def _calculate_trend(self, metric_name: str, current_value: float) -> Tuple[str, float]:
        """Calculate trend direction and strength"""
        if metric_name not in self.metrics_history:
            return "stable", 0.0

        history = list(self.metrics_history[metric_name])
        if len(history) < 5:
            return "stable", 0.0

        # Get recent values
        recent_values = [m.value for m in history[-10:]]

        # Simple trend calculation
        if len(recent_values) >= 3:
            slope = (recent_values[-1] - recent_values[0]) / len(recent_values)

            # Normalize slope to get strength (0-1)
            avg_value = statistics.mean(recent_values)
            if avg_value > 0:
                strength = min(1.0, abs(slope) / (avg_value * 0.1))
            else:
                strength = 0.0

            if slope > 0.01:
                return "up", strength
            elif slope < -0.01:
                return "down", strength
            else:
                return "stable", strength

        return "stable", 0.0

    def _calculate_anomaly_score(self, metric_name: str, current_value: float) -> float:
        """Calculate anomaly score for metric"""
        if metric_name not in self.baseline_metrics:
            return 0.0

        baseline = self.baseline_metrics[metric_name]
        baseline_value = baseline.get('baseline', current_value)
        std_dev = baseline.get('std_dev', 1.0)

        if std_dev == 0:
            return 0.0

        # Z-score based anomaly detection
        z_score = abs(current_value - baseline_value) / std_dev

        # Convert to 0-1 scale
        anomaly_score = min(1.0, z_score / 3.0)  # 3 sigma rule

        return anomaly_score

    def _process_metric(self, metric: PerformanceMetric) -> None:
        """Process metric for real-time analysis"""
        # Add to history
        self.metrics_history[metric.name].append(metric)

        # Update baseline if learning is enabled
        if self.config['ai_analysis']['baseline_learning_enabled']:
            self._update_baseline(metric)

    def _update_baseline(self, metric: PerformanceMetric) -> None:
        """Update baseline for metric"""
        try:
            history = list(self.metrics_history[metric.name])
            if len(history) >= 100:  # Need sufficient data
                values = [m.value for m in history[-100:]]  # Last 100 values

                baseline_value = statistics.mean(values)
                std_deviation = statistics.stdev(values)

                # Update in-memory baseline
                self.baseline_metrics[metric.name] = {
                    'baseline': baseline_value,
                    'std_dev': std_deviation
                }

                # Update database periodically
                if len(history) % 50 == 0:  # Every 50 data points
                    cursor = self.db_connection.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO baselines
                        (metric_name, baseline_value, std_deviation, last_updated, sample_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        metric.name,
                        baseline_value,
                        std_deviation,
                        datetime.now().isoformat(),
                        len(values)
                    ))
                    self.db_connection.commit()

        except Exception as e:
            self.logger.debug(f"Error updating baseline for {metric.name}: {e}")

    def _store_metric(self, metric: PerformanceMetric) -> None:
        """Store metric in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO metrics (
                    timestamp, name, value, unit, metric_type, source,
                    performance_level, trend_direction, trend_strength,
                    anomaly_score, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp.isoformat(),
                metric.name,
                metric.value,
                metric.unit,
                metric.metric_type.value,
                metric.source,
                metric.performance_level.value[0],
                metric.trend_direction,
                metric.trend_strength,
                metric.anomaly_score,
                json.dumps(metric.tags),
                json.dumps(metric.metadata)
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing metric {metric.name}: {e}")

    def _system_monitoring_loop(self) -> None:
        """Detailed system monitoring loop"""
        self.logger.info("System monitoring loop started")
        detailed_interval = self.config['monitoring']['detailed_interval_seconds']

        while not self.shutdown_event.is_set():
            try:
                # Detailed system analysis
                self._analyze_system_performance()

                # Check for bottlenecks
                self._detect_bottlenecks()

                # Update capacity planning
                self._update_capacity_planning()

                time.sleep(detailed_interval)

            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(60)

    def _analyze_system_performance(self) -> None:
        """Perform detailed system performance analysis"""
        try:
            current_time = datetime.now()

            # CPU analysis
            cpu_times = psutil.cpu_times_percent()
            if hasattr(cpu_times, 'user'):
                self.logger.debug(f"CPU breakdown - User: {cpu_times.user}%, System: {cpu_times.system}%, "
                                f"Idle: {cpu_times.idle}%")

            # Memory analysis
            memory = psutil.virtual_memory()
            self.logger.debug(f"Memory details - Total: {memory.total/(1024**3):.1f}GB, "
                            f"Available: {memory.available/(1024**3):.1f}GB, "
                            f"Used: {memory.used/(1024**3):.1f}GB")

            # Process analysis
            top_processes = sorted(
                psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']),
                key=lambda p: p.info['cpu_percent'] or 0,
                reverse=True
            )[:5]

            self.logger.debug("Top CPU processes:")
            for proc in top_processes:
                if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 1.0:
                    self.logger.debug(f"  {proc.info['name']}: {proc.info['cpu_percent']:.1f}% CPU, "
                                    f"{proc.info['memory_percent']:.1f}% Memory")

        except Exception as e:
            self.logger.error(f"Error in system performance analysis: {e}")

    def _detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect system bottlenecks"""
        bottlenecks = []

        try:
            current_time = datetime.now()

            # Check recent metrics for bottleneck patterns
            for metric_name, history in self.metrics_history.items():
                if not history:
                    continue

                recent_metrics = [m for m in history if
                                (current_time - m.timestamp).total_seconds() < 300]  # Last 5 minutes

                if len(recent_metrics) < 5:
                    continue

                # Check for sustained high utilization
                high_util_count = sum(1 for m in recent_metrics
                                    if m.performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL])

                if high_util_count >= len(recent_metrics) * 0.8:  # 80% of samples
                    avg_value = statistics.mean(m.value for m in recent_metrics)

                    bottleneck_type = self._classify_bottleneck(metric_name)
                    if bottleneck_type:
                        bottlenecks.append({
                            'type': bottleneck_type.value,
                            'metric': metric_name,
                            'severity': 'HIGH' if avg_value > 90 else 'MEDIUM',
                            'average_value': avg_value,
                            'duration_minutes': len(recent_metrics) * self.config['monitoring']['interval_seconds'] / 60,
                            'timestamp': current_time.isoformat()
                        })

        except Exception as e:
            self.logger.error(f"Error detecting bottlenecks: {e}")

        return bottlenecks

    def _classify_bottleneck(self, metric_name: str) -> Optional[BottleneckType]:
        """Classify bottleneck type based on metric name"""
        metric_lower = metric_name.lower()

        if 'cpu' in metric_lower or 'load' in metric_lower:
            return BottleneckType.CPU_BOUND
        elif 'memory' in metric_lower or 'swap' in metric_lower:
            return BottleneckType.MEMORY_BOUND
        elif 'disk' in metric_lower:
            return BottleneckType.DISK_BOUND
        elif 'network' in metric_lower:
            return BottleneckType.NETWORK_BOUND
        elif 'gpu' in metric_lower:
            return BottleneckType.GPU_BOUND

        return None

    def _update_capacity_planning(self) -> None:
        """Update capacity planning analysis"""
        try:
            # Analyze growth trends
            for metric_name, history in self.metrics_history.items():
                if len(history) < 100:  # Need sufficient data
                    continue

                # Get trend over different time periods
                recent_values = [m.value for m in list(history)[-100:]]

                if len(recent_values) >= 10:
                    # Simple linear regression for trend
                    X = np.array(range(len(recent_values))).reshape(-1, 1)
                    y = np.array(recent_values)

                    try:
                        model = LinearRegression().fit(X, y)
                        slope = model.coef_[0]

                        # Project future capacity needs
                        if slope > 0:  # Growing trend
                            current_value = recent_values[-1]
                            # Project 30 days ahead
                            future_projection = current_value + (slope * 30 * 24 * 6)  # Assuming 10min intervals

                            # Check if projected value exceeds thresholds
                            if metric_name in self.config['thresholds']:
                                thresholds = self.config['thresholds'][metric_name]
                                if future_projection > thresholds.get('acceptable', 100):
                                    self.logger.warning(
                                        f"Capacity planning alert: {metric_name} projected to reach "
                                        f"{future_projection:.1f} in 30 days"
                                    )

                    except Exception as e:
                        self.logger.debug(f"Error in capacity planning for {metric_name}: {e}")

        except Exception as e:
            self.logger.error(f"Error in capacity planning update: {e}")

    def _application_monitoring_loop(self) -> None:
        """Application-specific monitoring loop"""
        self.logger.info("Application monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                # Monitor trading application performance
                self._monitor_trading_performance()

                # Check API endpoints
                self._monitor_api_performance()

                # Database performance monitoring
                self._monitor_database_performance()

                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in application monitoring loop: {e}")
                time.sleep(60)

    def _monitor_trading_performance(self) -> None:
        """Monitor trading system specific performance"""
        try:
            current_time = datetime.now()

            # Check if services are responding
            services = {
                'backend': 'http://localhost:8000/health',
                'frontend': 'http://localhost:3000',
                'streamlit': 'http://localhost:8501'
            }

            for service_name, url in services.items():
                try:
                    import requests
                    start_time = time.time()
                    response = requests.get(url, timeout=5)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms

                    # Create response time metric
                    metric = self._create_metric(
                        f"{service_name}_response_time", response_time, "ms",
                        MetricType.LATENCY, current_time, "application",
                        self.config['thresholds']['response_time_ms']
                    )

                    self._process_metric(metric)
                    self._store_metric(metric)

                    # Create availability metric
                    availability = 1 if response.status_code < 400 else 0
                    avail_metric = self._create_metric(
                        f"{service_name}_availability", availability, "boolean",
                        MetricType.AVAILABILITY, current_time, "application",
                        {"excellent": 1, "good": 1, "acceptable": 1}
                    )

                    self._process_metric(avail_metric)
                    self._store_metric(avail_metric)

                except Exception as e:
                    self.logger.debug(f"Error monitoring {service_name}: {e}")

                    # Create unavailable metric
                    avail_metric = self._create_metric(
                        f"{service_name}_availability", 0, "boolean",
                        MetricType.AVAILABILITY, current_time, "application",
                        {"excellent": 1, "good": 1, "acceptable": 1}
                    )

                    self._process_metric(avail_metric)
                    self._store_metric(avail_metric)

        except Exception as e:
            self.logger.error(f"Error in trading performance monitoring: {e}")

    def _monitor_api_performance(self) -> None:
        """Monitor API endpoint performance"""
        # This would integrate with API access logs or metrics
        # For now, it's a placeholder for future implementation
        pass

    def _monitor_database_performance(self) -> None:
        """Monitor database performance"""
        try:
            # Monitor SQLite database file size and query performance
            db_path = self.data_dir / "performance_metrics.db"
            if db_path.exists():
                db_size_mb = db_path.stat().st_size / (1024 * 1024)

                current_time = datetime.now()
                size_metric = self._create_metric(
                    "database_size_mb", db_size_mb, "MB", MetricType.CAPACITY,
                    current_time, "database",
                    {"excellent": 100, "good": 500, "acceptable": 1000}
                )

                self._process_metric(size_metric)
                self._store_metric(size_metric)

                # Test query performance
                start_time = time.time()
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM metrics WHERE timestamp > ?",
                             ((datetime.now() - timedelta(hours=1)).isoformat(),))
                cursor.fetchone()
                query_time = (time.time() - start_time) * 1000

                query_metric = self._create_metric(
                    "database_query_time", query_time, "ms", MetricType.LATENCY,
                    current_time, "database",
                    {"excellent": 10, "good": 50, "acceptable": 200}
                )

                self._process_metric(query_metric)
                self._store_metric(query_metric)

        except Exception as e:
            self.logger.error(f"Error in database performance monitoring: {e}")

    def _ai_analysis_loop(self) -> None:
        """AI-powered analysis loop"""
        self.logger.info("AI analysis loop started")

        while not self.shutdown_event.is_set():
            try:
                if self.config['ai_analysis']['anomaly_detection_enabled']:
                    self._perform_anomaly_detection()

                if self.config['ai_analysis']['trend_analysis_enabled']:
                    self._perform_trend_analysis()

                if self.config['ai_analysis']['predictive_analysis_enabled']:
                    self._perform_predictive_analysis()

                time.sleep(300)  # Run AI analysis every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in AI analysis loop: {e}")
                time.sleep(300)

    def _perform_anomaly_detection(self) -> None:
        """Perform ML-based anomaly detection"""
        try:
            # Get recent metrics for analysis
            recent_time = datetime.now() - timedelta(hours=1)
            anomalies = []

            for metric_name, history in self.metrics_history.items():
                recent_metrics = [m for m in history if m.timestamp > recent_time]

                if len(recent_metrics) < 10:
                    continue

                values = np.array([[m.value] for m in recent_metrics])

                # Fit anomaly detector if we have enough data
                if len(values) >= 20:
                    try:
                        anomaly_scores = self.anomaly_detector.fit_predict(values)

                        for i, score in enumerate(anomaly_scores):
                            if score == -1:  # Anomaly detected
                                anomalies.append({
                                    'metric': metric_name,
                                    'value': recent_metrics[i].value,
                                    'timestamp': recent_metrics[i].timestamp.isoformat(),
                                    'anomaly_type': 'statistical_outlier'
                                })

                    except Exception as e:
                        self.logger.debug(f"Error in anomaly detection for {metric_name}: {e}")

            if anomalies:
                self.logger.warning(f"Detected {len(anomalies)} performance anomalies")
                for anomaly in anomalies[:5]:  # Log top 5
                    self.logger.warning(f"Anomaly: {anomaly['metric']} = {anomaly['value']} "
                                      f"at {anomaly['timestamp']}")

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")

    def _perform_trend_analysis(self) -> None:
        """Perform trend analysis on metrics"""
        try:
            trends = {}

            for metric_name, history in self.metrics_history.items():
                if len(history) < 50:
                    continue

                # Get values for trend analysis
                values = [m.value for m in list(history)[-50:]]  # Last 50 values

                if len(values) >= 10:
                    # Calculate trend strength and direction
                    X = np.array(range(len(values))).reshape(-1, 1)
                    y = np.array(values)

                    try:
                        model = self.trend_analyzer.fit(X, y)
                        slope = model.coef_[0]
                        r_squared = model.score(X, y)

                        trends[metric_name] = {
                            'slope': slope,
                            'r_squared': r_squared,
                            'trend_strength': abs(slope) * r_squared,
                            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                        }

                    except Exception as e:
                        self.logger.debug(f"Error in trend analysis for {metric_name}: {e}")

            # Log significant trends
            for metric, trend in trends.items():
                if trend['trend_strength'] > 0.1 and trend['r_squared'] > 0.5:
                    self.logger.info(f"Strong trend detected: {metric} is {trend['direction']} "
                                   f"(strength: {trend['trend_strength']:.2f})")

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")

    def _perform_predictive_analysis(self) -> None:
        """Perform predictive performance analysis"""
        try:
            # Simple prediction based on trends
            predictions = {}

            for metric_name, history in self.metrics_history.items():
                if len(history) < 100:
                    continue

                values = [m.value for m in list(history)[-100:]]

                if len(values) >= 20:
                    # Simple moving average prediction
                    recent_avg = statistics.mean(values[-10:])
                    older_avg = statistics.mean(values[-20:-10])

                    predicted_change = recent_avg - older_avg
                    predicted_value = recent_avg + predicted_change

                    predictions[metric_name] = {
                        'current_value': recent_avg,
                        'predicted_value': predicted_value,
                        'predicted_change': predicted_change,
                        'confidence': 0.7  # Simplified confidence score
                    }

            # Check for problematic predictions
            for metric, prediction in predictions.items():
                if abs(prediction['predicted_change']) > prediction['current_value'] * 0.2:
                    self.logger.warning(f"Significant change predicted for {metric}: "
                                      f"current={prediction['current_value']:.1f}, "
                                      f"predicted={prediction['predicted_value']:.1f}")

        except Exception as e:
            self.logger.error(f"Error in predictive analysis: {e}")

    def _report_generation_loop(self) -> None:
        """Generate periodic performance reports"""
        self.logger.info("Report generation loop started")

        while not self.shutdown_event.is_set():
            try:
                # Generate hourly reports
                report = self._generate_performance_report(duration_minutes=60)
                self._store_performance_report(report)

                if report.performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                    self.logger.warning(f"Performance report shows {report.performance_level.value[0]} "
                                      f"performance (score: {report.overall_score:.2f})")

                time.sleep(3600)  # Generate report every hour

            except Exception as e:
                self.logger.error(f"Error in report generation loop: {e}")
                time.sleep(3600)

    def _generate_performance_report(self, duration_minutes: int = 60) -> PerformanceReport:
        """Generate comprehensive performance report"""
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=duration_minutes)

        # Collect metrics for the period
        period_metrics = []
        for history in self.metrics_history.values():
            period_metrics.extend([m for m in history if m.timestamp >= start_time])

        # Calculate overall performance score
        if period_metrics:
            performance_scores = [m.performance_level.value[1] for m in period_metrics]
            overall_score = statistics.mean(performance_scores)
        else:
            overall_score = 0.5

        # Determine overall performance level
        if overall_score >= 0.8:
            overall_level = PerformanceLevel.EXCELLENT
        elif overall_score >= 0.6:
            overall_level = PerformanceLevel.GOOD
        elif overall_score >= 0.4:
            overall_level = PerformanceLevel.ACCEPTABLE
        elif overall_score >= 0.2:
            overall_level = PerformanceLevel.POOR
        else:
            overall_level = PerformanceLevel.CRITICAL

        # Generate metric summary
        metric_summary = {}
        for metric in period_metrics:
            if metric.name not in metric_summary:
                metric_summary[metric.name] = {
                    'count': 0,
                    'sum': 0,
                    'min': float('inf'),
                    'max': float('-inf')
                }

            summary = metric_summary[metric.name]
            summary['count'] += 1
            summary['sum'] += metric.value
            summary['min'] = min(summary['min'], metric.value)
            summary['max'] = max(summary['max'], metric.value)

        # Calculate averages
        for name, summary in metric_summary.items():
            if summary['count'] > 0:
                summary['average'] = summary['sum'] / summary['count']

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks()

        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(period_metrics, bottlenecks)

        # Check SLA compliance
        sla_compliance, sla_violations = self._check_sla_compliance(period_metrics)

        report = PerformanceReport(
            timestamp=current_time,
            duration_minutes=duration_minutes,
            overall_score=overall_score,
            performance_level=overall_level,
            metrics=period_metrics,
            metric_summary=metric_summary,
            bottlenecks=bottlenecks,
            trends={},  # Simplified for now
            anomalies=[],  # Simplified for now
            optimization_recommendations=recommendations,
            capacity_recommendations=[],
            sla_compliance=sla_compliance,
            sla_violations=sla_violations
        )

        return report

    def _generate_optimization_recommendations(self, metrics: List[PerformanceMetric],
                                             bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze bottlenecks
        bottleneck_types = [b['type'] for b in bottlenecks]

        if BottleneckType.CPU_BOUND.value in bottleneck_types:
            recommendations.append("Consider CPU optimization: reduce computational complexity or scale horizontally")

        if BottleneckType.MEMORY_BOUND.value in bottleneck_types:
            recommendations.append("Memory optimization needed: implement caching or increase available memory")

        if BottleneckType.DISK_BOUND.value in bottleneck_types:
            recommendations.append("Disk I/O bottleneck detected: consider SSD upgrade or I/O optimization")

        if BottleneckType.NETWORK_BOUND.value in bottleneck_types:
            recommendations.append("Network bottleneck detected: optimize network usage or upgrade bandwidth")

        # Analyze performance levels
        poor_metrics = [m for m in metrics if m.performance_level == PerformanceLevel.POOR]
        critical_metrics = [m for m in metrics if m.performance_level == PerformanceLevel.CRITICAL]

        if critical_metrics:
            recommendations.append(f"Critical performance issues detected in {len(critical_metrics)} metrics")

        if poor_metrics:
            recommendations.append(f"Performance degradation detected in {len(poor_metrics)} metrics")

        # General recommendations
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")

        return recommendations

    def _check_sla_compliance(self, metrics: List[PerformanceMetric]) -> Tuple[Dict[str, bool], List[Dict[str, Any]]]:
        """Check SLA compliance"""
        sla_targets = self.config['sla_targets']
        compliance = {}
        violations = []

        # Check availability
        availability_metrics = [m for m in metrics if 'availability' in m.name]
        if availability_metrics:
            avg_availability = statistics.mean(m.value for m in availability_metrics) * 100
            compliance['availability'] = avg_availability >= sla_targets['availability_percent']

            if not compliance['availability']:
                violations.append({
                    'sla': 'availability',
                    'target': sla_targets['availability_percent'],
                    'actual': avg_availability,
                    'severity': 'HIGH'
                })

        # Check response time
        response_time_metrics = [m for m in metrics if 'response_time' in m.name]
        if response_time_metrics:
            avg_response_time = statistics.mean(m.value for m in response_time_metrics)
            compliance['response_time'] = avg_response_time <= sla_targets['response_time_ms']

            if not compliance['response_time']:
                violations.append({
                    'sla': 'response_time',
                    'target': sla_targets['response_time_ms'],
                    'actual': avg_response_time,
                    'severity': 'MEDIUM'
                })

        return compliance, violations

    def _store_performance_report(self, report: PerformanceReport) -> None:
        """Store performance report in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO performance_reports (
                    timestamp, duration_minutes, overall_score, performance_level,
                    bottlenecks, trends, anomalies, recommendations, sla_compliance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.timestamp.isoformat(),
                report.duration_minutes,
                report.overall_score,
                report.performance_level.value[0],
                json.dumps(report.bottlenecks),
                json.dumps(report.trends),
                json.dumps(report.anomalies),
                json.dumps({
                    'optimization': report.optimization_recommendations,
                    'capacity': report.capacity_recommendations
                }),
                json.dumps(report.sla_compliance)
            ))
            self.db_connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing performance report: {e}")

    def _sla_monitoring_loop(self) -> None:
        """SLA monitoring and alerting loop"""
        self.logger.info("SLA monitoring loop started")

        while not self.shutdown_event.is_set():
            try:
                # Check recent performance against SLA targets
                recent_time = datetime.now() - timedelta(minutes=15)  # Last 15 minutes
                recent_metrics = []

                for history in self.metrics_history.values():
                    recent_metrics.extend([m for m in history if m.timestamp > recent_time])

                if recent_metrics:
                    compliance, violations = self._check_sla_compliance(recent_metrics)

                    for violation in violations:
                        self.logger.error(f"SLA VIOLATION: {violation['sla']} - "
                                        f"Target: {violation['target']}, "
                                        f"Actual: {violation['actual']:.2f}")

                time.sleep(300)  # Check SLA every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in SLA monitoring loop: {e}")
                time.sleep(300)

    def get_current_performance_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        current_time = datetime.now()
        recent_time = current_time - timedelta(minutes=5)

        # Get recent metrics
        recent_metrics = []
        for history in self.metrics_history.values():
            recent_metrics.extend([m for m in history if m.timestamp > recent_time])

        if not recent_metrics:
            return {
                'status': 'NO_DATA',
                'timestamp': current_time.isoformat(),
                'metrics_count': 0
            }

        # Calculate current performance
        performance_scores = [m.performance_level.value[1] for m in recent_metrics]
        current_score = statistics.mean(performance_scores)

        # Get current bottlenecks
        bottlenecks = self._detect_bottlenecks()

        # Get metric summary
        metric_counts = defaultdict(int)
        for metric in recent_metrics:
            metric_counts[metric.performance_level.value[0]] += 1

        return {
            'status': 'ACTIVE',
            'timestamp': current_time.isoformat(),
            'overall_score': current_score,
            'metrics_count': len(recent_metrics),
            'performance_distribution': dict(metric_counts),
            'active_bottlenecks': len(bottlenecks),
            'bottleneck_types': list(set(b['type'] for b in bottlenecks)),
            'monitoring_threads': len(self.monitoring_threads),
            'database_size_mb': (self.data_dir / "performance_metrics.db").stat().st_size / (1024*1024)
                               if (self.data_dir / "performance_metrics.db").exists() else 0
        }

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Get metrics for the period
        period_metrics = []
        for history in self.metrics_history.values():
            period_metrics.extend([m for m in history
                                 if start_time <= m.timestamp <= end_time])

        if not period_metrics:
            return {'status': 'NO_DATA', 'period_hours': hours}

        # Calculate summary statistics
        performance_scores = [m.performance_level.value[1] for m in period_metrics]

        summary = {
            'period_hours': hours,
            'total_metrics': len(period_metrics),
            'average_score': statistics.mean(performance_scores),
            'min_score': min(performance_scores),
            'max_score': max(performance_scores),
            'score_std_dev': statistics.stdev(performance_scores) if len(performance_scores) > 1 else 0,
            'performance_distribution': {}
        }

        # Performance level distribution
        level_counts = defaultdict(int)
        for metric in period_metrics:
            level_counts[metric.performance_level.value[0]] += 1

        summary['performance_distribution'] = dict(level_counts)

        # Top metrics by performance issues
        poor_metrics = [m for m in period_metrics
                       if m.performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]]

        metric_issues = defaultdict(int)
        for metric in poor_metrics:
            metric_issues[metric.name] += 1

        summary['top_problematic_metrics'] = dict(sorted(metric_issues.items(),
                                                        key=lambda x: x[1], reverse=True)[:5])

        return summary

    def shutdown(self) -> None:
        """Shutdown the performance monitor"""
        self.logger.info("Shutting down advanced performance monitor...")

        self.shutdown_event.set()

        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=10)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Close database connection
        if self.db_connection:
            self.db_connection.close()

        self.logger.info("Advanced performance monitor shutdown complete")


def main():
    """Main entry point for performance monitoring"""
    performance_monitor = AdvancedPerformanceMonitor()

    try:
        print("="*80)
        print("      ADVANCED PERFORMANCE MONITOR - INVESTMENT GRADE ANALYTICS")
        print("                Professional Quantitative Trading System")
        print("="*80)
        print("Starting comprehensive performance monitoring...")
        print("Features: AI-powered analysis, bottleneck detection, SLA monitoring")
        print("Press Ctrl+C to stop")
        print("="*80)

        # Start performance monitoring
        performance_monitor.start_monitoring()

        # Display real-time status
        while True:
            time.sleep(30)  # Status update every 30 seconds

            status = performance_monitor.get_current_performance_status()
            print(f"\nPerformance Status [{datetime.now().strftime('%H:%M:%S')}]:")
            print(f"  Overall Score: {status.get('overall_score', 0):.3f}")
            print(f"  Metrics Collected: {status.get('metrics_count', 0)}")
            print(f"  Active Bottlenecks: {status.get('active_bottlenecks', 0)}")
            print(f"  Performance Distribution: {status.get('performance_distribution', {})}")

            if status.get('bottleneck_types'):
                print(f"  Bottleneck Types: {', '.join(status['bottleneck_types'])}")

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in performance monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        performance_monitor.shutdown()

if __name__ == "__main__":
    sys.exit(main())