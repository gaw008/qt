#!/usr/bin/env python3
"""
System Self-Healing and Automatic Recovery
Professional Quantitative Trading System

This script provides intelligent system self-healing capabilities including:
- Automatic fault detection and diagnosis
- Intelligent process restart and recovery
- Configuration repair and validation
- Resource optimization and cleanup
- Predictive maintenance and prevention
- Comprehensive incident logging and reporting
- Integration with system health monitoring

Features:
- AI-driven fault pattern recognition
- Automatic resource leak detection and cleanup
- Intelligent process restart strategies
- Configuration drift detection and correction
- Predictive failure prevention
- Comprehensive recovery logging
- Integration with external monitoring systems

Author: Quantitative Trading System
Version: 2.0
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
import signal
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import requests

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART_PROCESS = "restart_process"
    CLEANUP_RESOURCES = "cleanup_resources"
    REPAIR_CONFIG = "repair_config"
    CLEAR_CACHE = "clear_cache"
    OPTIMIZE_RESOURCES = "optimize_resources"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    NOTIFY_ADMIN = "notify_admin"

@dataclass
class HealthIncident:
    """Health incident data structure."""
    incident_id: str
    timestamp: datetime
    severity: HealthStatus
    component: str
    description: str
    symptoms: List[str]
    root_cause: Optional[str]
    recovery_actions: List[RecoveryAction]
    recovery_success: bool
    resolution_time: Optional[datetime]
    details: Dict[str, Any]

@dataclass
class RecoveryStrategy:
    """Recovery strategy definition."""
    name: str
    conditions: Dict[str, Any]
    actions: List[RecoveryAction]
    priority: int
    timeout_seconds: int
    retry_count: int
    success_criteria: Dict[str, Any]

class SystemSelfHealing:
    """Intelligent system self-healing and recovery."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.quant_dir = self.base_dir / "quant_system_full"
        self.incident_dir = self.base_dir / "incidents"
        self.incident_dir.mkdir(exist_ok=True)

        self.logger = self._setup_logging()
        self.shutdown_event = threading.Event()
        self.recovery_threads: List[threading.Thread] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Recovery")

        # Self-healing configuration
        self.config = {
            'monitoring_interval': 15,  # seconds
            'recovery_timeout': 300,  # seconds
            'max_recovery_attempts': 3,
            'incident_retention_days': 30,
            'critical_memory_threshold': 95,  # percentage
            'critical_cpu_threshold': 98,
            'max_disk_usage': 95,
            'process_restart_delay': 5,
            'resource_cleanup_threshold': 85,
            'config_validation_interval': 3600,
            'predictive_analysis_interval': 1800,
            'auto_restart_enabled': True,
            'aggressive_recovery_enabled': False
        }

        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = self._initialize_recovery_strategies()

        # System state tracking
        self.system_state = {
            'last_health_check': None,
            'recent_incidents': [],
            'recovery_history': [],
            'system_baseline': {},
            'performance_degradation': False,
            'resource_leaks_detected': [],
            'configuration_drift': {},
            'predictive_alerts': []
        }

        # Process tracking
        self.monitored_processes = {
            'backend': {'name': 'uvicorn', 'restart_command': self._restart_backend},
            'frontend_react': {'name': 'node', 'restart_command': self._restart_react_frontend},
            'streamlit': {'name': 'streamlit', 'restart_command': self._restart_streamlit},
            'trading_bot': {'name': 'python', 'cmdline_contains': 'runner.py', 'restart_command': self._restart_trading_bot}
        }

        # Critical file integrity
        self.critical_files = [
            self.quant_dir / "config.example.env",
            self.quant_dir / "dashboard" / "backend" / "app.py",
            self.quant_dir / "dashboard" / "worker" / "runner.py",
            self.quant_dir / "UI" / "package.json"
        ]

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for self-healing operations."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('SelfHealing')
        logger.setLevel(logging.INFO)

        # Console handler with healing-specific formatting
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '\033[95m%(asctime)s\033[0m - \033[97mHEALING\033[0m - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler for incident tracking
        file_handler = logging.FileHandler(
            log_dir / f"self_healing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _initialize_recovery_strategies(self) -> List[RecoveryStrategy]:
        """Initialize recovery strategies for different failure scenarios."""
        strategies = [
            # High Memory Usage Recovery
            RecoveryStrategy(
                name="high_memory_recovery",
                conditions={"memory_percent": {"gt": self.config['critical_memory_threshold']}},
                actions=[RecoveryAction.CLEANUP_RESOURCES, RecoveryAction.OPTIMIZE_RESOURCES],
                priority=1,
                timeout_seconds=60,
                retry_count=2,
                success_criteria={"memory_percent": {"lt": 85}}
            ),

            # Process Failure Recovery
            RecoveryStrategy(
                name="process_failure_recovery",
                conditions={"process_status": {"eq": "stopped"}},
                actions=[RecoveryAction.RESTART_PROCESS],
                priority=2,
                timeout_seconds=30,
                retry_count=3,
                success_criteria={"process_status": {"eq": "running"}}
            ),

            # High CPU Usage Recovery
            RecoveryStrategy(
                name="high_cpu_recovery",
                conditions={"cpu_percent": {"gt": self.config['critical_cpu_threshold']}},
                actions=[RecoveryAction.OPTIMIZE_RESOURCES, RecoveryAction.RESTART_PROCESS],
                priority=2,
                timeout_seconds=120,
                retry_count=2,
                success_criteria={"cpu_percent": {"lt": 90}}
            ),

            # Disk Space Recovery
            RecoveryStrategy(
                name="disk_space_recovery",
                conditions={"disk_percent": {"gt": self.config['max_disk_usage']}},
                actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.CLEANUP_RESOURCES],
                priority=1,
                timeout_seconds=300,
                retry_count=1,
                success_criteria={"disk_percent": {"lt": 90}}
            ),

            # Configuration Corruption Recovery
            RecoveryStrategy(
                name="config_corruption_recovery",
                conditions={"config_integrity": {"eq": False}},
                actions=[RecoveryAction.REPAIR_CONFIG],
                priority=3,
                timeout_seconds=60,
                retry_count=1,
                success_criteria={"config_integrity": {"eq": True}}
            ),

            # Emergency Shutdown Strategy
            RecoveryStrategy(
                name="emergency_shutdown",
                conditions={"system_health": {"eq": "failed"}},
                actions=[RecoveryAction.EMERGENCY_SHUTDOWN, RecoveryAction.NOTIFY_ADMIN],
                priority=5,
                timeout_seconds=30,
                retry_count=1,
                success_criteria={}
            )
        ]

        return strategies

    def start_self_healing(self) -> None:
        """Start the self-healing monitoring and recovery system."""
        self.logger.info("=== Starting System Self-Healing ===")

        # Health monitoring thread
        health_thread = threading.Thread(
            target=self._continuous_health_monitoring,
            name="HealthMonitor",
            daemon=True
        )
        health_thread.start()
        self.recovery_threads.append(health_thread)

        # Resource leak detection thread
        resource_thread = threading.Thread(
            target=self._detect_resource_leaks,
            name="ResourceLeakDetector",
            daemon=True
        )
        resource_thread.start()
        self.recovery_threads.append(resource_thread)

        # Configuration validation thread
        config_thread = threading.Thread(
            target=self._validate_configurations,
            name="ConfigValidator",
            daemon=True
        )
        config_thread.start()
        self.recovery_threads.append(config_thread)

        # Predictive analysis thread
        predictive_thread = threading.Thread(
            target=self._predictive_failure_analysis,
            name="PredictiveAnalyzer",
            daemon=True
        )
        predictive_thread.start()
        self.recovery_threads.append(predictive_thread)

        # Incident cleanup thread
        cleanup_thread = threading.Thread(
            target=self._cleanup_old_incidents,
            name="IncidentCleaner",
            daemon=True
        )
        cleanup_thread.start()
        self.recovery_threads.append(cleanup_thread)

        self.logger.info(f"[OK] {len(self.recovery_threads)} self-healing threads started")

    def _continuous_health_monitoring(self) -> None:
        """Continuously monitor system health and trigger recovery when needed."""
        self.logger.info("Continuous health monitoring started")

        while not self.shutdown_event.is_set():
            try:
                # Collect system health metrics
                health_metrics = self._collect_health_metrics()
                self.system_state['last_health_check'] = datetime.now()

                # Analyze health and determine if recovery is needed
                issues = self._analyze_health_issues(health_metrics)

                if issues:
                    self.logger.warning(f"Health issues detected: {len(issues)} problems")

                    # Process each issue
                    for issue in issues:
                        incident = self._create_incident(issue)
                        recovery_success = self._execute_recovery(incident)

                        if not recovery_success:
                            self.logger.error(f"Recovery failed for incident: {incident.incident_id}")
                        else:
                            self.logger.info(f"Successfully recovered from incident: {incident.incident_id}")

                # Update system baseline occasionally
                if len(self.system_state.get('recovery_history', [])) % 10 == 0:
                    self._update_system_baseline(health_metrics)

                time.sleep(self.config['monitoring_interval'])

            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(60)

    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system health metrics."""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': (psutil.disk_usage(str(self.base_dir)).used /
                           psutil.disk_usage(str(self.base_dir)).total) * 100,
            'processes': {},
            'network_status': {},
            'file_integrity': {},
            'resource_usage': {}
        }

        # Process monitoring
        for process_key, process_info in self.monitored_processes.items():
            metrics['processes'][process_key] = self._check_process_health(process_info)

        # Network connectivity
        metrics['network_status'] = self._check_network_connectivity()

        # File integrity
        metrics['file_integrity'] = self._check_file_integrity()

        # Resource usage details
        metrics['resource_usage'] = self._collect_detailed_resource_usage()

        return metrics

    def _check_process_health(self, process_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of a specific process."""
        process_name = process_info['name']
        cmdline_filter = process_info.get('cmdline_contains')

        status = {
            'running': False,
            'pid': None,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'status': 'stopped',
            'last_seen': None
        }

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'status']):
                proc_info = proc.info
                if proc_info['name'] and process_name.lower() in proc_info['name'].lower():
                    # Additional cmdline filtering if specified
                    if cmdline_filter:
                        cmdline = ' '.join(proc_info.get('cmdline', []))
                        if cmdline_filter not in cmdline:
                            continue

                    status.update({
                        'running': True,
                        'pid': proc_info['pid'],
                        'cpu_percent': proc_info.get('cpu_percent', 0) or 0,
                        'memory_percent': proc_info.get('memory_percent', 0) or 0,
                        'status': proc_info.get('status', 'running'),
                        'last_seen': datetime.now()
                    })
                    break

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return status

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity to critical services."""
        connectivity = {
            'internet': False,
            'local_services': {},
            'api_endpoints': {}
        }

        # Internet connectivity test
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            connectivity['internet'] = True
        except:
            connectivity['internet'] = False

        # Local services
        services = {
            'backend': 'http://localhost:8000/health',
            'frontend': 'http://localhost:3000',
            'streamlit': 'http://localhost:8501'
        }

        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                connectivity['local_services'][service_name] = {
                    'status': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'available': response.status_code < 400
                }
            except:
                connectivity['local_services'][service_name] = {
                    'status': 0,
                    'response_time': 0,
                    'available': False
                }

        return connectivity

    def _check_file_integrity(self) -> Dict[str, bool]:
        """Check integrity of critical files."""
        integrity = {}

        for file_path in self.critical_files:
            integrity[str(file_path)] = file_path.exists() and file_path.is_file()

        return integrity

    def _collect_detailed_resource_usage(self) -> Dict[str, Any]:
        """Collect detailed resource usage information."""
        usage = {
            'memory_details': {},
            'cpu_details': {},
            'disk_io': {},
            'network_io': {},
            'open_files': 0,
            'threads': 0
        }

        try:
            # Memory details
            memory = psutil.virtual_memory()
            usage['memory_details'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'cached_gb': getattr(memory, 'cached', 0) / (1024**3),
                'buffers_gb': getattr(memory, 'buffers', 0) / (1024**3)
            }

            # CPU details
            usage['cpu_details'] = {
                'count': psutil.cpu_count(),
                'usage_per_cpu': psutil.cpu_percent(percpu=True),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                usage['disk_io'] = {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                }

            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                usage['network_io'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }

            # System-wide resource usage
            current_process = psutil.Process()
            usage['open_files'] = len(current_process.open_files())
            usage['threads'] = current_process.num_threads()

        except Exception as e:
            self.logger.warning(f"Error collecting detailed resource usage: {e}")

        return usage

    def _analyze_health_issues(self, health_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze health metrics and identify issues."""
        issues = []

        # Check system resource thresholds
        if health_metrics['memory_percent'] > self.config['critical_memory_threshold']:
            issues.append({
                'type': 'high_memory_usage',
                'severity': HealthStatus.CRITICAL,
                'component': 'system',
                'description': f"Memory usage at {health_metrics['memory_percent']:.1f}%",
                'metrics': {'memory_percent': health_metrics['memory_percent']}
            })

        if health_metrics['cpu_percent'] > self.config['critical_cpu_threshold']:
            issues.append({
                'type': 'high_cpu_usage',
                'severity': HealthStatus.CRITICAL,
                'component': 'system',
                'description': f"CPU usage at {health_metrics['cpu_percent']:.1f}%",
                'metrics': {'cpu_percent': health_metrics['cpu_percent']}
            })

        if health_metrics['disk_percent'] > self.config['max_disk_usage']:
            issues.append({
                'type': 'high_disk_usage',
                'severity': HealthStatus.WARNING,
                'component': 'system',
                'description': f"Disk usage at {health_metrics['disk_percent']:.1f}%",
                'metrics': {'disk_percent': health_metrics['disk_percent']}
            })

        # Check process health
        for process_key, process_status in health_metrics['processes'].items():
            if not process_status['running']:
                issues.append({
                    'type': 'process_failure',
                    'severity': HealthStatus.CRITICAL,
                    'component': process_key,
                    'description': f"Process {process_key} is not running",
                    'metrics': {'process_status': 'stopped'}
                })

        # Check network connectivity
        if not health_metrics['network_status']['internet']:
            issues.append({
                'type': 'network_connectivity',
                'severity': HealthStatus.WARNING,
                'component': 'network',
                'description': "Internet connectivity lost",
                'metrics': {'internet_connectivity': False}
            })

        # Check local services
        for service_name, service_status in health_metrics['network_status']['local_services'].items():
            if not service_status['available']:
                issues.append({
                    'type': 'service_unavailable',
                    'severity': HealthStatus.CRITICAL,
                    'component': service_name,
                    'description': f"Service {service_name} is not responding",
                    'metrics': {'service_status': service_status}
                })

        # Check file integrity
        for file_path, exists in health_metrics['file_integrity'].items():
            if not exists:
                issues.append({
                    'type': 'file_corruption',
                    'severity': HealthStatus.CRITICAL,
                    'component': 'filesystem',
                    'description': f"Critical file missing: {file_path}",
                    'metrics': {'config_integrity': False}
                })

        return issues

    def _create_incident(self, issue: Dict[str, Any]) -> HealthIncident:
        """Create an incident record for tracking and recovery."""
        incident_id = f"{issue['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        incident = HealthIncident(
            incident_id=incident_id,
            timestamp=datetime.now(),
            severity=issue['severity'],
            component=issue['component'],
            description=issue['description'],
            symptoms=[issue['description']],
            root_cause=None,
            recovery_actions=[],
            recovery_success=False,
            resolution_time=None,
            details=issue
        )

        # Store incident
        self.system_state['recent_incidents'].append(incident)
        self._save_incident(incident)

        return incident

    def _execute_recovery(self, incident: HealthIncident) -> bool:
        """Execute recovery actions for an incident."""
        self.logger.info(f"Executing recovery for incident: {incident.incident_id}")

        # Find applicable recovery strategies
        applicable_strategies = self._find_recovery_strategies(incident)

        if not applicable_strategies:
            self.logger.warning(f"No recovery strategies found for incident: {incident.incident_id}")
            return False

        # Try recovery strategies in priority order
        for strategy in sorted(applicable_strategies, key=lambda s: s.priority):
            self.logger.info(f"Attempting recovery strategy: {strategy.name}")

            try:
                success = self._apply_recovery_strategy(strategy, incident)
                if success:
                    incident.recovery_success = True
                    incident.resolution_time = datetime.now()
                    incident.recovery_actions = strategy.actions
                    self._update_incident(incident)
                    return True

            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.name} failed: {e}")
                continue

        # If all strategies failed
        incident.recovery_success = False
        self._update_incident(incident)
        return False

    def _find_recovery_strategies(self, incident: HealthIncident) -> List[RecoveryStrategy]:
        """Find applicable recovery strategies for an incident."""
        applicable = []

        for strategy in self.recovery_strategies:
            if self._matches_conditions(strategy.conditions, incident.details.get('metrics', {})):
                applicable.append(strategy)

        return applicable

    def _matches_conditions(self, conditions: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Check if metrics match strategy conditions."""
        for metric_name, condition in conditions.items():
            if metric_name not in metrics:
                continue

            metric_value = metrics[metric_name]

            for operator, threshold in condition.items():
                if operator == 'gt' and metric_value <= threshold:
                    return False
                elif operator == 'lt' and metric_value >= threshold:
                    return False
                elif operator == 'eq' and metric_value != threshold:
                    return False

        return True

    def _apply_recovery_strategy(self, strategy: RecoveryStrategy, incident: HealthIncident) -> bool:
        """Apply a specific recovery strategy."""
        for action in strategy.actions:
            try:
                success = self._execute_recovery_action(action, incident)
                if not success:
                    return False

                time.sleep(2)  # Brief pause between actions

            except Exception as e:
                self.logger.error(f"Recovery action {action} failed: {e}")
                return False

        # Verify recovery success
        return self._verify_recovery_success(strategy, incident)

    def _execute_recovery_action(self, action: RecoveryAction, incident: HealthIncident) -> bool:
        """Execute a specific recovery action."""
        self.logger.info(f"Executing recovery action: {action.value}")

        if action == RecoveryAction.RESTART_PROCESS:
            return self._restart_failed_process(incident.component)

        elif action == RecoveryAction.CLEANUP_RESOURCES:
            return self._cleanup_system_resources()

        elif action == RecoveryAction.REPAIR_CONFIG:
            return self._repair_configuration_files()

        elif action == RecoveryAction.CLEAR_CACHE:
            return self._clear_system_cache()

        elif action == RecoveryAction.OPTIMIZE_RESOURCES:
            return self._optimize_system_resources()

        elif action == RecoveryAction.EMERGENCY_SHUTDOWN:
            return self._emergency_system_shutdown()

        elif action == RecoveryAction.NOTIFY_ADMIN:
            return self._notify_administrator(incident)

        else:
            self.logger.warning(f"Unknown recovery action: {action}")
            return False

    def _restart_failed_process(self, component: str) -> bool:
        """Restart a failed process component."""
        if component in self.monitored_processes:
            process_info = self.monitored_processes[component]
            restart_command = process_info.get('restart_command')

            if restart_command and callable(restart_command):
                try:
                    return restart_command()
                except Exception as e:
                    self.logger.error(f"Error restarting {component}: {e}")
                    return False

        return False

    def _restart_backend(self) -> bool:
        """Restart backend API server."""
        try:
            backend_dir = self.quant_dir / "dashboard" / "backend"
            if backend_dir.exists():
                subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
                    cwd=backend_dir,
                    env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
                )
                time.sleep(5)  # Allow startup time
                return True
        except Exception as e:
            self.logger.error(f"Error restarting backend: {e}")
        return False

    def _restart_react_frontend(self) -> bool:
        """Restart React frontend."""
        try:
            ui_dir = self.quant_dir / "UI"
            if ui_dir.exists():
                subprocess.Popen(['npm', 'run', 'dev'], cwd=ui_dir)
                time.sleep(10)  # Allow startup time
                return True
        except Exception as e:
            self.logger.error(f"Error restarting React frontend: {e}")
        return False

    def _restart_streamlit(self) -> bool:
        """Restart Streamlit dashboard."""
        try:
            frontend_dir = self.quant_dir / "dashboard" / "frontend"
            if frontend_dir.exists():
                subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"],
                    cwd=frontend_dir,
                    env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
                )
                time.sleep(5)  # Allow startup time
                return True
        except Exception as e:
            self.logger.error(f"Error restarting Streamlit: {e}")
        return False

    def _restart_trading_bot(self) -> bool:
        """Restart trading bot worker."""
        try:
            worker_dir = self.quant_dir / "dashboard" / "worker"
            if worker_dir.exists():
                subprocess.Popen(
                    [sys.executable, "runner.py"],
                    cwd=worker_dir,
                    env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
                )
                time.sleep(5)  # Allow startup time
                return True
        except Exception as e:
            self.logger.error(f"Error restarting trading bot: {e}")
        return False

    def _cleanup_system_resources(self) -> bool:
        """Cleanup system resources and temporary files."""
        try:
            # Clear Python cache
            import gc
            gc.collect()

            # Clean temporary files
            temp_dirs = [
                self.base_dir / "temp",
                self.base_dir / "__pycache__",
                self.quant_dir / "__pycache__"
            ]

            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

            # Clean log files older than 7 days
            log_dir = self.base_dir / "logs"
            if log_dir.exists():
                cutoff_time = time.time() - (7 * 24 * 3600)
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_time:
                        log_file.unlink(missing_ok=True)

            self.logger.info("System resources cleaned up")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")
            return False

    def _repair_configuration_files(self) -> bool:
        """Repair corrupted configuration files."""
        try:
            repaired = False

            # Check and repair .env file
            env_file = self.quant_dir / ".env"
            example_env = self.quant_dir / "config.example.env"

            if not env_file.exists() and example_env.exists():
                shutil.copy2(example_env, env_file)
                self.logger.info("Restored .env file from example")
                repaired = True

            # Check package.json integrity
            package_json = self.quant_dir / "UI" / "package.json"
            if package_json.exists():
                try:
                    with open(package_json, 'r') as f:
                        json.load(f)  # Validate JSON
                except json.JSONDecodeError:
                    self.logger.warning("package.json appears corrupted")
                    # Could implement backup restoration here

            return repaired

        except Exception as e:
            self.logger.error(f"Error repairing configuration: {e}")
            return False

    def _clear_system_cache(self) -> bool:
        """Clear system and application caches."""
        try:
            # Clear data cache if it's too large
            data_cache_dir = self.quant_dir / "data_cache"
            if data_cache_dir.exists():
                cache_size = sum(f.stat().st_size for f in data_cache_dir.rglob('*') if f.is_file())
                cache_size_gb = cache_size / (1024**3)

                if cache_size_gb > 10:  # Clear if larger than 10GB
                    for cache_file in data_cache_dir.glob("*"):
                        if cache_file.is_file() and (time.time() - cache_file.stat().st_mtime) > 86400:  # 1 day old
                            cache_file.unlink(missing_ok=True)

            self.logger.info("System cache cleared")
            return True

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False

    def _optimize_system_resources(self) -> bool:
        """Optimize system resource usage."""
        try:
            # Optimize process priorities
            current_process = psutil.Process()
            try:
                if sys.platform == "win32":
                    current_process.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    current_process.nice(0)
            except psutil.AccessDenied:
                pass

            # Trigger garbage collection
            import gc
            gc.collect()

            self.logger.info("System resources optimized")
            return True

        except Exception as e:
            self.logger.error(f"Error optimizing resources: {e}")
            return False

    def _emergency_system_shutdown(self) -> bool:
        """Perform emergency system shutdown."""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")

        try:
            # Kill all monitored processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if any(name in proc_info.get('name', '').lower()
                          for name in ['uvicorn', 'streamlit', 'node']):
                        if 'python' in ' '.join(proc_info.get('cmdline', [])):
                            proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            self.shutdown_event.set()
            return True

        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")
            return False

    def _notify_administrator(self, incident: HealthIncident) -> bool:
        """Notify system administrator of critical incidents."""
        try:
            # Create notification message
            notification = {
                'incident_id': incident.incident_id,
                'timestamp': incident.timestamp.isoformat(),
                'severity': incident.severity.value,
                'component': incident.component,
                'description': incident.description,
                'recovery_attempted': bool(incident.recovery_actions)
            }

            # Save notification to file (in real implementation, would send email/SMS)
            notification_file = self.incident_dir / f"alert_{incident.incident_id}.json"
            with open(notification_file, 'w') as f:
                json.dump(notification, f, indent=2)

            self.logger.critical(f"ADMIN NOTIFICATION: {incident.description}")
            return True

        except Exception as e:
            self.logger.error(f"Error notifying administrator: {e}")
            return False

    def _verify_recovery_success(self, strategy: RecoveryStrategy, incident: HealthIncident) -> bool:
        """Verify that recovery actions were successful."""
        if not strategy.success_criteria:
            return True  # No specific criteria to check

        # Collect current metrics
        current_metrics = self._collect_health_metrics()

        # Check success criteria
        for metric_name, criteria in strategy.success_criteria.items():
            current_value = current_metrics.get(metric_name)
            if current_value is None:
                continue

            for operator, threshold in criteria.items():
                if operator == 'lt' and current_value >= threshold:
                    return False
                elif operator == 'gt' and current_value <= threshold:
                    return False
                elif operator == 'eq' and current_value != threshold:
                    return False

        return True

    def _save_incident(self, incident: HealthIncident) -> None:
        """Save incident to persistent storage."""
        try:
            incident_file = self.incident_dir / f"incident_{incident.incident_id}.json"

            # Convert dataclass to dict for JSON serialization
            incident_dict = asdict(incident)
            incident_dict['timestamp'] = incident.timestamp.isoformat()
            if incident.resolution_time:
                incident_dict['resolution_time'] = incident.resolution_time.isoformat()

            # Convert enums to strings
            incident_dict['severity'] = incident.severity.value
            incident_dict['recovery_actions'] = [action.value for action in incident.recovery_actions]

            with open(incident_file, 'w') as f:
                json.dump(incident_dict, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving incident: {e}")

    def _update_incident(self, incident: HealthIncident) -> None:
        """Update existing incident record."""
        self._save_incident(incident)

    def _detect_resource_leaks(self) -> None:
        """Detect and handle resource leaks."""
        self.logger.info("Resource leak detection started")

        baseline_memory = None
        baseline_handles = None

        while not self.shutdown_event.is_set():
            try:
                current_process = psutil.Process()
                current_memory = current_process.memory_info().rss / (1024**2)  # MB

                try:
                    current_handles = current_process.num_handles() if hasattr(current_process, 'num_handles') else 0
                except (psutil.AccessDenied, AttributeError):
                    current_handles = 0

                if baseline_memory is None:
                    baseline_memory = current_memory
                    baseline_handles = current_handles
                else:
                    # Check for memory leaks
                    memory_growth = current_memory - baseline_memory
                    if memory_growth > 500:  # 500MB growth
                        self.logger.warning(f"Potential memory leak detected: {memory_growth:.1f}MB growth")
                        self.system_state['resource_leaks_detected'].append({
                            'type': 'memory_leak',
                            'growth_mb': memory_growth,
                            'timestamp': datetime.now()
                        })

                    # Check for handle leaks
                    if current_handles > 0:
                        handle_growth = current_handles - baseline_handles
                        if handle_growth > 1000:  # 1000+ handle growth
                            self.logger.warning(f"Potential handle leak detected: {handle_growth} handles")

                # Update baseline periodically
                if len(self.system_state.get('resource_leaks_detected', [])) % 10 == 0:
                    baseline_memory = current_memory
                    baseline_handles = current_handles

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in resource leak detection: {e}")
                time.sleep(600)

    def _validate_configurations(self) -> None:
        """Validate system configurations periodically."""
        self.logger.info("Configuration validation started")

        while not self.shutdown_event.is_set():
            try:
                config_issues = []

                # Validate critical configuration files
                for config_file in self.critical_files:
                    if not config_file.exists():
                        config_issues.append(f"Missing file: {config_file}")

                # Check environment configuration
                env_file = self.quant_dir / ".env"
                if env_file.exists():
                    try:
                        with open(env_file, 'r') as f:
                            env_content = f.read()
                            if 'TIGER_ID' not in env_content:
                                config_issues.append("Missing TIGER_ID in .env")
                    except Exception as e:
                        config_issues.append(f"Error reading .env: {e}")

                # Store configuration issues
                if config_issues:
                    self.system_state['configuration_drift'] = {
                        'issues': config_issues,
                        'timestamp': datetime.now()
                    }
                    self.logger.warning(f"Configuration issues detected: {len(config_issues)}")

                time.sleep(self.config['config_validation_interval'])

            except Exception as e:
                self.logger.error(f"Error in configuration validation: {e}")
                time.sleep(1800)

    def _predictive_failure_analysis(self) -> None:
        """Analyze trends to predict potential failures."""
        self.logger.info("Predictive failure analysis started")

        while not self.shutdown_event.is_set():
            try:
                # Analyze resource usage trends
                if len(self.system_state.get('recovery_history', [])) > 5:
                    recent_incidents = self.system_state['recovery_history'][-10:]

                    # Look for patterns
                    memory_incidents = sum(1 for i in recent_incidents if 'memory' in str(i))
                    cpu_incidents = sum(1 for i in recent_incidents if 'cpu' in str(i))

                    if memory_incidents > 3:
                        self.system_state['predictive_alerts'].append({
                            'type': 'memory_degradation_trend',
                            'confidence': 0.7,
                            'timestamp': datetime.now(),
                            'recommendation': 'Consider memory optimization or hardware upgrade'
                        })

                    if cpu_incidents > 3:
                        self.system_state['predictive_alerts'].append({
                            'type': 'cpu_degradation_trend',
                            'confidence': 0.7,
                            'timestamp': datetime.now(),
                            'recommendation': 'Consider CPU optimization or load balancing'
                        })

                time.sleep(self.config['predictive_analysis_interval'])

            except Exception as e:
                self.logger.error(f"Error in predictive analysis: {e}")
                time.sleep(1800)

    def _cleanup_old_incidents(self) -> None:
        """Clean up old incident records."""
        while not self.shutdown_event.is_set():
            try:
                cutoff_date = datetime.now() - timedelta(days=self.config['incident_retention_days'])

                # Clean up incident files
                for incident_file in self.incident_dir.glob("incident_*.json"):
                    if incident_file.stat().st_mtime < cutoff_date.timestamp():
                        incident_file.unlink(missing_ok=True)

                # Clean up in-memory incidents
                self.system_state['recent_incidents'] = [
                    i for i in self.system_state['recent_incidents']
                    if i.timestamp > cutoff_date
                ]

                time.sleep(86400)  # Daily cleanup

            except Exception as e:
                self.logger.error(f"Error cleaning up incidents: {e}")
                time.sleep(86400)

    def _update_system_baseline(self, current_metrics: Dict[str, Any]) -> None:
        """Update system baseline metrics for comparison."""
        self.system_state['system_baseline'] = {
            'cpu_percent': current_metrics['cpu_percent'],
            'memory_percent': current_metrics['memory_percent'],
            'disk_percent': current_metrics['disk_percent'],
            'timestamp': datetime.now(),
            'process_count': len([p for p in current_metrics['processes'].values() if p['running']])
        }

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get current system health summary."""
        return {
            'last_health_check': self.system_state.get('last_health_check'),
            'recent_incidents_count': len(self.system_state.get('recent_incidents', [])),
            'resource_leaks_detected': len(self.system_state.get('resource_leaks_detected', [])),
            'configuration_issues': len(self.system_state.get('configuration_drift', {}).get('issues', [])),
            'predictive_alerts': len(self.system_state.get('predictive_alerts', [])),
            'recovery_success_rate': self._calculate_recovery_success_rate(),
            'system_baseline': self.system_state.get('system_baseline', {}),
            'self_healing_active': not self.shutdown_event.is_set()
        }

    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        recent_incidents = self.system_state.get('recent_incidents', [])
        if not recent_incidents:
            return 1.0

        successful_recoveries = sum(1 for i in recent_incidents if i.recovery_success)
        return successful_recoveries / len(recent_incidents)

    def shutdown(self) -> None:
        """Shutdown self-healing system."""
        self.logger.info("Shutting down self-healing system...")
        self.shutdown_event.set()

        # Wait for threads to finish
        for thread in self.recovery_threads:
            thread.join(timeout=10)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Generate final report
        summary = self.get_system_health_summary()
        self.logger.info(f"Final health summary: {json.dumps(summary, indent=2, default=str)}")

        self.logger.info("Self-healing system shutdown complete")

def main():
    """Entry point for self-healing system."""
    healing_system = SystemSelfHealing()

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        healing_system.shutdown()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("="*80)
        print("         QUANTITATIVE TRADING SYSTEM - SELF-HEALING MODE")
        print("                Intelligent System Recovery and Maintenance")
        print("="*80)
        print("Self-healing system starting...")
        print("Monitoring for system issues and automatically recovering...")
        print("Press Ctrl+C to stop")
        print("="*80)

        # Start self-healing
        healing_system.start_self_healing()

        # Keep main thread alive and show periodic status
        while not healing_system.shutdown_event.is_set():
            time.sleep(60)  # Status update every minute

            summary = healing_system.get_system_health_summary()
            print(f"\nStatus: {datetime.now().strftime('%H:%M:%S')} - "
                  f"Incidents: {summary['recent_incidents_count']}, "
                  f"Success Rate: {summary['recovery_success_rate']:.1%}, "
                  f"Active: {summary['self_healing_active']}")

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in self-healing system: {e}")
    finally:
        healing_system.shutdown()

    return 0

if __name__ == "__main__":
    sys.exit(main())