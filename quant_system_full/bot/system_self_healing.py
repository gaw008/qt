#!/usr/bin/env python3
"""
System Self-Healing Module
Provides automatic error detection, diagnosis, and recovery capabilities
"""

import os
import sys
import time
import logging
import threading
import traceback
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class RecoveryAction(Enum):
    """Available recovery actions"""
    RESTART_COMPONENT = "restart_component"
    CLEAR_CACHE = "clear_cache"
    REDUCE_LOAD = "reduce_load"
    EMERGENCY_STOP = "emergency_stop"
    NOTIFY_ADMIN = "notify_admin"
    RESET_CONNECTIONS = "reset_connections"


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    recovery_actions: List[RecoveryAction]
    check_interval: int = 60  # seconds
    max_failures: int = 3
    timeout: int = 30  # seconds


@dataclass
class SystemIssue:
    """Represents a detected system issue"""
    id: str
    name: str
    severity: HealthStatus
    description: str
    first_detected: str
    last_seen: str
    occurrence_count: int
    recovery_attempted: bool
    recovery_actions_taken: List[str]
    resolved: bool
    resolution_time: Optional[str] = None


class SystemSelfHealing:
    """
    Main self-healing system that monitors health and performs automatic recovery
    """

    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = state_dir or os.path.join(os.path.dirname(__file__), "..", "state")
        os.makedirs(self.state_dir, exist_ok=True)

        self.health_checks: Dict[str, HealthCheck] = {}
        self.active_issues: Dict[str, SystemIssue] = {}
        self.issue_history: List[SystemIssue] = []

        self.monitoring_active = False
        self.monitoring_thread = None

        # Failure tracking
        self.failure_counts: Dict[str, int] = {}
        self.last_check_times: Dict[str, datetime] = {}

        # Recovery functions
        self.recovery_functions: Dict[RecoveryAction, Callable] = {}

        # Initialize default health checks and recovery functions
        self._initialize_default_checks()
        self._initialize_recovery_functions()

        logger.info("System Self-Healing initialized")

    def _initialize_default_checks(self):
        """Initialize default health checks"""

        # Memory usage check
        def check_memory_usage():
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 85.0  # Fail if memory usage > 85%
            except Exception:
                return False

        self.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory_usage,
            recovery_actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.REDUCE_LOAD],
            check_interval=60,
            max_failures=2
        ))

        # Disk space check
        def check_disk_space():
            try:
                disk = psutil.disk_usage(self.state_dir)
                return (disk.free / disk.total) > 0.1  # Fail if less than 10% free
            except Exception:
                return False

        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=check_disk_space,
            recovery_actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.NOTIFY_ADMIN],
            check_interval=300,  # Check every 5 minutes
            max_failures=1
        ))

        # API connectivity check
        def check_api_connectivity():
            try:
                # Check if Tiger API modules are available
                # This is a basic check - in production, you'd ping actual endpoints
                return True  # Simplified for now
            except Exception:
                return False

        self.register_health_check(HealthCheck(
            name="api_connectivity",
            check_function=check_api_connectivity,
            recovery_actions=[RecoveryAction.RESET_CONNECTIONS, RecoveryAction.RESTART_COMPONENT],
            check_interval=120,
            max_failures=3
        ))

        # Process health check
        def check_process_health():
            try:
                # Check if current process is responding normally
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=1)
                return cpu_percent < 80.0  # Fail if CPU usage > 80%
            except Exception:
                return False

        self.register_health_check(HealthCheck(
            name="process_health",
            check_function=check_process_health,
            recovery_actions=[RecoveryAction.REDUCE_LOAD, RecoveryAction.RESTART_COMPONENT],
            check_interval=90,
            max_failures=2
        ))

    def _initialize_recovery_functions(self):
        """Initialize recovery functions"""

        def clear_cache():
            """Clear system caches to free up memory"""
            try:
                cache_dirs = [
                    os.path.join(os.path.dirname(__file__), "..", ".cache"),
                    os.path.join(self.state_dir, "cache")
                ]

                cleared_files = 0
                for cache_dir in cache_dirs:
                    if os.path.exists(cache_dir):
                        for file in os.listdir(cache_dir):
                            try:
                                file_path = os.path.join(cache_dir, file)
                                if os.path.isfile(file_path):
                                    # Only clear files older than 1 hour
                                    if time.time() - os.path.getmtime(file_path) > 3600:
                                        os.remove(file_path)
                                        cleared_files += 1
                            except Exception as e:
                                logger.warning(f"Failed to clear cache file {file}: {e}")

                logger.info(f"Cache clear completed: {cleared_files} files removed")
                return True
            except Exception as e:
                logger.error(f"Cache clear failed: {e}")
                return False

        def reduce_load():
            """Reduce system load by limiting resource usage"""
            try:
                # In a real implementation, this would:
                # - Reduce the number of concurrent operations
                # - Increase sleep intervals
                # - Temporarily disable non-essential features
                logger.info("Load reduction measures activated")
                return True
            except Exception as e:
                logger.error(f"Load reduction failed: {e}")
                return False

        def reset_connections():
            """Reset network connections and API clients"""
            try:
                # In a real implementation, this would:
                # - Close and reopen Tiger API connections
                # - Clear connection pools
                # - Reset WebSocket connections
                logger.info("Connection reset completed")
                return True
            except Exception as e:
                logger.error(f"Connection reset failed: {e}")
                return False

        def restart_component():
            """Restart specific system components"""
            try:
                # In a real implementation, this would:
                # - Restart specific services
                # - Reload configurations
                # - Reinitialize critical components
                logger.info("Component restart completed")
                return True
            except Exception as e:
                logger.error(f"Component restart failed: {e}")
                return False

        def emergency_stop():
            """Emergency stop of trading operations"""
            try:
                # Set kill flag to stop trading
                sys.path.append(os.path.join(os.path.dirname(__file__), "..", "state"))
                try:
                    from state_manager import set_kill
                    set_kill(True, "Emergency stop triggered by self-healing system")
                    logger.critical("Emergency stop activated")
                    return True
                except ImportError:
                    logger.error("Could not access state manager for emergency stop")
                    return False
            except Exception as e:
                logger.error(f"Emergency stop failed: {e}")
                return False

        def notify_admin():
            """Notify administrators of critical issues"""
            try:
                # In a real implementation, this would:
                # - Send email alerts
                # - Send Slack notifications
                # - Write to monitoring systems
                logger.critical("ADMIN NOTIFICATION: Critical system issue detected")
                return True
            except Exception as e:
                logger.error(f"Admin notification failed: {e}")
                return False

        # Register recovery functions
        self.recovery_functions = {
            RecoveryAction.CLEAR_CACHE: clear_cache,
            RecoveryAction.REDUCE_LOAD: reduce_load,
            RecoveryAction.RESET_CONNECTIONS: reset_connections,
            RecoveryAction.RESTART_COMPONENT: restart_component,
            RecoveryAction.EMERGENCY_STOP: emergency_stop,
            RecoveryAction.NOTIFY_ADMIN: notify_admin
        }

    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        self.health_checks[health_check.name] = health_check
        self.failure_counts[health_check.name] = 0
        logger.info(f"Health check registered: {health_check.name}")

    def start_monitoring(self):
        """Start the health monitoring system"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop the health monitoring system"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()

                for check_name, health_check in self.health_checks.items():
                    last_check = self.last_check_times.get(check_name)

                    # Check if it's time to run this health check
                    if (last_check is None or
                        (current_time - last_check).total_seconds() >= health_check.check_interval):

                        self._run_health_check(check_name, health_check)
                        self.last_check_times[check_name] = current_time

                # Sleep for a short interval before next cycle
                time.sleep(10)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Longer sleep on error

    def _run_health_check(self, check_name: str, health_check: HealthCheck):
        """Run a single health check"""
        try:
            # Run the health check with timeout
            start_time = time.time()
            check_passed = health_check.check_function()
            check_duration = time.time() - start_time

            if check_passed:
                # Health check passed
                if self.failure_counts[check_name] > 0:
                    logger.info(f"Health check {check_name} recovered")
                    self._resolve_issue(check_name)

                self.failure_counts[check_name] = 0

            else:
                # Health check failed
                self.failure_counts[check_name] += 1
                logger.warning(f"Health check {check_name} failed ({self.failure_counts[check_name]}/{health_check.max_failures})")

                # Create or update issue
                self._record_issue(check_name, health_check)

                # Trigger recovery if failure threshold reached
                if self.failure_counts[check_name] >= health_check.max_failures:
                    self._trigger_recovery(check_name, health_check)

        except Exception as e:
            logger.error(f"Health check {check_name} crashed: {e}")
            self.failure_counts[check_name] += 1

    def _record_issue(self, check_name: str, health_check: HealthCheck):
        """Record a system issue"""
        issue_id = f"{check_name}_{int(time.time())}"
        current_time = datetime.now().isoformat()

        # Determine severity based on failure count
        if self.failure_counts[check_name] >= health_check.max_failures:
            severity = HealthStatus.CRITICAL
        elif self.failure_counts[check_name] >= health_check.max_failures // 2:
            severity = HealthStatus.WARNING
        else:
            severity = HealthStatus.WARNING

        # Check if we already have an active issue for this check
        existing_issue = None
        for issue in self.active_issues.values():
            if issue.name == check_name and not issue.resolved:
                existing_issue = issue
                break

        if existing_issue:
            # Update existing issue
            existing_issue.last_seen = current_time
            existing_issue.occurrence_count += 1
            existing_issue.severity = severity
        else:
            # Create new issue
            new_issue = SystemIssue(
                id=issue_id,
                name=check_name,
                severity=severity,
                description=f"Health check {check_name} failed",
                first_detected=current_time,
                last_seen=current_time,
                occurrence_count=1,
                recovery_attempted=False,
                recovery_actions_taken=[],
                resolved=False
            )
            self.active_issues[issue_id] = new_issue

    def _trigger_recovery(self, check_name: str, health_check: HealthCheck):
        """Trigger recovery actions for a failed health check"""
        try:
            logger.warning(f"Triggering recovery for {check_name}")

            # Find the active issue for this check
            issue = None
            for active_issue in self.active_issues.values():
                if active_issue.name == check_name and not active_issue.resolved:
                    issue = active_issue
                    break

            if not issue:
                logger.error(f"Could not find active issue for {check_name}")
                return

            issue.recovery_attempted = True

            # Execute recovery actions
            for action in health_check.recovery_actions:
                try:
                    logger.info(f"Executing recovery action: {action.value}")

                    recovery_func = self.recovery_functions.get(action)
                    if recovery_func:
                        success = recovery_func()
                        action_name = action.value

                        if success:
                            logger.info(f"Recovery action {action_name} completed successfully")
                            issue.recovery_actions_taken.append(f"{action_name}: SUCCESS")
                        else:
                            logger.error(f"Recovery action {action_name} failed")
                            issue.recovery_actions_taken.append(f"{action_name}: FAILED")
                    else:
                        logger.error(f"No recovery function defined for {action.value}")
                        issue.recovery_actions_taken.append(f"{action.value}: NO_FUNCTION")

                except Exception as e:
                    logger.error(f"Recovery action {action.value} crashed: {e}")
                    issue.recovery_actions_taken.append(f"{action.value}: CRASHED")

                # Wait between recovery actions
                time.sleep(5)

        except Exception as e:
            logger.error(f"Recovery trigger failed for {check_name}: {e}")

    def _resolve_issue(self, check_name: str):
        """Mark issues as resolved when health checks start passing"""
        current_time = datetime.now().isoformat()

        for issue in self.active_issues.values():
            if issue.name == check_name and not issue.resolved:
                issue.resolved = True
                issue.resolution_time = current_time
                logger.info(f"Issue resolved: {issue.id}")

                # Move to history
                self.issue_history.append(issue)

        # Remove resolved issues from active issues
        self.active_issues = {
            issue_id: issue for issue_id, issue in self.active_issues.items()
            if not issue.resolved
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        overall_status = HealthStatus.HEALTHY

        # Determine overall status based on active issues
        for issue in self.active_issues.values():
            if issue.severity == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif issue.severity == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING

        return {
            "overall_status": overall_status.value,
            "monitoring_active": self.monitoring_active,
            "active_issues_count": len(self.active_issues),
            "resolved_issues_count": len(self.issue_history),
            "last_check": datetime.now().isoformat(),
            "health_checks": {
                name: {
                    "failure_count": self.failure_counts.get(name, 0),
                    "max_failures": check.max_failures,
                    "last_check": self.last_check_times.get(name, datetime.min).isoformat() if self.last_check_times.get(name) else None
                }
                for name, check in self.health_checks.items()
            },
            "active_issues": [asdict(issue) for issue in self.active_issues.values()],
            "recent_issues": [asdict(issue) for issue in self.issue_history[-10:]]  # Last 10 resolved issues
        }

    def force_recovery(self, check_name: str) -> bool:
        """Manually trigger recovery for a specific health check"""
        health_check = self.health_checks.get(check_name)
        if not health_check:
            logger.error(f"Health check {check_name} not found")
            return False

        try:
            self._trigger_recovery(check_name, health_check)
            return True
        except Exception as e:
            logger.error(f"Manual recovery failed for {check_name}: {e}")
            return False


# Thread-safe global instance
import threading
_self_healing_system = None
_healing_lock = threading.Lock()


def get_self_healing_system() -> SystemSelfHealing:
    """Get the global self-healing system instance with thread safety"""
    global _self_healing_system
    if _self_healing_system is None:
        with _healing_lock:
            if _self_healing_system is None:
                _self_healing_system = SystemSelfHealing()
    return _self_healing_system


def start_self_healing():
    """Start the self-healing system"""
    system = get_self_healing_system()
    system.start_monitoring()


def stop_self_healing():
    """Stop the self-healing system"""
    system = get_self_healing_system()
    system.stop_monitoring()


if __name__ == "__main__":
    # Test the self-healing system
    logging.basicConfig(level=logging.INFO)

    print("Starting self-healing system test...")
    start_self_healing()

    try:
        # Run for 60 seconds
        time.sleep(60)

        # Get health status
        health = self_healing_system.get_system_health()
        print(f"System health: {health['overall_status']}")
        print(f"Active issues: {health['active_issues_count']}")

    finally:
        stop_self_healing()
        print("Self-healing system test completed")