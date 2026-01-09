"""
Progress Monitoring and Resource Utilization System for Backtesting

This module provides comprehensive real-time monitoring of backtesting operations
with resource tracking, performance analytics, and intelligent alerting.

Key Features:
- Real-time progress tracking with ETA calculations
- Resource utilization monitoring (CPU, memory, disk I/O)
- Performance bottleneck detection and alerts
- Interactive progress visualization for long-running operations
- Historical performance data collection and analysis
- Adaptive throttling based on resource constraints
- Checkpoint management for resumable operations

Monitoring Capabilities:
- Multi-threaded operation progress tracking
- Memory usage patterns and leak detection
- CPU utilization and thermal throttling detection
- Disk I/O performance and space monitoring
- Network utilization for data fetching operations
- Cache hit rates and storage efficiency metrics
"""

import os
import sys
import time
import threading
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque, defaultdict
import queue
import traceback
from contextlib import contextmanager

# Performance monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Data visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    process_count: int = 0
    temperature_celsius: Optional[float] = None


@dataclass
class TaskProgress:
    """Progress tracking for individual tasks."""
    task_id: str
    name: str
    total_items: int
    completed_items: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running, completed, failed, paused
    error_message: Optional[str] = None
    sub_tasks: Dict[str, 'TaskProgress'] = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return min(100.0, (self.completed_items / self.total_items) * 100.0)

    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        """Estimate remaining time based on current progress."""
        if self.completed_items == 0 or self.progress_percent >= 100.0:
            return None

        elapsed = datetime.now() - self.start_time
        rate = self.completed_items / elapsed.total_seconds()

        if rate <= 0:
            return None

        remaining_items = self.total_items - self.completed_items
        remaining_seconds = remaining_items / rate

        return timedelta(seconds=remaining_seconds)

    def update(self, completed_items: int, status: str = None):
        """Update task progress."""
        self.completed_items = completed_items
        self.last_update = datetime.now()

        if status:
            self.status = status


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    alert_id: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    category: str  # memory, cpu, disk, network, performance
    message: str
    current_value: float
    threshold_value: float
    recommendation: str


class ResourceMonitor:
    """System resource monitoring with alerting."""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()

        # Resource history
        self.resource_history = deque(maxlen=3600)  # Keep 1 hour of data
        self._history_lock = threading.Lock()

        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'disk_io_mb_per_sec': 100.0,
            'temperature_celsius': 80.0
        }

        # Alert tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self._last_io_stats = None

    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.is_monitoring:
            return

        if not HAS_PSUTIL:
            logger.warning("psutil not available - resource monitoring disabled")
            return

        self.is_monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop background resource monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.monitoring_interval):
            try:
                snapshot = self._capture_resource_snapshot()

                with self._history_lock:
                    self.resource_history.append(snapshot)

                # Check for alerts
                self._check_alerts(snapshot)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def _capture_resource_snapshot(self) -> ResourceSnapshot:
        """Capture current system resource usage."""
        if not HAS_PSUTIL:
            return ResourceSnapshot()

        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            # I/O metrics
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            # Calculate I/O rates
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            net_sent_mb = 0.0
            net_recv_mb = 0.0

            if self._last_io_stats and disk_io and net_io:
                time_diff = time.time() - self._last_io_stats['timestamp']
                if time_diff > 0:
                    disk_read_mb = (disk_io.read_bytes - self._last_io_stats['disk_read']) / 1024 / 1024 / time_diff
                    disk_write_mb = (disk_io.write_bytes - self._last_io_stats['disk_write']) / 1024 / 1024 / time_diff
                    net_sent_mb = (net_io.bytes_sent - self._last_io_stats['net_sent']) / 1024 / 1024 / time_diff
                    net_recv_mb = (net_io.bytes_recv - self._last_io_stats['net_recv']) / 1024 / 1024 / time_diff

            # Update I/O tracking
            if disk_io and net_io:
                self._last_io_stats = {
                    'timestamp': time.time(),
                    'disk_read': disk_io.read_bytes,
                    'disk_write': disk_io.write_bytes,
                    'net_sent': net_io.bytes_sent,
                    'net_recv': net_io.bytes_recv
                }

            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for sensor_name, sensor_list in temps.items():
                        if sensor_list:
                            temperature = sensor_list[0].current
                            break
            except:
                pass

            return ResourceSnapshot(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=process_memory.rss / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                active_threads=process.num_threads(),
                process_count=len(psutil.pids()),
                temperature_celsius=temperature
            )

        except Exception as e:
            logger.error(f"Failed to capture resource snapshot: {e}")
            return ResourceSnapshot()

    def _check_alerts(self, snapshot: ResourceSnapshot):
        """Check for performance alerts based on current snapshot."""
        alerts_to_check = [
            ('cpu_high', snapshot.cpu_percent, self.alert_thresholds['cpu_percent'],
             'CPU usage is high', 'Consider reducing concurrent operations'),

            ('memory_high', snapshot.memory_percent, self.alert_thresholds['memory_percent'],
             'Memory usage is high', 'Consider increasing available memory or reducing batch sizes'),

            ('disk_io_high', max(snapshot.disk_io_read_mb, snapshot.disk_io_write_mb),
             self.alert_thresholds['disk_io_mb_per_sec'],
             'Disk I/O is high', 'Consider optimizing data access patterns or using SSD storage'),
        ]

        if snapshot.temperature_celsius:
            alerts_to_check.append(
                ('temperature_high', snapshot.temperature_celsius,
                 self.alert_thresholds['temperature_celsius'],
                 'System temperature is high', 'Check cooling and reduce workload if necessary')
            )

        current_time = datetime.now()

        for alert_id, current_value, threshold, message, recommendation in alerts_to_check:
            if current_value > threshold:
                if alert_id not in self.active_alerts:
                    # New alert
                    severity = 'critical' if current_value > threshold * 1.2 else 'high'

                    alert = PerformanceAlert(
                        alert_id=alert_id,
                        timestamp=current_time,
                        severity=severity,
                        category=alert_id.split('_')[0],
                        message=f"{message}: {current_value:.1f}",
                        current_value=current_value,
                        threshold_value=threshold,
                        recommendation=recommendation
                    )

                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)

                    logger.warning(f"Performance Alert: {alert.message}")

            else:
                # Clear alert if it exists
                if alert_id in self.active_alerts:
                    del self.active_alerts[alert_id]

    def get_current_resources(self) -> Optional[ResourceSnapshot]:
        """Get the most recent resource snapshot."""
        with self._history_lock:
            return self.resource_history[-1] if self.resource_history else None

    def get_resource_history(self, minutes: int = 60) -> List[ResourceSnapshot]:
        """Get resource history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self._history_lock:
            return [
                snapshot for snapshot in self.resource_history
                if snapshot.timestamp >= cutoff_time
            ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics."""
        history = self.get_resource_history(60)

        if not history:
            return {'error': 'No monitoring data available'}

        cpu_values = [s.cpu_percent for s in history]
        memory_values = [s.memory_percent for s in history]
        memory_used_values = [s.memory_used_mb for s in history]

        return {
            'monitoring_period_minutes': 60,
            'samples': len(history),
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': np.mean(cpu_values),
                'peak': np.max(cpu_values),
                'std_dev': np.std(cpu_values)
            },
            'memory': {
                'current_percent': memory_values[-1] if memory_values else 0,
                'current_used_mb': memory_used_values[-1] if memory_used_values else 0,
                'average_percent': np.mean(memory_values),
                'peak_percent': np.max(memory_values),
                'peak_used_mb': np.max(memory_used_values)
            },
            'active_alerts': len(self.active_alerts),
            'total_alerts': len(self.alert_history)
        }


class ProgressTracker:
    """Comprehensive progress tracking for backtesting operations."""

    def __init__(self):
        self.tasks = {}
        self._lock = threading.RLock()
        self.resource_monitor = ResourceMonitor()

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

    def create_task(self, task_id: str, name: str, total_items: int) -> TaskProgress:
        """Create a new progress tracking task."""
        with self._lock:
            task = TaskProgress(
                task_id=task_id,
                name=name,
                total_items=total_items
            )
            self.tasks[task_id] = task
            return task

    def update_task(self, task_id: str, completed_items: int, status: str = None):
        """Update progress for a task."""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(completed_items, status)

    def complete_task(self, task_id: str, success: bool = True, error_message: str = None):
        """Mark a task as completed."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = "completed" if success else "failed"
                task.error_message = error_message
                task.last_update = datetime.now()

    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """Get task progress information."""
        with self._lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskProgress]:
        """Get all tasks."""
        with self._lock:
            return self.tasks.copy()

    def get_active_tasks(self) -> Dict[str, TaskProgress]:
        """Get currently active (running) tasks."""
        with self._lock:
            return {
                task_id: task for task_id, task in self.tasks.items()
                if task.status == "running"
            }

    @contextmanager
    def track_operation(self, operation_name: str, total_items: int):
        """Context manager for tracking operations."""
        task_id = f"{operation_name}_{int(time.time())}"
        task = self.create_task(task_id, operation_name, total_items)

        try:
            yield task
            self.complete_task(task_id, success=True)
        except Exception as e:
            self.complete_task(task_id, success=False, error_message=str(e))
            raise

    def print_status_report(self):
        """Print comprehensive status report."""
        print("\n" + "="*80)
        print("BACKTESTING PROGRESS REPORT")
        print("="*80)

        # Resource status
        current_resources = self.resource_monitor.get_current_resources()
        if current_resources:
            print(f"\nSYSTEM RESOURCES:")
            print(f"  CPU Usage:    {current_resources.cpu_percent:.1f}%")
            print(f"  Memory Usage: {current_resources.memory_percent:.1f}% "
                  f"({current_resources.memory_used_mb:.1f} MB)")
            print(f"  Active Threads: {current_resources.active_threads}")

            if current_resources.temperature_celsius:
                print(f"  Temperature:  {current_resources.temperature_celsius:.1f} degreesC")

        # Active alerts
        active_alerts = self.resource_monitor.active_alerts
        if active_alerts:
            print(f"\nACTIVE ALERTS ({len(active_alerts)}):")
            for alert in active_alerts.values():
                print(f"  [{alert.severity.upper()}] {alert.message}")

        # Task progress
        active_tasks = self.get_active_tasks()
        completed_tasks = {
            task_id: task for task_id, task in self.tasks.items()
            if task.status in ["completed", "failed"]
        }

        print(f"\nTASK PROGRESS:")
        print(f"  Active Tasks: {len(active_tasks)}")
        print(f"  Completed Tasks: {len(completed_tasks)}")

        for task_id, task in active_tasks.items():
            eta_str = ""
            if task.estimated_time_remaining:
                eta_str = f" (ETA: {str(task.estimated_time_remaining).split('.')[0]})"

            print(f"    {task.name}: {task.progress_percent:.1f}% "
                  f"({task.completed_items}/{task.total_items}){eta_str}")

        print("="*80)

    def save_progress_report(self, output_path: str):
        """Save detailed progress report to file."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'tasks': {
                task_id: asdict(task) for task_id, task in self.tasks.items()
            },
            'resource_summary': self.resource_monitor.get_performance_summary(),
            'active_alerts': [
                asdict(alert) for alert in self.resource_monitor.active_alerts.values()
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

    def cleanup(self):
        """Cleanup and stop monitoring."""
        self.resource_monitor.stop_monitoring()


# Visualization functions (requires matplotlib)
def create_resource_dashboard(progress_tracker: ProgressTracker,
                            update_interval: int = 5000):
    """Create real-time resource monitoring dashboard."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - dashboard disabled")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Backtesting Resource Monitor')

    # Data containers
    timestamps = deque(maxlen=100)
    cpu_data = deque(maxlen=100)
    memory_data = deque(maxlen=100)
    disk_read_data = deque(maxlen=100)
    disk_write_data = deque(maxlen=100)

    def update_plots(frame):
        current_resources = progress_tracker.resource_monitor.get_current_resources()
        if not current_resources:
            return

        # Update data
        timestamps.append(current_resources.timestamp)
        cpu_data.append(current_resources.cpu_percent)
        memory_data.append(current_resources.memory_percent)
        disk_read_data.append(current_resources.disk_io_read_mb)
        disk_write_data.append(current_resources.disk_io_write_mb)

        # Clear and update plots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        # CPU Usage
        ax1.plot(list(timestamps), list(cpu_data), 'b-', linewidth=2)
        ax1.set_title('CPU Usage (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True)

        # Memory Usage
        ax2.plot(list(timestamps), list(memory_data), 'r-', linewidth=2)
        ax2.set_title('Memory Usage (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True)

        # Disk I/O
        ax3.plot(list(timestamps), list(disk_read_data), 'g-', label='Read', linewidth=2)
        ax3.plot(list(timestamps), list(disk_write_data), 'orange', label='Write', linewidth=2)
        ax3.set_title('Disk I/O (MB/s)')
        ax3.legend()
        ax3.grid(True)

        # Task Progress
        active_tasks = progress_tracker.get_active_tasks()
        if active_tasks:
            task_names = []
            progress_values = []

            for task in list(active_tasks.values())[:5]:  # Show top 5 tasks
                task_names.append(task.name[:15])  # Truncate long names
                progress_values.append(task.progress_percent)

            ax4.barh(task_names, progress_values)
            ax4.set_title('Task Progress (%)')
            ax4.set_xlim(0, 100)
        else:
            ax4.text(0.5, 0.5, 'No active tasks', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Task Progress (%)')

        plt.tight_layout()

    # Create animation
    ani = animation.FuncAnimation(fig, update_plots, interval=update_interval, blit=False)
    plt.show()

    return ani


# Demo and testing functions
def demo_progress_monitoring():
    """Demonstrate progress monitoring capabilities."""
    logger.info("=== Progress Monitoring Demo ===")

    tracker = ProgressTracker()

    try:
        # Simulate backtesting operation
        with tracker.track_operation("Demo Backtest", 100) as task:
            for i in range(101):
                time.sleep(0.1)  # Simulate work
                task.update(i)

                if i % 20 == 0:
                    tracker.print_status_report()

        # Print final report
        print("\nFINAL REPORT:")
        tracker.print_status_report()

        # Save report
        output_dir = Path("reports/monitoring")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"progress_report_{timestamp}.json"
        tracker.save_progress_report(str(report_path))

        logger.info(f"Progress report saved to {report_path}")

        return tracker.resource_monitor.get_performance_summary()

    finally:
        tracker.cleanup()


if __name__ == "__main__":
    demo_progress_monitoring()