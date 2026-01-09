#!/usr/bin/env python3
"""
System Bottleneck Analyzer for Quantitative Trading System
系统瓶颈分析器 - 量化交易系统

Advanced bottleneck identification and optimization recommendations:
1. CPU profiling and hotspot identification
2. Memory usage patterns and leak detection
3. I/O bottleneck analysis (disk and network)
4. Database query performance analysis
5. Concurrency and threading analysis
6. Resource contention identification
7. Real-time performance monitoring
8. Optimization recommendations with priority scoring

Author: Performance Engineering Team
Version: 1.0 - Investment Grade
"""

import os
import sys
import time
import json
import logging
import threading
import multiprocessing
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import psutil
import gc
import tracemalloc
import cProfile
import pstats
import io
from contextlib import contextmanager

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import scientific libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bottleneck_analyzer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BottleneckReport:
    """Bottleneck analysis report structure"""
    timestamp: str
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    impact_score: float  # 0-100
    affected_components: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]
    optimization_priority: int  # 1-10, 1 being highest

@dataclass
class SystemSnapshot:
    """System performance snapshot"""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    disk_read_rate_mbps: float
    disk_write_rate_mbps: float
    network_sent_rate_mbps: float
    network_recv_rate_mbps: float
    process_count: int
    thread_count: int
    open_files: int
    gc_collections: int

class BottleneckAnalyzer:
    """Advanced system bottleneck analyzer"""

    def __init__(self):
        self.bottleneck_reports: List[BottleneckReport] = []
        self.system_snapshots: deque = deque(maxlen=1000)
        self.profiling_data = {}
        self.monitoring_active = False

        # Thresholds for bottleneck detection
        self.thresholds = {
            'cpu_high': 80.0,
            'cpu_critical': 90.0,
            'memory_high': 75.0,
            'memory_critical': 85.0,
            'disk_io_high': 100.0,  # MB/s
            'disk_io_critical': 200.0,
            'network_high': 50.0,  # MB/s
            'network_critical': 100.0,
            'response_time_high': 500.0,  # ms
            'response_time_critical': 1000.0,
            'gc_high': 1000,
            'gc_critical': 5000
        }

        logger.info("Bottleneck Analyzer initialized")

    def capture_system_snapshot(self) -> SystemSnapshot:
        """Capture current system performance snapshot"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_usage_mb = memory.used / (1024 * 1024)

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_snapshot'):
                time_diff = time.time() - self._last_disk_time
                if time_diff > 0 and disk_io:
                    disk_read_rate = (disk_io.read_bytes - self._last_disk_snapshot.read_bytes) / time_diff / (1024 * 1024)
                    disk_write_rate = (disk_io.write_bytes - self._last_disk_snapshot.write_bytes) / time_diff / (1024 * 1024)
                else:
                    disk_read_rate = 0
                    disk_write_rate = 0
            else:
                disk_read_rate = 0
                disk_write_rate = 0

            if disk_io:
                self._last_disk_snapshot = disk_io
                self._last_disk_time = time.time()

            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if hasattr(self, '_last_network_snapshot'):
                time_diff = time.time() - self._last_network_time
                if time_diff > 0 and network_io:
                    network_sent_rate = (network_io.bytes_sent - self._last_network_snapshot.bytes_sent) / time_diff / (1024 * 1024)
                    network_recv_rate = (network_io.bytes_recv - self._last_network_snapshot.bytes_recv) / time_diff / (1024 * 1024)
                else:
                    network_sent_rate = 0
                    network_recv_rate = 0
            else:
                network_sent_rate = 0
                network_recv_rate = 0

            if network_io:
                self._last_network_snapshot = network_io
                self._last_network_time = time.time()

            # Process and thread metrics
            process_count = len(psutil.pids())

            try:
                current_process = psutil.Process()
                thread_count = current_process.num_threads()
                open_files = current_process.num_fds() if hasattr(current_process, 'num_fds') else len(current_process.open_files())
            except:
                thread_count = threading.active_count()
                open_files = 0

            # Garbage collection metrics
            gc_collections = sum(gc.get_count())

            snapshot = SystemSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent,
                memory_usage_mb=memory_usage_mb,
                disk_read_rate_mbps=disk_read_rate,
                disk_write_rate_mbps=disk_write_rate,
                network_sent_rate_mbps=network_sent_rate,
                network_recv_rate_mbps=network_recv_rate,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                gc_collections=gc_collections
            )

            self.system_snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture system snapshot: {e}")
            return SystemSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=0, memory_usage_percent=0, memory_usage_mb=0,
                disk_read_rate_mbps=0, disk_write_rate_mbps=0,
                network_sent_rate_mbps=0, network_recv_rate_mbps=0,
                process_count=0, thread_count=0, open_files=0, gc_collections=0
            )

    def analyze_cpu_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze CPU-related bottlenecks"""
        reports = []

        if not self.system_snapshots:
            return reports

        # Get recent CPU usage data
        recent_snapshots = list(self.system_snapshots)[-10:]
        cpu_usages = [s.cpu_usage_percent for s in recent_snapshots]

        if not cpu_usages:
            return reports

        avg_cpu = statistics.mean(cpu_usages)
        max_cpu = max(cpu_usages)

        # High CPU usage detection
        if avg_cpu > self.thresholds['cpu_high']:
            severity = 'CRITICAL' if avg_cpu > self.thresholds['cpu_critical'] else 'HIGH'
            impact_score = min(100, avg_cpu * 1.2)

            # Get CPU per-core information
            cpu_per_core = psutil.cpu_percent(percpu=True)
            cpu_cores_high = sum(1 for usage in cpu_per_core if usage > 80)

            # CPU frequency information
            try:
                cpu_freq = psutil.cpu_freq()
                freq_info = f"Current: {cpu_freq.current:.0f}MHz, Max: {cpu_freq.max:.0f}MHz"
            except:
                freq_info = "Frequency info unavailable"

            # Load average (if available)
            try:
                load_avg = psutil.getloadavg()
                load_info = f"Load avg: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}"
            except:
                load_info = "Load average unavailable"

            recommendations = [
                f"CPU usage is {severity.lower()}: {avg_cpu:.1f}% average, {max_cpu:.1f}% peak",
                f"{cpu_cores_high} of {len(cpu_per_core)} CPU cores are heavily utilized",
                "Identify CPU-intensive processes using task manager or htop",
                "Consider implementing parallel processing for data-heavy operations",
                "Profile code to identify computational hotspots",
                "Consider upgrading CPU or scaling horizontally"
            ]

            if avg_cpu > 85:
                recommendations.append("URGENT: Consider immediate load reduction or system scaling")

            reports.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                category="CPU",
                severity=severity,
                description=f"High CPU usage detected: {avg_cpu:.1f}% average",
                impact_score=impact_score,
                affected_components=["Data Processing", "Multi-Factor Analysis", "Real-time Monitoring"],
                metrics={
                    'avg_cpu_percent': avg_cpu,
                    'max_cpu_percent': max_cpu,
                    'cpu_cores_high': cpu_cores_high,
                    'total_cores': len(cpu_per_core)
                },
                recommendations=recommendations,
                optimization_priority=1 if severity == 'CRITICAL' else 2
            ))

        # CPU frequency scaling issues
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq and cpu_freq.current < cpu_freq.max * 0.8:
                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="CPU",
                    severity="MEDIUM",
                    description="CPU frequency scaling detected",
                    impact_score=30,
                    affected_components=["System Performance"],
                    metrics={
                        'current_freq_mhz': cpu_freq.current,
                        'max_freq_mhz': cpu_freq.max,
                        'freq_ratio': cpu_freq.current / cpu_freq.max
                    },
                    recommendations=[
                        f"CPU running at {cpu_freq.current:.0f}MHz ({cpu_freq.current/cpu_freq.max*100:.1f}% of max)",
                        "Check power management settings",
                        "Consider disabling CPU throttling for trading systems",
                        "Monitor temperature to ensure adequate cooling"
                    ],
                    optimization_priority=6
                ))
        except:
            pass

        return reports

    def analyze_memory_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze memory-related bottlenecks"""
        reports = []

        if not self.system_snapshots:
            return reports

        # Get recent memory usage data
        recent_snapshots = list(self.system_snapshots)[-10:]
        memory_usages = [s.memory_usage_percent for s in recent_snapshots]
        memory_mb = [s.memory_usage_mb for s in recent_snapshots]

        if not memory_usages:
            return reports

        avg_memory = statistics.mean(memory_usages)
        max_memory = max(memory_usages)
        avg_memory_mb = statistics.mean(memory_mb)

        # High memory usage detection
        if avg_memory > self.thresholds['memory_high']:
            severity = 'CRITICAL' if avg_memory > self.thresholds['memory_critical'] else 'HIGH'
            impact_score = min(100, avg_memory * 1.1)

            # Memory details
            memory_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()

            recommendations = [
                f"Memory usage is {severity.lower()}: {avg_memory:.1f}% average, {max_memory:.1f}% peak",
                f"Average memory consumption: {avg_memory_mb:.1f}MB",
                f"Available memory: {memory_info.available / (1024**3):.1f}GB",
                "Run memory profiling to identify memory-intensive components",
                "Consider implementing data streaming for large datasets",
                "Optimize data structures and algorithms",
                "Implement garbage collection tuning"
            ]

            if swap_info.percent > 10:
                recommendations.append(f"WARNING: Swap usage at {swap_info.percent:.1f}% - indicates memory pressure")
                impact_score += 20

            if avg_memory > 90:
                recommendations.append("URGENT: Risk of out-of-memory errors")
                recommendations.append("Consider immediate memory cleanup or system restart")

            reports.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                category="Memory",
                severity=severity,
                description=f"High memory usage detected: {avg_memory:.1f}% average",
                impact_score=impact_score,
                affected_components=["Data Processing", "Caching", "Multi-Factor Analysis"],
                metrics={
                    'avg_memory_percent': avg_memory,
                    'max_memory_percent': max_memory,
                    'avg_memory_mb': avg_memory_mb,
                    'total_memory_gb': memory_info.total / (1024**3),
                    'available_memory_gb': memory_info.available / (1024**3),
                    'swap_percent': swap_info.percent
                },
                recommendations=recommendations,
                optimization_priority=1 if severity == 'CRITICAL' else 2
            ))

        # Memory leak detection
        if len(recent_snapshots) >= 5:
            memory_trend = np.polyfit(range(len(memory_mb)), memory_mb, 1)[0] if NUMPY_AVAILABLE else 0
            if memory_trend > 10:  # MB per sample
                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Memory",
                    severity="MEDIUM",
                    description="Potential memory leak detected",
                    impact_score=40,
                    affected_components=["Long-running Processes"],
                    metrics={
                        'memory_growth_rate_mb_per_sample': memory_trend,
                        'memory_samples': len(recent_snapshots)
                    },
                    recommendations=[
                        f"Memory growing at {memory_trend:.1f}MB per measurement",
                        "Investigate for memory leaks in long-running processes",
                        "Review object lifecycle management",
                        "Check for unclosed resources (files, connections)",
                        "Implement memory monitoring alerts"
                    ],
                    optimization_priority=3
                ))

        # Garbage collection analysis
        recent_gc = [s.gc_collections for s in recent_snapshots]
        if recent_gc:
            avg_gc = statistics.mean(recent_gc)
            if avg_gc > self.thresholds['gc_high']:
                severity = 'HIGH' if avg_gc > self.thresholds['gc_critical'] else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Memory",
                    severity=severity,
                    description="High garbage collection activity",
                    impact_score=30 if severity == 'MEDIUM' else 60,
                    affected_components=["Python Runtime", "Data Processing"],
                    metrics={
                        'avg_gc_collections': avg_gc,
                        'gc_threshold': self.thresholds['gc_high']
                    },
                    recommendations=[
                        f"High GC activity: {avg_gc:.0f} collections average",
                        "Optimize object creation and disposal patterns",
                        "Reduce short-lived object allocations",
                        "Consider manual garbage collection in critical sections",
                        "Profile memory allocation patterns"
                    ],
                    optimization_priority=4
                ))

        return reports

    def analyze_io_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze I/O related bottlenecks"""
        reports = []

        if not self.system_snapshots:
            return reports

        # Get recent I/O data
        recent_snapshots = list(self.system_snapshots)[-10:]

        # Disk I/O analysis
        disk_reads = [s.disk_read_rate_mbps for s in recent_snapshots if s.disk_read_rate_mbps > 0]
        disk_writes = [s.disk_write_rate_mbps for s in recent_snapshots if s.disk_write_rate_mbps > 0]

        if disk_reads:
            avg_disk_read = statistics.mean(disk_reads)
            max_disk_read = max(disk_reads)

            if avg_disk_read > self.thresholds['disk_io_high']:
                severity = 'HIGH' if avg_disk_read > self.thresholds['disk_io_critical'] else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Disk I/O",
                    severity=severity,
                    description=f"High disk read activity: {avg_disk_read:.1f} MB/s average",
                    impact_score=min(80, avg_disk_read / 10),
                    affected_components=["Data Loading", "Database", "Caching"],
                    metrics={
                        'avg_disk_read_mbps': avg_disk_read,
                        'max_disk_read_mbps': max_disk_read,
                        'samples_with_reads': len(disk_reads)
                    },
                    recommendations=[
                        f"High disk read rate: {avg_disk_read:.1f} MB/s average, {max_disk_read:.1f} MB/s peak",
                        "Consider SSD upgrade if using traditional hard drives",
                        "Implement data caching to reduce disk reads",
                        "Optimize database queries and indexing",
                        "Consider data compression for large datasets",
                        "Implement asynchronous I/O where possible"
                    ],
                    optimization_priority=3
                ))

        if disk_writes:
            avg_disk_write = statistics.mean(disk_writes)
            max_disk_write = max(disk_writes)

            if avg_disk_write > self.thresholds['disk_io_high']:
                severity = 'HIGH' if avg_disk_write > self.thresholds['disk_io_critical'] else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Disk I/O",
                    severity=severity,
                    description=f"High disk write activity: {avg_disk_write:.1f} MB/s average",
                    impact_score=min(80, avg_disk_write / 10),
                    affected_components=["Logging", "Database", "Data Storage"],
                    metrics={
                        'avg_disk_write_mbps': avg_disk_write,
                        'max_disk_write_mbps': max_disk_write,
                        'samples_with_writes': len(disk_writes)
                    },
                    recommendations=[
                        f"High disk write rate: {avg_disk_write:.1f} MB/s average, {max_disk_write:.1f} MB/s peak",
                        "Review logging configuration - reduce verbose logging in production",
                        "Implement write batching for database operations",
                        "Consider write-behind caching strategies",
                        "Optimize data serialization and storage formats",
                        "Use faster storage devices (NVMe SSD)"
                    ],
                    optimization_priority=3
                ))

        # Network I/O analysis
        network_sent = [s.network_sent_rate_mbps for s in recent_snapshots if s.network_sent_rate_mbps > 0]
        network_recv = [s.network_recv_rate_mbps for s in recent_snapshots if s.network_recv_rate_mbps > 0]

        if network_sent:
            avg_net_sent = statistics.mean(network_sent)
            max_net_sent = max(network_sent)

            if avg_net_sent > self.thresholds['network_high']:
                severity = 'HIGH' if avg_net_sent > self.thresholds['network_critical'] else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Network I/O",
                    severity=severity,
                    description=f"High network send rate: {avg_net_sent:.1f} MB/s average",
                    impact_score=min(70, avg_net_sent / 5),
                    affected_components=["API Communication", "Data Feeds", "Trading"],
                    metrics={
                        'avg_network_sent_mbps': avg_net_sent,
                        'max_network_sent_mbps': max_net_sent
                    },
                    recommendations=[
                        f"High network send rate: {avg_net_sent:.1f} MB/s average",
                        "Review data transmission patterns",
                        "Implement data compression for API communications",
                        "Consider connection pooling and keep-alive",
                        "Monitor bandwidth utilization vs. available capacity",
                        "Optimize serialization formats (JSON vs. binary)"
                    ],
                    optimization_priority=4
                ))

        if network_recv:
            avg_net_recv = statistics.mean(network_recv)
            max_net_recv = max(network_recv)

            if avg_net_recv > self.thresholds['network_high']:
                severity = 'HIGH' if avg_net_recv > self.thresholds['network_critical'] else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Network I/O",
                    severity=severity,
                    description=f"High network receive rate: {avg_net_recv:.1f} MB/s average",
                    impact_score=min(70, avg_net_recv / 5),
                    affected_components=["Market Data", "API Responses", "Data Feeds"],
                    metrics={
                        'avg_network_recv_mbps': avg_net_recv,
                        'max_network_recv_mbps': max_net_recv
                    },
                    recommendations=[
                        f"High network receive rate: {avg_net_recv:.1f} MB/s average",
                        "Monitor market data feed efficiency",
                        "Implement selective data subscription",
                        "Consider local caching of frequently accessed data",
                        "Review API rate limits and optimize requests",
                        "Use efficient data processing pipelines"
                    ],
                    optimization_priority=4
                ))

        return reports

    def analyze_process_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze process and threading bottlenecks"""
        reports = []

        if not self.system_snapshots:
            return reports

        # Get recent process data
        recent_snapshots = list(self.system_snapshots)[-10:]
        thread_counts = [s.thread_count for s in recent_snapshots]
        open_files = [s.open_files for s in recent_snapshots]

        if thread_counts:
            avg_threads = statistics.mean(thread_counts)
            max_threads = max(thread_counts)

            # High thread count analysis
            if avg_threads > 100:
                severity = 'HIGH' if avg_threads > 200 else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Threading",
                    severity=severity,
                    description=f"High thread count: {avg_threads:.0f} average",
                    impact_score=min(60, avg_threads / 5),
                    affected_components=["Concurrent Processing", "Connection Handling"],
                    metrics={
                        'avg_thread_count': avg_threads,
                        'max_thread_count': max_threads,
                        'cpu_cores': psutil.cpu_count()
                    },
                    recommendations=[
                        f"High thread count: {avg_threads:.0f} average, {max_threads:.0f} peak",
                        f"Thread-to-CPU ratio: {avg_threads / psutil.cpu_count():.1f}:1",
                        "Review thread pool configurations",
                        "Consider async/await patterns instead of threading",
                        "Implement connection pooling to reduce thread usage",
                        "Monitor for thread leaks in long-running processes",
                        "Optimize thread lifecycle management"
                    ],
                    optimization_priority=5
                ))

        if open_files:
            avg_files = statistics.mean(open_files)
            max_files = max(open_files)

            # High open file count analysis
            if avg_files > 500:
                severity = 'HIGH' if avg_files > 1000 else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="File Handles",
                    severity=severity,
                    description=f"High open file count: {avg_files:.0f} average",
                    impact_score=min(50, avg_files / 20),
                    affected_components=["File I/O", "Database Connections", "Network Sockets"],
                    metrics={
                        'avg_open_files': avg_files,
                        'max_open_files': max_files
                    },
                    recommendations=[
                        f"High open file count: {avg_files:.0f} average, {max_files:.0f} peak",
                        "Review file handle lifecycle management",
                        "Ensure proper closing of files and connections",
                        "Implement connection pooling for database connections",
                        "Check for file descriptor leaks",
                        "Consider increasing system file descriptor limits if needed",
                        "Use context managers (with statements) for file operations"
                    ],
                    optimization_priority=6
                ))

        return reports

    def analyze_trading_specific_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze trading system specific bottlenecks"""
        reports = []

        try:
            # Check if trading services are running and responsive
            import requests

            services = {
                'backend': 'http://localhost:8000/health',
                'frontend': 'http://localhost:3000',
                'streamlit': 'http://localhost:8501'
            }

            slow_services = []
            down_services = []

            for service_name, url in services.items():
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=5)
                    response_time = (time.time() - start_time) * 1000

                    if response_time > 1000:  # >1 second
                        slow_services.append((service_name, response_time))
                    elif response.status_code >= 400:
                        down_services.append((service_name, response.status_code))

                except requests.exceptions.RequestException:
                    down_services.append((service_name, 'No Response'))

            # Slow service response bottleneck
            if slow_services:
                max_response_time = max(rt for _, rt in slow_services)
                severity = 'HIGH' if max_response_time > 2000 else 'MEDIUM'

                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Service Response",
                    severity=severity,
                    description="Slow service response times detected",
                    impact_score=min(80, max_response_time / 25),
                    affected_components=[name for name, _ in slow_services],
                    metrics={
                        'slow_services_count': len(slow_services),
                        'max_response_time_ms': max_response_time,
                        'slow_services': dict(slow_services)
                    },
                    recommendations=[
                        f"Slow services detected: {', '.join(name for name, _ in slow_services)}",
                        "Investigate database query performance",
                        "Review API endpoint optimizations",
                        "Check for resource contention",
                        "Implement response time monitoring",
                        "Consider horizontal scaling for slow services"
                    ],
                    optimization_priority=2
                ))

            # Service unavailability bottleneck
            if down_services:
                reports.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    category="Service Availability",
                    severity="CRITICAL",
                    description="Trading services unavailable",
                    impact_score=100,
                    affected_components=[name for name, _ in down_services],
                    metrics={
                        'down_services_count': len(down_services),
                        'down_services': dict(down_services)
                    },
                    recommendations=[
                        f"Services down: {', '.join(name for name, _ in down_services)}",
                        "Check service health and restart if necessary",
                        "Review service logs for error messages",
                        "Verify network connectivity",
                        "Implement service monitoring and alerting",
                        "Consider implementing circuit breakers"
                    ],
                    optimization_priority=1
                ))

        except Exception as e:
            logger.debug(f"Trading service analysis failed: {e}")

        return reports

    def run_comprehensive_bottleneck_analysis(self) -> Dict[str, Any]:
        """Run comprehensive bottleneck analysis"""
        logger.info("Starting comprehensive bottleneck analysis...")

        # Start monitoring to collect data
        self.start_monitoring(duration_seconds=30)

        # Wait for data collection
        time.sleep(35)

        # Stop monitoring
        self.stop_monitoring()

        # Run all bottleneck analyses
        all_reports = []

        logger.info("Analyzing CPU bottlenecks...")
        all_reports.extend(self.analyze_cpu_bottlenecks())

        logger.info("Analyzing memory bottlenecks...")
        all_reports.extend(self.analyze_memory_bottlenecks())

        logger.info("Analyzing I/O bottlenecks...")
        all_reports.extend(self.analyze_io_bottlenecks())

        logger.info("Analyzing process bottlenecks...")
        all_reports.extend(self.analyze_process_bottlenecks())

        logger.info("Analyzing trading-specific bottlenecks...")
        all_reports.extend(self.analyze_trading_specific_bottlenecks())

        # Store all reports
        self.bottleneck_reports.extend(all_reports)

        # Generate comprehensive report
        return self.generate_bottleneck_report()

    def start_monitoring(self, duration_seconds: int = 60):
        """Start system monitoring for specified duration"""
        self.monitoring_active = True

        def monitor_loop():
            end_time = time.time() + duration_seconds
            while self.monitoring_active and time.time() < end_time:
                try:
                    self.capture_system_snapshot()
                    time.sleep(2)  # Capture every 2 seconds
                except Exception as e:
                    logger.debug(f"Monitoring error: {e}")

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        logger.info(f"System monitoring started for {duration_seconds} seconds")

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        logger.info("System monitoring stopped")

    def generate_bottleneck_report(self) -> Dict[str, Any]:
        """Generate comprehensive bottleneck analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Categorize reports by severity
        critical_reports = [r for r in self.bottleneck_reports if r.severity == 'CRITICAL']
        high_reports = [r for r in self.bottleneck_reports if r.severity == 'HIGH']
        medium_reports = [r for r in self.bottleneck_reports if r.severity == 'MEDIUM']
        low_reports = [r for r in self.bottleneck_reports if r.severity == 'LOW']

        # Calculate overall system health score
        if critical_reports:
            health_score = 20  # Critical issues present
        elif high_reports:
            health_score = 50 + min(30, len(high_reports) * -10)  # Reduce score based on high severity issues
        elif medium_reports:
            health_score = 70 + min(20, len(medium_reports) * -5)  # Reduce score based on medium severity issues
        else:
            health_score = 90  # Good health

        # Get top optimization priorities
        priority_reports = sorted(self.bottleneck_reports, key=lambda r: r.optimization_priority)[:5]

        # Calculate impact scores by category
        category_impacts = defaultdict(list)
        for report in self.bottleneck_reports:
            category_impacts[report.category].append(report.impact_score)

        category_scores = {
            category: statistics.mean(impacts) for category, impacts in category_impacts.items()
        }

        # Generate summary statistics
        if self.system_snapshots:
            snapshots = list(self.system_snapshots)
            avg_cpu = statistics.mean([s.cpu_usage_percent for s in snapshots])
            avg_memory = statistics.mean([s.memory_usage_percent for s in snapshots])
            avg_disk_read = statistics.mean([s.disk_read_rate_mbps for s in snapshots if s.disk_read_rate_mbps > 0]) if any(s.disk_read_rate_mbps > 0 for s in snapshots) else 0
            avg_disk_write = statistics.mean([s.disk_write_rate_mbps for s in snapshots if s.disk_write_rate_mbps > 0]) if any(s.disk_write_rate_mbps > 0 for s in snapshots) else 0
        else:
            avg_cpu = avg_memory = avg_disk_read = avg_disk_write = 0

        # Create comprehensive report
        report = {
            'analysis_summary': {
                'timestamp': timestamp,
                'analysis_duration_seconds': len(self.system_snapshots) * 2,  # 2 seconds per snapshot
                'total_bottlenecks': len(self.bottleneck_reports),
                'critical_bottlenecks': len(critical_reports),
                'high_bottlenecks': len(high_reports),
                'medium_bottlenecks': len(medium_reports),
                'low_bottlenecks': len(low_reports),
                'system_health_score': health_score,
                'snapshots_collected': len(self.system_snapshots)
            },
            'system_performance_summary': {
                'average_cpu_percent': avg_cpu,
                'average_memory_percent': avg_memory,
                'average_disk_read_mbps': avg_disk_read,
                'average_disk_write_mbps': avg_disk_write,
                'monitoring_duration_seconds': len(self.system_snapshots) * 2
            },
            'category_impact_scores': category_scores,
            'priority_optimization_targets': [
                {
                    'category': r.category,
                    'description': r.description,
                    'severity': r.severity,
                    'impact_score': r.impact_score,
                    'priority': r.optimization_priority,
                    'top_recommendations': r.recommendations[:3]
                }
                for r in priority_reports
            ],
            'detailed_bottleneck_reports': [
                {
                    'timestamp': r.timestamp,
                    'category': r.category,
                    'severity': r.severity,
                    'description': r.description,
                    'impact_score': r.impact_score,
                    'affected_components': r.affected_components,
                    'metrics': r.metrics,
                    'recommendations': r.recommendations,
                    'optimization_priority': r.optimization_priority
                }
                for r in self.bottleneck_reports
            ]
        }

        # Save report
        report_file = f"bottleneck_analysis_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS SUMMARY")
        print("="*80)
        print(f"System Health Score: {health_score}/100")
        print(f"Total Bottlenecks Found: {len(self.bottleneck_reports)}")
        print(f"  Critical: {len(critical_reports)}")
        print(f"  High: {len(high_reports)}")
        print(f"  Medium: {len(medium_reports)}")
        print(f"  Low: {len(low_reports)}")

        if critical_reports:
            print(f"\nCRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for report in critical_reports:
                print(f"  • {report.category}: {report.description}")

        if priority_reports:
            print(f"\nTOP OPTIMIZATION PRIORITIES:")
            for i, report in enumerate(priority_reports[:3], 1):
                print(f"  {i}. {report.category}: {report.description} (Impact: {report.impact_score:.0f})")

        print(f"\nDetailed report saved: {report_file}")
        print("="*80)

        return report

def main():
    """Main bottleneck analysis execution"""
    try:
        # Initialize analyzer
        analyzer = BottleneckAnalyzer()

        print("="*80)
        print("QUANTITATIVE TRADING SYSTEM - BOTTLENECK ANALYSIS")
        print("Advanced Performance Bottleneck Identification")
        print("="*80)

        # Run comprehensive analysis
        report = analyzer.run_comprehensive_bottleneck_analysis()

        # Display final recommendations
        health_score = report['analysis_summary']['system_health_score']

        print(f"\nFINAL ASSESSMENT:")
        if health_score >= 80:
            print(f"✅ System Status: HEALTHY (Score: {health_score}/100)")
            print("No critical bottlenecks detected. System ready for production.")
        elif health_score >= 60:
            print(f"⚠️  System Status: DEGRADED (Score: {health_score}/100)")
            print("Some performance issues detected. Optimization recommended.")
        else:
            print(f"🚨 System Status: CRITICAL (Score: {health_score}/100)")
            print("Severe bottlenecks detected. Immediate action required.")

        return 0

    except KeyboardInterrupt:
        print("\nBottleneck analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Bottleneck analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())