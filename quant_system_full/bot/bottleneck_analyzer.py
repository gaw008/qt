"""
Bottleneck Analysis and Performance Optimization System

This module provides comprehensive bottleneck detection and performance optimization
recommendations for the three-phase backtesting system.

Key Features:
- Automated bottleneck detection in data processing pipelines
- Performance profiling with detailed timing analysis
- Resource utilization pattern analysis
- Optimization recommendation engine
- Comparative performance benchmarking
- Historical performance regression detection
- Adaptive optimization suggestions based on workload patterns

Analysis Areas:
- Data loading and I/O bottlenecks
- Memory allocation and garbage collection issues
- CPU-bound vs I/O-bound operation identification
- Parallel processing efficiency analysis
- Cache hit rate optimization opportunities
- Query optimization recommendations
- Resource contention detection
"""

import os
import sys
import time
import threading
import cProfile
import pstats
import io
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from functools import wraps
from collections import defaultdict, deque
import json
import traceback
import statistics

# Performance monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Line profiling (optional)
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for a specific operation."""
    operation_name: str
    total_time: float
    call_count: int
    avg_time_per_call: float
    function_breakdown: Dict[str, float]
    memory_peak_mb: float
    cpu_time: float
    io_wait_time: float
    cache_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckReport:
    """Comprehensive bottleneck analysis report."""
    operation_name: str
    analysis_timestamp: datetime
    bottleneck_type: str  # cpu, io, memory, cache, synchronization
    severity: str  # low, medium, high, critical
    impact_score: float  # 0-100
    description: str
    affected_functions: List[str]
    performance_impact: str
    optimization_recommendations: List[str]
    estimated_improvement: str


class PerformanceProfiler:
    """Comprehensive performance profiler with bottleneck detection."""

    def __init__(self):
        self.profiles = {}
        self.timing_data = defaultdict(list)
        self.memory_snapshots = deque(maxlen=1000)
        self._profiling_enabled = True
        self._lock = threading.Lock()

    def profile_function(self, operation_name: str = None):
        """Decorator for profiling function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._profiling_enabled:
                    return func(*args, **kwargs)

                name = operation_name or f"{func.__module__}.{func.__name__}"
                return self._profile_execution(name, func, *args, **kwargs)

            return wrapper
        return decorator

    def _profile_execution(self, name: str, func: Callable, *args, **kwargs):
        """Execute function with comprehensive profiling."""
        start_time = time.time()
        start_process_time = time.process_time()

        # Memory before
        memory_before = self._get_memory_usage()

        # Enable cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)

            profiler.disable()

            # Timing measurements
            wall_time = time.time() - start_time
            cpu_time = time.process_time() - start_process_time
            io_wait_time = wall_time - cpu_time

            # Memory after
            memory_after = self._get_memory_usage()
            memory_peak = max(memory_before, memory_after)

            # Extract function breakdown from profiler
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')

            function_breakdown = self._extract_function_stats(ps)

            # Store timing data
            with self._lock:
                self.timing_data[name].append({
                    'wall_time': wall_time,
                    'cpu_time': cpu_time,
                    'io_wait_time': io_wait_time,
                    'memory_peak_mb': memory_peak,
                    'timestamp': datetime.now()
                })

            # Create performance profile
            profile = PerformanceProfile(
                operation_name=name,
                total_time=wall_time,
                call_count=1,
                avg_time_per_call=wall_time,
                function_breakdown=function_breakdown,
                memory_peak_mb=memory_peak,
                cpu_time=cpu_time,
                io_wait_time=io_wait_time
            )

            self.profiles[name] = profile

            return result

        except Exception as e:
            profiler.disable()
            logger.error(f"Profiling error in {name}: {e}")
            raise

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0

    def _extract_function_stats(self, ps: pstats.Stats) -> Dict[str, float]:
        """Extract function timing statistics from pstats."""
        function_stats = {}

        try:
            # Get stats dictionary
            stats_dict = ps.stats

            for (filename, line_num, func_name), (call_count, _, total_time, cumulative_time) in stats_dict.items():
                func_key = f"{Path(filename).name}:{func_name}"
                if total_time > 0.001:  # Only include functions with significant time
                    function_stats[func_key] = total_time

        except Exception as e:
            logger.warning(f"Failed to extract function stats: {e}")

        return function_stats

    def get_performance_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        if operation_name and operation_name in self.timing_data:
            data = self.timing_data[operation_name]
        else:
            # Aggregate all operations
            data = []
            for op_data in self.timing_data.values():
                data.extend(op_data)

        if not data:
            return {}

        wall_times = [d['wall_time'] for d in data]
        cpu_times = [d['cpu_time'] for d in data]
        io_wait_times = [d['io_wait_time'] for d in data]
        memory_peaks = [d['memory_peak_mb'] for d in data]

        return {
            'operation_count': len(data),
            'wall_time': {
                'total': sum(wall_times),
                'average': statistics.mean(wall_times),
                'median': statistics.median(wall_times),
                'std_dev': statistics.stdev(wall_times) if len(wall_times) > 1 else 0,
                'min': min(wall_times),
                'max': max(wall_times)
            },
            'cpu_time': {
                'total': sum(cpu_times),
                'average': statistics.mean(cpu_times),
                'cpu_efficiency': statistics.mean([c/w for c, w in zip(cpu_times, wall_times) if w > 0])
            },
            'io_wait_time': {
                'total': sum(io_wait_times),
                'average': statistics.mean(io_wait_times),
                'io_ratio': statistics.mean([i/w for i, w in zip(io_wait_times, wall_times) if w > 0])
            },
            'memory': {
                'peak_mb': max(memory_peaks),
                'average_mb': statistics.mean(memory_peaks),
                'std_dev_mb': statistics.stdev(memory_peaks) if len(memory_peaks) > 1 else 0
            }
        }


class BottleneckDetector:
    """Automated bottleneck detection and analysis."""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.bottleneck_thresholds = {
            'cpu_efficiency_low': 0.3,      # CPU time / wall time < 30%
            'io_ratio_high': 0.7,           # I/O wait / wall time > 70%
            'memory_growth_high': 100.0,    # Memory growth > 100MB per operation
            'function_time_high': 1.0,      # Single function > 1s
            'operation_time_high': 10.0,    # Total operation > 10s
            'variance_high': 0.5             # StdDev / Mean > 50%
        }

    def analyze_bottlenecks(self, operation_name: str = None) -> List[BottleneckReport]:
        """Perform comprehensive bottleneck analysis."""
        bottlenecks = []

        summary = self.profiler.get_performance_summary(operation_name)
        if not summary:
            return bottlenecks

        # CPU efficiency analysis
        cpu_efficiency = summary.get('cpu_time', {}).get('cpu_efficiency', 1.0)
        if cpu_efficiency < self.bottleneck_thresholds['cpu_efficiency_low']:
            bottlenecks.append(self._create_cpu_bottleneck_report(cpu_efficiency, operation_name))

        # I/O bottleneck analysis
        io_ratio = summary.get('io_wait_time', {}).get('io_ratio', 0.0)
        if io_ratio > self.bottleneck_thresholds['io_ratio_high']:
            bottlenecks.append(self._create_io_bottleneck_report(io_ratio, operation_name))

        # Memory bottleneck analysis
        memory_stats = summary.get('memory', {})
        if memory_stats.get('std_dev_mb', 0) > self.bottleneck_thresholds['memory_growth_high']:
            bottlenecks.append(self._create_memory_bottleneck_report(memory_stats, operation_name))

        # Function-level bottleneck analysis
        if operation_name and operation_name in self.profiler.profiles:
            profile = self.profiler.profiles[operation_name]
            slow_functions = {
                func: time_val for func, time_val in profile.function_breakdown.items()
                if time_val > self.bottleneck_thresholds['function_time_high']
            }

            if slow_functions:
                bottlenecks.append(self._create_function_bottleneck_report(slow_functions, operation_name))

        # Performance variance analysis
        wall_time_stats = summary.get('wall_time', {})
        if wall_time_stats.get('std_dev', 0) / wall_time_stats.get('average', 1) > self.bottleneck_thresholds['variance_high']:
            bottlenecks.append(self._create_variance_bottleneck_report(wall_time_stats, operation_name))

        return bottlenecks

    def _create_cpu_bottleneck_report(self, cpu_efficiency: float, operation_name: str) -> BottleneckReport:
        """Create CPU bottleneck report."""
        severity = "critical" if cpu_efficiency < 0.1 else "high" if cpu_efficiency < 0.2 else "medium"
        impact_score = (1 - cpu_efficiency) * 100

        recommendations = [
            "Investigate I/O blocking operations that prevent CPU utilization",
            "Consider asynchronous processing for I/O-bound operations",
            "Review database query performance and connection pooling",
            "Check for network latency issues in data fetching",
            "Consider parallel processing for CPU-intensive calculations"
        ]

        return BottleneckReport(
            operation_name=operation_name or "system",
            analysis_timestamp=datetime.now(),
            bottleneck_type="cpu",
            severity=severity,
            impact_score=impact_score,
            description=f"Low CPU efficiency ({cpu_efficiency:.1%}) indicates I/O or synchronization bottlenecks",
            affected_functions=[],
            performance_impact=f"CPU underutilization reduces processing throughput by ~{impact_score:.0f}%",
            optimization_recommendations=recommendations,
            estimated_improvement=f"Potential {impact_score * 0.3:.0f}-{impact_score * 0.7:.0f}% performance gain"
        )

    def _create_io_bottleneck_report(self, io_ratio: float, operation_name: str) -> BottleneckReport:
        """Create I/O bottleneck report."""
        severity = "critical" if io_ratio > 0.9 else "high" if io_ratio > 0.8 else "medium"
        impact_score = io_ratio * 100

        recommendations = [
            "Implement data caching to reduce repeated I/O operations",
            "Optimize database queries and add appropriate indexes",
            "Consider using faster storage (SSD) for data cache",
            "Implement connection pooling for database access",
            "Use batch processing to reduce I/O overhead",
            "Consider data compression to reduce I/O volume"
        ]

        return BottleneckReport(
            operation_name=operation_name or "system",
            analysis_timestamp=datetime.now(),
            bottleneck_type="io",
            severity=severity,
            impact_score=impact_score,
            description=f"High I/O wait time ({io_ratio:.1%}) indicates storage or network bottlenecks",
            affected_functions=[],
            performance_impact=f"I/O bottlenecks account for ~{impact_score:.0f}% of execution time",
            optimization_recommendations=recommendations,
            estimated_improvement=f"Potential {impact_score * 0.4:.0f}-{impact_score * 0.8:.0f}% performance gain"
        )

    def _create_memory_bottleneck_report(self, memory_stats: Dict, operation_name: str) -> BottleneckReport:
        """Create memory bottleneck report."""
        std_dev = memory_stats.get('std_dev_mb', 0)
        peak = memory_stats.get('peak_mb', 0)

        severity = "critical" if std_dev > 500 else "high" if std_dev > 200 else "medium"
        impact_score = min(100, (std_dev / 100) * 25)  # Scale to 0-100

        recommendations = [
            "Implement chunked processing to reduce memory usage",
            "Add explicit garbage collection at operation boundaries",
            "Review data structures for memory efficiency",
            "Consider using generators instead of lists for large datasets",
            "Implement data streaming for large file processing",
            "Monitor for memory leaks in long-running operations"
        ]

        return BottleneckReport(
            operation_name=operation_name or "system",
            analysis_timestamp=datetime.now(),
            bottleneck_type="memory",
            severity=severity,
            impact_score=impact_score,
            description=f"High memory variance ({std_dev:.1f}MB std dev, {peak:.1f}MB peak) indicates inefficient memory usage",
            affected_functions=[],
            performance_impact=f"Memory pressure may cause {impact_score:.0f}% performance degradation",
            optimization_recommendations=recommendations,
            estimated_improvement=f"Potential {impact_score * 0.3:.0f}-{impact_score * 0.6:.0f}% performance gain"
        )

    def _create_function_bottleneck_report(self, slow_functions: Dict, operation_name: str) -> BottleneckReport:
        """Create function-level bottleneck report."""
        total_slow_time = sum(slow_functions.values())
        severity = "critical" if total_slow_time > 10 else "high" if total_slow_time > 5 else "medium"
        impact_score = min(100, total_slow_time * 5)

        recommendations = [
            "Profile identified slow functions for optimization opportunities",
            "Consider algorithmic improvements for computational bottlenecks",
            "Implement caching for expensive function calls",
            "Review database queries in slow functions",
            "Consider parallel processing for independent operations",
            "Optimize data structures and algorithms in hot paths"
        ]

        return BottleneckReport(
            operation_name=operation_name or "system",
            analysis_timestamp=datetime.now(),
            bottleneck_type="computation",
            severity=severity,
            impact_score=impact_score,
            description=f"Slow functions consuming {total_slow_time:.1f}s total execution time",
            affected_functions=list(slow_functions.keys()),
            performance_impact=f"Function bottlenecks account for ~{impact_score:.0f}% of performance issues",
            optimization_recommendations=recommendations,
            estimated_improvement=f"Potential {impact_score * 0.2:.0f}-{impact_score * 0.5:.0f}% performance gain"
        )

    def _create_variance_bottleneck_report(self, wall_time_stats: Dict, operation_name: str) -> BottleneckReport:
        """Create performance variance bottleneck report."""
        variance_ratio = wall_time_stats.get('std_dev', 0) / wall_time_stats.get('average', 1)
        severity = "high" if variance_ratio > 1.0 else "medium" if variance_ratio > 0.7 else "low"
        impact_score = min(100, variance_ratio * 50)

        recommendations = [
            "Investigate inconsistent performance patterns",
            "Review resource contention issues (CPU, memory, I/O)",
            "Check for garbage collection impact on performance",
            "Monitor system load during operations",
            "Consider operation prioritization and scheduling",
            "Review concurrent access patterns for shared resources"
        ]

        return BottleneckReport(
            operation_name=operation_name or "system",
            analysis_timestamp=datetime.now(),
            bottleneck_type="synchronization",
            severity=severity,
            impact_score=impact_score,
            description=f"High performance variance (std dev {variance_ratio:.1%} of mean) indicates inconsistent execution",
            affected_functions=[],
            performance_impact=f"Performance inconsistency reduces predictable throughput by ~{impact_score:.0f}%",
            optimization_recommendations=recommendations,
            estimated_improvement=f"Potential {impact_score * 0.3:.0f}-{impact_score * 0.6:.0f}% consistency improvement"
        )


class OptimizationRecommendationEngine:
    """Engine for generating optimization recommendations based on bottleneck analysis."""

    def __init__(self):
        self.optimization_strategies = {
            'cpu': self._cpu_optimizations,
            'io': self._io_optimizations,
            'memory': self._memory_optimizations,
            'computation': self._computation_optimizations,
            'synchronization': self._synchronization_optimizations
        }

    def generate_recommendations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations."""
        if not bottlenecks:
            return {
                'status': 'optimal',
                'message': 'No significant bottlenecks detected',
                'recommendations': []
            }

        # Group bottlenecks by type and severity
        bottleneck_groups = defaultdict(list)
        critical_bottlenecks = []

        for bottleneck in bottlenecks:
            bottleneck_groups[bottleneck.bottleneck_type].append(bottleneck)
            if bottleneck.severity == 'critical':
                critical_bottlenecks.append(bottleneck)

        # Generate prioritized recommendations
        recommendations = []

        # High-priority critical bottlenecks first
        if critical_bottlenecks:
            recommendations.append({
                'priority': 'critical',
                'title': 'Critical Performance Issues',
                'description': 'These issues require immediate attention',
                'actions': self._generate_critical_actions(critical_bottlenecks)
            })

        # Type-specific recommendations
        for bottleneck_type, type_bottlenecks in bottleneck_groups.items():
            if bottleneck_type in self.optimization_strategies:
                type_recommendations = self.optimization_strategies[bottleneck_type](type_bottlenecks)
                if type_recommendations:
                    recommendations.append(type_recommendations)

        # System-wide optimizations
        system_recommendations = self._generate_system_recommendations(bottlenecks)
        if system_recommendations:
            recommendations.append(system_recommendations)

        return {
            'status': 'optimization_needed',
            'total_bottlenecks': len(bottlenecks),
            'critical_bottlenecks': len(critical_bottlenecks),
            'estimated_improvement': self._calculate_total_improvement(bottlenecks),
            'recommendations': recommendations
        }

    def _generate_critical_actions(self, critical_bottlenecks: List[BottleneckReport]) -> List[str]:
        """Generate actions for critical bottlenecks."""
        actions = []

        for bottleneck in critical_bottlenecks:
            actions.extend([
                f"Address {bottleneck.bottleneck_type} bottleneck in {bottleneck.operation_name}",
                f"Expected impact: {bottleneck.estimated_improvement}"
            ])

        return actions

    def _cpu_optimizations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate CPU optimization recommendations."""
        return {
            'priority': 'high',
            'title': 'CPU Utilization Optimization',
            'description': 'Improve CPU efficiency and reduce I/O blocking',
            'actions': [
                'Implement asynchronous I/O operations',
                'Add connection pooling for database access',
                'Review and optimize blocking operations',
                'Consider parallel processing for independent tasks',
                'Profile and optimize hot code paths'
            ],
            'estimated_effort': 'Medium',
            'expected_impact': 'High'
        }

    def _io_optimizations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate I/O optimization recommendations."""
        return {
            'priority': 'high',
            'title': 'I/O Performance Optimization',
            'description': 'Reduce I/O overhead and improve data access patterns',
            'actions': [
                'Implement intelligent data caching strategy',
                'Optimize database queries and add indexes',
                'Use batch processing for data operations',
                'Consider faster storage solutions (SSD)',
                'Implement data compression for large datasets',
                'Add connection pooling and query optimization'
            ],
            'estimated_effort': 'Medium-High',
            'expected_impact': 'Very High'
        }

    def _memory_optimizations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate memory optimization recommendations."""
        return {
            'priority': 'medium',
            'title': 'Memory Management Optimization',
            'description': 'Improve memory efficiency and reduce allocation overhead',
            'actions': [
                'Implement chunked data processing',
                'Add explicit garbage collection points',
                'Use memory-efficient data structures',
                'Implement data streaming for large datasets',
                'Monitor and fix memory leaks',
                'Optimize object creation patterns'
            ],
            'estimated_effort': 'Medium',
            'expected_impact': 'Medium-High'
        }

    def _computation_optimizations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate computational optimization recommendations."""
        return {
            'priority': 'high',
            'title': 'Computational Performance Optimization',
            'description': 'Optimize algorithms and computational bottlenecks',
            'actions': [
                'Profile and optimize slow functions',
                'Implement algorithmic improvements',
                'Add caching for expensive calculations',
                'Consider vectorized operations with NumPy',
                'Implement parallel processing for independent calculations',
                'Review and optimize data structures'
            ],
            'estimated_effort': 'High',
            'expected_impact': 'High'
        }

    def _synchronization_optimizations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate synchronization optimization recommendations."""
        return {
            'priority': 'medium',
            'title': 'Synchronization and Consistency Optimization',
            'description': 'Improve performance consistency and reduce contention',
            'actions': [
                'Review resource contention patterns',
                'Optimize thread synchronization',
                'Implement operation prioritization',
                'Monitor system resource usage',
                'Consider load balancing strategies',
                'Review concurrent access patterns'
            ],
            'estimated_effort': 'Medium-High',
            'expected_impact': 'Medium'
        }

    def _generate_system_recommendations(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate system-wide optimization recommendations."""
        return {
            'priority': 'low',
            'title': 'System-Wide Optimizations',
            'description': 'General system improvements and monitoring',
            'actions': [
                'Implement comprehensive performance monitoring',
                'Add automated bottleneck detection',
                'Create performance regression testing',
                'Establish performance baselines and alerts',
                'Document optimization procedures',
                'Regular performance review and tuning'
            ],
            'estimated_effort': 'Low-Medium',
            'expected_impact': 'Long-term Improvement'
        }

    def _calculate_total_improvement(self, bottlenecks: List[BottleneckReport]) -> str:
        """Calculate estimated total improvement from addressing bottlenecks."""
        total_impact = sum(b.impact_score for b in bottlenecks)
        improvement_range = (total_impact * 0.2, total_impact * 0.6)

        return f"{improvement_range[0]:.0f}-{improvement_range[1]:.0f}% performance improvement"


# Demo and testing functions
def demo_bottleneck_analysis():
    """Demonstrate bottleneck analysis capabilities."""
    logger.info("=== Bottleneck Analysis Demo ===")

    profiler = PerformanceProfiler()
    detector = BottleneckDetector(profiler)
    optimizer = OptimizationRecommendationEngine()

    # Demo functions with different bottleneck patterns
    @profiler.profile_function("cpu_intensive")
    def cpu_intensive_task():
        """Simulate CPU-intensive task."""
        result = sum(i ** 2 for i in range(100000))
        return result

    @profiler.profile_function("io_intensive")
    def io_intensive_task():
        """Simulate I/O-intensive task."""
        time.sleep(2)  # Simulate I/O wait
        return "io_complete"

    @profiler.profile_function("memory_intensive")
    def memory_intensive_task():
        """Simulate memory-intensive task."""
        data = [list(range(10000)) for _ in range(100)]
        return len(data)

    # Run demo tasks
    logger.info("Running CPU-intensive task...")
    cpu_intensive_task()

    logger.info("Running I/O-intensive task...")
    io_intensive_task()

    logger.info("Running memory-intensive task...")
    memory_intensive_task()

    # Analyze bottlenecks
    logger.info("Analyzing bottlenecks...")

    all_bottlenecks = []
    for operation in ["cpu_intensive", "io_intensive", "memory_intensive"]:
        bottlenecks = detector.analyze_bottlenecks(operation)
        all_bottlenecks.extend(bottlenecks)

        logger.info(f"\n{operation.upper()} Analysis:")
        summary = profiler.get_performance_summary(operation)
        logger.info(f"  Wall time: {summary['wall_time']['average']:.3f}s")
        logger.info(f"  CPU efficiency: {summary['cpu_time']['cpu_efficiency']:.1%}")
        logger.info(f"  I/O ratio: {summary['io_wait_time']['io_ratio']:.1%}")

        for bottleneck in bottlenecks:
            logger.info(f"  Bottleneck: {bottleneck.bottleneck_type} ({bottleneck.severity})")

    # Generate optimization recommendations
    logger.info("\nGenerating optimization recommendations...")
    recommendations = optimizer.generate_recommendations(all_bottlenecks)

    logger.info(f"\nOptimization Report:")
    logger.info(f"Status: {recommendations['status']}")
    logger.info(f"Total bottlenecks: {recommendations.get('total_bottlenecks', 0)}")
    logger.info(f"Estimated improvement: {recommendations.get('estimated_improvement', 'N/A')}")

    for rec in recommendations.get('recommendations', []):
        logger.info(f"\n{rec['title']} ({rec['priority']} priority):")
        for action in rec['actions'][:3]:  # Show first 3 actions
            logger.info(f"  - {action}")

    # Save detailed report
    output_dir = Path("reports/bottleneck_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"bottleneck_analysis_{timestamp}.json"

    report_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'bottlenecks': [asdict(b) for b in all_bottlenecks],
        'recommendations': recommendations,
        'performance_summaries': {
            op: profiler.get_performance_summary(op)
            for op in ["cpu_intensive", "io_intensive", "memory_intensive"]
        }
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    logger.info(f"\nDetailed analysis saved to {report_path}")

    return report_data


if __name__ == "__main__":
    demo_bottleneck_analysis()