#!/usr/bin/env python3
"""
Master Integration Test Suite Runner - Orchestrated Comprehensive System Validation
?????????????????????????????? - ???????????????????????????

This is the master test orchestrator that coordinates all integration test suites:
- Complete system integration testing
- AI/ML system integration validation
- Trading system integration testing
- Monitoring system integration validation
- System management integration testing
- Parallel test execution with intelligent coordination
- Comprehensive reporting and analysis
- Performance benchmarking and validation
- Compliance reporting for audit requirements

Test Suite Orchestration:
- Sequential dependency management between test suites
- Parallel execution of independent test modules
- Resource utilization optimization during testing
- Professional test documentation and audit trails
- Failure analysis and debugging support
- Performance metrics collection and analysis
"""

import os
import sys
import asyncio
import logging
import time
import json
import traceback
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import threading
from collections import defaultdict
import platform

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
import warnings
warnings.filterwarnings('ignore')

# Configure master logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('master_integration_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestSuiteResult:
    """Test suite result data structure"""
    suite_name: str
    test_file: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    return_code: int = -1
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    success_rate: float = 0.0
    status: str = "PENDING"
    output: str = ""
    error_output: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance metrics during testing"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    active_processes: int = 0
    open_files: int = 0

class MasterIntegrationTestRunner:
    """
    Master orchestrator for all integration test suites.
    Coordinates execution, monitors resources, and generates comprehensive reports.
    """

    def __init__(self):
        self.test_start_time = datetime.now()
        self.test_results: List[TestSuiteResult] = []
        self.system_metrics: List[SystemMetrics] = []

        # Test configuration
        self.max_parallel_tests = min(4, os.cpu_count() or 4)  # Max 4 parallel tests
        self.individual_test_timeout = 1800.0  # 30 minutes per test suite
        self.total_test_timeout = 7200.0  # 2 hours total

        # Test suite definitions with dependencies
        self.test_suites = {
            'complete_system_integration': {
                'file': 'test_complete_system_integration.py',
                'description': 'Complete End-to-End System Integration',
                'priority': 1,  # Highest priority
                'dependencies': [],
                'timeout': 1800,
                'critical': True,
                'parallel_safe': False  # Should run alone initially
            },
            'ai_ml_integration': {
                'file': 'test_ai_ml_integration.py',
                'description': 'AI/ML System Integration',
                'priority': 2,
                'dependencies': ['complete_system_integration'],
                'timeout': 1200,
                'critical': True,
                'parallel_safe': True
            },
            'trading_system_integration': {
                'file': 'test_trading_system_integration.py',
                'description': 'Trading System Integration',
                'priority': 2,
                'dependencies': ['complete_system_integration'],
                'timeout': 1200,
                'critical': True,
                'parallel_safe': True
            },
            'monitoring_system_integration': {
                'file': 'test_monitoring_system_integration.py',
                'description': 'Monitoring System Integration',
                'priority': 3,
                'dependencies': ['complete_system_integration'],
                'timeout': 900,
                'critical': False,
                'parallel_safe': True
            },
            'system_management_integration': {
                'file': 'test_system_management_integration.py',
                'description': 'System Management Integration',
                'priority': 3,
                'dependencies': ['complete_system_integration'],
                'timeout': 900,
                'critical': False,
                'parallel_safe': True
            }
        }

        # Results and reporting
        self.report_path = Path("integration_test_results")
        self.report_path.mkdir(exist_ok=True)

        logger.info("Master Integration Test Runner Initialized")
        logger.info(f"Max parallel tests: {self.max_parallel_tests}")
        logger.info(f"Total test suites: {len(self.test_suites)}")

    async def run_all_integration_test_suites(self) -> bool:
        """
        Run all integration test suites with intelligent orchestration.
        Returns True if all critical tests pass.
        """
        logger.info("=" * 100)
        logger.info("MASTER INTEGRATION TEST SUITE EXECUTION")
        logger.info("Comprehensive Professional Trading System Validation")
        logger.info("=" * 100)
        logger.info(f"Test Start Time: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info("=" * 100)

        try:
            # Start system monitoring
            monitoring_task = asyncio.create_task(self.monitor_system_resources())

            # Pre-test system validation
            pre_test_valid = await self.validate_test_environment()
            if not pre_test_valid:
                logger.error("??? Pre-test environment validation failed")
                return False

            # Execute test suites with intelligent orchestration
            execution_success = await self.execute_test_suites_orchestrated()

            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

            # Generate comprehensive reports
            await self.generate_master_test_report()

            # Print final summary
            await self.print_final_summary()

            return execution_success

        except Exception as e:
            logger.error(f"Master test execution failed: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return False

        finally:
            # Cleanup
            await self.cleanup_test_environment()

    async def validate_test_environment(self) -> bool:
        """Validate test environment before execution."""
        try:
            logger.info("???? Validating test environment...")

            validation_results = {}

            # Check system resources
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')

                # Minimum requirements for integration testing
                min_memory_gb = 4.0
                min_disk_gb = 10.0

                available_memory_gb = memory.available / (1024**3)
                available_disk_gb = disk.free / (1024**3)

                memory_ok = available_memory_gb >= min_memory_gb
                disk_ok = available_disk_gb >= min_disk_gb

                validation_results['system_resources'] = {
                    'memory_ok': memory_ok,
                    'disk_ok': disk_ok,
                    'memory_available_gb': round(available_memory_gb, 2),
                    'disk_available_gb': round(available_disk_gb, 2),
                    'cpu_count': psutil.cpu_count()
                }

                if memory_ok and disk_ok:
                    logger.info(f"??? System resources adequate: {available_memory_gb:.1f}GB RAM, {available_disk_gb:.1f}GB disk")
                else:
                    logger.warning(f"?????? System resources may be insufficient: {available_memory_gb:.1f}GB RAM, {available_disk_gb:.1f}GB disk")

            except Exception as e:
                logger.warning(f"System resource check failed: {e}")
                validation_results['system_resources'] = {'error': str(e)}

            # Check test files availability
            try:
                test_files_available = {}
                for suite_name, suite_config in self.test_suites.items():
                    test_file = Path(suite_config['file'])
                    available = test_file.exists() and test_file.is_file()
                    test_files_available[suite_name] = available

                    if available:
                        logger.info(f"??? Test suite available: {suite_name}")
                    else:
                        logger.warning(f"??? Test suite missing: {suite_name} ({suite_config['file']})")

                available_count = sum(test_files_available.values())
                total_count = len(test_files_available)

                validation_results['test_files'] = {
                    'available': available_count,
                    'total': total_count,
                    'availability_rate': (available_count / total_count) * 100,
                    'files': test_files_available
                }

            except Exception as e:
                logger.warning(f"Test file availability check failed: {e}")
                validation_results['test_files'] = {'error': str(e)}

            # Check Python environment
            try:
                python_version = sys.version_info
                python_ok = python_version.major == 3 and python_version.minor >= 8

                validation_results['python_environment'] = {
                    'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    'compatible': python_ok,
                    'executable': sys.executable
                }

                if python_ok:
                    logger.info(f"??? Python version compatible: {python_version.major}.{python_version.minor}.{python_version.micro}")
                else:
                    logger.warning(f"?????? Python version may be incompatible: {python_version.major}.{python_version.minor}.{python_version.micro}")

            except Exception as e:
                logger.warning(f"Python environment check failed: {e}")
                validation_results['python_environment'] = {'error': str(e)}

            # Calculate overall validation score
            validation_scores = []

            # System resources score
            if validation_results.get('system_resources', {}).get('memory_ok', False) and \
               validation_results.get('system_resources', {}).get('disk_ok', False):
                validation_scores.append(1.0)
            else:
                validation_scores.append(0.5)  # Partial credit

            # Test files availability score
            test_files_rate = validation_results.get('test_files', {}).get('availability_rate', 0) / 100
            validation_scores.append(test_files_rate)

            # Python environment score
            if validation_results.get('python_environment', {}).get('compatible', False):
                validation_scores.append(1.0)
            else:
                validation_scores.append(0.0)

            overall_validation_score = (sum(validation_scores) / len(validation_scores)) * 100

            logger.info(f"???? Environment validation score: {overall_validation_score:.1f}%")

            # Minimum 70% validation score required
            return overall_validation_score >= 70.0

        except Exception as e:
            logger.error(f"Test environment validation failed: {e}")
            return False

    async def execute_test_suites_orchestrated(self) -> bool:
        """Execute test suites with intelligent orchestration."""
        try:
            logger.info("???? Executing test suites with intelligent orchestration...")

            # Group test suites by priority and dependencies
            execution_phases = self.plan_execution_phases()

            overall_success = True
            critical_failures = []

            for phase_num, phase_suites in execution_phases.items():
                logger.info(f"\n???? Executing Phase {phase_num}: {len(phase_suites)} test suites")

                # Execute phase
                phase_success = await self.execute_phase(phase_num, phase_suites)

                if not phase_success:
                    # Check for critical failures
                    phase_critical_failures = [
                        suite_name for suite_name in phase_suites
                        if self.test_suites[suite_name].get('critical', False)
                        and any(r.suite_name == suite_name and r.status == 'FAILED' for r in self.test_results)
                    ]

                    critical_failures.extend(phase_critical_failures)

                    if phase_critical_failures:
                        logger.error(f"??? Critical failures in Phase {phase_num}: {phase_critical_failures}")
                        overall_success = False

                        # Decide whether to continue
                        if self.should_stop_on_critical_failure(phase_critical_failures):
                            logger.error("???? Stopping execution due to critical failures")
                            break

                logger.info(f"??? Phase {phase_num} completed")

            # Final success determination
            if critical_failures:
                logger.error(f"??? Integration tests failed due to critical failures: {critical_failures}")
                overall_success = False
            else:
                # Check overall success rate
                total_tests = len(self.test_results)
                passed_tests = len([r for r in self.test_results if r.status == 'PASSED'])
                success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

                if success_rate >= 85.0:  # 85% minimum success rate
                    logger.info(f"??? Integration tests passed: {success_rate:.1f}% success rate")
                    overall_success = True
                else:
                    logger.error(f"??? Integration tests failed: {success_rate:.1f}% success rate below 85%")
                    overall_success = False

            return overall_success

        except Exception as e:
            logger.error(f"Test suite orchestration failed: {e}")
            return False

    def plan_execution_phases(self) -> Dict[int, List[str]]:
        """Plan test suite execution phases based on dependencies and priorities."""
        phases = defaultdict(list)

        # Sort by priority first
        sorted_suites = sorted(self.test_suites.items(), key=lambda x: x[1]['priority'])

        # Phase 1: High priority suites with no dependencies or only completed dependencies
        for suite_name, suite_config in sorted_suites:
            if suite_config['priority'] == 1:
                phases[1].append(suite_name)

        # Phase 2: Medium priority suites
        for suite_name, suite_config in sorted_suites:
            if suite_config['priority'] == 2:
                phases[2].append(suite_name)

        # Phase 3: Lower priority suites
        for suite_name, suite_config in sorted_suites:
            if suite_config['priority'] >= 3:
                phases[3].append(suite_name)

        return dict(phases)

    async def execute_phase(self, phase_num: int, suite_names: List[str]) -> bool:
        """Execute a specific phase of test suites."""
        try:
            logger.info(f"?????? Starting Phase {phase_num} with {len(suite_names)} suites")

            # Determine execution strategy
            if phase_num == 1:
                # Phase 1: Sequential execution for critical dependencies
                for suite_name in suite_names:
                    await self.execute_single_test_suite(suite_name)
            else:
                # Phase 2+: Parallel execution where safe
                parallel_suites = [s for s in suite_names if self.test_suites[s].get('parallel_safe', True)]
                sequential_suites = [s for s in suite_names if not self.test_suites[s].get('parallel_safe', True)]

                # Run sequential suites first
                for suite_name in sequential_suites:
                    await self.execute_single_test_suite(suite_name)

                # Run parallel suites
                if parallel_suites:
                    await self.execute_parallel_test_suites(parallel_suites)

            # Check phase success
            phase_results = [r for r in self.test_results if r.suite_name in suite_names]
            phase_success = all(r.status == 'PASSED' for r in phase_results)

            logger.info(f"???? Phase {phase_num} completed: {len(phase_results)} suites, success: {phase_success}")

            return phase_success

        except Exception as e:
            logger.error(f"Phase {phase_num} execution failed: {e}")
            return False

    async def execute_single_test_suite(self, suite_name: str) -> TestSuiteResult:
        """Execute a single test suite."""
        suite_config = self.test_suites[suite_name]

        result = TestSuiteResult(
            suite_name=suite_name,
            test_file=suite_config['file'],
            start_time=datetime.now()
        )

        try:
            logger.info(f"???? Starting test suite: {suite_name}")
            logger.info(f"   File: {suite_config['file']}")
            logger.info(f"   Description: {suite_config['description']}")

            # Execute test suite
            process = await asyncio.create_subprocess_exec(
                sys.executable, suite_config['file'],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            try:
                # Wait for completion with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=suite_config.get('timeout', self.individual_test_timeout)
                )

                result.end_time = datetime.now()
                result.duration = (result.end_time - result.start_time).total_seconds()
                result.return_code = process.returncode
                result.output = stdout.decode('utf-8') if stdout else ""
                result.error_output = stderr.decode('utf-8') if stderr else ""

                # Parse test results from output
                await self.parse_test_results(result)

                # Determine status
                if result.return_code == 0:
                    result.status = 'PASSED'
                    logger.info(f"??? {suite_name} PASSED ({result.duration:.1f}s)")
                else:
                    result.status = 'FAILED'
                    logger.error(f"??? {suite_name} FAILED ({result.duration:.1f}s)")

            except asyncio.TimeoutError:
                # Kill the process
                process.kill()
                await process.wait()

                result.end_time = datetime.now()
                result.duration = (result.end_time - result.start_time).total_seconds()
                result.return_code = -1
                result.status = 'TIMEOUT'
                result.error_output = f"Test suite timed out after {suite_config.get('timeout', self.individual_test_timeout)} seconds"

                logger.error(f"?????? {suite_name} TIMEOUT ({result.duration:.1f}s)")

        except Exception as e:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.return_code = -1
            result.status = 'ERROR'
            result.error_output = f"Test suite execution error: {str(e)}"

            logger.error(f"???? {suite_name} ERROR ({result.duration:.1f}s): {e}")

        # Add to results
        self.test_results.append(result)

        return result

    async def execute_parallel_test_suites(self, suite_names: List[str]) -> List[TestSuiteResult]:
        """Execute multiple test suites in parallel."""
        logger.info(f"??? Executing {len(suite_names)} test suites in parallel")

        # Limit parallel execution
        semaphore = asyncio.Semaphore(min(self.max_parallel_tests, len(suite_names)))

        async def execute_with_semaphore(suite_name):
            async with semaphore:
                return await self.execute_single_test_suite(suite_name)

        # Execute all suites concurrently
        tasks = [execute_with_semaphore(suite_name) for suite_name in suite_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel execution error: {result}")
            else:
                successful_results.append(result)

        return successful_results

    async def parse_test_results(self, result: TestSuiteResult):
        """Parse test results from output."""
        try:
            output_lines = result.output.split('\n')

            # Look for common test result patterns
            for line in output_lines:
                if 'PASSED' in line and 'FAILED' in line:
                    # Try to extract numbers
                    words = line.split()
                    for i, word in enumerate(words):
                        if 'Passed:' in word or 'PASSED' in word:
                            try:
                                # Look for number after "Passed:"
                                if '/' in words[i+1]:
                                    result.tests_passed = int(words[i+1].split('/')[0])
                                    total = int(words[i+1].split('/')[1])
                                    result.tests_failed = total - result.tests_passed
                            except (IndexError, ValueError):
                                pass

                elif 'Success Rate:' in line:
                    try:
                        # Extract success rate
                        rate_str = line.split('Success Rate:')[1].strip().replace('%', '')
                        result.success_rate = float(rate_str)
                    except (IndexError, ValueError):
                        pass

            # Calculate success rate if not found
            if result.success_rate == 0.0 and (result.tests_passed > 0 or result.tests_failed > 0):
                total_tests = result.tests_passed + result.tests_failed + result.tests_errors
                if total_tests > 0:
                    result.success_rate = (result.tests_passed / total_tests) * 100

        except Exception as e:
            logger.warning(f"Failed to parse test results for {result.suite_name}: {e}")

    def should_stop_on_critical_failure(self, critical_failures: List[str]) -> bool:
        """Determine if execution should stop based on critical failures."""
        # Stop if complete system integration fails
        if 'complete_system_integration' in critical_failures:
            return True

        # Stop if more than 1 critical suite fails
        if len(critical_failures) > 1:
            return True

        return False

    async def monitor_system_resources(self):
        """Monitor system resources during test execution."""
        try:
            while True:
                try:
                    metrics = SystemMetrics(
                        timestamp=datetime.now(),
                        cpu_usage=psutil.cpu_percent(),
                        memory_usage=psutil.virtual_memory().percent,
                        disk_usage=psutil.disk_usage('.').used / psutil.disk_usage('.').total * 100,
                        active_processes=len(psutil.pids()),
                    )

                    try:
                        metrics.open_files = psutil.Process().num_fds() if hasattr(psutil.Process(), 'num_fds') else 0
                    except:
                        metrics.open_files = 0

                    self.system_metrics.append(metrics)

                    # Log resource warnings
                    if metrics.cpu_usage > 90:
                        logger.warning(f"?????? High CPU usage: {metrics.cpu_usage:.1f}%")
                    if metrics.memory_usage > 90:
                        logger.warning(f"?????? High memory usage: {metrics.memory_usage:.1f}%")

                except Exception as e:
                    logger.debug(f"Resource monitoring error: {e}")

                await asyncio.sleep(10)  # Monitor every 10 seconds

        except asyncio.CancelledError:
            logger.info("Resource monitoring stopped")

    async def generate_master_test_report(self):
        """Generate comprehensive master test report."""
        try:
            report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.report_path / f"master_integration_test_report_{report_timestamp}.json"

            # Calculate comprehensive statistics
            total_duration = (datetime.now() - self.test_start_time).total_seconds()
            total_suites = len(self.test_results)
            passed_suites = len([r for r in self.test_results if r.status == 'PASSED'])
            failed_suites = len([r for r in self.test_results if r.status == 'FAILED'])
            error_suites = len([r for r in self.test_results if r.status == 'ERROR'])
            timeout_suites = len([r for r in self.test_results if r.status == 'TIMEOUT'])

            overall_success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0

            # System resource statistics
            if self.system_metrics:
                avg_cpu = sum(m.cpu_usage for m in self.system_metrics) / len(self.system_metrics)
                max_cpu = max(m.cpu_usage for m in self.system_metrics)
                avg_memory = sum(m.memory_usage for m in self.system_metrics) / len(self.system_metrics)
                max_memory = max(m.memory_usage for m in self.system_metrics)
            else:
                avg_cpu = max_cpu = avg_memory = max_memory = 0

            # Generate comprehensive report
            report = {
                'test_run_info': {
                    'start_time': self.test_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_duration_seconds': total_duration,
                    'platform': platform.platform(),
                    'python_version': sys.version,
                    'hostname': platform.node(),
                    'test_runner_version': '1.0.0'
                },
                'execution_summary': {
                    'total_test_suites': total_suites,
                    'passed_suites': passed_suites,
                    'failed_suites': failed_suites,
                    'error_suites': error_suites,
                    'timeout_suites': timeout_suites,
                    'overall_success_rate': round(overall_success_rate, 2),
                    'critical_failures': [r.suite_name for r in self.test_results
                                        if r.status in ['FAILED', 'ERROR'] and
                                        self.test_suites.get(r.suite_name, {}).get('critical', False)]
                },
                'performance_metrics': {
                    'average_cpu_usage': round(avg_cpu, 2),
                    'maximum_cpu_usage': round(max_cpu, 2),
                    'average_memory_usage': round(avg_memory, 2),
                    'maximum_memory_usage': round(max_memory, 2),
                    'resource_samples': len(self.system_metrics)
                },
                'test_suite_results': [
                    {
                        'suite_name': result.suite_name,
                        'test_file': result.test_file,
                        'status': result.status,
                        'duration_seconds': round(result.duration, 2),
                        'tests_passed': result.tests_passed,
                        'tests_failed': result.tests_failed,
                        'tests_errors': result.tests_errors,
                        'success_rate': round(result.success_rate, 2),
                        'return_code': result.return_code,
                        'critical': self.test_suites.get(result.suite_name, {}).get('critical', False),
                        'start_time': result.start_time.isoformat(),
                        'end_time': result.end_time.isoformat() if result.end_time else None
                    }
                    for result in self.test_results
                ],
                'system_metrics_summary': [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'cpu_usage': round(metric.cpu_usage, 2),
                        'memory_usage': round(metric.memory_usage, 2),
                        'disk_usage': round(metric.disk_usage, 2),
                        'active_processes': metric.active_processes
                    }
                    for metric in self.system_metrics[-10:]  # Last 10 samples
                ],
                'recommendations': self.generate_recommendations(),
                'detailed_outputs': {
                    result.suite_name: {
                        'stdout': result.output[-2000:],  # Last 2000 characters
                        'stderr': result.error_output[-2000:] if result.error_output else ""
                    }
                    for result in self.test_results
                }
            }

            # Save report
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"???? Master test report generated: {report_file}")

            # Also generate a summary report
            await self.generate_summary_report(report, report_timestamp)

        except Exception as e:
            logger.error(f"Failed to generate master test report: {e}")

    async def generate_summary_report(self, full_report: Dict, timestamp: str):
        """Generate executive summary report."""
        try:
            summary_file = self.report_path / f"integration_test_executive_summary_{timestamp}.md"

            summary_content = f"""
# Integration Test Executive Summary

**Test Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Duration:** {full_report['test_run_info']['total_duration_seconds'] / 60:.1f} minutes
**Platform:** {full_report['test_run_info']['platform']}

## Overall Results

- **Total Test Suites:** {full_report['execution_summary']['total_test_suites']}
- **Success Rate:** {full_report['execution_summary']['overall_success_rate']:.1f}%
- **Passed:** {full_report['execution_summary']['passed_suites']}
- **Failed:** {full_report['execution_summary']['failed_suites']}
- **Errors:** {full_report['execution_summary']['error_suites']}
- **Timeouts:** {full_report['execution_summary']['timeout_suites']}

## Test Suite Status

"""

            for result in full_report['test_suite_results']:
                status_emoji = {
                    'PASSED': '???',
                    'FAILED': '???',
                    'ERROR': '????',
                    'TIMEOUT': '??????'
                }.get(result['status'], '???')

                critical_marker = " (CRITICAL)" if result['critical'] else ""

                summary_content += f"- {status_emoji} **{result['suite_name']}**{critical_marker}: {result['duration_seconds']:.1f}s\n"

            summary_content += f"""

## Performance Metrics

- **Average CPU Usage:** {full_report['performance_metrics']['average_cpu_usage']:.1f}%
- **Maximum CPU Usage:** {full_report['performance_metrics']['maximum_cpu_usage']:.1f}%
- **Average Memory Usage:** {full_report['performance_metrics']['average_memory_usage']:.1f}%
- **Maximum Memory Usage:** {full_report['performance_metrics']['maximum_memory_usage']:.1f}%

## Recommendations

"""

            for recommendation in full_report['recommendations']:
                summary_content += f"- {recommendation}\n"

            summary_content += f"""

## Critical Issues

"""
            critical_failures = full_report['execution_summary']['critical_failures']
            if critical_failures:
                for failure in critical_failures:
                    summary_content += f"- **CRITICAL:** {failure} test suite failed\n"
            else:
                summary_content += "- No critical issues detected\n"

            # Save summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)

            logger.info(f"???? Executive summary generated: {summary_file}")

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Analyze failures
        failed_results = [r for r in self.test_results if r.status in ['FAILED', 'ERROR']]
        if failed_results:
            recommendations.append(f"Address {len(failed_results)} failed test suites before production deployment")

        # Analyze critical failures
        critical_failures = [r for r in failed_results if self.test_suites.get(r.suite_name, {}).get('critical', False)]
        if critical_failures:
            recommendations.append(f"URGENT: {len(critical_failures)} critical test suites failed - system not ready for production")

        # Analyze performance
        if self.system_metrics:
            max_cpu = max(m.cpu_usage for m in self.system_metrics)
            max_memory = max(m.memory_usage for m in self.system_metrics)

            if max_cpu > 90:
                recommendations.append("High CPU usage detected during testing - consider performance optimization")
            if max_memory > 90:
                recommendations.append("High memory usage detected - monitor memory usage in production")

        # Analyze timeouts
        timeout_results = [r for r in self.test_results if r.status == 'TIMEOUT']
        if timeout_results:
            recommendations.append(f"{len(timeout_results)} test suites timed out - investigate performance issues")

        # Success recommendations
        if not failed_results:
            recommendations.append("All integration tests passed - system ready for production deployment")

        overall_success_rate = len([r for r in self.test_results if r.status == 'PASSED']) / len(self.test_results) * 100
        if overall_success_rate >= 95:
            recommendations.append("Excellent test coverage and success rate - system demonstrates high quality")
        elif overall_success_rate >= 85:
            recommendations.append("Good test success rate - monitor failed tests for improvements")

        return recommendations

    async def print_final_summary(self):
        """Print final test execution summary."""
        try:
            total_duration = (datetime.now() - self.test_start_time).total_seconds()
            total_suites = len(self.test_results)
            passed_suites = len([r for r in self.test_results if r.status == 'PASSED'])
            failed_suites = len([r for r in self.test_results if r.status == 'FAILED'])
            error_suites = len([r for r in self.test_results if r.status == 'ERROR'])
            timeout_suites = len([r for r in self.test_results if r.status == 'TIMEOUT'])

            overall_success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0

            critical_failures = [r.suite_name for r in self.test_results
                               if r.status in ['FAILED', 'ERROR'] and
                               self.test_suites.get(r.suite_name, {}).get('critical', False)]

            print("\n" + "=" * 100)
            print("???? MASTER INTEGRATION TEST EXECUTION COMPLETE")
            print("=" * 100)
            print(f"???? EXECUTION SUMMARY")
            print(f"   Total Duration: {total_duration / 60:.1f} minutes")
            print(f"   Test Suites: {total_suites}")
            print(f"   Success Rate: {overall_success_rate:.1f}%")
            print("")
            print(f"???? DETAILED RESULTS")
            print(f"   ??? Passed: {passed_suites}")
            print(f"   ??? Failed: {failed_suites}")
            print(f"   ???? Errors: {error_suites}")
            print(f"   ?????? Timeouts: {timeout_suites}")
            print("")

            if critical_failures:
                print(f"???? CRITICAL FAILURES")
                for failure in critical_failures:
                    print(f"   ??? {failure}")
                print("")

            print(f"???? FINAL VERDICT")
            if not critical_failures and overall_success_rate >= 85:
                print("   ???? INTEGRATION TESTS PASSED")
                print("   ??? System is ready for professional trading operations")
            elif not critical_failures and overall_success_rate >= 70:
                print("   ??????  INTEGRATION TESTS PARTIALLY SUCCESSFUL")
                print("   ???? Review failed tests but system may be acceptable for production")
            else:
                print("   ??? INTEGRATION TESTS FAILED")
                print("   ???? System requires significant attention before production deployment")

            print("=" * 100)

        except Exception as e:
            logger.error(f"Failed to print final summary: {e}")

    async def cleanup_test_environment(self):
        """Cleanup test environment."""
        try:
            logger.info("???? Cleaning up test environment...")

            # Add any cleanup logic here
            # For example: temporary files, test databases, etc.

            logger.info("??? Test environment cleanup completed")

        except Exception as e:
            logger.warning(f"Test environment cleanup failed: {e}")


async def main():
    """Main entry point for master integration test runner."""
    print("???? QUANTITATIVE TRADING SYSTEM")
    print("???? MASTER INTEGRATION TEST SUITE")
    print("=" * 100)
    print("Professional Trading System Comprehensive Validation")
    print(f"Test Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    try:
        # Initialize and run master test suite
        runner = MasterIntegrationTestRunner()
        success = await runner.run_all_integration_test_suites()

        if success:
            print("\n???? MASTER INTEGRATION TESTS PASSED!")
            print("??? Complete quantitative trading system validated for production")
            return 0
        else:
            print("\n??? MASTER INTEGRATION TESTS FAILED!")
            print("?????? System requires attention before production deployment")
            return 1

    except KeyboardInterrupt:
        print("\n?????? Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Master integration test runner failed: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        print(f"\n???? MASTER TEST RUNNER ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))