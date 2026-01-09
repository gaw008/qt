#!/usr/bin/env python3
"""
Execution Robustness Framework

Provides robust execution capabilities for trading operations including:
- Intelligent retry mechanisms
- Execution monitoring and validation
- Error handling and recovery
- Performance tracking and optimization
- Circuit breaker patterns for system protection

This framework ensures reliable trade execution even under adverse market conditions.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status types"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class FailureType(Enum):
    """Types of execution failures"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    MARKET_CLOSED = "market_closed"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_ORDER = "invalid_order"
    TIMEOUT_ERROR = "timeout_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ExecutionAttempt:
    """Record of a single execution attempt"""
    attempt_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    error_message: Optional[str] = None
    failure_type: Optional[FailureType] = None
    execution_time_ms: Optional[float] = None
    result: Optional[Any] = None


@dataclass
class ExecutionTask:
    """Task for robust execution"""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    max_retries: int = 3
    timeout_seconds: float = 30.0
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    priority: int = 1  # Higher numbers = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    attempts: List[ExecutionAttempt] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    final_error: Optional[str] = None


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes needed to close
    timeout_duration: float = 60.0  # Seconds to wait before retry
    monitoring_window: float = 300.0  # Window for failure counting


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()
        self.recent_calls = deque(maxlen=100)  # Track recent call results
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.timeout_duration

    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            self.recent_calls.append(('success', datetime.now()))

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after recovery")
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self, error: Exception):
        """Handle failed execution"""
        with self._lock:
            self.recent_calls.append(('failure', datetime.now(), str(error)))
            self.last_failure_time = datetime.now()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # Failed during test - go back to open
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker {self.name} reopened after failed test")
            elif self.state == CircuitBreakerState.CLOSED:
                # Count recent failures
                recent_failures = self._count_recent_failures()
                if recent_failures >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.last_state_change = datetime.now()
                    logger.error(f"Circuit breaker {self.name} opened after {recent_failures} failures")

    def _count_recent_failures(self) -> int:
        """Count failures in recent monitoring window"""
        cutoff_time = datetime.now() - timedelta(seconds=self.config.monitoring_window)
        failures = 0

        for call in reversed(self.recent_calls):
            if call[1] < cutoff_time:
                break
            if call[0] == 'failure':
                failures += 1

        return failures

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            recent_failures = self._count_recent_failures()
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'recent_failures': recent_failures,
                'last_failure_time': self.last_failure_time,
                'last_state_change': self.last_state_change,
                'total_calls': len(self.recent_calls)
            }


class ExecutionRobustnessFramework:
    """
    Robust execution framework for trading operations

    Provides:
    - Intelligent retry mechanisms with exponential backoff
    - Circuit breakers for service protection
    - Execution monitoring and performance tracking
    - Error classification and handling
    - Graceful degradation strategies
    """

    def __init__(self,
                 default_max_retries: int = 3,
                 default_timeout: float = 30.0,
                 enable_circuit_breakers: bool = True):
        """
        Initialize execution framework

        Args:
            default_max_retries: Default maximum retry attempts
            default_timeout: Default timeout in seconds
            enable_circuit_breakers: Enable circuit breaker protection
        """
        self.default_max_retries = default_max_retries
        self.default_timeout = default_timeout
        self.enable_circuit_breakers = enable_circuit_breakers

        # Task management
        self.tasks: Dict[str, ExecutionTask] = {}
        self.task_counter = 0
        self._task_lock = threading.Lock()

        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Performance tracking
        self.execution_stats = defaultdict(list)
        self.error_stats = defaultdict(int)

        # Background processing
        self._running = False
        self._worker_thread = None

        logger.info("Execution Robustness Framework initialized")

    def start(self):
        """Start background processing"""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._background_worker, daemon=True)
            self._worker_thread.start()
            logger.info("Execution framework background worker started")

    def stop(self):
        """Stop background processing"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Execution framework stopped")

    def execute_robust(self,
                      function: Callable,
                      *args,
                      task_id: Optional[str] = None,
                      max_retries: Optional[int] = None,
                      timeout: Optional[float] = None,
                      circuit_breaker: Optional[str] = None,
                      priority: int = 1,
                      **kwargs) -> Any:
        """
        Execute function with robustness features

        Args:
            function: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            max_retries: Maximum retry attempts
            timeout: Timeout in seconds
            circuit_breaker: Circuit breaker name
            priority: Execution priority
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        if task_id is None:
            with self._task_lock:
                self.task_counter += 1
                task_id = f"task_{self.task_counter}"

        # Create execution task
        task = ExecutionTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries or self.default_max_retries,
            timeout_seconds=timeout or self.default_timeout,
            priority=priority
        )

        # Execute with circuit breaker if specified
        if circuit_breaker and self.enable_circuit_breakers:
            breaker = self._get_circuit_breaker(circuit_breaker)
            return breaker.call(self._execute_task, task)
        else:
            return self._execute_task(task)

    def execute_async(self,
                     function: Callable,
                     *args,
                     task_id: Optional[str] = None,
                     callback: Optional[Callable] = None,
                     **kwargs) -> str:
        """
        Execute function asynchronously

        Args:
            function: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            callback: Optional completion callback
            **kwargs: Function keyword arguments

        Returns:
            Task ID for tracking
        """
        if task_id is None:
            with self._task_lock:
                self.task_counter += 1
                task_id = f"async_task_{self.task_counter}"

        task = ExecutionTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            max_retries=kwargs.pop('max_retries', self.default_max_retries),
            timeout_seconds=kwargs.pop('timeout', self.default_timeout)
        )

        # Store callback if provided
        if callback:
            task.kwargs['_completion_callback'] = callback

        with self._task_lock:
            self.tasks[task_id] = task

        return task_id

    def get_task_status(self, task_id: str) -> Optional[ExecutionTask]:
        """Get status of async task"""
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel async task"""
        task = self.tasks.get(task_id)
        if task and task.status in [ExecutionStatus.PENDING, ExecutionStatus.RETRYING]:
            task.status = ExecutionStatus.CANCELLED
            return True
        return False

    def _execute_task(self, task: ExecutionTask) -> Any:
        """Execute a single task with retries"""
        task.status = ExecutionStatus.EXECUTING

        for attempt_num in range(task.max_retries + 1):
            attempt = ExecutionAttempt(
                attempt_number=attempt_num + 1,
                start_time=datetime.now()
            )
            task.attempts.append(attempt)

            try:
                # Execute with timeout
                result = self._execute_with_timeout(
                    task.function,
                    task.args,
                    task.kwargs,
                    task.timeout_seconds
                )

                # Success
                attempt.end_time = datetime.now()
                attempt.status = ExecutionStatus.COMPLETED
                attempt.result = result
                attempt.execution_time_ms = (attempt.end_time - attempt.start_time).total_seconds() * 1000

                task.status = ExecutionStatus.COMPLETED
                task.result = result

                # Update stats
                self._update_execution_stats(task, attempt)

                logger.debug(f"Task {task.task_id} completed successfully on attempt {attempt_num + 1}")
                return result

            except Exception as e:
                attempt.end_time = datetime.now()
                attempt.status = ExecutionStatus.FAILED
                attempt.error_message = str(e)
                attempt.failure_type = self._classify_error(e)
                attempt.execution_time_ms = (attempt.end_time - attempt.start_time).total_seconds() * 1000

                logger.warning(f"Task {task.task_id} attempt {attempt_num + 1} failed: {e}")

                # Update error stats
                self.error_stats[attempt.failure_type.value] += 1

                # Check if we should retry
                if attempt_num < task.max_retries:
                    if self._should_retry(attempt.failure_type):
                        task.status = ExecutionStatus.RETRYING
                        delay = self._calculate_retry_delay(attempt_num, task.retry_delays)
                        logger.info(f"Retrying task {task.task_id} in {delay} seconds")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Task {task.task_id} not retryable due to {attempt.failure_type}")
                        break
                else:
                    logger.error(f"Task {task.task_id} exhausted all {task.max_retries} retries")

        # All retries failed
        task.status = ExecutionStatus.FAILED
        task.final_error = f"Failed after {len(task.attempts)} attempts"

        if task.attempts:
            last_attempt = task.attempts[-1]
            raise Exception(f"Execution failed: {last_attempt.error_message}")
        else:
            raise Exception("Execution failed with no attempts recorded")

    def _execute_with_timeout(self, function: Callable, args: tuple, kwargs: dict, timeout: float) -> Any:
        """Execute function with timeout"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        # Set timeout alarm (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

            try:
                result = function(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)  # Clear alarm

        except AttributeError:
            # Windows doesn't have signal.SIGALRM, use threading approach
            result_container = [None]
            exception_container = [None]

            def target():
                try:
                    result_container[0] = function(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                # Thread is still running, consider it timed out
                raise TimeoutError(f"Execution timed out after {timeout} seconds")

            if exception_container[0]:
                raise exception_container[0]

            return result_container[0]

    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for retry logic"""
        error_str = str(error).lower()

        if isinstance(error, TimeoutError) or 'timeout' in error_str:
            return FailureType.TIMEOUT_ERROR
        elif 'network' in error_str or 'connection' in error_str:
            return FailureType.NETWORK_ERROR
        elif 'api' in error_str or 'server' in error_str:
            return FailureType.API_ERROR
        elif 'market closed' in error_str or 'trading hours' in error_str:
            return FailureType.MARKET_CLOSED
        elif 'insufficient' in error_str or 'funds' in error_str:
            return FailureType.INSUFFICIENT_FUNDS
        elif 'invalid' in error_str or 'order' in error_str:
            return FailureType.INVALID_ORDER
        else:
            return FailureType.UNKNOWN_ERROR

    def _should_retry(self, failure_type: FailureType) -> bool:
        """Determine if error type is retryable"""
        retryable_errors = {
            FailureType.NETWORK_ERROR,
            FailureType.API_ERROR,
            FailureType.TIMEOUT_ERROR,
            FailureType.SYSTEM_ERROR,
            FailureType.UNKNOWN_ERROR
        }
        return failure_type in retryable_errors

    def _calculate_retry_delay(self, attempt_num: int, delays: List[float]) -> float:
        """Calculate retry delay with exponential backoff"""
        if attempt_num < len(delays):
            return delays[attempt_num]
        else:
            # Exponential backoff for additional attempts
            base_delay = delays[-1] if delays else 1.0
            return min(base_delay * (2 ** (attempt_num - len(delays) + 1)), 30.0)

    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def _update_execution_stats(self, task: ExecutionTask, attempt: ExecutionAttempt):
        """Update execution performance statistics"""
        stats_key = f"{task.function.__name__}"
        self.execution_stats[stats_key].append({
            'execution_time_ms': attempt.execution_time_ms,
            'attempts': len(task.attempts),
            'timestamp': attempt.end_time,
            'success': True
        })

        # Keep only recent stats (last 1000 executions)
        if len(self.execution_stats[stats_key]) > 1000:
            self.execution_stats[stats_key] = self.execution_stats[stats_key][-1000:]

    def _background_worker(self):
        """Background worker for async task processing"""
        while self._running:
            try:
                # Process pending async tasks
                pending_tasks = []
                with self._task_lock:
                    for task_id, task in list(self.tasks.items()):
                        if task.status == ExecutionStatus.PENDING:
                            pending_tasks.append(task)

                # Sort by priority
                pending_tasks.sort(key=lambda t: t.priority, reverse=True)

                for task in pending_tasks[:5]:  # Process up to 5 tasks at once
                    try:
                        result = self._execute_task(task)

                        # Call completion callback if provided
                        callback = task.kwargs.get('_completion_callback')
                        if callback:
                            try:
                                callback(task.task_id, result, None)
                            except Exception as e:
                                logger.error(f"Completion callback failed for task {task.task_id}: {e}")

                    except Exception as e:
                        task.final_error = str(e)

                        # Call completion callback with error
                        callback = task.kwargs.get('_completion_callback')
                        if callback:
                            try:
                                callback(task.task_id, None, e)
                            except Exception as cb_error:
                                logger.error(f"Completion callback failed for task {task.task_id}: {cb_error}")

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Background worker error: {e}")
                time.sleep(1.0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}

        for func_name, executions in self.execution_stats.items():
            if executions:
                execution_times = [ex['execution_time_ms'] for ex in executions]
                stats[func_name] = {
                    'total_executions': len(executions),
                    'avg_execution_time_ms': sum(execution_times) / len(execution_times),
                    'min_execution_time_ms': min(execution_times),
                    'max_execution_time_ms': max(execution_times),
                    'success_rate': sum(1 for ex in executions if ex['success']) / len(executions)
                }

        # Circuit breaker stats
        cb_stats = {}
        for name, breaker in self.circuit_breakers.items():
            cb_stats[name] = breaker.get_stats()

        return {
            'execution_stats': stats,
            'error_stats': dict(self.error_stats),
            'circuit_breakers': cb_stats,
            'framework_info': {
                'running': self._running,
                'active_tasks': len([t for t in self.tasks.values() if t.status == ExecutionStatus.PENDING]),
                'total_tasks': len(self.tasks)
            }
        }


# Decorators for easy usage
def robust_execution(max_retries: int = 3,
                    timeout: float = 30.0,
                    circuit_breaker: Optional[str] = None):
    """Decorator for robust function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            framework = ExecutionRobustnessFramework()
            return framework.execute_robust(
                func,
                *args,
                max_retries=max_retries,
                timeout=timeout,
                circuit_breaker=circuit_breaker,
                **kwargs
            )
        return wrapper
    return decorator


# Global framework instance
_global_framework = None

def get_global_framework() -> ExecutionRobustnessFramework:
    """Get global execution framework instance"""
    global _global_framework
    if _global_framework is None:
        _global_framework = ExecutionRobustnessFramework()
        _global_framework.start()
    return _global_framework


# Example usage and testing
if __name__ == "__main__":
    import random

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    framework = ExecutionRobustnessFramework()
    framework.start()

    def unreliable_function(fail_rate: float = 0.3):
        """Test function that fails randomly"""
        if random.random() < fail_rate:
            raise Exception("Random failure occurred")
        return f"Success at {datetime.now()}"

    def slow_function(delay: float = 2.0):
        """Test function that takes time"""
        time.sleep(delay)
        return f"Completed after {delay} seconds"

    try:
        print("Testing robust execution...")

        # Test basic robust execution
        result = framework.execute_robust(
            unreliable_function,
            fail_rate=0.7,  # High failure rate
            max_retries=5,
            circuit_breaker="test_service"
        )
        print(f"Robust execution result: {result}")

        # Test timeout handling
        try:
            result = framework.execute_robust(
                slow_function,
                delay=5.0,  # Longer than default timeout
                timeout=2.0,
                max_retries=1
            )
        except Exception as e:
            print(f"Timeout test failed as expected: {e}")

        # Test async execution
        task_id = framework.execute_async(
            unreliable_function,
            fail_rate=0.2,
            callback=lambda tid, result, error: print(f"Async task {tid} completed: {result or error}")
        )
        print(f"Started async task: {task_id}")

        # Wait for async task
        time.sleep(3.0)

        # Get performance stats
        stats = framework.get_performance_stats()
        print(f"Performance stats: {stats}")

    finally:
        framework.stop()
        print("Framework stopped")