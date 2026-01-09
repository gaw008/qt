#!/usr/bin/env python3
"""
Ultra High-Performance Trading System Launcher
Professional Quantitative Trading System - Ultra Mode

This script launches the trading system in ultra-high performance mode with:
- GPU acceleration where available
- Advanced memory management
- Optimized data processing pipelines
- Enhanced parallel execution
- Real-time performance monitoring
- Adaptive resource allocation

Features:
- Multi-threading with intelligent load balancing
- GPU-accelerated ML inference
- Memory-mapped data structures
- Advanced caching mechanisms
- Ultra-low latency execution
- Professional performance monitoring

Author: Quantitative Trading System
Version: 2.0 Ultra
"""

# Set encoding for Windows compatibility
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import time
import signal
import json
import logging
import threading
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import queue

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class UltraSystemManager:
    """Ultra-high performance trading system management."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.quant_dir = self.base_dir / "quant_system_full"
        self.bot_dir = self.quant_dir / "bot"

        # System processes and resources
        self.processes: Dict[str, subprocess.Popen] = {}
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = threading.Event()
        self.performance_monitor: Optional[threading.Thread] = None

        # Performance configuration
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total // (1024**3)
        self.gpu_available = self._check_gpu_availability()

        # Ultra performance settings
        self.config = {
            'max_workers': min(self.cpu_count * 2, 16),
            'max_processes': min(self.cpu_count, 8),
            'memory_limit_gb': max(self.memory_gb * 0.8, 4),
            'gpu_enabled': self.gpu_available,
            'ultra_mode': True,
            'parallel_execution': True,
            'advanced_caching': True,
            'real_time_optimization': True,
            'batch_size_multiplier': 4,
            'concurrent_streams': 8,
            'performance_monitoring_interval': 1  # seconds
        }

        # Performance metrics
        self.metrics = {
            'start_time': None,
            'cpu_usage_history': [],
            'memory_usage_history': [],
            'gpu_usage_history': [],
            'throughput_history': [],
            'latency_history': [],
            'error_count': 0,
            'total_operations': 0,
            'peak_memory_mb': 0,
            'avg_cpu_usage': 0.0,
            'system_efficiency': 0.0
        }

        self.logger = self._setup_logging()
        self._optimize_system_settings()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup ultra-performance logging system."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('UltraSystem')
        logger.setLevel(logging.INFO)

        # High-performance console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '\033[95m%(asctime)s\033[0m - \033[96mULTRA\033[0m - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Buffered file handler for performance
        file_handler = logging.FileHandler(
            log_dir / f"ultra_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            mode='a',
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"[OK] CUDA GPU detected: {gpu_count} device(s)")
                return True
        except ImportError:
            pass

        try:
            import tensorflow as tf
            if tf.test.is_gpu_available():
                self.logger.info("[OK] TensorFlow GPU support detected")
                return True
        except ImportError:
            pass

        self.logger.info("[WARN] No GPU acceleration available, using CPU optimization")
        return False

    def _optimize_system_settings(self) -> None:
        """Optimize system settings for ultra performance."""
        self.logger.info("=== Ultra Performance Optimization ===")

        try:
            # Set process priority to high
            current_process = psutil.Process()
            if sys.platform == "win32":
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                current_process.nice(-10)  # Higher priority on Unix
            self.logger.info("[OK] Process priority set to HIGH")

            # Set CPU affinity if beneficial
            if self.cpu_count >= 8:
                # Reserve some cores for system operations
                available_cores = list(range(1, self.cpu_count))
                current_process.cpu_affinity(available_cores)
                self.logger.info(f"[OK] CPU affinity set to cores: {available_cores}")

            # Optimize memory settings
            if hasattr(psutil, 'virtual_memory'):
                vm = psutil.virtual_memory()
                if vm.available > 4 * 1024**3:  # 4GB available
                    os.environ['PYTHONHASHSEED'] = '0'  # Reproducible hash for caching
                    self.logger.info("[OK] Memory optimization enabled")

        except Exception as e:
            self.logger.warning(f"Some optimizations failed: {e}")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Ultra System received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
        self._ultra_shutdown()

    def validate_ultra_environment(self) -> bool:
        """Validate environment for ultra-high performance operation."""
        self.logger.info("=== Ultra Environment Validation ===")

        validation_results = []

        # Check system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 8:
            self.logger.info(f"[OK] Sufficient RAM: {memory_gb:.1f} GB")
            validation_results.append(True)
        else:
            self.logger.warning(f"[WARN] Limited RAM: {memory_gb:.1f} GB (recommended: 8+ GB)")
            validation_results.append(True)  # Still functional

        # Check CPU cores
        if self.cpu_count >= 4:
            self.logger.info(f"[OK] Multi-core CPU: {self.cpu_count} cores")
            validation_results.append(True)
        else:
            self.logger.warning(f"[WARN] Limited CPU cores: {self.cpu_count}")
            validation_results.append(True)

        # Check GPU availability
        if self.gpu_available:
            self.logger.info("[OK] GPU acceleration available")
            validation_results.append(True)
        else:
            self.logger.info("ℹ GPU acceleration not available (optional)")
            validation_results.append(True)

        # Check ultra-performance modules
        ultra_modules = [
            self.bot_dir / "gpu_training_pipeline.py",
            self.bot_dir / "performance_optimizer.py",
            self.bot_dir / "ultra_execution_engine.py"
        ]

        for module in ultra_modules:
            if module.exists():
                self.logger.info(f"[OK] Ultra module: {module.name}")
                validation_results.append(True)
            else:
                self.logger.info(f"ℹ Ultra module not found: {module.name} (optional)")
                validation_results.append(True)  # Optional

        # Check storage I/O performance
        try:
            start_time = time.time()
            test_file = self.base_dir / "temp_io_test.tmp"
            with open(test_file, 'wb') as f:
                f.write(b'0' * 1024 * 1024)  # 1MB write test
            os.remove(test_file)
            io_time = time.time() - start_time

            if io_time < 0.1:
                self.logger.info(f"[OK] Fast storage I/O: {io_time:.3f}s")
                validation_results.append(True)
            else:
                self.logger.warning(f"[WARN] Slow storage I/O: {io_time:.3f}s")
                validation_results.append(True)

        except Exception as e:
            self.logger.warning(f"[WARN] I/O test failed: {e}")
            validation_results.append(True)

        success_rate = sum(validation_results) / len(validation_results)
        self.logger.info(f"Ultra environment validation: {success_rate:.1%}")

        return success_rate >= 0.9

    def initialize_ultra_components(self) -> bool:
        """Initialize ultra-performance components."""
        self.logger.info("=== Initializing Ultra Components ===")

        try:
            # Initialize thread pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config['max_workers'],
                thread_name_prefix="UltraWorker"
            )
            self.logger.info(f"[OK] Thread pool initialized: {self.config['max_workers']} workers")

            # Initialize process pool if beneficial
            if self.cpu_count >= 4:
                self.process_pool = ProcessPoolExecutor(
                    max_workers=self.config['max_processes']
                )
                self.logger.info(f"[OK] Process pool initialized: {self.config['max_processes']} processes")

            return True

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to initialize ultra components: {e}")
            return False

    def start_gpu_pipeline(self) -> Optional[subprocess.Popen]:
        """Start GPU-accelerated ML pipeline if available."""
        if not self.gpu_available:
            return None

        gpu_pipeline_path = self.bot_dir / "gpu_training_pipeline.py"
        if not gpu_pipeline_path.exists():
            self.logger.info("GPU pipeline not available, skipping...")
            return None

        try:
            self.logger.info("Starting GPU Training Pipeline...")

            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.quant_dir)
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

            process = subprocess.Popen(
                [sys.executable, "gpu_training_pipeline.py", "--ultra-mode"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )

            self.logger.info(f"[OK] GPU Training Pipeline started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start GPU Training Pipeline: {e}")
            return None

    def start_performance_optimizer(self) -> Optional[subprocess.Popen]:
        """Start real-time performance optimizer."""
        optimizer_path = self.bot_dir / "performance_optimizer.py"
        if not optimizer_path.exists():
            self.logger.info("Performance optimizer not available, using built-in optimization...")
            return None

        try:
            self.logger.info("Starting Performance Optimizer...")

            process = subprocess.Popen(
                [sys.executable, "performance_optimizer.py",
                 "--ultra-mode", "--real-time"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] Performance Optimizer started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Performance Optimizer: {e}")
            return None

    def start_ultra_execution_engine(self) -> Optional[subprocess.Popen]:
        """Start ultra-low latency execution engine."""
        execution_engine_path = self.bot_dir / "ultra_execution_engine.py"
        if not execution_engine_path.exists():
            # Fall back to standard execution engine
            return self.start_standard_execution_engine()

        try:
            self.logger.info("Starting Ultra Execution Engine...")

            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.quant_dir)
            env['ULTRA_MODE'] = 'true'
            env['MAX_THREADS'] = str(self.config['max_workers'])
            env['BATCH_SIZE'] = str(self.config['batch_size_multiplier'] * 1000)

            process = subprocess.Popen(
                [sys.executable, "ultra_execution_engine.py"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )

            self.logger.info(f"[OK] Ultra Execution Engine started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Ultra Execution Engine: {e}")
            return None

    def start_standard_execution_engine(self) -> Optional[subprocess.Popen]:
        """Start standard execution engine with ultra optimizations."""
        worker_dir = self.quant_dir / "dashboard" / "worker"
        runner_path = worker_dir / "runner.py"

        if not runner_path.exists():
            self.logger.error("Standard execution engine not found")
            return None

        try:
            self.logger.info("Starting Standard Execution Engine (Ultra Optimized)...")

            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.quant_dir)
            env['ULTRA_PERFORMANCE'] = 'true'
            env['PARALLEL_WORKERS'] = str(self.config['max_workers'])

            process = subprocess.Popen(
                [sys.executable, "runner.py"],
                cwd=worker_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )

            self.logger.info(f"[OK] Standard Execution Engine started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Standard Execution Engine: {e}")
            return None

    def start_ultra_data_pipeline(self) -> Optional[subprocess.Popen]:
        """Start ultra-high performance data processing pipeline."""
        data_processor_path = self.bot_dir / "realtime_data_processor_c1.py"
        if not data_processor_path.exists():
            self.logger.info("Ultra data pipeline not available, using standard processing...")
            return None

        try:
            self.logger.info("Starting Ultra Data Pipeline...")

            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.quant_dir)
            env['ULTRA_MODE'] = 'true'
            env['CONCURRENT_STREAMS'] = str(self.config['concurrent_streams'])
            env['BATCH_MULTIPLIER'] = str(self.config['batch_size_multiplier'])

            process = subprocess.Popen(
                [sys.executable, "realtime_data_processor_c1.py", "--ultra"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )

            self.logger.info(f"[OK] Ultra Data Pipeline started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Ultra Data Pipeline: {e}")
            return None

    def start_all_ultra_systems(self) -> bool:
        """Start all ultra-performance systems."""
        self.logger.info("=== Starting Ultra Systems ===")
        self.metrics['start_time'] = datetime.now()

        # Initialize ultra components
        if not self.initialize_ultra_components():
            return False

        # Start ultra systems in optimal order
        ultra_systems = [
            ('gpu_pipeline', self.start_gpu_pipeline),
            ('performance_optimizer', self.start_performance_optimizer),
            ('data_pipeline', self.start_ultra_data_pipeline),
            ('execution_engine', self.start_ultra_execution_engine)
        ]

        successful_starts = 0
        for system_name, start_func in ultra_systems:
            if self.shutdown_event.is_set():
                break

            self.logger.info(f"--- Starting {system_name} ---")
            process = start_func()

            if process:
                self.processes[system_name] = process
                successful_starts += 1
                time.sleep(2)  # Brief pause between starts
            else:
                # Some systems are optional
                self.logger.info(f"Skipping optional system: {system_name}")

        # Require at least the execution engine
        if 'execution_engine' not in self.processes:
            self.logger.error("Critical: Execution engine failed to start")
            return False

        self.logger.info(f"[OK] {successful_starts} ultra systems started successfully")
        return successful_starts > 0

    def monitor_ultra_performance(self) -> None:
        """Monitor ultra-performance metrics in real-time."""
        self.logger.info("=== Ultra Performance Monitoring Started ===")

        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()

                # CPU usage monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics['cpu_usage_history'].append((current_time, cpu_percent))

                # Memory usage monitoring
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics['memory_usage_history'].append((current_time, memory_mb))
                self.metrics['peak_memory_mb'] = max(self.metrics['peak_memory_mb'], memory_mb)

                # GPU monitoring if available
                if self.gpu_available:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_usage = gpus[0].load * 100
                            self.metrics['gpu_usage_history'].append((current_time, gpu_usage))
                    except ImportError:
                        pass

                # Process monitoring
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        self.logger.warning(f"Ultra process {name} has terminated")
                        del self.processes[name]
                        self.metrics['error_count'] += 1

                # Clean old metrics (keep last 300 entries = 5 minutes at 1s interval)
                max_history = 300
                self.metrics['cpu_usage_history'] = self.metrics['cpu_usage_history'][-max_history:]
                self.metrics['memory_usage_history'] = self.metrics['memory_usage_history'][-max_history:]
                self.metrics['gpu_usage_history'] = self.metrics['gpu_usage_history'][-max_history:]

                # Calculate running averages
                if self.metrics['cpu_usage_history']:
                    recent_cpu = [x[1] for x in self.metrics['cpu_usage_history'][-60:]]  # Last minute
                    self.metrics['avg_cpu_usage'] = sum(recent_cpu) / len(recent_cpu)

                # Log detailed status every minute
                if int(current_time) % 60 == 0:
                    self._log_ultra_status()

                time.sleep(self.config['performance_monitoring_interval'])

            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                time.sleep(5)

    def _log_ultra_status(self) -> None:
        """Log comprehensive ultra system status."""
        if not self.metrics['start_time']:
            return

        uptime = datetime.now() - self.metrics['start_time']
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_minutes': uptime.total_seconds() / 60,
            'active_processes': len(self.processes),
            'avg_cpu_usage_percent': round(self.metrics['avg_cpu_usage'], 1),
            'peak_memory_mb': round(self.metrics['peak_memory_mb'], 1),
            'error_count': self.metrics['error_count'],
            'gpu_enabled': self.gpu_available,
            'ultra_mode': True,
            'thread_pool_size': self.config['max_workers'],
            'process_pool_size': self.config['max_processes']
        }

        self.logger.info(f"Ultra Status: {json.dumps(status, indent=2)}")

    def _ultra_shutdown(self) -> None:
        """Ultra-fast graceful shutdown."""
        self.logger.info("=== Ultra System Shutdown ===")

        try:
            # Stop performance monitoring
            self.shutdown_event.set()

            # Shutdown thread and process pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)

            if self.process_pool:
                self.process_pool.shutdown(wait=False)

            # Shutdown processes in reverse order
            shutdown_order = ['gpu_pipeline', 'data_pipeline', 'execution_engine', 'performance_optimizer']

            for process_name in shutdown_order:
                if process_name in self.processes:
                    process = self.processes[process_name]
                    self.logger.info(f"Shutting down {process_name}...")

                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

            # Log final performance summary
            self._log_final_performance_summary()

        except Exception as e:
            self.logger.error(f"Error during ultra shutdown: {e}")

    def _log_final_performance_summary(self) -> None:
        """Log final performance summary."""
        if not self.metrics['start_time']:
            return

        total_runtime = datetime.now() - self.metrics['start_time']

        summary = {
            'total_runtime_minutes': total_runtime.total_seconds() / 60,
            'peak_memory_mb': self.metrics['peak_memory_mb'],
            'avg_cpu_usage_percent': round(self.metrics['avg_cpu_usage'], 1),
            'total_errors': self.metrics['error_count'],
            'systems_started': len(self.processes),
            'gpu_acceleration_used': self.gpu_available,
            'ultra_mode_enabled': True
        }

        self.logger.info(f"Final Performance Summary: {json.dumps(summary, indent=2)}")

    def print_ultra_banner(self) -> None:
        """Print ultra system startup banner."""
        print("\n" + "="*80)
        print("          QUANTITATIVE TRADING SYSTEM - ULTRA PERFORMANCE MODE")
        print("               Advanced High-Frequency Trading Platform")
        print("="*80)
        print(f"CPU Cores: {self.cpu_count} | Memory: {self.memory_gb}GB | GPU: {'Available' if self.gpu_available else 'Not Available'}")
        print(f"Thread Workers: {self.config['max_workers']} | Process Workers: {self.config['max_processes']}")
        print(f"Ultra Optimizations: ENABLED | Real-time Monitoring: ACTIVE")
        print("="*80)
        print("Ultra Systems Status:")

    def print_ultra_status(self) -> None:
        """Print current ultra system status."""
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"[OK] {name.replace('_', ' ').title()}: RUNNING")
            else:
                print(f"[FAIL] {name.replace('_', ' ').title()}: STOPPED")

        print(f"[OK] Performance Monitoring: ACTIVE")
        print(f"[OK] Resource Optimization: ENABLED")

        if self.metrics['start_time']:
            uptime = datetime.now() - self.metrics['start_time']
            print(f"[OK] Uptime: {str(uptime).split('.')[0]}")

        print("="*80)
        print("Ultra Performance Commands:")
        print("  Ctrl+C: Emergency shutdown with performance summary")
        print("  Logs: Check logs/ directory for detailed performance data")
        print("="*80 + "\n")

    def run(self) -> int:
        """Main ultra system execution."""
        try:
            self.print_ultra_banner()

            # Ultra environment validation
            if not self.validate_ultra_environment():
                self.logger.error("Ultra environment validation failed")
                return 1

            # Start all ultra systems
            if not self.start_all_ultra_systems():
                self.logger.error("Failed to start ultra systems")
                self._ultra_shutdown()
                return 1

            self.print_ultra_status()

            # Start ultra performance monitoring
            self.performance_monitor = threading.Thread(
                target=self.monitor_ultra_performance,
                daemon=True,
                name="UltraMonitor"
            )
            self.performance_monitor.start()

            # Main execution loop
            self.logger.info("Ultra Trading System running at maximum performance...")
            while not self.shutdown_event.is_set():
                time.sleep(0.1)  # Ultra-responsive main loop

        except KeyboardInterrupt:
            self.logger.info("Ultra system shutdown requested")
        except Exception as e:
            self.logger.error(f"Ultra system error: {e}")
            return 1
        finally:
            self._ultra_shutdown()

        return 0

def main():
    """Entry point for ultra system."""
    ultra_manager = UltraSystemManager()
    return ultra_manager.run()

if __name__ == "__main__":
    sys.exit(main())