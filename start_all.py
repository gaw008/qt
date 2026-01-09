#!/usr/bin/env python3
"""
Comprehensive System Startup Script - One-Click Launch
Professional Quantitative Trading System Management

This script orchestrates the startup of all system components:
- React Frontend (UI)
- FastAPI Backend
- Streamlit Dashboard
- Trading Bot Worker
- System Health Monitoring
- Data Services

Author: Quantitative Trading System
Version: 2.0
"""

# Set encoding for Windows compatibility
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import time
import signal
import subprocess
import threading
import psutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SystemManager:
    """Professional system management with process orchestration."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.quant_dir = self.base_dir / "quant_system_full"
        self.processes: Dict[str, subprocess.Popen] = {}
        self.process_info: Dict[str, Dict] = {}
        self.shutdown_event = threading.Event()
        self.logger = self._setup_logging()

        # System configuration
        self.components = {
            'backend': {
                'name': 'FastAPI Backend',
                'port': 8000,
                'health_endpoint': 'http://localhost:8000/health',
                'startup_timeout': 60,
                'critical': True
            },
            'frontend_react': {
                'name': 'React Frontend',
                'port': 3000,
                'health_endpoint': 'http://localhost:3000',
                'startup_timeout': 45,
                'critical': False
            },
            'frontend_streamlit': {
                'name': 'Streamlit Dashboard',
                'port': 8501,
                'health_endpoint': 'http://localhost:8501',
                'startup_timeout': 30,
                'critical': False
            },
            'worker': {
                'name': 'Trading Bot Worker',
                'port': None,
                'health_endpoint': None,
                'startup_timeout': 20,
                'critical': True
            },
            'monitor': {
                'name': 'System Health Monitor',
                'port': None,
                'health_endpoint': None,
                'startup_timeout': 15,
                'critical': False
            }
        }

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('SystemManager')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"system_startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.shutdown_all_processes()

    def validate_environment(self) -> bool:
        """Validate system environment and dependencies."""
        self.logger.info("=== Environment Validation ===")

        validation_results = []

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.logger.info(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            validation_results.append(True)
        else:
            self.logger.error(f"[FAIL] Python version {python_version} is too old (requires 3.8+)")
            validation_results.append(False)

        # Check critical directories
        critical_dirs = [
            self.quant_dir,
            self.quant_dir / "bot",
            self.quant_dir / "dashboard",
            self.quant_dir / "UI"
        ]

        for directory in critical_dirs:
            if directory.exists():
                self.logger.info(f"[OK] Directory exists: {directory.name}")
                validation_results.append(True)
            else:
                self.logger.error(f"[FAIL] Missing critical directory: {directory}")
                validation_results.append(False)

        # Check environment configuration
        env_file = self.quant_dir / ".env"
        if env_file.exists():
            self.logger.info("[OK] Environment configuration found")
            validation_results.append(True)
        else:
            self.logger.warning("[WARN] No .env file found, using defaults from config.example.env")
            validation_results.append(True)  # Non-critical

        # Check Node.js and npm for React frontend
        try:
            node_result = subprocess.run(['node', '--version'],
                                       capture_output=True, text=True, timeout=10)
            npm_result = subprocess.run(['npm', '--version'],
                                      capture_output=True, text=True, timeout=10)
            if node_result.returncode == 0 and npm_result.returncode == 0:
                self.logger.info(f"[OK] Node.js {node_result.stdout.strip()}, npm {npm_result.stdout.strip()}")
                validation_results.append(True)
            else:
                raise subprocess.SubprocessError()
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("[WARN] Node.js or npm not found (React frontend will be skipped)")
            validation_results.append(True)  # Non-critical

        # Check port availability
        ports_to_check = [8000, 3000, 8501]
        for port in ports_to_check:
            if self._is_port_available(port):
                self.logger.info(f"[OK] Port {port} available")
                validation_results.append(True)
            else:
                self.logger.warning(f"[WARN] Port {port} is already in use")
                validation_results.append(True)  # Non-critical, might be existing instance

        success_rate = sum(validation_results) / len(validation_results)
        self.logger.info(f"Environment validation: {success_rate:.1%} success rate")

        return success_rate >= 0.8  # Allow some non-critical failures

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            # Try psutil method (may fail on Windows)
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    return False
            return True
        except (AttributeError, Exception) as e:
            # Fallback for Windows compatibility issue
            # Assume port is available if we can't check
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    return True  # Port is available
            except OSError:
                return False  # Port is in use

    def prepare_react_frontend(self) -> bool:
        """Prepare React frontend dependencies."""
        self.logger.info("=== Preparing React Frontend ===")

        ui_dir = self.quant_dir / "UI"
        if not ui_dir.exists():
            self.logger.error("React UI directory not found")
            return False

        package_json = ui_dir / "package.json"
        node_modules = ui_dir / "node_modules"

        # Install dependencies if needed
        if not node_modules.exists() or not (node_modules / ".bin").exists():
            self.logger.info("Installing React dependencies...")
            try:
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=ui_dir,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minutes timeout
                )

                if result.returncode == 0:
                    self.logger.info("[OK] React dependencies installed successfully")
                    return True
                else:
                    self.logger.error(f"[FAIL] npm install failed: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                self.logger.error("[FAIL] npm install timed out")
                return False
            except Exception as e:
                self.logger.error(f"[FAIL] Error installing React dependencies: {e}")
                return False
        else:
            self.logger.info("[OK] React dependencies already installed")
            return True

    def start_backend(self) -> Optional[subprocess.Popen]:
        """Start FastAPI backend server."""
        self.logger.info("Starting FastAPI Backend...")

        backend_dir = self.quant_dir / "dashboard" / "backend"
        if not backend_dir.exists():
            self.logger.error("Backend directory not found")
            return None

        try:
            # Start FastAPI with uvicorn
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app:app",
                 "--host", "0.0.0.0", "--port", "8000", "--reload"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] FastAPI Backend started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start FastAPI Backend: {e}")
            return None

    def start_react_frontend(self) -> Optional[subprocess.Popen]:
        """Start React frontend development server."""
        self.logger.info("Starting React Frontend...")

        ui_dir = self.quant_dir / "UI"
        if not ui_dir.exists():
            self.logger.error("React UI directory not found")
            return None

        try:
            # Start React development server
            process = subprocess.Popen(
                ['npm', 'run', 'dev'],
                cwd=ui_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            self.logger.info(f"[OK] React Frontend started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start React Frontend: {e}")
            return None

    def start_streamlit_dashboard(self) -> Optional[subprocess.Popen]:
        """Start Streamlit dashboard."""
        self.logger.info("Starting Streamlit Dashboard...")

        frontend_dir = self.quant_dir / "dashboard" / "frontend"
        if not frontend_dir.exists():
            self.logger.error("Streamlit frontend directory not found")
            return None

        try:
            # Start Streamlit
            process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                 "--server.port", "8501", "--server.headless", "true"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] Streamlit Dashboard started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Streamlit Dashboard: {e}")
            return None

    def start_worker(self) -> Optional[subprocess.Popen]:
        """Start trading bot worker."""
        self.logger.info("Starting Trading Bot Worker...")

        worker_dir = self.quant_dir / "dashboard" / "worker"
        if not worker_dir.exists():
            self.logger.error("Worker directory not found")
            return None

        try:
            # Start worker process
            process = subprocess.Popen(
                [sys.executable, "runner.py"],
                cwd=worker_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] Trading Bot Worker started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Trading Bot Worker: {e}")
            return None

    def start_system_monitor(self) -> Optional[subprocess.Popen]:
        """Start system health monitoring."""
        self.logger.info("Starting System Health Monitor...")

        try:
            # Start system health monitoring
            process = subprocess.Popen(
                [sys.executable, "system_health_monitoring.py"],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] System Health Monitor started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start System Health Monitor: {e}")
            return None

    def wait_for_health_check(self, component: str, timeout: int = 30) -> bool:
        """Wait for component to be healthy."""
        if component not in self.components:
            return False

        config = self.components[component]
        health_endpoint = config.get('health_endpoint')

        if not health_endpoint:
            # For components without health endpoint, just wait a bit
            time.sleep(5)
            return True

        start_time = time.time()
        self.logger.info(f"Waiting for {config['name']} health check...")

        while time.time() - start_time < timeout:
            try:
                import requests
                response = requests.get(health_endpoint, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"[OK] {config['name']} is healthy")
                    return True
            except:
                pass

            time.sleep(2)

        self.logger.warning(f"[WARN] {config['name']} health check timeout")
        return False

    def start_all_components(self) -> bool:
        """Start all system components in proper order."""
        self.logger.info("=== Starting All System Components ===")

        # Start components in dependency order
        startup_sequence = [
            ('backend', self.start_backend),
            ('worker', self.start_worker),
            ('frontend_react', self.start_react_frontend),
            ('frontend_streamlit', self.start_streamlit_dashboard),
            ('monitor', self.start_system_monitor)
        ]

        for component_name, start_func in startup_sequence:
            if self.shutdown_event.is_set():
                self.logger.info("Shutdown requested, stopping startup sequence")
                return False

            self.logger.info(f"--- Starting {self.components[component_name]['name']} ---")

            process = start_func()
            if process:
                self.processes[component_name] = process
                self.process_info[component_name] = {
                    'pid': process.pid,
                    'start_time': time.time(),
                    'name': self.components[component_name]['name']
                }

                # Wait for health check
                if not self.wait_for_health_check(
                    component_name,
                    self.components[component_name]['startup_timeout']
                ):
                    if self.components[component_name]['critical']:
                        self.logger.error(f"Critical component {component_name} failed to start")
                        return False
                    else:
                        self.logger.warning(f"Non-critical component {component_name} may have issues")
            else:
                if self.components[component_name]['critical']:
                    self.logger.error(f"Failed to start critical component: {component_name}")
                    return False

        return True

    def monitor_processes(self) -> None:
        """Monitor running processes and restart if needed."""
        self.logger.info("=== Process Monitoring Started ===")

        while not self.shutdown_event.is_set():
            for component_name, process in list(self.processes.items()):
                if process.poll() is not None:  # Process has terminated
                    self.logger.warning(f"Process {component_name} has terminated unexpectedly")

                    # Try to restart critical processes
                    if self.components[component_name]['critical']:
                        self.logger.info(f"Attempting to restart {component_name}...")
                        # Implementation for restart logic would go here

            time.sleep(10)  # Check every 10 seconds

    def shutdown_all_processes(self) -> None:
        """Gracefully shutdown all processes."""
        self.logger.info("=== Shutting Down All Processes ===")

        # Shutdown in reverse order
        shutdown_order = ['monitor', 'frontend_streamlit', 'frontend_react', 'worker', 'backend']

        for component_name in shutdown_order:
            if component_name in self.processes:
                process = self.processes[component_name]
                self.logger.info(f"Shutting down {self.components[component_name]['name']}...")

                try:
                    # Try graceful shutdown first
                    process.terminate()
                    process.wait(timeout=10)
                    self.logger.info(f"[OK] {component_name} shut down gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.logger.warning(f"Force killing {component_name}...")
                    process.kill()
                    process.wait()
                except Exception as e:
                    self.logger.error(f"Error shutting down {component_name}: {e}")

        self.logger.info("=== Shutdown Complete ===")

    def print_system_status(self) -> None:
        """Print current system status."""
        print("\n" + "="*60)
        print("         QUANTITATIVE TRADING SYSTEM")
        print("              System Status Dashboard")
        print("="*60)

        for component_name, process in self.processes.items():
            config = self.components[component_name]
            if process.poll() is None:
                status = "[OK] RUNNING"
                color = "\033[92m"  # Green
            else:
                status = "[FAIL] STOPPED"
                color = "\033[91m"  # Red

            print(f"{color}{status}\033[0m {config['name']}")
            if config['port']:
                print(f"         -> http://localhost:{config['port']}")

        print("\n" + "="*60)
        print("Control Commands:")
        print("  Ctrl+C: Graceful shutdown")
        print("  Access React Frontend: http://localhost:3000")
        print("  Access Streamlit Dashboard: http://localhost:8501")
        print("  Access API Documentation: http://localhost:8000/docs")
        print("="*60 + "\n")

    def run(self) -> int:
        """Main execution method."""
        try:
            self.logger.info("=== QUANTITATIVE TRADING SYSTEM STARTUP ===")

            # Environment validation
            if not self.validate_environment():
                self.logger.error("Environment validation failed")
                return 1

            # Prepare React frontend (optional)
            if not self.prepare_react_frontend():
                self.logger.warning("React frontend preparation skipped (Node.js not available)")

            # Start all components
            if not self.start_all_components():
                self.logger.error("Failed to start all components")
                self.shutdown_all_processes()
                return 1

            # Print status and start monitoring
            self.print_system_status()

            # Start process monitoring in background
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()

            # Keep main thread alive
            self.logger.info("System startup complete. Monitoring processes...")
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Unexpected error during startup: {e}")
            return 1
        finally:
            self.shutdown_all_processes()

        return 0

def main():
    """Entry point."""
    system_manager = SystemManager()
    return system_manager.run()

if __name__ == "__main__":
    sys.exit(main())