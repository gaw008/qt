#!/usr/bin/env python3
"""
Standalone Trading Bot Launcher
Professional Quantitative Trading System

This script launches the trading bot with AI/ML integration,
comprehensive risk management, and real-time monitoring.

Features:
- AI/ML learning engines integration
- Enhanced risk management with ES@97.5%
- Real-time performance monitoring
- Emergency stop capabilities
- Configuration management
- Comprehensive logging

Author: Quantitative Trading System
Version: 2.0
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
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import psutil

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TradingBotManager:
    """Professional trading bot management with AI/ML integration."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.quant_dir = self.base_dir / "quant_system_full"
        self.bot_dir = self.quant_dir / "bot"
        self.worker_dir = self.quant_dir / "dashboard" / "worker"

        self.bot_process: Optional[subprocess.Popen] = None
        self.ai_processes: Dict[str, subprocess.Popen] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        self.logger = self._setup_logging()
        self.config = self._load_configuration()

        # Performance metrics
        self.start_time = None
        self.metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0,
            'risk_score': 0.0,
            'ai_predictions_accuracy': 0.0
        }

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)

        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '\033[92m%(asctime)s\033[0m - \033[94m%(name)s\033[0m - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration."""
        config = {
            'dry_run': True,
            'log_level': 'INFO',
            'max_positions': 20,
            'risk_limit': 0.02,
            'update_interval': 30,
            'ai_enabled': True,
            'enhanced_risk': True,
            'real_time_monitoring': True
        }

        # Load from .env file
        env_file = self.quant_dir / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            if key == 'DRY_RUN':
                                config['dry_run'] = value.lower() == 'true'
                            elif key == 'MAX_POSITIONS':
                                config['max_positions'] = int(value)
                            elif key == 'RISK_LIMIT':
                                config['risk_limit'] = float(value)
                            elif key == 'UPDATE_INTERVAL':
                                config['update_interval'] = int(value)
            except Exception as e:
                self.logger.warning(f"Error loading configuration: {e}")

        return config

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self._emergency_stop()

    def validate_environment(self) -> bool:
        """Validate trading environment and dependencies."""
        self.logger.info("=== Trading Environment Validation ===")

        validation_results = []

        # Check critical directories
        critical_dirs = [self.quant_dir, self.bot_dir, self.worker_dir]
        for directory in critical_dirs:
            if directory.exists():
                self.logger.info(f"[OK] Directory exists: {directory.name}")
                validation_results.append(True)
            else:
                self.logger.error(f"[FAIL] Missing critical directory: {directory}")
                validation_results.append(False)

        # Check configuration files
        config_files = [
            self.quant_dir / "config.example.env",
            self.bot_dir / "__init__.py"
        ]

        for config_file in config_files:
            if config_file.exists():
                self.logger.info(f"[OK] Configuration file: {config_file.name}")
                validation_results.append(True)
            else:
                self.logger.warning(f"[WARN] Missing configuration: {config_file.name}")
                validation_results.append(True)  # Non-critical

        # Check AI/ML modules
        ai_modules = [
            self.bot_dir / "ai_learning_engine.py",
            self.bot_dir / "ai_strategy_optimizer.py",
            self.bot_dir / "enhanced_risk_manager.py"
        ]

        for module in ai_modules:
            if module.exists():
                self.logger.info(f"[OK] AI/ML module: {module.name}")
                validation_results.append(True)
            else:
                self.logger.warning(f"[WARN] AI/ML module missing: {module.name}")
                validation_results.append(True)  # Non-critical for basic operation

        # Check Python dependencies
        try:
            import numpy
            import pandas
            import sklearn
            import requests
            self.logger.info("[OK] Core Python dependencies available")
            validation_results.append(True)
        except ImportError as e:
            self.logger.error(f"[FAIL] Missing Python dependencies: {e}")
            validation_results.append(False)

        # Check system resources
        memory = psutil.virtual_memory()
        if memory.available > 1024 * 1024 * 1024:  # 1GB
            self.logger.info(f"[OK] Available memory: {memory.available // (1024*1024)} MB")
            validation_results.append(True)
        else:
            self.logger.warning("[WARN] Low available memory")
            validation_results.append(True)  # Non-critical

        success_rate = sum(validation_results) / len(validation_results)
        self.logger.info(f"Environment validation: {success_rate:.1%} success rate")

        return success_rate >= 0.8

    def start_ai_learning_engine(self) -> Optional[subprocess.Popen]:
        """Start AI learning engine if available."""
        ai_engine_path = self.bot_dir / "ai_learning_engine.py"
        if not ai_engine_path.exists():
            self.logger.warning("AI Learning Engine not found, skipping...")
            return None

        try:
            self.logger.info("Starting AI Learning Engine...")
            process = subprocess.Popen(
                [sys.executable, "ai_learning_engine.py", "--daemon"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] AI Learning Engine started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start AI Learning Engine: {e}")
            return None

    def start_strategy_optimizer(self) -> Optional[subprocess.Popen]:
        """Start AI strategy optimizer if available."""
        optimizer_path = self.bot_dir / "ai_strategy_optimizer.py"
        if not optimizer_path.exists():
            self.logger.warning("Strategy Optimizer not found, skipping...")
            return None

        try:
            self.logger.info("Starting Strategy Optimizer...")
            process = subprocess.Popen(
                [sys.executable, "ai_strategy_optimizer.py", "--continuous"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] Strategy Optimizer started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Strategy Optimizer: {e}")
            return None

    def start_enhanced_risk_manager(self) -> Optional[subprocess.Popen]:
        """Start enhanced risk management system."""
        risk_manager_path = self.bot_dir / "enhanced_risk_manager.py"
        if not risk_manager_path.exists():
            self.logger.warning("Enhanced Risk Manager not found, using basic risk management...")
            return None

        try:
            self.logger.info("Starting Enhanced Risk Manager...")
            process = subprocess.Popen(
                [sys.executable, "enhanced_risk_manager.py", "--monitor"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] Enhanced Risk Manager started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Enhanced Risk Manager: {e}")
            return None

    def start_main_bot(self) -> Optional[subprocess.Popen]:
        """Start the main trading bot process."""
        runner_path = self.worker_dir / "runner.py"
        if not runner_path.exists():
            self.logger.error("Main bot runner not found")
            return None

        try:
            self.logger.info("Starting Main Trading Bot...")

            # Prepare environment variables
            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.quant_dir)
            if self.config['dry_run']:
                env['DRY_RUN'] = 'true'

            process = subprocess.Popen(
                [sys.executable, "runner.py"],
                cwd=self.worker_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )

            self.logger.info(f"[OK] Main Trading Bot started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Main Trading Bot: {e}")
            return None

    def start_real_time_monitor(self) -> Optional[subprocess.Popen]:
        """Start real-time monitoring system."""
        monitor_path = self.bot_dir / "real_time_monitor.py"
        if not monitor_path.exists():
            self.logger.warning("Real-time monitor not found, using basic monitoring...")
            return None

        try:
            self.logger.info("Starting Real-time Monitor...")
            process = subprocess.Popen(
                [sys.executable, "real_time_monitor.py"],
                cwd=self.bot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(self.quant_dir))
            )

            self.logger.info(f"[OK] Real-time Monitor started (PID: {process.pid})")
            return process

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start Real-time Monitor: {e}")
            return None

    def start_all_systems(self) -> bool:
        """Start all trading bot systems."""
        self.logger.info("=== Starting Trading Bot Systems ===")
        self.start_time = datetime.now()

        # Start AI/ML components if enabled
        if self.config.get('ai_enabled', True):
            # AI Learning Engine
            ai_engine = self.start_ai_learning_engine()
            if ai_engine:
                self.ai_processes['ai_engine'] = ai_engine

            # Strategy Optimizer
            strategy_optimizer = self.start_strategy_optimizer()
            if strategy_optimizer:
                self.ai_processes['strategy_optimizer'] = strategy_optimizer

        # Start Enhanced Risk Manager if enabled
        if self.config.get('enhanced_risk', True):
            risk_manager = self.start_enhanced_risk_manager()
            if risk_manager:
                self.ai_processes['risk_manager'] = risk_manager

        # Start Real-time Monitoring if enabled
        if self.config.get('real_time_monitoring', True):
            monitor = self.start_real_time_monitor()
            if monitor:
                self.ai_processes['monitor'] = monitor

        # Start main trading bot (critical)
        self.bot_process = self.start_main_bot()
        if not self.bot_process:
            self.logger.error("Failed to start main trading bot")
            return False

        # Wait for systems to initialize
        self.logger.info("Waiting for systems to initialize...")
        time.sleep(10)

        return True

    def monitor_systems(self) -> None:
        """Monitor all running systems."""
        self.logger.info("=== System Monitoring Started ===")

        while not self.shutdown_event.is_set():
            try:
                # Check main bot process
                if self.bot_process and self.bot_process.poll() is not None:
                    self.logger.error("Main trading bot has terminated unexpectedly!")
                    self._emergency_stop()
                    break

                # Check AI/ML processes
                for name, process in list(self.ai_processes.items()):
                    if process.poll() is not None:
                        self.logger.warning(f"AI process {name} has terminated")
                        del self.ai_processes[name]

                # Update performance metrics
                self._update_performance_metrics()

                # Log system status every 5 minutes
                if int(time.time()) % 300 == 0:
                    self._log_system_status()

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)

    def _update_performance_metrics(self) -> None:
        """Update performance metrics from system state."""
        try:
            # This would integrate with actual trading system metrics
            # For now, we'll simulate basic metric tracking
            if self.start_time:
                runtime = datetime.now() - self.start_time
                self.metrics['runtime_hours'] = runtime.total_seconds() / 3600

            # In a real implementation, these would come from:
            # - Trading system logs
            # - Position management system
            # - Risk management system
            # - AI prediction accuracy tracking

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def _log_system_status(self) -> None:
        """Log comprehensive system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': self.metrics.get('runtime_hours', 0),
            'main_bot_status': 'running' if self.bot_process and self.bot_process.poll() is None else 'stopped',
            'ai_processes': len(self.ai_processes),
            'dry_run_mode': self.config['dry_run'],
            'memory_usage_mb': psutil.Process().memory_info().rss // (1024 * 1024),
            'cpu_percent': psutil.Process().cpu_percent()
        }

        self.logger.info(f"System Status: {json.dumps(status, indent=2)}")

    def _emergency_stop(self) -> None:
        """Emergency stop all trading activities."""
        self.logger.warning("=== EMERGENCY STOP ACTIVATED ===")

        try:
            # Stop main bot process
            if self.bot_process:
                self.logger.info("Stopping main trading bot...")
                self.bot_process.terminate()
                try:
                    self.bot_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.bot_process.kill()
                    self.bot_process.wait()

            # Stop AI/ML processes
            for name, process in self.ai_processes.items():
                self.logger.info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            self.logger.info("=== EMERGENCY STOP COMPLETE ===")

        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")

    def print_startup_banner(self) -> None:
        """Print system startup banner."""
        print("\n" + "="*70)
        print("        QUANTITATIVE TRADING BOT - PROFESSIONAL EDITION")
        print("             Advanced AI/ML Integrated Trading System")
        print("="*70)
        print(f"Mode: {'DRY RUN' if self.config['dry_run'] else 'LIVE TRADING'}")
        print(f"Max Positions: {self.config['max_positions']}")
        print(f"Risk Limit: {self.config['risk_limit']:.1%}")
        print(f"AI/ML Enabled: {'Yes' if self.config['ai_enabled'] else 'No'}")
        print(f"Enhanced Risk: {'Yes' if self.config['enhanced_risk'] else 'No'}")
        print("="*70)
        print("Status:")

    def print_running_status(self) -> None:
        """Print current running status."""
        if self.bot_process and self.bot_process.poll() is None:
            print("[OK] Main Trading Bot: RUNNING")
        else:
            print("[FAIL] Main Trading Bot: STOPPED")

        print(f"[OK] AI/ML Processes: {len(self.ai_processes)} active")

        if self.start_time:
            runtime = datetime.now() - self.start_time
            print(f"[OK] Uptime: {str(runtime).split('.')[0]}")

        print("="*70)
        print("Control Commands:")
        print("  Ctrl+C: Emergency stop and shutdown")
        print("  Monitor: Check logs/ directory for detailed logs")
        print("="*70 + "\n")

    def run(self) -> int:
        """Main execution method."""
        try:
            self.print_startup_banner()

            # Validate environment
            if not self.validate_environment():
                self.logger.error("Environment validation failed")
                return 1

            # Start all systems
            if not self.start_all_systems():
                self.logger.error("Failed to start trading systems")
                self._emergency_stop()
                return 1

            self.print_running_status()

            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self.monitor_systems, daemon=True)
            self.monitoring_thread.start()

            # Keep main thread alive
            self.logger.info("Trading bot systems started successfully. Monitoring...")
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            self._emergency_stop()

        return 0

def main():
    """Entry point."""
    bot_manager = TradingBotManager()
    return bot_manager.run()

if __name__ == "__main__":
    sys.exit(main())