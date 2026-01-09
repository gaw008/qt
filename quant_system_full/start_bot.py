#!/usr/bin/env python3
"""
Complete Quantitative Trading Bot Startup Script

This script provides a comprehensive startup process for the quantitative trading bot system.
"""

import os
import sys

# Configure Unicode encoding for Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'
import time
import subprocess
import requests
import argparse
from pathlib import Path
from datetime import datetime

class BotLauncher:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.backend_port = 8001
        self.frontend_port = 8502
        self.admin_token = "wgyjd0508"  # Read from .env
        
    def print_banner(self):
        print("""
================================================================
          Quantitative Trading Bot Launcher
================================================================
  Automated startup for complete trading system
  Dashboard + Worker + API deployment
  Supports live/simulation trading modes
================================================================
        """)
    
    def check_dependencies(self):
        """Check dependencies and environment"""
        print("[*] Checking system dependencies...")

        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8+ required")
        print(f"[OK] Python version: {sys.version_info.major}.{sys.version_info.minor}")

        # Check required packages
        required_packages = ['fastapi', 'uvicorn', 'streamlit', 'pandas', 'tigeropen']
        missing = []
        for pkg in required_packages:
            try:
                __import__(pkg)
                print(f"[OK] {pkg}")
            except ImportError:
                missing.append(pkg)
                print(f"[MISSING] {pkg}")

        if missing:
            print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
            install = input("Install automatically? (y/n): ")
            if install.lower() == 'y':
                for pkg in missing:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])
            else:
                raise Exception("Please install required dependencies first")

        # Check .env file
        env_file = self.base_dir / '.env'
        if not env_file.exists():
            print("[ERROR] .env file not found")
            self.create_env_template()
        else:
            print("[OK] .env configuration file")

        print("[OK] Dependency check complete\n")
    
    def create_env_template(self):
        """Create .env template"""
        template = """# === AI API ===
GEMINI_API_KEY=your_api_key_here

# === Bot / TradeUP SDK ===
TIGER_ID=your_tiger_id
ACCOUNT=your_account_number
PRIVATE_KEY_PATH=private_key.pem
SECRET_KEY=
TIMEZONE=US/Eastern
LANG=en_US
DRY_RUN=true              # true=simulation, false=live trading

# === Data Source Configuration ===
DATA_SOURCE=auto          # "tiger", "yahoo_api", "yahoo_mcp", "auto"

# === Yahoo Finance API Settings ===
YAHOO_API_TIMEOUT=10.0
YAHOO_API_RETRIES=3
YAHOO_API_RETRY_DELAY=1.0

# === Yahoo Finance MCP Settings ===
USE_MCP_TOOLS=true

# === Dashboard ===
ADMIN_TOKEN=your_secure_token
API_BASE=http://localhost:8001
"""
        with open(self.base_dir / '.env', 'w') as f:
            f.write(template)
        print("[INFO] .env template created. Please fill in configuration and re-run.")
        sys.exit(1)
    
    def load_config(self):
        """Load configuration"""
        from dotenv import load_dotenv
        load_dotenv(self.base_dir / '.env')

        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.admin_token = os.getenv('ADMIN_TOKEN', 'changeme')
        self.tiger_id = os.getenv('TIGER_ID', '')
        self.account = os.getenv('ACCOUNT', '')

        print(f"[*] Configuration loaded:")
        print(f"   Simulation mode: {'Yes' if self.dry_run else 'No (Live Trading)'}")
        print(f"   Tiger ID: {self.tiger_id}")
        print(f"   Account: {self.account}")
        print()
    
    def kill_existing_processes(self):
        """Stop existing processes"""
        print("[*] Checking and stopping existing processes...")

        # Find and terminate processes on Windows
        try:
            # Find uvicorn processes
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                                  capture_output=True, text=True)
            if 'uvicorn' in result.stdout:
                print("[*] Found existing backend process, attempting to stop...")
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe'],
                             capture_output=True)
        except:
            pass

        print("[OK] Process cleanup complete\n")
    
    def start_backend(self):
        """Start backend API"""
        print("[*] Starting Dashboard backend...")

        backend_dir = self.base_dir / 'dashboard' / 'backend'
        os.chdir(backend_dir)

        cmd = [sys.executable, '-m', 'uvicorn', 'app:app',
               '--host', '0.0.0.0', '--port', str(self.backend_port)]

        # Start backend process
        self.backend_process = subprocess.Popen(cmd,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)

        # Wait for backend to start
        for i in range(30):
            try:
                response = requests.get(f'http://localhost:{self.backend_port}/health',
                                      timeout=1)
                if response.status_code == 200:
                    print("[OK] Backend API started successfully")
                    break
            except:
                time.sleep(1)
                print(f"[*] Waiting for backend... ({i+1}/30)")
        else:
            raise Exception("Backend startup timeout")
    
    def start_frontend(self):
        """Start frontend interface"""
        print("[*] Starting Dashboard frontend...")

        frontend_dir = self.base_dir / 'dashboard' / 'frontend'
        os.chdir(frontend_dir)

        cmd = [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
               '--server.port', str(self.frontend_port)]

        # Start frontend process
        self.frontend_process = subprocess.Popen(cmd,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)

        # Wait for frontend to start
        time.sleep(5)

        try:
            response = requests.get(f'http://localhost:{self.frontend_port}', timeout=2)
            print("[OK] Frontend interface started successfully")
        except:
            print("[WARNING] Frontend may need more time to start")
    
    def test_tiger_api(self):
        """Test Tiger API connection"""
        print("[*] Testing Tiger API connection...")

        os.chdir(self.base_dir)

        test_script = '''
import sys
sys.path.append('.')
from bot.tradeup_client import build_clients

try:
    quote_client, trade_client = build_clients()
    if quote_client is None:
        print("DRY_RUN mode - No API connection needed")
    else:
        permissions = quote_client.grab_quote_permission()
        print(f"API connection successful - Permissions: {permissions}")
except Exception as e:
    print(f"API connection failed: {e}")
'''

        result = subprocess.run([sys.executable, '-c', test_script],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warning: {result.stderr}")
    
    def start_trading_bot(self, sectors="IT,HEALTHCARE", interval=300):
        """Start trading bot"""
        print(f"[*] Starting quantitative trading bot...")
        print(f"   Sectors: {sectors}")
        print(f"   Interval: {interval} seconds")

        os.chdir(self.base_dir)

        cmd = [sys.executable, 'live.py',
               '--sectors', sectors,
               '--interval', str(interval)]

        # Start trading bot
        self.bot_process = subprocess.Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)

        # Wait for bot initialization
        time.sleep(3)

        # Check status
        try:
            response = requests.get(f'http://localhost:{self.backend_port}/status',
                                  headers={'Authorization': f'Bearer {self.admin_token}'})
            if response.status_code == 200:
                status = response.json()
                print(f"[OK] Trading bot started successfully")
                print(f"   Status: {status.get('bot', 'unknown')}")
            else:
                print("[WARNING] Unable to retrieve bot status")
        except Exception as e:
            print(f"[WARNING] Status check failed: {e}")
    
    def display_summary(self, sectors, interval):
        """Display startup summary"""
        print(f"""
================================================================
                   Startup Complete!
================================================================
  Dashboard Frontend:  http://localhost:{self.frontend_port}
  API Backend:         http://localhost:{self.backend_port}
  Trading Bot:         Running ({sectors})
  Trading Interval:    {interval} seconds
  Trading Mode:        {'Simulation' if self.dry_run else 'Live Trading'}

  Control Panel Configuration:
     API Base: http://localhost:{self.backend_port}
     Token: {self.admin_token}
================================================================

IMPORTANT REMINDERS:
* Use Ctrl+C to safely stop all services
* Monitor trading status through Dashboard frontend
* Ensure thorough testing before live trading

System ready - Start quantitative trading!
        """)
    
    def cleanup(self):
        """Clean up processes"""
        print("\n[*] Stopping all services...")

        processes = []
        if hasattr(self, 'backend_process'):
            processes.append(('Backend API', self.backend_process))
        if hasattr(self, 'frontend_process'):
            processes.append(('Frontend Interface', self.frontend_process))
        if hasattr(self, 'bot_process'):
            processes.append(('Trading Bot', self.bot_process))

        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"[OK] {name} stopped")
            except:
                try:
                    process.kill()
                    print(f"[FORCE] {name} force stopped")
                except:
                    print(f"[WARNING] Unable to stop {name}")

        print("[OK] Cleanup complete")
    
    def run(self, sectors="IT,HEALTHCARE", interval=300, skip_frontend=False, skip_bot=False):
        """Main run flow"""
        try:
            self.print_banner()
            self.check_dependencies()
            self.load_config()

            if not self.dry_run:
                confirm = input("[WARNING] Will start live trading mode, confirm to continue? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("Startup cancelled")
                    return

            self.kill_existing_processes()

            # Start core services
            self.start_backend()

            if not skip_frontend:
                self.start_frontend()

            # Test API connection
            if not self.dry_run:
                self.test_tiger_api()

            # Start trading bot
            if not skip_bot:
                self.start_trading_bot(sectors, interval)

            # Display summary
            self.display_summary(sectors, interval)

            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nReceived stop signal...")

        except KeyboardInterrupt:
            print("\nUser interrupted")
        except Exception as e:
            print(f"\n[ERROR] Startup failed: {e}")
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Quantitative Trading Bot Launcher')
    parser.add_argument('--sectors', default='IT,HEALTHCARE',
                       help='Trading sectors (default: IT,HEALTHCARE)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Trading interval in seconds (default: 300)')
    parser.add_argument('--skip-frontend', action='store_true',
                       help='Skip starting frontend interface')
    parser.add_argument('--skip-bot', action='store_true',
                       help='Skip starting trading bot')

    args = parser.parse_args()

    launcher = BotLauncher()
    launcher.run(args.sectors, args.interval, args.skip_frontend, args.skip_bot)

if __name__ == '__main__':
    main()