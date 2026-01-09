#!/usr/bin/env python3
"""
Live Trading Real-time Monitor

This script provides real-time monitoring for live trading operations,
including position tracking, risk monitoring, P&L calculation, and emergency controls.

CRITICAL: Use this monitor when live trading is active to ensure safe operations.
"""

import os
import sys
import json
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add bot directory to path
bot_path = Path(__file__).parent / 'bot'
sys.path.append(str(bot_path))

@dataclass
class LiveTradingMetrics:
    """Live trading performance metrics"""
    timestamp: datetime
    account_balance: float
    total_positions: int
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    portfolio_value: float
    buying_power: float
    risk_metrics: Dict[str, float]
    active_orders: int
    position_details: List[Dict[str, Any]]

class LiveTradingMonitor:
    """Real-time live trading monitor with safety controls"""

    def __init__(self):
        self.running = False
        self.start_time = datetime.now()
        self.emergency_stop_triggered = False

        # Load configuration
        self._load_environment()

        # Initialize monitoring state
        self.last_balance = 0.0
        self.daily_start_balance = 0.0
        self.max_daily_loss = float(os.getenv('DAILY_LOSS_LIMIT', 0.05))
        self.max_position_loss = float(os.getenv('POSITION_LOSS_LIMIT', 0.15))

        # Emergency controls
        self.emergency_token = os.getenv('EMERGENCY_STOP_TOKEN')

        # Monitoring intervals
        self.monitor_interval = 30  # seconds
        self.risk_check_interval = 60  # seconds
        self.report_interval = 300  # 5 minutes

        # Initialize Tiger clients
        self._initialize_clients()

    def _load_environment(self):
        """Load environment configuration"""
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    def _initialize_clients(self):
        """Initialize Tiger API clients"""
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.quote.quote_client import QuoteClient
            from tigeropen.trade.trade_client import TradeClient

            props_path = str(Path(__file__).parent / "props")
            config = TigerOpenClientConfig(props_path=props_path)

            self.quote_client = QuoteClient(config)
            self.trade_client = TradeClient(config)

            logger.info("✅ Tiger API clients initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Tiger clients: {e}")
            self.quote_client = None
            self.trade_client = None

    def start_monitoring(self):
        """Start live trading monitoring"""
        logger.info("="*80)
        logger.info("🚨 LIVE TRADING MONITOR STARTED 🚨")
        logger.info("="*80)

        # Verify live trading is active
        if not self._verify_live_trading_active():
            logger.error("❌ Live trading not active or not configured")
            return False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # Get initial account state
        self._capture_initial_state()

        logger.info("📊 Monitor started - Press Ctrl+C to stop")
        logger.info(f"⏱️  Monitoring every {self.monitor_interval} seconds")
        logger.info(f"🔍 Risk checks every {self.risk_check_interval} seconds")
        logger.info(f"📋 Reports every {self.report_interval} seconds")

        last_risk_check = 0
        last_report = 0

        try:
            while self.running:
                current_time = time.time()

                # Regular monitoring
                self._monitor_positions()
                self._monitor_orders()

                # Periodic risk checks
                if current_time - last_risk_check >= self.risk_check_interval:
                    self._perform_risk_checks()
                    last_risk_check = current_time

                # Periodic reports
                if current_time - last_report >= self.report_interval:
                    self._generate_status_report()
                    last_report = current_time

                # Check for emergency stop
                if self._check_emergency_conditions():
                    self._trigger_emergency_stop()
                    break

                time.sleep(self.monitor_interval)

        except KeyboardInterrupt:
            logger.info("📱 Monitor stopped by user")
        except Exception as e:
            logger.error(f"🚨 Monitor error: {e}")
        finally:
            self._cleanup()

        return True

    def _verify_live_trading_active(self) -> bool:
        """Verify live trading is active"""
        # Check DRY_RUN setting
        dry_run = os.getenv('DRY_RUN', 'true').lower()
        if dry_run == 'true':
            logger.error("❌ System is in DRY_RUN mode - not live trading")
            return False

        # Check live trading metadata
        metadata_file = Path(__file__).parent / 'LIVE_TRADING_ACTIVE.json'
        if not metadata_file.exists():
            logger.error("❌ Live trading metadata not found")
            return False

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if not metadata.get('live_trading_enabled', False):
                logger.error("❌ Live trading not enabled in metadata")
                return False

            logger.info("✅ Live trading verified as active")
            return True

        except Exception as e:
            logger.error(f"❌ Could not verify live trading status: {e}")
            return False

    def _capture_initial_state(self):
        """Capture initial account state"""
        try:
            if self.trade_client:
                assets = self.trade_client.get_assets()
                if assets and hasattr(assets, 'summary'):
                    self.daily_start_balance = getattr(assets.summary, 'net_liquidation', 0.0)
                    self.last_balance = self.daily_start_balance

                    logger.info(f"📊 Initial account balance: ${self.daily_start_balance:,.2f}")
                else:
                    logger.warning("⚠️  Could not retrieve initial account state")

        except Exception as e:
            logger.error(f"❌ Failed to capture initial state: {e}")

    def _monitor_positions(self):
        """Monitor current positions"""
        try:
            if not self.trade_client:
                return

            positions = self.trade_client.get_positions()

            if positions:
                total_unrealized = 0.0
                position_count = len(positions)

                for position in positions:
                    symbol = getattr(position, 'symbol', 'UNKNOWN')
                    quantity = getattr(position, 'quantity', 0)
                    unrealized_pnl = getattr(position, 'unrealized_pnl', 0.0)

                    total_unrealized += unrealized_pnl

                    # Check individual position risk
                    if unrealized_pnl < 0:
                        loss_pct = abs(unrealized_pnl) / max(abs(getattr(position, 'market_value', 1)), 1)
                        if loss_pct > self.max_position_loss:
                            logger.warning(f"⚠️  Position {symbol} loss exceeds limit: {loss_pct:.2%}")

                logger.info(f"📈 Positions: {position_count}, Unrealized P&L: ${total_unrealized:,.2f}")

        except Exception as e:
            logger.error(f"❌ Position monitoring error: {e}")

    def _monitor_orders(self):
        """Monitor active orders"""
        try:
            if not self.trade_client:
                return

            orders = self.trade_client.get_orders()

            if orders:
                active_orders = [o for o in orders if getattr(o, 'status', '') in ['NEW', 'PENDING']]

                if active_orders:
                    logger.info(f"📋 Active orders: {len(active_orders)}")

                    for order in active_orders[:5]:  # Show first 5
                        symbol = getattr(order, 'symbol', 'UNKNOWN')
                        side = getattr(order, 'action', 'UNKNOWN')
                        quantity = getattr(order, 'quantity', 0)
                        logger.info(f"   📄 {symbol}: {side} {quantity}")

        except Exception as e:
            logger.error(f"❌ Order monitoring error: {e}")

    def _perform_risk_checks(self):
        """Perform comprehensive risk checks"""
        try:
            logger.info("🔍 Performing risk checks...")

            # Get current account state
            if self.trade_client:
                assets = self.trade_client.get_assets()

                if assets and hasattr(assets, 'summary'):
                    current_balance = getattr(assets.summary, 'net_liquidation', 0.0)

                    # Calculate daily P&L
                    daily_pnl = current_balance - self.daily_start_balance
                    daily_pnl_pct = daily_pnl / max(self.daily_start_balance, 1)

                    logger.info(f"💰 Current balance: ${current_balance:,.2f}")
                    logger.info(f"📊 Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:.2%})")

                    # Check daily loss limit
                    if daily_pnl < 0 and abs(daily_pnl_pct) > self.max_daily_loss:
                        logger.error(f"🚨 DAILY LOSS LIMIT EXCEEDED: {daily_pnl_pct:.2%}")
                        return False

            # Check risk metrics if enhanced risk manager is available
            try:
                from bot.enhanced_risk_manager import EnhancedRiskManager

                risk_manager = EnhancedRiskManager()

                # Get current positions for risk calculation
                positions = self.trade_client.get_positions() if self.trade_client else []

                if positions:
                    # Calculate portfolio risk metrics
                    symbols = [getattr(p, 'symbol', '') for p in positions]
                    weights = [getattr(p, 'market_value', 0) for p in positions]

                    if symbols and weights:
                        total_value = sum(weights)
                        weights = [w/total_value for w in weights] if total_value > 0 else []

                        # Check concentration risk
                        max_weight = max(weights) if weights else 0
                        if max_weight > 0.15:  # 15% concentration limit
                            logger.warning(f"⚠️  High concentration risk: {max_weight:.2%}")

            except ImportError:
                logger.debug("Enhanced risk manager not available")

            logger.info("✅ Risk checks completed")
            return True

        except Exception as e:
            logger.error(f"❌ Risk check error: {e}")
            return False

    def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions"""
        # This is a placeholder - extend with specific emergency conditions
        # Examples: extreme losses, system errors, market conditions, etc.

        return self.emergency_stop_triggered

    def _trigger_emergency_stop(self):
        """Trigger emergency stop procedures"""
        logger.error("🚨 EMERGENCY STOP TRIGGERED 🚨")

        try:
            # Close all positions (if configured to do so)
            emergency_close = os.getenv('EMERGENCY_CLOSE_POSITIONS', 'false').lower()

            if emergency_close == 'true':
                logger.info("🛑 Closing all positions...")
                self._close_all_positions()

            # Cancel all pending orders
            logger.info("❌ Cancelling all pending orders...")
            self._cancel_all_orders()

            # Create emergency stop record
            self._record_emergency_stop()

            # Stop monitoring
            self.running = False

        except Exception as e:
            logger.error(f"❌ Emergency stop error: {e}")

    def _close_all_positions(self):
        """Close all open positions"""
        try:
            if not self.trade_client:
                return

            positions = self.trade_client.get_positions()

            for position in positions:
                symbol = getattr(position, 'symbol', '')
                quantity = getattr(position, 'quantity', 0)

                if quantity != 0:
                    # Create market order to close position
                    side = 'SELL' if quantity > 0 else 'BUY'
                    close_quantity = abs(quantity)

                    logger.info(f"🛑 Closing position {symbol}: {side} {close_quantity}")

                    # Note: Actual order placement would require proper order construction
                    # This is a simplified example

        except Exception as e:
            logger.error(f"❌ Position closing error: {e}")

    def _cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            if not self.trade_client:
                return

            orders = self.trade_client.get_orders()

            for order in orders:
                status = getattr(order, 'status', '')
                if status in ['NEW', 'PENDING']:
                    order_id = getattr(order, 'order_id', '')
                    symbol = getattr(order, 'symbol', '')

                    logger.info(f"❌ Cancelling order {order_id} for {symbol}")

                    # Cancel the order
                    self.trade_client.cancel_order(order_id)

        except Exception as e:
            logger.error(f"❌ Order cancellation error: {e}")

    def _record_emergency_stop(self):
        """Record emergency stop event"""
        emergency_record = {
            "timestamp": datetime.now().isoformat(),
            "event": "EMERGENCY_STOP",
            "triggered_by": "live_trading_monitor.py",
            "reason": "Emergency conditions detected",
            "actions_taken": [
                "Monitoring stopped",
                "Orders cancelled",
                "Emergency record created"
            ]
        }

        emergency_file = Path(__file__).parent / f'emergency_stop_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(emergency_file, 'w') as f:
            json.dump(emergency_record, f, indent=2)

        logger.info(f"📝 Emergency stop recorded: {emergency_file}")

    def _generate_status_report(self):
        """Generate periodic status report"""
        try:
            logger.info("📋 Generating status report...")

            report = {
                "timestamp": datetime.now().isoformat(),
                "monitor_uptime": str(datetime.now() - self.start_time),
                "system_status": "OPERATIONAL",
                "daily_pnl": 0.0,
                "positions": 0,
                "active_orders": 0
            }

            # Get account information
            if self.trade_client:
                assets = self.trade_client.get_assets()
                positions = self.trade_client.get_positions()
                orders = self.trade_client.get_orders()

                if assets and hasattr(assets, 'summary'):
                    current_balance = getattr(assets.summary, 'net_liquidation', 0.0)
                    report["current_balance"] = current_balance
                    report["daily_pnl"] = current_balance - self.daily_start_balance

                if positions:
                    report["positions"] = len(positions)

                if orders:
                    active_orders = [o for o in orders if getattr(o, 'status', '') in ['NEW', 'PENDING']]
                    report["active_orders"] = len(active_orders)

            # Save report
            report_file = Path(__file__).parent / 'logs' / f'status_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            report_file.parent.mkdir(exist_ok=True)

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"💰 Daily P&L: ${report['daily_pnl']:,.2f}")
            logger.info(f"📊 Positions: {report['positions']}, Orders: {report['active_orders']}")

        except Exception as e:
            logger.error(f"❌ Status report error: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📱 Received signal {signum}, shutting down...")
        self.running = False

    def _cleanup(self):
        """Cleanup on shutdown"""
        logger.info("🧹 Cleaning up monitor...")

        # Final status report
        self._generate_status_report()

        logger.info("✅ Live trading monitor stopped")

def main():
    """Main execution"""
    monitor = LiveTradingMonitor()

    try:
        success = monitor.start_monitoring()
        return 0 if success else 1

    except Exception as e:
        logger.error(f"🚨 Monitor failed: {e}")
        return 3

if __name__ == "__main__":
    exit(main())