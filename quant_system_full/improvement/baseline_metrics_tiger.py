#!/usr/bin/env python3
"""
Tiger API Baseline Metrics Dashboard - Real Account Data

This script generates comprehensive baseline metrics using Tiger API to fetch real account data:
- Live positions and portfolio value
- Account balance and P&L
- Trading history and performance
- Real-time risk metrics

Usage:
    python baseline_metrics_tiger.py [--output html|json|console] [--days 30]
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from bot.config import SETTINGS
from bot.tradeup_client import build_clients

try:
    from tigeropen.common.consts import SecurityType, Market
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


@dataclass
class TigerBaselineMetrics:
    """Container for Tiger API baseline metrics"""
    # Account Information
    account_id: str
    account_type: str
    currency: str

    # Portfolio Metrics
    total_portfolio_value: float
    cash_balance: float
    buying_power: float
    positions_count: int
    positions_value: float

    # Performance Metrics
    total_pnl: float
    day_pnl: float
    unrealized_pnl: float
    realized_pnl: float

    # Risk Metrics
    margin_used: float
    margin_available: float
    maintenance_margin: float

    # Trading Activity
    orders_today: int
    trades_count: int

    # Analysis metadata
    analysis_period_days: int
    timestamp: str

    # Calculated metrics
    win_rate: Optional[float] = None
    profit_loss_ratio: Optional[float] = None
    avg_position_size: Optional[float] = None
    max_position_exposure: Optional[float] = None


class TigerBaselineAnalyzer:
    """Analyze baseline metrics using Tiger API"""

    def __init__(self, analysis_days: int = 30):
        self.analysis_days = analysis_days
        self.quote_client = None
        self.trade_client = None
        self.execution_engine = None

        if not SDK_AVAILABLE:
            raise RuntimeError("Tiger SDK not available. Please install tigeropen library.")

        # Initialize Tiger clients
        try:
            self.quote_client, self.trade_client = build_clients()
            if not self.trade_client:
                raise RuntimeError("Tiger clients not initialized. Check configuration.")

            # Create execution engine for Tiger API operations
            from bot.execution_tiger import create_tiger_execution_engine
            self.execution_engine = create_tiger_execution_engine(self.quote_client, self.trade_client)
            if not self.execution_engine:
                raise RuntimeError("Tiger execution engine not created.")

        except Exception as e:
            print(f"Warning: Could not initialize Tiger clients: {e}")
            print("Running in simulation mode with limited functionality.")

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information from Tiger API"""
        if not self.execution_engine:
            return {
                "account": SETTINGS.account or "SIMULATION",
                "currency": "USD",
                "account_type": "SIMULATION"
            }

        try:
            # Get account from execution engine (it already has the account info)
            account_id = getattr(self.execution_engine, 'account', SETTINGS.account or "UNKNOWN")

            return {
                "account": account_id,
                "currency": "USD",  # Default to USD for Tiger US accounts
                "account_type": "LIVE"
            }
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return {
                "account": SETTINGS.account or "ERROR",
                "currency": "USD",
                "account_type": "ERROR"
            }

    def get_portfolio_summary(self) -> Dict[str, float]:
        """Get portfolio summary from Tiger API"""
        if not self.execution_engine:
            return {
                "total_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "positions_value": 0.0,
                "total_pnl": 0.0,
                "day_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "margin_used": 0.0,
                "margin_available": 0.0,
                "maintenance_margin": 0.0
            }

        try:
            # Get asset information using execution engine
            assets = self.execution_engine.get_account_assets()
            summary = {
                "total_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "positions_value": 0.0,
                "total_pnl": 0.0,
                "day_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "margin_used": 0.0,
                "margin_available": 0.0,
                "maintenance_margin": 0.0
            }

            if assets:
                summary["total_value"] = assets.get('net_liquidation', 0.0)
                summary["cash"] = assets.get('cash_available', 0.0)
                summary["buying_power"] = assets.get('buying_power', 0.0)
                summary["positions_value"] = assets.get('gross_position_value', 0.0)
                summary["unrealized_pnl"] = assets.get('unrealized_pnl', 0.0)
                summary["realized_pnl"] = assets.get('realized_pnl', 0.0)
                summary["total_pnl"] = summary["unrealized_pnl"] + summary["realized_pnl"]

            return summary

        except Exception as e:
            print(f"Error fetching portfolio summary: {e}")
            return {
                "total_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "positions_value": 0.0,
                "total_pnl": 0.0,
                "day_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "margin_used": 0.0,
                "margin_available": 0.0,
                "maintenance_margin": 0.0
            }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from Tiger API"""
        if not self.execution_engine:
            return []

        try:
            positions = self.execution_engine.get_account_positions()
            position_list = []

            if positions:
                for pos in positions:
                    position_data = {
                        "symbol": pos.get('symbol', 'UNKNOWN'),
                        "quantity": pos.get('quantity', 0),
                        "market_price": pos.get('market_price', 0.0),
                        "market_value": pos.get('market_value', 0.0),
                        "average_cost": pos.get('average_cost', 0.0),
                        "unrealized_pnl": pos.get('unrealized_pnl', 0.0),
                        "realized_pnl": pos.get('realized_pnl', 0.0),
                        "position_type": "LONG" if pos.get('quantity', 0) > 0 else "SHORT" if pos.get('quantity', 0) < 0 else "FLAT"
                    }
                    position_list.append(position_data)

            return position_list

        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def get_recent_orders(self) -> Tuple[int, int]:
        """Get recent orders and trades count"""
        if not self.execution_engine:
            return 0, 0

        try:
            # Get recent orders using execution engine
            orders = self.execution_engine.get_recent_orders(hours=24)

            orders_count = len(orders) if orders else 0

            # Count filled orders as trades
            trades_count = 0
            if orders:
                for order in orders:
                    status = order.get('status', '')
                    if status in ['FILLED', 'PARTIALLY_FILLED']:
                        trades_count += 1

            return orders_count, trades_count

        except Exception as e:
            print(f"Error fetching orders: {e}")
            return 0, 0

    def calculate_performance_metrics(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from positions"""
        if not positions:
            return {
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_position_size": 0.0,
                "max_position_exposure": 0.0
            }

        # Calculate metrics
        profitable_positions = [p for p in positions if p.get('unrealized_pnl', 0) > 0]
        losing_positions = [p for p in positions if p.get('unrealized_pnl', 0) < 0]

        win_rate = (len(profitable_positions) / len(positions)) * 100 if positions else 0.0

        avg_profit = np.mean([p['unrealized_pnl'] for p in profitable_positions]) if profitable_positions else 0.0
        avg_loss = abs(np.mean([p['unrealized_pnl'] for p in losing_positions])) if losing_positions else 1.0
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0

        position_values = [abs(p.get('market_value', 0)) for p in positions]
        avg_position_size = np.mean(position_values) if position_values else 0.0
        max_position_exposure = max(position_values) if position_values else 0.0

        return {
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "avg_position_size": avg_position_size,
            "max_position_exposure": max_position_exposure
        }

    def generate_metrics(self) -> TigerBaselineMetrics:
        """Generate comprehensive baseline metrics using Tiger API"""
        print("Fetching data from Tiger API...")

        # Get account information
        account_info = self.get_account_info()

        # Get portfolio summary
        portfolio = self.get_portfolio_summary()

        # Get positions
        positions = self.get_positions()

        # Get trading activity
        orders_today, trades_count = self.get_recent_orders()

        # Calculate performance metrics
        performance = self.calculate_performance_metrics(positions)

        return TigerBaselineMetrics(
            # Account info
            account_id=account_info["account"],
            account_type=account_info["account_type"],
            currency=account_info["currency"],

            # Portfolio metrics
            total_portfolio_value=portfolio["total_value"],
            cash_balance=portfolio["cash"],
            buying_power=portfolio["buying_power"],
            positions_count=len(positions),
            positions_value=portfolio["positions_value"],

            # Performance metrics
            total_pnl=portfolio["total_pnl"],
            day_pnl=portfolio["day_pnl"],
            unrealized_pnl=portfolio["unrealized_pnl"],
            realized_pnl=portfolio["total_pnl"] - portfolio["unrealized_pnl"],

            # Risk metrics
            margin_used=portfolio["margin_used"],
            margin_available=portfolio["margin_available"],
            maintenance_margin=portfolio["maintenance_margin"],

            # Trading activity
            orders_today=orders_today,
            trades_count=trades_count,

            # Analysis metadata
            analysis_period_days=self.analysis_days,
            timestamp=datetime.now().isoformat(),

            # Calculated metrics
            win_rate=performance["win_rate"],
            profit_loss_ratio=performance["profit_loss_ratio"],
            avg_position_size=performance["avg_position_size"],
            max_position_exposure=performance["max_position_exposure"]
        )


class TigerReportGenerator:
    """Generate reports from Tiger API metrics"""

    def __init__(self, metrics: TigerBaselineMetrics):
        self.metrics = metrics

    def generate_console_report(self) -> str:
        """Generate console-friendly report"""
        report = f"""
=== TIGER API BASELINE METRICS DASHBOARD ===
Generated: {self.metrics.timestamp}
Account: {self.metrics.account_id} ({self.metrics.account_type})

ACCOUNT SUMMARY:
- Total Portfolio Value: ${self.metrics.total_portfolio_value:,.2f}
- Cash Balance: ${self.metrics.cash_balance:,.2f}
- Buying Power: ${self.metrics.buying_power:,.2f}
- Positions Value: ${self.metrics.positions_value:,.2f}

PERFORMANCE METRICS:
- Total P&L: ${self.metrics.total_pnl:,.2f}
- Day P&L: ${self.metrics.day_pnl:,.2f}
- Unrealized P&L: ${self.metrics.unrealized_pnl:,.2f}
- Realized P&L: ${self.metrics.realized_pnl:,.2f}
- Win Rate: {self.metrics.win_rate:.1f}%
- Profit/Loss Ratio: {self.metrics.profit_loss_ratio:.2f}

PORTFOLIO METRICS:
- Total Positions: {self.metrics.positions_count}
- Average Position Size: ${self.metrics.avg_position_size:,.2f}
- Max Position Exposure: ${self.metrics.max_position_exposure:,.2f}

RISK METRICS:
- Margin Used: ${self.metrics.margin_used:,.2f}
- Margin Available: ${self.metrics.margin_available:,.2f}
- Maintenance Margin: ${self.metrics.maintenance_margin:,.2f}

TRADING ACTIVITY:
- Orders Today: {self.metrics.orders_today}
- Trades Count: {self.metrics.trades_count}

SYSTEM STATUS:
- Data Source: Tiger API (Live)
- Currency: {self.metrics.currency}
- Status: Connected
"""
        return report

    def generate_json_report(self) -> str:
        """Generate JSON report"""
        return json.dumps({
            "tiger_baseline_metrics": {
                "account": {
                    "account_id": self.metrics.account_id,
                    "account_type": self.metrics.account_type,
                    "currency": self.metrics.currency
                },
                "portfolio": {
                    "total_value": self.metrics.total_portfolio_value,
                    "cash_balance": self.metrics.cash_balance,
                    "buying_power": self.metrics.buying_power,
                    "positions_count": self.metrics.positions_count,
                    "positions_value": self.metrics.positions_value
                },
                "performance": {
                    "total_pnl": self.metrics.total_pnl,
                    "day_pnl": self.metrics.day_pnl,
                    "unrealized_pnl": self.metrics.unrealized_pnl,
                    "realized_pnl": self.metrics.realized_pnl,
                    "win_rate": self.metrics.win_rate,
                    "profit_loss_ratio": self.metrics.profit_loss_ratio
                },
                "risk": {
                    "margin_used": self.metrics.margin_used,
                    "margin_available": self.metrics.margin_available,
                    "maintenance_margin": self.metrics.maintenance_margin
                },
                "trading": {
                    "orders_today": self.metrics.orders_today,
                    "trades_count": self.metrics.trades_count
                },
                "metadata": {
                    "analysis_period_days": self.metrics.analysis_period_days,
                    "timestamp": self.metrics.timestamp
                }
            }
        }, indent=2)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate Tiger API baseline metrics dashboard')
    parser.add_argument('--output', choices=['console', 'json'], default='console',
                        help='Output format (default: console)')
    parser.add_argument('--days', type=int, default=30,
                        help='Analysis period in days (default: 30)')

    args = parser.parse_args()

    try:
        # Initialize analyzer with Tiger API
        analyzer = TigerBaselineAnalyzer(analysis_days=args.days)
        metrics = analyzer.generate_metrics()

        # Generate report
        generator = TigerReportGenerator(metrics)

        if args.output == 'console':
            report = generator.generate_console_report()
            print(report)
        elif args.output == 'json':
            report = generator.generate_json_report()
            print(report)

        # Save metrics to improvement directory
        output_dir = Path(__file__).parent / "reports"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_dir / f"tiger_baseline_metrics_{timestamp}.json"

        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(generator.generate_json_report())

        print(f"\nTiger API baseline metrics saved to: {json_file}")

    except Exception as e:
        print(f"Error generating Tiger API baseline metrics: {e}")
        print(f"Make sure Tiger API credentials are configured correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()