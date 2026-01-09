"""
Strategy Monitoring Dashboard - Real-time Strategy Performance Dashboard

Streamlit-based dashboard for monitoring improved strategies V2 performance.
Displays real-time metrics, performance trends, and strategy comparisons.

Usage:
    streamlit run strategy_monitor_dashboard.py
    streamlit run strategy_monitor_dashboard.py --server.port 8503
"""

import os
import sys
import json
import csv
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess

# Import monitoring tools
from compare_strategies import StrategyComparer
from performance_tracker import PerformanceTracker
from weekly_report_generator import WeeklyReportGenerator


class StrategyMonitorDashboard:
    """Real-time strategy monitoring dashboard."""

    def __init__(self):
        """Initialize dashboard."""
        self.status_file = os.path.join(
            os.path.dirname(__file__),
            'dashboard', 'state', 'status.json'
        )
        self.csv_file = 'performance_tracking.csv'
        self.reports_dir = 'reports'

    def load_status(self) -> Dict[str, Any]:
        """Load current status from status.json."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Error loading status: {e}")
            return {}

    def load_performance_data(self, days: int = 30) -> pd.DataFrame:
        """Load performance tracking data."""
        try:
            if not os.path.exists(self.csv_file):
                return pd.DataFrame()

            df = pd.read_csv(self.csv_file)
            df['date'] = pd.to_datetime(df['date'])

            # Get recent data
            if days > 0:
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[df['date'] >= cutoff_date]

            return df.sort_values('date')

        except Exception as e:
            st.error(f"Error loading performance data: {e}")
            return pd.DataFrame()

    def display_key_metrics(self, status: Dict[str, Any]):
        """Display key metrics in columns."""
        try:
            rt_metrics = status.get('real_time_metrics', {})
            portfolio_value = status.get('real_portfolio_value', 0)
            selection_results = status.get('selection_results', {})

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Portfolio Value",
                    f"${portfolio_value:,.2f}",
                    delta=f"{rt_metrics.get('daily_pnl', 0):+,.2f}"
                )

            with col2:
                sharpe = rt_metrics.get('sharpe_ratio_ytd', 0)
                st.metric(
                    "Sharpe Ratio YTD",
                    f"{sharpe:.2f}",
                    delta=None
                )

            with col3:
                drawdown = rt_metrics.get('current_drawdown', 0)
                st.metric(
                    "Current Drawdown",
                    f"{drawdown:.2%}",
                    delta=None,
                    delta_color="inverse"
                )

            with col4:
                num_selections = selection_results.get('total_selections', 0)
                st.metric(
                    "Active Selections",
                    f"{num_selections}",
                    delta=None
                )

        except Exception as e:
            st.error(f"Error displaying metrics: {e}")

    def display_performance_charts(self, df: pd.DataFrame):
        """Display performance charts."""
        try:
            if df.empty:
                st.warning("No performance data available for charts")
                return

            st.subheader("Performance Trends")

            # Portfolio value chart
            st.line_chart(df.set_index('date')['portfolio_value'])
            st.caption("Portfolio Value Over Time")

            # Returns chart
            if 'daily_return' in df.columns:
                st.bar_chart(df.set_index('date')['daily_return'])
                st.caption("Daily Returns")

            # Sharpe ratio chart
            if 'sharpe_ratio_ytd' in df.columns:
                st.line_chart(df.set_index('date')['sharpe_ratio_ytd'])
                st.caption("Sharpe Ratio YTD")

        except Exception as e:
            st.error(f"Error displaying charts: {e}")

    def display_risk_metrics(self, status: Dict[str, Any]):
        """Display risk metrics."""
        try:
            st.subheader("Risk Metrics")

            rt_metrics = status.get('real_time_metrics', {})
            crowding = status.get('factor_crowding_analysis', {})

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Portfolio Risk**")
                st.write(f"- ES@97.5%: {rt_metrics.get('portfolio_es_975', 0):.4f}")
                st.write(f"- Risk Budget Util: {rt_metrics.get('risk_budget_utilization', 0):.2%}")
                st.write(f"- Max Drawdown YTD: {rt_metrics.get('max_drawdown_ytd', 0):.2%}")
                st.write(f"- Tail Dependence: {rt_metrics.get('tail_dependence', 0):.4f}")

            with col2:
                st.write("**Factor Crowding**")
                st.write(f"- Crowding Level: {crowding.get('crowding_level', 'UNKNOWN')}")
                st.write(f"- Factor HHI: {rt_metrics.get('factor_hhi', 0):.4f}")
                st.write(f"- Max Correlation: {rt_metrics.get('max_correlation', 0):.4f}")
                st.write(f"- Crowding Risk Score: {rt_metrics.get('crowding_risk_score', 0):.4f}")

        except Exception as e:
            st.error(f"Error displaying risk metrics: {e}")

    def display_selection_info(self, status: Dict[str, Any]):
        """Display selection information."""
        try:
            st.subheader("Current Selections")

            selection_results = status.get('selection_results', {})
            top_picks = selection_results.get('top_picks', [])

            if not top_picks:
                st.info("No current selections available")
                return

            # Create dataframe for display
            df = pd.DataFrame(top_picks)

            # Select and rename columns for display
            display_cols = ['symbol', 'avg_score', 'dominant_action', 'strategy_count']
            if all(col in df.columns for col in display_cols):
                df_display = df[display_cols].copy()
                df_display.columns = ['Symbol', 'Score', 'Action', 'Strategies']
                df_display['Score'] = df_display['Score'].round(2)

                st.dataframe(df_display, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying selections: {e}")

    def display_compliance_status(self, status: Dict[str, Any]):
        """Display compliance status."""
        try:
            st.subheader("Compliance Status")

            compliance_status = status.get('compliance_status', 'UNKNOWN')
            violations = status.get('compliance_violations', [])

            if compliance_status == 'COMPLIANT':
                st.success(f"Status: {compliance_status}")
            else:
                st.error(f"Status: {compliance_status}")

            if violations:
                st.warning(f"Found {len(violations)} violation(s)")
                for violation in violations[:5]:  # Show first 5
                    st.write(f"- {violation.get('rule_id', 'N/A')}: {violation.get('message', 'N/A')}")
            else:
                st.info("No compliance violations")

        except Exception as e:
            st.error(f"Error displaying compliance: {e}")

    def display_system_info(self, status: Dict[str, Any]):
        """Display system information."""
        try:
            st.subheader("System Information")

            market_regime = status.get('market_regime', {})
            rt_metrics = status.get('real_time_metrics', {})

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Market Status**")
                st.write(f"- Regime: {market_regime.get('regime', 'unknown')}")
                st.write(f"- Confidence: {market_regime.get('confidence', 0):.2%}")
                st.write(f"- System Uptime: {rt_metrics.get('system_uptime', 0):.1f}h")

            with col2:
                st.write("**Data Quality**")
                st.write(f"- Data Freshness: {rt_metrics.get('data_freshness', 0):.1f}min")
                st.write(f"- Active Positions: {rt_metrics.get('active_positions', 0)}")
                st.write(f"- Last Update: {status.get('timestamp', 'N/A')}")

        except Exception as e:
            st.error(f"Error displaying system info: {e}")


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Strategy Monitor",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Strategy Monitoring Dashboard")
    st.markdown("Real-time monitoring of Improved Strategies V2")

    # Initialize dashboard
    dashboard = StrategyMonitorDashboard()

    # Sidebar controls
    st.sidebar.title("Controls")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Time range selector
    days_range = st.sidebar.slider("Performance History (days)", 7, 90, 30)

    # Manual actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Actions")

    if st.sidebar.button("📊 Run Strategy Comparison"):
        with st.spinner("Running comparison..."):
            try:
                comparer = StrategyComparer()
                status = comparer.load_status()
                current_info = comparer.extract_selection_info(status)
                comparison = comparer.compare_selections(current_info)
                comparer.generate_report(comparison, 'comparison_report.json')
                st.sidebar.success("Comparison completed!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    if st.sidebar.button("📈 Record Daily Metrics"):
        with st.spinner("Recording metrics..."):
            try:
                tracker = PerformanceTracker()
                success = tracker.run()
                if success:
                    st.sidebar.success("Metrics recorded!")
                else:
                    st.sidebar.error("Failed to record metrics")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    if st.sidebar.button("📝 Generate Weekly Report"):
        with st.spinner("Generating report..."):
            try:
                generator = WeeklyReportGenerator()
                success = generator.run()
                if success:
                    st.sidebar.success("Report generated!")
                else:
                    st.sidebar.error("Failed to generate report")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    # Auto-refresh settings
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60)
        st.sidebar.info(f"Auto-refreshing every {refresh_interval}s")

    # Main content
    try:
        # Load data
        status = dashboard.load_status()
        df = dashboard.load_performance_data(days=days_range)

        if not status:
            st.error("Unable to load system status")
            st.info("Make sure the trading system is running and status.json is available")
            return

        # Display sections
        with st.container():
            dashboard.display_key_metrics(status)

        st.markdown("---")

        # Two-column layout
        col1, col2 = st.columns([2, 1])

        with col1:
            if not df.empty:
                dashboard.display_performance_charts(df)
            else:
                st.info("No performance history available yet")

        with col2:
            dashboard.display_risk_metrics(status)

        st.markdown("---")

        # Selection and compliance
        col3, col4 = st.columns(2)

        with col3:
            dashboard.display_selection_info(status)

        with col4:
            dashboard.display_compliance_status(status)

        st.markdown("---")

        # System info
        dashboard.display_system_info(status)

        # Performance statistics
        if not df.empty:
            st.markdown("---")
            st.subheader("Performance Statistics")

            col5, col6, col7, col8 = st.columns(4)

            with col5:
                if 'weekly_return' in df.columns or 'daily_return' in df.columns:
                    returns = df['daily_return'] if 'daily_return' in df.columns else df['weekly_return']
                    total_return = returns.sum()
                    st.metric("Cumulative Return", f"{total_return:.2%}")

            with col6:
                if 'daily_return' in df.columns:
                    volatility = df['daily_return'].std()
                    st.metric("Daily Volatility", f"{volatility:.4%}")

            with col7:
                if 'daily_return' in df.columns:
                    wins = (df['daily_return'] > 0).sum()
                    total = len(df)
                    win_rate = wins / total if total > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.2%}")

            with col8:
                if 'sharpe_ratio_ytd' in df.columns:
                    latest_sharpe = df['sharpe_ratio_ytd'].iloc[-1]
                    st.metric("Latest Sharpe", f"{latest_sharpe:.2f}")

        # Footer
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()
