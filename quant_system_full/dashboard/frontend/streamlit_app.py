"""
Enhanced Quant Trading Dashboard with Intelligent Stock Selection
================================================================

This enhanced dashboard provides comprehensive monitoring and control capabilities
for the quantitative trading system, including:

- Real-time portfolio monitoring with live P&L charts
- Multi-factor scoring visualization and analysis
- Interactive stock screening controls
- Real-time trade execution monitoring
- Portfolio performance analytics
- Risk management dashboard

Features:
- Professional charts using Plotly and Seaborn
- Mobile-responsive layouts
- Real-time data updates
- Interactive controls for risk parameters
- Multi-page navigation for different functionalities
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Quant Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE = st.sidebar.text_input("API Base", os.getenv("API_BASE", "http://localhost:8000"))
TOKEN = st.sidebar.text_input("Bearer Token (ADMIN_TOKEN)", os.getenv("ADMIN_TOKEN", "changeme"), type="password")
AUTO_REFRESH = st.sidebar.checkbox("Auto refresh", True)
REFRESH_SEC = st.sidebar.slider("Refresh interval (seconds)", 2, 60, 5)

# Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Stock Screening", "Portfolio Analytics", "Risk Management", "Performance Analysis", "System Logs"]
)

def headers():
    """Get HTTP headers for API requests."""
    if TOKEN and TOKEN != "changeme":
        return {"Authorization": f"Bearer {TOKEN}"}
    return {}

def call_api(endpoint: str, method: str = "GET", json_data: dict = None) -> dict:
    """Make API call with error handling."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, headers=headers(), timeout=10)
        else:
            r = requests.post(url, headers=headers(), json=json_data, timeout=10)
        
        if r.ok:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error"}
    except Exception as e:
        return {"error": str(e)}

def format_currency(value: float) -> str:
    """Format currency values."""
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format percentage values."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 100, 
                      threshold_low: float = 30, threshold_high: float = 70) -> go.Figure:
    """Create a gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        delta = {'reference': (threshold_low + threshold_high) / 2},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, threshold_low], 'color': "lightgray"},
                {'range': [threshold_low, threshold_high], 'color': "gray"},
                {'range': [threshold_high, max_val], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_factor_radar_chart(factor_scores: dict, symbol: str) -> go.Figure:
    """Create radar chart for factor scores."""
    factors = list(factor_scores.keys())
    values = list(factor_scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=factors,
        fill='toself',
        name=symbol,
        line_color='rgb(1,90,200)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title=f"Factor Analysis - {symbol}",
        height=400
    )
    
    return fig

def create_portfolio_treemap(positions: List[dict]) -> go.Figure:
    """Create treemap visualization of portfolio positions."""
    if not positions:
        return go.Figure()
    
    df = pd.DataFrame(positions)
    if df.empty:
        return go.Figure()
    
    # Calculate position values
    df['abs_value'] = abs(df.get('quantity', 1) * df.get('current_price', 100))
    df['color_val'] = df.get('pnl_percent', 0)
    
    fig = go.Figure(go.Treemap(
        labels=df.get('symbol', ['Unknown']),
        values=df['abs_value'],
        parents=[""] * len(df),
        textinfo="label+value+percent parent",
        marker=dict(
            colorscale='RdYlGn',
            cmid=0,
            colorbar=dict(title="P&L %"),
            colors=df['color_val']
        )
    ))
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def display_control_buttons():
    """Display system control buttons."""
    st.markdown("### System Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔴 Emergency STOP", use_container_width=True):
            result = call_api("/kill", method="POST", json_data={"reason": "manual via UI"})
            if "error" not in result:
                st.success("✅ Emergency stop activated")
            else:
                st.error(f"❌ Error: {result['error']}")
    
    with col2:
        if st.button("🟢 Resume Trading", use_container_width=True):
            result = call_api("/resume", method="POST", json_data={"note": "resume via UI"})
            if "error" not in result:
                st.success("✅ Trading resumed")
            else:
                st.error(f"❌ Error: {result['error']}")
    
    with col3:
        if st.button("📝 Generate Summary", use_container_width=True):
            result = call_api("/summary")
            if "error" not in result:
                st.success("✅ Daily summary generated")
                if "report" in result:
                    st.text_area("Summary", result["report"], height=200)
            else:
                st.error(f"❌ Error: {result['error']}")
    
    with col4:
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.rerun()

def display_overview_page():
    """Display the main overview dashboard."""
    st.markdown('<h1 class="main-header">📊 Quantitative Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Control buttons
    display_control_buttons()
    
    # Get system status
    status = call_api("/status")
    
    if "error" in status:
        st.error(f"❌ Cannot connect to backend: {status['error']}")
        return
    
    # System status metrics
    st.markdown("### System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bot_status = status.get("bot", "unknown")
        status_color = "🟢" if bot_status == "running" else "🟡" if bot_status == "paused" else "🔴"
        st.metric("Bot Status", f"{status_color} {bot_status.title()}")
    
    with col2:
        is_paused = status.get("paused", False)
        pause_status = "🔴 Paused" if is_paused else "🟢 Active"
        st.metric("Trading Status", pause_status)
    
    with col3:
        heartbeat = status.get("heartbeat")
        if heartbeat:
            last_seen = datetime.fromtimestamp(heartbeat)
            minutes_ago = int((datetime.now() - last_seen).total_seconds() / 60)
            st.metric("Last Heartbeat", f"{minutes_ago} min ago")
        else:
            st.metric("Last Heartbeat", "N/A")
    
    with col4:
        pnl = status.get("pnl", 0)
        pnl_color = "normal" if pnl >= 0 else "inverse"
        st.metric("Total P&L", format_currency(pnl), delta_color=pnl_color)
    
    # Positions overview
    st.markdown("### Portfolio Overview")
    
    positions = status.get("positions", [])
    if positions:
        df_positions = pd.DataFrame(positions)
        
        # Portfolio summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_portfolio_treemap(positions), use_container_width=True)
        
        with col2:
            # Portfolio metrics
            total_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) for pos in positions)
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
            
            st.metric("Portfolio Value", format_currency(total_value))
            st.metric("Unrealized P&L", format_currency(total_pnl))
            st.metric("Number of Positions", len(positions))
            
            # Top positions
            if len(df_positions) > 0:
                st.markdown("**Top Positions:**")
                for _, pos in df_positions.head(5).iterrows():
                    pnl_pct = pos.get('pnl_percent', 0)
                    color = "🟢" if pnl_pct >= 0 else "🔴"
                    st.write(f"{color} {pos.get('symbol', 'N/A')}: {format_percentage(pnl_pct)}")
        
        # Detailed positions table
        st.markdown("### Position Details")
        st.dataframe(df_positions, use_container_width=True)
    
    else:
        st.info("📊 No active positions")
    
    # Recent activity
    st.markdown("### Recent Activity")
    logs_response = call_api("/logs?n=50")
    if "error" not in logs_response and "lines" in logs_response:
        recent_logs = logs_response["lines"][-10:]  # Show last 10 entries
        
        for log_line in recent_logs:
            if any(keyword in log_line for keyword in ["BUY", "SELL"]):
                st.success(f"📈 {log_line}")
            elif "ERROR" in log_line:
                st.error(f"❌ {log_line}")
            elif "WARNING" in log_line:
                st.warning(f"⚠️ {log_line}")
            else:
                st.info(f"ℹ️ {log_line}")
    else:
        st.info("No recent activity logs available")

def display_stock_screening_page():
    """Display stock screening interface."""
    st.markdown('<h1 class="main-header">🔍 Intelligent Stock Screening</h1>', unsafe_allow_html=True)
    
    # Screening controls
    st.markdown("### Screening Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.number_input("Number of stocks to select", min_value=5, max_value=100, value=20)
        sectors = st.multiselect("Select sectors", 
                                ["Technology", "Healthcare", "Financial", "Energy", "Consumer", "Industrial"])
        min_market_cap = st.number_input("Min Market Cap (B)", min_value=0.1, max_value=1000.0, value=1.0)
    
    with col2:
        min_price = st.number_input("Min Stock Price", min_value=1.0, max_value=1000.0, value=5.0)
        max_price = st.number_input("Max Stock Price", min_value=10.0, max_value=5000.0, value=1000.0)
        min_volume = st.number_input("Min Daily Volume", min_value=10000, max_value=10000000, value=100000)
    
    with col3:
        st.markdown("**Factor Weights**")
        valuation_weight = st.slider("Valuation", 0.0, 1.0, 0.25, 0.05)
        volume_weight = st.slider("Volume", 0.0, 1.0, 0.20, 0.05)
        momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.20, 0.05)
        technical_weight = st.slider("Technical", 0.0, 1.0, 0.20, 0.05)
        quality_weight = st.slider("Quality", 0.0, 1.0, 0.15, 0.05)
    
    # Screening simulation (mock data for demonstration)
    if st.button("🔍 Run Stock Screening", use_container_width=True):
        with st.spinner("Screening stocks..."):
            time.sleep(2)  # Simulate processing time
            
            # Generate mock screening results
            mock_results = generate_mock_screening_results(top_n)
            
            st.success(f"✅ Screening completed! Found {len(mock_results)} candidates.")
            
            # Display results
            display_screening_results(mock_results)

def generate_mock_screening_results(n: int) -> List[dict]:
    """Generate mock screening results for demonstration."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
               "MA", "UNH", "HD", "PG", "BAC", "DIS", "ADBE", "CRM", "NFLX", "PYPL"]
    
    results = []
    for i, symbol in enumerate(symbols[:n]):
        result = {
            "symbol": symbol,
            "rank": i + 1,
            "final_score": np.random.uniform(6.0, 9.5),
            "valuation_score": np.random.uniform(5.0, 10.0),
            "volume_score": np.random.uniform(5.0, 10.0),
            "momentum_score": np.random.uniform(5.0, 10.0),
            "technical_score": np.random.uniform(5.0, 10.0),
            "quality_score": np.random.uniform(5.0, 10.0),
            "current_price": np.random.uniform(50.0, 500.0),
            "market_cap": np.random.uniform(10e9, 2000e9),
            "volume": np.random.randint(100000, 10000000),
            "sector": np.random.choice(["Technology", "Healthcare", "Financial", "Energy", "Consumer"]),
            "rsi": np.random.uniform(30.0, 70.0),
            "volatility": np.random.uniform(0.15, 0.45)
        }
        results.append(result)
    
    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def display_screening_results(results: List[dict]):
    """Display screening results with visualizations."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Summary metrics
    st.markdown("### Screening Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df["final_score"].mean()
        st.metric("Average Score", f"{avg_score:.2f}")
    
    with col2:
        top_sector = df["sector"].value_counts().index[0]
        st.metric("Top Sector", top_sector)
    
    with col3:
        avg_market_cap = df["market_cap"].mean()
        st.metric("Avg Market Cap", format_currency(avg_market_cap))
    
    with col4:
        high_momentum = (df["momentum_score"] > 7.5).sum()
        st.metric("High Momentum Stocks", high_momentum)
    
    # Factor analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 10 Stocks")
        top_stocks = df.head(10)
        
        fig = px.bar(
            top_stocks, 
            x="symbol", 
            y="final_score",
            color="final_score",
            color_continuous_scale="Blues",
            title="Top Scoring Stocks"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Sector Distribution")
        sector_counts = df["sector"].value_counts()
        
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Sector Allocation"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Factor heatmap
    st.markdown("### Factor Analysis Heatmap")
    
    factor_cols = ["valuation_score", "volume_score", "momentum_score", "technical_score", "quality_score"]
    factor_data = df[["symbol"] + factor_cols].set_index("symbol")
    
    fig = px.imshow(
        factor_data.T,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Factor Scores Heatmap"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual stock analysis
    st.markdown("### Individual Stock Analysis")
    selected_stock = st.selectbox("Select stock for detailed analysis", df["symbol"].tolist())
    
    if selected_stock:
        stock_data = df[df["symbol"] == selected_stock].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stock metrics
            st.metric("Rank", stock_data["rank"])
            st.metric("Final Score", f"{stock_data['final_score']:.2f}")
            st.metric("Current Price", format_currency(stock_data["current_price"]))
            st.metric("Market Cap", format_currency(stock_data["market_cap"]))
        
        with col2:
            # Factor radar chart
            factor_scores = {
                "Valuation": stock_data["valuation_score"],
                "Volume": stock_data["volume_score"], 
                "Momentum": stock_data["momentum_score"],
                "Technical": stock_data["technical_score"],
                "Quality": stock_data["quality_score"]
            }
            
            fig = create_factor_radar_chart(factor_scores, selected_stock)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.markdown("### Detailed Results")
    st.dataframe(df, use_container_width=True)

def display_portfolio_analytics_page():
    """Display portfolio analytics and performance tracking."""
    st.markdown('<h1 class="main-header">📈 Portfolio Analytics</h1>', unsafe_allow_html=True)
    
    # Get portfolio data
    status = call_api("/status")
    positions = status.get("positions", []) if "error" not in status else []
    
    if not positions:
        st.info("📊 No portfolio data available")
        return
    
    df = pd.DataFrame(positions)
    
    # Portfolio overview
    st.markdown("### Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) for pos in positions)
        st.metric("Total Portfolio Value", format_currency(total_value))
    
    with col2:
        total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
        st.metric("Total Unrealized P&L", format_currency(total_pnl))
    
    with col3:
        avg_pnl_pct = df.get('pnl_percent', pd.Series([0])).mean()
        st.metric("Average P&L %", format_percentage(avg_pnl_pct))
    
    with col4:
        winning_positions = (df.get('unrealized_pnl', pd.Series([0])) > 0).sum()
        win_rate = (winning_positions / len(df) * 100) if len(df) > 0 else 0
        st.metric("Win Rate", format_percentage(win_rate))
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### P&L Distribution")
        
        pnl_data = df.get('unrealized_pnl', pd.Series([0]))
        fig = px.histogram(
            x=pnl_data,
            nbins=20,
            title="P&L Distribution",
            labels={'x': 'P&L ($)', 'y': 'Frequency'}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Metrics")
        
        # Calculate portfolio metrics
        portfolio_volatility = np.random.uniform(15, 25)  # Mock data
        sharpe_ratio = np.random.uniform(1.2, 2.5)
        max_drawdown = np.random.uniform(5, 15)
        
        fig = create_gauge_chart(sharpe_ratio, "Sharpe Ratio", 0, 3, 1.0, 2.0)
        st.plotly_chart(fig, use_container_width=True)
    
    # Position analysis
    st.markdown("### Position Analysis")
    
    # Top performers
    if len(df) > 0:
        df_sorted = df.sort_values('unrealized_pnl', ascending=False) if 'unrealized_pnl' in df.columns else df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Performers")
            top_performers = df_sorted.head(5)
            for _, pos in top_performers.iterrows():
                pnl = pos.get('unrealized_pnl', 0)
                color = "🟢" if pnl >= 0 else "🔴"
                st.write(f"{color} **{pos.get('symbol', 'N/A')}**: {format_currency(pnl)}")
        
        with col2:
            st.markdown("#### Bottom Performers")
            bottom_performers = df_sorted.tail(5)
            for _, pos in bottom_performers.iterrows():
                pnl = pos.get('unrealized_pnl', 0)
                color = "🟢" if pnl >= 0 else "🔴"
                st.write(f"{color} **{pos.get('symbol', 'N/A')}**: {format_currency(pnl)}")
    
    # Sector allocation
    st.markdown("### Sector Analysis")
    
    if 'sector' in df.columns:
        sector_allocation = df.groupby('sector').agg({
            'quantity': 'sum',
            'unrealized_pnl': 'sum'
        }).reset_index()
        
        fig = px.sunburst(
            sector_allocation,
            path=['sector'],
            values='quantity',
            title="Portfolio Sector Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_risk_management_page():
    """Display risk management dashboard."""
    st.markdown('<h1 class="main-header">⚠️ Risk Management</h1>', unsafe_allow_html=True)
    
    # Risk settings
    st.markdown("### Risk Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Portfolio Limits")
        max_position_size = st.slider("Max Position Size (%)", 1, 20, 10)
        max_sector_exposure = st.slider("Max Sector Exposure (%)", 10, 50, 30)
        max_portfolio_leverage = st.slider("Max Portfolio Leverage", 1.0, 3.0, 2.0, 0.1)
    
    with col2:
        st.markdown("#### Stop Loss Settings")
        individual_stop_loss = st.slider("Individual Stop Loss (%)", 1, 20, 10)
        portfolio_stop_loss = st.slider("Portfolio Stop Loss (%)", 5, 30, 15)
        trailing_stop = st.checkbox("Enable Trailing Stop")
    
    with col3:
        st.markdown("#### Risk Alerts")
        volatility_threshold = st.slider("Volatility Alert (%)", 10, 50, 25)
        correlation_threshold = st.slider("Correlation Alert", 0.5, 0.95, 0.8, 0.05)
        drawdown_alert = st.slider("Drawdown Alert (%)", 5, 25, 10)
    
    # Current risk metrics
    st.markdown("### Current Risk Metrics")
    
    # Mock risk data
    current_volatility = np.random.uniform(15, 30)
    portfolio_beta = np.random.uniform(0.8, 1.5)
    var_95 = np.random.uniform(5, 15)
    current_drawdown = np.random.uniform(2, 12)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_gauge_chart(current_volatility, "Portfolio Volatility (%)", 0, 50, 20, 35)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gauge_chart(portfolio_beta, "Portfolio Beta", 0, 2, 0.8, 1.2)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_gauge_chart(var_95, "VaR (95%)", 0, 25, 5, 15)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = create_gauge_chart(current_drawdown, "Current Drawdown (%)", 0, 30, 5, 15)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk alerts
    st.markdown("### Risk Alerts")
    
    # Generate mock alerts
    alerts = [
        {"level": "warning", "message": "Portfolio volatility above 25% threshold", "time": "2 min ago"},
        {"level": "info", "message": "Sector exposure rebalanced - Technology reduced to 25%", "time": "15 min ago"},
        {"level": "error", "message": "Individual position TSLA triggered stop loss", "time": "1 hour ago"},
    ]
    
    for alert in alerts:
        if alert["level"] == "error":
            st.error(f"🔴 **{alert['time']}**: {alert['message']}")
        elif alert["level"] == "warning":
            st.warning(f"🟡 **{alert['time']}**: {alert['message']}")
        else:
            st.info(f"🔵 **{alert['time']}**: {alert['message']}")
    
    # Position risk analysis
    st.markdown("### Position Risk Analysis")
    
    # Mock position risk data
    position_risks = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'weight': [0.15, 0.12, 0.10, 0.08, 0.05],
        'volatility': [0.25, 0.22, 0.28, 0.30, 0.45],
        'beta': [1.1, 0.9, 1.2, 1.3, 1.8],
        'var_contribution': [0.03, 0.02, 0.025, 0.028, 0.035]
    })
    
    fig = px.scatter(
        position_risks,
        x='weight',
        y='volatility',
        size='var_contribution',
        color='beta',
        hover_data=['symbol'],
        title="Position Risk-Return Profile",
        labels={'weight': 'Portfolio Weight', 'volatility': 'Volatility'}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_performance_analysis_page():
    """Display performance analysis dashboard."""
    st.markdown('<h1 class="main-header">📊 Performance Analysis</h1>', unsafe_allow_html=True)
    
    # Performance period selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period = st.selectbox("Analysis Period", ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD"])
    
    with col2:
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "VTI", "IWM"])
    
    with col3:
        show_drawdown = st.checkbox("Show Drawdown", True)
    
    # Generate mock performance data
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    portfolio_returns = np.random.normal(0.001, 0.02, 252)
    benchmark_returns = np.random.normal(0.0008, 0.015, 252)
    
    portfolio_cumulative = (1 + pd.Series(portfolio_returns)).cumprod()
    benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod()
    
    # Performance chart
    st.markdown("### Portfolio Performance")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_cumulative,
        mode='lines',
        name='Portfolio',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_cumulative,
        mode='lines',
        name=benchmark,
        line=dict(color='red', width=2)
    ))
    
    if show_drawdown:
        # Calculate drawdown
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative / running_max - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown (%)',
            line=dict(color='orange', width=1),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="Portfolio Performance vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis2=dict(
            title="Drawdown (%)",
            overlaying='y',
            side='right'
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    
    # Calculate metrics
    portfolio_return = (portfolio_cumulative.iloc[-1] - 1) * 100
    benchmark_return = (benchmark_cumulative.iloc[-1] - 1) * 100
    excess_return = portfolio_return - benchmark_return
    
    portfolio_vol = np.std(portfolio_returns) * np.sqrt(252) * 100
    benchmark_vol = np.std(benchmark_returns) * np.sqrt(252) * 100
    
    sharpe_ratio = (np.mean(portfolio_returns) * 252) / (np.std(portfolio_returns) * np.sqrt(252))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Return", f"{portfolio_return:.2f}%")
        st.metric("Portfolio Volatility", f"{portfolio_vol:.2f}%")
    
    with col2:
        st.metric("Benchmark Return", f"{benchmark_return:.2f}%")
        st.metric("Benchmark Volatility", f"{benchmark_vol:.2f}%")
    
    with col3:
        st.metric("Excess Return", f"{excess_return:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        max_drawdown = drawdown.min() if show_drawdown else np.random.uniform(-15, -5)
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        calmar_ratio = portfolio_return / abs(max_drawdown) if max_drawdown != 0 else 0
        st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
    
    # Monthly returns heatmap
    st.markdown("### Monthly Returns Heatmap")
    
    # Generate mock monthly returns
    monthly_returns = np.random.normal(0.02, 0.05, 12)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = px.imshow(
        [monthly_returns],
        x=months,
        y=['2024'],
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title='Monthly Returns (%)'
    )
    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)

def display_system_logs_page():
    """Display system logs and activity."""
    st.markdown('<h1 class="main-header">📋 System Logs</h1>', unsafe_allow_html=True)
    
    # Log controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_lines = st.number_input("Number of log lines", min_value=10, max_value=1000, value=200)
    
    with col2:
        log_level = st.selectbox("Log Level", ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"])
    
    with col3:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    
    # Get logs
    logs_response = call_api(f"/logs?n={log_lines}")
    
    if "error" in logs_response:
        st.error(f"❌ Cannot fetch logs: {logs_response['error']}")
        return
    
    logs = logs_response.get("lines", [])
    
    if not logs:
        st.info("📋 No logs available")
        return
    
    # Filter logs by level
    if log_level != "ALL":
        logs = [log for log in logs if log_level in log]
    
    # Log statistics
    st.markdown("### Log Statistics")
    
    error_count = sum(1 for log in logs if "ERROR" in log)
    warning_count = sum(1 for log in logs if "WARNING" in log)
    info_count = sum(1 for log in logs if "INFO" in log)
    trade_count = sum(1 for log in logs if any(word in log for word in ["BUY", "SELL"]))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Logs", len(logs))
    
    with col2:
        st.metric("Errors", error_count)
    
    with col3:
        st.metric("Warnings", warning_count)
    
    with col4:
        st.metric("Trade Actions", trade_count)
    
    # Log analysis
    if logs:
        st.markdown("### Recent Activity")
        
        # Show recent logs with color coding
        for log_line in logs[-50:]:  # Show last 50 logs
            if "ERROR" in log_line:
                st.error(f"❌ {log_line}")
            elif "WARNING" in log_line:
                st.warning(f"⚠️ {log_line}")
            elif any(keyword in log_line for keyword in ["BUY", "SELL"]):
                st.success(f"📈 {log_line}")
            elif "INFO" in log_line:
                st.info(f"ℹ️ {log_line}")
            else:
                st.text(log_line)
    
    # Full log viewer
    st.markdown("### Full Log Viewer")
    
    log_text = "\n".join(logs)
    st.text_area("Logs", log_text, height=400)

# Main application logic
def main():
    """Main application function."""
    
    # Page routing
    if page == "Overview":
        display_overview_page()
    elif page == "Stock Screening":
        display_stock_screening_page()
    elif page == "Portfolio Analytics":
        display_portfolio_analytics_page()
    elif page == "Risk Management":
        display_risk_management_page()
    elif page == "Performance Analysis":
        display_performance_analysis_page()
    elif page == "System Logs":
        display_system_logs_page()
    
    # Auto-refresh functionality
    if AUTO_REFRESH and page == "Overview":
        time.sleep(REFRESH_SEC)
        st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Quant Trading Dashboard v2.0**")
st.sidebar.markdown("*Enhanced with intelligent stock selection*")

if __name__ == "__main__":
    main()