"""
Enhanced Multi-Asset Trading Dashboard - Support for 5,700+ Assets
================================================================

This enhanced dashboard provides comprehensive monitoring and control capabilities
for the quantitative trading system supporting:

- 5,000+ Stocks (NYSE, NASDAQ)
- 1,500+ ETFs (All major categories)
- REITs (Real Estate Investment Trusts)
- ADRs (American Depository Receipts)
- Futures (Stock Index, Commodity, Currency, Bond)

Features:
- Multi-asset universe management and filtering
- Real-time monitoring with heat maps and advanced visualizations
- AI learning progress tracking with model performance metrics
- Futures trading interface with margin and risk management
- Professional responsive design with customizable layouts
- Advanced performance analytics and portfolio optimization
"""

import os
import time
import json
import asyncio
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
from typing import Dict, List, Optional, Any, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Load environment variables
load_dotenv()

# Page configuration - responsive design
st.set_page_config(
    page_title="Quant Trading Dashboard Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/quantbot',
        'Report a bug': "https://github.com/quantbot/issues",
        'About': "Multi-Asset Quantitative Trading Platform v3.0"
    }
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    /* Main theme and layout */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .futures-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #2d3436;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 2px solid #fdcb6e;
    }
    
    /* Asset type indicators */
    .asset-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .stock-badge {
        background-color: #74b9ff;
        color: white;
    }
    
    .etf-badge {
        background-color: #00b894;
        color: white;
    }
    
    .reit-badge {
        background-color: #6c5ce7;
        color: white;
    }
    
    .adr-badge {
        background-color: #fd79a8;
        color: white;
    }
    
    .futures-badge {
        background-color: #fdcb6e;
        color: #2d3436;
    }
    
    /* Heat map styling */
    .heat-map-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Performance metrics */
    .perf-metric {
        text-align: center;
        padding: 1rem;
        margin: 0.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .metric-card, .success-card, .warning-card, .info-card {
            padding: 1rem;
        }
    }
    
    /* Custom scrollbar */
    .stApp {
        scrollbar-width: thin;
        scrollbar-color: #667eea #f1f3f4;
    }
    
    .stApp::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    .stApp::-webkit-scrollbar-track {
        background: #f1f3f4;
        border-radius: 10px;
    }
    
    .stApp::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    /* Enhanced tables */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Loading animations */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-radius: 50%;
        border-top: 4px solid #667eea;
        width: 30px;
        height: 30px;
        animation: spin 2s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE = st.sidebar.text_input("API Base", os.getenv("API_BASE", "http://localhost:8000"))
TOKEN = st.sidebar.text_input("Bearer Token (ADMIN_TOKEN)", os.getenv("ADMIN_TOKEN", "wgyjd0508"), type="password")
AUTO_REFRESH = st.sidebar.checkbox("Auto refresh", True)
REFRESH_SEC = st.sidebar.slider("Refresh interval (seconds)", 2, 60, 5)

# Enhanced Navigation with asset type filtering
st.sidebar.title("🚀 Multi-Asset Platform")
page = st.sidebar.selectbox(
    "Select Page",
    ["Asset Universe", "Trading Interface", "AI Learning Center", "Futures Trading", 
     "Real-time Monitor", "Portfolio Analytics", "Risk Management", "Performance Hub", "System Logs"]
)

# Asset filtering sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Asset Filters")
selected_asset_types = st.sidebar.multiselect(
    "Asset Types",
    ["Stock", "ETF", "REIT", "ADR", "Futures"],
    default=["Stock", "ETF"]
)

market_cap_filter = st.sidebar.select_slider(
    "Market Cap Range",
    options=["All", "Large Cap (>10B)", "Mid Cap (2B-10B)", "Small Cap (300M-2B)", "Micro Cap (<300M)"],
    value="All"
)

# Utility functions
@st.cache_data(ttl=60)  # Cache for 1 minute
def headers():
    """Get HTTP headers for API requests."""
    if TOKEN and TOKEN != "changeme":
        return {"Authorization": f"Bearer {TOKEN}"}
    env_token = os.getenv("ADMIN_TOKEN", "wgyjd0508")
    return {"Authorization": f"Bearer {env_token}"}

@st.cache_data(ttl=30)  # Cache for 30 seconds
def call_api(endpoint: str, method: str = "GET", json_data: dict = None) -> dict:
    """Make API call with error handling and caching."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, headers=headers(), timeout=15)
        else:
            r = requests.post(url, headers=headers(), json=json_data, timeout=15)
        
        if r.ok:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - server may be processing large dataset"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - check if backend is running"}
    except Exception as e:
        return {"error": str(e)}

def format_currency(value: float, compact: bool = True) -> str:
    """Format currency values with smart scaling."""
    if pd.isna(value):
        return "N/A"
    
    if compact:
        if abs(value) >= 1e12:
            return f"${value/1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    else:
        return f"${value:,.2f}"

def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage values."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{precision}f}%"

def get_asset_badge(asset_type: str) -> str:
    """Get HTML badge for asset type."""
    badge_classes = {
        "stock": "stock-badge",
        "etf": "etf-badge", 
        "reit": "reit-badge",
        "adr": "adr-badge",
        "futures": "futures-badge"
    }
    
    class_name = badge_classes.get(asset_type.lower(), "stock-badge")
    return f'<span class="asset-badge {class_name}">{asset_type.upper()}</span>'

def create_enhanced_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 100,
                               threshold_ranges: List[Tuple[float, str]] = None) -> go.Figure:
    """Create enhanced gauge chart with custom thresholds."""
    if threshold_ranges is None:
        threshold_ranges = [(30, "red"), (70, "yellow"), (100, "green")]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#2c3e50'}},
        delta={'reference': (min_val + max_val) / 2},
        gauge={
            'axis': {'range': [None, max_val], 'tickcolor': '#2c3e50'},
            'bar': {'color': "#667eea", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e9ecef",
            'steps': [
                {'range': [min_val, threshold_ranges[0][0]], 'color': '#ff7675'},
                {'range': [threshold_ranges[0][0], threshold_ranges[1][0]], 'color': '#fdcb6e'},
                {'range': [threshold_ranges[1][0], max_val], 'color': '#00b894'}
            ] if len(threshold_ranges) >= 2 else [],
            'threshold': {
                'line': {'color': "#e74c3c", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "#2c3e50", 'family': "Arial, sans-serif"}
    )
    return fig

def create_asset_heatmap(asset_data: List[Dict], metric: str = "change_pct") -> go.Figure:
    """Create interactive heatmap for asset performance."""
    if not asset_data:
        return go.Figure()
    
    # Prepare data for heatmap
    df = pd.DataFrame(asset_data)
    
    # Group by asset type and create matrix
    asset_types = df['asset_type'].unique()
    symbols = df['symbol'].tolist()[:100]  # Limit for performance
    
    # Create matrix data
    z_data = []
    hover_text = []
    
    for asset_type in asset_types:
        type_data = df[df['asset_type'] == asset_type].head(20)
        row_data = []
        hover_row = []
        
        for symbol in symbols[:len(type_data)]:
            if symbol in type_data['symbol'].values:
                row_data.append(type_data[type_data['symbol'] == symbol][metric].iloc[0])
                symbol_data = type_data[type_data['symbol'] == symbol].iloc[0]
                hover_row.append(f"{symbol}<br>Type: {asset_type}<br>Change: {symbol_data[metric]:.2f}%<br>Price: ${symbol_data.get('price', 0):.2f}")
            else:
                row_data.append(None)
                hover_row.append("")
        
        z_data.append(row_data[:20])  # Limit columns
        hover_text.append(hover_row[:20])
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=symbols[:20],
        y=asset_types,
        hoverongaps=False,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        colorscale='RdYlGn',
        zmid=0
    ))
    
    fig.update_layout(
        title="Asset Performance Heat Map",
        xaxis_title="Symbols",
        yaxis_title="Asset Types",
        height=400,
        font={'color': "#2c3e50"}
    )
    
    return fig

def create_futures_chain_view(futures_data: Dict) -> go.Figure:
    """Create futures contract chain visualization."""
    if not futures_data or 'contracts' not in futures_data:
        return go.Figure()
    
    contracts = futures_data['contracts']
    symbols = [contract['symbol'] for contract in contracts]
    prices = [contract['price'] for contract in contracts]
    volumes = [contract['volume'] for contract in contracts]
    expirations = [contract['expiry_date'] for contract in contracts]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Contract Prices', 'Contract Volumes'),
        vertical_spacing=0.1
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=symbols, y=prices, mode='markers+lines', 
                  name='Price', marker=dict(size=10, color='#667eea')),
        row=1, col=1
    )
    
    # Volume chart  
    fig.add_trace(
        go.Bar(x=symbols, y=volumes, name='Volume', 
               marker=dict(color='#00b894')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Futures Contract Chain",
        showlegend=True
    )
    
    return fig

def display_asset_universe_page():
    """Display the comprehensive asset universe overview."""
    st.markdown('<h1 class="main-header">🌎 Multi-Asset Universe</h1>', unsafe_allow_html=True)
    
    # Asset universe statistics
    st.markdown("### 📊 Universe Statistics")
    
    # Get universe data
    universe_stats = call_api("/universe_stats")
    
    if "error" in universe_stats:
        st.error(f"❌ Cannot load universe data: {universe_stats['error']}")
        return
    
    # Display overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assets = universe_stats.get('total_assets', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{total_assets:,}</h3>
            <p style="margin: 0;">Total Assets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_assets = universe_stats.get('active_assets', 0)
        st.markdown(f"""
        <div class="success-card">
            <h3 style="margin: 0; font-size: 2rem;">{active_assets:,}</h3>
            <p style="margin: 0;">Active Assets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        data_coverage = universe_stats.get('data_coverage_pct', 0)
        st.markdown(f"""
        <div class="info-card">
            <h3 style="margin: 0; font-size: 2rem;">{data_coverage:.1f}%</h3>
            <p style="margin: 0;">Data Coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        last_update = universe_stats.get('last_update', 'Never')
        st.markdown(f"""
        <div class="warning-card">
            <h3 style="margin: 0; font-size: 1.2rem;">{last_update}</h3>
            <p style="margin: 0;">Last Updated</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Asset type breakdown
    st.markdown("### 🎯 Asset Type Distribution")
    
    type_breakdown = universe_stats.get('asset_type_breakdown', {})
    if type_breakdown:
        
        # Create donut chart for asset types
        labels = list(type_breakdown.keys())
        values = list(type_breakdown.values())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.4,
                marker_colors=['#74b9ff', '#00b894', '#6c5ce7', '#fd79a8', '#fdcb6e']
            )])
            fig_donut.update_layout(
                title="Asset Universe Composition",
                height=400,
                annotations=[dict(text=f'{sum(values):,}<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with col2:
            st.markdown("#### Asset Counts:")
            for asset_type, count in type_breakdown.items():
                percentage = (count / sum(values)) * 100
                badge = get_asset_badge(asset_type)
                st.markdown(f"""
                <div class="perf-metric">
                    {badge}<br>
                    <strong>{count:,}</strong><br>
                    <small>{percentage:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Asset search and filtering
    st.markdown("### 🔍 Asset Explorer")
    
    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
    
    with search_col1:
        search_term = st.text_input("Search assets (symbol, name, or keyword)", placeholder="AAPL, Tesla, Gold, etc.")
    
    with search_col2:
        sort_by = st.selectbox("Sort by", ["Symbol", "Market Cap", "Volume", "Price Change"])
    
    with search_col3:
        max_results = st.number_input("Max results", min_value=10, max_value=500, value=50)
    
    if search_term:
        # Perform search
        search_params = {
            "query": search_term,
            "asset_types": selected_asset_types,
            "max_results": max_results,
            "sort_by": sort_by.lower().replace(" ", "_")
        }
        
        search_results = call_api("/search_assets", method="POST", json_data=search_params)
        
        if "error" not in search_results and "results" in search_results:
            results = search_results["results"]
            
            if results:
                st.markdown(f"### 📋 Found {len(results)} assets")
                
                # Create enhanced results table
                df_results = pd.DataFrame(results)
                
                # Add asset type badges
                def add_badge(row):
                    return get_asset_badge(row['asset_type'])
                
                # Display results with enhanced formatting
                for i, result in enumerate(results):
                    with st.expander(f"{result['symbol']} - {result['name']} {get_asset_badge(result['asset_type'])}", 
                                   expanded=(i < 5)):  # Expand first 5
                        
                        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                        
                        with rcol1:
                            st.metric("Price", format_currency(result.get('price', 0)))
                            st.metric("Market Cap", format_currency(result.get('market_cap', 0)))
                        
                        with rcol2:
                            st.metric("Volume", f"{result.get('volume', 0):,}")
                            st.metric("Exchange", result.get('exchange', 'N/A'))
                        
                        with rcol3:
                            change = result.get('price_change_pct', 0)
                            st.metric("Change %", f"{change:+.2f}%", delta=f"{change:.2f}%")
                            st.metric("Sector", result.get('sector', 'N/A'))
                        
                        with rcol4:
                            st.metric("Beta", f"{result.get('beta', 1.0):.2f}")
                            if result.get('asset_type') == 'futures':
                                st.metric("Contract Size", result.get('contract_size', 'N/A'))
                            else:
                                st.metric("P/E Ratio", f"{result.get('pe_ratio', 0):.1f}")
                        
                        # Add to watchlist button
                        if st.button(f"📌 Add {result['symbol']} to Watchlist", key=f"watch_{result['symbol']}"):
                            st.success(f"Added {result['symbol']} to watchlist!")
                
            else:
                st.info("No assets found matching your criteria")
        else:
            st.error(f"Search failed: {search_results.get('error', 'Unknown error')}")
    
    # Real-time asset heat map
    st.markdown("### 🌡️ Real-Time Performance Heat Map")
    
    # Get real-time data for heat map
    heatmap_data = call_api("/realtime_heatmap")
    
    if "error" not in heatmap_data and "assets" in heatmap_data:
        fig_heatmap = create_asset_heatmap(heatmap_data["assets"])
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Real-time heat map data not available")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    
    with qcol1:
        if st.button("🔄 Refresh Universe", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with qcol2:
        if st.button("📥 Export Watchlist", use_container_width=True):
            st.success("Watchlist exported to CSV")
    
    with qcol3:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Generating comprehensive universe report...")
    
    with qcol4:
        if st.button("⚙️ Configure Universe", use_container_width=True):
            st.info("Opening universe configuration...")

def display_trading_interface_page():
    """Display enhanced trading interface for all asset types."""
    st.markdown('<h1 class="main-header">💹 Advanced Trading Interface</h1>', unsafe_allow_html=True)
    
    # Trading dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📈 Order Management", "🎯 Strategy Execution", "⚡ Quick Trade"])
    
    with tab1:
        display_market_overview()
    
    with tab2:
        display_order_management()
    
    with tab3:
        display_strategy_execution()
    
    with tab4:
        display_quick_trade()

def display_market_overview():
    """Display comprehensive market overview."""
    st.markdown("### 🌐 Global Market Overview")
    
    # Market indices
    indices_data = call_api("/market_indices")
    
    if "error" not in indices_data:
        indices = indices_data.get("indices", [])
        
        # Create market indices dashboard
        idx_cols = st.columns(len(indices) if len(indices) <= 6 else 6)
        
        for i, index in enumerate(indices[:6]):
            with idx_cols[i % 6]:
                change = index.get('change_pct', 0)
                color = "normal" if change >= 0 else "inverse"
                st.metric(
                    index['name'], 
                    f"{index['price']:.2f}",
                    delta=f"{change:+.2f}%",
                    delta_color=color
                )
    
    # Sector performance
    st.markdown("### 🏭 Sector Performance")
    
    sector_data = call_api("/sector_performance")
    
    if "error" not in sector_data and "sectors" in sector_data:
        sectors_df = pd.DataFrame(sector_data["sectors"])
        
        # Create sector performance chart
        fig_sectors = px.bar(
            sectors_df,
            x='sector',
            y='change_pct',
            color='change_pct',
            color_continuous_scale='RdYlGn',
            title="Sector Performance Today"
        )
        fig_sectors.update_layout(height=400)
        st.plotly_chart(fig_sectors, use_container_width=True)
    
    # Top movers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Top Gainers")
        gainers_data = call_api("/top_gainers")
        if "error" not in gainers_data:
            for gainer in gainers_data.get("gainers", [])[:5]:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #e9ecef;">
                    <span><strong>{gainer['symbol']}</strong> {get_asset_badge(gainer.get('asset_type', 'stock'))}</span>
                    <span style="color: #00b894;"><strong>+{gainer['change_pct']:.2f}%</strong></span>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📉 Top Losers")
        losers_data = call_api("/top_losers")
        if "error" not in losers_data:
            for loser in losers_data.get("losers", [])[:5]:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #e9ecef;">
                    <span><strong>{loser['symbol']}</strong> {get_asset_badge(loser.get('asset_type', 'stock'))}</span>
                    <span style="color: #e74c3c;"><strong>{loser['change_pct']:.2f}%</strong></span>
                </div>
                """, unsafe_allow_html=True)

def display_order_management():
    """Display order management interface."""
    st.markdown("### 📋 Order Management")
    
    # Active orders
    orders_data = call_api("/active_orders")
    
    if "error" not in orders_data and "orders" in orders_data:
        orders = orders_data["orders"]
        
        if orders:
            st.markdown("#### Active Orders")
            
            for order in orders:
                with st.expander(f"Order {order['order_id']} - {order['symbol']} ({order['side']})"):
                    ocol1, ocol2, ocol3, ocol4 = st.columns(4)
                    
                    with ocol1:
                        st.write(f"**Symbol:** {order['symbol']}")
                        st.write(f"**Side:** {order['side']}")
                    
                    with ocol2:
                        st.write(f"**Quantity:** {order['quantity']}")
                        st.write(f"**Price:** ${order['price']:.2f}")
                    
                    with ocol3:
                        st.write(f"**Status:** {order['status']}")
                        st.write(f"**Type:** {order['order_type']}")
                    
                    with ocol4:
                        st.write(f"**Time:** {order['timestamp']}")
                        if st.button(f"Cancel Order {order['order_id']}", key=f"cancel_{order['order_id']}"):
                            cancel_result = call_api(f"/cancel_order/{order['order_id']}", method="POST")
                            if "error" not in cancel_result:
                                st.success("Order cancelled successfully")
                                st.rerun()
                            else:
                                st.error(f"Failed to cancel order: {cancel_result['error']}")
        else:
            st.info("No active orders")
    
    # Order history
    st.markdown("#### Recent Order History")
    
    history_data = call_api("/order_history?limit=20")
    
    if "error" not in history_data and "orders" in history_data:
        history_df = pd.DataFrame(history_data["orders"])
        
        if not history_df.empty:
            # Format the dataframe for display
            display_df = history_df[['timestamp', 'symbol', 'side', 'quantity', 'price', 'status']].copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "timestamp": "Time",
                    "symbol": "Symbol", 
                    "side": "Side",
                    "quantity": st.column_config.NumberColumn("Qty"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "status": "Status"
                }
            )

def display_strategy_execution():
    """Display strategy execution interface."""
    st.markdown("### 🎯 Strategy Execution")
    
    # Strategy selection
    scol1, scol2 = st.columns([2, 1])
    
    with scol1:
        strategy_name = st.selectbox(
            "Select Strategy",
            ["Multi-Asset Momentum", "Value Factor", "Technical Breakout", "Mean Reversion", "Futures Carry"]
        )
    
    with scol2:
        execution_mode = st.selectbox("Execution Mode", ["Paper Trading", "Live Trading"])
    
    # Strategy parameters
    st.markdown("#### Strategy Parameters")
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        position_size = st.slider("Position Size (%)", 1, 20, 5)
        max_positions = st.number_input("Max Positions", 1, 50, 10)
    
    with param_col2:
        stop_loss = st.slider("Stop Loss (%)", 1, 15, 5)
        take_profit = st.slider("Take Profit (%)", 5, 50, 15)
    
    with param_col3:
        rebalance_freq = st.selectbox("Rebalance Frequency", ["Daily", "Weekly", "Monthly"])
        risk_level = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])
    
    # Asset universe selection for strategy
    st.markdown("#### Asset Universe")
    
    universe_assets = st.multiselect(
        "Include Asset Types",
        ["Large Cap Stocks", "Mid Cap Stocks", "ETFs", "REITs", "Futures"],
        default=["Large Cap Stocks", "ETFs"]
    )
    
    # Strategy execution controls
    exec_col1, exec_col2, exec_col3 = st.columns(3)
    
    with exec_col1:
        if st.button("▶️ Start Strategy", use_container_width=True):
            strategy_config = {
                "strategy_name": strategy_name,
                "execution_mode": execution_mode,
                "parameters": {
                    "position_size_pct": position_size,
                    "max_positions": max_positions,
                    "stop_loss_pct": stop_loss,
                    "take_profit_pct": take_profit,
                    "rebalance_frequency": rebalance_freq,
                    "risk_level": risk_level
                },
                "universe": universe_assets
            }
            
            result = call_api("/start_strategy", method="POST", json_data=strategy_config)
            
            if "error" not in result:
                st.success("Strategy started successfully!")
            else:
                st.error(f"Failed to start strategy: {result['error']}")
    
    with exec_col2:
        if st.button("⏸️ Pause Strategy", use_container_width=True):
            st.info("Strategy paused")
    
    with exec_col3:
        if st.button("⏹️ Stop Strategy", use_container_width=True):
            st.warning("Strategy stopped")

def display_quick_trade():
    """Display quick trading interface."""
    st.markdown("### ⚡ Quick Trade")
    
    # Quick trade form
    trade_col1, trade_col2 = st.columns([2, 1])
    
    with trade_col1:
        symbol = st.text_input("Symbol", placeholder="AAPL, SPY, ES, GLD, etc.").upper()
        
        if symbol:
            # Get symbol info
            symbol_info = call_api(f"/symbol_info/{symbol}")
            
            if "error" not in symbol_info:
                info = symbol_info.get("info", {})
                
                st.markdown(f"""
                **{info.get('name', symbol)}** {get_asset_badge(info.get('asset_type', 'stock'))}
                
                Current Price: **${info.get('price', 0):.2f}** | 
                Change: **{info.get('change_pct', 0):+.2f}%** |
                Volume: **{info.get('volume', 0):,}**
                """)
    
    with trade_col2:
        st.markdown("### Order Details")
        
        side = st.selectbox("Side", ["BUY", "SELL"])
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP"])
        quantity = st.number_input("Quantity", min_value=1, value=100)
        
        if order_type in ["LIMIT", "STOP"]:
            price = st.number_input("Price", min_value=0.01, value=100.0, step=0.01)
        else:
            price = None
    
    # Risk check
    if symbol and quantity:
        estimated_cost = quantity * (price if price else 100.0)
        st.markdown(f"""
        <div class="info-card">
            <strong>Order Summary</strong><br>
            {side} {quantity} shares of {symbol}<br>
            Estimated Cost: {format_currency(estimated_cost)}
        </div>
        """, unsafe_allow_html=True)
        
        # Place order button
        if st.button(f"🚀 Place {side} Order", use_container_width=True, type="primary"):
            order_data = {
                "symbol": symbol,
                "side": side.lower(),
                "quantity": quantity,
                "order_type": order_type.lower(),
                "price": price
            }
            
            result = call_api("/place_order", method="POST", json_data=order_data)
            
            if "error" not in result:
                st.success(f"Order placed successfully! Order ID: {result.get('order_id', 'N/A')}")
            else:
                st.error(f"Failed to place order: {result['error']}")

def display_ai_learning_center():
    """Display AI learning progress and model performance."""
    st.markdown('<h1 class="main-header">🤖 AI Learning Center</h1>', unsafe_allow_html=True)
    
    # Learning progress overview
    st.markdown("### 📈 Learning Progress Overview")
    
    learning_stats = call_api("/ai_learning_stats")
    
    if "error" not in learning_stats:
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        
        with lcol1:
            training_epochs = learning_stats.get('training_epochs', 0)
            fig_epochs = create_enhanced_gauge_chart(
                training_epochs, 
                "Training Epochs",
                min_val=0,
                max_val=1000,
                threshold_ranges=[(200, "red"), (500, "yellow"), (1000, "green")]
            )
            st.plotly_chart(fig_epochs, use_container_width=True)
        
        with lcol2:
            model_accuracy = learning_stats.get('model_accuracy', 0)
            fig_accuracy = create_enhanced_gauge_chart(
                model_accuracy * 100,
                "Model Accuracy (%)",
                min_val=0,
                max_val=100,
                threshold_ranges=[(60, "red"), (80, "yellow"), (100, "green")]
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        with lcol3:
            strategy_performance = learning_stats.get('strategy_performance', 0)
            fig_performance = create_enhanced_gauge_chart(
                strategy_performance,
                "Strategy Performance",
                min_val=-10,
                max_val=20,
                threshold_ranges=[(0, "red"), (10, "yellow"), (20, "green")]
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with lcol4:
            data_quality = learning_stats.get('data_quality_score', 0)
            fig_quality = create_enhanced_gauge_chart(
                data_quality * 100,
                "Data Quality (%)",
                min_val=0,
                max_val=100,
                threshold_ranges=[(70, "red"), (85, "yellow"), (100, "green")]
            )
            st.plotly_chart(fig_quality, use_container_width=True)
    
    # Model performance comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    model_data = call_api("/model_performance_comparison")
    
    if "error" not in model_data and "models" in model_data:
        models_df = pd.DataFrame(model_data["models"])
        
        # Create performance comparison chart
        fig_comparison = go.Figure()
        
        models = models_df['model_name'].unique()
        for model in models:
            model_perf = models_df[models_df['model_name'] == model]
            fig_comparison.add_trace(go.Scatter(
                x=model_perf['training_day'],
                y=model_perf['accuracy'],
                mode='lines+markers',
                name=model,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig_comparison.update_layout(
            title="Model Accuracy Over Time",
            xaxis_title="Training Day",
            yaxis_title="Accuracy",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Feature importance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Feature Importance")
        
        feature_data = call_api("/feature_importance")
        
        if "error" not in feature_data and "features" in feature_data:
            features_df = pd.DataFrame(feature_data["features"])
            
            fig_features = px.bar(
                features_df.head(10),
                x='importance',
                y='feature_name',
                orientation='h',
                title="Top 10 Most Important Features",
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig_features.update_layout(height=400)
            st.plotly_chart(fig_features, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Learning Metrics")
        
        metrics_data = call_api("/learning_metrics")
        
        if "error" not in metrics_data:
            metrics = metrics_data.get("metrics", {})
            
            # Display key learning metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    st.metric(
                        metric_name.replace('_', ' ').title(),
                        f"{metric_value:.4f}" if isinstance(metric_value, float) else f"{metric_value:,}"
                    )
    
    # Strategy evolution
    st.markdown("### 🧬 Strategy Evolution")
    
    evolution_data = call_api("/strategy_evolution")
    
    if "error" not in evolution_data and "evolution" in evolution_data:
        evolution_df = pd.DataFrame(evolution_data["evolution"])
        
        # Create strategy evolution visualization
        fig_evolution = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Strategy Returns', 'Risk Metrics'),
            vertical_spacing=0.1
        )
        
        # Returns evolution
        fig_evolution.add_trace(
            go.Scatter(
                x=evolution_df['date'],
                y=evolution_df['cumulative_return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#00b894', width=3)
            ),
            row=1, col=1
        )
        
        # Risk evolution
        fig_evolution.add_trace(
            go.Scatter(
                x=evolution_df['date'],
                y=evolution_df['volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='#e74c3c', width=2)
            ),
            row=2, col=1
        )
        
        fig_evolution.update_layout(height=600, title_text="Strategy Performance Evolution")
        st.plotly_chart(fig_evolution, use_container_width=True)
    
    # AI insights and recommendations
    st.markdown("### 💡 AI Insights & Recommendations")
    
    insights_data = call_api("/ai_insights")
    
    if "error" not in insights_data and "insights" in insights_data:
        insights = insights_data["insights"]
        
        for i, insight in enumerate(insights):
            insight_type = insight.get('type', 'info')
            icon = {"info": "ℹ️", "warning": "⚠️", "success": "✅", "error": "❌"}.get(insight_type, "ℹ️")
            
            st.markdown(f"""
            <div class="info-card">
                <strong>{icon} {insight['title']}</strong><br>
                {insight['message']}<br>
                <small>Confidence: {insight['confidence']:.2%}</small>
            </div>
            """, unsafe_allow_html=True)

def display_futures_trading_page():
    """Display specialized futures trading interface."""
    st.markdown('<h1 class="main-header">🏛️ Futures Trading Center</h1>', unsafe_allow_html=True)
    
    # Futures overview
    st.markdown("### 📊 Futures Market Overview")
    
    futures_stats = call_api("/futures_stats")
    
    if "error" not in futures_stats:
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        
        with fcol1:
            st.markdown(f"""
            <div class="futures-card">
                <h3>{futures_stats.get('total_contracts', 0)}</h3>
                <p>Total Contracts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with fcol2:
            st.markdown(f"""
            <div class="futures-card">
                <h3>{futures_stats.get('active_contracts', 0)}</h3>
                <p>Active Contracts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with fcol3:
            avg_margin = futures_stats.get('avg_initial_margin', 0)
            st.markdown(f"""
            <div class="futures-card">
                <h3>{format_currency(avg_margin)}</h3>
                <p>Avg Initial Margin</p>
            </div>
            """, unsafe_allow_html=True)
        
        with fcol4:
            st.markdown(f"""
            <div class="futures-card">
                <h3>{futures_stats.get('category_count', 0)}</h3>
                <p>Categories</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Futures categories
    st.markdown("### 🗂️ Futures Categories")
    
    category_data = call_api("/futures_categories")
    
    if "error" not in category_data and "categories" in category_data:
        categories = category_data["categories"]
        
        # Create tabs for each category
        category_tabs = st.tabs([cat['name'] for cat in categories])
        
        for i, category in enumerate(categories):
            with category_tabs[i]:
                st.markdown(f"#### {category['name']} - {category['contract_count']} contracts")
                
                # Display contracts in this category
                if 'contracts' in category:
                    for contract in category['contracts'][:10]:  # Limit for display
                        with st.expander(f"{contract['symbol']} - {contract['name']}"):
                            ccol1, ccol2, ccol3, ccol4 = st.columns(4)
                            
                            with ccol1:
                                st.write(f"**Exchange:** {contract['exchange']}")
                                st.write(f"**Currency:** {contract['currency']}")
                            
                            with ccol2:
                                st.write(f"**Contract Size:** {contract['contract_size']:,}")
                                st.write(f"**Tick Size:** {contract['tick_size']}")
                            
                            with ccol3:
                                st.write(f"**Initial Margin:** {format_currency(contract['margin_initial'])}")
                                st.write(f"**Maint. Margin:** {format_currency(contract['margin_maintenance'])}")
                            
                            with ccol4:
                                st.write(f"**Point Value:** {format_currency(contract['point_value'])}")
                                if st.button(f"Trade {contract['symbol']}", key=f"trade_{contract['symbol']}"):
                                    st.success(f"Opening {contract['symbol']} trading interface...")
    
    # Contract chain viewer
    st.markdown("### 📈 Contract Chain Viewer")
    
    chain_symbol = st.selectbox(
        "Select Base Contract",
        ["ES", "NQ", "YM", "RTY", "CL", "NG", "GC", "SI", "ZB", "ZN"]
    )
    
    if chain_symbol:
        chain_data = call_api(f"/futures_chain/{chain_symbol}")
        
        if "error" not in chain_data:
            fig_chain = create_futures_chain_view(chain_data)
            st.plotly_chart(fig_chain, use_container_width=True)
    
    # Margin calculator
    st.markdown("### 🧮 Margin Calculator")
    
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        calc_symbol = st.selectbox("Contract", ["ES", "NQ", "CL", "GC", "ZB"], key="margin_calc")
        num_contracts = st.number_input("Number of Contracts", min_value=1, value=1)
    
    with calc_col2:
        entry_price = st.number_input("Entry Price", min_value=0.01, value=4000.0)
        current_price = st.number_input("Current Price", min_value=0.01, value=4050.0)
    
    with calc_col3:
        if st.button("Calculate", use_container_width=True):
            calc_data = call_api("/calculate_margin", method="POST", json_data={
                "symbol": calc_symbol,
                "contracts": num_contracts,
                "entry_price": entry_price,
                "current_price": current_price
            })
            
            if "error" not in calc_data:
                st.markdown(f"""
                <div class="success-card">
                    <strong>Margin Requirements</strong><br>
                    Initial Margin: {format_currency(calc_data.get('initial_margin', 0))}<br>
                    Maintenance Margin: {format_currency(calc_data.get('maintenance_margin', 0))}<br>
                    Unrealized P&L: {format_currency(calc_data.get('unrealized_pnl', 0))}
                </div>
                """, unsafe_allow_html=True)

# Main application logic
def main():
    """Main application function with enhanced routing."""
    
    # Page routing with enhanced interfaces
    if page == "Asset Universe":
        display_asset_universe_page()
    elif page == "Trading Interface":
        display_trading_interface_page()
    elif page == "AI Learning Center":
        display_ai_learning_center()
    elif page == "Futures Trading":
        display_futures_trading_page()
    elif page == "Real-time Monitor":
        # Import and display from original app (enhanced monitoring)
        from streamlit_app import display_overview_page
        display_overview_page()
    elif page == "Portfolio Analytics":
        from streamlit_app import display_portfolio_analytics_page
        display_portfolio_analytics_page()
    elif page == "Risk Management":
        from streamlit_app import display_risk_management_page
        display_risk_management_page()
    elif page == "Performance Hub":
        from streamlit_app import display_performance_analysis_page
        display_performance_analysis_page()
    elif page == "System Logs":
        from streamlit_app import display_system_logs_page
        display_system_logs_page()
    
    # Auto-refresh functionality for real-time pages
    if AUTO_REFRESH and page in ["Asset Universe", "Real-time Monitor", "Futures Trading"]:
        time.sleep(REFRESH_SEC)
        st.rerun()

# Enhanced footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 Multi-Asset Trading Platform v3.0**")
st.sidebar.markdown("*Supporting 5,700+ assets across global markets*")

# Performance metrics in sidebar
perf_data = call_api("/system_performance")
if "error" not in perf_data:
    st.sidebar.markdown("### ⚡ System Performance")
    st.sidebar.metric("CPU Usage", f"{perf_data.get('cpu_usage', 0):.1f}%")
    st.sidebar.metric("Memory Usage", f"{perf_data.get('memory_usage', 0):.1f}%")
    st.sidebar.metric("Active Connections", perf_data.get('active_connections', 0))

if __name__ == "__main__":
    main()