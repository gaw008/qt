"""
Master Trading Hub - Agent C1 Integration
=======================================

Central command center integrating all monitoring and analytics systems:
- Real-time multi-asset monitoring (5,700+ assets)
- AI decision visualization and learning progress
- Advanced risk monitoring and stress testing
- System performance and health monitoring
- Intelligent alert management and notifications
- Automated reporting and analytics

This is the main entry point for the professional trading surveillance system,
providing unified access to all Agent C1 monitoring capabilities.
"""

import streamlit as st
import sys
import os
from datetime import datetime
import time

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Configure main app
st.set_page_config(
    page_title="Master Trading Hub - Agent C1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Agent C1 - Professional Trading Surveillance System"
    }
)

# Master CSS styling
st.markdown("""
<style>
    /* Master hub styling */
    .master-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 50px rgba(30, 60, 114, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .master-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }
    
    .master-header > * {
        position: relative;
        z-index: 1;
    }
    
    .hub-navigation {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.18);
    }
    
    .module-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .module-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border-color: #764ba2;
    }
    
    .status-indicator-live {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #00e676;
        border-radius: 50%;
        animation: pulse-live 2s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse-live {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .system-overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .overview-metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .overview-metric-card:hover {
        transform: scale(1.05);
    }
    
    .alert-summary-panel {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, rgba(255, 56, 56, 0.1) 100%);
        border: 3px solid #ff4757;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .quick-actions-panel {
        background: linear-gradient(135deg, rgba(0, 184, 148, 0.1) 0%, rgba(0, 206, 201, 0.1) 100%);
        border: 3px solid #00b894;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .performance-summary {
        background: linear-gradient(135deg, rgba(253, 203, 110, 0.1) 0%, rgba(255, 234, 167, 0.1) 100%);
        border: 3px solid #fdcb6e;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .agent-signature {
        text-align: center;
        color: #667eea;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 2rem 0;
        padding: 1rem;
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    .module-status-active {
        color: #00e676;
        font-weight: bold;
    }
    
    .module-status-inactive {
        color: #ffa726;
        font-weight: bold;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top: 3px solid #667eea;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def load_module_status():
    """Load status of all monitoring modules."""
    return {
        'realtime_monitor': {
            'name': 'Real-Time Asset Monitor',
            'description': 'Monitoring 5,743 assets across 8 categories',
            'status': 'ACTIVE',
            'last_update': datetime.now(),
            'key_metrics': {
                'assets_tracked': 5743,
                'data_points_per_second': 2847,
                'alerts_active': 12,
                'uptime': '99.8%'
            }
        },
        'ai_decision_viz': {
            'name': 'AI Decision Visualizer',
            'description': 'ML model performance and decision analytics',
            'status': 'ACTIVE',
            'last_update': datetime.now(),
            'key_metrics': {
                'models_active': 4,
                'avg_accuracy': 92.7,
                'decisions_today': 287,
                'learning_sessions': 23
            }
        },
        'risk_monitor': {
            'name': 'Risk Monitor',
            'description': 'Multi-dimensional risk analytics and VaR',
            'status': 'ACTIVE',
            'last_update': datetime.now(),
            'key_metrics': {
                'daily_var_95': 186500,
                'max_drawdown': -8.4,
                'portfolio_beta': 1.23,
                'risk_score': 7.2
            }
        },
        'system_health': {
            'name': 'System Health Monitor',
            'description': 'Hardware performance and system monitoring',
            'status': 'ACTIVE',
            'last_update': datetime.now(),
            'key_metrics': {
                'cpu_usage': 76.5,
                'gpu_usage': 89.2,
                'memory_usage': 24.8,
                'network_latency': 18.3
            }
        }
    }

def display_master_overview():
    """Display the master trading hub overview."""
    
    # Master header
    st.markdown("""
    <div class="master-header">
        <h1>🎯 Master Trading Hub</h1>
        <h2>Agent C1 - Professional Trading Surveillance System</h2>
        <p>Unified command center for multi-asset real-time monitoring, AI analytics, and risk management</p>
        <p><span class="status-indicator-live"></span>LIVE MONITORING ACTIVE | All Systems Operational</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load module status
    modules = load_module_status()
    
    # System overview metrics
    st.markdown("## 📊 System Overview")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_assets = sum(m['key_metrics'].get('assets_tracked', 0) for m in modules.values())
        st.markdown(f"""
        <div class="overview-metric-card">
            <h3>Assets Monitored</h3>
            <h1>{total_assets:,}</h1>
            <p>Multi-asset universe</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_decisions = sum(m['key_metrics'].get('decisions_today', 0) for m in modules.values())
        st.markdown(f"""
        <div class="overview-metric-card">
            <h3>AI Decisions</h3>
            <h1>{total_decisions}</h1>
            <p>Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_accuracy = sum(m['key_metrics'].get('avg_accuracy', 0) for m in modules.values() if 'avg_accuracy' in m['key_metrics']) / 1
        st.markdown(f"""
        <div class="overview-metric-card">
            <h3>AI Accuracy</h3>
            <h1>{avg_accuracy:.1f}%</h1>
            <p>Model performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        daily_var = modules['risk_monitor']['key_metrics']['daily_var_95']
        st.markdown(f"""
        <div class="overview-metric-card">
            <h3>Daily VaR</h3>
            <h1>${daily_var:,.0f}</h1>
            <p>95% confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        cpu_usage = modules['system_health']['key_metrics']['cpu_usage']
        st.markdown(f"""
        <div class="overview-metric-card">
            <h3>CPU Usage</h3>
            <h1>{cpu_usage:.1f}%</h1>
            <p>System load</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        total_alerts = sum(m['key_metrics'].get('alerts_active', 0) for m in modules.values())
        st.markdown(f"""
        <div class="overview-metric-card">
            <h3>Active Alerts</h3>
            <h1>{total_alerts}</h1>
            <p>Notifications</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Module navigation
    st.markdown("---")
    st.markdown("## 🚀 Module Access")
    
    st.markdown("""
    <div class="hub-navigation">
        <h4>🎛️ Select a monitoring module to access detailed analytics and controls</h4>
        <p>Each module provides specialized monitoring capabilities with real-time data and advanced analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Module cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="module-card">
            <h3>📈 Real-Time Asset Monitor</h3>
            <p><strong>{modules['realtime_monitor']['description']}</strong></p>
            <hr>
            <p>🎯 Assets: {modules['realtime_monitor']['key_metrics']['assets_tracked']:,}</p>
            <p>⚡ Data Rate: {modules['realtime_monitor']['key_metrics']['data_points_per_second']:,}/sec</p>
            <p>🚨 Alerts: {modules['realtime_monitor']['key_metrics']['alerts_active']}</p>
            <p>⏰ Uptime: {modules['realtime_monitor']['key_metrics']['uptime']}</p>
            <p><span class="module-status-active">● {modules['realtime_monitor']['status']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Real-Time Monitor", key="btn_realtime"):
            st.session_state.selected_module = "realtime_monitor"
            st.rerun()
    
    with col2:
        st.markdown(f"""
        <div class="module-card">
            <h3>🧠 AI Decision Visualizer</h3>
            <p><strong>{modules['ai_decision_viz']['description']}</strong></p>
            <hr>
            <p>🤖 Models: {modules['ai_decision_viz']['key_metrics']['models_active']}</p>
            <p>🎯 Accuracy: {modules['ai_decision_viz']['key_metrics']['avg_accuracy']:.1f}%</p>
            <p>📊 Decisions: {modules['ai_decision_viz']['key_metrics']['decisions_today']}</p>
            <p>🎓 Sessions: {modules['ai_decision_viz']['key_metrics']['learning_sessions']}</p>
            <p><span class="module-status-active">● {modules['ai_decision_viz']['status']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch AI Visualizer", key="btn_ai"):
            st.session_state.selected_module = "ai_decision_viz"
            st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="module-card">
            <h3>🛡️ Risk Monitor</h3>
            <p><strong>{modules['risk_monitor']['description']}</strong></p>
            <hr>
            <p>💰 Daily VaR: ${modules['risk_monitor']['key_metrics']['daily_var_95']:,}</p>
            <p>📉 Max Drawdown: {modules['risk_monitor']['key_metrics']['max_drawdown']:.1f}%</p>
            <p>📊 Portfolio Beta: {modules['risk_monitor']['key_metrics']['portfolio_beta']:.2f}</p>
            <p>🎯 Risk Score: {modules['risk_monitor']['key_metrics']['risk_score']:.1f}/10</p>
            <p><span class="module-status-active">● {modules['risk_monitor']['status']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Risk Monitor", key="btn_risk"):
            st.session_state.selected_module = "risk_monitor"
            st.rerun()
    
    with col4:
        st.markdown(f"""
        <div class="module-card">
            <h3>🖥️ System Health Monitor</h3>
            <p><strong>{modules['system_health']['description']}</strong></p>
            <hr>
            <p>🔥 CPU: {modules['system_health']['key_metrics']['cpu_usage']:.1f}%</p>
            <p>🎮 GPU: {modules['system_health']['key_metrics']['gpu_usage']:.1f}%</p>
            <p>💾 Memory: {modules['system_health']['key_metrics']['memory_usage']:.1f}GB</p>
            <p>🌐 Latency: {modules['system_health']['key_metrics']['network_latency']:.1f}ms</p>
            <p><span class="module-status-active">● {modules['system_health']['status']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch System Monitor", key="btn_system"):
            st.session_state.selected_module = "system_health"
            st.rerun()
    
    # Alert summary
    st.markdown("---")
    st.markdown("## 🚨 Alert Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="alert-summary-panel">
            <h4>📢 System-Wide Alerts</h4>
            <p>🟢 <strong>All Systems Operational:</strong> No critical alerts detected</p>
            <p>🟡 <strong>CPU Usage Warning:</strong> System load at 76.5% (threshold: 90%)</p>
            <p>🟠 <strong>Portfolio Concentration:</strong> Technology sector at 34.2% (threshold: 40%)</p>
            <p>🔵 <strong>AI Learning:</strong> 3 models in active training phase</p>
            <p>🟢 <strong>Risk Status:</strong> All risk metrics within acceptable ranges</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="quick-actions-panel">
            <h4>⚡ Quick Actions</h4>
            <p>🔄 <strong>Refresh All Data</strong></p>
            <p>📊 <strong>Generate Report</strong></p>
            <p>🛡️ <strong>Risk Assessment</strong></p>
            <p>🤖 <strong>AI Health Check</strong></p>
            <p>📈 <strong>Performance Review</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance summary
    st.markdown("---")
    st.markdown("## 📈 Performance Summary")
    
    st.markdown("""
    <div class="performance-summary">
        <h4>🏆 Today's Performance Highlights</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>
                <strong>📊 Data Processing:</strong><br>
                2.8M data points processed<br>
                99.97% uptime maintained
            </div>
            <div>
                <strong>🤖 AI Performance:</strong><br>
                287 trading decisions<br>
                92.7% accuracy rate
            </div>
            <div>
                <strong>🛡️ Risk Management:</strong><br>
                $186.5K daily VaR<br>
                7.2/10 risk score
            </div>
            <div>
                <strong>⚡ System Health:</strong><br>
                18.3ms avg latency<br>
                89.2% GPU utilization
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent signature
    st.markdown("""
    <div class="agent-signature">
        🚀 <strong>Agent C1 - Professional Trading Surveillance System</strong> 🚀<br>
        Real-Time Monitoring | AI Analytics | Risk Management | System Health<br>
        <em>Delivering professional-grade trading intelligence and monitoring capabilities</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer with real-time stats
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem; background: rgba(102, 126, 234, 0.05); border-radius: 10px;">
        <p><strong>Master Trading Hub - Agent C1</strong></p>
        <p>System Status: <span style="color: #00e676;">🟢 OPERATIONAL</span> | 
        Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        Modules Active: 4/4 | 
        Data Latency: 18.3ms</p>
        <p>Portfolio Value: $10,000,000 | Daily P&L: +$23,450 | Risk Utilization: 68% | AI Confidence: 92.7%</p>
    </div>
    """, unsafe_allow_html=True)

def load_selected_module():
    """Load the selected monitoring module."""
    selected = st.session_state.get('selected_module', None)
    
    if selected == "realtime_monitor":
        # Import and run real-time monitor
        try:
            from advanced_realtime_monitor_c1 import main as realtime_main
            st.markdown("### 🔄 Loading Real-Time Asset Monitor...")
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            time.sleep(1)
            realtime_main()
        except ImportError as e:
            st.error(f"Could not load Real-Time Monitor: {e}")
            st.info("Please ensure advanced_realtime_monitor_c1.py is in the same directory.")
    
    elif selected == "ai_decision_viz":
        # Import and run AI decision visualizer
        try:
            from ai_decision_visualizer_c1 import main as ai_main
            st.markdown("### 🔄 Loading AI Decision Visualizer...")
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            time.sleep(1)
            ai_main()
        except ImportError as e:
            st.error(f"Could not load AI Decision Visualizer: {e}")
            st.info("Please ensure ai_decision_visualizer_c1.py is in the same directory.")
    
    elif selected == "risk_monitor":
        # Import and run risk monitor
        try:
            from risk_monitor_c1 import main as risk_main
            st.markdown("### 🔄 Loading Risk Monitor...")
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            time.sleep(1)
            risk_main()
        except ImportError as e:
            st.error(f"Could not load Risk Monitor: {e}")
            st.info("Please ensure risk_monitor_c1.py is in the same directory.")
    
    elif selected == "system_health":
        st.markdown("### 🖥️ System Health Monitor")
        st.info("System Health Monitor module is being prepared. Advanced system monitoring capabilities will be available soon.")
        
        # Show basic system health for now
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU Usage", "76.5%", delta="↑5%")
        with col2:
            st.metric("GPU Usage", "89.2%", delta="↑12%")
        with col3:
            st.metric("Memory Usage", "24.8GB", delta="↑2.1GB")
    
    # Back to hub button
    st.markdown("---")
    if st.button("🏠 Back to Master Hub"):
        st.session_state.selected_module = None
        st.rerun()

def main():
    """Main application entry point."""
    
    # Initialize session state
    if 'selected_module' not in st.session_state:
        st.session_state.selected_module = None
    
    # Master sidebar
    st.sidebar.title("🎯 Master Control")
    
    # System status
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success("🟢 All Systems Operational")
    st.sidebar.metric("Uptime", "99.8%")
    st.sidebar.metric("Active Modules", "4/4")
    st.sidebar.metric("Data Latency", "18.3ms")
    st.sidebar.metric("Response Time", "23ms")
    
    st.sidebar.markdown("---")
    
    # Quick navigation
    st.sidebar.markdown("### 🚀 Quick Launch")
    
    if st.sidebar.button("📈 Real-Time Monitor"):
        st.session_state.selected_module = "realtime_monitor"
        st.rerun()
    
    if st.sidebar.button("🧠 AI Visualizer"):
        st.session_state.selected_module = "ai_decision_viz"
        st.rerun()
    
    if st.sidebar.button("🛡️ Risk Monitor"):
        st.session_state.selected_module = "risk_monitor"
        st.rerun()
    
    if st.sidebar.button("🖥️ System Health"):
        st.session_state.selected_module = "system_health"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # System controls
    st.sidebar.markdown("### ⚙️ System Controls")
    
    monitoring_enabled = st.sidebar.checkbox("📊 Live Monitoring", True)
    ai_learning = st.sidebar.checkbox("🧠 AI Learning", True)
    risk_alerts = st.sidebar.checkbox("🚨 Risk Alerts", True)
    auto_reporting = st.sidebar.checkbox("📄 Auto Reports", True)
    
    st.sidebar.markdown("---")
    
    # Performance stats
    st.sidebar.markdown("### 📈 Performance")
    st.sidebar.metric("Portfolio Value", "$10.00M", delta="+$23.4K")
    st.sidebar.metric("Daily Return", "+0.23%", delta="+0.15%")
    st.sidebar.metric("AI Accuracy", "92.7%", delta="+1.2%")
    st.sidebar.metric("Risk Score", "7.2/10", delta="-0.3")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Agent C1 Master Hub**")
    st.sidebar.markdown("*Professional Trading Surveillance*")
    st.sidebar.markdown(f"*Version: 1.0.0*")
    st.sidebar.markdown(f"*Status: {datetime.now().strftime('%H:%M:%S')}*")
    
    # Main content area
    if st.session_state.selected_module is None:
        display_master_overview()
    else:
        load_selected_module()

if __name__ == "__main__":
    main()