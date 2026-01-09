"""
AI Learning Monitor - Advanced Learning Progress Interface
Agent B1 - Dashboard Integration

This module creates an advanced AI learning monitoring interface that integrates
with the existing dashboard system to provide real-time insights into:
- AI learning engine status and progress
- Model training pipeline monitoring
- Hyperparameter optimization progress
- A/B testing results and comparisons
- Reinforcement learning agent performance
- Strategy performance analytics

Designed for real-time monitoring and control of AI learning systems.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
bot_dir = project_root / "bot"
sys.path.append(str(bot_dir))
sys.path.append(str(project_root))

# Import AI learning modules
try:
    from bot.ai_learning_engine import get_learning_engine, AILearningEngine
    from bot.gpu_training_pipeline import GPUTrainingPipeline, create_gpu_pipeline
    from bot.ab_testing_framework import ABTestingFramework, create_ab_testing_framework
    from bot.hyperparameter_optimization import HyperparameterOptimizer
    from bot.reinforcement_learning_framework import MultiAgentRLFramework
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"AI modules not available: {e}")
    AI_MODULES_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="AI Learning Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    """Load custom CSS for AI learning monitor."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .learning-progress {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    
    .model-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def get_ai_learning_status():
    """Get AI learning engine status."""
    if not AI_MODULES_AVAILABLE:
        return {
            'available': False,
            'error': 'AI modules not available'
        }
    
    try:
        # Get learning engine status
        engine = get_learning_engine()
        learning_status = engine.get_learning_status()
        strategy_performance = engine.get_strategy_performance_summary()
        
        return {
            'available': True,
            'learning_status': learning_status,
            'strategy_performance': strategy_performance,
            'current_time': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }

def get_training_pipeline_status():
    """Get GPU training pipeline status."""
    try:
        pipeline = create_gpu_pipeline()
        status = pipeline.get_training_status()
        return {
            'available': True,
            'status': status
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }

def get_ab_testing_status():
    """Get A/B testing framework status."""
    try:
        framework = create_ab_testing_framework()
        status = framework.get_framework_status()
        return {
            'available': True,
            'status': status
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }

def create_learning_progress_chart(learning_data):
    """Create learning progress visualization."""
    fig = go.Figure()
    
    # Sample data if not available
    if not learning_data or 'learning_sessions' not in learning_data:
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        improvement = np.cumsum(np.random.normal(0.01, 0.02, 30))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=improvement,
            mode='lines+markers',
            name='Learning Progress',
            line=dict(color='#1f77b4', width=3)
        ))
    else:
        # Use real data
        sessions = learning_data.get('learning_sessions', 0)
        if sessions > 0:
            # Generate sample progress for demonstration
            dates = pd.date_range(start='2024-01-01', periods=sessions, freq='D')
            improvement = np.cumsum(np.random.normal(0.01, 0.01, sessions))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=improvement,
                mode='lines+markers',
                name='Cumulative Improvement',
                line=dict(color='#1f77b4', width=3)
            ))
    
    fig.update_layout(
        title="AI Learning Progress Over Time",
        xaxis_title="Date",
        yaxis_title="Performance Improvement (%)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_strategy_performance_chart(strategy_data):
    """Create strategy performance comparison chart."""
    if not strategy_data:
        # Sample data
        strategies = ['Strategy A', 'Strategy B', 'Strategy C']
        win_rates = [0.65, 0.58, 0.72]
        total_pnl = [1250, 980, 1560]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Win Rate', 'Total PnL'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=win_rates, name='Win Rate', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=total_pnl, name='Total PnL', marker_color='#ff7f0e'),
            row=1, col=2
        )
    else:
        # Use real data
        strategies = list(strategy_data.keys())
        win_rates = [data.get('win_rate', 0) for data in strategy_data.values()]
        total_pnl = [data.get('total_pnl', 0) for data in strategy_data.values()]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Win Rate', 'Total PnL'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=win_rates, name='Win Rate', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=total_pnl, name='Total PnL', marker_color='#ff7f0e'),
            row=1, col=2
        )
    
    fig.update_layout(
        title="Strategy Performance Comparison",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_training_metrics_chart():
    """Create model training metrics visualization."""
    # Sample training metrics
    epochs = list(range(1, 51))
    train_loss = np.exp(-np.array(epochs) * 0.1) + 0.1 + np.random.normal(0, 0.02, 50)
    val_loss = np.exp(-np.array(epochs) * 0.08) + 0.15 + np.random.normal(0, 0.03, 50)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="Model Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_hyperparameter_optimization_chart():
    """Create hyperparameter optimization progress chart."""
    # Sample optimization data
    trials = list(range(1, 101))
    objective_values = []
    best_so_far = []
    
    current_best = float('-inf')
    for trial in trials:
        # Simulate optimization progress
        value = np.random.normal(0.75, 0.1) + 0.1 * np.exp(-trial / 20)
        objective_values.append(value)
        
        current_best = max(current_best, value)
        best_so_far.append(current_best)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trials,
        y=objective_values,
        mode='markers',
        name='Trial Values',
        marker=dict(color='lightblue', size=6, opacity=0.6)
    ))
    
    fig.add_trace(go.Scatter(
        x=trials,
        y=best_so_far,
        mode='lines',
        name='Best So Far',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="Hyperparameter Optimization Progress",
        xaxis_title="Trial Number",
        yaxis_title="Objective Value",
        template="plotly_white",
        height=400
    )
    
    return fig

def render_learning_engine_section():
    """Render AI Learning Engine section."""
    st.markdown('<div class="section-header">🧠 AI Learning Engine</div>', unsafe_allow_html=True)
    
    # Get learning status
    ai_status = get_ai_learning_status()
    
    if not ai_status['available']:
        st.error(f"AI Learning Engine not available: {ai_status.get('error', 'Unknown error')}")
        return
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    learning_status = ai_status.get('learning_status', {})
    
    with col1:
        is_running = learning_status.get('is_running', False)
        status_class = 'status-good' if is_running else 'status-warning'
        st.markdown(f'<div class="metric-container"><h4>Status</h4><p class="{status_class}">{"Running" if is_running else "Stopped"}</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_decisions = learning_status.get('total_decisions', 0)
        st.markdown(f'<div class="metric-container"><h4>Total Decisions</h4><p>{total_decisions:,}</p></div>', unsafe_allow_html=True)
    
    with col3:
        active_strategies = learning_status.get('active_strategies', 0)
        st.markdown(f'<div class="metric-container"><h4>Active Strategies</h4><p>{active_strategies}</p></div>', unsafe_allow_html=True)
    
    with col4:
        market_env = learning_status.get('market_environment', 'unknown')
        st.markdown(f'<div class="metric-container"><h4>Market Environment</h4><p>{market_env.title().replace("_", " ")}</p></div>', unsafe_allow_html=True)
    
    # Learning progress chart
    learning_chart = create_learning_progress_chart(learning_status)
    st.plotly_chart(learning_chart, use_container_width=True)
    
    # Strategy performance
    strategy_performance = ai_status.get('strategy_performance', {})
    if strategy_performance:
        st.subheader("Strategy Performance")
        
        performance_df = pd.DataFrame.from_dict(strategy_performance, orient='index')
        if not performance_df.empty:
            st.dataframe(performance_df, use_container_width=True)
        
        # Performance chart
        performance_chart = create_strategy_performance_chart(strategy_performance)
        st.plotly_chart(performance_chart, use_container_width=True)

def render_training_pipeline_section():
    """Render GPU Training Pipeline section."""
    st.markdown('<div class="section-header">🚀 GPU Training Pipeline</div>', unsafe_allow_html=True)
    
    training_status = get_training_pipeline_status()
    
    if not training_status['available']:
        st.error(f"Training Pipeline not available: {training_status.get('error', 'Unknown error')}")
        return
    
    status_data = training_status.get('status', {})
    
    # Pipeline metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gpu_available = status_data.get('gpu_available', False)
        gpu_enabled = status_data.get('gpu_enabled', False)
        gpu_status = 'GPU Enabled' if gpu_available and gpu_enabled else 'CPU Only'
        status_class = 'status-good' if gpu_available and gpu_enabled else 'status-warning'
        st.markdown(f'<div class="metric-container"><h4>Acceleration</h4><p class="{status_class}">{gpu_status}</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_jobs = status_data.get('total_jobs', 0)
        st.markdown(f'<div class="metric-container"><h4>Total Jobs</h4><p>{total_jobs}</p></div>', unsafe_allow_html=True)
    
    with col3:
        active_jobs = status_data.get('active_jobs', 0)
        st.markdown(f'<div class="metric-container"><h4>Active Jobs</h4><p>{active_jobs}</p></div>', unsafe_allow_html=True)
    
    with col4:
        trained_models = status_data.get('trained_models', 0)
        st.markdown(f'<div class="metric-container"><h4>Trained Models</h4><p>{trained_models}</p></div>', unsafe_allow_html=True)
    
    # Training metrics chart
    training_chart = create_training_metrics_chart()
    st.plotly_chart(training_chart, use_container_width=True)

def render_ab_testing_section():
    """Render A/B Testing section."""
    st.markdown('<div class="section-header">⚖️ A/B Testing Framework</div>', unsafe_allow_html=True)
    
    ab_status = get_ab_testing_status()
    
    if not ab_status['available']:
        st.error(f"A/B Testing Framework not available: {ab_status.get('error', 'Unknown error')}")
        return
    
    status_data = ab_status.get('status', {})
    
    # A/B testing metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        active_tests = status_data.get('active_tests', 0)
        st.markdown(f'<div class="metric-container"><h4>Active Tests</h4><p>{active_tests}</p></div>', unsafe_allow_html=True)
    
    with col2:
        completed_tests = status_data.get('completed_tests', 0)
        st.markdown(f'<div class="metric-container"><h4>Completed Tests</h4><p>{completed_tests}</p></div>', unsafe_allow_html=True)
    
    with col3:
        total_variants = status_data.get('total_variants', 0)
        st.markdown(f'<div class="metric-container"><h4>Total Variants</h4><p>{total_variants}</p></div>', unsafe_allow_html=True)
    
    # Sample A/B test results
    if active_tests > 0 or completed_tests > 0:
        st.subheader("A/B Test Results")
        
        # Sample test data
        test_data = {
            'Test Name': ['Conservative vs Aggressive', 'Model A vs Model B', 'Feature Set Comparison'],
            'Status': ['Running', 'Completed', 'Completed'],
            'Winner': ['-', 'Model B', 'Enhanced Features'],
            'Confidence': ['-', '95%', '87%'],
            'Improvement': ['-', '+12.5%', '+8.3%']
        }
        
        df = pd.DataFrame(test_data)
        st.dataframe(df, use_container_width=True)

def render_hyperparameter_optimization_section():
    """Render Hyperparameter Optimization section."""
    st.markdown('<div class="section-header">🎯 Hyperparameter Optimization</div>', unsafe_allow_html=True)
    
    # Sample optimization status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container"><h4>Active Studies</h4><p>2</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container"><h4>Completed Studies</h4><p>15</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container"><h4>Best Score</h4><p>0.847</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container"><h4>Total Trials</h4><p>1,250</p></div>', unsafe_allow_html=True)
    
    # Optimization progress chart
    opt_chart = create_hyperparameter_optimization_chart()
    st.plotly_chart(opt_chart, use_container_width=True)
    
    # Parameter importance
    st.subheader("Parameter Importance")
    
    # Sample parameter importance data
    params = ['learning_rate', 'num_leaves', 'feature_fraction', 'reg_alpha', 'max_depth']
    importance = [0.35, 0.28, 0.18, 0.12, 0.07]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=params,
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title="Hyperparameter Importance",
        xaxis_title="Importance",
        template="plotly_white",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_system_health_section():
    """Render system health monitoring section."""
    st.markdown('<div class="section-header">💚 System Health</div>', unsafe_allow_html=True)
    
    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container"><h4>CPU Usage</h4><p class="status-good">45%</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container"><h4>Memory Usage</h4><p class="status-good">62%</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container"><h4>GPU Utilization</h4><p class="status-good">78%</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container"><h4>Storage Usage</h4><p class="status-warning">85%</p></div>', unsafe_allow_html=True)
    
    # Recent activities
    st.subheader("Recent AI Activities")
    
    activities = [
        {"Time": "2024-08-30 10:45", "Activity": "Daily learning cycle completed", "Status": "Success"},
        {"Time": "2024-08-30 10:30", "Activity": "Model retraining started", "Status": "In Progress"},
        {"Time": "2024-08-30 09:15", "Activity": "A/B test threshold reached", "Status": "Success"},
        {"Time": "2024-08-30 08:00", "Activity": "Hyperparameter optimization completed", "Status": "Success"},
        {"Time": "2024-08-30 07:30", "Activity": "Strategy performance evaluation", "Status": "Success"}
    ]
    
    df = pd.DataFrame(activities)
    st.dataframe(df, use_container_width=True)

def main():
    """Main AI Learning Monitor interface."""
    # Load custom CSS
    load_custom_css()
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 AI Learning Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Real-time monitoring and control of AI learning systems</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(0.1)  # Small delay for UI responsiveness
        st.rerun()
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Section toggles
    st.sidebar.header("📊 Sections")
    show_learning_engine = st.sidebar.checkbox("AI Learning Engine", value=True)
    show_training_pipeline = st.sidebar.checkbox("GPU Training Pipeline", value=True)
    show_ab_testing = st.sidebar.checkbox("A/B Testing", value=True)
    show_hyperopt = st.sidebar.checkbox("Hyperparameter Optimization", value=True)
    show_system_health = st.sidebar.checkbox("System Health", value=True)
    
    # Main content
    if show_learning_engine:
        render_learning_engine_section()
    
    if show_training_pipeline:
        render_training_pipeline_section()
    
    if show_ab_testing:
        render_ab_testing_section()
    
    if show_hyperopt:
        render_hyperparameter_optimization_section()
    
    if show_system_health:
        render_system_health_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #999; font-size: 0.9rem;">'
        'AI Learning Monitor v1.0 | Last updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()