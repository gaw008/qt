#!/usr/bin/env python3
"""
Market Regime Visualization and Reporting Module

This module provides comprehensive visualization and reporting capabilities for the
Market Regime Classification System, including:

- Historical regime timeline visualization
- Regime transition analysis and statistics
- Performance metrics and validation charts
- Real-time regime monitoring dashboards
- Export capabilities for reports and analysis

Features:
- Interactive regime timeline plots
- Regime probability heatmaps
- Transition frequency analysis
- Crisis period overlay validation
- Performance attribution by regime
- Export to various formats (HTML, PDF, PNG)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

# Plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import regime classifier
try:
    from market_regime_classifier import (
        MarketRegimeClassifier,
        MarketRegime,
        RegimePrediction,
        RegimeTransition
    )
    REGIME_CLASSIFIER_AVAILABLE = True
except ImportError:
    REGIME_CLASSIFIER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class RegimeVisualization:
    """
    Comprehensive visualization system for market regime analysis
    """

    def __init__(self, classifier: Optional['MarketRegimeClassifier'] = None):
        """
        Initialize visualization system

        Args:
            classifier: MarketRegimeClassifier instance (creates new if None)
        """
        if not REGIME_CLASSIFIER_AVAILABLE:
            raise ImportError("Market regime classifier not available")

        self.classifier = classifier or MarketRegimeClassifier()

        # Color scheme for regimes
        self.regime_colors = {
            MarketRegime.NORMAL: '#2E8B57',      # Sea Green
            MarketRegime.VOLATILE: '#FF8C00',     # Dark Orange
            MarketRegime.CRISIS: '#DC143C',       # Crimson
            'NORMAL': '#2E8B57',
            'VOLATILE': '#FF8C00',
            'CRISIS': '#DC143C'
        }

        # Style configuration
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
            sns.set_palette("husl")

    def plot_regime_timeline(self,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            include_probabilities: bool = True,
                            include_indicators: bool = True,
                            save_path: Optional[str] = None,
                            interactive: bool = False) -> Optional[Any]:
        """
        Create regime timeline visualization

        Args:
            start_date: Start date for timeline
            end_date: End date for timeline
            include_probabilities: Whether to include probability bands
            include_indicators: Whether to include key indicators
            save_path: Path to save the plot
            interactive: Use Plotly for interactive plot

        Returns:
            Figure object or None if error
        """
        logger.info("Creating regime timeline visualization...")

        try:
            # Get historical regime data
            history_df = self.classifier.get_historical_regimes(start_date, end_date)

            if history_df.empty:
                logger.warning("No historical regime data available")
                return None

            if interactive and PLOTLY_AVAILABLE:
                return self._plot_regime_timeline_plotly(
                    history_df, include_probabilities, include_indicators, save_path
                )
            elif MATPLOTLIB_AVAILABLE:
                return self._plot_regime_timeline_matplotlib(
                    history_df, include_probabilities, include_indicators, save_path
                )
            else:
                logger.error("No plotting libraries available")
                return None

        except Exception as e:
            logger.error(f"Failed to create regime timeline: {e}")
            return None

    def _plot_regime_timeline_matplotlib(self,
                                       history_df: pd.DataFrame,
                                       include_probabilities: bool,
                                       include_indicators: bool,
                                       save_path: Optional[str]) -> plt.Figure:
        """Create matplotlib regime timeline"""
        n_subplots = 1 + int(include_probabilities) + int(include_indicators)
        fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 4 * n_subplots))

        if n_subplots == 1:
            axes = [axes]

        subplot_idx = 0

        # Main regime timeline
        ax_main = axes[subplot_idx]
        subplot_idx += 1

        # Map regime to numeric values for plotting
        regime_map = {'NORMAL': 0, 'VOLATILE': 1, 'CRISIS': 2}
        numeric_regimes = history_df['regime'].map(regime_map)

        # Create colored background
        for i, (date, regime) in enumerate(zip(history_df['date'], history_df['regime'])):
            color = self.regime_colors[regime]
            if i < len(history_df) - 1:
                next_date = history_df['date'].iloc[i + 1]
                width = (next_date - date).days
            else:
                width = 1

            rect = Rectangle((mdates.date2num(date), -0.5), width, 3,
                           facecolor=color, alpha=0.7, edgecolor='none')
            ax_main.add_patch(rect)

        ax_main.set_ylim(-0.5, 2.5)
        ax_main.set_yticks([0, 1, 2])
        ax_main.set_yticklabels(['Normal', 'Volatile', 'Crisis'])
        ax_main.set_title('Market Regime Timeline', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)

        # Format x-axis
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)

        # Add crisis period overlays
        self._add_crisis_overlays_matplotlib(ax_main)

        # Probability timeline
        if include_probabilities:
            ax_prob = axes[subplot_idx]
            subplot_idx += 1

            ax_prob.stackplot(history_df['date'],
                            history_df['prob_normal'],
                            history_df['prob_volatile'],
                            history_df['prob_crisis'],
                            labels=['Normal', 'Volatile', 'Crisis'],
                            colors=[self.regime_colors['NORMAL'],
                                  self.regime_colors['VOLATILE'],
                                  self.regime_colors['CRISIS']],
                            alpha=0.7)

            ax_prob.set_ylabel('Probability')
            ax_prob.set_title('Regime Probabilities', fontsize=12, fontweight='bold')
            ax_prob.legend(loc='upper right')
            ax_prob.grid(True, alpha=0.3)

        # Key indicators
        if include_indicators:
            ax_ind = axes[subplot_idx]

            # Try to get VIX data for overlay
            try:
                data = self.classifier.load_market_data(limit=len(history_df))
                if 'vix_data' in data and data['vix_data'] is not None:
                    vix_data = data['vix_data']
                    if len(vix_data) >= len(history_df):
                        vix_subset = vix_data.tail(len(history_df))
                        ax_ind.plot(history_df['date'], vix_subset['close'],
                                  color='purple', linewidth=2, label='VIX')

                        # Add VIX thresholds
                        ax_ind.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Normal Threshold')
                        ax_ind.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Volatile Threshold')
                        ax_ind.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='Crisis Threshold')

            except Exception as e:
                logger.warning(f"Could not overlay VIX data: {e}")

            ax_ind.set_ylabel('VIX Level')
            ax_ind.set_title('Key Market Indicators', fontsize=12, fontweight='bold')
            ax_ind.legend()
            ax_ind.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regime timeline saved to {save_path}")

        return fig

    def _plot_regime_timeline_plotly(self,
                                   history_df: pd.DataFrame,
                                   include_probabilities: bool,
                                   include_indicators: bool,
                                   save_path: Optional[str]) -> go.Figure:
        """Create Plotly interactive regime timeline"""
        n_subplots = 1 + int(include_probabilities) + int(include_indicators)
        subplot_titles = ['Market Regime Timeline']

        if include_probabilities:
            subplot_titles.append('Regime Probabilities')
        if include_indicators:
            subplot_titles.append('Key Market Indicators')

        fig = make_subplots(
            rows=n_subplots, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )

        subplot_idx = 1

        # Main regime timeline
        for regime in ['NORMAL', 'VOLATILE', 'CRISIS']:
            regime_data = history_df[history_df['regime'] == regime]
            if not regime_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['date'],
                        y=[{'NORMAL': 0, 'VOLATILE': 1, 'CRISIS': 2}[regime]] * len(regime_data),
                        mode='markers',
                        marker=dict(
                            color=self.regime_colors[regime],
                            size=8,
                            symbol='square'
                        ),
                        name=regime.title(),
                        hovertemplate=f'<b>{regime.title()}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Confidence: %{customdata:.3f}<extra></extra>',
                        customdata=regime_data['confidence']
                    ),
                    row=subplot_idx, col=1
                )

        fig.update_yaxes(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Normal', 'Volatile', 'Crisis'],
            title_text='Regime',
            row=subplot_idx, col=1
        )

        subplot_idx += 1

        # Probability timeline
        if include_probabilities:
            fig.add_trace(
                go.Scatter(
                    x=history_df['date'],
                    y=history_df['prob_normal'],
                    mode='lines',
                    fill='tonexty',
                    name='Normal Probability',
                    line=dict(color=self.regime_colors['NORMAL']),
                    stackgroup='one'
                ),
                row=subplot_idx, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=history_df['date'],
                    y=history_df['prob_volatile'],
                    mode='lines',
                    fill='tonexty',
                    name='Volatile Probability',
                    line=dict(color=self.regime_colors['VOLATILE']),
                    stackgroup='one'
                ),
                row=subplot_idx, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=history_df['date'],
                    y=history_df['prob_crisis'],
                    mode='lines',
                    fill='tonexty',
                    name='Crisis Probability',
                    line=dict(color=self.regime_colors['CRISIS']),
                    stackgroup='one'
                ),
                row=subplot_idx, col=1
            )

            fig.update_yaxes(title_text='Probability', row=subplot_idx, col=1)
            subplot_idx += 1

        # Key indicators
        if include_indicators:
            try:
                data = self.classifier.load_market_data(limit=len(history_df))
                if 'vix_data' in data and data['vix_data'] is not None:
                    vix_data = data['vix_data']
                    if len(vix_data) >= len(history_df):
                        vix_subset = vix_data.tail(len(history_df))

                        fig.add_trace(
                            go.Scatter(
                                x=history_df['date'],
                                y=vix_subset['close'],
                                mode='lines',
                                name='VIX',
                                line=dict(color='purple', width=2)
                            ),
                            row=subplot_idx, col=1
                        )

                        # Add threshold lines
                        for level, color, name in [(20, 'green', 'Normal'), (30, 'orange', 'Volatile'), (35, 'red', 'Crisis')]:
                            fig.add_hline(
                                y=level,
                                line_dash="dash",
                                line_color=color,
                                annotation_text=f"{name} Threshold",
                                row=subplot_idx, col=1
                            )

            except Exception:
                pass

            fig.update_yaxes(title_text='VIX Level', row=subplot_idx, col=1)

        fig.update_layout(
            title='Market Regime Analysis Dashboard',
            height=300 * n_subplots,
            showlegend=True
        )

        if save_path:
            if save_path.endswith('.html'):
                pyo.plot(fig, filename=save_path, auto_open=False)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive regime timeline saved to {save_path}")

        return fig

    def _add_crisis_overlays_matplotlib(self, ax: plt.Axes):
        """Add known crisis period overlays to matplotlib plot"""
        crisis_periods = [
            ('2008-09-01', '2009-03-31', '2008 Financial Crisis'),
            ('2011-08-01', '2012-06-30', 'European Debt Crisis'),
            ('2020-02-15', '2020-04-30', 'COVID-19 Crash'),
            ('2022-01-01', '2022-06-30', '2022 Inflation Crisis')
        ]

        for start_str, end_str, name in crisis_periods:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)

            # Add shaded region
            ax.axvspan(start_date, end_date, alpha=0.2, color='red', label=name if 'Crisis' not in ax.get_legend_handles_labels()[1] else "")

    def plot_transition_analysis(self,
                               save_path: Optional[str] = None,
                               interactive: bool = False) -> Optional[Any]:
        """
        Create regime transition analysis visualization

        Args:
            save_path: Path to save the plot
            interactive: Use Plotly for interactive plot

        Returns:
            Figure object or None if error
        """
        logger.info("Creating regime transition analysis...")

        try:
            transitions = self.classifier.transition_history

            if not transitions:
                logger.warning("No transition history available")
                return None

            # Prepare transition data
            transition_data = []
            for t in transitions:
                transition_data.append({
                    'date': t.transition_date,
                    'from_regime': t.from_regime.value,
                    'to_regime': t.to_regime.value,
                    'confidence': t.confidence,
                    'duration_days': t.duration_days
                })

            transition_df = pd.DataFrame(transition_data)

            if interactive and PLOTLY_AVAILABLE:
                return self._plot_transition_analysis_plotly(transition_df, save_path)
            elif MATPLOTLIB_AVAILABLE:
                return self._plot_transition_analysis_matplotlib(transition_df, save_path)
            else:
                logger.error("No plotting libraries available")
                return None

        except Exception as e:
            logger.error(f"Failed to create transition analysis: {e}")
            return None

    def _plot_transition_analysis_matplotlib(self,
                                           transition_df: pd.DataFrame,
                                           save_path: Optional[str]) -> plt.Figure:
        """Create matplotlib transition analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Transition matrix heatmap
        transition_matrix = pd.crosstab(transition_df['from_regime'], transition_df['to_regime'])
        sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Regime Transition Matrix')
        axes[0, 0].set_xlabel('To Regime')
        axes[0, 0].set_ylabel('From Regime')

        # Transition frequency over time
        transition_df['month'] = pd.to_datetime(transition_df['date']).dt.to_period('M')
        monthly_transitions = transition_df.groupby('month').size()

        axes[0, 1].plot(monthly_transitions.index.to_timestamp(), monthly_transitions.values)
        axes[0, 1].set_title('Monthly Transition Frequency')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Transitions')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Duration distribution
        axes[1, 0].hist(transition_df['duration_days'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Regime Duration Distribution')
        axes[1, 0].set_xlabel('Duration (Days)')
        axes[1, 0].set_ylabel('Frequency')

        # Confidence distribution by transition type
        transition_df['transition_type'] = transition_df['from_regime'] + ' -> ' + transition_df['to_regime']
        transition_types = transition_df['transition_type'].unique()

        for i, tt in enumerate(transition_types):
            subset = transition_df[transition_df['transition_type'] == tt]
            axes[1, 1].scatter([i] * len(subset), subset['confidence'], alpha=0.6, s=30)

        axes[1, 1].set_xticks(range(len(transition_types)))
        axes[1, 1].set_xticklabels(transition_types, rotation=45, ha='right')
        axes[1, 1].set_title('Confidence by Transition Type')
        axes[1, 1].set_ylabel('Confidence')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Transition analysis saved to {save_path}")

        return fig

    def _plot_transition_analysis_plotly(self,
                                       transition_df: pd.DataFrame,
                                       save_path: Optional[str]) -> go.Figure:
        """Create Plotly interactive transition analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Transition Matrix', 'Monthly Transitions', 'Duration Distribution', 'Confidence Analysis'],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                  [{"type": "histogram"}, {"type": "box"}]]
        )

        # Transition matrix
        regimes = ['NORMAL', 'VOLATILE', 'CRISIS']
        transition_matrix = pd.crosstab(transition_df['from_regime'], transition_df['to_regime'])

        # Ensure all regimes are present
        for regime in regimes:
            if regime not in transition_matrix.index:
                transition_matrix.loc[regime] = 0
            if regime not in transition_matrix.columns:
                transition_matrix[regime] = 0

        transition_matrix = transition_matrix.reindex(regimes).reindex(columns=regimes).fillna(0)

        fig.add_trace(
            go.Heatmap(
                z=transition_matrix.values,
                x=transition_matrix.columns,
                y=transition_matrix.index,
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=1
        )

        # Monthly transitions
        transition_df['month'] = pd.to_datetime(transition_df['date']).dt.to_period('M')
        monthly_transitions = transition_df.groupby('month').size()

        fig.add_trace(
            go.Scatter(
                x=monthly_transitions.index.to_timestamp(),
                y=monthly_transitions.values,
                mode='lines+markers',
                name='Monthly Transitions'
            ),
            row=1, col=2
        )

        # Duration distribution
        fig.add_trace(
            go.Histogram(
                x=transition_df['duration_days'],
                nbinsx=20,
                name='Duration Distribution'
            ),
            row=2, col=1
        )

        # Confidence by transition type
        transition_df['transition_type'] = transition_df['from_regime'] + ' -> ' + transition_df['to_regime']

        for tt in transition_df['transition_type'].unique():
            subset = transition_df[transition_df['transition_type'] == tt]
            fig.add_trace(
                go.Box(
                    y=subset['confidence'],
                    name=tt,
                    boxpoints='all'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title='Regime Transition Analysis',
            height=800,
            showlegend=False
        )

        if save_path:
            if save_path.endswith('.html'):
                pyo.plot(fig, filename=save_path, auto_open=False)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive transition analysis saved to {save_path}")

        return fig

    def create_regime_dashboard(self,
                              save_path: Optional[str] = None,
                              auto_open: bool = False) -> Optional[str]:
        """
        Create comprehensive regime analysis dashboard

        Args:
            save_path: Path to save HTML dashboard
            auto_open: Whether to open dashboard in browser

        Returns:
            Path to saved dashboard or None if error
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly required for dashboard creation")
            return None

        logger.info("Creating comprehensive regime dashboard...")

        try:
            # Get current regime summary
            summary = self.classifier.get_regime_summary()

            # Create dashboard HTML
            html_content = self._generate_dashboard_html(summary)

            # Save dashboard
            if save_path is None:
                save_path = f"regime_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Dashboard saved to {save_path}")

            if auto_open:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(save_path)}")

            return save_path

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return None

    def _generate_dashboard_html(self, summary: Dict[str, Any]) -> str:
        """Generate HTML dashboard content"""
        # Get current prediction for detailed info
        current_prediction = self.classifier.predict_regime()

        # Create timeline plot
        timeline_fig = self.plot_regime_timeline(interactive=True)
        timeline_html = pyo.plot(timeline_fig, output_type='div', include_plotlyjs=False) if timeline_fig else ""

        # Create transition analysis
        transition_fig = self.plot_transition_analysis(interactive=True)
        transition_html = pyo.plot(transition_fig, output_type='div', include_plotlyjs=False) if transition_fig else ""

        # Create current state summary
        current_state_html = f"""
        <div class="current-state">
            <h2>Current Market Regime</h2>
            <div class="regime-indicator {summary['current_regime'].lower()}">
                {summary['current_regime']}
            </div>
            <div class="confidence">
                Confidence: {summary['confidence']:.1%}
            </div>
            <div class="probabilities">
                <div class="prob-bar">
                    <span>Normal: {summary['probabilities']['normal']:.1%}</span>
                    <div class="bar">
                        <div class="fill normal" style="width: {summary['probabilities']['normal']*100}%"></div>
                    </div>
                </div>
                <div class="prob-bar">
                    <span>Volatile: {summary['probabilities']['volatile']:.1%}</span>
                    <div class="bar">
                        <div class="fill volatile" style="width: {summary['probabilities']['volatile']*100}%"></div>
                    </div>
                </div>
                <div class="prob-bar">
                    <span>Crisis: {summary['probabilities']['crisis']:.1%}</span>
                    <div class="bar">
                        <div class="fill crisis" style="width: {summary['probabilities']['crisis']*100}%"></div>
                    </div>
                </div>
            </div>
        </div>
        """

        # Key indicators
        indicators_html = f"""
        <div class="indicators">
            <h3>Key Indicators</h3>
            <div class="indicator-grid">
                <div class="indicator">
                    <span class="label">VIX Level:</span>
                    <span class="value">{summary['indicators']['vix_level']:.1f}</span>
                </div>
                <div class="indicator">
                    <span class="label">Volatility Percentile:</span>
                    <span class="value">{summary['indicators']['volatility_percentile']:.1f}%</span>
                </div>
                <div class="indicator">
                    <span class="label">Correlation Level:</span>
                    <span class="value">{summary['indicators']['correlation_level']:.3f}</span>
                </div>
                <div class="indicator">
                    <span class="label">Fear Index:</span>
                    <span class="value">{summary['indicators']['fear_index']:.3f}</span>
                </div>
            </div>
        </div>
        """

        # Recent activity
        recent_html = f"""
        <div class="recent-activity">
            <h3>Recent Activity</h3>
            <div class="activity-grid">
                <div class="activity-item">
                    <span class="label">Recent Transitions:</span>
                    <span class="value">{summary['recent_transitions']}</span>
                </div>
                <div class="activity-item">
                    <span class="label">Last Update:</span>
                    <span class="value">{summary['last_update'] or 'Never'}</span>
                </div>
            </div>
            <div class="regime-distribution">
                <h4>Recent Regime Distribution</h4>
                <div class="dist-bars">
                    <div class="dist-bar">
                        <span>Normal: {summary['recent_distribution']['NORMAL']}</span>
                        <div class="bar">
                            <div class="fill normal" style="width: {summary['recent_distribution']['NORMAL']/30*100}%"></div>
                        </div>
                    </div>
                    <div class="dist-bar">
                        <span>Volatile: {summary['recent_distribution']['VOLATILE']}</span>
                        <div class="bar">
                            <div class="fill volatile" style="width: {summary['recent_distribution']['VOLATILE']/30*100}%"></div>
                        </div>
                    </div>
                    <div class="dist-bar">
                        <span>Crisis: {summary['recent_distribution']['CRISIS']}</span>
                        <div class="bar">
                            <div class="fill crisis" style="width: {summary['recent_distribution']['CRISIS']/30*100}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

        # Complete HTML
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Regime Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }}
                .summary-section {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .current-state {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .regime-indicator {{
                    font-size: 24px;
                    font-weight: bold;
                    text-align: center;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .regime-indicator.normal {{ background-color: #2E8B57; color: white; }}
                .regime-indicator.volatile {{ background-color: #FF8C00; color: white; }}
                .regime-indicator.crisis {{ background-color: #DC143C; color: white; }}
                .confidence {{
                    text-align: center;
                    font-size: 18px;
                    margin: 10px 0;
                }}
                .probabilities {{
                    margin-top: 15px;
                }}
                .prob-bar, .dist-bar {{
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                }}
                .prob-bar span, .dist-bar span {{
                    width: 100px;
                    font-size: 14px;
                }}
                .bar {{
                    flex: 1;
                    height: 20px;
                    background-color: #eee;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-left: 10px;
                }}
                .fill {{
                    height: 100%;
                    border-radius: 10px;
                }}
                .fill.normal {{ background-color: #2E8B57; }}
                .fill.volatile {{ background-color: #FF8C00; }}
                .fill.crisis {{ background-color: #DC143C; }}
                .indicators, .recent-activity {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .indicator-grid, .activity-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    margin-top: 10px;
                }}
                .indicator, .activity-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                }}
                .label {{
                    font-weight: 600;
                    color: #666;
                }}
                .value {{
                    font-weight: bold;
                    color: #333;
                }}
                .charts-section {{
                    margin-top: 30px;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    margin-top: 20px;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>Market Regime Analysis Dashboard</h1>

                <div class="summary-section">
                    {current_state_html}
                    <div>
                        {indicators_html}
                        {recent_html}
                    </div>
                </div>

                <div class="charts-section">
                    <div class="chart-container">
                        <h2>Regime Timeline</h2>
                        {timeline_html}
                    </div>

                    <div class="chart-container">
                        <h2>Transition Analysis</h2>
                        {transition_html}
                    </div>
                </div>

                <div class="timestamp">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """

        return html_template

    def export_regime_report(self,
                           output_path: str,
                           format: str = 'json',
                           include_history: bool = True,
                           include_validation: bool = True) -> bool:
        """
        Export comprehensive regime analysis report

        Args:
            output_path: Path for output file
            format: Export format ('json', 'csv', 'excel')
            include_history: Include historical regime data
            include_validation: Include validation results

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Exporting regime report to {output_path} (format: {format})")

        try:
            # Gather all data
            report_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'classifier_config': self.classifier.config,
                    'ensemble_weights': self.classifier.ensemble_weights
                },
                'current_state': self.classifier.get_regime_summary(),
                'transitions': []
            }

            # Add transition history
            for t in self.classifier.transition_history:
                report_data['transitions'].append({
                    'date': t.transition_date.isoformat(),
                    'from_regime': t.from_regime.value,
                    'to_regime': t.to_regime.value,
                    'confidence': t.confidence,
                    'duration_days': t.duration_days
                })

            # Add historical regimes if requested
            if include_history:
                history_df = self.classifier.get_historical_regimes()
                if not history_df.empty:
                    report_data['historical_regimes'] = history_df.to_dict('records')

            # Add validation results if requested
            if include_validation:
                validation_results = self.classifier.validate_crisis_periods()
                report_data['validation'] = validation_results

            # Export based on format
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

            elif format.lower() == 'csv':
                # Export main data as CSV
                if 'historical_regimes' in report_data:
                    df = pd.DataFrame(report_data['historical_regimes'])
                    df.to_csv(output_path, index=False)
                else:
                    # Export transitions if no history
                    df = pd.DataFrame(report_data['transitions'])
                    df.to_csv(output_path, index=False)

            elif format.lower() == 'excel':
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_df = pd.DataFrame([report_data['current_state']])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                    # Transitions sheet
                    if report_data['transitions']:
                        transitions_df = pd.DataFrame(report_data['transitions'])
                        transitions_df.to_excel(writer, sheet_name='Transitions', index=False)

                    # Historical data sheet
                    if 'historical_regimes' in report_data:
                        history_df = pd.DataFrame(report_data['historical_regimes'])
                        history_df.to_excel(writer, sheet_name='Historical_Regimes', index=False)

                    # Validation sheet
                    if 'validation' in report_data and 'crisis_periods_detected' in report_data['validation']:
                        validation_df = pd.DataFrame(report_data['validation']['crisis_periods_detected'])
                        validation_df.to_excel(writer, sheet_name='Validation', index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Regime report exported successfully to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export regime report: {e}")
            return False


# Factory function for easy creation
def create_regime_visualization(classifier: Optional['MarketRegimeClassifier'] = None) -> RegimeVisualization:
    """
    Factory function to create regime visualization system

    Args:
        classifier: Optional MarketRegimeClassifier instance

    Returns:
        RegimeVisualization instance
    """
    return RegimeVisualization(classifier)


if __name__ == "__main__":
    # Example usage and demonstration
    print("Market Regime Visualization System")
    print("=" * 40)

    if not REGIME_CLASSIFIER_AVAILABLE:
        print("Error: Market regime classifier not available")
        exit(1)

    # Create visualization system
    from market_regime_classifier import MarketRegimeClassifier

    classifier = MarketRegimeClassifier()
    viz = RegimeVisualization(classifier)

    # Try to load or fit models
    if not classifier.load_models():
        print("Fitting new models...")
        classifier.fit_models()

    print("\nGenerating visualizations...")

    # Create regime timeline
    print("1. Creating regime timeline...")
    timeline_fig = viz.plot_regime_timeline(
        include_probabilities=True,
        include_indicators=True,
        save_path="regime_timeline.png",
        interactive=False
    )

    if timeline_fig:
        print("   Timeline saved to regime_timeline.png")

    # Create transition analysis
    print("2. Creating transition analysis...")
    transition_fig = viz.plot_transition_analysis(
        save_path="regime_transitions.png",
        interactive=False
    )

    if transition_fig:
        print("   Transition analysis saved to regime_transitions.png")

    # Create dashboard
    print("3. Creating interactive dashboard...")
    dashboard_path = viz.create_regime_dashboard(
        save_path="regime_dashboard.html",
        auto_open=False
    )

    if dashboard_path:
        print(f"   Dashboard saved to {dashboard_path}")

    # Export report
    print("4. Exporting regime report...")
    report_success = viz.export_regime_report(
        output_path="regime_report.json",
        format='json',
        include_history=True,
        include_validation=True
    )

    if report_success:
        print("   Report exported to regime_report.json")

    print("\nVisualization system demonstration complete!")
    print("Files generated:")
    print("- regime_timeline.png (static timeline plot)")
    print("- regime_transitions.png (transition analysis)")
    print("- regime_dashboard.html (interactive dashboard)")
    print("- regime_report.json (comprehensive report)")