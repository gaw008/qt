"""
Professional PDF Report Generator for Backtesting Analysis
专业PDF报告生成器用于回测分析

This module provides high-quality PDF generation capabilities for institutional reports:
- Professional layout and styling
- Charts and visualization embedding
- Executive summary formatting
- Multi-page report structure
- Print-optimized design
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import tempfile
import base64
from io import BytesIO

# PDF generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus import Image as RLImage, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.widgets.markers import makeMarker
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Configure logging
logger = logging.getLogger(__name__)

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')


class ProfessionalPDFGenerator:
    """
    Professional PDF report generator for backtesting analysis.

    Features:
    - Institutional-quality layout and typography
    - Embedded charts and visualizations
    - Executive summary and detailed analysis
    - Performance metrics tables
    - Risk analysis sections
    - Statistical significance reporting
    """

    def __init__(self, output_dir: str = "reports/pdf"):
        """Initialize PDF generator."""
        if not HAS_REPORTLAB:
            raise ImportError("ReportLab is required for PDF generation")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Page configuration
        self.page_size = A4
        self.margins = {
            'top': 2.5*cm,
            'bottom': 2.5*cm,
            'left': 2.5*cm,
            'right': 2.5*cm
        }

        logger.info("Professional PDF generator initialized")

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for professional appearance."""

        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold'
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#34495e'),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))

        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=16,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold'
        ))

        # Executive summary style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            backColor=colors.HexColor('#f8f9fa'),
            borderWidth=1,
            borderColor=colors.HexColor('#e9ecef'),
            borderPadding=10
        ))

        # Metric value style
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=18,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        ))

        # Metric label style
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7f8c8d')
        ))

        # Recommendation style
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=6
        ))

    def generate_comprehensive_report(self,
                                    strategy_name: str,
                                    report_data: Dict[str, Any],
                                    filename: Optional[str] = None) -> str:
        """
        Generate comprehensive PDF backtesting report.

        Args:
            strategy_name: Name of the trading strategy
            report_data: Processed backtesting data
            filename: Optional custom filename

        Returns:
            Path to generated PDF file
        """

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_comprehensive_report_{timestamp}.pdf"

        pdf_path = self.output_dir / filename

        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=self.page_size,
            topMargin=self.margins['top'],
            bottomMargin=self.margins['bottom'],
            leftMargin=self.margins['left'],
            rightMargin=self.margins['right'],
            title=f"{strategy_name} - Backtesting Report",
            author="Quantitative Trading System",
            subject="Three-Phase Backtesting Validation Report"
        )

        # Build PDF content
        story = []

        # Title page
        story.extend(self._create_title_page(strategy_name, report_data))
        story.append(PageBreak())

        # Executive summary
        story.extend(self._create_executive_summary(report_data))
        story.append(PageBreak())

        # Performance overview
        story.extend(self._create_performance_overview(report_data))

        # Phase-by-phase analysis
        story.extend(self._create_phase_analysis(report_data))

        # Crisis period analysis
        if report_data.get('crisis_results'):
            story.extend(self._create_crisis_analysis(report_data))

        # Risk analysis
        story.extend(self._create_risk_analysis(report_data))

        # Statistical significance
        if report_data.get('statistical_tests'):
            story.extend(self._create_statistical_analysis(report_data))

        # Charts and visualizations
        story.extend(self._create_visualizations(report_data))

        # Recommendations
        story.extend(self._create_recommendations(report_data))

        # Appendices
        story.extend(self._create_appendices(report_data))

        # Build PDF
        doc.build(story)

        logger.info(f"PDF report generated: {pdf_path}")
        return str(pdf_path)

    def _create_title_page(self, strategy_name: str, report_data: Dict[str, Any]) -> List[Any]:
        """Create professional title page."""

        story = []

        # Main title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(f"{strategy_name}", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))

        # Subtitle
        story.append(Paragraph(
            "Comprehensive Three-Phase Backtesting Validation Report",
            self.styles['Heading2']
        ))
        story.append(Spacer(1, 1*inch))

        # Key metrics summary table
        overall_performance = report_data.get('overall_performance', {})

        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{overall_performance.get('total_return', 0):.2%}"],
            ['Annualized Return', f"{overall_performance.get('annualized_return', 0):.2%}"],
            ['Sharpe Ratio', f"{overall_performance.get('sharpe_ratio', 0):.2f}"],
            ['Maximum Drawdown', f"{overall_performance.get('max_drawdown', 0):.2%}"],
            ['Win Rate', f"{overall_performance.get('win_rate', 0):.2%}"]
        ]

        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 11)
        ]))

        story.append(metrics_table)
        story.append(Spacer(1, 1*inch))

        # Report metadata
        generation_time = report_data.get('generation_time', datetime.now().isoformat())
        story.append(Paragraph(f"Generated: {generation_time}", self.styles['Normal']))
        story.append(Paragraph("Quantitative Trading System v2.1", self.styles['Normal']))

        return story

    def _create_executive_summary(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create executive summary section."""

        story = []

        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

        executive_summary = report_data.get('executive_summary', {})

        # Key findings
        key_findings = executive_summary.get('key_findings', 'No key findings available.')
        story.append(Paragraph(f"<b>Key Findings:</b> {key_findings}", self.styles['ExecutiveSummary']))

        # Viability assessment
        viability = executive_summary.get('viability_assessment', 'Assessment not available.')
        story.append(Paragraph(f"<b>Strategy Viability:</b> {viability}", self.styles['ExecutiveSummary']))

        # Risk assessment
        risk_assessment = executive_summary.get('risk_assessment', 'Risk assessment not available.')
        story.append(Paragraph(f"<b>Risk Profile:</b> {risk_assessment}", self.styles['ExecutiveSummary']))

        story.append(Spacer(1, 0.5*inch))

        return story

    def _create_performance_overview(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create performance overview section."""

        story = []

        story.append(Paragraph("Performance Overview", self.styles['SectionHeader']))

        overall_performance = report_data.get('overall_performance', {})

        # Create performance metrics table
        perf_data = [
            ['Performance Metric', 'Value', 'Assessment'],
            ['Total Return', f"{overall_performance.get('total_return', 0):.2%}",
             self._assess_metric('return', overall_performance.get('total_return', 0))],
            ['Annualized Return', f"{overall_performance.get('annualized_return', 0):.2%}",
             self._assess_metric('return', overall_performance.get('annualized_return', 0))],
            ['Volatility', f"{overall_performance.get('volatility', 0):.2%}",
             self._assess_metric('volatility', overall_performance.get('volatility', 0))],
            ['Sharpe Ratio', f"{overall_performance.get('sharpe_ratio', 0):.2f}",
             self._assess_metric('sharpe', overall_performance.get('sharpe_ratio', 0))],
            ['Sortino Ratio', f"{overall_performance.get('sortino_ratio', 0):.2f}",
             self._assess_metric('sortino', overall_performance.get('sortino_ratio', 0))],
            ['Calmar Ratio', f"{overall_performance.get('calmar_ratio', 0):.2f}",
             self._assess_metric('calmar', overall_performance.get('calmar_ratio', 0))],
            ['Maximum Drawdown', f"{overall_performance.get('max_drawdown', 0):.2%}",
             self._assess_metric('drawdown', overall_performance.get('max_drawdown', 0))],
            ['Win Rate', f"{overall_performance.get('win_rate', 0):.2%}",
             self._assess_metric('win_rate', overall_performance.get('win_rate', 0))]
        ]

        perf_table = Table(perf_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))

        story.append(perf_table)
        story.append(Spacer(1, 0.3*inch))

        return story

    def _create_phase_analysis(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create phase-by-phase analysis section."""

        story = []

        story.append(Paragraph("Phase-by-Phase Analysis", self.styles['SectionHeader']))

        phase_results = report_data.get('phase_results', {})

        for phase_name, phase_data in phase_results.items():
            story.append(Paragraph(phase_name, self.styles['SubsectionHeader']))

            # Phase performance table
            phase_table_data = [
                ['Metric', 'Value'],
                ['Total Return', f"{phase_data.get('total_return', 0):.2%}"],
                ['Annualized Return', f"{phase_data.get('annualized_return', 0):.2%}"],
                ['Sharpe Ratio', f"{phase_data.get('sharpe_ratio', 0):.2f}"],
                ['Maximum Drawdown', f"{phase_data.get('max_drawdown', 0):.2%}"],
                ['Win Rate', f"{phase_data.get('win_rate', 0):.2%}"],
                ['Total Trades', f"{phase_data.get('total_trades', 0):,}"]
            ]

            phase_table = Table(phase_table_data, colWidths=[2*inch, 1.5*inch])
            phase_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))

            story.append(phase_table)
            story.append(Spacer(1, 0.2*inch))

        return story

    def _create_crisis_analysis(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create crisis period analysis section."""

        story = []

        story.append(Paragraph("Crisis Period Performance Analysis", self.styles['SectionHeader']))

        crisis_results = report_data.get('crisis_results', {})

        # Create crisis performance table
        crisis_data = [['Crisis Period', 'Return', 'Max Drawdown', 'Recovery Days']]

        for crisis_name, crisis_metrics in crisis_results.items():
            crisis_data.append([
                crisis_name,
                f"{crisis_metrics.get('total_return', 0):.2%}",
                f"{crisis_metrics.get('max_drawdown', 0):.2%}",
                f"{crisis_metrics.get('max_drawdown_duration', 0):,}"
            ])

        crisis_table = Table(crisis_data, colWidths=[2.5*inch, 1*inch, 1.2*inch, 1*inch])
        crisis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))

        story.append(crisis_table)
        story.append(Spacer(1, 0.3*inch))

        # Crisis analysis summary
        story.append(Paragraph(
            "Crisis period analysis demonstrates the strategy's resilience during market stress. "
            "Performance during these periods is critical for institutional deployment.",
            self.styles['Normal']
        ))

        return story

    def _create_risk_analysis(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create comprehensive risk analysis section."""

        story = []

        story.append(Paragraph("Risk Analysis", self.styles['SectionHeader']))

        overall_performance = report_data.get('overall_performance', {})

        # Risk metrics table
        risk_data = [
            ['Risk Metric', 'Value', 'Risk Level'],
            ['Value at Risk (95%)', f"{overall_performance.get('var_95', 0):.2%}",
             self._assess_risk_level('var', overall_performance.get('var_95', 0))],
            ['Expected Shortfall', f"{overall_performance.get('expected_shortfall', 0):.2%}",
             self._assess_risk_level('es', overall_performance.get('expected_shortfall', 0))],
            ['Maximum Drawdown', f"{overall_performance.get('max_drawdown', 0):.2%}",
             self._assess_risk_level('drawdown', overall_performance.get('max_drawdown', 0))],
            ['Volatility', f"{overall_performance.get('volatility', 0):.2%}",
             self._assess_risk_level('volatility', overall_performance.get('volatility', 0))]
        ]

        risk_table = Table(risk_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))

        story.append(risk_table)
        story.append(Spacer(1, 0.3*inch))

        return story

    def _create_statistical_analysis(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create statistical significance analysis section."""

        story = []

        story.append(Paragraph("Statistical Significance Analysis", self.styles['SectionHeader']))

        statistical_tests = report_data.get('statistical_tests', [])

        if not statistical_tests:
            story.append(Paragraph("No statistical tests available.", self.styles['Normal']))
            return story

        # Statistical tests table
        stats_data = [['Test Name', 'Statistic', 'P-Value', 'Significant', 'Interpretation']]

        for test in statistical_tests:
            stats_data.append([
                test.get('test_name', ''),
                f"{test.get('statistic', 0):.4f}",
                f"{test.get('p_value', 0):.4f}",
                'Yes' if test.get('is_significant', False) else 'No',
                test.get('interpretation', '')[:50] + '...' if len(test.get('interpretation', '')) > 50 else test.get('interpretation', '')
            ])

        stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 0.8*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))

        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))

        return story

    def _create_visualizations(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create charts and visualizations section."""

        story = []

        story.append(Paragraph("Performance Visualizations", self.styles['SectionHeader']))

        # Generate and embed charts
        try:
            # Equity curve chart
            equity_chart_path = self._generate_equity_curve_chart(report_data)
            if equity_chart_path:
                story.append(Paragraph("Equity Curve Performance", self.styles['SubsectionHeader']))
                story.append(RLImage(equity_chart_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 0.2*inch))

            # Returns distribution chart
            returns_chart_path = self._generate_returns_distribution_chart(report_data)
            if returns_chart_path:
                story.append(Paragraph("Returns Distribution", self.styles['SubsectionHeader']))
                story.append(RLImage(returns_chart_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 0.2*inch))

        except Exception as e:
            logger.warning(f"Failed to generate charts: {e}")
            story.append(Paragraph("Charts could not be generated.", self.styles['Normal']))

        return story

    def _create_recommendations(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create recommendations section."""

        story = []

        story.append(Paragraph("Strategic Recommendations", self.styles['SectionHeader']))

        recommendations = report_data.get('recommendations', [])

        if not recommendations:
            story.append(Paragraph("No specific recommendations available.", self.styles['Normal']))
            return story

        for i, recommendation in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {recommendation}", self.styles['Recommendation']))

        return story

    def _create_appendices(self, report_data: Dict[str, Any]) -> List[Any]:
        """Create appendices section."""

        story = []

        story.append(PageBreak())
        story.append(Paragraph("Appendices", self.styles['SectionHeader']))

        # Appendix A: Methodology
        story.append(Paragraph("Appendix A: Methodology", self.styles['SubsectionHeader']))
        methodology_text = """
        This report employs a three-phase backtesting methodology to validate strategy performance across different market regimes:

        Phase 1 (2006-2016): Pre-crisis to recovery period, including the 2008 financial crisis
        Phase 2 (2017-2020): Modern bull market with low volatility environment
        Phase 3 (2021-2025): Post-pandemic era with inflation and monetary policy changes

        Statistical significance testing includes normality tests, autocorrelation analysis, and heteroscedasticity testing.
        Risk metrics are calculated using industry-standard methodologies including Value at Risk and Expected Shortfall.
        """
        story.append(Paragraph(methodology_text, self.styles['Normal']))

        # Appendix B: Disclaimers
        story.append(Paragraph("Appendix B: Important Disclaimers", self.styles['SubsectionHeader']))
        disclaimer_text = """
        This backtesting report is for informational purposes only and does not constitute investment advice.
        Past performance does not guarantee future results. All investment strategies carry risk of loss.
        The results presented are based on historical data and may not reflect future market conditions.
        This analysis assumes perfect execution and does not account for all real-world trading constraints.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))

        return story

    def _generate_equity_curve_chart(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Generate equity curve chart and return file path."""

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get phase results for plotting
            phase_results = report_data.get('phase_results', {})

            for phase_name, phase_data in phase_results.items():
                equity_curve = phase_data.get('equity_curve')
                if equity_curve and isinstance(equity_curve, list):
                    # Create date range for x-axis
                    start_date = pd.to_datetime(phase_data.get('start_date', '2006-01-01'))
                    dates = pd.date_range(start_date, periods=len(equity_curve), freq='D')

                    ax.plot(dates, equity_curve, label=phase_name, linewidth=2)

            ax.set_title('Strategy Equity Curve Across Phases', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format dates on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Save to temporary file
            chart_path = tempfile.mktemp(suffix='.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return chart_path

        except Exception as e:
            logger.error(f"Failed to generate equity curve chart: {e}")
            return None

    def _generate_returns_distribution_chart(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Generate returns distribution chart and return file path."""

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Collect all returns data
            all_returns = []
            overall_performance = report_data.get('overall_performance', {})

            if 'returns_series' in overall_performance:
                returns_series = overall_performance['returns_series']
                if isinstance(returns_series, list):
                    all_returns = returns_series

            if all_returns:
                # Create histogram
                ax.hist(all_returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

                # Add normal distribution overlay
                mu, sigma = np.mean(all_returns), np.std(all_returns)
                x = np.linspace(min(all_returns), max(all_returns), 100)
                y = ((1/(sigma * np.sqrt(2 * np.pi))) *
                     np.exp(-0.5 * ((x - mu) / sigma) ** 2))

                # Scale normal distribution to match histogram
                y_scaled = y * len(all_returns) * (max(all_returns) - min(all_returns)) / 50

                ax.plot(x, y_scaled, 'r-', linewidth=2, label='Normal Distribution')

                ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Daily Return')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add statistics text
                stats_text = f'Mean: {mu:.4f}\nStd: {sigma:.4f}\nSkewness: {pd.Series(all_returns).skew():.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save to temporary file
            chart_path = tempfile.mktemp(suffix='.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return chart_path

        except Exception as e:
            logger.error(f"Failed to generate returns distribution chart: {e}")
            return None

    def _assess_metric(self, metric_type: str, value: float) -> str:
        """Assess performance metric and return qualitative assessment."""

        if metric_type == 'return':
            if value > 0.15:
                return 'Excellent'
            elif value > 0.08:
                return 'Good'
            elif value > 0.04:
                return 'Moderate'
            else:
                return 'Poor'

        elif metric_type == 'sharpe':
            if value > 1.5:
                return 'Excellent'
            elif value > 1.0:
                return 'Good'
            elif value > 0.5:
                return 'Moderate'
            else:
                return 'Poor'

        elif metric_type == 'sortino':
            if value > 1.5:
                return 'Excellent'
            elif value > 1.0:
                return 'Good'
            elif value > 0.5:
                return 'Moderate'
            else:
                return 'Poor'

        elif metric_type == 'calmar':
            if value > 2.0:
                return 'Excellent'
            elif value > 1.0:
                return 'Good'
            elif value > 0.5:
                return 'Moderate'
            else:
                return 'Poor'

        elif metric_type == 'drawdown':
            if value > -0.05:
                return 'Excellent'
            elif value > -0.10:
                return 'Good'
            elif value > -0.20:
                return 'Moderate'
            else:
                return 'Poor'

        elif metric_type == 'volatility':
            if value < 0.10:
                return 'Low Risk'
            elif value < 0.20:
                return 'Moderate Risk'
            else:
                return 'High Risk'

        elif metric_type == 'win_rate':
            if value > 0.60:
                return 'Excellent'
            elif value > 0.50:
                return 'Good'
            elif value > 0.40:
                return 'Moderate'
            else:
                return 'Poor'

        return 'Unknown'

    def _assess_risk_level(self, risk_type: str, value: float) -> str:
        """Assess risk level and return qualitative assessment."""

        if risk_type == 'var':
            if value > -0.02:
                return 'Low Risk'
            elif value > -0.04:
                return 'Moderate Risk'
            else:
                return 'High Risk'

        elif risk_type == 'es':
            if value > -0.03:
                return 'Low Risk'
            elif value > -0.06:
                return 'Moderate Risk'
            else:
                return 'High Risk'

        elif risk_type == 'drawdown':
            if value > -0.05:
                return 'Low Risk'
            elif value > -0.15:
                return 'Moderate Risk'
            else:
                return 'High Risk'

        elif risk_type == 'volatility':
            if value < 0.15:
                return 'Low Risk'
            elif value < 0.25:
                return 'Moderate Risk'
            else:
                return 'High Risk'

        return 'Unknown'


# Utility functions

def generate_professional_pdf_report(strategy_name: str,
                                   report_data: Dict[str, Any],
                                   output_dir: str = "reports/pdf") -> str:
    """
    Main entry point for generating professional PDF reports.

    Args:
        strategy_name: Name of the trading strategy
        report_data: Processed backtesting data
        output_dir: Output directory for PDF files

    Returns:
        Path to generated PDF file
    """

    if not HAS_REPORTLAB:
        raise ImportError("ReportLab is required for PDF generation")

    generator = ProfessionalPDFGenerator(output_dir)
    return generator.generate_comprehensive_report(strategy_name, report_data)


if __name__ == "__main__":
    # Test PDF generation with sample data
    if HAS_REPORTLAB:
        # Create sample data structure
        sample_data = {
            'strategy_name': 'Sample Multi-Factor Strategy',
            'generation_time': datetime.now().isoformat(),
            'overall_performance': {
                'total_return': 0.234,
                'annualized_return': 0.145,
                'sharpe_ratio': 1.23,
                'max_drawdown': -0.087,
                'win_rate': 0.56,
                'volatility': 0.18,
                'var_95': -0.025,
                'expected_shortfall': -0.035,
                'sortino_ratio': 1.45,
                'calmar_ratio': 1.67
            },
            'phase_results': {
                'Phase 1 (2006-2016)': {
                    'total_return': 0.156,
                    'annualized_return': 0.142,
                    'sharpe_ratio': 1.12,
                    'max_drawdown': -0.123,
                    'win_rate': 0.54,
                    'total_trades': 1250,
                    'start_date': '2006-01-01',
                    'equity_curve': [100000 * (1.001 ** i) for i in range(2500)]
                }
            },
            'executive_summary': {
                'key_findings': 'Strategy demonstrates strong risk-adjusted performance across multiple market regimes.',
                'viability_assessment': 'Highly viable for institutional deployment.',
                'risk_assessment': 'Moderate risk profile suitable for balanced portfolios.'
            },
            'recommendations': [
                'Continue monitoring performance with quarterly reviews',
                'Consider gradual scaling based on live trading validation',
                'Implement additional risk controls for crisis periods'
            ]
        }

        pdf_path = generate_professional_pdf_report("Sample Strategy", sample_data)
        print(f"Sample PDF report generated: {pdf_path}")
    else:
        print("ReportLab not available - skipping PDF generation test")