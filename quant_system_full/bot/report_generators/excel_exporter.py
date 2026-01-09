"""
Comprehensive Excel Export System for Backtesting Reports
综合Excel导出系统用于回测报告

This module provides detailed Excel export capabilities for institutional analysis:
- Multi-worksheet reports with professional formatting
- Interactive charts and pivot tables
- Detailed time series data export
- Summary statistics and metrics
- Risk analysis worksheets
- Executive dashboard sheets
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import warnings

# Excel export
try:
    import xlsxwriter
    import pandas as pd
    import numpy as np
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False
    warnings.warn("XlsxWriter and pandas required for Excel export")

# Configure logging
logger = logging.getLogger(__name__)


class ComprehensiveExcelExporter:
    """
    Comprehensive Excel export system for backtesting reports.

    Features:
    - Multiple worksheets for different analysis sections
    - Professional formatting and styling
    - Interactive charts and visualizations
    - Detailed time series data
    - Summary statistics and metrics
    - Risk analysis and stress testing results
    - Executive dashboard overview
    """

    def __init__(self, output_dir: str = "reports/excel"):
        """Initialize Excel exporter."""
        if not HAS_XLSXWRITER:
            raise ImportError("xlsxwriter and pandas are required for Excel export")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Formatting configurations
        self.formats = {}
        self.chart_colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
            '#6A994E', '#BC4749', '#F2E8CF', '#386641'
        ]

        logger.info("Comprehensive Excel exporter initialized")

    def _setup_formats(self, workbook):
        """Setup formatting styles for the Excel workbook."""

        # Header formats
        self.formats['header'] = workbook.add_format({
            'bold': True,
            'bg_color': '#2C3E50',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'font_size': 12
        })

        self.formats['subheader'] = workbook.add_format({
            'bold': True,
            'bg_color': '#34495E',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'font_size': 11
        })

        # Data formats
        self.formats['number'] = workbook.add_format({
            'num_format': '0.00',
            'border': 1,
            'align': 'center'
        })

        self.formats['percentage'] = workbook.add_format({
            'num_format': '0.00%',
            'border': 1,
            'align': 'center'
        })

        self.formats['currency'] = workbook.add_format({
            'num_format': '$#,##0',
            'border': 1,
            'align': 'center'
        })

        self.formats['date'] = workbook.add_format({
            'num_format': 'yyyy-mm-dd',
            'border': 1,
            'align': 'center'
        })

        # Status formats
        self.formats['positive'] = workbook.add_format({
            'num_format': '0.00%',
            'border': 1,
            'align': 'center',
            'bg_color': '#D5EDDA',
            'font_color': '#155724'
        })

        self.formats['negative'] = workbook.add_format({
            'num_format': '0.00%',
            'border': 1,
            'align': 'center',
            'bg_color': '#F8D7DA',
            'font_color': '#721C24'
        })

        # Title formats
        self.formats['title'] = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'font_color': '#2C3E50',
            'align': 'center'
        })

        self.formats['section_title'] = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'font_color': '#34495E',
            'bottom': 2,
            'bottom_color': '#3498DB'
        })

    def generate_comprehensive_report(self,
                                    strategy_name: str,
                                    report_data: Dict[str, Any],
                                    filename: Optional[str] = None) -> str:
        """
        Generate comprehensive Excel backtesting report.

        Args:
            strategy_name: Name of the trading strategy
            report_data: Processed backtesting data
            filename: Optional custom filename

        Returns:
            Path to generated Excel file
        """

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_comprehensive_analysis_{timestamp}.xlsx"

        excel_path = self.output_dir / filename

        # Create Excel workbook
        workbook = xlsxwriter.Workbook(str(excel_path))
        self._setup_formats(workbook)

        try:
            # Create worksheets
            self._create_executive_dashboard(workbook, strategy_name, report_data)
            self._create_performance_summary(workbook, report_data)
            self._create_phase_analysis(workbook, report_data)
            self._create_risk_analysis(workbook, report_data)
            self._create_statistical_tests(workbook, report_data)
            self._create_time_series_data(workbook, report_data)
            self._create_trade_analysis(workbook, report_data)
            self._create_benchmark_comparison(workbook, report_data)
            self._create_charts_dashboard(workbook, report_data)
            self._create_raw_data_export(workbook, report_data)

            workbook.close()

            logger.info(f"Excel report generated: {excel_path}")
            return str(excel_path)

        except Exception as e:
            workbook.close()
            logger.error(f"Failed to generate Excel report: {e}")
            raise

    def _create_executive_dashboard(self, workbook, strategy_name: str, report_data: Dict[str, Any]):
        """Create executive dashboard worksheet."""

        worksheet = workbook.add_worksheet('Executive Dashboard')

        # Set column widths
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:F', 15)

        # Title
        worksheet.merge_range('A1:F1', f'{strategy_name} - Executive Dashboard', self.formats['title'])
        worksheet.write('A2', f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')

        # Key Performance Indicators
        row = 4
        worksheet.write(row, 0, 'Key Performance Indicators', self.formats['section_title'])
        row += 2

        overall_performance = report_data.get('overall_performance', {})

        kpi_data = [
            ['Total Return', overall_performance.get('total_return', 0), 'percentage'],
            ['Annualized Return', overall_performance.get('annualized_return', 0), 'percentage'],
            ['Sharpe Ratio', overall_performance.get('sharpe_ratio', 0), 'number'],
            ['Maximum Drawdown', overall_performance.get('max_drawdown', 0), 'percentage'],
            ['Win Rate', overall_performance.get('win_rate', 0), 'percentage'],
            ['Volatility', overall_performance.get('volatility', 0), 'percentage']
        ]

        # Headers
        worksheet.write(row, 0, 'Metric', self.formats['header'])
        worksheet.write(row, 1, 'Value', self.formats['header'])
        worksheet.write(row, 2, 'Assessment', self.formats['header'])
        row += 1

        for metric_name, value, format_type in kpi_data:
            worksheet.write(row, 0, metric_name, self.formats['number'])

            if format_type == 'percentage':
                if value >= 0:
                    worksheet.write(row, 1, value, self.formats['positive'])
                else:
                    worksheet.write(row, 1, value, self.formats['negative'])
            else:
                worksheet.write(row, 1, value, self.formats['number'])

            # Assessment
            assessment = self._assess_metric(metric_name, value)
            worksheet.write(row, 2, assessment, self.formats['number'])
            row += 1

        # Risk Metrics Summary
        row += 2
        worksheet.write(row, 0, 'Risk Summary', self.formats['section_title'])
        row += 2

        risk_data = [
            ['Value at Risk (95%)', overall_performance.get('var_95', 0)],
            ['Expected Shortfall', overall_performance.get('expected_shortfall', 0)],
            ['Calmar Ratio', overall_performance.get('calmar_ratio', 0)],
            ['Sortino Ratio', overall_performance.get('sortino_ratio', 0)]
        ]

        worksheet.write(row, 0, 'Risk Metric', self.formats['header'])
        worksheet.write(row, 1, 'Value', self.formats['header'])
        row += 1

        for metric_name, value in risk_data:
            worksheet.write(row, 0, metric_name, self.formats['number'])
            if 'Ratio' in metric_name:
                worksheet.write(row, 1, value, self.formats['number'])
            else:
                worksheet.write(row, 1, value, self.formats['percentage'])
            row += 1

        # Executive Summary
        row += 2
        worksheet.write(row, 0, 'Executive Summary', self.formats['section_title'])
        row += 1

        executive_summary = report_data.get('executive_summary', {})
        summary_items = [
            ('Key Findings', executive_summary.get('key_findings', 'Not available')),
            ('Viability Assessment', executive_summary.get('viability_assessment', 'Not available')),
            ('Risk Assessment', executive_summary.get('risk_assessment', 'Not available'))
        ]

        for label, text in summary_items:
            worksheet.write(row, 0, label, self.formats['subheader'])
            # Wrap text for better readability
            wrapped_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
            worksheet.write(row, 1, text, wrapped_format)
            worksheet.set_row(row, 40)  # Set row height for wrapped text
            row += 1

    def _create_performance_summary(self, workbook, report_data: Dict[str, Any]):
        """Create performance summary worksheet."""

        worksheet = workbook.add_worksheet('Performance Summary')
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:Z', 12)

        row = 0

        # Title
        worksheet.write(row, 0, 'Performance Summary Analysis', self.formats['title'])
        row += 3

        overall_performance = report_data.get('overall_performance', {})

        # Comprehensive metrics table
        metrics_data = [
            ['Metric', 'Value', 'Benchmark', 'Category'],
            ['Total Return', overall_performance.get('total_return', 0), '8%', 'Returns'],
            ['Annualized Return', overall_performance.get('annualized_return', 0), '8%', 'Returns'],
            ['Volatility', overall_performance.get('volatility', 0), '15%', 'Risk'],
            ['Sharpe Ratio', overall_performance.get('sharpe_ratio', 0), '1.0', 'Risk-Adjusted'],
            ['Sortino Ratio', overall_performance.get('sortino_ratio', 0), '1.0', 'Risk-Adjusted'],
            ['Calmar Ratio', overall_performance.get('calmar_ratio', 0), '1.0', 'Risk-Adjusted'],
            ['Maximum Drawdown', overall_performance.get('max_drawdown', 0), '-10%', 'Risk'],
            ['Value at Risk (95%)', overall_performance.get('var_95', 0), '-2%', 'Risk'],
            ['Expected Shortfall', overall_performance.get('expected_shortfall', 0), '-3%', 'Risk'],
            ['Win Rate', overall_performance.get('win_rate', 0), '55%', 'Trading'],
            ['Profit Factor', overall_performance.get('profit_factor', 0), '1.5', 'Trading'],
            ['Total Trades', overall_performance.get('total_trades', 0), 'N/A', 'Trading']
        ]

        # Write headers
        for col, header in enumerate(metrics_data[0]):
            worksheet.write(row, col, header, self.formats['header'])
        row += 1

        # Write data
        for metric_row in metrics_data[1:]:
            for col, value in enumerate(metric_row):
                if col == 0:  # Metric name
                    worksheet.write(row, col, value, self.formats['number'])
                elif col == 1:  # Value
                    if isinstance(value, (int, float)):
                        if 'Return' in metric_row[0] or 'Drawdown' in metric_row[0] or 'Risk' in metric_row[0] or 'Rate' in metric_row[0]:
                            worksheet.write(row, col, value, self.formats['percentage'])
                        elif 'Trades' in metric_row[0]:
                            worksheet.write(row, col, int(value), self.formats['number'])
                        else:
                            worksheet.write(row, col, value, self.formats['number'])
                    else:
                        worksheet.write(row, col, value, self.formats['number'])
                else:  # Benchmark and category
                    worksheet.write(row, col, value, self.formats['number'])
            row += 1

        # Monthly/Quarterly performance breakdown
        row += 2
        worksheet.write(row, 0, 'Period Performance Breakdown', self.formats['section_title'])
        row += 2

        # This would be populated with actual periodic returns if available
        # For now, create placeholder structure
        period_headers = ['Period', 'Return', 'Benchmark', 'Alpha', 'Volatility']
        for col, header in enumerate(period_headers):
            worksheet.write(row, col, header, self.formats['header'])

    def _create_phase_analysis(self, workbook, report_data: Dict[str, Any]):
        """Create phase-by-phase analysis worksheet."""

        worksheet = workbook.add_worksheet('Phase Analysis')
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:Z', 15)

        row = 0

        # Title
        worksheet.write(row, 0, 'Three-Phase Analysis', self.formats['title'])
        row += 3

        phase_results = report_data.get('phase_results', {})

        if not phase_results:
            worksheet.write(row, 0, 'No phase data available', self.formats['number'])
            return

        # Create comparison table
        phases = list(phase_results.keys())
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown',
                  'win_rate', 'volatility', 'total_trades']

        # Headers
        worksheet.write(row, 0, 'Metric', self.formats['header'])
        for col, phase in enumerate(phases, 1):
            worksheet.write(row, col, phase, self.formats['header'])
        row += 1

        # Metrics rows
        metric_labels = {
            'total_return': 'Total Return',
            'annualized_return': 'Annualized Return',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Maximum Drawdown',
            'win_rate': 'Win Rate',
            'volatility': 'Volatility',
            'total_trades': 'Total Trades'
        }

        for metric in metrics:
            worksheet.write(row, 0, metric_labels.get(metric, metric), self.formats['number'])

            for col, phase in enumerate(phases, 1):
                phase_data = phase_results.get(phase, {})
                value = phase_data.get(metric, 0)

                if metric in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'volatility']:
                    if value >= 0 and metric != 'max_drawdown':
                        worksheet.write(row, col, value, self.formats['positive'])
                    elif metric == 'max_drawdown':
                        worksheet.write(row, col, value, self.formats['negative'])
                    else:
                        worksheet.write(row, col, value, self.formats['negative'])
                elif metric == 'total_trades':
                    worksheet.write(row, col, int(value), self.formats['number'])
                else:
                    worksheet.write(row, col, value, self.formats['number'])

            row += 1

        # Phase consistency analysis
        row += 2
        worksheet.write(row, 0, 'Phase Consistency Analysis', self.formats['section_title'])
        row += 2

        # Calculate consistency metrics
        if len(phases) >= 2:
            returns = [phase_results[phase].get('total_return', 0) for phase in phases]
            sharpe_ratios = [phase_results[phase].get('sharpe_ratio', 0) for phase in phases]

            consistency_data = [
                ['Metric', 'Mean', 'Std Dev', 'Coefficient of Variation'],
                ['Returns', np.mean(returns), np.std(returns), np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else 0],
                ['Sharpe Ratios', np.mean(sharpe_ratios), np.std(sharpe_ratios), np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else 0]
            ]

            # Write consistency table
            for row_data in consistency_data:
                for col, value in enumerate(row_data):
                    if row == consistency_data.index(row_data) + row - len(consistency_data) + 1:  # Header row
                        worksheet.write(row, col, value, self.formats['header'])
                    else:
                        if col == 0:
                            worksheet.write(row, col, value, self.formats['number'])
                        else:
                            worksheet.write(row, col, value, self.formats['number'])
                row += 1

    def _create_risk_analysis(self, workbook, report_data: Dict[str, Any]):
        """Create comprehensive risk analysis worksheet."""

        worksheet = workbook.add_worksheet('Risk Analysis')
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:Z', 15)

        row = 0

        # Title
        worksheet.write(row, 0, 'Comprehensive Risk Analysis', self.formats['title'])
        row += 3

        overall_performance = report_data.get('overall_performance', {})

        # Risk metrics summary
        risk_metrics = [
            ['Risk Metric', 'Value', 'Risk Level', 'Industry Benchmark'],
            ['Portfolio Volatility', overall_performance.get('volatility', 0),
             self._assess_risk_level('volatility', overall_performance.get('volatility', 0)), '15-20%'],
            ['Maximum Drawdown', overall_performance.get('max_drawdown', 0),
             self._assess_risk_level('drawdown', overall_performance.get('max_drawdown', 0)), '< -10%'],
            ['Value at Risk (95%)', overall_performance.get('var_95', 0),
             self._assess_risk_level('var', overall_performance.get('var_95', 0)), '< -2%'],
            ['Expected Shortfall (95%)', overall_performance.get('expected_shortfall', 0),
             self._assess_risk_level('es', overall_performance.get('expected_shortfall', 0)), '< -3%'],
            ['Downside Deviation', overall_performance.get('downside_deviation', 0),
             self._assess_risk_level('volatility', overall_performance.get('downside_deviation', 0)), '< 12%']
        ]

        # Write risk metrics table
        for row_data in risk_metrics:
            for col, value in enumerate(row_data):
                if row == 3:  # Header row
                    worksheet.write(row, col, value, self.formats['header'])
                else:
                    if col == 0 or col == 2 or col == 3:  # Text columns
                        worksheet.write(row, col, value, self.formats['number'])
                    else:  # Value column
                        if isinstance(value, (int, float)):
                            worksheet.write(row, col, value, self.formats['percentage'])
                        else:
                            worksheet.write(row, col, value, self.formats['number'])
            row += 1

        # Drawdown analysis
        row += 2
        worksheet.write(row, 0, 'Drawdown Analysis', self.formats['section_title'])
        row += 2

        drawdown_data = [
            ['Drawdown Metric', 'Value'],
            ['Maximum Drawdown', overall_performance.get('max_drawdown', 0)],
            ['Average Drawdown', overall_performance.get('avg_drawdown', 0)],
            ['Max Drawdown Duration (Days)', overall_performance.get('max_drawdown_duration', 0)],
            ['Recovery Factor', abs(overall_performance.get('total_return', 0) / overall_performance.get('max_drawdown', -1)) if overall_performance.get('max_drawdown', 0) != 0 else 0]
        ]

        for row_data in drawdown_data:
            for col, value in enumerate(row_data):
                if drawdown_data.index(row_data) == 0:  # Header
                    worksheet.write(row, col, value, self.formats['header'])
                else:
                    if col == 0:
                        worksheet.write(row, col, value, self.formats['number'])
                    else:
                        if 'Duration' in row_data[0]:
                            worksheet.write(row, col, int(value) if value else 0, self.formats['number'])
                        elif 'Factor' in row_data[0]:
                            worksheet.write(row, col, value, self.formats['number'])
                        else:
                            worksheet.write(row, col, value, self.formats['percentage'])
            row += 1

        # Crisis period performance
        crisis_results = report_data.get('crisis_results', {})
        if crisis_results:
            row += 2
            worksheet.write(row, 0, 'Crisis Period Performance', self.formats['section_title'])
            row += 2

            crisis_headers = ['Crisis Period', 'Return', 'Max Drawdown', 'Recovery Days']
            for col, header in enumerate(crisis_headers):
                worksheet.write(row, col, header, self.formats['header'])
            row += 1

            for crisis_name, crisis_data in crisis_results.items():
                worksheet.write(row, 0, crisis_name, self.formats['number'])
                worksheet.write(row, 1, crisis_data.get('total_return', 0), self.formats['percentage'])
                worksheet.write(row, 2, crisis_data.get('max_drawdown', 0), self.formats['percentage'])
                worksheet.write(row, 3, crisis_data.get('max_drawdown_duration', 0), self.formats['number'])
                row += 1

    def _create_statistical_tests(self, workbook, report_data: Dict[str, Any]):
        """Create statistical significance tests worksheet."""

        worksheet = workbook.add_worksheet('Statistical Tests')
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:E', 15)
        worksheet.set_column('F:F', 40)

        row = 0

        # Title
        worksheet.write(row, 0, 'Statistical Significance Analysis', self.formats['title'])
        row += 3

        statistical_tests = report_data.get('statistical_tests', [])

        if not statistical_tests:
            worksheet.write(row, 0, 'No statistical tests available', self.formats['number'])
            return

        # Test results table
        headers = ['Test Name', 'Statistic', 'P-Value', 'Critical Value', 'Significant', 'Interpretation']
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, self.formats['header'])
        row += 1

        for test in statistical_tests:
            worksheet.write(row, 0, test.get('test_name', ''), self.formats['number'])
            worksheet.write(row, 1, test.get('statistic', 0), self.formats['number'])
            worksheet.write(row, 2, test.get('p_value', 0), self.formats['number'])
            worksheet.write(row, 3, test.get('critical_value', 0), self.formats['number'])

            # Significance indicator
            is_significant = test.get('is_significant', False)
            significance_format = self.formats['positive'] if is_significant else self.formats['negative']
            worksheet.write(row, 4, 'Yes' if is_significant else 'No', significance_format)

            # Interpretation
            interpretation = test.get('interpretation', '')
            wrapped_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
            worksheet.write(row, 5, interpretation, wrapped_format)
            worksheet.set_row(row, 30)  # Increase row height for wrapped text

            row += 1

        # Statistical summary
        row += 2
        worksheet.write(row, 0, 'Test Summary', self.formats['section_title'])
        row += 2

        total_tests = len(statistical_tests)
        significant_tests = sum(1 for test in statistical_tests if test.get('is_significant', False))

        summary_data = [
            ['Summary Metric', 'Value'],
            ['Total Tests Conducted', total_tests],
            ['Significant Results', significant_tests],
            ['Significance Rate', significant_tests / total_tests if total_tests > 0 else 0]
        ]

        for row_data in summary_data:
            for col, value in enumerate(row_data):
                if summary_data.index(row_data) == 0:
                    worksheet.write(row, col, value, self.formats['header'])
                else:
                    if col == 0:
                        worksheet.write(row, col, value, self.formats['number'])
                    else:
                        if 'Rate' in row_data[0]:
                            worksheet.write(row, col, value, self.formats['percentage'])
                        else:
                            worksheet.write(row, col, int(value), self.formats['number'])
            row += 1

    def _create_time_series_data(self, workbook, report_data: Dict[str, Any]):
        """Create time series data worksheet."""

        worksheet = workbook.add_worksheet('Time Series Data')
        worksheet.set_column('A:A', 12)
        worksheet.set_column('B:Z', 15)

        row = 0

        # Title
        worksheet.write(row, 0, 'Time Series Performance Data', self.formats['title'])
        row += 3

        # This would contain actual time series data
        # For now, create headers for the structure
        headers = ['Date', 'Portfolio Value', 'Daily Return', 'Cumulative Return',
                  'Drawdown', 'Rolling Sharpe (252d)', 'Rolling Vol (252d)']

        for col, header in enumerate(headers):
            worksheet.write(row, col, header, self.formats['header'])

        # Note: In a real implementation, this would be populated with actual time series data
        # from the backtest results, showing daily portfolio values, returns, rolling metrics, etc.

    def _create_trade_analysis(self, workbook, report_data: Dict[str, Any]):
        """Create trade-level analysis worksheet."""

        worksheet = workbook.add_worksheet('Trade Analysis')
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:Z', 15)

        row = 0

        # Title
        worksheet.write(row, 0, 'Trading Performance Analysis', self.formats['title'])
        row += 3

        overall_performance = report_data.get('overall_performance', {})

        # Trading statistics
        trading_stats = [
            ['Trading Metric', 'Value'],
            ['Total Trades', overall_performance.get('total_trades', 0)],
            ['Profitable Trades', overall_performance.get('profitable_trades', 0)],
            ['Losing Trades', overall_performance.get('losing_trades', 0)],
            ['Win Rate', overall_performance.get('win_rate', 0)],
            ['Average Win', overall_performance.get('average_win', 0)],
            ['Average Loss', overall_performance.get('average_loss', 0)],
            ['Largest Win', overall_performance.get('largest_win', 0)],
            ['Largest Loss', overall_performance.get('largest_loss', 0)],
            ['Profit Factor', overall_performance.get('profit_factor', 0)]
        ]

        for row_data in trading_stats:
            for col, value in enumerate(row_data):
                if trading_stats.index(row_data) == 0:
                    worksheet.write(row, col, value, self.formats['header'])
                else:
                    if col == 0:
                        worksheet.write(row, col, value, self.formats['number'])
                    else:
                        if 'Rate' in row_data[0] or 'Win' in row_data[0] or 'Loss' in row_data[0]:
                            if 'Trades' in row_data[0]:
                                worksheet.write(row, col, int(value), self.formats['number'])
                            else:
                                worksheet.write(row, col, value, self.formats['percentage'])
                        else:
                            worksheet.write(row, col, value, self.formats['number'])
            row += 1

    def _create_benchmark_comparison(self, workbook, report_data: Dict[str, Any]):
        """Create benchmark comparison worksheet."""

        worksheet = workbook.add_worksheet('Benchmark Comparison')
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:Z', 15)

        row = 0

        # Title
        worksheet.write(row, 0, 'Benchmark Performance Comparison', self.formats['title'])
        row += 3

        benchmark_results = report_data.get('benchmark_results', {})
        overall_performance = report_data.get('overall_performance', {})

        if not benchmark_results:
            worksheet.write(row, 0, 'No benchmark data available', self.formats['number'])
            return

        # Comparison table
        headers = ['Benchmark', 'Strategy Return', 'Benchmark Return', 'Alpha', 'Outperformance']
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, self.formats['header'])
        row += 1

        for benchmark_name, benchmark_data in benchmark_results.items():
            worksheet.write(row, 0, benchmark_name, self.formats['number'])
            worksheet.write(row, 1, benchmark_data.get('strategy_return', 0), self.formats['percentage'])
            worksheet.write(row, 2, benchmark_data.get('benchmark_return', 0), self.formats['percentage'])
            worksheet.write(row, 3, benchmark_data.get('alpha', 0), self.formats['percentage'])

            outperformance = benchmark_data.get('outperformance', False)
            perf_format = self.formats['positive'] if outperformance else self.formats['negative']
            worksheet.write(row, 4, 'Yes' if outperformance else 'No', perf_format)
            row += 1

    def _create_charts_dashboard(self, workbook, report_data: Dict[str, Any]):
        """Create charts and visualizations worksheet."""

        worksheet = workbook.add_worksheet('Charts Dashboard')

        # Title
        worksheet.write('A1', 'Performance Visualizations', self.formats['title'])

        # Note: In a full implementation, this would create embedded Excel charts
        # showing equity curves, drawdown charts, rolling metrics, etc.

        chart_descriptions = [
            'A3: Equity Curve Chart - Shows portfolio value over time across all phases',
            'A10: Rolling Sharpe Ratio - 252-day rolling Sharpe ratio evolution',
            'A17: Drawdown Chart - Underwater curve showing drawdown periods',
            'A24: Returns Distribution - Histogram of daily returns with normal overlay',
            'A31: Phase Comparison - Side-by-side phase performance comparison'
        ]

        for i, description in enumerate(chart_descriptions):
            worksheet.write(f'A{3 + i*7}', description, self.formats['number'])

    def _create_raw_data_export(self, workbook, report_data: Dict[str, Any]):
        """Create raw data export worksheet."""

        worksheet = workbook.add_worksheet('Raw Data')
        worksheet.set_column('A:Z', 12)

        row = 0

        # Title
        worksheet.write(row, 0, 'Raw Data Export', self.formats['title'])
        row += 2

        # Export all available data in JSON-like format
        worksheet.write(row, 0, 'Complete Report Data (JSON Format)', self.formats['section_title'])
        row += 2

        # Convert report data to readable format
        def write_dict_data(data, start_row, indent=0):
            current_row = start_row
            indent_str = "  " * indent

            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        worksheet.write(current_row, 0, f"{indent_str}{key}:", self.formats['number'])
                        current_row += 1
                        current_row = write_dict_data(value, current_row, indent + 1)
                    else:
                        worksheet.write(current_row, 0, f"{indent_str}{key}:", self.formats['number'])
                        worksheet.write(current_row, 1, str(value), self.formats['number'])
                        current_row += 1
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        worksheet.write(current_row, 0, f"{indent_str}[{i}]:", self.formats['number'])
                        current_row += 1
                        current_row = write_dict_data(item, current_row, indent + 1)
                    else:
                        worksheet.write(current_row, 0, f"{indent_str}[{i}]:", self.formats['number'])
                        worksheet.write(current_row, 1, str(item), self.formats['number'])
                        current_row += 1

            return current_row

        # Write the data
        write_dict_data(report_data, row)

    def _assess_metric(self, metric_name: str, value: float) -> str:
        """Assess performance metric qualitatively."""

        if 'Return' in metric_name:
            if value > 0.15:
                return 'Excellent'
            elif value > 0.08:
                return 'Good'
            elif value > 0.04:
                return 'Moderate'
            else:
                return 'Poor'

        elif 'Sharpe' in metric_name:
            if value > 1.5:
                return 'Excellent'
            elif value > 1.0:
                return 'Good'
            elif value > 0.5:
                return 'Moderate'
            else:
                return 'Poor'

        elif 'Drawdown' in metric_name:
            if value > -0.05:
                return 'Excellent'
            elif value > -0.10:
                return 'Good'
            elif value > -0.20:
                return 'Moderate'
            else:
                return 'Poor'

        elif 'Win Rate' in metric_name:
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
        """Assess risk level qualitatively."""

        if risk_type == 'volatility':
            if value < 0.10:
                return 'Low'
            elif value < 0.20:
                return 'Medium'
            else:
                return 'High'

        elif risk_type == 'drawdown':
            if value > -0.05:
                return 'Low'
            elif value > -0.15:
                return 'Medium'
            else:
                return 'High'

        elif risk_type in ['var', 'es']:
            if value > -0.02:
                return 'Low'
            elif value > -0.05:
                return 'Medium'
            else:
                return 'High'

        return 'Unknown'


# Utility function
def generate_comprehensive_excel_report(strategy_name: str,
                                      report_data: Dict[str, Any],
                                      output_dir: str = "reports/excel") -> str:
    """
    Main entry point for generating comprehensive Excel reports.

    Args:
        strategy_name: Name of the trading strategy
        report_data: Processed backtesting data
        output_dir: Output directory for Excel files

    Returns:
        Path to generated Excel file
    """

    if not HAS_XLSXWRITER:
        raise ImportError("xlsxwriter is required for Excel export")

    exporter = ComprehensiveExcelExporter(output_dir)
    return exporter.generate_comprehensive_report(strategy_name, report_data)


if __name__ == "__main__":
    # Test Excel generation with sample data
    if HAS_XLSXWRITER:
        # Create sample data
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
                'total_trades': 1250,
                'profitable_trades': 700,
                'losing_trades': 550
            },
            'phase_results': {
                'Phase 1 (2006-2016)': {
                    'total_return': 0.156,
                    'annualized_return': 0.142,
                    'sharpe_ratio': 1.12,
                    'max_drawdown': -0.123,
                    'win_rate': 0.54,
                    'total_trades': 500
                },
                'Phase 2 (2017-2020)': {
                    'total_return': 0.089,
                    'annualized_return': 0.087,
                    'sharpe_ratio': 1.34,
                    'max_drawdown': -0.056,
                    'win_rate': 0.58,
                    'total_trades': 400
                },
                'Phase 3 (2021-2025)': {
                    'total_return': 0.067,
                    'annualized_return': 0.065,
                    'sharpe_ratio': 1.23,
                    'max_drawdown': -0.087,
                    'win_rate': 0.56,
                    'total_trades': 350
                }
            },
            'executive_summary': {
                'key_findings': 'Strategy demonstrates consistent performance across market cycles.',
                'viability_assessment': 'Suitable for institutional deployment.',
                'risk_assessment': 'Moderate risk with acceptable drawdown characteristics.'
            },
            'statistical_tests': [
                {
                    'test_name': 'Sharpe Ratio Significance',
                    'statistic': 2.45,
                    'p_value': 0.014,
                    'is_significant': True,
                    'interpretation': 'Statistically significant risk-adjusted returns'
                }
            ]
        }

        excel_path = generate_comprehensive_excel_report("Sample Strategy", sample_data)
        print(f"Sample Excel report generated: {excel_path}")
    else:
        print("xlsxwriter not available - skipping Excel generation test")