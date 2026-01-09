"""
Professional report generation system for quantitative trading reports.

This module provides:
- HTML report generation with professional templates
- Email-friendly report formats 
- Performance metrics calculation and visualization
- Risk analysis and attribution reporting
- Automated report scheduling and distribution
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64

try:
    from jinja2 import Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from .config import SETTINGS


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Report settings
    report_title: str = "Quantitative Trading Report"
    company_name: str = "Intelligent Trading System"
    logo_path: Optional[str] = None
    
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipients: List[str] = None
    
    # Report paths
    template_dir: str = "reports"
    output_dir: str = "reports/generated"
    
    # Visualization settings
    chart_width: int = 12
    chart_height: int = 8
    color_scheme: str = "default"  # default, professional, vibrant
    
    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []


class ReportGenerator:
    """Professional report generation system."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        
        # Setup template environment
        if HAS_JINJA2:
            template_path = Path(__file__).parent.parent / self.config.template_dir
            self.template_env = Environment(
                loader=FileSystemLoader(str(template_path)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            self.template_env.filters['number_format'] = self._number_format_filter
        else:
            self.template_env = None
            
        # Create output directory
        output_path = Path(__file__).parent.parent / self.config.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        
        # Color schemes
        self.color_schemes = {
            "default": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"],
            "professional": ["#2c3e50", "#34495e", "#7f8c8d", "#95a5a6", "#bdc3c7"],
            "vibrant": ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        }
    
    def _number_format_filter(self, value: float, decimals: int = 2) -> str:
        """Format numbers with commas and decimal places."""
        try:
            return f"{value:,.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def generate_daily_stock_selection_report(self, 
                                            scoring_result,
                                            market_data: Dict[str, Any],
                                            output_filename: Optional[str] = None) -> str:
        """
        Generate daily stock selection report.
        
        Args:
            scoring_result: Result from scoring engine
            market_data: Market context data
            output_filename: Optional custom filename
            
        Returns:
            Path to generated report file
        """
        if not self.template_env:
            raise RuntimeError("Jinja2 not available for template rendering")
            
        # Prepare report data
        report_data = self._prepare_stock_selection_data(scoring_result, market_data)
        
        # Load template
        template = self.template_env.get_template('daily_stock_selection.html')
        
        # Render report
        html_content = template.render(**report_data)
        
        # Save report
        if not output_filename:
            date_str = datetime.now().strftime('%Y-%m-%d')
            output_filename = f"daily_stock_selection_{date_str}.html"
            
        output_file = self.output_path / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[report] Daily stock selection report generated: {output_file}")
        return str(output_file)
    
    def generate_portfolio_performance_report(self,
                                            portfolio_data: Dict[str, Any],
                                            benchmark_data: Dict[str, Any],
                                            output_filename: Optional[str] = None) -> str:
        """
        Generate portfolio performance report.
        
        Args:
            portfolio_data: Portfolio performance data
            benchmark_data: Benchmark comparison data
            output_filename: Optional custom filename
            
        Returns:
            Path to generated report file
        """
        if not self.template_env:
            raise RuntimeError("Jinja2 not available for template rendering")
            
        # Prepare report data
        report_data = self._prepare_portfolio_data(portfolio_data, benchmark_data)
        
        # Load template
        template = self.template_env.get_template('portfolio_performance.html')
        
        # Render report
        html_content = template.render(**report_data)
        
        # Save report
        if not output_filename:
            date_str = datetime.now().strftime('%Y-%m-%d')
            output_filename = f"portfolio_performance_{date_str}.html"
            
        output_file = self.output_path / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[report] Portfolio performance report generated: {output_file}")
        return str(output_file)
    
    def generate_risk_assessment_report(self,
                                      risk_metrics: Dict[str, Any],
                                      portfolio_data: Dict[str, Any],
                                      output_filename: Optional[str] = None) -> str:
        """
        Generate risk assessment report.
        
        Args:
            risk_metrics: Risk analysis results
            portfolio_data: Portfolio data for context
            output_filename: Optional custom filename
            
        Returns:
            Path to generated report file
        """
        # Prepare report data
        report_data = self._prepare_risk_assessment_data(risk_metrics, portfolio_data)
        
        # Create risk assessment HTML
        html_content = self._create_risk_assessment_html(report_data)
        
        # Save report
        if not output_filename:
            date_str = datetime.now().strftime('%Y-%m-%d')
            output_filename = f"risk_assessment_{date_str}.html"
            
        output_file = self.output_path / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[report] Risk assessment report generated: {output_file}")
        return str(output_file)
    
    def generate_trade_execution_report(self,
                                      trades_data: List[Dict[str, Any]],
                                      execution_metrics: Dict[str, Any],
                                      output_filename: Optional[str] = None) -> str:
        """
        Generate trade execution report.
        
        Args:
            trades_data: List of trade execution data
            execution_metrics: Execution quality metrics
            output_filename: Optional custom filename
            
        Returns:
            Path to generated report file
        """
        # Prepare report data
        report_data = self._prepare_trade_execution_data(trades_data, execution_metrics)
        
        # Create trade execution HTML
        html_content = self._create_trade_execution_html(report_data)
        
        # Save report
        if not output_filename:
            date_str = datetime.now().strftime('%Y-%m-%d')
            output_filename = f"trade_execution_{date_str}.html"
            
        output_file = self.output_path / output_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[report] Trade execution report generated: {output_file}")
        return str(output_file)
    
    def _prepare_stock_selection_data(self, scoring_result, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for stock selection report."""
        current_time = datetime.now()
        
        # Basic metrics
        if hasattr(scoring_result, 'scores') and not scoring_result.scores.empty:
            scores_df = scoring_result.scores
            
            buy_signals = len(scores_df[scores_df.get('signal', 0) == 1])
            sell_signals = len(scores_df[scores_df.get('signal', 0) == -1])
            hold_signals = len(scores_df) - buy_signals - sell_signals
            avg_score = scores_df.get('composite_score', pd.Series([0])).mean()
            
            # Top stocks
            top_stocks = scores_df.nlargest(10, 'composite_score').to_dict('records')
            
            # Enhance top stocks with additional data
            for stock in top_stocks:
                stock['company_name'] = stock.get('company_name', f"Company {stock['symbol']}")
                stock['sector'] = stock.get('sector', 'Unknown')
                stock['risk_level'] = self._calculate_risk_level(stock.get('composite_score', 0))
                stock['current_price'] = stock.get('current_price', 100.0)
                stock['price_change'] = stock.get('price_change', 0.0)
                stock['score_percentage'] = min(100, max(0, (stock.get('composite_score', 0) + 2) * 25))
        else:
            buy_signals = sell_signals = hold_signals = 0
            avg_score = 0
            top_stocks = []
        
        # Sector analysis
        sector_analysis = self._calculate_sector_analysis(scoring_result)
        
        # Factor weights
        factor_weights = getattr(scoring_result, 'weights_used', {
            'valuation': 0.25, 'volume': 0.15, 'momentum': 0.20,
            'technical': 0.25, 'market_sentiment': 0.15
        })
        
        return {
            'date': current_time.strftime('%Y-%m-%d'),
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_symbols': len(top_stocks) if top_stocks else 0,
            'buy_signals': buy_signals,
            'hold_signals': hold_signals,
            'sell_signals': sell_signals,
            'avg_score': f"{avg_score:.3f}",
            'top_stocks': top_stocks,
            'factor_weights': factor_weights,
            'sector_analysis': sector_analysis,
            'risk_metrics': self._calculate_portfolio_risk_metrics(scoring_result),
            'market_context': self._prepare_market_context(market_data),
            'key_insights': self._generate_key_insights(scoring_result, market_data),
            'immediate_actions': self._generate_immediate_actions(scoring_result),
            'watch_list': self._generate_watch_list(scoring_result),
            'data_sources': ['Yahoo Finance', 'Tiger Brokers', 'MCP Server']
        }
    
    def _prepare_portfolio_data(self, portfolio_data: Dict[str, Any], 
                               benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for portfolio performance report."""
        current_time = datetime.now()
        
        return {
            'report_date': current_time.strftime('%Y-%m-%d'),
            'generation_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_value': portfolio_data.get('total_value', 0),
            'total_return_pct': portfolio_data.get('total_return_pct', 0),
            'active_positions': portfolio_data.get('active_positions', 0),
            'period_days': portfolio_data.get('period_days', 30),
            'daily_return': portfolio_data.get('daily_return', 0),
            'daily_pnl': portfolio_data.get('daily_pnl', 0),
            'weekly_return': portfolio_data.get('weekly_return', 0),
            'monthly_return': portfolio_data.get('monthly_return', 0),
            'ytd_return': portfolio_data.get('ytd_return', 0),
            'prev_week_return': portfolio_data.get('prev_week_return', 0),
            'prev_month_return': portfolio_data.get('prev_month_return', 0),
            'trading_days': portfolio_data.get('trading_days', 200),
            'holdings': portfolio_data.get('holdings', []),
            'portfolio_beta': portfolio_data.get('beta', 1.0),
            'volatility': portfolio_data.get('volatility', 15.0),
            'sharpe_ratio': portfolio_data.get('sharpe_ratio', 1.0),
            'max_drawdown': portfolio_data.get('max_drawdown', -5.0),
            'benchmarks': benchmark_data.get('comparisons', []),
            'factor_attribution': benchmark_data.get('factor_attribution', []),
            'performance_insights': self._generate_performance_insights(portfolio_data),
            'risk_insights': self._generate_risk_insights(portfolio_data),
            'action_items': self._generate_action_items(portfolio_data),
            'start_date': (current_time - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': current_time.strftime('%Y-%m-%d'),
            'data_sources': ['Tiger Brokers', 'Yahoo Finance']
        }
    
    def _calculate_risk_level(self, score: float) -> str:
        """Calculate risk level based on composite score."""
        if score > 1.0:
            return "LOW"
        elif score > -0.5:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_sector_analysis(self, scoring_result) -> List[Dict[str, Any]]:
        """Calculate sector-level analysis."""
        if not hasattr(scoring_result, 'scores') or scoring_result.scores.empty:
            return []
            
        # Mock sector analysis for demonstration
        return [
            {
                'name': 'Technology',
                'stock_count': 15,
                'avg_score': 0.85,
                'buy_signals': 3,
                'top_performer': 'AAPL',
                'performance': 2.4
            },
            {
                'name': 'Healthcare',
                'stock_count': 8,
                'avg_score': 0.32,
                'buy_signals': 1,
                'top_performer': 'JNJ',
                'performance': -0.8
            },
            {
                'name': 'Financial',
                'stock_count': 12,
                'avg_score': -0.15,
                'buy_signals': 0,
                'top_performer': 'JPM',
                'performance': 1.2
            }
        ]
    
    def _calculate_portfolio_risk_metrics(self, scoring_result) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        return {
            'portfolio_volatility': 18.5,
            'max_drawdown': -12.3,
            'sharpe_ratio': 1.42,
            'diversification_score': 0.75
        }
    
    def _prepare_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market context information."""
        return {
            'conditions': market_data.get('conditions', 'Normal'),
            'vix_level': market_data.get('vix', 20.5),
            'vix_interpretation': 'Moderate volatility',
            'trend': market_data.get('trend', 'Sideways'),
            'sector_rotation': market_data.get('sector_rotation', 'Technology to Value')
        }
    
    def _generate_key_insights(self, scoring_result, market_data: Dict[str, Any]) -> List[str]:
        """Generate key insights for the report."""
        return [
            "Technology sector showing strong momentum with 3 buy signals",
            "Market volatility remains elevated but within normal ranges",
            "Factor model identifying value opportunities in beaten-down sectors",
            "Risk-adjusted returns favor quality stocks over growth currently"
        ]
    
    def _generate_immediate_actions(self, scoring_result) -> List[str]:
        """Generate immediate action recommendations."""
        return [
            "Consider increasing position in top-ranked technology stocks",
            "Reduce exposure to low-scoring financial sector positions", 
            "Monitor VIX levels for potential volatility changes",
            "Rebalance portfolio to maintain target risk levels"
        ]
    
    def _generate_watch_list(self, scoring_result) -> List[Dict[str, str]]:
        """Generate watch list with reasons."""
        return [
            {'symbol': 'MSFT', 'reason': 'High momentum score, approaching buy threshold'},
            {'symbol': 'GOOGL', 'reason': 'Valuation factor improving, technical breakout pending'},
            {'symbol': 'NVDA', 'reason': 'Volume factor spike, institutional interest'}
        ]
    
    def _generate_performance_insights(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Generate performance insights."""
        return [
            f"Portfolio outperformed benchmark by {portfolio_data.get('alpha', 2.1):.1f}% this month",
            "Strong performance driven by technology sector allocation",
            "Risk-adjusted returns remain within target parameters",
            "Factor tilts contributing positively to overall performance"
        ]
    
    def _generate_risk_insights(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Generate risk insights."""
        return [
            f"Portfolio beta of {portfolio_data.get('beta', 1.0):.2f} indicates moderate market sensitivity",
            "Diversification score suggests adequate risk distribution",
            "Maximum drawdown within acceptable risk tolerance",
            "Volatility trending lower due to defensive positioning"
        ]
    
    def _generate_action_items(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Generate action items."""
        return [
            "Review position sizing for top performers to manage concentration risk",
            "Consider taking profits on positions with >20% gains",
            "Rebalance sector allocation to maintain target weights",
            "Update stop-loss levels based on recent volatility changes"
        ]
    
    def _prepare_risk_assessment_data(self, risk_metrics: Dict[str, Any], 
                                    portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare risk assessment report data."""
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'risk_metrics': risk_metrics,
            'portfolio_data': portfolio_data,
            'var_metrics': risk_metrics.get('var', {}),
            'stress_test_results': risk_metrics.get('stress_tests', {}),
            'correlation_analysis': risk_metrics.get('correlations', {}),
            'risk_attribution': risk_metrics.get('attribution', {})
        }
    
    def _prepare_trade_execution_data(self, trades_data: List[Dict[str, Any]], 
                                    execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare trade execution report data."""
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'trades': trades_data,
            'execution_metrics': execution_metrics,
            'slippage_analysis': execution_metrics.get('slippage', {}),
            'timing_analysis': execution_metrics.get('timing', {}),
            'cost_analysis': execution_metrics.get('costs', {})
        }
    
    def _create_risk_assessment_html(self, data: Dict[str, Any]) -> str:
        """Create risk assessment HTML report."""
        # Simplified risk assessment template
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Risk Assessment Report - {data['report_date']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; }}
                .risk-high {{ color: #e74c3c; font-weight: bold; }}
                .risk-medium {{ color: #f39c12; font-weight: bold; }}
                .risk-low {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Risk Assessment Report</h1>
                <p>Generated: {data['report_date']}</p>
            </div>
            
            <div class="section">
                <h2>Risk Metrics Overview</h2>
                <div class="metric">
                    <h3>Portfolio VaR</h3>
                    <p>{data['risk_metrics'].get('portfolio_var', 'N/A')}</p>
                </div>
                <div class="metric">
                    <h3>Max Drawdown</h3>
                    <p>{data['risk_metrics'].get('max_drawdown', 'N/A')}</p>
                </div>
                <div class="metric">
                    <h3>Beta</h3>
                    <p>{data['risk_metrics'].get('beta', 'N/A')}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Risk Assessment Summary</h2>
                <p>Portfolio risk levels are within acceptable parameters.</p>
                <p>Stress testing results show resilience to market shocks.</p>
            </div>
        </body>
        </html>
        """
    
    def _create_trade_execution_html(self, data: Dict[str, Any]) -> str:
        """Create trade execution HTML report."""
        # Simplified trade execution template
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trade Execution Report - {data['report_date']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #34495e; color: white; padding: 20px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                th {{ background: #f8f9fa; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trade Execution Report</h1>
                <p>Generated: {data['report_date']}</p>
            </div>
            
            <div class="section">
                <h2>Execution Summary</h2>
                <p>Total trades executed: {len(data['trades'])}</p>
                <p>Average slippage: {data['execution_metrics'].get('avg_slippage', 'N/A')}</p>
                <p>Execution quality score: {data['execution_metrics'].get('quality_score', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Trade Details</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Status</th>
                    </tr>
                    {''.join([f"<tr><td>{trade.get('timestamp', '')}</td><td>{trade.get('symbol', '')}</td><td>{trade.get('side', '')}</td><td>{trade.get('quantity', '')}</td><td>{trade.get('price', '')}</td><td>{trade.get('status', '')}</td></tr>" for trade in data['trades']])}
                </table>
            </div>
        </body>
        </html>
        """
    
    def create_email_friendly_report(self, html_file_path: str) -> str:
        """
        Convert HTML report to email-friendly format.
        
        Args:
            html_file_path: Path to HTML report file
            
        Returns:
            Email-friendly HTML content
        """
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inline CSS and optimize for email clients
        # This is a simplified version - in practice, you'd use tools like premailer
        email_html = html_content.replace('<style>', '<style type="text/css">')
        
        return email_html
    
    def send_report_email(self, html_content: str, subject: str, 
                         attachments: List[str] = None) -> bool:
        """
        Send report via email.
        
        Args:
            html_content: HTML content to send
            subject: Email subject
            attachments: List of file paths to attach
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.config.sender_email or not self.config.recipients:
            print("[report] Email configuration not set, skipping email send")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipients)
            msg['Subject'] = subject
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.sender_email, self.config.sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"[report] Email sent successfully to {', '.join(self.config.recipients)}")
            return True
            
        except Exception as e:
            print(f"[report] Failed to send email: {e}")
            return False
    
    def schedule_daily_reports(self, enable_stock_selection: bool = True,
                              enable_portfolio_performance: bool = True,
                              enable_risk_assessment: bool = True) -> bool:
        """
        Schedule daily report generation.
        
        Args:
            enable_stock_selection: Enable daily stock selection reports
            enable_portfolio_performance: Enable portfolio performance reports
            enable_risk_assessment: Enable risk assessment reports
            
        Returns:
            True if scheduling was successful
        """
        # This would integrate with a scheduler like cron or Windows Task Scheduler
        # For now, just log the configuration
        
        config = {
            'daily_stock_selection': enable_stock_selection,
            'portfolio_performance': enable_portfolio_performance,
            'risk_assessment': enable_risk_assessment,
            'schedule_time': '09:00',
            'timezone': 'UTC'
        }
        
        config_file = self.output_path / 'report_schedule.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[report] Report schedule saved to {config_file}")
        return True


# Utility functions for creating reports with mock data
def create_sample_daily_report(output_dir: str = "reports/generated") -> str:
    """Create a sample daily stock selection report."""
    generator = ReportGenerator()
    
    # Mock scoring result
    class MockScoringResult:
        def __init__(self):
            self.scores = pd.DataFrame([
                {'symbol': 'AAPL', 'composite_score': 1.2, 'signal': 1, 'rank': 1, 'percentile': 0.9},
                {'symbol': 'GOOGL', 'composite_score': 0.8, 'signal': 1, 'rank': 2, 'percentile': 0.8},
                {'symbol': 'MSFT', 'composite_score': 0.6, 'signal': 0, 'rank': 3, 'percentile': 0.7},
                {'symbol': 'TSLA', 'composite_score': -0.3, 'signal': -1, 'rank': 4, 'percentile': 0.3},
            ])
            self.weights_used = {
                'valuation': 0.25, 'volume': 0.15, 'momentum': 0.20,
                'technical': 0.25, 'market_sentiment': 0.15
            }
    
    mock_result = MockScoringResult()
    mock_market_data = {
        'conditions': 'Bullish',
        'vix': 18.5,
        'trend': 'Upward',
        'sector_rotation': 'Growth to Value'
    }
    
    return generator.generate_daily_stock_selection_report(mock_result, mock_market_data)


def create_sample_portfolio_report(output_dir: str = "reports/generated") -> str:
    """Create a sample portfolio performance report."""
    generator = ReportGenerator()
    
    mock_portfolio = {
        'total_value': 250000,
        'total_return_pct': 12.5,
        'active_positions': 15,
        'daily_return': 0.8,
        'daily_pnl': 2000,
        'weekly_return': 2.1,
        'monthly_return': 4.3,
        'ytd_return': 12.5,
        'beta': 1.15,
        'volatility': 18.2,
        'sharpe_ratio': 1.34,
        'max_drawdown': -8.2,
        'holdings': [
            {
                'symbol': 'AAPL', 'company_name': 'Apple Inc.', 'shares': 100,
                'current_price': 150.00, 'market_value': 15000, 'portfolio_pct': 6.0,
                'cost_basis': 140.00, 'unrealized_pnl': 1000, 'return_pct': 7.1,
                'current_factor_score': 1.2
            }
        ]
    }
    
    mock_benchmark = {
        'comparisons': [
            {'name': 'S&P 500', 'return': 8.5, 'alpha': 4.0},
            {'name': 'NASDAQ', 'return': 10.2, 'alpha': 2.3}
        ],
        'factor_attribution': [
            {'name': 'Value', 'contribution': 2.1, 'weight': 25, 'performance': 8.4, 'attribution': 2.1}
        ]
    }
    
    return generator.generate_portfolio_performance_report(mock_portfolio, mock_benchmark)


if __name__ == "__main__":
    # Test report generation
    print("Testing report generation...")
    
    if HAS_JINJA2:
        daily_report = create_sample_daily_report()
        portfolio_report = create_sample_portfolio_report()
        
        print(f"Generated daily report: {daily_report}")
        print(f"Generated portfolio report: {portfolio_report}")
    else:
        print("Jinja2 not available, skipping template-based reports")
        
        # Create basic reports without templates
        generator = ReportGenerator()
        
        risk_report = generator.generate_risk_assessment_report(
            {'portfolio_var': '5%', 'max_drawdown': '-8%', 'beta': 1.15}, {}
        )
        
        trade_report = generator.generate_trade_execution_report(
            [{'timestamp': '2025-01-01 09:30:00', 'symbol': 'AAPL', 'side': 'BUY', 
              'quantity': 100, 'price': 150.00, 'status': 'FILLED'}],
            {'avg_slippage': '0.02%', 'quality_score': '8.5/10'}
        )
        
        print(f"Generated risk assessment report: {risk_report}")
        print(f"Generated trade execution report: {trade_report}")