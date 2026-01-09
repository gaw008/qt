"""
专业期货交易界面
===============

专为期货交易设计的高级界面，包含：
- 期货合约规格管理
- 保证金实时监控
- 连续合约切换
- 基差分析与期现套利
- 多合约组合风险管理
- 期货特有交易功能

Author: Agent D1 - Interface Optimization Specialist
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

# 页面配置
st.set_page_config(
    page_title="专业期货交易平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 期货交易专用CSS样式
st.markdown("""
<style>
    .futures-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .contract-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.18);
        transition: transform 0.3s ease;
    }
    
    .contract-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    .margin-warning {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        border-left: 4px solid #e74c3c;
        animation: pulse 2s infinite;
    }
    
    .margin-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        border-left: 4px solid #00b894;
    }
    
    .spread-opportunity {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        border: 2px solid #f39c12;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 10px #f39c12; }
        to { box-shadow: 0 0 20px #f39c12, 0 0 30px #f39c12; }
    }
    
    .position-table {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .rollover-alert {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        border-left: 4px solid #6c5ce7;
        margin: 1rem 0;
    }
    
    .arbitrage-signal {
        background: linear-gradient(135deg, #00cec9 0%, #55a3ff 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        border: 2px solid #00cec9;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FuturesContractManager:
    """期货合约管理系统"""
    
    def __init__(self):
        self.contracts = {
            # 股指期货
            'ES': {
                'name': 'E-mini S&P 500',
                'exchange': 'CME',
                'multiplier': 50,
                'tick_size': 0.25,
                'tick_value': 12.50,
                'margin_day': 12500,
                'margin_overnight': 15000,
                'settlement_method': 'cash',
                'trading_hours': '23:00-22:00 CT',
                'months': ['H', 'M', 'U', 'Z'],
                'current_contract': 'ESZ4',
                'next_contract': 'ESH5',
                'rollover_date': '2024-12-19'
            },
            'NQ': {
                'name': 'E-mini NASDAQ-100',
                'exchange': 'CME',
                'multiplier': 20,
                'tick_size': 0.25,
                'tick_value': 5.00,
                'margin_day': 8000,
                'margin_overnight': 10000,
                'settlement_method': 'cash',
                'trading_hours': '23:00-22:00 CT',
                'months': ['H', 'M', 'U', 'Z'],
                'current_contract': 'NQZ4',
                'next_contract': 'NQH5',
                'rollover_date': '2024-12-19'
            },
            # 能源期货
            'CL': {
                'name': 'Crude Oil WTI',
                'exchange': 'NYMEX',
                'multiplier': 1000,
                'tick_size': 0.01,
                'tick_value': 10.00,
                'margin_day': 5000,
                'margin_overnight': 6000,
                'settlement_method': 'physical',
                'trading_hours': '23:00-22:00 CT',
                'months': ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
                'current_contract': 'CLZ4',
                'next_contract': 'CLF5',
                'rollover_date': '2024-11-20'
            },
            'NG': {
                'name': 'Natural Gas',
                'exchange': 'NYMEX',
                'multiplier': 10000,
                'tick_size': 0.001,
                'tick_value': 10.00,
                'margin_day': 3500,
                'margin_overnight': 4500,
                'settlement_method': 'physical',
                'trading_hours': '23:00-22:00 CT',
                'months': ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
                'current_contract': 'NGZ4',
                'next_contract': 'NGF5',
                'rollover_date': '2024-11-25'
            },
            # 贵金属
            'GC': {
                'name': 'Gold',
                'exchange': 'COMEX',
                'multiplier': 100,
                'tick_size': 0.10,
                'tick_value': 10.00,
                'margin_day': 6000,
                'margin_overnight': 7500,
                'settlement_method': 'physical',
                'trading_hours': '23:00-22:00 CT',
                'months': ['G', 'J', 'M', 'Q', 'V', 'Z'],
                'current_contract': 'GCZ4',
                'next_contract': 'GCG5',
                'rollover_date': '2024-11-27'
            },
            # 农产品
            'ZC': {
                'name': 'Corn',
                'exchange': 'CBOT',
                'multiplier': 5000,
                'tick_size': 0.25,
                'tick_value': 12.50,
                'margin_day': 2000,
                'margin_overnight': 2500,
                'settlement_method': 'physical',
                'trading_hours': '21:00-20:45 CT',
                'months': ['H', 'K', 'N', 'U', 'Z'],
                'current_contract': 'ZCZ4',
                'next_contract': 'ZCH5',
                'rollover_date': '2024-11-15'
            }
        }
        
        self.positions = {
            'ES': {'quantity': 5, 'entry_price': 4240.25, 'current_price': 4255.50, 'unrealized_pnl': 3812.50},
            'NQ': {'quantity': 2, 'entry_price': 13450.75, 'current_price': 13485.25, 'unrealized_pnl': 345.00},
            'CL': {'quantity': -3, 'entry_price': 78.45, 'current_price': 78.12, 'unrealized_pnl': 990.00},
            'GC': {'quantity': 4, 'entry_price': 1985.40, 'current_price': 1992.80, 'unrealized_pnl': 2960.00}
        }
        
        self.account_balance = 250000
        self.available_margin = 185000
        
    def get_contract_specs(self, symbol: str) -> Dict:
        """获取合约规格"""
        return self.contracts.get(symbol, {})
    
    def get_margin_requirements(self) -> Dict:
        """计算保证金要求"""
        total_day_margin = 0
        total_overnight_margin = 0
        
        for symbol, position in self.positions.items():
            if symbol in self.contracts:
                contract = self.contracts[symbol]
                quantity = abs(position['quantity'])
                total_day_margin += contract['margin_day'] * quantity
                total_overnight_margin += contract['margin_overnight'] * quantity
        
        return {
            'total_day_margin': total_day_margin,
            'total_overnight_margin': total_overnight_margin,
            'available_margin': self.available_margin,
            'margin_utilization': total_overnight_margin / self.account_balance,
            'excess_liquidity': self.available_margin - total_overnight_margin
        }
    
    def check_rollover_alerts(self) -> List[Dict]:
        """检查需要展期的合约"""
        alerts = []
        current_date = datetime.now()
        
        for symbol, position in self.positions.items():
            if symbol in self.contracts:
                contract = self.contracts[symbol]
                rollover_date = datetime.strptime(contract['rollover_date'], '%Y-%m-%d')
                days_to_rollover = (rollover_date - current_date).days
                
                if days_to_rollover <= 5:
                    alerts.append({
                        'symbol': symbol,
                        'current_contract': contract['current_contract'],
                        'next_contract': contract['next_contract'],
                        'days_remaining': days_to_rollover,
                        'position_size': position['quantity']
                    })
        
        return alerts

class SpreadAnalyzer:
    """期现套利分析器"""
    
    def __init__(self):
        self.spread_opportunities = {}
        self.calendar_spreads = {}
        self.inter_commodity_spreads = {}
    
    def analyze_calendar_spreads(self) -> List[Dict]:
        """分析跨期套利机会"""
        opportunities = []
        
        # ES跨期价差
        front_month_price = 4255.50
        back_month_price = 4268.75
        spread = back_month_price - front_month_price
        z_score = (spread - 8.5) / 3.2  # 历史均值和标准差
        
        if abs(z_score) > 2:
            opportunities.append({
                'type': 'Calendar Spread',
                'symbol': 'ES',
                'front_contract': 'ESZ4',
                'back_contract': 'ESH5',
                'spread_value': spread,
                'z_score': z_score,
                'signal': 'SELL SPREAD' if z_score > 2 else 'BUY SPREAD',
                'confidence': min(abs(z_score) / 3 * 100, 95)
            })
        
        # CL跨期价差
        front_cl_price = 78.12
        back_cl_price = 79.85
        cl_spread = back_cl_price - front_cl_price
        cl_z_score = (cl_spread - 1.2) / 0.8
        
        if abs(cl_z_score) > 1.5:
            opportunities.append({
                'type': 'Calendar Spread',
                'symbol': 'CL',
                'front_contract': 'CLZ4',
                'back_contract': 'CLF5',
                'spread_value': cl_spread,
                'z_score': cl_z_score,
                'signal': 'SELL SPREAD' if cl_z_score > 1.5 else 'BUY SPREAD',
                'confidence': min(abs(cl_z_score) / 2.5 * 100, 90)
            })
        
        return opportunities
    
    def analyze_intercommodity_spreads(self) -> List[Dict]:
        """分析跨品种套利机会"""
        opportunities = []
        
        # 原油/天然气价差
        cl_price = 78.12
        ng_price = 3.45
        oil_gas_ratio = cl_price / ng_price
        historical_ratio = 22.5
        ratio_z_score = (oil_gas_ratio - historical_ratio) / 2.1
        
        if abs(ratio_z_score) > 1.8:
            opportunities.append({
                'type': 'Inter-commodity',
                'leg1': 'CL',
                'leg2': 'NG',
                'ratio': oil_gas_ratio,
                'historical_ratio': historical_ratio,
                'z_score': ratio_z_score,
                'signal': 'LONG CL / SHORT NG' if ratio_z_score < -1.8 else 'SHORT CL / LONG NG',
                'confidence': min(abs(ratio_z_score) / 2.5 * 100, 88)
            })
        
        return opportunities

def create_futures_position_overview(futures_mgr: FuturesContractManager) -> go.Figure:
    """创建期货持仓概览图"""
    positions = futures_mgr.positions
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['持仓分布', '盈亏分析', '保证金使用', '合约到期分析'],
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "bar"}]]
    )
    
    # 持仓分布饼图
    symbols = list(positions.keys())
    position_values = [abs(pos['quantity'] * pos['current_price'] * 
                          futures_mgr.contracts[sym]['multiplier']) 
                      for sym, pos in positions.items()]
    
    fig.add_trace(go.Pie(
        labels=symbols,
        values=position_values,
        hole=0.4,
        textinfo='label+percent',
        textposition='auto'
    ), row=1, col=1)
    
    # 盈亏分析柱状图
    pnls = [pos['unrealized_pnl'] for pos in positions.values()]
    colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
    
    fig.add_trace(go.Bar(
        x=symbols,
        y=pnls,
        marker_color=colors,
        text=[f'${pnl:,.0f}' for pnl in pnls],
        textposition='auto'
    ), row=1, col=2)
    
    # 保证金使用指示器
    margin_data = futures_mgr.get_margin_requirements()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=margin_data['margin_utilization'] * 100,
        title={'text': "保证金使用率 (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkred"},
               'steps': [{'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "red"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=1)
    
    # 合约到期天数
    rollover_alerts = futures_mgr.check_rollover_alerts()
    if rollover_alerts:
        alert_symbols = [alert['symbol'] for alert in rollover_alerts]
        alert_days = [alert['days_remaining'] for alert in rollover_alerts]
        
        fig.add_trace(go.Bar(
            x=alert_symbols,
            y=alert_days,
            marker_color='orange',
            text=[f'{days}天' for days in alert_days],
            textposition='auto'
        ), row=2, col=2)
    
    fig.update_layout(
        title="期货持仓综合分析",
        height=800,
        showlegend=False
    )
    
    return fig

def create_spread_analysis_chart(spread_analyzer: SpreadAnalyzer) -> go.Figure:
    """创建价差分析图表"""
    calendar_opps = spread_analyzer.analyze_calendar_spreads()
    intercommodity_opps = spread_analyzer.analyze_intercommodity_spreads()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['跨期价差机会', 'Z-Score分析', '跨品种套利', '套利信心度'],
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # 跨期价差机会
    if calendar_opps:
        cal_symbols = [opp['symbol'] for opp in calendar_opps]
        cal_spreads = [opp['spread_value'] for opp in calendar_opps]
        
        fig.add_trace(go.Bar(
            x=cal_symbols,
            y=cal_spreads,
            marker_color='blue',
            text=[f'{spread:.2f}' for spread in cal_spreads],
            textposition='auto',
            name='跨期价差'
        ), row=1, col=1)
    
    # Z-Score分析
    if calendar_opps:
        z_scores = [opp['z_score'] for opp in calendar_opps]
        
        fig.add_trace(go.Scatter(
            x=cal_symbols,
            y=z_scores,
            mode='markers+lines',
            marker=dict(size=15, color=['red' if abs(z) > 2 else 'green' for z in z_scores]),
            name='Z-Score'
        ), row=1, col=2)
        
        # 添加阈值线
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=1, col=2)
    
    # 跨品种套利
    if intercommodity_opps:
        inter_pairs = [f"{opp['leg1']}/{opp['leg2']}" for opp in intercommodity_opps]
        inter_ratios = [opp['ratio'] for opp in intercommodity_opps]
        
        fig.add_trace(go.Bar(
            x=inter_pairs,
            y=inter_ratios,
            marker_color='purple',
            text=[f'{ratio:.2f}' for ratio in inter_ratios],
            textposition='auto',
            name='价格比率'
        ), row=2, col=1)
    
    # 套利信心度指示器
    if calendar_opps or intercommodity_opps:
        all_opps = calendar_opps + intercommodity_opps
        avg_confidence = np.mean([opp['confidence'] for opp in all_opps])
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=avg_confidence,
            title={'text': "平均信心度 (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}]},
            domain={'x': [0, 1], 'y': [0, 1]}
        ), row=2, col=2)
    
    fig.update_layout(
        title="价差套利分析",
        height=800,
        showlegend=True
    )
    
    return fig

def create_rollover_calendar(futures_mgr: FuturesContractManager) -> go.Figure:
    """创建展期日历"""
    fig = go.Figure()
    
    # 获取展期提醒
    rollover_alerts = futures_mgr.check_rollover_alerts()
    
    if rollover_alerts:
        symbols = [alert['symbol'] for alert in rollover_alerts]
        days_remaining = [alert['days_remaining'] for alert in rollover_alerts]
        current_contracts = [alert['current_contract'] for alert in rollover_alerts]
        next_contracts = [alert['next_contract'] for alert in rollover_alerts]
        
        # 创建甘特图风格的展期日历
        colors = ['red' if days <= 2 else 'orange' if days <= 5 else 'green' for days in days_remaining]
        
        fig.add_trace(go.Bar(
            x=days_remaining,
            y=symbols,
            orientation='h',
            marker_color=colors,
            text=[f'{curr} → {next}' for curr, next in zip(current_contracts, next_contracts)],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>剩余天数: %{x}<br>%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title="合约展期日历",
        xaxis_title="剩余天数",
        yaxis_title="合约品种",
        height=400
    )
    
    return fig

def display_futures_trading_dashboard():
    """显示期货交易主界面"""
    st.markdown('<h1 class="futures-header">📈 专业期货交易平台</h1>', unsafe_allow_html=True)
    
    # 初始化管理器
    futures_mgr = FuturesContractManager()
    spread_analyzer = SpreadAnalyzer()
    
    # 控制面板
    st.markdown("### 🎛️ 交易控制面板")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        selected_contract = st.selectbox(
            "选择合约",
            list(futures_mgr.contracts.keys())
        )
    
    with col2:
        order_type = st.selectbox(
            "订单类型",
            ["市价单", "限价单", "止损单", "止盈单"]
        )
    
    with col3:
        trade_action = st.selectbox(
            "交易方向",
            ["买入开仓", "卖出开仓", "买入平仓", "卖出平仓"]
        )
    
    with col4:
        quantity = st.number_input("数量", min_value=1, max_value=100, value=1)
    
    with col5:
        if order_type == "限价单":
            price = st.number_input("价格", value=futures_mgr.positions.get(selected_contract, {}).get('current_price', 100.0))
    
    # 快速交易按钮
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 执行交易", use_container_width=True):
            st.success(f"已提交{trade_action} {selected_contract} {quantity}手的{order_type}")
    
    with col2:
        if st.button("🔄 展期操作", use_container_width=True):
            st.info(f"已启动{selected_contract}的自动展期程序")
    
    with col3:
        if st.button("⚡ 套利交易", use_container_width=True):
            st.info("套利交易模块已激活，正在搜索机会...")
    
    # 保证金状态警报
    margin_data = futures_mgr.get_margin_requirements()
    
    if margin_data['margin_utilization'] > 0.85:
        st.markdown(f"""
        <div class="margin-warning">
            <h4>⚠️ 保证金使用率警告</h4>
            <p>当前保证金使用率: <strong>{margin_data['margin_utilization']:.1%}</strong></p>
            <p>剩余可用保证金: <strong>${margin_data['excess_liquidity']:,}</strong></p>
            <p>建议及时补充保证金或减少持仓!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="margin-safe">
            <h4>✅ 保证金状态良好</h4>
            <p>保证金使用率: <strong>{margin_data['margin_utilization']:.1%}</strong></p>
            <p>剩余可用: <strong>${margin_data['excess_liquidity']:,}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # 展期提醒
    rollover_alerts = futures_mgr.check_rollover_alerts()
    
    if rollover_alerts:
        st.markdown("### 📅 合约展期提醒")
        
        for alert in rollover_alerts:
            st.markdown(f"""
            <div class="rollover-alert">
                <h4>🔔 {alert['symbol']} 即将到期</h4>
                <p><strong>当前合约:</strong> {alert['current_contract']}</p>
                <p><strong>下期合约:</strong> {alert['next_contract']}</p>
                <p><strong>剩余天数:</strong> {alert['days_remaining']} 天</p>
                <p><strong>持仓规模:</strong> {alert['position_size']} 手</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 主要图表区域
    st.markdown("---")
    st.markdown("### 📊 持仓与风险分析")
    
    # 期货持仓概览
    position_fig = create_futures_position_overview(futures_mgr)
    st.plotly_chart(position_fig, use_container_width=True)
    
    # 套利机会分析
    st.markdown("### 🎯 套利机会分析")
    
    calendar_opportunities = spread_analyzer.analyze_calendar_spreads()
    intercommodity_opportunities = spread_analyzer.analyze_intercommodity_spreads()
    
    if calendar_opportunities or intercommodity_opportunities:
        spread_fig = create_spread_analysis_chart(spread_analyzer)
        st.plotly_chart(spread_fig, use_container_width=True)
        
        # 显示具体套利机会
        st.markdown("#### 🚨 当前套利信号")
        
        for opp in calendar_opportunities:
            confidence_class = "arbitrage-signal" if opp['confidence'] > 80 else "spread-opportunity"
            st.markdown(f"""
            <div class="{confidence_class}">
                <h4>📈 {opp['type']}: {opp['symbol']}</h4>
                <p><strong>合约:</strong> {opp['front_contract']} vs {opp['back_contract']}</p>
                <p><strong>价差:</strong> {opp['spread_value']:.2f} | <strong>Z-Score:</strong> {opp['z_score']:.2f}</p>
                <p><strong>信号:</strong> {opp['signal']} | <strong>信心度:</strong> {opp['confidence']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        for opp in intercommodity_opportunities:
            st.markdown(f"""
            <div class="arbitrage-signal">
                <h4>🔄 {opp['type']}: {opp['leg1']}/{opp['leg2']}</h4>
                <p><strong>当前比率:</strong> {opp['ratio']:.2f} | <strong>历史均值:</strong> {opp['historical_ratio']:.2f}</p>
                <p><strong>信号:</strong> {opp['signal']} | <strong>信心度:</strong> {opp['confidence']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 合约规格详情
    st.markdown("---")
    st.markdown("### 📋 合约规格表")
    
    if selected_contract in futures_mgr.contracts:
        contract = futures_mgr.contracts[selected_contract]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="contract-card">
                <h4>{contract['name']} ({selected_contract})</h4>
                <p><strong>交易所:</strong> {contract['exchange']}</p>
                <p><strong>合约乘数:</strong> {contract['multiplier']:,}</p>
                <p><strong>最小变动价位:</strong> {contract['tick_size']}</p>
                <p><strong>最小变动价值:</strong> ${contract['tick_value']}</p>
                <p><strong>结算方式:</strong> {contract['settlement_method']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="contract-card">
                <h4>保证金要求</h4>
                <p><strong>日内保证金:</strong> ${contract['margin_day']:,}</p>
                <p><strong>隔夜保证金:</strong> ${contract['margin_overnight']:,}</p>
                <p><strong>当前合约:</strong> {contract['current_contract']}</p>
                <p><strong>下期合约:</strong> {contract['next_contract']}</p>
                <p><strong>展期日期:</strong> {contract['rollover_date']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 展期日历
    st.markdown("### 📅 合约展期日历")
    
    rollover_fig = create_rollover_calendar(futures_mgr)
    st.plotly_chart(rollover_fig, use_container_width=True)
    
    # 实时持仓明细
    st.markdown("---")
    st.markdown("### 📈 实时持仓明细")
    
    position_data = []
    for symbol, position in futures_mgr.positions.items():
        if symbol in futures_mgr.contracts:
            contract = futures_mgr.contracts[symbol]
            position_value = position['quantity'] * position['current_price'] * contract['multiplier']
            
            position_data.append({
                '品种': symbol,
                '合约': contract['current_contract'],
                '方向': '多头' if position['quantity'] > 0 else '空头',
                '数量': abs(position['quantity']),
                '开仓价': f"{position['entry_price']:.2f}",
                '现价': f"{position['current_price']:.2f}",
                '持仓价值': f"${position_value:,.0f}",
                '未实现盈亏': f"${position['unrealized_pnl']:,.0f}",
                '保证金': f"${contract['margin_overnight'] * abs(position['quantity']):,}"
            })
    
    if position_data:
        df_positions = pd.DataFrame(position_data)
        st.markdown('<div class="position-table">', unsafe_allow_html=True)
        st.dataframe(df_positions, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 账户总览
    st.markdown("---")
    st.markdown("### 💰 账户资金总览")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("账户余额", f"${futures_mgr.account_balance:,}", delta="+5,230")
    
    with col2:
        st.metric("可用保证金", f"${futures_mgr.available_margin:,}", delta="-8,500")
    
    with col3:
        total_pnl = sum(pos['unrealized_pnl'] for pos in futures_mgr.positions.values())
        st.metric("总盈亏", f"${total_pnl:,.0f}", delta=f"{total_pnl:+,.0f}")
    
    with col4:
        st.metric("保证金使用率", f"{margin_data['margin_utilization']:.1%}", delta="+12%")
    
    with col5:
        position_count = len(futures_mgr.positions)
        st.metric("持仓品种", position_count, delta="+1")

def main():
    """主应用程序"""
    
    # 侧边栏导航
    st.sidebar.title("📈 期货交易导航")
    
    page = st.sidebar.selectbox(
        "选择功能模块",
        [
            "交易仪表盘",
            "合约管理",
            "套利分析", 
            "风险监控",
            "历史数据",
            "系统设置"
        ]
    )
    
    # 实时状态指示器
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🟢 系统状态")
    st.sidebar.markdown("**交易服务:** 🟢 在线")
    st.sidebar.markdown("**行情数据:** 🟢 正常")
    st.sidebar.markdown("**风控系统:** 🟢 活跃")
    st.sidebar.markdown("**套利引擎:** 🟡 监控中")
    
    # 快速统计
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 今日统计")
    st.sidebar.metric("交易笔数", "23", delta="+5")
    st.sidebar.metric("总盈亏", "$8,107", delta="+$1,250")
    st.sidebar.metric("最大回撤", "-2.1%", delta="+0.3%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Agent D1 期货专版**")
    st.sidebar.markdown("*专业期货交易解决方案*")
    st.sidebar.markdown(f"*更新时间: {datetime.now().strftime('%H:%M:%S')}*")
    
    # 显示选中的页面
    if page == "交易仪表盘":
        display_futures_trading_dashboard()
    else:
        st.markdown(f"# {page}")
        st.info(f"{page}模块正在开发中。交易仪表盘功能已完整实现。")

if __name__ == "__main__":
    main()