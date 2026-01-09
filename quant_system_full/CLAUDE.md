# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repository contains a quantitative trading system with the following structure:
- `quant_system_full/` - Main quantitative trading system implementation
- `.vscode/` - VSCode configuration files
- `GEMINI.md` - Alternative AI assistant documentation

## Main System Overview (quant_system_full)

A comprehensive multi-factor quantitative trading system with live trading capabilities via Tiger Brokers API, intelligent stock selection, and automated portfolio management.

### Core Components

- **bot/** - Trading bot core with multi-factor analysis and Tiger API integration
- **dashboard/** - Web-based monitoring and control system (FastAPI backend + Streamlit legacy interface)
- **UI/** - Modern React + TypeScript frontend (primary trading interface)
- **openapi-docs/** - API documentation
- **scripts/** - Utility scripts for system management
- **data_cache/** - Cached market data storage
- **reports/** - Trading reports and analysis output

## Common Development Commands

### Build and Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install Python dependencies
pip install -r bot/requirements.txt
pip install git+https://github.com/tigerbrokers/openapi-python-sdk.git

# Configure environment
cp config.example.env .env
# Edit .env with your Tiger API credentials and settings
```

### Running the System

#### One-Click Startup (Recommended)
```bash
# Complete System Launch (All Components)
python start_all.py

# Standalone Trading Bot (AI/ML Enhanced)
python start_bot.py

# Ultra High-Performance Mode (GPU Accelerated)
python start_ultra_system.py

# Advanced AI-Driven Trading (Agent C1)
python start_agent_c1_system.py
```

#### Manual Component Startup
```bash
# React Frontend (Primary Interface) - Port 3000/5173
cd UI && npm install && npm run dev

# Backend API
cd dashboard/backend
uvicorn app:app --host 0.0.0.0 --port 8000

# Worker Process
cd dashboard/worker
python runner.py

# Streamlit Management Interface (Legacy) - Port 8501
cd dashboard/frontend
streamlit run streamlit_app.py

# Bot Trading System
python start_bot.py
```

### Testing Commands
```bash
# Core System Tests
python test_system_integration.py
python test_scoring_system.py
python test_data_management.py
python test_portfolio_integration.py
python test_risk_systems.py

# Performance Tests
python test_full_system.py
python test_multi_factor_trading.py

# Trading Tests
python test_trading_system.py
python test_tiger_order.py  # CAUTION: May place real orders

# Quick Testing
python quick_test.py
python simple_test.py

# Backtesting
python backtest.py --symbol AAPL --short 5 --long 20 --csv test_data.csv
```

### System Status and Monitoring
```bash
# Comprehensive System Health Monitoring
python system_health_monitoring.py

# Intelligent System Self-Healing
python system_self_healing.py

# System status check
python system_status_check.py
python system_check.py

# Check Tiger account
python check_tiger_account.py
python check_account_positions.py

# Monitor deployments
python deploy_system.py
tail -f deployment.log
```

## High-Level Architecture

### Data Flow Pipeline

#### 1. Market Data Acquisition Layer
- **Multi-Source Integration**: Yahoo Finance API → Yahoo Finance MCP → Tiger SDK → Fallback
- **Data Cache System**: Persistent storage in `data_cache/` for offline development
- **Real-time Processing**: Live market feeds for 4000+ stocks simultaneously
- **Batch Processing**: Optimized for handling large stock universes

#### 2. Multi-Factor Analysis Engine
- **Factor Categories** (60+ indicators):
  - Valuation: P/E, P/B, EV/EBITDA, industry-normalized metrics
  - Volume: OBV, VWAP, Money Flow Index, accumulation patterns
  - Momentum: RSI, ROC, Stochastic, price/volume momentum
  - Technical: MACD, Bollinger Bands, ADX, breakout patterns
  - Market Sentiment: VIX, breadth, sector rotation metrics

#### 3. Intelligent Stock Selection
- **Sector Management**: Pre-defined universes (Technology, Healthcare, Financial, etc.)
- **Scoring Engine**: Weighted multi-factor composite scoring with normalization (20-100 range)
- **Selection Strategies**:
  - Value Momentum: Undervaluation + positive momentum
  - Technical Breakout: Resistance breaks with volume
  - Earnings Momentum: Growth and earnings surprises
- **Quality Filter**: Only selects stocks with avg_score >= 85 points
- **Dynamic Selection**: Selects up to 10 stocks that meet quality criteria (not fixed count)
- **Risk Filters**: Market cap, liquidity, volatility constraints

#### 4. Portfolio Management
- **Multi-Stock Tracking**: Handles portfolios of 20+ positions
- **Position Sizing**: Risk-based allocation algorithms
- **Rebalancing**: Automated periodic portfolio optimization
- **Performance Monitoring**: Real-time P&L and risk metrics

#### 5. Trade Execution
- **Tiger API Integration**: Direct connection to Tiger Brokers
- **Order Types**: Market, limit, stop-loss orders
- **Execution Engine**: `bot/execution_tiger.py` with retry logic
- **Transaction Logging**: Complete audit trail of all trades

#### 6. System Control & Monitoring
- **React Frontend**: Modern trading UI at http://localhost:3000+
- **Streamlit Dashboard**: Management interface at http://localhost:8501
- **API Backend**: RESTful API at http://localhost:8000
- **WebSocket Updates**: Real-time data via WS at ws://localhost:8000/ws
- **Kill Switch**: Emergency stop functionality
- **Health Monitoring**: System diagnostics and self-healing
- **Alert System**: `bot/intelligent_alert_system_c1.py`

### React Frontend Architecture

The modern React + TypeScript frontend provides a comprehensive trading interface with the following structure:

#### Page Structure
- **Index** (`/`) - Dashboard overview with portfolio summary
- **Markets** (`/markets`) - Market data, heatmaps, and real-time quotes
- **Trading** (`/trade`) - Order placement, position management, and execution
- **Risk** (`/risk`) - Risk metrics, VaR calculations, and exposure analysis

#### Technology Stack
- **React 18** with TypeScript for type safety
- **Vite** for fast development and hot reloading
- **TanStack Query** for server state management
- **React Router** for client-side routing
- **Radix UI** for accessible component primitives
- **Tailwind CSS** for styling and responsive design
- **Recharts** for financial charts and visualization
- **WebSocket** integration for real-time updates

#### API Integration
- **Base URL**: `http://localhost:8000` (configurable via API_CONFIG)
- **Authentication**: Bearer token authentication (`wgyjd0508`)
- **Endpoints**: Full REST API coverage for all trading operations
- **Error Handling**: Automatic retry with exponential backoff
- **Real-time**: WebSocket connection for live data updates

#### Development Commands
```bash
# Start development server
npm run dev          # Runs on http://localhost:3000 (or next available port)

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run typecheck
```

### Key Configuration Files

#### .env Configuration
```bash
# Tiger API Settings
TIGER_ID=20550012
ACCOUNT=41169270
PRIVATE_KEY_PATH=private_key.pem
DRY_RUN=true  # Set false for live trading

# Data Sources
DATA_SOURCE=auto  # auto|tiger|yahoo_api|yahoo_mcp
USE_MCP_TOOLS=true

# Market Settings
PRIMARY_MARKET=US
SELECTION_UNIVERSE_SIZE=4000
SELECTION_RESULT_SIZE=10          # Maximum stocks to select (dynamic based on score filter)
SELECTION_MIN_SCORE=80.0          # Minimum avg_score threshold (only select stocks >= 80)

# Scheduling (seconds)
SELECTION_TASK_INTERVAL=3600
TRADING_TASK_INTERVAL=30
MONITORING_TASK_INTERVAL=120

# Performance Tuning
BATCH_SIZE=1000
MAX_CONCURRENT_REQUESTS=10
```

### Critical File Locations

- **Configuration**: `config.example.env`, `.env`
- **Tiger API Config**: `props/tiger_openapi_config.properties`
- **Private Key**: `private_key.pem`
- **State Management**: `dashboard/state/`
- **Logs**: `deployment.log`, `dashboard/state/bot.log`
- **Reports**: `reports/`, various `*_REPORT.md` files
- **Documentation**: Multiple `*_GUIDE.md` files

## Important Notes

### Rules
- 当遇到问题需要解决问题，不能降级为简化版本
- 不能使用虚拟数据，一切都需要使用真实数据
- 当测试完之后，需要删除测试脚本，保证项目简洁
- 不要再文件中使用表情之类的特殊符号，避免Unicode error
- 每次更新完都需要更新到系统说明

### Security Considerations
- Never commit `.env` files or `private_key.pem`
- Use `ADMIN_TOKEN` for dashboard authentication
- Test with `DRY_RUN=true` before live trading
- Verify Tiger account credentials before production

### Performance Optimization
- System optimized for 4000+ stock universe
- Batch processing for efficiency
- Concurrent request handling
- Data caching to reduce API calls

### Testing Best Practices
- Always run integration tests before deployment
- Use `test_data.csv` for backtesting
- Monitor system health with status check scripts
- Validate Tiger API connection regularly

### Known Working Configuration
- Tiger Account: 41169270
- Tiger ID: 20550012
- Python: 3.11+ recommended
- All scripts use English output (no Unicode issues)

## Advanced System Management Scripts

### Professional System Launchers

#### start_all.py - Complete System Orchestration
Comprehensive one-click system startup with intelligent orchestration:
- **Environment Validation**: Pre-flight checks for dependencies and resources
- **Component Coordination**: Starts all system components in proper dependency order
- **Health Monitoring**: Real-time health checks and status reporting
- **Process Management**: Intelligent process lifecycle management
- **Graceful Shutdown**: Clean shutdown with resource cleanup
- **Error Recovery**: Automatic error detection and recovery
- **Resource Optimization**: Memory and CPU usage optimization

```bash
python start_all.py
# Features: React Frontend + FastAPI Backend + Streamlit Dashboard + Trading Bot + Health Monitoring
```

#### start_bot.py - Standalone AI/ML Trading Bot
Advanced trading bot with AI/ML integration:
- **AI Learning Engine**: Real-time machine learning model training
- **Strategy Optimization**: Automatic strategy parameter tuning
- **Enhanced Risk Management**: ES@97.5% risk calculations
- **Real-time Monitoring**: Performance metrics and trading analytics
- **Emergency Controls**: Instant stop functionality with safety checks
- **Multi-threaded Execution**: Parallel processing for optimal performance

```bash
python start_bot.py
# Features: AI/ML Models + Risk Management + Real-time Analytics + Emergency Controls
```

#### start_ultra_system.py - Ultra High-Performance Mode
Maximum performance trading system with advanced optimizations:
- **GPU Acceleration**: CUDA/OpenCL support for ML inference
- **Multi-core Optimization**: Intelligent CPU core utilization
- **Memory Management**: Advanced memory allocation and cleanup
- **Ultra-low Latency**: Optimized execution paths for speed
- **Performance Monitoring**: Real-time performance metrics and bottleneck detection
- **Resource Scaling**: Dynamic resource allocation based on market conditions

```bash
python start_ultra_system.py
# Features: GPU Acceleration + Multi-core Processing + Ultra-low Latency + Advanced Analytics
```

#### start_agent_c1_system.py - Advanced AI-Driven Trading
Most sophisticated AI trading system with intelligent agents:
- **AI Agent Ensemble**: Multiple specialized AI agents for different trading aspects
- **Reinforcement Learning**: Continuous strategy learning and adaptation
- **Sentiment Analysis**: Real-time news and social media sentiment processing
- **Feature Engineering**: Automatic feature discovery and selection
- **Model Evolution**: Automatic model architecture optimization
- **Predictive Analytics**: Advanced market prediction and trend analysis

```bash
python start_agent_c1_system.py
# Features: AI Agent Ensemble + Reinforcement Learning + Sentiment Analysis + Predictive Models
```

### Professional System Management

#### system_health_monitoring.py - Comprehensive Health Monitoring
Real-time system health monitoring with intelligent analysis:
- **Resource Monitoring**: CPU, memory, disk, and network usage tracking
- **Process Health**: Critical process monitoring and performance analysis
- **Service Availability**: API endpoint and service connectivity testing
- **Performance Trending**: Historical analysis and bottleneck identification
- **Predictive Alerts**: Early warning system for potential issues
- **Dashboard Interface**: Real-time health status dashboard

```bash
python system_health_monitoring.py
# Features: Real-time Monitoring + Predictive Analysis + Health Dashboard + Alert System
```

#### system_self_healing.py - Intelligent Auto-Recovery
Advanced self-healing system with automatic fault recovery:
- **Fault Detection**: AI-driven pattern recognition for system issues
- **Automatic Recovery**: Intelligent restart and repair strategies
- **Resource Optimization**: Automatic cleanup and resource management
- **Configuration Repair**: Automatic configuration validation and repair
- **Predictive Maintenance**: Proactive issue prevention
- **Incident Tracking**: Comprehensive incident logging and analysis

```bash
python system_self_healing.py
# Features: Auto-recovery + Predictive Maintenance + Configuration Repair + Incident Management
```

### System Management Architecture

#### Professional Features
- **Cross-platform Compatibility**: Windows, Linux, and macOS support
- **Professional Logging**: Structured logging with multiple output formats
- **Performance Optimization**: Intelligent resource management and optimization
- **Error Handling**: Comprehensive error recovery and reporting
- **Security**: Secure process management and credential handling
- **Scalability**: Support for high-frequency trading and large portfolios

#### Integration Points
- **Health Monitoring Integration**: All launchers integrate with health monitoring
- **Self-healing Integration**: Automatic recovery for all system components
- **Performance Tracking**: Real-time performance metrics and optimization
- **Alert System**: Comprehensive alerting and notification system
- **Process Coordination**: Intelligent process lifecycle management

## Intelligent Trading Decision System

Location: `bot/intelligent_trading_decision/`

A 4-layer architecture with 2 critical gates to reduce excessive trading costs.

### Architecture Layers
- **Layer 1**: Stock Selection (Daily) - 12-1 Month Momentum + Historical Win Rate
- **Layer 2**: Daily Regime (Daily) - VIX-based position sizing and threshold adjustment
- **Layer 3**: Signal Execution (Minute) - Directional scoring with stability, volume, price action
- **Layer 4**: Risk Control (System) - Cooldowns, daily limits, earnings blackout

### Critical Gates
- **Gate 1 (TriggerGate)**: Convert continuous signals to discrete events (key level crosses, volume shocks)
- **Gate 2 (CostBenefitGate)**: Only trade when Expected Edge > 2.5x Transaction Cost

### Trade History Modules (Phase 0)
Required for historical win rate calculation and system learning:

1. **trade_history_sync.py** - Historical trade sync from Tiger API
   - `TradeHistorySync.sync_from_tiger(days=90)` - Sync filled orders
   - Matches buy/sell pairs for round-trip P&L calculation

2. **trade_history_recorder.py** - Real-time trade recording
   - `TradeHistoryRecorder.record_entry()` - Record with decision context
   - `TradeHistoryRecorder.record_exit()` - Calculate P&L with FIX 15 (proper ID capture)
   - Direction-aware P&L (LONG vs SHORT)

3. **trade_history_analyzer.py** - Win rate calculations
   - `TradeHistoryAnalyzer.calculate_win_rate_score()` - Cold start handling
   - `TradeHistoryAnalyzer.get_symbol_performance()` - Per-symbol stats
   - `TradeHistoryAnalyzer.get_sector_stats()` - Sector-level aggregates

### Key Design Decisions
- All costs in $/share units for consistency
- Uses `bar_time` (not `datetime.now()`) for hold duration calculation
- Historical sync marked with `source='sync_from_tiger'`
- Real-time trades marked with `source='system'`

### Usage Example
```python
from bot.intelligent_trading_decision import (
    init_decision_system,
    filter_signals_through_decision_chain,
    get_trade_history_recorder,
)

# Initialize
decision_system = init_decision_system(data_provider=engine)
recorder = get_trade_history_recorder()

# Record entry with decision context
entry_id = recorder.record_entry(
    symbol='AAPL',
    action='BUY',
    quantity=100,
    fill_price=185.50,
    trade_time=bar_time,
    decision_context={
        'decision_score': 78.5,
        'gate1_reason': 'vwap_cross_up',
        'gate2_edge': 0.85,
        'gate2_cost': 0.32,
    }
)

# Record exit
exit_id = recorder.record_exit(
    entry_trade_id=entry_id,
    exit_price=188.20,
    exit_reason='Take profit',
    bar_time=bar_time
)
```

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.