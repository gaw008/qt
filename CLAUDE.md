# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repository contains a quantitative trading system with the following structure:
- `quant_system_full/` - Main quantitative trading system implementation
- `.vscode/` - VSCode configuration files
- `GEMINI.md` - Alternative AI assistant documentation

## Main System Overview (quant_system_full)

An **investment-grade** multi-factor quantitative trading system with institutional-quality risk management, three-phase backtesting validation (2006-2025), and live trading capabilities via Tiger Brokers API. The system features Expected Shortfall (ES) @ 97.5% risk management, adaptive execution algorithms, factor crowding detection, and comprehensive compliance monitoring.

### Core Components

- **bot/** - Trading bot core with multi-factor analysis and Tiger API integration
- **dashboard/** - Web-based monitoring and control system (FastAPI backend + Streamlit legacy interface)
- **UI/** - Modern React + TypeScript frontend (primary trading interface)
- **openapi-java-sdk/** - Java SDK for Tiger Brokers OpenAPI
- **openapi-docs/** - API documentation
- **scripts/** - Utility scripts for system management
- **data_cache/** - Cached market data storage
- **reports/** - Trading reports and analysis output
- **yahoo-finance-server/** - Yahoo Finance data server integration

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
# Windows
start_all.bat

# Cross-platform
python start_all.py
```

#### Manual Component Startup
```bash
# React Frontend (Primary Interface) - Port 3000/5173
python start_react_ui.py
# OR: cd UI && npm install && npm run dev

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

# Ultra System (Enhanced)
python start_ultra_system.py
```

### Testing Commands
```bash
# Core System Tests
python test_system_integration.py
python test_tiger_system.py
python test_scoring_system.py
python test_data_management.py
python test_portfolio_integration.py
python test_risk_systems.py

# Performance Tests
python test_optimization.py
python test_gpu_performance.py
python standalone_performance_test.py

# Multi-Asset Tests
python test_multi_asset_integration.py
python test_30k_integration.py

# Trading Tests
python test_automated_trading.py
python test_trading_system.py
python test_tiger_order.py  # CAUTION: May place real orders

# Data Quality
python test_data_quality.py
python test_datacache_integration.py

# Backtesting
python backtest.py --symbol AAPL --short 5 --long 20 --csv test_data.csv
```

### System Status and Monitoring
```bash
# System status check
python system_status_check.py
python system_check.py

# Check Tiger account
python check_tiger_account.py
python check_account_positions.py
python check_account_proper.py

# Monitor deployments
python deploy_ultra_system.py
tail -f deployment.log
```

### Quick Testing
```bash
python quick_test.py
python simple_test.py
python quick_start.py
```

## High-Level Architecture

### Data Flow Pipeline

#### 1. Market Data Acquisition Layer
- **Multi-Source Integration**: Yahoo Finance API → Yahoo Finance MCP → Tiger SDK → Fallback
- **Data Cache System**: Persistent storage in `data_cache/` for offline development
- **Real-time Processing**: `bot/realtime_data_processor_c1.py` for live market feeds
- **Batch Processing**: Optimized for handling 4000+ stocks simultaneously

#### 2. Multi-Factor Analysis Engine
- **Factor Categories** (60+ indicators):
  - Valuation: P/E, P/B, EV/EBITDA, industry-normalized metrics
  - Volume: OBV, VWAP, Money Flow Index, accumulation patterns
  - Momentum: RSI, ROC, Stochastic, price/volume momentum
  - Technical: MACD, Bollinger Bands, ADX, breakout patterns
  - Market Sentiment: VIX, breadth, sector rotation metrics

#### 3. Intelligent Stock Selection
- **Sector Management**: Pre-defined universes (Technology, Healthcare, Financial, etc.)
- **Scoring Engine**: Weighted multi-factor composite scoring with normalization
- **Selection Strategies**:
  - Value Momentum: Undervaluation + positive momentum
  - Technical Breakout: Resistance breaks with volume
  - Earnings Momentum: Growth and earnings surprises
- **Risk Filters**: Market cap, liquidity, volatility constraints

#### 4. Investment-Grade Portfolio Management
- **Multi-Stock Tracking**: Handles portfolios of 20+ positions with institutional controls
- **ES@97.5% Risk Management**: Expected Shortfall replaces VaR for tail risk measurement
- **Position Sizing**: Risk-based allocation with drawdown budgeting and capacity analysis
- **Rebalancing**: Automated periodic optimization with transaction cost modeling
- **Performance Monitoring**: Real-time P&L, ES@97.5%, factor crowding, and attribution analysis
- **Compliance Monitoring**: Automated regulatory compliance with 8+ core rules

#### 5. Adaptive Trade Execution
- **Tiger API Integration**: Direct connection to Tiger Brokers with enhanced monitoring
- **Smart Execution**: Adaptive participation rate optimization based on market conditions
- **Order Types**: Market, limit, stop-loss with Implementation Shortfall analysis
- **Cost Attribution**: Real-time transaction cost analysis vs VWAP/TWAP/arrival price
- **Execution Engine**: `bot/adaptive_execution_engine.py` with market impact modeling
- **Transaction Logging**: Complete audit trail with cost breakdown and performance attribution

#### 6. Investment-Grade Control & Monitoring
- **React Frontend**: Modern trading UI at http://localhost:3000+ with real-time ES@97.5% monitoring
- **Streamlit Dashboard**: Management interface at http://localhost:8501 with compliance alerts
- **API Backend**: RESTful API at http://localhost:8000 with institutional endpoints
- **WebSocket Updates**: Real-time data via WS at ws://localhost:8000/ws including risk metrics
- **Kill Switch**: Emergency stop with compliance notifications and audit logging
- **Real-time Monitor**: `bot/real_time_monitor.py` with 17 institutional-quality metrics
- **Health Monitoring**: System diagnostics, self-healing, and factor crowding detection
- **Alert System**: `bot/intelligent_alert_system_c1.py` with compliance escalation

### React Frontend Architecture

The modern React + TypeScript frontend provides a comprehensive trading interface with the following structure:

#### Page Structure
- **Index** (`/`) - Dashboard overview with portfolio summary
- **Markets** (`/markets`) - Market data, heatmaps, and real-time quotes
- **Trading** (`/trade`) - Order placement, position management, and execution
- **Risk** (`/risk`) - Risk metrics, ES@97.5% calculations, factor crowding, and exposure analysis
- **AI Center** (`/ai`) - AI training status, model performance, and analytics
- **Strategies** (`/strategies`) - Strategy selection, weights, and performance
- **Screener** (`/screener`) - Stock screening with multi-factor criteria

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
SELECTION_RESULT_SIZE=20

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

## Advanced Features

### Investment-Grade Risk Management & Validation
- **Enhanced Risk Manager**: `bot/enhanced_risk_manager.py` - ES@97.5% with tail dependence analysis
- **Transaction Cost Analyzer**: `bot/transaction_cost_analyzer.py` - Market impact & capacity modeling
- **Purged K-Fold Validator**: `bot/purged_kfold_validator.py` - Time series cross-validation
- **Factor Crowding Detector**: `bot/factor_crowding_detector.py` - HHI, Gini, correlation analysis
- **Adaptive Execution Engine**: `bot/adaptive_execution_engine.py` - Smart participation rates
- **Real-time Monitor**: `bot/real_time_monitor.py` - 17 institutional metrics
- **Compliance System**: `bot/compliance_monitoring_system.py` - Automated regulatory compliance
- **Backtesting Framework**: Three-phase validation (2006-2016, 2017-2020, 2021-2025)

### AI/ML Integration
- **AI Learning Engine**: `bot/ai_learning_engine.py`
- **Strategy Optimizer**: `bot/ai_strategy_optimizer.py`
- **GPU Pipeline**: `bot/gpu_training_pipeline.py`, `setup_gpu.py`
- **Reinforcement Learning**: `bot/reinforcement_learning_framework.py`
- **Feature Engineering**: `bot/feature_engineering.py`

### Enhanced Systems
- **Agent C1 System**: `start_agent_c1_system.py` - Advanced intelligent trading
- **Ultra System**: `start_ultra_system.py`, `demo_ultra_system.py` - High-performance trading
- **Self-Healing**: `system_self_healing.py` - Automatic error recovery
- **Health Monitoring**: `system_health_monitoring.py` - Comprehensive diagnostics

### Multi-Asset Support
- **ETF Manager**: `bot/etf_manager.py`
- **Futures Manager**: `bot/futures_manager.py`
- **REITs/ADR Manager**: `bot/reits_adr_manager.py`
- **Cross-Asset Arbitrage**: `bot/cross_asset_arbitrage.py`

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