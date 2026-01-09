-- ============================================
-- Quant Trading System - Supabase Schema
-- Run this in Supabase SQL Editor
-- ============================================

-- Enum types
CREATE TYPE order_side AS ENUM ('BUY', 'SELL');
CREATE TYPE order_status AS ENUM ('PENDING', 'FILLED', 'PARTIAL', 'CANCELLED', 'REJECTED');
CREATE TYPE run_type AS ENUM ('trading', 'selection', 'monitoring', 'ai_training', 'compliance', 'factor_crowding', 'daily_close');

-- ============================================
-- CORE TABLES (Hot Data)
-- ============================================

-- Task runs (hot: 30 days)
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type run_type NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_ms INTEGER,
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_runs_started_at ON runs(started_at DESC);
CREATE INDEX idx_runs_type ON runs(run_type);

-- Orders (hot: 90 days)
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    side order_side NOT NULL,
    order_type VARCHAR(20) DEFAULT 'MARKET',
    quantity INTEGER NOT NULL,
    price DECIMAL(12,4),
    stop_price DECIMAL(12,4),
    status order_status NOT NULL DEFAULT 'PENDING',
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(12,4),
    commission DECIMAL(10,4),
    run_id UUID REFERENCES runs(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX idx_orders_status ON orders(status);

-- Fills (hot: 90 days)
CREATE TABLE fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id) ON DELETE CASCADE,
    fill_price DECIMAL(12,4) NOT NULL,
    fill_quantity INTEGER NOT NULL,
    commission DECIMAL(10,4),
    filled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_fills_order_id ON fills(order_id);
CREATE INDEX idx_fills_filled_at ON fills(filled_at DESC);

-- Positions snapshot (hot: 7 days)
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost DECIMAL(12,4) NOT NULL,
    market_price DECIMAL(12,4),
    market_value DECIMAL(14,2),
    unrealized_pnl DECIMAL(12,2),
    position_type VARCHAR(20) DEFAULT 'REAL',
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_snapshot_at ON positions(snapshot_at DESC);

-- Strategy configuration (permanent)
CREATE TABLE strategy_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(50) NOT NULL UNIQUE,
    parameters JSONB NOT NULL DEFAULT '{}',
    weights JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Real-time metrics (hot: 7 days)
CREATE TABLE metrics_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metrics JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_metrics_recorded_at ON metrics_snapshots(recorded_at DESC);

-- Selection results (hot: 30 days)
CREATE TABLE selection_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES runs(id),
    strategy_type VARCHAR(50),
    top_picks JSONB NOT NULL,
    total_evaluated INTEGER,
    selected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_selection_selected_at ON selection_results(selected_at DESC);

-- Archival tracking (permanent)
CREATE TABLE archive_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(50) NOT NULL,
    records_archived INTEGER NOT NULL,
    archive_file VARCHAR(255),
    archived_before TIMESTAMPTZ NOT NULL,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================
-- ANALYSIS & IMPROVEMENT TABLES
-- ============================================

-- Trade signals (hot: 90 days)
CREATE TABLE trade_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES runs(id),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    score DECIMAL(6,2) NOT NULL,
    component_scores JSONB DEFAULT '{}',
    price_at_signal DECIMAL(12,4) NOT NULL,
    volume_at_signal BIGINT,
    market_cap DECIMAL(18,2),
    sector VARCHAR(50),
    reasoning TEXT,
    was_executed BOOLEAN DEFAULT false,
    order_id UUID REFERENCES orders(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_signals_symbol ON trade_signals(symbol);
CREATE INDEX idx_signals_strategy ON trade_signals(strategy_name);
CREATE INDEX idx_signals_created_at ON trade_signals(created_at DESC);
CREATE INDEX idx_signals_was_executed ON trade_signals(was_executed);

-- Execution analysis (hot: 90 days)
CREATE TABLE execution_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id) ON DELETE CASCADE,
    signal_price DECIMAL(12,4),
    arrival_price DECIMAL(12,4),
    fill_price DECIMAL(12,4),
    vwap_price DECIMAL(12,4),
    twap_price DECIMAL(12,4),
    slippage_bps DECIMAL(8,2),
    market_impact_bps DECIMAL(8,2),
    implementation_shortfall DECIMAL(12,4),
    execution_duration_ms INTEGER,
    market_volatility DECIMAL(8,4),
    spread_bps DECIMAL(8,2),
    volume_participation DECIMAL(6,4),
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_exec_order_id ON execution_analysis(order_id);
CREATE INDEX idx_exec_analyzed_at ON execution_analysis(analyzed_at DESC);

-- Daily performance (hot: 365 days)
CREATE TABLE daily_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL UNIQUE,
    starting_equity DECIMAL(14,2) NOT NULL,
    ending_equity DECIMAL(14,2) NOT NULL,
    daily_pnl DECIMAL(12,2) NOT NULL,
    daily_return DECIMAL(8,6),
    cumulative_return DECIMAL(10,6),
    drawdown DECIMAL(8,6),
    max_drawdown DECIMAL(8,6),
    sharpe_ratio DECIMAL(6,3),
    sortino_ratio DECIMAL(6,3),
    trades_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    avg_win DECIMAL(12,2),
    avg_loss DECIMAL(12,2),
    profit_factor DECIMAL(6,3),
    sector_pnl JSONB DEFAULT '{}',
    strategy_pnl JSONB DEFAULT '{}',
    risk_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_daily_perf_date ON daily_performance(date DESC);

-- Market regime snapshots (hot: 365 days)
CREATE TABLE market_regimes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime_type VARCHAR(30) NOT NULL,
    vix_level DECIMAL(6,2),
    market_trend VARCHAR(20),
    sector_rotation JSONB DEFAULT '{}',
    breadth_advance_decline DECIMAL(6,3),
    fear_greed_index INTEGER,
    yield_curve_slope DECIMAL(6,4),
    dollar_index DECIMAL(8,3),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_regime_detected_at ON market_regimes(detected_at DESC);
CREATE INDEX idx_regime_type ON market_regimes(regime_type);

-- AI/ML training history (permanent)
CREATE TABLE ai_training_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    training_started_at TIMESTAMPTZ NOT NULL,
    training_ended_at TIMESTAMPTZ,
    training_duration_seconds INTEGER,
    hyperparameters JSONB NOT NULL DEFAULT '{}',
    training_data_range TSTZRANGE,
    training_samples INTEGER,
    validation_samples INTEGER,
    train_loss DECIMAL(10,6),
    val_loss DECIMAL(10,6),
    train_accuracy DECIMAL(5,4),
    val_accuracy DECIMAL(5,4),
    sharpe_backtest DECIMAL(6,3),
    max_drawdown_backtest DECIMAL(6,4),
    live_sharpe DECIMAL(6,3),
    live_accuracy DECIMAL(5,4),
    days_in_production INTEGER DEFAULT 0,
    improvement_vs_previous DECIMAL(6,4),
    is_current_best BOOLEAN DEFAULT false,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_ai_model_name ON ai_training_history(model_name);
CREATE INDEX idx_ai_created_at ON ai_training_history(created_at DESC);

-- Strategy performance tracking (hot: 365 days)
CREATE TABLE strategy_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    signals_generated INTEGER DEFAULT 0,
    signals_executed INTEGER DEFAULT 0,
    hit_rate DECIMAL(5,4),
    avg_return DECIMAL(8,6),
    total_pnl DECIMAL(12,2),
    sharpe_contribution DECIMAL(6,3),
    drawdown_contribution DECIMAL(6,4),
    sector_exposure JSONB DEFAULT '{}',
    factor_exposure JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(strategy_name, date)
);
CREATE INDEX idx_strat_perf_name ON strategy_performance(strategy_name);
CREATE INDEX idx_strat_perf_date ON strategy_performance(date DESC);

-- Compliance events (permanent)
CREATE TABLE compliance_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(30) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    symbol VARCHAR(20),
    position_value DECIMAL(14,2),
    limit_value DECIMAL(14,2),
    actual_value DECIMAL(14,2),
    breach_percentage DECIMAL(6,4),
    was_prevented BOOLEAN DEFAULT false,
    action_taken TEXT,
    resolved_at TIMESTAMPTZ,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_compliance_type ON compliance_events(event_type);
CREATE INDEX idx_compliance_detected_at ON compliance_events(detected_at DESC);
CREATE INDEX idx_compliance_severity ON compliance_events(severity);

-- Factor crowding history (hot: 90 days)
CREATE TABLE factor_crowding_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_name VARCHAR(30) NOT NULL,
    hhi DECIMAL(6,4),
    gini_coefficient DECIMAL(6,4),
    crowding_score DECIMAL(6,2),
    crowding_level VARCHAR(20),
    portfolio_exposure DECIMAL(6,4),
    market_exposure DECIMAL(6,4),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_factor_crowding_at ON factor_crowding_history(recorded_at DESC);
CREATE INDEX idx_factor_name ON factor_crowding_history(factor_name);

-- ============================================
-- INTRADAY TRADING ANALYSIS TABLES
-- For minute-level high-frequency strategy tuning
-- ============================================

-- Intraday signals with component scores (hot: 30 days)
CREATE TABLE intraday_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES runs(id),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    score DECIMAL(6,4),
    component_scores JSONB DEFAULT '{}',
    fast_ema DECIMAL(12,4),
    slow_ema DECIMAL(12,4),
    momentum_pct DECIMAL(8,6),
    volume_ratio DECIMAL(8,4),
    atr DECIMAL(12,4),
    price_at_signal DECIMAL(12,4),
    volume_at_signal BIGINT,
    spread_bps DECIMAL(8,2),
    data_freshness_seconds INTEGER,
    data_coverage_pct DECIMAL(5,4),
    was_executed BOOLEAN DEFAULT FALSE,
    order_id VARCHAR(50),
    exit_price DECIMAL(12,4),
    exit_reason VARCHAR(50),
    pnl_bps DECIMAL(10,2),
    hold_duration_minutes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_intraday_signals_symbol_time ON intraday_signals(symbol, created_at DESC);
CREATE INDEX idx_intraday_signals_type ON intraday_signals(signal_type);
CREATE INDEX idx_intraday_signals_run ON intraday_signals(run_id);

-- Intraday execution quality analysis (hot: 30 days)
CREATE TABLE intraday_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID REFERENCES intraday_signals(id),
    order_id VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    signal_price DECIMAL(12,4),
    limit_price DECIMAL(12,4),
    fill_price DECIMAL(12,4),
    vwap_5min DECIMAL(12,4),
    slippage_bps DECIMAL(8,2),
    market_impact_bps DECIMAL(8,2),
    execution_time_ms INTEGER,
    order_quantity INTEGER,
    filled_quantity INTEGER,
    volume_participation_pct DECIMAL(6,4),
    commission DECIMAL(10,4),
    slippage_cost DECIMAL(10,4),
    total_cost DECIMAL(10,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_intraday_exec_symbol ON intraday_executions(symbol, created_at DESC);
CREATE INDEX idx_intraday_exec_signal ON intraday_executions(signal_id);

-- Intraday risk snapshots (hot: 7 days)
CREATE TABLE intraday_risk_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES runs(id),
    equity DECIMAL(14,2),
    day_start_equity DECIMAL(14,2),
    daily_pnl DECIMAL(12,2),
    daily_loss_pct DECIMAL(8,6),
    positions_count INTEGER,
    total_position_value DECIMAL(14,2),
    cash_balance DECIMAL(14,2),
    buying_power DECIMAL(14,2),
    max_position_weight DECIMAL(6,4),
    es_97_5 DECIMAL(12,2),
    portfolio_beta DECIMAL(6,4),
    factor_hhi DECIMAL(6,4),
    daily_costs_total DECIMAL(10,4),
    daily_cost_pct DECIMAL(8,6),
    circuit_breaker_active BOOLEAN DEFAULT FALSE,
    halt_reason VARCHAR(100),
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_risk_snapshots_time ON intraday_risk_snapshots(snapshot_at DESC);
CREATE INDEX idx_risk_snapshots_run ON intraday_risk_snapshots(run_id);

-- Signal performance aggregation (hot: 365 days)
CREATE TABLE signal_performance_daily (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    symbol VARCHAR(20),
    total_signals INTEGER DEFAULT 0,
    buy_signals INTEGER DEFAULT 0,
    sell_signals INTEGER DEFAULT 0,
    executed_signals INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    avg_win_bps DECIMAL(10,2),
    avg_loss_bps DECIMAL(10,2),
    profit_factor DECIMAL(8,3),
    avg_score DECIMAL(6,4),
    score_vs_outcome_corr DECIMAL(6,4),
    ema_contribution DECIMAL(6,4),
    momentum_contribution DECIMAL(6,4),
    volume_contribution DECIMAL(6,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(date, symbol)
);
CREATE INDEX idx_signal_perf_date ON signal_performance_daily(date DESC);
CREATE INDEX idx_signal_perf_symbol ON signal_performance_daily(symbol);

-- ============================================
-- SUCCESS MESSAGE
-- ============================================
-- Schema created successfully!
-- Total tables: 20
-- Core tables: 8 (runs, orders, fills, positions, strategy_configs, metrics_snapshots, selection_results, archive_log)
-- Analysis tables: 8 (trade_signals, execution_analysis, daily_performance, market_regimes, ai_training_history, strategy_performance, compliance_events, factor_crowding_history)
-- Intraday tables: 4 (intraday_signals, intraday_executions, intraday_risk_snapshots, signal_performance_daily)
