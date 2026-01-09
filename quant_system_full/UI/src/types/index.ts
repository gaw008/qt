// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
}

// Market Data Types
export interface Asset {
  symbol: string;
  name: string;
  type: 'stock' | 'etf' | 'future' | 'reit' | 'adr';
  sector?: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
  last_update: string;
}

export interface HeatmapData {
  symbol: string;
  name: string;
  sector: string;
  change_percent_1d: number;
  change_percent_1w: number;
  change_percent_1m: number;
  volume_change: number;
  market_cap: number;
  price: number;
}

// Position Types
export interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  realized_pnl?: number;
  stop_loss?: number;
  take_profit?: number;
  entry_time: string;
  last_update: string;
}

// Order Types
export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stop_price?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected' | 'partially_filled';
  filled_quantity?: number;
  avg_fill_price?: number;
  created_at: string;
  updated_at: string;
  cost_analysis?: CostAnalysis;
}

// Cost Analysis Types
export interface CostAnalysis {
  total_cost: number;
  cost_basis_points: number;
  recommended_order_type: string;
  market_data: {
    symbol: string;
    price: number;
    spread: number;
    volume: number;
  };
}

// Market State Types
export interface MarketState {
  status: string;
  market_trend?: number;
  volatility?: number;
  volume_ratio?: number;
  fear_greed_index?: number;
  next_open?: string;
  regime?: string;
  risk_level?: 'low' | 'medium' | 'high';
  market_interpretation?: string;
  current_state?: 'bull_market' | 'bear_market' | 'sideways_market' | 'high_volatility' | 'crisis_mode';
  state_duration?: number;
  parameters?: {
    position_size_multiplier: number;
    risk_threshold_multiplier: number;
    stop_loss_multiplier: number;
    max_positions: number;
    volatility_target: number;
  };
  factor_weights?: {
    valuation: number;
    momentum: number;
    technical: number;
    volume: number;
    market_sentiment: number;
  };
  latest_signals?: {
    volatility_percentile: number;
    momentum_score: number;
    sentiment_score: number;
    market_stress: number;
    timestamp: string;
  };
}

// Alert Types
export interface Alert {
  id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  status: 'active' | 'resolved' | 'acknowledged';
  title: string;
  message: string;
  symbol?: string;
  price_change?: number;
  context?: Record<string, any>;
  acknowledged?: boolean;
  resolved?: boolean;
}

// Portfolio Summary Types
export interface PortfolioSummary {
  total_value: number;
  total_pnl: number;
  total_pnl_percent: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  positions_count: number;
  cash_balance: number;
  buying_power: number;
  margin_used: number;
  risk_metrics: {
    portfolio_beta: number;
    sharpe_ratio: number;
    max_drawdown: number;
    volatility: number;
  };
}

// Filter Types
export interface AssetFilter {
  asset_types?: string[];
  sectors?: string[];
  min_market_cap?: number;
  max_market_cap?: number;
  min_volume?: number;
  max_volume?: number;
  min_price?: number;
  max_price?: number;
  min_change?: number;
  max_change?: number;
}

// Chart Data Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  volume?: number;
}

export interface PerformanceData {
  timestamp: string;
  portfolio_value: number;
  benchmark_value: number;
  daily_return: number;
  cumulative_return: number;
}

// Stock Selection Types
export interface StockSelectionItem {
  symbol: string;
  score?: number;
  rank?: number;
  action?: string;
  confidence?: number;
  metrics?: Record<string, any>;
  component_scores?: Record<string, number>;
}

export interface StockSelectionResponse {
  stocks: StockSelectionItem[];
  timestamp?: string | null;
  strategy?: string;
  total_analyzed?: number;
  execution_time?: number;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: 'price_update' | 'position_update' | 'order_update' | 'alert' | 'market_state';
  data: any;
  timestamp: string;
}
