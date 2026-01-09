## Strategy Catalog

### Strategy A - Open-Only Daily Rebalance
Recorded: 2026-01-04 16:34:12 -08:00

Purpose:
- Trade only at the market open based on prior close selection.
- No intraday monitoring or continuous rebalancing.

Execution Flow:
- During market closed hours, run stock selection.
- Select top picks, use AI to reorder.
- At next market open: sell existing positions, buy the planned set.
- No further trades until the next cycle.

Key Selection Parameters (from local .env at time of record):
- DEFAULT_SELECTION_STRATEGY: value_momentum
- SELECTION_UNIVERSE_SIZE: 5202
- SELECTION_RESULT_SIZE: 10
- SELECTION_MIN_SCORE: 80.0
- SELECTION_MIN_MARKET_CAP: 100000000
- SELECTION_MAX_MARKET_CAP: 5000000000000
- SELECTION_MIN_VOLUME: 50000
- SELECTION_MIN_PRICE: 1.0
- SELECTION_MAX_PRICE: 2000.0

Global Universe Filters (from local .env at time of record):
- MIN_MARKET_CAP: 100000000
- MAX_MARKET_CAP: 5000000000000
- MIN_DAILY_VOLUME: 50000
- MIN_STOCK_PRICE: 1.0
- MAX_STOCK_PRICE: 2000.0

### Strategy B - Intraday Coordinated (Selection + 5-Min Execution)
Recorded: 2026-01-04 16:35:50 -08:00

Purpose:
- Use daily selection to define a higher-quality candidate pool.
- Execute intraday 5-minute trend signals for dynamic entries, exits, and sizing.
- Enforce daily loss and transaction cost limits for frequent trading.

Execution Flow:
- During market closed hours, run selection and build candidate pool.
- Intraday watchlist = current positions + top picks (deduped, size limited by `INTRADAY_WATCHLIST_SIZE`, default 30).
- At market open + `open_buffer_minutes`, begin 5-minute signal loop.
- If `min_data_coverage` not met, skip the cycle to avoid forced liquidation.
- Generate target weights; apply cooldown and risk checks.
- Submit orders with de-duplication and track status; update daily cost totals.

Key Selection Parameters (current .env):
- DEFAULT_SELECTION_STRATEGY: value_momentum
- SELECTION_UNIVERSE_SIZE: 5202
- SELECTION_RESULT_SIZE: 30
- SELECTION_MIN_SCORE: 70.0
- SELECTION_MIN_MARKET_CAP: 1000000000
- SELECTION_MAX_MARKET_CAP: 5000000000000
- SELECTION_MIN_VOLUME: 1000000
- SELECTION_MIN_PRICE: 5.0
- SELECTION_MAX_PRICE: 2000.0

Global Universe Filters (current .env):
- MIN_MARKET_CAP: 1000000000
- MAX_MARKET_CAP: 5000000000000
- MIN_DAILY_VOLUME: 1000000
- MIN_STOCK_PRICE: 5.0
- MAX_STOCK_PRICE: 2000.0

Intraday Strategy Parameters (current intraday_strategy.json):
- signal_period: 5min
- lookback_bars: 120
- fast_ema: 9
- slow_ema: 21
- atr_period: 14
- trail_atr: 2.5
- hard_stop_atr: 3.0
- momentum_lookback: 6
- min_volume_ratio: 1.0
- entry_score_threshold: 0.6
- weight_power: 1.4
- max_positions: 10
- max_position_percent: 0.12
- min_trade_value: 200
- min_data_coverage: 0.6
- cooldown_seconds: 600
- buy_price_buffer_pct: 0.005
- max_daily_loss_pct: 0.02
- open_buffer_minutes: 5

Transaction Cost Controls:
- trading_costs.json (system defaults)
  - commission_per_share: 0.005
  - min_commission: 1.0
  - fee_per_order: 0.0
  - slippage_bps: 5.0
  - max_daily_cost_pct: 0.003
  - max_daily_cost_usd: 0.0

Operational Notes:
- Set `INTRADAY_TRADING_ENABLED=true` and restart runner to activate.
- Adjust intraday parameters via Dashboard -> Strategies; restart runner after saving.
- Bot control (Stop/Resume) is in Dashboard.
- Strategy switching is available in Dashboard -> Strategy Switcher (Apply & Restart Bot).
