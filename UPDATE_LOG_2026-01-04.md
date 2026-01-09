## Update Log - 2026-01-04 16:28:17 -08:00

### Purpose
- Replace "end-of-day rebalance only" behavior with intraday monitoring and dynamic position management.
- Reduce failed orders and duplicate submissions, add cost controls for frequent trading.
- Expose intraday parameters to the frontend and add bot stop/start controls to Dashboard.

### Trading Logic Adjustments (Intraday)
- **Intraday signal engine** now uses market timezone day reset (instead of UTC) to avoid premature daily resets.
- **Data coverage guard** added: if the percentage of symbols with valid bars is below `min_data_coverage`, the intraday rebalance is skipped to avoid accidental full liquidation.
- **Price buffer for buys**: buy sizing uses `buy_price_buffer_pct` to avoid underestimation and reduce failed buy orders.
- **Trailing stop logic** retained; adds "trend break" exits when fast EMA dips below slow EMA or momentum turns negative.
- **Intraday engine state** persists across cycles to avoid duplicated orders and to enforce daily limits consistently.

### Order Execution & Duplicate Protection
- **Order status refresh** added: each cycle polls broker order status and updates filled quantity.
- **Submitted order tracking** persists within the intraday session to block duplicate orders.
- **Daily trade counters reset** once per market day (not UTC day) to match market calendar.

### Transaction Cost Controls (System Level)
- Added **system-wide cost config** file: `quant_system_full/config/trading_costs.json`.
- **Cost model** now applied to every turnover:
  - commission per share
  - minimum commission
  - per-order fees
  - slippage (bps)
- **Daily cost limit** enforced via `max_daily_cost_pct` or `max_daily_cost_usd`.
- Costs are estimated pre-trade and accumulated post-trade with order status refresh.

### Frontend Adjustments
- **Bot Control** (Stop/Resume) moved to **Dashboard** page (not Strategies).
- **Strategy Switcher** added to Dashboard to select profile + restart runner.
- **Intraday Strategy Settings** expanded with cost/coverage fields:
  - `min_data_coverage`
  - `buy_price_buffer_pct`
  - `commission_per_share`
  - `min_commission`
  - `fee_per_order`
  - `slippage_bps`
  - `max_daily_cost_pct`
- Build version updated in `UI/src/lib/api.ts` to force cache busting.

### Backend/API Adjustments
- Intraday config endpoints now validate and persist the new cost/coverage fields.
- Intraday state/status includes:
  - `intraday_costs`
  - `intraday_order_statuses`
  - `data_coverage` and skip reasons
- Strategy profile endpoints:
  - `GET /api/strategy/profiles`
  - `PUT /api/strategy/active`
  - `POST /api/runner/restart` (restart request flag)

### New/Updated Configuration Fields
- `quant_system_full/config/intraday_strategy.json`
  - `min_data_coverage` (float 0-1)
  - `buy_price_buffer_pct` (float)
  - `commission_per_share` (float)
  - `min_commission` (float)
  - `fee_per_order` (float)
  - `slippage_bps` (float)
  - `max_daily_cost_pct` (float)
- `quant_system_full/config/trading_costs.json`
  - Same cost fields as above, **system-wide default** for all trading components.
- `quant_system_full/config/strategy_profiles.json`
  - Strategy profile definitions (A/B) and env overrides for switching.
- `quant_system_full/config/active_strategy.json`
  - Active strategy profile marker.

### Operational Notes
- After changing intraday config, **restart runner** to apply changes.
- Frontend changes require **`npm run build`** and **`pm2 restart frontend`**.
- If UI still shows old content, hard refresh or purge Cloudflare cache.

### Files Updated
- quant_system_full/bot/intraday_signal_engine.py
- quant_system_full/dashboard/worker/runner.py
- quant_system_full/dashboard/worker/auto_trading_engine.py
- quant_system_full/dashboard/backend/app.py
- quant_system_full/UI/src/pages/Strategies.tsx
- quant_system_full/UI/src/pages/Dashboard.tsx
- quant_system_full/UI/src/lib/api.ts
- quant_system_full/config/intraday_strategy.json
- quant_system_full/config/trading_costs.json

### Tests/Checks
- Python AST parse checks (local)
- `npm run typecheck` (local)
