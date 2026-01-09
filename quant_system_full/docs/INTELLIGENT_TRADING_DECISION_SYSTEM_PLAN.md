# Intelligent Trading Decision System - Implementation Plan

## Problem Summary

Current system generates 5-9 buy/sell signals per minute, causing excessive trading costs ($1,681.73 realized loss today). User wants a smarter system that:
- Uses multi-factor scoring for trade decisions
- Performs cost/benefit analysis before executing
- Learns optimal parameters automatically

---

## Core Architecture: 4-Layer System

```
Layer 1: Stock Selection (Weekly/Daily) - Universe Filter
Layer 2: Daily Regime (Daily) - Position/Trade Control
Layer 3: Signal Execution (Minute) - Directional Scoring
Layer 4: Risk Control (System) - Hard Rules
```

### Layer 1: Stock Selection (Slow Factors)

**Frequency**: Daily close (NOT hourly - avoid unnecessary churn)

**Hysteresis Buffer** (avoid flip-flop):
- Entry threshold: Score >= 70
- Exit threshold: Score < 60 (10-point buffer)

**Factors**:
- 12-1 Month Momentum (decides if stock is worth watching)
- Historical Win Rate (per-stock performance)

**12-1 Month Momentum Formula** (CRITICAL - must use correct direction):
```python
# CORRECT: Skip most recent month to avoid reversal
# price_21 = 21 trading days ago (about 1 month)
# price_252 = 252 trading days ago (about 12 months)
mom_12_1 = (price_21_days_ago / price_252_days_ago) - 1

# WRONG (inverted, would mark strong stocks as weak):
# mom_12_1 = (price_252 - price_21) / price_252  # DO NOT USE
```

**Mandatory Tradability Filters** (hard rules, no scoring):
```python
# Must pass ALL to enter pool
filters = {
    'min_adv': 20_000_000,      # Average Daily Volume > $20M
    'min_price': 5.0,            # Price > $5 (avoid penny stock noise)
    'max_spread_pct': 0.5,       # Spread < 0.5%
    'min_atr_pct': 0.5,          # ATR% > 0.5% (enough volatility)
    'max_atr_pct': 5.0,          # ATR% < 5% (not too crazy)
}
```

### Layer 2: Daily Regime Control

**Frequency**: Pre-market + 1-2 times during session (NOT every minute)

**VIX Control** (continuous function, not step):
```python
# Smooth position sizing based on VIX
def get_max_position_pct(vix):
    # Linear decay: 100% at VIX=15, 0% at VIX=35
    pct = max(0, min(1, (35 - vix) / 20))
    return pct

# With hysteresis to avoid jitter
if vix > vix_threshold_upper and not in_cautious_mode:
    enter_cautious_mode()
elif vix < vix_threshold_lower and in_cautious_mode:
    exit_cautious_mode()
```

**Sector Rotation**: Boost/reduce weights for sectors in/out of favor

### Layer 3: Signal Execution (Minute-Level)

**Key Change**: Directional scoring (score_long vs score_short)

```python
# Two separate scores
score_long = (
    0.25 * stability_long +
    0.45 * volume_confirmation_long +
    0.30 * price_action_long
)
score_short = (
    0.25 * stability_short +
    0.45 * volume_confirmation_short +
    0.30 * price_action_short
)

# Use based on signal direction
if signal == "BUY":
    final_score = score_long
elif signal == "SELL":
    final_score = score_short
```

**Weights** (adjusted based on reliability):
- Signal Stability: 25% (reduced - easily fooled by oscillation)
- Volume Confirmation: 45% (increased - hardest to fake)
- Price Action: 30% (unchanged)

#### Factor 1: Signal Stability (25%) - Two-Component Version

```python
def calculate_stability(signals, prices, current_signal, N=10):
    """
    Two-component stability score:
    - 60% direction consistency (ratio of same-direction signals)
    - 40% trend strength (magnitude of price move)

    This prevents being fooled by oscillation with small moves.
    """
    # BUG FIX: Only calculate when signal is BUY or SELL
    if current_signal == 0:  # HOLD
        return 0  # Don't calculate stability for HOLD

    # Remove HOLD from statistics
    non_zero_signals = [s for s in signals if s != 0]

    if len(non_zero_signals) < 3:
        return 0  # Not enough data, skip trade

    # Component 1: Direction consistency (60%)
    same_direction = sum(1 for s in non_zero_signals if s == current_signal)
    direction_ratio = same_direction / len(non_zero_signals)

    # Component 2: Trend strength (40%)
    # ret_N = cumulative return over N bars
    ret_N = prices[-1] / prices[-N] - 1 if len(prices) >= N else 0
    cap = 0.005  # 0.5% cap for minute bars (adjust based on bar period)
    trend_strength = min(abs(ret_N), cap) / cap  # Clamp to [0, 1]

    # Combined score
    stability = 0.6 * direction_ratio + 0.4 * trend_strength
    return stability * 100
```

#### Factor 2: Volume Confirmation (45%) - Time-of-Day Adjusted

**BUG FIX (from pre-launch audit):**
- Add avg_floor protection to prevent division by near-zero

```python
def calculate_volume_score(symbol, current_time, current_price):
    current_volume = get_current_bar_volume()

    # FIX: Use same time period historical average
    # NOT simple daily average (opening 15-30 min is different)
    historical_avg = get_same_time_avg_volume(
        symbol,
        time=current_time,  # e.g., 10:05
        lookback_days=20
    )

    # FIX: Protect against near-zero historical_avg
    # Use daily average / 390 minutes * 10% as floor
    daily_avg = get_daily_avg_volume(symbol)
    avg_floor = max(1000, daily_avg / 390 * 0.1)  # At least 1000 shares
    historical_avg = max(historical_avg, avg_floor)

    volume_ratio = current_volume / historical_avg

    # VWAP deviation - USE ATR for normalization (not std)
    # This ensures: 1) stable early in day, 2) cross-stock comparable
    vwap = get_vwap()
    atr_intraday = get_intraday_atr(periods=20)  # 20-bar ATR
    vwap_deviation = (current_price - vwap) / atr_intraday

    # Intraday OBV trend
    obv_trend = get_intraday_obv_trend()

    return (
        score_volume_ratio(volume_ratio) * 0.4 +
        score_vwap_deviation(vwap_deviation) * 0.3 +
        score_obv_trend(obv_trend) * 0.3
    )

def score_volume_ratio(ratio):
    """Volume ratio scoring: higher is better for confirmation"""
    if ratio >= 2.0: return 100
    if ratio >= 1.5: return 80
    if ratio >= 1.0: return 60
    if ratio >= 0.7: return 40
    return 20

def score_vwap_deviation(deviation):
    """VWAP deviation scoring: for BUY, above VWAP is good"""
    # deviation is in ATR units
    if deviation >= 1.0: return 100   # 1+ ATR above VWAP
    if deviation >= 0.5: return 80
    if deviation >= 0.0: return 60    # At or above VWAP
    if deviation >= -0.5: return 40
    return 20                         # Well below VWAP
```

#### Factor 3: Price Action (30%) - Edge-Trigger Key Levels

**CRITICAL: Key Levels Calculation Frequency** (from final validation):
| Level | Calculation Time | Update Frequency |
|-------|-----------------|------------------|
| OR_High/Low | 9:45 AM (15min after open) | Fixed for day |
| PrevClose/High/Low | Pre-market | Fixed for day |
| VWAP | Real-time | Every minute bar |

> **Important**: Gate1 and L3 must use the SAME key_levels object to avoid "triggered but no breakout" inconsistency.

```python
import pytz
from datetime import datetime, time, date

NY_TZ = pytz.timezone('America/New_York')

class KeyLevelProvider:
    """
    FIX 12: Auto-reset on session date change (O(1) global clear)
    FIX 13: Use America/New_York timezone for OR lock timing
    """
    def __init__(self):
        self._cache = {}           # symbol -> key_levels dict
        self._or_locked = {}       # symbol -> bool (OR period ended)
        self._current_session_date = None  # FIX 12: Track trading day

    def get_key_levels(self, symbol, bar_time):
        """
        Get key levels for symbol, auto-resetting on new trading day.
        """
        # FIX 12: Auto-reset when session date changes (O(1) global clear)
        bar_date = self._extract_date(bar_time)
        if self._current_session_date != bar_date:
            self._reset_all_caches()
            self._current_session_date = bar_date

        # Check if OR period is locked
        if not self._or_locked.get(symbol, False):
            # FIX 13: Use NY timezone for time comparison
            bar_time_ny = self._extract_time_ny(bar_time)
            OR_LOCK_TIME = time(9, 45)  # 9:45 AM ET
            if bar_time_ny >= OR_LOCK_TIME:
                self._or_locked[symbol] = True

        # Get or compute key levels
        if symbol not in self._cache:
            self._cache[symbol] = self._compute_key_levels(symbol)

        # VWAP is real-time, always update
        levels = self._cache[symbol].copy()
        levels['vwap'] = get_vwap(symbol)
        levels['vwap_upper'] = levels['vwap'] + 2 * get_intraday_atr(symbol, 20)
        levels['vwap_lower'] = levels['vwap'] - 2 * get_intraday_atr(symbol, 20)

        return levels

    def _compute_key_levels(self, symbol):
        """Compute static key levels (called once per day per symbol)."""
        return {
            # Opening Range (first 15 min high/low) - FIXED after 9:45 AM
            'or_high': get_opening_range_high(symbol, minutes=15),
            'or_low': get_opening_range_low(symbol, minutes=15),
            # Yesterday's levels - FIXED for day
            'prev_high': get_previous_day_high(symbol),
            'prev_low': get_previous_day_low(symbol),
            'prev_close': get_previous_day_close(symbol),
        }

    def _reset_all_caches(self):
        """FIX 12: Clear all caches globally - O(1) operation."""
        self._cache = {}
        self._or_locked = {}

    def _extract_date(self, bar_time):
        """Extract date from bar_time (handles datetime and date)."""
        if isinstance(bar_time, datetime):
            return bar_time.date()
        return bar_time

    def _extract_time_ny(self, bar_time):
        """FIX 13: Extract time component in NY timezone."""
        if isinstance(bar_time, datetime):
            if bar_time.tzinfo is None:
                # Assume already in NY timezone if no tzinfo
                return bar_time.time()
            # Convert to NY timezone
            ny_time = bar_time.astimezone(NY_TZ)
            return ny_time.time()
        return bar_time

# Global singleton
key_level_provider = KeyLevelProvider()

def calculate_price_action_score(price, prev_price, signal_direction, key_levels, breakout_state):
    """
    Edge-trigger scoring: High score only on ACTUAL breakout bar.
    After breakout, switch to continuation mode (pullback logic).

    breakout_state: dict tracking which levels have been broken
    """
    score = 0

    if signal_direction == "BUY":
        # EDGE TRIGGER: Only score high on the breakout bar itself
        # Use "cross" not ">" to detect actual breakout moment

        # OR High breakout (40 points on breakout bar, 20 after)
        if prev_price <= key_levels['or_high'] and price > key_levels['or_high']:
            score += 40  # Fresh breakout
            breakout_state['or_high'] = True
        elif breakout_state.get('or_high') and price > key_levels['or_high']:
            # Already broke out, now in continuation mode
            # Give points for holding above OR high (pullback didn't break it)
            score += 20

        # VWAP cross (30 points on cross, 15 after)
        if prev_price <= key_levels['vwap'] and price > key_levels['vwap']:
            score += 30  # Fresh cross
            breakout_state['vwap'] = True
        elif breakout_state.get('vwap') and price > key_levels['vwap']:
            score += 15

        # Prev close cross (30 points on cross, 15 after)
        if prev_price <= key_levels['prev_close'] and price > key_levels['prev_close']:
            score += 30
            breakout_state['prev_close'] = True
        elif breakout_state.get('prev_close') and price > key_levels['prev_close']:
            score += 15

    else:  # SELL - mirror logic for breakdowns
        if prev_price >= key_levels['or_low'] and price < key_levels['or_low']:
            score += 40
            breakout_state['or_low'] = True
        elif breakout_state.get('or_low') and price < key_levels['or_low']:
            score += 20

        if prev_price >= key_levels['vwap'] and price < key_levels['vwap']:
            score += 30
            breakout_state['vwap_down'] = True
        elif breakout_state.get('vwap_down') and price < key_levels['vwap']:
            score += 15

        if prev_price >= key_levels['prev_close'] and price < key_levels['prev_close']:
            score += 30
            breakout_state['prev_close_down'] = True
        elif breakout_state.get('prev_close_down') and price < key_levels['prev_close']:
            score += 15

    return score, breakout_state
```

### Layer 4: Risk Control (System Level)

```python
class RiskControl:
    def __init__(self):
        self.cooldown_minutes = 5  # Min time between trades per symbol
        self.max_daily_loss = 2000  # $2000 daily loss limit
        self.max_single_position = 0.15  # 15% max in single stock
        self.last_trade_time = {}  # symbol -> datetime

    def check_trade(self, signal_score, symbol, direction):
        """
        Hard rules (not scoring) - must pass ALL to execute
        """
        # 1. Daily loss limit
        if get_daily_pnl() < -self.max_daily_loss:
            return False, "Daily loss limit hit"

        # 2. Position concentration
        if get_position_pct(symbol) > self.max_single_position:
            return False, "Position too large"

        # 3. Trade cooldown (prevent oscillation whipsaws)
        last_trade = self.last_trade_time.get(symbol)
        if last_trade:
            minutes_since = (datetime.now() - last_trade).total_seconds() / 60
            if minutes_since < self.cooldown_minutes:
                return False, f"Cooldown: {self.cooldown_minutes - minutes_since:.1f} min left"

        # 4. Earnings blackout (avoid earnings volatility)
        if is_earnings_window(symbol, days_before=1, days_after=1):
            return False, "Earnings blackout period"

        # 5. Low score special handling
        if signal_score < 50:
            # Allow only specific patterns
            if is_breakout() or is_trend_continuation():
                return True, "Low score but valid pattern"
            return False, "Low score, no pattern"

        return True, "OK"

    def record_trade(self, symbol):
        """Call this after successful trade execution"""
        self.last_trade_time[symbol] = datetime.now()


def is_earnings_window(symbol, days_before=1, days_after=1):
    """
    Check if symbol is within earnings blackout window.
    Data source: Tiger API or third-party earnings calendar.
    """
    earnings_date = get_next_earnings_date(symbol)
    if earnings_date is None:
        return False

    today = datetime.now().date()
    window_start = earnings_date - timedelta(days=days_before)
    window_end = earnings_date + timedelta(days=days_after)

    return window_start <= today <= window_end
```

---

## Final Execution Chain (Decision Flow) - WITH TWO CRITICAL GATES

```python
def execute_decision_chain(symbol, signal, price, prev_price, bar_volume):
    """
    Fixed execution order with TWO NEW GATES to prevent over-trading.
    Returns: (should_trade, target_position, reason)
    """
    # Layer 1: Pool check (daily, cached)
    if not stock_pool.is_in_pool(symbol):
        return False, 0, "Not in pool"

    # Layer 2: Regime check (pre-market + 1-2x/day, cached)
    trade_enabled, max_position_pct, sector_boost, threshold_boost = regime_controller.check()
    if not trade_enabled:
        return False, 0, "Regime disabled trading"

    # ========== NEW GATE 1: TRIGGER GATE ==========
    # Convert from continuous stream to event-triggered
    # Must have ONE of these events to proceed
    # FIX: Pass signal for direction-aware cross checking
    triggered, trigger_reason = trigger_gate.check(symbol, signal, price, prev_price, bar_volume)
    if not triggered:
        return False, 0, "No trigger event"

    # Layer 3: Signal scoring (minute-level)
    if signal == "BUY":
        final_score = directional_scorer.score_long(symbol, price, prev_price)
    elif signal == "SELL":
        final_score = directional_scorer.score_short(symbol, price, prev_price)
    else:
        return False, 0, "No signal"

    # Apply VIX-based threshold boost
    adjusted_threshold = 65 + threshold_boost
    if final_score < adjusted_threshold:
        return False, 0, f"Score {final_score:.1f} < threshold {adjusted_threshold}"

    # ========== NEW GATE 2: COST/BENEFIT GATE ==========
    # Expected Edge must exceed transaction cost
    passed, reason = cost_benefit_gate.check(symbol, price, final_score)
    if not passed:
        return False, 0, reason

    # Layer 4: Risk control (hard rules)
    passed, reason = risk_control.check_trade(final_score, symbol, signal)
    if not passed:
        return False, 0, reason

    # Position sizing
    score_multiplier = get_score_multiplier(final_score)
    target_position = (
        base_position_size *
        score_multiplier *
        max_position_pct *
        sector_boost
    )

    return True, target_position, f"Score {final_score:.1f}"
```

---

## NEW: Trigger Gate (Event-Triggered, Not Stream)

**Purpose**: Convert 5-9 signals/minute into 0-1 signals per price move

**BUG FIXES (from pre-launch audit + final validation):**
1. Data structure: Use `(symbol, level, direction) -> datetime` triple key for clarity
2. Direction-aware cross: Only trigger on same-direction cross as signal
3. Volume z-score: Add std floor + dual condition to prevent early morning false triggers
4. Separate cooldown by direction: BUY and SELL can trigger independently on same level
5. **FIX 14**: Use cooldown_seconds (not bars) - bar-interval independent

```python
class TriggerGate:
    """
    Must satisfy ONE of these events to proceed to scoring.
    This is the KEY to reducing trade frequency.
    """
    def __init__(self):
        # FIX 1: Clear data structure - triple key -> datetime
        # FIX 4: Include direction so BUY/SELL cooldowns are independent
        self.last_trigger_time = {}  # (symbol, level_name, direction) -> datetime
        self.volume_zscore_threshold = 2.5  # Conservative first week
        self.volume_ratio_threshold = 2.0   # FIX 3: Dual condition
        # FIX 14: Use SECONDS not "bars" - works for any bar interval (1min, 5min, etc.)
        self.cooldown_seconds = 30 * 60  # 30 minutes = 1800 seconds

    def check(self, symbol, signal, price, prev_price, bar_volume, bar_time):
        """
        FIX 2: Now takes `signal` parameter to enforce direction-aware crossing
        FIX 14: Now takes `bar_time` parameter for consistent time comparison
        """
        key_levels = key_level_provider.get_key_levels(symbol, bar_time)
        triggered = False
        trigger_reason = None

        # Trigger 1: Key Level Cross (direction-aware)
        # FIX 2: Only allow crosses in same direction as signal
        # FIX 14: Pass bar_time to _can_trigger and _record_trigger
        if signal == "BUY":
            # For BUY: only upward crosses trigger
            if self._is_cross_up(prev_price, price, key_levels['vwap']):
                if self._can_trigger(symbol, 'vwap', 'BUY', bar_time):
                    triggered = True
                    trigger_reason = "VWAP cross up"
                    self._record_trigger(symbol, 'vwap', 'BUY', bar_time)

            if self._is_cross_up(prev_price, price, key_levels['or_high']):
                if self._can_trigger(symbol, 'or_high', 'BUY', bar_time):
                    triggered = True
                    trigger_reason = "OR High breakout"
                    self._record_trigger(symbol, 'or_high', 'BUY', bar_time)

            if self._is_cross_up(prev_price, price, key_levels['prev_close']):
                if self._can_trigger(symbol, 'prev_close', 'BUY', bar_time):
                    triggered = True
                    trigger_reason = "Prev close cross up"
                    self._record_trigger(symbol, 'prev_close', 'BUY', bar_time)

        elif signal == "SELL":
            # For SELL: only downward crosses trigger
            if self._is_cross_down(prev_price, price, key_levels['vwap']):
                if self._can_trigger(symbol, 'vwap', 'SELL', bar_time):
                    triggered = True
                    trigger_reason = "VWAP cross down"
                    self._record_trigger(symbol, 'vwap', 'SELL', bar_time)

            if self._is_cross_down(prev_price, price, key_levels['or_low']):
                if self._can_trigger(symbol, 'or_low', 'SELL', bar_time):
                    triggered = True
                    trigger_reason = "OR Low breakdown"
                    self._record_trigger(symbol, 'or_low', 'SELL', bar_time)

            if self._is_cross_down(prev_price, price, key_levels['prev_close']):
                if self._can_trigger(symbol, 'prev_close', 'SELL', bar_time):
                    triggered = True
                    trigger_reason = "Prev close cross down"
                    self._record_trigger(symbol, 'prev_close', 'SELL', bar_time)

        # Trigger 2: Volume Shock
        # FIX 3: Dual condition to prevent early morning false triggers
        volume_ratio, volume_zscore = self._get_volume_metrics(symbol, bar_volume)
        if volume_ratio > self.volume_ratio_threshold and volume_zscore > self.volume_zscore_threshold:
            triggered = True
            trigger_reason = f"Volume shock ratio={volume_ratio:.1f}x z={volume_zscore:.1f}"

        return triggered, trigger_reason

    def _is_cross_up(self, prev_price, price, level):
        """True if price just crossed UP through the level"""
        return prev_price <= level < price

    def _is_cross_down(self, prev_price, price, level):
        """True if price just crossed DOWN through the level"""
        return prev_price >= level > price

    def _can_trigger(self, symbol, level_name, direction, bar_time):
        """Prevent same level+direction from triggering repeatedly"""
        key = (symbol, level_name, direction)  # FIX 4: Triple key with direction
        last_trigger = self.last_trigger_time.get(key)
        if last_trigger is None:
            return True
        # FIX 14: Compare in seconds using bar_time (not datetime.now())
        seconds_since = (bar_time - last_trigger).total_seconds()
        return seconds_since >= self.cooldown_seconds

    def _record_trigger(self, symbol, level_name, direction, bar_time):
        key = (symbol, level_name, direction)  # FIX 4: Triple key with direction
        # FIX 14: Use bar_time (not datetime.now()) for consistent time tracking
        self.last_trigger_time[key] = bar_time

    def _get_volume_metrics(self, symbol, bar_volume):
        """
        FIX 3: Return both ratio and z-score for dual condition check
        """
        same_time_avg = get_same_time_avg_volume(symbol)
        same_time_std = get_same_time_std_volume(symbol)

        # FIX 3a: Protect against near-zero average
        avg_floor = get_daily_avg_volume(symbol) / 390 * 0.1  # 10% of per-minute average
        same_time_avg = max(same_time_avg, avg_floor)

        # FIX 3b: Protect against near-zero std (early morning)
        std_floor = same_time_avg * 0.3  # At least 30% of avg as std
        same_time_std = max(same_time_std, std_floor)

        volume_ratio = bar_volume / same_time_avg
        volume_zscore = (bar_volume - same_time_avg) / same_time_std

        return volume_ratio, volume_zscore

    def reset_daily(self):
        """Call at market open"""
        self.last_trigger_time = {}
```

---

## NEW: Cost/Benefit Gate

**Purpose**: Only trade when Expected Edge > 2.5x Transaction Cost (conservative first week)

**BUG FIX (from pre-launch audit):**
- All costs must be in consistent units: **$/share**
- Use **half-spread** (more realistic execution)
- Fee handling: distinguish per-share vs per-order fees

```python
class CostBenefitGate:
    """
    Expected edge must exceed transaction cost by a multiple.
    This kills all the small meaningless trades.

    CRITICAL: All values in $/share units for consistency
    """
    def __init__(self):
        self.edge_multiple = 2.5  # Conservative first week (was 2.0)
        self.edge_atr_factor = 0.7  # Conservative: 70% of ATR

    def check(self, symbol, price, signal_score, shares=100):
        """
        Args:
            shares: Expected trade size for per-order cost conversion
        """
        # Calculate expected edge ($/share)
        atr_intraday = get_intraday_atr(symbol, periods=20)
        expected_edge_per_share = self.edge_atr_factor * atr_intraday

        # Calculate transaction cost (ALL in $/share)
        total_cost_per_share = self._calculate_total_cost(symbol, atr_intraday, shares)

        # Check if edge justifies cost
        if total_cost_per_share <= 0:
            return True, "Zero cost"

        edge_ratio = expected_edge_per_share / total_cost_per_share

        if edge_ratio < self.edge_multiple:
            return False, f"Edge/Cost={edge_ratio:.1f} < {self.edge_multiple}"

        return True, f"Edge/Cost={edge_ratio:.1f} OK"

    def _calculate_total_cost(self, symbol, atr, shares):
        """
        Calculate total cost per share with consistent units.
        All returns in $/share.
        """
        # 1. Half-spread (more realistic than full spread)
        bid, ask = get_bid_ask(symbol)
        half_spread = (ask - bid) / 2  # $/share

        # 2. Slippage estimate
        slippage = self._estimate_slippage(symbol, atr)  # $/share

        # 3. Commission/Fee
        # Tiger Brokers: $0.005/share, min $1.00/order
        fee_per_share = 0.005
        min_order_fee = 1.00
        # If min fee dominates, convert to per-share equivalent
        fee = max(fee_per_share, min_order_fee / shares)  # $/share

        total = half_spread + slippage + fee
        return total

    def _estimate_slippage(self, symbol, atr):
        """
        Estimate slippage in $/share based on volatility and liquidity.
        """
        adv = get_average_daily_volume(symbol)  # in shares
        avg_price = get_current_price(symbol)
        adv_dollars = adv * avg_price

        # Base slippage: 1bp of price
        base_slippage = avg_price * 0.0001  # 1bp

        # Volatility factor: higher ATR = more slippage
        vol_factor = min(2.0, atr / (avg_price * 0.01))  # Relative to 1%

        # Liquidity factor: lower ADV$ = more slippage
        liq_factor = max(0.5, min(2.0, 50_000_000 / adv_dollars))

        return base_slippage * vol_factor * liq_factor
```

---

## UPDATED: Layer 2 Regime Controller (VIX Controls Threshold Too)

```python
class RegimeController:
    def check(self):
        vix = get_vix()

        # Position sizing (continuous)
        max_position_pct = max(0, min(1, (35 - vix) / 20))

        # NEW: Threshold boost based on VIX
        # Higher VIX = higher threshold = fewer trades
        if vix < 20:
            threshold_boost = 0
        elif vix < 25:
            threshold_boost = 5   # threshold becomes 70
        elif vix < 30:
            threshold_boost = 10  # threshold becomes 75
        else:
            threshold_boost = 15  # threshold becomes 80

        # Trade enabled/disabled
        trade_enabled = (vix < 40) and is_market_hours()

        # Sector boost (existing)
        sector_boost = get_sector_rotation_boost()

        return trade_enabled, max_position_pct, sector_boost, threshold_boost
```

---

## UPDATED: Layer 4 Risk Control (Stricter Limits)

```python
class RiskControl:
    def __init__(self):
        self.cooldown_minutes = 20  # INCREASED from 5 to 20
        self.max_daily_loss = 2000
        self.max_single_position = 0.15
        self.max_trades_per_symbol_per_day = 2  # Conservative first week (was 3)
        self.max_trades_per_day = 20  # NEW: Global system-level hard cap
        self.last_trade_time = {}
        self.trades_today = {}  # symbol -> count
        self.total_trades_today = 0  # NEW: Global counter

    def check_trade(self, signal_score, symbol, direction):
        # 0. NEW: Global daily trade limit (prevents "trading cost hell")
        if self.total_trades_today >= self.max_trades_per_day:
            return False, f"Daily system cap: {self.total_trades_today}/{self.max_trades_per_day}"

        # 1. Daily loss limit
        if get_daily_pnl() < -self.max_daily_loss:
            return False, "Daily loss limit hit"

        # 2. Position concentration
        if get_position_pct(symbol) > self.max_single_position:
            return False, "Position too large"

        # 3. Trade cooldown (INCREASED to 20 min)
        last_trade = self.last_trade_time.get(symbol)
        if last_trade:
            minutes_since = (datetime.now() - last_trade).total_seconds() / 60
            if minutes_since < self.cooldown_minutes:
                return False, f"Cooldown: {self.cooldown_minutes - minutes_since:.1f} min left"

        # 4. NEW: Max trades per symbol per day
        today_count = self.trades_today.get(symbol, 0)
        if today_count >= self.max_trades_per_symbol_per_day:
            return False, f"Daily cap: {today_count}/{self.max_trades_per_symbol_per_day}"

        # 5. Earnings blackout
        if is_earnings_window(symbol, days_before=1, days_after=1):
            return False, "Earnings blackout period"

        # 6. Low score handling
        if signal_score < 50:
            if is_breakout() or is_trend_continuation():
                return True, "Low score but valid pattern"
            return False, "Low score, no pattern"

        return True, "OK"

    def record_trade(self, symbol):
        self.last_trade_time[symbol] = datetime.now()
        self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
        self.total_trades_today += 1  # NEW: Increment global counter

    def reset_daily_counts(self):
        """Call at market open"""
        self.trades_today = {}
        self.total_trades_today = 0  # NEW: Reset global counter
```

---

## NEW: Exit Rules (Critical Missing Piece)

**From pre-launch audit:**
> "你这套系统把'进场'管得很好，但如果没有标准化退出，会出现：进场次数下降了，但单笔亏损扩大，最终还是不赚钱。"

**Purpose**: Standardized exit rules to prevent single-trade losses from expanding

**CRITICAL: ATR Consistency** (from final validation):
> Entry (Gate2) and Exit must use the SAME ATR scale:
> - Gate2: `get_intraday_atr(symbol, periods=20)` - 20-bar intraday ATR
> - ExitManager: `get_intraday_atr(symbol, periods=20)` - SAME function
>
> If mismatch (e.g., exit uses daily ATR), risk/reward assumption breaks.

```python
class ExitManager:
    """
    Minimum viable exit rules to control single-trade risk.
    Managed at Layer 4 (system level) for uniform execution.
    """
    def __init__(self):
        # ATR-based dynamic stops
        self.stop_loss_atr_multiple = 1.0   # 1.0 * ATR_intraday
        self.take_profit_atr_multiple = 1.5 # 1.5 * ATR_intraday (can increase to 2.0)

        # Time stop
        self.time_stop_minutes = 30  # Exit if no profit after 30 min

    def check_exit(self, position):
        """
        Check if position should be exited.
        Returns: (should_exit: bool, reason: str)
        """
        symbol = position['symbol']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        current_price = get_current_price(symbol)
        atr = get_intraday_atr(symbol, periods=20)

        # Calculate P&L
        if position['direction'] == 'LONG':
            pnl_per_share = current_price - entry_price
        else:  # SHORT
            pnl_per_share = entry_price - current_price

        # 1. Stop Loss: 1.0 * ATR
        stop_loss_threshold = -1 * self.stop_loss_atr_multiple * atr
        if pnl_per_share <= stop_loss_threshold:
            return True, f"Stop loss: {pnl_per_share:.2f} <= {stop_loss_threshold:.2f}"

        # 2. Take Profit: 1.5 * ATR
        take_profit_threshold = self.take_profit_atr_multiple * atr
        if pnl_per_share >= take_profit_threshold:
            return True, f"Take profit: {pnl_per_share:.2f} >= {take_profit_threshold:.2f}"

        # 3. Time Stop: 30 min without profit
        minutes_held = (datetime.now() - entry_time).total_seconds() / 60
        if minutes_held >= self.time_stop_minutes and pnl_per_share <= 0:
            return True, f"Time stop: {minutes_held:.0f}min held, no profit"

        return False, "Hold"

    def get_exit_levels(self, entry_price, direction, symbol):
        """
        Calculate stop loss and take profit price levels for order placement.
        """
        atr = get_intraday_atr(symbol, periods=20)

        if direction == 'LONG':
            stop_loss_price = entry_price - self.stop_loss_atr_multiple * atr
            take_profit_price = entry_price + self.take_profit_atr_multiple * atr
        else:  # SHORT
            stop_loss_price = entry_price + self.stop_loss_atr_multiple * atr
            take_profit_price = entry_price - self.take_profit_atr_multiple * atr

        return {
            'stop_loss': round(stop_loss_price, 2),
            'take_profit': round(take_profit_price, 2),
            'time_stop_minutes': self.time_stop_minutes
        }
```

**Exit Rule Integration:**
```python
# In runner.py main loop:
def check_all_positions():
    exit_manager = ExitManager()

    for position in get_open_positions():
        should_exit, reason = exit_manager.check_exit(position)
        if should_exit:
            close_position(position['symbol'], reason)
            log(f"EXIT: {position['symbol']} - {reason}")
```

**Exit Reason Categories:**
- `"Stop loss hit"` - 1.0 * ATR triggered
- `"Take profit"` - 1.5 * ATR reached
- `"Time stop"` - 30 min without profit
- `"Signal reversal"` - Opposite signal triggered
- `"Manual"` - User intervention

---

## UPDATED: Stability Factor (ATR-Based Cap + Sparse Signal Check)

**BUG FIX (from pre-launch audit):**
- Add sparse signal density check: if too many HOLDs, stability is unreliable

```python
def calculate_stability(signals, prices, current_signal, N=10):
    """
    Two-component stability score with sparse signal protection.

    Returns 0-100 score.
    """
    if current_signal == 0:  # HOLD
        return 0

    non_zero_signals = [s for s in signals if s != 0]

    # FIX 1: Minimum signal count
    if len(non_zero_signals) < 3:
        return 0

    # FIX 2: Sparse signal density check
    # If most recent N bars are mostly HOLD, direction_ratio is unreliable
    signal_density = len(non_zero_signals) / N
    if signal_density < 0.4:
        # Too sparse - reduce weight or return 0
        return 0  # Or: apply penalty factor

    # Component 1: Direction consistency (60%)
    same_direction = sum(1 for s in non_zero_signals if s == current_signal)
    direction_ratio = same_direction / len(non_zero_signals)

    # Component 2: Trend strength (40%)
    # Use ATR-based cap for cross-stock comparability
    ret_N = prices[-1] / prices[-N] - 1 if len(prices) >= N else 0
    atr_intraday = get_intraday_atr(periods=20)
    price = prices[-1]
    cap = 1.5 * atr_intraday / price  # Dynamic cap based on ATR
    trend_strength = min(abs(ret_N), cap) / cap

    stability = 0.6 * direction_ratio + 0.4 * trend_strength
    return stability * 100
```

---

## UPDATED: Price Action (Decaying Continuation Score)

**BUG FIX (from pre-launch audit):**
- Use `.get()` with default to prevent KeyError when level_name not in state

```python
class PriceActionScorer:
    def __init__(self):
        self.breakout_state = {}  # symbol -> {level_name: {triggered: bool, count: int}}

    def calculate_score(self, symbol, price, prev_price, signal_direction, key_levels):
        # FIX: Use .get() with empty dict default
        state = self.breakout_state.get(symbol, {})
        score = 0

        if signal_direction == "BUY":
            # OR High
            score += self._score_level(
                prev_price, price, key_levels['or_high'],
                state, 'or_high', direction='up',
                breakout_pts=40, continuation_decay=[20, 10, 0]
            )
            # VWAP
            score += self._score_level(
                prev_price, price, key_levels['vwap'],
                state, 'vwap', direction='up',
                breakout_pts=30, continuation_decay=[15, 7, 0]
            )
            # Prev close
            score += self._score_level(
                prev_price, price, key_levels['prev_close'],
                state, 'prev_close', direction='up',
                breakout_pts=30, continuation_decay=[15, 7, 0]
            )

        elif signal_direction == "SELL":
            # OR Low
            score += self._score_level(
                prev_price, price, key_levels['or_low'],
                state, 'or_low', direction='down',
                breakout_pts=40, continuation_decay=[20, 10, 0]
            )
            # VWAP
            score += self._score_level(
                prev_price, price, key_levels['vwap'],
                state, 'vwap_down', direction='down',
                breakout_pts=30, continuation_decay=[15, 7, 0]
            )
            # Prev close
            score += self._score_level(
                prev_price, price, key_levels['prev_close'],
                state, 'prev_close_down', direction='down',
                breakout_pts=30, continuation_decay=[15, 7, 0]
            )

        self.breakout_state[symbol] = state
        return score

    def _score_level(self, prev_price, price, level, state, level_name,
                     direction, breakout_pts, continuation_decay):
        """
        Decaying continuation score:
        - Fresh breakout: full points
        - 1st continuation: decay[0]
        - 2nd continuation: decay[1]
        - 3rd+ continuation: decay[2] (usually 0)
        """
        # FIX: Use .get() with default dict to prevent KeyError
        lvl = state.get(level_name, {'triggered': False, 'count': 0})

        is_cross = (
            (direction == 'up' and prev_price <= level < price) or
            (direction == 'down' and prev_price >= level > price)
        )
        is_holding = (
            (direction == 'up' and price > level) or
            (direction == 'down' and price < level)
        )

        if is_cross:
            # Fresh breakout
            lvl = {'triggered': True, 'count': 0}
            state[level_name] = lvl
            return breakout_pts
        elif lvl['triggered'] and is_holding:
            # Continuation mode with decay
            count = lvl['count']
            lvl['count'] = count + 1
            state[level_name] = lvl
            if count < len(continuation_decay):
                return continuation_decay[count]
            return 0
        else:
            # Reset if price retreats below level
            lvl = {'triggered': False, 'count': 0}
            state[level_name] = lvl
            return 0

    def reset_daily(self, symbol):
        """Call at market open"""
        self.breakout_state[symbol] = {}
```

---

## Score-to-Position Mapping

| Score | Multiplier | Notes |
|-------|-----------|-------|
| >= 80 | 1.0 | Full position |
| 65-79 | 0.6 | Normal signal |
| 50-64 | 0.3 | Weak signal |
| < 50 | 0.2 | Exception only |

---

## Quick Win Parameters (First Week - Conservative)

```python
# FIRST WEEK: Use conservative values to stop bleeding
# After confirming signal quality improvement, can relax gradually

COOLDOWN_MINUTES = 20                   # Was 5
MAX_TRADES_PER_SYMBOL_DAY = 2           # Conservative: was 3
MAX_TRADES_PER_DAY = 20                 # NEW: Global system-level hard cap
EDGE_MULTIPLE = 2.5                     # Conservative: was 2.0
VOLUME_ZSCORE_THRESHOLD = 2.5           # Conservative: was 2.0
VIX_THRESHOLD_BOOST = {
    'vix < 20': 0,
    '20-25': 5,
    '25-30': 10,
    '>30': 15
}

# AFTER 1 WEEK: If signal quality confirmed, can relax to:
# MAX_TRADES_PER_SYMBOL_DAY = 3
# MAX_TRADES_PER_DAY = 30
# EDGE_MULTIPLE = 2.0
# VOLUME_ZSCORE_THRESHOLD = 2.0
```

---

## Phase 0: Trade History Sync (PREREQUISITE)

### Problem
- Tiger API has 100+ filled orders but data not persisted
- Need historical data to calculate per-stock win rates
- Supabase package not installed on server

### Solution: `trade_history` Table in Supabase

**Table Schema: `trade_history`**
```sql
CREATE TABLE trade_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(4) NOT NULL,           -- BUY or SELL
    quantity INTEGER NOT NULL,
    fill_price DECIMAL(12,4) NOT NULL,
    total_value DECIMAL(12,2) NOT NULL,   -- quantity * fill_price
    trade_time TIMESTAMP NOT NULL,
    tiger_order_id VARCHAR(50) UNIQUE,

    -- Decision Reasoning (WHY this trade was made)
    decision_score DECIMAL(5,2),          -- Final score from L3 (0-100)
    adjusted_threshold DECIMAL(5,2),      -- VIX-adjusted threshold used
    score_components JSONB,               -- {stability: 25, volume: 45, price_action: 30}

    -- Gate 1: Trigger Info
    gate1_reason VARCHAR(50),             -- 'vwap_cross_up', 'or_high_breakout', 'volume_shock'

    -- Gate 2: Cost/Benefit
    gate2_edge DECIMAL(12,4),             -- Expected edge $/share
    gate2_cost DECIMAL(12,4),             -- Total cost $/share
    edge_multiple DECIMAL(5,2),           -- edge/cost ratio used

    -- Regime (L2) at trade time
    regime JSONB,                         -- {vix, max_position_pct, threshold_boost, sector_boost}

    -- Cost estimates (for post-trade analysis)
    fees_estimated DECIMAL(12,4),         -- Commission/fee $/share
    slippage_estimated DECIMAL(12,4),     -- Estimated slippage $/share
    spread_at_entry DECIMAL(12,4),        -- Bid-ask spread at entry

    -- Round-trip P&L (updated when position closed)
    is_position_closed BOOLEAN DEFAULT FALSE,
    paired_trade_id UUID REFERENCES trade_history(id),
    entry_price DECIMAL(12,4),
    exit_price DECIMAL(12,4),
    pnl_amount DECIMAL(12,2),             -- Profit/Loss in dollars
    pnl_percent DECIMAL(8,4),             -- Profit/Loss percentage
    hold_duration_minutes INTEGER,
    exit_reason TEXT,                     -- "Stop loss hit", "Take profit", "Signal reversal"

    -- Learning Feedback
    was_profitable BOOLEAN,               -- Quick lookup for win rate
    decision_quality VARCHAR(20),         -- 'correct', 'incorrect', 'neutral' (post-analysis)

    -- Metadata
    sector VARCHAR(50),
    source VARCHAR(20) DEFAULT 'system',  -- 'system', 'manual', 'sync_from_tiger'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX idx_trade_history_time ON trade_history(trade_time DESC);
CREATE INDEX idx_trade_history_profitable ON trade_history(symbol, was_profitable);
```

**Example Record:**
```json
{
  "symbol": "AAPL",
  "action": "BUY",
  "quantity": 50,
  "fill_price": 185.50,

  "decision_score": 78.5,
  "adjusted_threshold": 70,
  "score_components": {
    "stability": 22,
    "volume": 38,
    "price_action": 18.5
  },

  "gate1_reason": "vwap_cross_up",
  "gate2_edge": 0.85,
  "gate2_cost": 0.32,
  "edge_multiple": 2.66,

  "regime": {
    "vix": 18.5,
    "max_position_pct": 0.825,
    "threshold_boost": 5,
    "sector_boost": 1.1
  },

  "fees_estimated": 0.005,
  "slippage_estimated": 0.02,
  "spread_at_entry": 0.03,

  "is_position_closed": true,
  "exit_price": 188.20,
  "pnl_amount": 135.00,
  "pnl_percent": 1.46,
  "hold_duration_minutes": 45,
  "exit_reason": "Take profit",
  "was_profitable": true,
  "decision_quality": "correct"
}
```

**Two Recording Modes:**

### Mode 1: Historical Sync (Past Trades)
```python
class TradeHistorySync:
    def sync_from_tiger(self, days=90):
        """Sync filled orders from Tiger API to Supabase (historical data)"""
        orders = tiger_client.get_filled_orders(start_date=90_days_ago)

        for order in orders:
            if not exists_in_supabase(order.id):
                insert_trade(
                    symbol=order.symbol,
                    action=order.action,
                    quantity=order.filled,
                    fill_price=order.avg_fill_price,
                    trade_time=order.trade_time,
                    tiger_order_id=order.id,
                    source='sync_from_tiger',  # Mark as historical sync
                    # No decision_score/reasoning for historical trades
                )

        # Match buy/sell pairs and calculate P&L
        self.calculate_round_trip_pnl()
```

### Mode 2: Real-time Recording (Future Trades)
```python
# In auto_trading_engine.py, when executing a trade:
def execute_trade(self, signal):
    # Execute via Tiger API
    order_result = tiger_client.place_order(...)

    if order_result.success:
        # Record with full decision context
        supabase.insert_trade(
            symbol=signal['symbol'],
            action=signal['action'],
            quantity=signal['quantity'],
            fill_price=order_result.avg_fill_price,
            trade_time=datetime.now(),
            tiger_order_id=order_result.order_id,

            # Decision Reasoning (NEW)
            decision_score=signal['final_score'],
            score_breakdown={
                'signal_continuity': signal['scores']['continuity'],
                'price_momentum': signal['scores']['momentum'],
                'volume_confirmation': signal['scores']['volume'],
                'market_sentiment': signal['scores']['sentiment'],
                'historical_win_rate': signal['scores']['history']
            },
            trigger_reason=f"{signal['signal_type']}: score {signal['final_score']:.1f}, {signal['reason']}",
            signal_type=signal['signal_type'],
            strategy_name=signal['strategy'],
            market_conditions={
                'vix': get_vix(),
                'sector': signal['sector'],
                'phase': get_market_phase()
            },
            source='system'
        )

# When position is closed:
def record_exit_to_supabase(position, order_result, exit_reason, bar_time):
    """
    FIX 15: Properly capture exit_trade_id from insert_trade() return value.
    """
    # Find the entry trade by position_id
    entry_trade = supabase.get_entry_by_position_id(position['position_id'])
    if entry_trade is None:
        import logging
        logging.error(f"No entry trade found for position {position['position_id']}")
        return None

    # Direction-aware P&L calculation
    direction = entry_trade.direction
    if direction == 'LONG':
        pnl_amount = (order_result.avg_fill_price - entry_trade.fill_price) * order_result.quantity
    else:  # SHORT
        pnl_amount = (entry_trade.fill_price - order_result.avg_fill_price) * order_result.quantity

    pnl_percent = pnl_amount / (entry_trade.fill_price * order_result.quantity) * 100

    # FIX 15: CAPTURE the returned ID from insert_trade()
    exit_trade_id = supabase.insert_trade(
        position_id=position['position_id'],
        role='EXIT',
        direction=direction,
        symbol=position['symbol'],
        action='SELL' if direction == 'LONG' else 'BUY',  # Opposite action to close
        quantity=order_result.quantity,
        fill_price=order_result.avg_fill_price,
        trade_time=bar_time,  # Use bar_time, not datetime.now()
        tiger_order_id=order_result.order_id,
        source='system'
    )

    # FIX 15: Handle None return case
    if exit_trade_id is None:
        import logging
        logging.error(f"Failed to insert exit trade for {position['symbol']}")
        # Continue to update entry even if exit insert failed

    # Calculate hold duration using bar_time (not datetime.now())
    hold_duration_minutes = int((bar_time - entry_trade.trade_time).total_seconds() / 60)

    # Update the entry record with exit info
    supabase.update_trade(
        entry_trade.id,
        is_position_closed=True,
        paired_trade_id=exit_trade_id,  # FIX 15: Now properly defined
        exit_price=order_result.avg_fill_price,
        pnl_amount=pnl_amount,
        pnl_percent=pnl_percent,
        hold_duration_minutes=hold_duration_minutes,
        exit_reason=exit_reason,  # "Stop loss hit", "Take profit", "Signal reversal"
        was_profitable=(pnl_amount > 0),
        decision_quality='correct' if pnl_amount > 0 else 'incorrect'
    )

    return exit_trade_id
```

**Expected Output Example:**
| Symbol | Action | Score | Reason | P&L | Hold | Quality |
|--------|--------|-------|--------|-----|------|---------|
| AAPL | BUY | 78.5 | Strong momentum, 3x volume | +$135 | 45min | correct |
| MSFT | BUY | 65.2 | Moderate buy signal | -$42 | 23min | incorrect |
| ZIM | BUY | - | (historical sync) | +$1.55 | 3min | - |

### Files to Create/Modify
1. `bot/trade_history_sync.py` - Tiger historical sync
2. `bot/trade_history_recorder.py` - Real-time trade recording
3. `bot/trade_history_analyzer.py` - Win rate calculations
4. `dashboard/worker/auto_trading_engine.py` - Add recording calls

### Prerequisites
- Install supabase package on server: `pip install supabase`
- Create `trade_history` table in Supabase dashboard

---

## Solution Architecture

```
Raw Signals -> TradeSignalAggregator -> CostBenefitAnalyzer -> Execute
                      |                        |
                      v                        v
               AdaptiveLearner <-------> TradeDecisionDB
```

## New Modules

### 1. LayeredSignalProcessor (`bot/layered_signal_processor.py`)

Implements the 4-layer architecture defined above.

### 2. StockSelectionFilter (`bot/stock_selection_filter.py`)

Layer 1 implementation:
- 12-1 momentum scoring
- Historical win rate calculation
- Tradability filters (ADDV, price, spread, ATR)
- Hysteresis logic for pool entry/exit

**Historical Win Rate Cold Start:**
```python
def calculate_win_rate_score(symbol, trade_history, sector_stats, system_stats):
    trades = trade_history.get(symbol, [])
    trade_count = len(trades)

    if trade_count >= 5:
        win_rate = sum(1 for t in trades if t.profit > 0) / trade_count
        confidence = min(1.0, trade_count / 20)
    elif trade_count >= 1:
        stock_win_rate = sum(1 for t in trades if t.profit > 0) / trade_count
        win_rate = (stock_win_rate + system_stats.win_rate) / 2
        confidence = 0.3 + 0.15 * trade_count
    else:
        sector = get_sector(symbol)
        win_rate = sector_stats.get(sector, system_stats.win_rate)
        confidence = 0.2

    final_score = win_rate * confidence + system_stats.win_rate * (1 - confidence)
    return final_score * 100
```

### 3. RegimeController (`bot/regime_controller.py`)

Layer 2 implementation:
- VIX-based position sizing (continuous function)
- Sector rotation detection
- Trade enable/disable logic

### 4. DirectionalScorer (`bot/directional_scorer.py`)

Layer 3 implementation:
- score_long and score_short calculation
- Signal stability (excluding HOLD)
- Volume confirmation (time-of-day adjusted)
- Price action (objective key levels)

### 5. CostBenefitAnalyzer (`bot/cost_benefit_analyzer.py`)

Pre-trade cost analysis:
```python
expected_profit = estimate_price_move(signal_score) * position_value
transaction_cost = commission + spread + slippage + market_impact
min_ratio = 2.0 * volatility_multiplier  # VIX-based

if expected_profit / transaction_cost >= min_ratio:
    EXECUTE
else:
    BLOCK
```

### 6. AdaptiveLearner (`bot/adaptive_learner.py`)

**Learning Objectives:**
1. Optimal score thresholds (Bayesian optimization)
2. Per-stock best strategies (Thompson Sampling)
3. Factor weights (Online gradient descent)
4. Time-of-day patterns (Time series analysis)

**Update Frequency:** Daily (not hourly - aligned with Layer 1)

## Files to Modify

| File | Change |
|------|--------|
| `dashboard/worker/auto_trading_engine.py` | Integrate aggregator and analyzer in `analyze_trading_opportunities()` |
| `dashboard/worker/runner.py` | Initialize AdaptiveLearner, add hourly learning task |
| `bot/intraday_signal_engine.py` | Record signal history for continuity |
| `bot/config_trading.py` | Add new configuration parameters |
| `dashboard/backend/supabase_client.py` | Add trade_decisions table methods |

## New Files to Create

1. `bot/trade_signal_aggregator.py` - Multi-factor scoring
2. `bot/cost_benefit_analyzer.py` - Pre-trade cost analysis
3. `bot/adaptive_learner.py` - Parameter optimization
4. `bot/trade_decision_db.py` - Decision persistence
5. `config/trade_decision_config.json` - Configuration

## Configuration Parameters

```bash
# .env additions
TRADE_DECISION_ENABLED=true
TRADE_SCORE_BUY_THRESHOLD=65
TRADE_SCORE_SELL_THRESHOLD=65
MIN_PROFIT_COST_RATIO=2.0
DYNAMIC_RATIO_ENABLED=true
ADAPTIVE_LEARNING_ENABLED=true
SIGNAL_CONTINUITY_LOOKBACK=5
```

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Signals/minute | 5-9 | 5-9 (same) |
| Approved trades/day | 200-400 | 20-50 |
| Trading cost/day | $1,600+ | ~$200-400 |
| Pass rate | 100% | 10-20% |

## Implementation Phases

### Phase 0: Trade History Sync (FIRST - see above)
- Install supabase package on Vultr server
- Create `trade_history` table in Supabase
- Implement `bot/trade_history_sync.py`
- Run initial sync from Tiger API (90 days)
- Verify P&L calculations are correct

### Phase 1: Core Infrastructure
- Create `trade_decisions` table schema
- Add Supabase table methods to supabase_client.py
- Create configuration loader for new parameters

### Phase 2: Signal Aggregation
- Implement TradeSignalAggregator
- Add signal history recording
- Implement all 5 scoring factors (AI factor reserved for future)

### Phase 3: Cost/Benefit Analysis (Day 3-4)
- Implement CostBenefitAnalyzer
- Dynamic ratio calculation
- Integration with existing cost model

### Phase 4: Integration (Day 4-5)
- Modify auto_trading_engine.py
- Add to runner.py initialization
- Test with dry_run=true

### Phase 5: Adaptive Learning (Day 5-7)
- Implement AdaptiveLearner
- Threshold optimization
- Per-stock strategy learning
- Weight optimization

### Phase 6: Testing & Tuning (Day 7+)
- Production monitoring
- Parameter tuning
- Enable full learning

## Key Integration Point

**`auto_trading_engine.py` modification:**

```python
def analyze_trading_opportunities(...):
    # 1. Generate raw signals (existing)
    raw_signals = self._generate_raw_signals(...)

    # 2. NEW: Score signals
    from bot.trade_signal_aggregator import TradeSignalAggregator
    aggregator = TradeSignalAggregator()
    scored_signals = [aggregator.score_signal(s) for s in raw_signals]

    # 3. NEW: Cost/benefit filter
    from bot.cost_benefit_analyzer import CostBenefitAnalyzer
    analyzer = CostBenefitAnalyzer()

    filtered_signals = {'buy': [], 'sell': []}
    for signal in scored_signals:
        if signal['final_score'] >= threshold:
            should_execute, reason = analyzer.should_execute_trade(signal)
            if should_execute:
                filtered_signals[signal['action']].append(signal)
                log(f"APPROVED: {signal['symbol']} score={signal['final_score']}")
            else:
                log(f"BLOCKED: {signal['symbol']} reason={reason}")

    return filtered_signals
```

## Critical Files

1. `quant_system_full/dashboard/worker/auto_trading_engine.py` (lines 681-833) - Main integration point
2. `quant_system_full/bot/intraday_signal_engine.py` - Signal history recording
3. `quant_system_full/bot/transaction_cost_analyzer.py` - Cost calculation reference
4. `quant_system_full/dashboard/backend/supabase_client.py` - Database operations
5. `quant_system_full/bot/selection_strategies/improved_strategies/improved_value_momentum_v2.py` - Existing scoring reference

## Future Enhancement (when AI model is trained)

Add AI factor (25%) by reducing other weights proportionally:
- signal_continuity: 20%
- price_momentum: 15%
- volume_confirmation: 15%
- market_sentiment: 10%
- historical_win_rate: 15%
- ai_signal_score: 25% (NEW)

---

## Bug Fixes Checklist (Pre-Launch Audit)

### Original Fixes (FIX 1-11)
| FIX | Component | Issue | Status |
|-----|-----------|-------|--------|
| 1 | TriggerGate | Clear data structure - triple key | Done |
| 2 | TriggerGate | Direction-aware cross checking | Done |
| 3 | TriggerGate | Volume z-score dual condition | Done |
| 4 | TriggerGate | Separate cooldown by direction | Done |
| 5 | CostBenefitGate | Consistent $/share units | Done |
| 6 | CostBenefitGate | Half-spread calculation | Done |
| 7 | CostBenefitGate | Per-share vs per-order fees | Done |
| 8 | Stability | Sparse signal density check | Done |
| 9 | PriceAction | Use .get() with default dict | Done |
| 10 | KeyLevelProvider | ATR consistency entry/exit | Done |
| 11 | KeyLevelProvider | OR/VWAP/PrevClose timing | Done |

### New Critical Fixes (FIX 12-17)
| FIX | Component | Issue | Solution | Status |
|-----|-----------|-------|----------|--------|
| 12 | KeyLevelProvider | reset_daily() loops 2000+ symbols (slow) | Auto-reset via session date tracking O(1) | Done |
| 13 | KeyLevelProvider | OR lock uses wrong timezone | Use America/New_York with pytz | Done |
| 14 | TriggerGate | cooldown_bars misleading (bar-dependent) | Rename to cooldown_seconds = 1800 | Done |
| 15 | record_exit | exit_trade.id undefined | Capture return: exit_trade_id = insert() | Done |
| 16 | PositionManager | active_positions keyed by symbol | Key by position_id instead | **TODO** |
| 17 | PositionSizer | SELL = close all (blocks short) | Add intent field (OPEN/CLOSE) | **TODO** |

---

## FIX 16: active_positions Structure (CRITICAL)

### Problem
Current `active_positions = {symbol: {...}}` prevents:
- Adding to positions (second ENTRY overwrites first)
- Partial exits (分批止盈/止损)
- Same symbol long/short simultaneously

### Solution
```python
class PositionManager:
    def __init__(self):
        # FIX 16: Key by position_id, not symbol
        self.active_positions = {}  # position_id -> position_dict

        # Optional: Quick lookup if enforcing 1 position per symbol
        self.symbol_to_position_id = {}  # symbol -> position_id (if single position mode)
        self.single_position_per_symbol = True  # MVP: explicit constraint

    def open_position(self, position_id, symbol, direction, entry_price, quantity, bar_time):
        """Open a new position."""
        # FIX 16: Enforce single position per symbol if configured
        if self.single_position_per_symbol:
            if symbol in self.symbol_to_position_id:
                existing_id = self.symbol_to_position_id[symbol]
                raise ValueError(f"Symbol {symbol} already has position {existing_id}")

        position = {
            'position_id': position_id,
            'symbol': symbol,
            'direction': direction,  # 'LONG' or 'SHORT'
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': bar_time,
            'status': 'OPEN'
        }

        self.active_positions[position_id] = position
        self.symbol_to_position_id[symbol] = position_id
        return position

    def close_position(self, position_id, exit_price, exit_reason, bar_time):
        """Close an existing position."""
        if position_id not in self.active_positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.active_positions[position_id]
        position['exit_price'] = exit_price
        position['exit_reason'] = exit_reason
        position['exit_time'] = bar_time
        position['status'] = 'CLOSED'

        # Calculate P&L
        if position['direction'] == 'LONG':
            position['pnl'] = (exit_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            position['pnl'] = (position['entry_price'] - exit_price) * position['quantity']

        # Clean up lookups
        symbol = position['symbol']
        if self.symbol_to_position_id.get(symbol) == position_id:
            del self.symbol_to_position_id[symbol]

        del self.active_positions[position_id]
        return position

    def get_position_by_symbol(self, symbol):
        """Get position for symbol (single position mode)."""
        position_id = self.symbol_to_position_id.get(symbol)
        if position_id:
            return self.active_positions.get(position_id)
        return None

    def has_open_position(self, symbol):
        """Check if symbol has an open position."""
        return symbol in self.symbol_to_position_id
```

---

## FIX 17: PositionSizer Signal vs Intent Separation (CRITICAL)

### Problem
Current SELL logic:
```python
else:  # SELL
    delta_shares = -current_shares  # Assumes SELL = close all
```

This breaks when:
- `signal == SELL` but intent is "open short" (not close long)
- `current_shares == 0` means delta_shares = 0, never opens short

### Solution: Add Intent Field

**MVP Approach (No Short for Now):**
```python
class PositionSizer:
    def __init__(self):
        self.allow_short = False  # MVP: Disable shorting

    def calculate_delta(self, signal, intent, symbol, current_shares, score, regime):
        """
        FIX 17: Separate signal from intent.

        Args:
            signal: 'BUY' or 'SELL' (direction of trade)
            intent: 'OPEN' | 'CLOSE' | 'REDUCE' | 'INCREASE'
            current_shares: Current position (positive = long, negative = short)
        """
        if signal == 'BUY':
            if intent == 'OPEN':
                # Open new long position
                if current_shares != 0:
                    return 0, "Already have position"
                target_shares = self._calculate_target_shares(symbol, score, regime)
                return target_shares, "Open long"

            elif intent == 'CLOSE':
                # Close short position (buy to cover)
                if current_shares >= 0:
                    return 0, "No short to close"
                return -current_shares, "Close short"  # Buy to cover

            elif intent == 'INCREASE':
                # Add to long position
                additional = self._calculate_add_shares(symbol, score, regime)
                return additional, "Increase long"

        elif signal == 'SELL':
            if intent == 'CLOSE':
                # Close long position
                if current_shares <= 0:
                    return 0, "No long to close"  # FIX 17: Explicit rejection
                return -current_shares, "Close long"

            elif intent == 'OPEN':
                # Open short position
                if not self.allow_short:
                    return 0, "Short disabled"  # FIX 17: Explicit rejection
                if current_shares != 0:
                    return 0, "Already have position"
                target_shares = -self._calculate_target_shares(symbol, score, regime)
                return target_shares, "Open short"

            elif intent == 'REDUCE':
                # Partial close of long
                reduce_qty = self._calculate_reduce_shares(current_shares, score)
                return -reduce_qty, "Reduce long"

        return 0, "Unknown signal/intent"


# In execute_decision_chain:
def execute_decision_chain(symbol, signal, price, prev_price, bar_volume, current_shares):
    # ... existing gate checks ...

    # FIX 17: Determine intent based on current position
    if signal == 'BUY':
        if current_shares < 0:
            intent = 'CLOSE'  # Buy to cover short
        elif current_shares == 0:
            intent = 'OPEN'   # Open new long
        else:
            intent = 'INCREASE'  # Add to long

    elif signal == 'SELL':
        if current_shares > 0:
            intent = 'CLOSE'  # Close long
        elif current_shares == 0:
            # FIX 17: Explicit handling - no position to close
            if not position_sizer.allow_short:
                return False, 0, "SELL with no position and short disabled"
            intent = 'OPEN'   # Open new short
        else:
            intent = 'INCREASE'  # Add to short

    delta_shares, reason = position_sizer.calculate_delta(
        signal, intent, symbol, current_shares, final_score, regime
    )

    if delta_shares == 0:
        return False, 0, reason

    return True, delta_shares, reason
```

**Key Behavior Changes:**
1. `SELL` when `current_shares == 0` is now explicitly rejected (not silently ignored)
2. Intent is derived from current position state
3. Short positions require explicit `allow_short = True`
4. Clear audit trail: reason field explains why delta = 0
