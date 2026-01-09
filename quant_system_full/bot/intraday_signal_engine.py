import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bot.config import SETTINGS
from bot.data import fetch_batch_history
from bot.market_time import get_market_manager

logger = logging.getLogger(__name__)

# Lazy import for Supabase client
_supabase_client = None


def _get_supabase():
    """Get Supabase client singleton for signal recording."""
    global _supabase_client
    if _supabase_client is None:
        try:
            from dashboard.backend.supabase_client import supabase_client
            _supabase_client = supabase_client
        except ImportError:
            logger.debug("Supabase client not available for signal recording")
            _supabase_client = False
    return _supabase_client if _supabase_client else None


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    defaults = {
        "signal_period": "5min",
        "lookback_bars": 120,
        "fast_ema": 9,
        "slow_ema": 21,
        "atr_period": 14,
        "trail_atr": 3.5,              # Widened from 2.5 to reduce whipsaw
        "hard_stop_atr": 3.0,
        "momentum_lookback": 6,
        "min_volume_ratio": 1.0,
        "entry_score_threshold": 0.8,  # Raised from 0.6 to reduce frequent entries
        "weight_power": 1.4,
        "max_positions": 10,
        "max_position_percent": 0.12,
        "min_trade_value": 500,        # Raised from 200 to reduce small trades
        "min_data_coverage": 0.6,
        "cooldown_seconds": 1200,      # Extended from 600 (10min) to 1200 (20min)
        "buy_price_buffer_pct": 0.005,
        "commission_per_share": 0.005,
        "min_commission": 1.0,
        "fee_per_order": 0.0,
        "slippage_bps": 5.0,
        "max_daily_cost_pct": 0.003,
        "max_daily_loss_pct": 0.02,
        "open_buffer_minutes": 5,
    }
    if not path or not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = defaults.copy()
        merged.update(data or {})
        return merged
    except Exception as exc:
        logger.warning("Failed to load intraday config at %s: %s", path, exc)
        return defaults


def _compute_atr(df: pd.DataFrame, period: int) -> Optional[float]:
    if df is None or df.empty or len(df) < period + 2:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    if pd.isna(atr):
        return None
    return float(atr)


class IntradaySignalEngine:
    def __init__(self, config_path: Optional[str] = None, quote_client=None):
        self.config = _load_config(config_path)
        self.quote_client = quote_client or self._init_quote_client()
        self._current_run_id = None

    def set_run_id(self, run_id: str):
        """Set current run ID for signal recording."""
        self._current_run_id = run_id

    def _record_signals_to_supabase(self, signals: List[Dict], data_coverage: float):
        """Record intraday signals to Supabase for analysis."""
        supabase = _get_supabase()
        if not supabase or not supabase.is_enabled():
            return

        try:
            records = []
            for sig in signals:
                metrics = sig.get("metrics", {})
                record = {
                    "run_id": self._current_run_id,
                    "symbol": sig["symbol"],
                    "signal_type": sig.get("action"),
                    "score": sig.get("score"),
                    "component_scores": {
                        "ema_trend": 0.5 if metrics.get("fast_ema", 0) > metrics.get("slow_ema", 0) else 0.0,
                        "momentum": 0.3 if metrics.get("momentum", 0) > 0 else 0.0,
                        "volume": 0.2 if metrics.get("vol_ratio", 0) >= self.config.get("min_volume_ratio", 1.0) else 0.0,
                    },
                    "fast_ema": metrics.get("fast_ema"),
                    "slow_ema": metrics.get("slow_ema"),
                    "momentum_pct": metrics.get("momentum"),
                    "volume_ratio": metrics.get("vol_ratio"),
                    "atr": sig.get("atr"),
                    "price_at_signal": sig.get("price"),
                    "data_coverage_pct": data_coverage,
                    "was_executed": False,
                }
                records.append(record)

            if records:
                supabase.batch_insert_intraday_signals(records, self._current_run_id)
                logger.debug(f"Recorded {len(records)} intraday signals to Supabase")
        except Exception as e:
            logger.warning(f"Failed to record intraday signals: {e}")

    def _init_quote_client(self):
        if SETTINGS.data_source.lower() not in ("tiger", "auto"):
            return None
        try:
            from tigeropen.quote.quote_client import QuoteClient
            from tigeropen.tiger_open_config import TigerOpenClientConfig
        except Exception:
            return None
        props_dir = str((Path(__file__).parent.parent / "props").resolve())
        try:
            client_config = TigerOpenClientConfig(props_path=props_dir)
            return QuoteClient(client_config)
        except Exception as exc:
            logger.warning("Failed to init Tiger QuoteClient: %s", exc)
            return None

    def _score_symbol(self, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        fast_ema = close.ewm(span=self.config["fast_ema"], adjust=False).mean()
        slow_ema = close.ewm(span=self.config["slow_ema"], adjust=False).mean()
        if len(close) <= self.config["momentum_lookback"]:
            return 0.0, {}

        momentum = close.iloc[-1] / close.iloc[-1 - self.config["momentum_lookback"]] - 1.0
        vol_ma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = (volume.iloc[-1] / vol_ma) if vol_ma and vol_ma > 0 else 0.0

        score = 0.0
        if fast_ema.iloc[-1] > slow_ema.iloc[-1]:
            score += 0.5
        if momentum > 0:
            score += 0.3
        if vol_ratio >= self.config["min_volume_ratio"]:
            score += 0.2

        metrics = {
            "fast_ema": float(fast_ema.iloc[-1]),
            "slow_ema": float(slow_ema.iloc[-1]),
            "momentum": float(momentum),
            "vol_ratio": float(vol_ratio),
        }
        return float(score), metrics

    def generate_targets(
        self,
        symbols: List[str],
        current_positions: List[Dict],
        available_funds: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = state or {}
        try:
            market_manager = get_market_manager(SETTINGS.primary_market)
            today = market_manager.get_current_market_time().date().isoformat()
        except Exception:
            today = datetime.now(timezone.utc).date().isoformat()

        equity = float(available_funds) + sum(p.get("market_value", 0.0) for p in current_positions)
        day_start_date = state.get("day_start_date")
        if day_start_date != today:
            state = {
                "day_start_date": today,
                "day_start_equity": equity,
                "trailing_highs": {},
                "last_trade_ts": {},
                "risk_paused": False,
                "risk_reason": None,
            }

        day_start_equity = state.get("day_start_equity") or equity
        loss_pct = (equity - day_start_equity) / day_start_equity if day_start_equity else 0.0
        max_daily_loss_pct = float(self.config["max_daily_loss_pct"])
        if loss_pct <= -max_daily_loss_pct:
            state["risk_paused"] = True
            state["risk_reason"] = f"Daily loss limit hit ({loss_pct:.2%})"
            return {
                "risk_paused": True,
                "risk_reason": state["risk_reason"],
                "state": state,
                "equity": equity,
                "loss_pct": loss_pct,
                "target_positions": [],
                "signals": [],
                "cooldown_blocked": {},
                "data_coverage": 0.0,
            }

        trailing_highs = state.get("trailing_highs", {})
        last_trade_ts = state.get("last_trade_ts", {})
        cooldown_seconds = int(self.config["cooldown_seconds"])
        cooldown_blocked = {}
        now_ts = int(time.time())
        for symbol, ts in last_trade_ts.items():
            if now_ts - ts < cooldown_seconds:
                cooldown_blocked[symbol] = cooldown_seconds - (now_ts - ts)

        symbols = list(dict.fromkeys([s for s in symbols if s]))
        if not symbols:
            return {
                "risk_paused": False,
                "state": state,
                "equity": equity,
                "loss_pct": loss_pct,
                "target_positions": [],
                "signals": [],
                "cooldown_blocked": cooldown_blocked,
                "data_coverage": 0.0,
            }

        bars_map = fetch_batch_history(
            quote_client=self.quote_client,
            symbols=symbols,
            period=self.config["signal_period"],
            limit=int(self.config["lookback_bars"]),
            dry_run=SETTINGS.dry_run,
            max_concurrent=5,
            delay_between_requests=0.05,
        )

        positions_map = {p.get("symbol"): p for p in current_positions if p.get("symbol")}
        candidates = []
        signals = []
        valid_symbols = 0
        required_len = max(int(self.config["slow_ema"]), int(self.config["atr_period"])) + 5

        # Stale data threshold (in seconds) - data older than this is considered stale
        stale_threshold_seconds = int(self.config.get("stale_data_threshold", 900))  # Default 15 minutes
        current_time = datetime.now()
        stale_symbols = 0

        for symbol in symbols:
            df = bars_map.get(symbol)
            if df is None or df.empty:
                continue
            if len(df) < required_len:
                continue

            # CRITICAL: Check data freshness (timestamp validation)
            try:
                # Get the timestamp of the last bar
                if 'time' in df.columns:
                    last_bar_time = pd.to_datetime(df['time'].iloc[-1])
                elif df.index.name == 'time' or isinstance(df.index, pd.DatetimeIndex):
                    last_bar_time = df.index[-1]
                else:
                    # Try to infer from index
                    last_bar_time = pd.to_datetime(df.index[-1])

                # Make timezone-aware comparison if needed
                if last_bar_time.tzinfo is not None:
                    current_time_tz = current_time.replace(tzinfo=last_bar_time.tzinfo)
                    time_diff = (current_time_tz - last_bar_time).total_seconds()
                else:
                    time_diff = (current_time - last_bar_time).total_seconds()

                if time_diff > stale_threshold_seconds:
                    stale_symbols += 1
                    if stale_symbols <= 5:  # Only log first 5 to avoid spam
                        logger.warning(f"[SIGNAL] Skipping {symbol}: data is {time_diff/60:.1f} minutes stale (threshold: {stale_threshold_seconds/60:.1f} min)")
                    continue

            except Exception as ts_error:
                # If we can't determine timestamp, log warning but continue
                logger.warning(f"[SIGNAL] Could not verify data freshness for {symbol}: {ts_error}")

            valid_symbols += 1

            score, metrics = self._score_symbol(df)
            last_close = float(df["close"].iloc[-1])
            atr = _compute_atr(df, int(self.config["atr_period"]))

            position = positions_map.get(symbol)
            entry_price = float(position.get("average_cost", 0.0)) if position else 0.0

            trail_high = trailing_highs.get(symbol, last_close)
            if position:
                trail_high = max(trail_high, last_close)
                trailing_highs[symbol] = trail_high

            exit_reason = None
            if position and atr:
                trail_stop = trail_high - atr * float(self.config["trail_atr"])
                hard_stop = entry_price - atr * float(self.config["hard_stop_atr"]) if entry_price else None
                if last_close <= trail_stop:
                    exit_reason = "trailing_stop"
                elif hard_stop and last_close <= hard_stop:
                    exit_reason = "hard_stop"

            fast_above = metrics.get("fast_ema", 0.0) > metrics.get("slow_ema", 0.0)
            trend_ok = fast_above and metrics.get("momentum", 0.0) >= 0
            entry_threshold = float(self.config["entry_score_threshold"])

            if position and not trend_ok and not exit_reason:
                exit_reason = "trend_break"

            if exit_reason:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "EXIT",
                        "score": score,
                        "price": last_close,
                        "reason": exit_reason,
                    }
                )
                continue

            if position or score >= entry_threshold:
                candidates.append(
                    {
                        "symbol": symbol,
                        "score": score,
                        "price": last_close,
                        "metrics": metrics,
                        "atr": atr,
                    }
                )
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "HOLD" if position else "ENTER",
                        "score": score,
                        "price": last_close,
                        "reason": "trend_ok",
                    }
                )

        max_positions = int(self.config["max_positions"])
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:max_positions]
        data_coverage = valid_symbols / len(symbols) if symbols else 0.0

        weights = {}
        if candidates:
            weight_power = float(self.config["weight_power"])
            raw_scores = [max(c["score"], 0.01) ** weight_power for c in candidates]
            raw_sum = sum(raw_scores)
            if raw_sum > 0:
                for idx, c in enumerate(candidates):
                    weight = raw_scores[idx] / raw_sum
                    weight = min(weight, float(self.config["max_position_percent"]))
                    weights[c["symbol"]] = weight

        target_positions = []
        for candidate in candidates:
            symbol = candidate["symbol"]
            weight = weights.get(symbol, 0.0)
            target_value = equity * weight
            target_positions.append(
                {
                    "symbol": symbol,
                    "price": candidate["price"],
                    "target_weight": weight,
                    "target_value": target_value,
                    "score": candidate["score"],
                }
            )

        state["trailing_highs"] = trailing_highs
        state["last_trade_ts"] = last_trade_ts
        state["risk_paused"] = False
        state["risk_reason"] = None

        # Enrich signals with metrics for recording
        signals_with_metrics = []
        for sig in signals:
            # Find matching candidate for metrics
            candidate = next((c for c in candidates if c["symbol"] == sig["symbol"]), None)
            if candidate:
                sig["metrics"] = candidate.get("metrics", {})
                sig["atr"] = candidate.get("atr")
            signals_with_metrics.append(sig)

        # Record signals to Supabase for analysis
        self._record_signals_to_supabase(signals_with_metrics, data_coverage)

        return {
            "risk_paused": False,
            "state": state,
            "equity": equity,
            "loss_pct": loss_pct,
            "target_positions": target_positions,
            "signals": signals,
            "cooldown_blocked": cooldown_blocked,
            "data_coverage": data_coverage,
        }
