"""
Supabase client for hot data persistence in Quant Trading System.

This module provides async-compatible CRUD operations for all trading data tables.
Data is automatically written by runner.py tasks and archived by archival_job.py.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Lazy import to avoid startup errors if supabase not installed
_supabase_client = None

def get_supabase_client():
    """Get or create Supabase client singleton."""
    global _supabase_client
    if _supabase_client is None:
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            if url and key:
                _supabase_client = create_client(url, key)
                logger.info(f"Supabase client initialized: {url[:30]}...")
            else:
                logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        except ImportError:
            logger.warning("supabase package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
    return _supabase_client


def supabase_enabled():
    """Check if Supabase is enabled and configured."""
    enabled = os.getenv("SUPABASE_ENABLED", "true").lower() == "true"
    return enabled and get_supabase_client() is not None


def safe_supabase(default_return=None):
    """Decorator to safely execute Supabase operations with fallback."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not supabase_enabled():
                return default_return
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Supabase error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


class SupabaseClient:
    """Supabase client wrapper with CRUD operations for all tables."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_supabase_client()
        return self._client

    def is_enabled(self) -> bool:
        return supabase_enabled()

    # ============================================
    # RUNS TABLE
    # ============================================

    @safe_supabase(default_return=None)
    def insert_run(self, run_type: str, metadata: Dict = None) -> Optional[str]:
        """Insert a new task run and return its ID."""
        data = {
            "run_type": run_type,
            "started_at": datetime.utcnow().isoformat(),
            "status": "running",
            "metadata": metadata or {}
        }
        result = self.client.table("runs").insert(data).execute()
        run_id = result.data[0]["id"] if result.data else None
        logger.debug(f"Inserted run: {run_type} -> {run_id}")
        return run_id

    @safe_supabase()
    def complete_run(self, run_id: str, duration_ms: int, error: str = None):
        """Mark a run as completed or failed."""
        if not run_id:
            return
        data = {
            "ended_at": datetime.utcnow().isoformat(),
            "duration_ms": duration_ms,
            "status": "error" if error else "completed",
            "error_message": error
        }
        self.client.table("runs").update(data).eq("id", run_id).execute()
        logger.debug(f"Completed run: {run_id} ({duration_ms}ms)")

    # ============================================
    # ORDERS TABLE
    # ============================================

    @safe_supabase(default_return=None)
    def insert_order(self, order: Dict, run_id: str = None) -> Optional[str]:
        """Insert an order and return its ID."""
        data = {
            "external_id": order.get("external_id") or order.get("tiger_order_id"),
            "symbol": order["symbol"],
            "side": order.get("side") or order.get("action"),
            "order_type": order.get("order_type") or order.get("type", "MARKET"),
            "quantity": order["quantity"],
            "price": order.get("price"),
            "stop_price": order.get("stop_price"),
            "status": order.get("status", "PENDING"),
            "filled_quantity": order.get("filled_quantity", 0),
            "avg_fill_price": order.get("avg_fill_price"),
            "commission": order.get("commission"),
            "run_id": run_id
        }
        result = self.client.table("orders").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    @safe_supabase()
    def update_order_status(self, order_id: str, status: str,
                            filled_qty: int = None, avg_price: float = None,
                            commission: float = None):
        """Update order status and fill information."""
        if not order_id:
            return
        data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if filled_qty is not None:
            data["filled_quantity"] = filled_qty
        if avg_price is not None:
            data["avg_fill_price"] = avg_price
        if commission is not None:
            data["commission"] = commission
        self.client.table("orders").update(data).eq("id", order_id).execute()

    @safe_supabase()
    def batch_insert_orders(self, orders: List[Dict], run_id: str = None):
        """Insert multiple orders at once."""
        if not orders:
            return
        data = [{
            "external_id": o.get("external_id") or o.get("tiger_order_id"),
            "symbol": o["symbol"],
            "side": o.get("side") or o.get("action"),
            "order_type": o.get("order_type") or o.get("type", "MARKET"),
            "quantity": o["quantity"],
            "price": o.get("price"),
            "status": o.get("status", "PENDING"),
            "run_id": run_id
        } for o in orders]
        self.client.table("orders").insert(data).execute()

    # ============================================
    # FILLS TABLE
    # ============================================

    @safe_supabase(default_return=None)
    def insert_fill(self, fill: Dict) -> Optional[str]:
        """Insert an order fill."""
        data = {
            "order_id": fill["order_id"],
            "fill_price": fill["fill_price"],
            "fill_quantity": fill["fill_quantity"],
            "commission": fill.get("commission", 0)
        }
        result = self.client.table("fills").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    # ============================================
    # POSITIONS TABLE
    # ============================================

    @safe_supabase()
    def snapshot_positions(self, positions: List[Dict]):
        """Insert current positions snapshot."""
        if not positions:
            return
        now = datetime.utcnow().isoformat()
        data = [{
            "symbol": p["symbol"],
            "quantity": p.get("quantity", 0),
            "avg_cost": p.get("average_cost") or p.get("avg_cost", 0),
            "market_price": p.get("market_price", 0),
            "market_value": p.get("market_value", 0),
            "unrealized_pnl": p.get("unrealized_pnl", 0),
            "position_type": p.get("position_type", "REAL"),
            "snapshot_at": now
        } for p in positions]
        self.client.table("positions").insert(data).execute()
        logger.debug(f"Snapshotted {len(positions)} positions")

    # ============================================
    # TRADE SIGNALS TABLE
    # ============================================

    @safe_supabase(default_return=None)
    def insert_trade_signal(self, signal: Dict) -> Optional[str]:
        """Insert a trade signal (executed or not)."""
        data = {
            "run_id": signal.get("run_id"),
            "symbol": signal["symbol"],
            "signal_type": signal.get("signal_type") or signal.get("action"),
            "strategy_name": signal.get("strategy_name") or signal.get("strategy", "unknown"),
            "score": signal["score"],
            "component_scores": signal.get("component_scores", {}),
            "price_at_signal": signal.get("price_at_signal") or signal.get("price"),
            "volume_at_signal": signal.get("volume_at_signal") or signal.get("volume"),
            "market_cap": signal.get("market_cap"),
            "sector": signal.get("sector"),
            "reasoning": signal.get("reasoning") or signal.get("reason"),
            "was_executed": signal.get("was_executed", False),
            "order_id": signal.get("order_id")
        }
        result = self.client.table("trade_signals").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    @safe_supabase()
    def update_signal_executed(self, signal_id: str, order_id: str):
        """Mark a signal as executed with its order ID."""
        if not signal_id:
            return
        self.client.table("trade_signals").update({
            "was_executed": True,
            "order_id": order_id
        }).eq("id", signal_id).execute()

    # ============================================
    # EXECUTION ANALYSIS TABLE
    # ============================================

    @safe_supabase(default_return=None)
    def insert_execution_analysis(self, analysis: Dict) -> Optional[str]:
        """Insert execution quality analysis."""
        data = {
            "order_id": analysis["order_id"],
            "signal_price": analysis.get("signal_price"),
            "arrival_price": analysis.get("arrival_price"),
            "fill_price": analysis.get("fill_price"),
            "vwap_price": analysis.get("vwap_price") or analysis.get("vwap"),
            "twap_price": analysis.get("twap_price") or analysis.get("twap"),
            "slippage_bps": analysis.get("slippage_bps"),
            "market_impact_bps": analysis.get("market_impact_bps"),
            "implementation_shortfall": analysis.get("implementation_shortfall"),
            "execution_duration_ms": analysis.get("execution_duration_ms"),
            "market_volatility": analysis.get("market_volatility"),
            "spread_bps": analysis.get("spread_bps"),
            "volume_participation": analysis.get("volume_participation")
        }
        result = self.client.table("execution_analysis").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    # ============================================
    # METRICS SNAPSHOTS TABLE
    # ============================================

    @safe_supabase()
    def insert_metrics(self, metrics: Dict):
        """Insert real-time metrics snapshot."""
        data = {
            "metrics": metrics,
            "recorded_at": datetime.utcnow().isoformat()
        }
        self.client.table("metrics_snapshots").insert(data).execute()
        logger.debug("Inserted metrics snapshot")

    # ============================================
    # SELECTION RESULTS TABLE
    # ============================================

    @safe_supabase()
    def insert_selection(self, run_id: str, strategy: str,
                         picks: List[Dict], total: int):
        """Insert stock selection results."""
        data = {
            "run_id": run_id,
            "strategy_type": strategy,
            "top_picks": picks,
            "total_evaluated": total
        }
        self.client.table("selection_results").insert(data).execute()
        logger.debug(f"Inserted selection: {len(picks)} picks from {total} evaluated")

    # ============================================
    # STRATEGY CONFIG TABLE
    # ============================================

    @safe_supabase()
    def upsert_strategy_config(self, name: str, params: Dict, weights: Dict = None):
        """Upsert strategy configuration."""
        data = {
            "strategy_name": name,
            "parameters": params,
            "weights": weights or {},
            "is_active": True,
            "updated_at": datetime.utcnow().isoformat()
        }
        self.client.table("strategy_configs").upsert(
            data, on_conflict="strategy_name"
        ).execute()

    # ============================================
    # DAILY PERFORMANCE TABLE
    # ============================================

    @safe_supabase()
    def insert_daily_performance(self, perf: Dict):
        """Insert daily performance summary."""
        data = {
            "date": str(perf["date"]),
            "starting_equity": perf["starting_equity"],
            "ending_equity": perf["ending_equity"],
            "daily_pnl": perf.get("daily_pnl") or perf.get("pnl"),
            "daily_return": perf.get("daily_return") or perf.get("return"),
            "cumulative_return": perf.get("cumulative_return"),
            "drawdown": perf.get("drawdown"),
            "max_drawdown": perf.get("max_drawdown"),
            "sharpe_ratio": perf.get("sharpe_ratio") or perf.get("sharpe"),
            "sortino_ratio": perf.get("sortino_ratio") or perf.get("sortino"),
            "trades_count": perf.get("trades_count") or perf.get("trades", 0),
            "win_count": perf.get("win_count") or perf.get("wins", 0),
            "loss_count": perf.get("loss_count") or perf.get("losses", 0),
            "win_rate": perf.get("win_rate"),
            "avg_win": perf.get("avg_win"),
            "avg_loss": perf.get("avg_loss"),
            "profit_factor": perf.get("profit_factor"),
            "sector_pnl": perf.get("sector_pnl") or perf.get("sector_breakdown", {}),
            "strategy_pnl": perf.get("strategy_pnl") or perf.get("strategy_breakdown", {}),
            "risk_metrics": perf.get("risk_metrics", {})
        }
        self.client.table("daily_performance").upsert(
            data, on_conflict="date"
        ).execute()

    # ============================================
    # STRATEGY PERFORMANCE TABLE
    # ============================================

    @safe_supabase()
    def insert_strategy_performance(self, perf: Dict):
        """Insert strategy performance for a day."""
        data = {
            "strategy_name": perf["strategy_name"],
            "date": str(perf["date"]),
            "signals_generated": perf.get("signals_generated", 0),
            "signals_executed": perf.get("signals_executed", 0),
            "hit_rate": perf.get("hit_rate"),
            "avg_return": perf.get("avg_return"),
            "total_pnl": perf.get("total_pnl") or perf.get("pnl"),
            "sharpe_contribution": perf.get("sharpe_contribution") or perf.get("sharpe_contrib"),
            "drawdown_contribution": perf.get("drawdown_contribution"),
            "sector_exposure": perf.get("sector_exposure", {}),
            "factor_exposure": perf.get("factor_exposure", {})
        }
        self.client.table("strategy_performance").upsert(
            data, on_conflict="strategy_name,date"
        ).execute()

    # ============================================
    # MARKET REGIMES TABLE
    # ============================================

    @safe_supabase()
    def insert_market_regime(self, regime: Dict):
        """Insert market regime detection."""
        data = {
            "regime_type": regime.get("regime_type") or regime.get("type"),
            "vix_level": regime.get("vix_level") or regime.get("vix"),
            "market_trend": regime.get("market_trend") or regime.get("trend"),
            "sector_rotation": regime.get("sector_rotation", {}),
            "breadth_advance_decline": regime.get("breadth_advance_decline") or regime.get("breadth"),
            "fear_greed_index": regime.get("fear_greed_index") or regime.get("fear_greed"),
            "yield_curve_slope": regime.get("yield_curve_slope"),
            "dollar_index": regime.get("dollar_index")
        }
        self.client.table("market_regimes").insert(data).execute()

    # ============================================
    # AI TRAINING HISTORY TABLE
    # ============================================

    @safe_supabase()
    def insert_ai_training(self, training: Dict):
        """Insert AI model training record."""
        data = {
            "model_name": training["model_name"],
            "model_version": training.get("model_version") or training.get("version"),
            "training_started_at": training["training_started_at"],
            "training_ended_at": training.get("training_ended_at"),
            "training_duration_seconds": training.get("training_duration_seconds") or training.get("duration_seconds"),
            "hyperparameters": training.get("hyperparameters", {}),
            "training_samples": training.get("training_samples"),
            "validation_samples": training.get("validation_samples"),
            "train_loss": training.get("train_loss"),
            "val_loss": training.get("val_loss"),
            "train_accuracy": training.get("train_accuracy"),
            "val_accuracy": training.get("val_accuracy"),
            "sharpe_backtest": training.get("sharpe_backtest"),
            "max_drawdown_backtest": training.get("max_drawdown_backtest") or training.get("max_dd_backtest"),
            "live_sharpe": training.get("live_sharpe"),
            "live_accuracy": training.get("live_accuracy"),
            "days_in_production": training.get("days_in_production", 0),
            "improvement_vs_previous": training.get("improvement_vs_previous") or training.get("improvement"),
            "is_current_best": training.get("is_current_best") or training.get("is_best", False),
            "notes": training.get("notes")
        }
        self.client.table("ai_training_history").insert(data).execute()

    # ============================================
    # COMPLIANCE EVENTS TABLE
    # ============================================

    @safe_supabase()
    def insert_compliance_event(self, event: Dict):
        """Insert compliance violation or warning."""
        data = {
            "event_type": event.get("event_type") or event.get("type"),
            "rule_name": event.get("rule_name") or event.get("rule"),
            "severity": event["severity"],
            "description": event["description"],
            "symbol": event.get("symbol"),
            "position_value": event.get("position_value"),
            "limit_value": event.get("limit_value") or event.get("limit"),
            "actual_value": event.get("actual_value") or event.get("actual"),
            "breach_percentage": event.get("breach_percentage") or event.get("breach_pct"),
            "was_prevented": event.get("was_prevented", False),
            "action_taken": event.get("action_taken") or event.get("action")
        }
        self.client.table("compliance_events").insert(data).execute()

    # ============================================
    # FACTOR CROWDING HISTORY TABLE
    # ============================================

    @safe_supabase()
    def insert_factor_crowding(self, crowding: Dict):
        """Insert factor crowding analysis."""
        data = {
            "factor_name": crowding.get("factor_name") or crowding.get("factor"),
            "hhi": crowding.get("hhi"),
            "gini_coefficient": crowding.get("gini_coefficient") or crowding.get("gini"),
            "crowding_score": crowding.get("crowding_score"),
            "crowding_level": crowding.get("crowding_level"),
            "portfolio_exposure": crowding.get("portfolio_exposure"),
            "market_exposure": crowding.get("market_exposure")
        }
        self.client.table("factor_crowding_history").insert(data).execute()

    # ============================================
    # INTRADAY SIGNALS TABLE (Minute-level)
    # ============================================

    @safe_supabase(default_return=None)
    def insert_intraday_signal(self, signal: Dict) -> Optional[str]:
        """Insert an intraday signal with component scores for factor attribution."""
        data = {
            "run_id": signal.get("run_id"),
            "symbol": signal["symbol"],
            "signal_type": signal.get("signal_type") or signal.get("action"),
            "score": signal.get("score"),
            "component_scores": signal.get("component_scores") or signal.get("metrics", {}),
            "fast_ema": signal.get("fast_ema"),
            "slow_ema": signal.get("slow_ema"),
            "momentum_pct": signal.get("momentum_pct") or signal.get("momentum"),
            "volume_ratio": signal.get("volume_ratio") or signal.get("vol_ratio"),
            "atr": signal.get("atr"),
            "price_at_signal": signal.get("price_at_signal") or signal.get("price"),
            "volume_at_signal": signal.get("volume_at_signal"),
            "spread_bps": signal.get("spread_bps"),
            "data_freshness_seconds": signal.get("data_freshness_seconds"),
            "data_coverage_pct": signal.get("data_coverage_pct") or signal.get("data_coverage"),
            "was_executed": signal.get("was_executed", False),
            "order_id": signal.get("order_id")
        }
        result = self.client.table("intraday_signals").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    @safe_supabase()
    def batch_insert_intraday_signals(self, signals: List[Dict], run_id: str = None):
        """Insert multiple intraday signals at once."""
        if not signals:
            return
        data = [{
            "run_id": run_id or s.get("run_id"),
            "symbol": s["symbol"],
            "signal_type": s.get("signal_type") or s.get("action"),
            "score": s.get("score"),
            "component_scores": s.get("component_scores") or s.get("metrics", {}),
            "fast_ema": s.get("fast_ema") or (s.get("metrics", {}).get("fast_ema")),
            "slow_ema": s.get("slow_ema") or (s.get("metrics", {}).get("slow_ema")),
            "momentum_pct": s.get("momentum_pct") or s.get("momentum") or (s.get("metrics", {}).get("momentum")),
            "volume_ratio": s.get("volume_ratio") or s.get("vol_ratio") or (s.get("metrics", {}).get("vol_ratio")),
            "atr": s.get("atr"),
            "price_at_signal": s.get("price_at_signal") or s.get("price"),
            "data_coverage_pct": s.get("data_coverage_pct") or s.get("data_coverage"),
            "was_executed": s.get("was_executed", False)
        } for s in signals]
        self.client.table("intraday_signals").insert(data).execute()
        logger.debug(f"Inserted {len(signals)} intraday signals")

    @safe_supabase()
    def update_intraday_signal_outcome(self, signal_id: str, exit_price: float,
                                       exit_reason: str, pnl_bps: float,
                                       hold_duration_minutes: int):
        """Update signal with exit outcome for performance tracking."""
        if not signal_id:
            return
        data = {
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_bps": pnl_bps,
            "hold_duration_minutes": hold_duration_minutes
        }
        self.client.table("intraday_signals").update(data).eq("id", signal_id).execute()

    # ============================================
    # INTRADAY EXECUTIONS TABLE
    # ============================================

    @safe_supabase(default_return=None)
    def insert_intraday_execution(self, execution: Dict) -> Optional[str]:
        """Insert intraday execution quality analysis."""
        data = {
            "signal_id": execution.get("signal_id"),
            "order_id": execution.get("order_id"),
            "symbol": execution["symbol"],
            "action": execution.get("action") or execution.get("side"),
            "signal_price": execution.get("signal_price"),
            "limit_price": execution.get("limit_price"),
            "fill_price": execution.get("fill_price"),
            "vwap_5min": execution.get("vwap_5min") or execution.get("vwap"),
            "slippage_bps": execution.get("slippage_bps"),
            "market_impact_bps": execution.get("market_impact_bps"),
            "execution_time_ms": execution.get("execution_time_ms"),
            "order_quantity": execution.get("order_quantity") or execution.get("quantity"),
            "filled_quantity": execution.get("filled_quantity"),
            "volume_participation_pct": execution.get("volume_participation_pct"),
            "commission": execution.get("commission"),
            "slippage_cost": execution.get("slippage_cost"),
            "total_cost": execution.get("total_cost")
        }
        result = self.client.table("intraday_executions").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    # ============================================
    # INTRADAY RISK SNAPSHOTS TABLE
    # ============================================

    @safe_supabase()
    def insert_intraday_risk_snapshot(self, snapshot: Dict):
        """Insert 5-minute risk snapshot for intraday monitoring."""
        data = {
            "run_id": snapshot.get("run_id"),
            "equity": snapshot.get("equity"),
            "day_start_equity": snapshot.get("day_start_equity"),
            "daily_pnl": snapshot.get("daily_pnl"),
            "daily_loss_pct": snapshot.get("daily_loss_pct") or snapshot.get("loss_pct"),
            "positions_count": snapshot.get("positions_count"),
            "total_position_value": snapshot.get("total_position_value"),
            "cash_balance": snapshot.get("cash_balance") or snapshot.get("available_funds"),
            "buying_power": snapshot.get("buying_power"),
            "max_position_weight": snapshot.get("max_position_weight"),
            "es_97_5": snapshot.get("es_97_5") or snapshot.get("es"),
            "portfolio_beta": snapshot.get("portfolio_beta") or snapshot.get("beta"),
            "factor_hhi": snapshot.get("factor_hhi") or snapshot.get("hhi"),
            "daily_costs_total": snapshot.get("daily_costs_total"),
            "daily_cost_pct": snapshot.get("daily_cost_pct"),
            "circuit_breaker_active": snapshot.get("circuit_breaker_active") or snapshot.get("risk_paused", False),
            "halt_reason": snapshot.get("halt_reason") or snapshot.get("risk_reason")
        }
        self.client.table("intraday_risk_snapshots").insert(data).execute()
        logger.debug("Inserted intraday risk snapshot")

    # ============================================
    # SIGNAL PERFORMANCE DAILY TABLE
    # ============================================

    @safe_supabase()
    def upsert_signal_performance_daily(self, perf: Dict):
        """Upsert daily signal performance aggregation."""
        data = {
            "date": str(perf["date"]),
            "symbol": perf.get("symbol"),
            "total_signals": perf.get("total_signals", 0),
            "buy_signals": perf.get("buy_signals", 0),
            "sell_signals": perf.get("sell_signals", 0),
            "executed_signals": perf.get("executed_signals", 0),
            "win_count": perf.get("win_count", 0),
            "loss_count": perf.get("loss_count", 0),
            "win_rate": perf.get("win_rate"),
            "avg_win_bps": perf.get("avg_win_bps"),
            "avg_loss_bps": perf.get("avg_loss_bps"),
            "profit_factor": perf.get("profit_factor"),
            "avg_score": perf.get("avg_score"),
            "score_vs_outcome_corr": perf.get("score_vs_outcome_corr"),
            "ema_contribution": perf.get("ema_contribution"),
            "momentum_contribution": perf.get("momentum_contribution"),
            "volume_contribution": perf.get("volume_contribution")
        }
        self.client.table("signal_performance_daily").upsert(
            data, on_conflict="date,symbol"
        ).execute()

    # ============================================
    # INTRADAY QUERY METHODS
    # ============================================

    @safe_supabase(default_return=[])
    def get_intraday_signals(self, days: int = 1, symbol: str = None,
                             signal_type: str = None, limit: int = 500) -> List[Dict]:
        """Get intraday signals history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = self.client.table("intraday_signals")\
            .select("*")\
            .gte("created_at", since)\
            .order("created_at", desc=True)\
            .limit(limit)
        if symbol:
            query = query.eq("symbol", symbol.upper())
        if signal_type:
            query = query.eq("signal_type", signal_type)
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_intraday_executions(self, days: int = 1, symbol: str = None,
                                 limit: int = 100) -> List[Dict]:
        """Get intraday execution quality history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = self.client.table("intraday_executions")\
            .select("*")\
            .gte("created_at", since)\
            .order("created_at", desc=True)\
            .limit(limit)
        if symbol:
            query = query.eq("symbol", symbol.upper())
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_intraday_risk_snapshots(self, hours: int = 8, limit: int = 100) -> List[Dict]:
        """Get intraday risk snapshots."""
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        result = self.client.table("intraday_risk_snapshots")\
            .select("*")\
            .gte("snapshot_at", since)\
            .order("snapshot_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data

    @safe_supabase(default_return=[])
    def get_signal_performance_daily(self, days: int = 30, symbol: str = None) -> List[Dict]:
        """Get daily signal performance aggregation."""
        since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        query = self.client.table("signal_performance_daily")\
            .select("*")\
            .gte("date", since)\
            .order("date", desc=True)
        if symbol:
            query = query.eq("symbol", symbol.upper())
        return query.execute().data

    # ============================================
    # QUERY METHODS
    # ============================================

    @safe_supabase(default_return=[])
    def get_recent_orders(self, days: int = 7, limit: int = 100) -> List[Dict]:
        """Get recent orders."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        result = self.client.table("orders")\
            .select("*")\
            .gte("created_at", since)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data

    @safe_supabase(default_return=[])
    def get_latest_positions(self, limit: int = 50) -> List[Dict]:
        """Get most recent position snapshot."""
        result = self.client.table("positions")\
            .select("*")\
            .order("snapshot_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data

    @safe_supabase(default_return=[])
    def get_recent_runs(self, days: int = 7, run_type: str = None, limit: int = 100) -> List[Dict]:
        """Get recent task runs."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = self.client.table("runs")\
            .select("*")\
            .gte("started_at", since)\
            .order("started_at", desc=True)\
            .limit(limit)
        if run_type:
            query = query.eq("run_type", run_type)
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_daily_performance(self, days: int = 30) -> List[Dict]:
        """Get daily performance history."""
        since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        result = self.client.table("daily_performance")\
            .select("*")\
            .gte("date", since)\
            .order("date", desc=True)\
            .execute()
        return result.data

    @safe_supabase(default_return=[])
    def get_trade_signals(self, days: int = 7, strategy: str = None,
                          symbol: str = None, limit: int = 100,
                          executed_only: bool = False) -> List[Dict]:
        """Get trade signals history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = self.client.table("trade_signals")\
            .select("*")\
            .gte("created_at", since)\
            .order("created_at", desc=True)\
            .limit(limit)
        if strategy:
            query = query.eq("strategy_name", strategy)
        if symbol:
            query = query.eq("symbol", symbol.upper())
        if executed_only:
            query = query.eq("was_executed", True)
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_strategy_performance(self, days: int = 30, strategy: str = None) -> List[Dict]:
        """Get strategy performance history."""
        since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        query = self.client.table("strategy_performance")\
            .select("*")\
            .gte("date", since)\
            .order("date", desc=True)
        if strategy:
            query = query.eq("strategy_name", strategy)
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_compliance_events(self, days: int = 30, severity: str = None) -> List[Dict]:
        """Get compliance events history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = self.client.table("compliance_events")\
            .select("*")\
            .gte("detected_at", since)\
            .order("detected_at", desc=True)
        if severity:
            query = query.eq("severity", severity)
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_metrics_history(self, days: int = 7, limit: int = 100) -> List[Dict]:
        """Get metrics snapshots history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        result = self.client.table("metrics_snapshots")\
            .select("*")\
            .gte("recorded_at", since)\
            .order("recorded_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data

    @safe_supabase(default_return=[])
    def get_factor_crowding_history(self, days: int = 30, factor: str = None) -> List[Dict]:
        """Get factor crowding history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = self.client.table("factor_crowding_history")\
            .select("*")\
            .gte("recorded_at", since)\
            .order("recorded_at", desc=True)
        if factor:
            query = query.eq("factor_name", factor)
        return query.execute().data

    @safe_supabase(default_return=[])
    def get_selection_results(self, days: int = 30, limit: int = 50) -> List[Dict]:
        """Get selection results history."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        result = self.client.table("selection_results")\
            .select("*")\
            .gte("selected_at", since)\
            .order("selected_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data


# Singleton instance
supabase_client = SupabaseClient()


# ============================================
# UTILITY FUNCTIONS
# ============================================

def calculate_slippage_bps(signal_price: float, fill_price: float) -> Optional[float]:
    """Calculate slippage in basis points."""
    if not signal_price or not fill_price:
        return None
    return ((fill_price - signal_price) / signal_price) * 10000
