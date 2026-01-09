#!/usr/bin/env python3
"""
自动交易执行引擎 - 动态资金管理版本
基于AI选股推荐和当前持仓状态，执行买入/卖出决策
实施动态资金管理，移除硬编码限制
集成Adaptive Execution Engine实现智能执行
"""

import logging
import time
import os
import asyncio
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Tiger API imports
try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.trade.trade_client import TradeClient
    from tigeropen.quote.quote_client import QuoteClient
    from tigeropen.common.util.contract_utils import stock_contract
    from tigeropen.common.util.order_utils import market_order, limit_order
    from tigeropen.common.consts import Market, OrderStatus, OrderType
    TIGER_SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Tiger SDK not available: {e}")
    TIGER_SDK_AVAILABLE = False

from state_manager import append_log

# Import Market Time Manager for trading hours enforcement
try:
    import sys
    bot_path_mt = str((Path(__file__).parent.parent.parent / "bot").resolve())
    if bot_path_mt not in sys.path:
        sys.path.insert(0, bot_path_mt)
    from market_time import MarketTimeManager, MarketPhase, MarketType, US_MARKET
    MARKET_TIME_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Market Time Manager not available: {e}")
    MARKET_TIME_AVAILABLE = False
    MarketPhase = None
    US_MARKET = None

# Import Order Rate Limiter
try:
    from order_rate_limiter import get_rate_limiter
    ORDER_RATE_LIMITER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Order Rate Limiter not available: {e}")
    ORDER_RATE_LIMITER_AVAILABLE = False

# Lazy import for Supabase client
_supabase_client = None


def _get_supabase():
    """Get Supabase client singleton for execution recording."""
    global _supabase_client
    if _supabase_client is None:
        try:
            import sys
            backend_path = str(Path(__file__).parent.parent / "backend")
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from supabase_client import supabase_client
            _supabase_client = supabase_client
        except ImportError:
            _supabase_client = False
    return _supabase_client if _supabase_client else None


# Import Adaptive Execution Engine
try:
    import sys
    bot_path = str((Path(__file__).parent.parent.parent / "bot").resolve())
    if bot_path not in sys.path:
        sys.path.insert(0, bot_path)
    from adaptive_execution_engine import AdaptiveExecutionEngine, ExecutionOrder, ExecutionUrgency
    ADAPTIVE_EXECUTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Adaptive Execution Engine not available: {e}")
    ADAPTIVE_EXECUTION_AVAILABLE = False

# Large cap symbols (S&P 500 high-liquidity stocks) - use market orders
LARGE_CAP_SYMBOLS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'BRK.B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL',
    'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO', 'PEP', 'ABT', 'KO', 'NKE',
    'MRK', 'PFE', 'TMO', 'COST', 'AVGO', 'ACN', 'WMT', 'MCD', 'DHR',
    'ORCL', 'VZ', 'T', 'XOM', 'CVX', 'BAC', 'WFC', 'C', 'GS', 'MS',
    'AMD', 'QCOM', 'TXN', 'IBM', 'NOW', 'UBER', 'ABNB', 'SQ', 'SHOP'
}

class AutoTradingEngine:
    """自动交易执行引擎 - 动态资金管理版本"""

    def __init__(self, dry_run=True, max_position_value=None, max_daily_trades=10):
        """
        初始化交易引擎

        Args:
            dry_run: 是否为模拟模式
            max_position_value: 单个持仓最大价值($) - 如果为None则动态计算
            max_daily_trades: 每日最大交易次数
        """
        self.dry_run = dry_run

        # 动态资金管理配置
        self.trading_safety_factor = float(os.getenv('TRADING_SAFETY_FACTOR', '0.8'))
        self.min_cash_reserve = float(os.getenv('MIN_CASH_RESERVE', '5000'))
        self.max_position_percent = float(os.getenv('MAX_POSITION_PERCENT', '0.20'))
        self.fallback_buying_power = float(os.getenv('FALLBACK_BUYING_POWER', '50000'))
        self.max_daily_outflow_percent = float(os.getenv('MAX_DAILY_OUTFLOW_PERCENT', '0.80'))

        # 如果没有指定max_position_value，将动态计算
        self.max_position_value = max_position_value
        self.max_daily_trades = max_daily_trades
        self.daily_trade_count = 0
        self.trade_history = []
        self.daily_cost_total = 0.0
        self.last_equity = None

        # CRITICAL: Daily loss limit enforcement (Fix #7)
        # Updated: 8% limit as requested by user (was 3%)
        self.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', '0.08'))  # 8% default
        self.day_start_equity = None  # Set on first equity update of the day
        self.trading_halted = False  # Circuit breaker flag
        self.halt_reason = None
        self._last_reset_date = None  # Track when we last reset day start equity

        # CRITICAL: Order execution mutex lock - prevents concurrent order placement
        self._order_lock = threading.Lock()

        # CRITICAL: Order rate limiter - prevents excessive order placement
        self.rate_limiter = None
        if ORDER_RATE_LIMITER_AVAILABLE:
            self.rate_limiter = get_rate_limiter()
            append_log(f"[AUTO_TRADING] Order Rate Limiter initialized")

        # Transaction cost model (commission + fees + slippage)
        cost_config = self._load_cost_config()
        self.commission_per_share = float(os.getenv('COMMISSION_PER_SHARE', cost_config["commission_per_share"]))
        self.min_commission = float(os.getenv('MIN_COMMISSION', cost_config["min_commission"]))
        self.fee_per_order = float(os.getenv('FEE_PER_ORDER', cost_config["fee_per_order"]))
        self.slippage_bps = float(os.getenv('SLIPPAGE_BPS', cost_config["slippage_bps"]))
        self.max_daily_cost_pct = float(os.getenv('MAX_DAILY_COST_PCT', cost_config["max_daily_cost_pct"]))
        self.max_daily_cost_usd = float(os.getenv('MAX_DAILY_COST_USD', cost_config["max_daily_cost_usd"]))

        # 失败股票黑名单：记录连续失败的股票
        self.failed_symbols = {}  # {symbol: failure_count}
        self.max_failures = 1    # 最大连续失败次数（改为1次立即生效）

        # 订单去重追踪：防止重复提交相同订单
        self.submitted_orders = {}  # {symbol: {'action': str, 'qty': int, 'timestamp': datetime, 'order_id': str}}
        self.order_expiry_minutes = 5  # 订单记录过期时间（分钟）- 缩短以适应15秒交易周期

        # CRITICAL FIX: Account ID from environment variable (not hardcoded)
        self.account = os.getenv('ACCOUNT', '41169270')

        # Initialize Tiger API clients
        # DEBUG: Log initialization parameters to stderr for PM2 capture
        print(f"[AUTO_TRADING_DEBUG] __init__ called: dry_run={dry_run}, TIGER_SDK_AVAILABLE={TIGER_SDK_AVAILABLE}")
        if TIGER_SDK_AVAILABLE and not dry_run:
            print(f"[AUTO_TRADING_DEBUG] Calling _init_tiger_clients()")
            self._init_tiger_clients()
            print(f"[AUTO_TRADING_DEBUG] After _init_tiger_clients(): trade_client={'SET' if self.trade_client else 'NONE'}")
        else:
            print(f"[AUTO_TRADING_DEBUG] Skipping Tiger client init: TIGER_SDK={TIGER_SDK_AVAILABLE}, dry_run={dry_run}")
            self.trade_client = None
            self.quote_client = None

        # Initialize Adaptive Execution Engine
        self.adaptive_execution = None
        self.use_adaptive_execution = False
        if ADAPTIVE_EXECUTION_AVAILABLE:
            try:
                config_path = str((Path(__file__).parent.parent.parent / "config" / "execution_config.json").resolve())
                # Pass tiger_client for live order execution
                self.adaptive_execution = AdaptiveExecutionEngine(
                    config_path=config_path,
                    tiger_client=self.trade_client if hasattr(self, 'trade_client') else None
                )
                self.use_adaptive_execution = not dry_run  # Only use adaptive execution in live mode
                append_log(f"[AUTO_TRADING] Adaptive Execution Engine initialized (enabled: {self.use_adaptive_execution})")
                if self.trade_client and not dry_run:
                    append_log(f"[AUTO_TRADING] Tiger API client connected for live order execution")
            except Exception as e:
                append_log(f"[AUTO_TRADING] Adaptive Execution initialization failed: {e}")
                self.use_adaptive_execution = False

        append_log(f"[AUTO_TRADING] Dynamic Fund Management Engine initialized")
        append_log(f"[AUTO_TRADING] DRY_RUN: {dry_run}")
        append_log(f"[AUTO_TRADING] Adaptive Execution: {self.use_adaptive_execution}")
        append_log(f"[AUTO_TRADING] Safety Factor: {self.trading_safety_factor:.1%}")
        append_log(f"[AUTO_TRADING] Min Cash Reserve: ${self.min_cash_reserve:,.2f}")
        append_log(f"[AUTO_TRADING] Max Position %: {self.max_position_percent:.1%}")
        append_log(f"[AUTO_TRADING] Max Daily Outflow %: {self.max_daily_outflow_percent:.1%}")

    def _load_cost_config(self) -> Dict[str, float]:
        defaults = {
            "commission_per_share": 0.0,
            "min_commission": 0.0,
            "fee_per_order": 0.0,
            "slippage_bps": 5.0,
            "max_daily_cost_pct": 0.0,
            "max_daily_cost_usd": 0.0,
        }
        config_path = os.getenv(
            "TRADING_COST_CONFIG",
            str((Path(__file__).parent.parent.parent / "config" / "trading_costs.json").resolve()),
        )
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    defaults.update({k: float(v) for k, v in data.items() if k in defaults})
        except Exception as exc:
            append_log(f"[AUTO_TRADING] Failed to load trading cost config: {exc}")
        return defaults

    def get_raw_buying_power(self):
        """获取原始账户购买力（未应用安全系数）"""
        if self.dry_run or not self.trade_client:
            # 模拟模式使用fallback值
            raw_buying_power = self.fallback_buying_power
            append_log(f"[AUTO_TRADING] DRY RUN mode: using fallback buying power ${raw_buying_power:,.2f}")
            return raw_buying_power

        try:
            assets_result = self.trade_client.get_assets()

            if hasattr(assets_result, '__iter__') and not isinstance(assets_result, str):
                assets_list = list(assets_result)
                if assets_list:
                    asset = assets_list[0]
                else:
                    append_log(f"[AUTO_TRADING] No assets found, using fallback")
                    return self.fallback_buying_power
            else:
                asset = assets_result

            if hasattr(asset, 'summary'):
                summary = asset.summary
            else:
                summary = asset

            # 优先获取可用资金 (available_funds from segments)
            # This is the actual available funds after accrued cash/fees
            available_funds = None

            # Try to get from segments['S'] (US Securities)
            if hasattr(asset, 'segments') and 'S' in asset.segments:
                seg = asset.segments['S']
                available_funds = getattr(seg, 'available_funds', None)
                if available_funds is not None and available_funds > 0:
                    raw_buying_power = float(available_funds)
                    append_log(f"[AUTO_TRADING] Tiger API available_funds (from segments): ${raw_buying_power:,.2f}")
                    return raw_buying_power

            # 备用1：使用现金余额
            cash_balance = getattr(summary, 'cash', None)
            if cash_balance is not None and cash_balance > 0:
                raw_buying_power = float(cash_balance)
                append_log(f"[AUTO_TRADING] Tiger API cash balance (fallback): ${raw_buying_power:,.2f}")
                return raw_buying_power

            # 备用2：获取购买力
            buying_power = getattr(summary, 'buying_power', None)
            if buying_power is not None and buying_power > 0:
                raw_buying_power = float(buying_power)
                append_log(f"[AUTO_TRADING] Tiger API buying_power (fallback): ${raw_buying_power:,.2f}")
                return raw_buying_power

            # 最终备用
            raw_buying_power = self.fallback_buying_power
            append_log(f"[AUTO_TRADING] Tiger API failed, using fallback: ${raw_buying_power:,.2f}")
            return raw_buying_power

        except Exception as e:
            append_log(f"[AUTO_TRADING] Error getting buying power from Tiger API: {e}")
            raw_buying_power = self.fallback_buying_power
            append_log(f"[AUTO_TRADING] Using fallback buying power: ${raw_buying_power:,.2f}")
            return raw_buying_power

    def get_buying_power(self):
        """
        获取可用于交易的购买力 - 使用原始购买力，减去待处理订单
        
        CRITICAL FIX #11: Deducts pending order value from buying power
        to prevent over-committing funds during partial fills.
        """
        raw_buying_power = self.get_raw_buying_power()
        
        # CRITICAL: Subtract pending order value to get true available buying power
        pending_value = self.get_pending_order_value()
        adjusted_buying_power = max(0, raw_buying_power - pending_value)

        # 计算安全购买力仅用于记录
        safe_buying_power = max(0, (adjusted_buying_power - self.min_cash_reserve) * self.trading_safety_factor)

        append_log(f"[AUTO_TRADING] Raw buying power: ${raw_buying_power:,.2f}")
        if pending_value > 0:
            append_log(f"[AUTO_TRADING] Pending order value (deducted): ${pending_value:,.2f}")
            append_log(f"[AUTO_TRADING] Adjusted buying power: ${adjusted_buying_power:,.2f}")
        append_log(f"[AUTO_TRADING] Safety factor: {self.trading_safety_factor:.1%}")
        append_log(f"[AUTO_TRADING] Min cash reserve: ${self.min_cash_reserve:,.2f}")
        append_log(f"[AUTO_TRADING] Safe buying power: ${safe_buying_power:,.2f}")
        append_log(f"[AUTO_TRADING] Available buying power: ${adjusted_buying_power:,.2f}")

        return adjusted_buying_power

    def get_max_position_value(self, available_buying_power):
        """动态计算单个持仓最大价值"""
        if self.max_position_value is None:
            # 根据可用购买力动态计算
            max_allocation = available_buying_power * self.max_position_percent
        else:
            # 使用固定值和百分比的较小值
            max_allocation = min(self.max_position_value, available_buying_power * self.max_position_percent)

        append_log(f"[AUTO_TRADING] Max position value: ${max_allocation:,.2f}")
        return max_allocation

    def get_pending_order_value(self) -> float:
        """
        Calculate total value of pending/partially filled BUY orders.
        
        CRITICAL FIX #11: Track partial fills to avoid over-committing buying power.
        Pending orders reduce available buying power until they're filled or cancelled.
        
        Returns:
            Total value of pending buy orders (remaining unfilled value)
        """
        pending_value = 0.0
        
        for symbol, order_info in self.submitted_orders.items():
            # Only count BUY orders (SELL orders release funds)
            action = order_info.get('action', '').upper()
            if action != 'BUY':
                continue
            
            status = order_info.get('status', 'unknown')
            # Count pending, submitted, and partially filled orders
            if status not in {'pending', 'submitted', 'initial', 'unknown', 'partial', 'partially_filled'}:
                continue
            
            # Get order details
            qty = int(order_info.get('qty', 0) or 0)
            filled_qty = int(order_info.get('filled_qty', 0) or 0)
            price = float(order_info.get('price', 0) or 0)
            
            # Calculate remaining unfilled value
            remaining_qty = qty - filled_qty
            if remaining_qty > 0 and price > 0:
                remaining_value = remaining_qty * price
                pending_value += remaining_value
                append_log(f"[PENDING_VALUE] {symbol}: {remaining_qty} shares @ ${price:.2f} = ${remaining_value:,.2f}")
        
        if pending_value > 0:
            append_log(f"[PENDING_VALUE] Total pending BUY value: ${pending_value:,.2f}")
        
        return pending_value

    def _init_tiger_clients(self):
        """初始化Tiger API客户端"""
        try:
            # Use props configuration
            props_dir = str((Path(__file__).parent.parent.parent / "props").resolve())
            client_config = TigerOpenClientConfig(props_path=props_dir)

            self.trade_client = TradeClient(client_config)
            self.quote_client = QuoteClient(client_config)

            append_log(f"[AUTO_TRADING] Tiger API clients initialized")
        except Exception as e:
            append_log(f"[AUTO_TRADING] Error initializing Tiger clients: {e}")
            self.trade_client = None
            self.quote_client = None

    def _cleanup_expired_orders(self):
        """清理过期的订单记录"""
        current_time = datetime.now()
        expired_symbols = []

        for symbol, order_info in self.submitted_orders.items():
            order_time = order_info.get('timestamp')
            if order_time and (current_time - order_time).total_seconds() > self.order_expiry_minutes * 60:
                expired_symbols.append(symbol)

        for symbol in expired_symbols:
            del self.submitted_orders[symbol]
            append_log(f"[ORDER_TRACKING] Expired order record removed: {symbol}")

    def update_cost_config(self, cost_config: Optional[Dict[str, float]] = None):
        """Update transaction cost configuration overrides."""
        if not cost_config:
            return
        if "commission_per_share" in cost_config:
            self.commission_per_share = float(cost_config["commission_per_share"])
        if "min_commission" in cost_config:
            self.min_commission = float(cost_config["min_commission"])
        if "fee_per_order" in cost_config:
            self.fee_per_order = float(cost_config["fee_per_order"])
        if "slippage_bps" in cost_config:
            self.slippage_bps = float(cost_config["slippage_bps"])
        if "max_daily_cost_pct" in cost_config:
            self.max_daily_cost_pct = float(cost_config["max_daily_cost_pct"])
        if "max_daily_cost_usd" in cost_config:
            self.max_daily_cost_usd = float(cost_config["max_daily_cost_usd"])

    def update_equity(self, equity: Optional[float]):
        """Store latest equity and check daily loss limit."""
        if equity is None:
            self.last_equity = None
            return
        try:
            equity_value = float(equity)
        except (TypeError, ValueError):
            equity_value = None
        self.last_equity = equity_value if equity_value and equity_value > 0 else None

        # CRITICAL FIX #7: Track day start equity and check daily loss limit
        if equity_value and equity_value > 0:
            today = datetime.now().date()
            # Reset day start equity at beginning of new trading day
            if self._last_reset_date != today:
                self.day_start_equity = equity_value
                self._last_reset_date = today
                self.trading_halted = False  # Reset circuit breaker for new day
                self.halt_reason = None
                append_log(f"[DAILY_LOSS] New trading day - start equity: ${equity_value:,.2f}")

            # Check daily loss limit
            if self.day_start_equity and self.day_start_equity > 0:
                loss_pct = (equity_value - self.day_start_equity) / self.day_start_equity
                if loss_pct <= -self.daily_loss_limit and not self.trading_halted:
                    self.trading_halted = True
                    self.halt_reason = f"Daily loss limit hit: {loss_pct:.2%} (limit: {-self.daily_loss_limit:.2%})"
                    append_log(f"[DAILY_LOSS] CIRCUIT BREAKER TRIGGERED: {self.halt_reason}")

    def check_daily_loss_limit(self) -> Tuple[bool, Optional[str]]:
        """
        Check if daily loss limit has been exceeded.
        
        Returns:
            Tuple of (can_trade: bool, reason: Optional[str])
        """
        if self.trading_halted:
            return False, self.halt_reason
        
        if self.last_equity and self.day_start_equity and self.day_start_equity > 0:
            loss_pct = (self.last_equity - self.day_start_equity) / self.day_start_equity
            if loss_pct <= -self.daily_loss_limit:
                self.trading_halted = True
                self.halt_reason = f"Daily loss limit hit: {loss_pct:.2%} (limit: {-self.daily_loss_limit:.2%})"
                return False, self.halt_reason
        
        return True, None

    def reset_daily_costs(self):
        """Reset daily cost counters."""
        self.daily_cost_total = 0.0

    def _normalize_order_status(self, status) -> str:
        if status is None:
            return "unknown"
        return str(status).lower().replace("orderstatus.", "")

    def _estimate_transaction_cost(self, qty: int, price: float) -> float:
        if qty <= 0 or price <= 0:
            return 0.0
        notional = qty * price
        commission = max(self.commission_per_share * qty, self.min_commission)
        fee_total = self.fee_per_order
        slippage = notional * (self.slippage_bps / 10000.0)
        return float(commission + fee_total + slippage)

    def _estimate_cost_for_signal(self, signal: Dict) -> float:
        qty = int(signal.get("qty", 0) or 0)
        price = float(signal.get("price", 0.0) or 0.0)
        if price <= 0:
            estimated_value = float(signal.get("estimated_value", 0.0) or 0.0)
            price = estimated_value / qty if qty > 0 else 0.0
        return self._estimate_transaction_cost(qty, price)

    def _estimate_total_cost(self, trading_signals: Dict[str, List[Dict]]) -> float:
        total = 0.0
        for bucket in ("buy", "sell"):
            for signal in trading_signals.get(bucket, []):
                total += self._estimate_cost_for_signal(signal)
        return total

    def _get_daily_cost_limit(self) -> float:
        limit = float(self.max_daily_cost_usd or 0.0)
        if self.max_daily_cost_pct and self.last_equity:
            pct_limit = self.last_equity * float(self.max_daily_cost_pct)
            limit = max(limit, pct_limit) if limit > 0 else pct_limit
        return float(limit)

    def refresh_order_statuses(self) -> List[Dict]:
        """Refresh submitted order statuses from broker."""
        if self.dry_run or not self.trade_client:
            return []

        self._cleanup_expired_orders()
        status_updates = []
        for symbol, order_info in list(self.submitted_orders.items()):
            order_id = order_info.get("order_id")
            if not order_id or str(order_id).startswith("TEMP_"):
                continue
            try:
                order = self.trade_client.get_order(id=order_id)
            except Exception as exc:
                append_log(f"[ORDER_TRACKING] Failed to fetch order {order_id}: {exc}")
                continue
            if not order:
                continue

            status = self._normalize_order_status(getattr(order, "status", None))
            filled_qty = int(getattr(order, "filled", 0) or 0)
            avg_fill_price = getattr(order, "avg_fill_price", None)
            avg_fill_price = float(avg_fill_price) if avg_fill_price else None
            prev_filled = int(order_info.get("filled_qty", 0) or 0)

            if filled_qty > prev_filled:
                delta_qty = filled_qty - prev_filled
                price = avg_fill_price or float(order_info.get("price", 0.0) or 0.0)
                added_cost = self._estimate_transaction_cost(delta_qty, price)
                self.daily_cost_total += added_cost
                order_info["filled_qty"] = filled_qty
                order_info["avg_fill_price"] = avg_fill_price
                order_info["cost_total"] = float(order_info.get("cost_total", 0.0) or 0.0) + added_cost

            order_info["status"] = status
            status_updates.append(
                {
                    "symbol": symbol,
                    "order_id": order_id,
                    "status": status,
                    "filled_qty": filled_qty,
                    "avg_fill_price": avg_fill_price,
                    "action": order_info.get("action"),
                }
            )

            if status in {"filled", "cancelled", "rejected", "expired"}:
                if status in {"cancelled", "rejected"}:
                    self._update_failure_count(symbol, False)
                del self.submitted_orders[symbol]

        return status_updates

    def _is_duplicate_order(self, symbol: str, action: str, qty: int) -> bool:
        """
        检查是否为重复订单

        Args:
            symbol: 股票代码
            action: 交易动作
            qty: 数量

        Returns:
            是否为重复订单
        """
        if symbol not in self.submitted_orders:
            return False

        order_info = self.submitted_orders[symbol]
        order_time = order_info.get('timestamp')

        # 检查是否在过期时间内
        if order_time:
            time_diff = (datetime.now() - order_time).total_seconds()
            if time_diff > self.order_expiry_minutes * 60:
                # 订单记录已过期，不是重复订单
                return False

        # 检查动作和数量是否相同
        if order_info.get('action') == action and order_info.get('qty') == qty:
            append_log(f"[ORDER_TRACKING] Duplicate order detected: {symbol} {action} {qty} (submitted {time_diff:.0f}s ago)")
            return True

        return False

    def _choose_order_type(self, symbol: str, action: str, urgency: int = 5) -> str:
        """
        Intelligently choose order type based on stock characteristics

        Args:
            symbol: Stock symbol
            action: BUY or SELL
            urgency: Urgency level (1-10), 8+ means urgent stop-loss

        Returns:
            "MARKET" or "LIMIT"
        """
        # Urgent stop-loss -> market order
        if urgency >= 8:
            append_log(f"[ORDER_TYPE] {symbol}: MARKET (urgency={urgency})")
            return "MARKET"

        # Large cap stocks -> market order (good liquidity)
        if symbol.upper() in LARGE_CAP_SYMBOLS:
            append_log(f"[ORDER_TYPE] {symbol}: MARKET (large cap)")
            return "MARKET"

        # Other stocks -> limit order (safer)
        append_log(f"[ORDER_TYPE] {symbol}: LIMIT (default safe)")
        return "LIMIT"

    def _has_pending_limit_order(self, symbol: str) -> bool:
        """
        Check if there's a pending limit order for this symbol

        Returns:
            True if there's a pending limit order that hasn't expired
        """
        if symbol not in self.submitted_orders:
            return False

        order_info = self.submitted_orders[symbol]
        order_type = order_info.get('order_type', 'MARKET')
        status = order_info.get('status', 'unknown')

        # Only check limit orders with pending status
        if order_type != 'LIMIT':
            return False

        if status in {'pending', 'submitted', 'initial', 'unknown', 'partial', 'partially_filled'}:
            order_time = order_info.get('timestamp')
            if order_time:
                age_seconds = (datetime.now() - order_time).total_seconds()
                # Limit order older than 3 minutes -> cancel and allow new order
                if age_seconds > 180:
                    self._cancel_stale_limit_order(symbol, order_info.get('order_id'))
                    return False
                append_log(f"[LIMIT_ORDER] {symbol} has pending limit order ({age_seconds:.0f}s old)")
                return True
        return False

    def _cancel_stale_limit_order(self, symbol: str, order_id: str):
        """Cancel a stale limit order that has been pending too long"""
        if not self.trade_client or not order_id or str(order_id).startswith('TEMP_'):
            return
        try:
            self.trade_client.cancel_order(id=int(order_id))
            append_log(f"[LIMIT_ORDER] Cancelled stale limit order: {symbol} (ID: {order_id})")
            if symbol in self.submitted_orders:
                del self.submitted_orders[symbol]
        except Exception as e:
            append_log(f"[LIMIT_ORDER] Failed to cancel order {order_id}: {e}")

    def _record_submitted_order(self, symbol: str, action: str, qty: int, order_id: str, price: Optional[float] = None, order_type: str = "MARKET"):
        """
        Record a submitted order for tracking

        Args:
            symbol: Stock symbol
            action: Trade action (BUY/SELL)
            qty: Quantity
            order_id: Order ID
            price: Order price
            order_type: Order type (MARKET or LIMIT)
        """
        self.submitted_orders[symbol] = {
            'action': action,
            'qty': qty,
            'timestamp': datetime.now(),
            'order_id': order_id,
            'price': price,
            'order_type': order_type,
            'status': 'submitted'
        }
        append_log(f"[ORDER_TRACKING] Recorded {order_type} order: {symbol} {action} {qty} (order_id: {order_id})")

    def analyze_trading_opportunities(self, current_positions: List[Dict],
                                    recommended_positions: List[Dict],
                                    available_buying_power: Optional[float] = None) -> Dict[str, List[Dict]]:
        """
        分析交易机会，生成买入/卖出信号

        Args:
            current_positions: 当前真实持仓
            recommended_positions: AI推荐持仓
            available_buying_power: 可用购买力（可选，如果提供则使用，否则从Tiger API获取）

        Returns:
            交易信号字典 {'buy': [...], 'sell': [...], 'hold': [...]}
        """

        # 获取当前购买力 - 使用提供的值以确保与推荐生成一致，否则从Tiger API获取
        if available_buying_power is None:
            available_buying_power = self.get_buying_power()
            append_log(f"[AUTO_TRADING] Available buying power from Tiger API: ${available_buying_power:,.2f}")
        else:
            append_log(f"[AUTO_TRADING] Available buying power (provided): ${available_buying_power:,.2f}")

        # 当前持仓映射
        current_symbols = {}
        for pos in current_positions:
            symbol = pos.get('symbol')  # Try symbol first
            if not symbol:  # Fall back to other possible keys
                continue
            current_symbols[symbol] = pos

        # AI推荐映射（考虑action="buy"或"strong_buy"的）
        recommended_buys = {}
        for pos in recommended_positions:
            action = pos.get("action", "").lower()
            if action in ["buy", "strong_buy"]:
                symbol = pos.get('symbol')
                if symbol:
                    recommended_buys[symbol] = pos

        trading_signals = {
            'buy': [],
            'sell': [],
            'hold': []
        }

        # 1. 买入信号：AI推荐买入但当前未持仓
        # 先计算要买多少只新股票，以便平均分配资金
        symbols_to_buy = []
        for symbol in recommended_buys.keys():
            if symbol not in current_symbols:
                # 检查是否在失败黑名单中
                if symbol in self.failed_symbols and self.failed_symbols[symbol] >= self.max_failures:
                    append_log(f"[AUTO_TRADING] Skipping {symbol} - failed {self.failed_symbols[symbol]} times")
                    continue
                symbols_to_buy.append(symbol)

        # Use SCORE-WEIGHTED allocation from recommendations (not equal allocation)
        # The recommendations already contain pre-calculated qty and value based on score weights
        num_stocks_to_buy = len(symbols_to_buy)
        if num_stocks_to_buy > 0:
            # Calculate total recommended value to check against buying power
            total_recommended_value = sum(
                recommended_buys[s].get('value', 0) for s in symbols_to_buy
            )
            # Scale factor if recommendations exceed buying power
            scale_factor = min(1.0, available_buying_power / total_recommended_value) if total_recommended_value > 0 else 1.0
            append_log(f"[AUTO_TRADING] Using SCORE-WEIGHTED allocation for {num_stocks_to_buy} stocks (scale: {scale_factor:.2f})")

        # Generate buy signals using RECOMMENDED quantities (score-weighted)
        for symbol in symbols_to_buy:
            rec_pos = recommended_buys[symbol]
            current_price = rec_pos.get('price', 0)

            # CRITICAL FIX: Use recommended qty/value instead of equal allocation
            # The recommendations contain pre-calculated score-weighted positions
            recommended_qty = rec_pos.get('qty', 0)
            recommended_value = rec_pos.get('value', 0)
            position_weight = rec_pos.get('weight', 0)  # Weight percentage

            if recommended_qty > 0 and current_price > 0:
                # Apply scale factor if needed (when buying power is limited)
                if num_stocks_to_buy > 0 and scale_factor < 1.0:
                    qty = max(1, int(recommended_qty * scale_factor))
                else:
                    qty = recommended_qty

                estimated_value = qty * current_price

                # Verify we have enough buying power for this position
                if estimated_value <= available_buying_power:
                    trading_signals['buy'].append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'qty': qty,
                        'price': current_price,
                        'estimated_value': estimated_value,
                        'target_value': recommended_value,
                        'weight': position_weight,
                        'reason': f"AI recommends BUY (score: {rec_pos.get('score', 0)}, weight: {position_weight:.1f}%)"
                    })
                    append_log(f"[AUTO_TRADING] Buy signal for {symbol}: {qty} shares @ ${current_price:.2f} = ${estimated_value:,.2f} (weight: {position_weight:.1f}%)")
                else:
                    append_log(f"[AUTO_TRADING] Skipping {symbol}: estimated ${estimated_value:,.2f} exceeds buying power ${available_buying_power:,.2f}")
            elif current_price > 0:
                # Fallback: calculate qty from recommended value if qty not provided
                if recommended_value > 0:
                    qty = int(recommended_value / current_price)
                    if qty > 0:
                        estimated_value = qty * current_price
                        trading_signals['buy'].append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'qty': qty,
                            'price': current_price,
                            'estimated_value': estimated_value,
                            'target_value': recommended_value,
                            'weight': position_weight,
                            'reason': f"AI recommends BUY (score: {rec_pos.get('score', 0)}, weight: {position_weight:.1f}%)"
                        })
                        append_log(f"[AUTO_TRADING] Buy signal for {symbol}: {qty} shares @ ${current_price:.2f} = ${estimated_value:,.2f} (from value)")
                else:
                    append_log(f"[AUTO_TRADING] Cannot buy {symbol}: no qty or value in recommendation")
            else:
                append_log(f"[AUTO_TRADING] Invalid price for {symbol}: ${current_price:.2f}")

        # 2. 卖出信号：当前持仓但不在AI推荐买入列表中
        for symbol, curr_pos in current_symbols.items():
            if symbol not in recommended_buys:
                trading_signals['sell'].append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'qty': curr_pos.get('quantity', 0),
                    'price': curr_pos.get('market_price', 0),
                    'estimated_value': curr_pos.get('market_value', 0),
                    'reason': "Not in AI recommendations, exit position"
                })
            else:
                # 3. 持有信号：当前持仓且在AI推荐中
                trading_signals['hold'].append({
                    'symbol': symbol,
                    'action': 'HOLD',
                    'qty': curr_pos.get('quantity', 0),
                    'price': curr_pos.get('market_price', 0),
                    'estimated_value': curr_pos.get('market_value', 0),
                    'reason': f"Hold position, AI still recommends (score: {recommended_buys[symbol].get('score', 0)})"
                })

        append_log(f"[AUTO_TRADING] Trading analysis complete:")
        append_log(f"  - Buy signals: {len(trading_signals['buy'])}")
        append_log(f"  - Sell signals: {len(trading_signals['sell'])}")
        append_log(f"  - Hold positions: {len(trading_signals['hold'])}")

        return trading_signals

    def build_rebalance_signals(
        self,
        current_positions: List[Dict],
        target_positions: List[Dict],
        buying_power: float,
        min_trade_value: float = 0.0,
        buy_price_buffer_pct: float = 0.0,
        cooldown_blocked: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Build buy/sell/hold signals based on target position values.

        target_positions items: {symbol, price, target_value, target_weight, score}
        """
        cooldown_blocked = cooldown_blocked or {}
        current_map = {p.get("symbol"): p for p in current_positions if p.get("symbol")}
        target_map = {t.get("symbol"): t for t in target_positions if t.get("symbol")}

        symbols = set(current_map.keys()) | set(target_map.keys())
        trading_signals = {"buy": [], "sell": [], "hold": []}

        buy_buffer = max(float(buy_price_buffer_pct or 0.0), 0.0)

        for symbol in symbols:
            current = current_map.get(symbol, {})
            target = target_map.get(symbol, {})
            current_value = float(current.get("market_value", 0.0) or 0.0)
            target_value = float(target.get("target_value", 0.0) or 0.0)
            price = float(target.get("price") or current.get("market_price", 0.0) or 0.0)

            if price <= 0:
                continue

            delta = target_value - current_value
            if abs(delta) < float(min_trade_value):
                trading_signals["hold"].append(
                    {
                        "symbol": symbol,
                        "action": "HOLD",
                        "qty": int(current.get("quantity", 0) or 0),
                        "price": price,
                        "estimated_value": current_value,
                        "reason": "Within min trade value",
                    }
                )
                continue

            if delta > 0:
                if symbol in cooldown_blocked:
                    trading_signals["hold"].append(
                        {
                            "symbol": symbol,
                            "action": "HOLD",
                            "qty": int(current.get("quantity", 0) or 0),
                            "price": price,
                            "estimated_value": current_value,
                            "reason": f"Cooldown {cooldown_blocked[symbol]}s",
                        }
                    )
                    continue

                effective_price = price * (1.0 + buy_buffer)
                qty = int(delta / effective_price)
                if qty <= 0:
                    continue
                trading_signals["buy"].append(
                    {
                        "symbol": symbol,
                        "action": "BUY",
                        "qty": qty,
                        "price": price,
                        "estimated_value": qty * effective_price,
                        "reason": "Rebalance to target weight",
                    }
                )
            else:
                current_qty = int(current.get("quantity", 0) or 0)
                qty = int(abs(delta) / price)
                if qty <= 0 or current_qty <= 0:
                    continue
                qty = min(qty, current_qty)
                trading_signals["sell"].append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "qty": qty,
                        "price": price,
                        "estimated_value": qty * price,
                        "reason": "Reduce to target weight",
                    }
                )

        total_buy_value = sum(s.get("estimated_value", 0.0) for s in trading_signals["buy"])
        if total_buy_value > 0 and buying_power > 0 and total_buy_value > buying_power:
            scale = buying_power / total_buy_value
            scaled_buys = []
            for buy in trading_signals["buy"]:
                scaled_value = buy["estimated_value"] * scale
                price = buy["price"]
                effective_price = price * (1.0 + buy_buffer)
                qty = int(scaled_value / effective_price)
                if qty <= 0:
                    continue
                buy = buy.copy()
                buy["qty"] = qty
                buy["estimated_value"] = qty * effective_price
                buy["reason"] = "Rebalance scaled to buying power"
                if buy["estimated_value"] >= float(min_trade_value):
                    scaled_buys.append(buy)
            trading_signals["buy"] = scaled_buys

        return trading_signals

    def execute_trading_signals(self, trading_signals: Dict[str, List[Dict]]) -> List[Dict]:
        """
        执行交易信号 - 实现分阶段执行以确保流动性管理

        Args:
            trading_signals: 交易信号字典

        Returns:
            执行结果列表
        """

        execution_results = []

        # 清理过期的订单记录
        self._cleanup_expired_orders()

        # 风险控制检查
        risk_check = self._perform_risk_checks(trading_signals)
        if not risk_check['passed']:
            append_log(f"[AUTO_TRADING] Risk check failed: {risk_check['reason']}")
            return [{
                'success': False,
                'error': f"Risk check failed: {risk_check['reason']}",
                'timestamp': datetime.now().isoformat()
            }]

        # 检查每日交易限制
        if self.daily_trade_count >= self.max_daily_trades:
            append_log(f"[AUTO_TRADING] Daily trade limit reached ({self.max_daily_trades})")
            return execution_results

        # Phase 1: 执行所有卖出订单
        sell_proceeds = 0.0
        successful_sells = 0
        append_log(f"[AUTO_TRADING] Phase 1: Executing {len(trading_signals.get('sell', []))} sell orders")

        for sell_signal in trading_signals.get('sell', []):
            if self.daily_trade_count >= self.max_daily_trades:
                break

            # 检查是否为重复订单
            symbol = sell_signal.get('symbol', '')
            action = sell_signal.get('action', '')
            qty = sell_signal.get('qty', 0)

            if self._is_duplicate_order(symbol, action, qty):
                append_log(f"[AUTO_TRADING] Skipped {symbol} - duplicate order")
                continue

            # 单个订单风险检查
            if self._validate_single_order(sell_signal):
                # CRITICAL: 在执行前立即记录订单,防止同一周期内的重复提交
                temp_order_id = f"TEMP_{symbol}_{action}_{int(time.time())}"
                self._record_submitted_order(symbol, action, qty, temp_order_id, sell_signal.get('price'))
                append_log(f"[ORDER_TRACKING] Pre-recorded order: {symbol} {action} {qty} (TEMP)")

                result = self._execute_order(sell_signal)
                execution_results.append(result)

                if result.get('success'):
                    self.daily_trade_count += 1
                    successful_sells += 1
                    # 累计卖出收益（用于重新计算购买力）
                    sell_proceeds += sell_signal.get('estimated_value', 0)
                    append_log(f"[AUTO_TRADING] Sell executed: {sell_signal['symbol']}, proceeds: ${sell_signal.get('estimated_value', 0):,.2f}")
            else:
                append_log(f"[AUTO_TRADING] Skipped {sell_signal['symbol']} - failed validation")

        # Phase 2: 等待并重新计算购买力（如果有卖出交易成功）
        if successful_sells > 0:
            append_log(f"[AUTO_TRADING] Phase 1 complete: {successful_sells} sells executed, ${sell_proceeds:,.2f} total proceeds")
            append_log(f"[AUTO_TRADING] Waiting for settlement before buy orders...")

            # 在实际生产中，这里可以添加短暂延迟或检查订单状态
            # 但对于现有系统，我们立即重新计算购买力
            time.sleep(1)  # 短暂延迟以允许系统更新

            # 重新获取更新后的购买力
            updated_buying_power = self.get_buying_power()
            append_log(f"[AUTO_TRADING] Updated buying power after sells: ${updated_buying_power:,.2f}")

            # 重新验证买入订单的购买力
            buy_signals = trading_signals.get('buy', [])
            validated_buys = self._revalidate_buy_orders(buy_signals, updated_buying_power)
            append_log(f"[AUTO_TRADING] Revalidated {len(validated_buys)}/{len(buy_signals)} buy orders")
        else:
            validated_buys = trading_signals.get('buy', [])
            append_log(f"[AUTO_TRADING] No sell orders executed, proceeding with original buy orders")

        # Phase 3: 执行验证后的买入订单
        append_log(f"[AUTO_TRADING] Phase 2: Executing {len(validated_buys)} buy orders")

        for buy_signal in validated_buys:
            if self.daily_trade_count >= self.max_daily_trades:
                break

            # 检查是否为重复订单
            symbol = buy_signal.get('symbol', '')
            action = buy_signal.get('action', '')
            qty = buy_signal.get('qty', 0)

            if self._is_duplicate_order(symbol, action, qty):
                append_log(f"[AUTO_TRADING] Skipped {symbol} - duplicate order")
                continue

            # 单个订单风险检查
            if self._validate_single_order(buy_signal):
                # CRITICAL: 在执行前立即记录订单,防止同一周期内的重复提交
                temp_order_id = f"TEMP_{symbol}_{action}_{int(time.time())}"
                self._record_submitted_order(symbol, action, qty, temp_order_id, buy_signal.get('price'))
                append_log(f"[ORDER_TRACKING] Pre-recorded order: {symbol} {action} {qty} (TEMP)")

                result = self._execute_order(buy_signal)
                execution_results.append(result)

                if result.get('success'):
                    self.daily_trade_count += 1
                    append_log(f"[AUTO_TRADING] Buy executed: {buy_signal['symbol']}, cost: ${buy_signal.get('estimated_value', 0):,.2f}")
            else:
                append_log(f"[AUTO_TRADING] Skipped {buy_signal['symbol']} - failed validation")

        append_log(f"[AUTO_TRADING] Execution complete: {len(execution_results)} total orders processed")
        return execution_results

    def _perform_risk_checks(self, trading_signals: Dict[str, List[Dict]]) -> Dict:
        """
        执行风险控制检查 - 使用动态限制

        Args:
            trading_signals: 交易信号

        Returns:
            风险检查结果
        """

        # 检查总交易价值
        total_buy_value = sum(signal.get('estimated_value', 0) for signal in trading_signals.get('buy', []))
        total_sell_value = sum(signal.get('estimated_value', 0) for signal in trading_signals.get('sell', []))

        # Daily buy limit check removed - allow full buying power usage
        available_buying_power = self.get_buying_power()

        # Skip daily buy limit check to allow flexible position building
        # User requested removal of this restriction

        # 检查交易数量限制
        total_trades = len(trading_signals.get('buy', [])) + len(trading_signals.get('sell', []))
        if total_trades > self.max_daily_trades:
            return {
                'passed': False,
                'reason': f'Too many trades: {total_trades} > {self.max_daily_trades}'
            }

        # 检查是否有有效的交易信号
        if total_trades == 0:
            return {
                'passed': True,
                'reason': 'No trades to execute'
            }

        estimated_cost = self._estimate_total_cost(trading_signals)
        cost_limit = self._get_daily_cost_limit()
        if cost_limit > 0 and (self.daily_cost_total + estimated_cost) > cost_limit:
            return {
                'passed': False,
                'reason': (
                    f'Transaction cost limit reached: '
                    f'${self.daily_cost_total + estimated_cost:,.2f} > ${cost_limit:,.2f}'
                )
            }

        # Net cash flow check removed - allow flexible fund allocation
        net_cash = total_sell_value - total_buy_value
        max_allowed_outflow = available_buying_power * self.max_daily_outflow_percent

        # Skip net cash flow limit check to allow full buying power utilization
        # User requested removal of this restriction

        return {
            'passed': True,
            'reason': f'Risk checks passed - {total_trades} trades, net cash: ${net_cash:,.2f}, max outflow: ${max_allowed_outflow:,.2f}'
        }

    def _validate_single_order(self, order_signal: Dict) -> bool:
        """
        验证单个订单 - 使用动态限制

        Args:
            order_signal: 订单信号

        Returns:
            是否通过验证
        """

        symbol = order_signal.get('symbol', '')
        qty = order_signal.get('qty', 0)
        price = order_signal.get('price', 0)
        estimated_value = order_signal.get('estimated_value', 0)
        action = order_signal.get('action', '').upper()

        # 基本字段检查
        if not symbol or qty <= 0 or price <= 0:
            append_log(f"[VALIDATION] Invalid order parameters for {symbol}")
            return False

        # 单笔交易价值检查 - 只对买入订单检查
        if action == 'BUY':
            buying_power = self.get_buying_power()
            max_allowed_value = self.get_max_position_value(buying_power) * 1.5  # 允许50%溢价

            if estimated_value > max_allowed_value:
                append_log(f"[VALIDATION] Order value too high: ${estimated_value:,.2f} > ${max_allowed_value:,.2f} for {symbol}")
                return False

        # 股价合理性检查 ($1 - $10,000)
        if price < 1.0 or price > 10000:
            append_log(f"[VALIDATION] Price out of range: ${price:.2f} for {symbol}")
            return False

        # 数量合理性检查 (1 - 10,000股)
        if qty < 1 or qty > 10000:
            append_log(f"[VALIDATION] Quantity out of range: {qty} shares for {symbol}")
            return False

        return True

    def _revalidate_buy_orders(self, buy_signals: List[Dict], updated_buying_power: float) -> List[Dict]:
        """
        重新验证买入订单（在卖出后更新购买力）

        Args:
            buy_signals: 原始买入信号
            updated_buying_power: 更新后的购买力

        Returns:
            验证后的买入信号列表
        """
        validated_buys = []

        for buy_signal in buy_signals:
            estimated_value = buy_signal.get('estimated_value', 0)

            # 检查是否仍在购买力范围内
            if estimated_value <= updated_buying_power:
                validated_buys.append(buy_signal)
            else:
                # 重新计算数量以适应更新的购买力
                symbol = buy_signal.get('symbol', '')
                price = buy_signal.get('price', 0)
                max_allocation = self.get_max_position_value(updated_buying_power)

                if price > 0 and max_allocation > price:
                    new_qty = int(max_allocation / price)
                    new_estimated_value = new_qty * price

                    if new_qty > 0 and new_estimated_value <= updated_buying_power:
                        updated_signal = buy_signal.copy()
                        updated_signal.update({
                            'qty': new_qty,
                            'estimated_value': new_estimated_value,
                            'reason': f"Adjusted for updated buying power: {new_estimated_value/updated_buying_power*100:.1f}%"
                        })
                        validated_buys.append(updated_signal)
                        append_log(f"[AUTO_TRADING] Adjusted {symbol}: {new_qty} shares @ ${price:.2f} = ${new_estimated_value:,.2f}")
                    else:
                        append_log(f"[AUTO_TRADING] Removed {symbol} - insufficient updated buying power")
                else:
                    append_log(f"[AUTO_TRADING] Removed {symbol} - invalid price or allocation")

        return validated_buys

    def _execute_order(self, order_signal: Dict) -> Dict:
        """
        执行单个订单 - 使用Adaptive Execution Engine（如果可用）

        Args:
            order_signal: 订单信号

        Returns:
            执行结果
        """
        execution_start_time = time.time()  # Track execution timing

        symbol = order_signal.get('symbol', '')
        action = order_signal.get('action', '')
        qty = order_signal.get('qty', 0)
        price = order_signal.get('price', 0)
        estimated_value = order_signal.get('estimated_value', 0)
        reason = order_signal.get('reason', 'No reason provided')

        append_log(f"[AUTO_TRADING] Executing {action} order: {qty} shares of {symbol} @ ${price:.2f} (Reason: {reason})")

        if self.dry_run:
            # 模拟执行
            order_id = f"SIM_{int(time.time())}"
            result = {
                'success': True,
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'price': price,
                'order_id': order_id,
                'reason': reason,
                'message': "Simulated order execution",
                'timestamp': datetime.now().isoformat()
            }
            append_log(f"[AUTO_TRADING] SIMULATED {action}: {qty} {symbol} @ ${price:.2f}")

        elif self.use_adaptive_execution and self.adaptive_execution:
            # 使用自适应执行引擎
            try:
                result = self._execute_with_adaptive_engine(symbol, action, qty, price, estimated_value)
            except Exception as e:
                append_log(f"[ADAPTIVE_EXEC] Execution failed: {e}, falling back to simple execution")
                result = self._place_real_order(symbol, action, qty, price)
        else:
            # 简单真实执行（fallback）
            result = self._place_real_order(symbol, action, qty, price)

        # 记录已提交的订单（防止重复提交）
        # IMPORTANT: 即使执行过程有问题,只要有订单ID就应该记录,防止重复提交
        order_id = result.get('order_id', '')
        order_type = result.get('order_type', 'MARKET')
        if order_id:
            self._record_submitted_order(symbol, action, qty, order_id, price, order_type)
            append_log(f"[ORDER_TRACKING] Recorded {order_type} order: {symbol} {action} {qty} shares (ID: {order_id})")

        # Record execution quality to Supabase for analysis
        if result.get('success'):
            self._record_execution_to_supabase(result, order_signal, execution_start_time)

        # 更新失败计数
        self._update_failure_count(symbol, result.get('success', False))

        # Ensure reason is included in result for all execution paths
        if 'reason' not in result:
            result['reason'] = reason

        # 记录交易历史
        self.trade_history.append(result)

        return result

    def _execute_with_adaptive_engine(self, symbol: str, action: str, qty: int,
                                     price: float, estimated_value: float) -> Dict:
        """
        使用Adaptive Execution Engine执行订单

        Args:
            symbol: 股票代码
            action: 交易动作 (BUY/SELL)
            qty: 数量
            price: 价格
            estimated_value: 预估价值

        Returns:
            执行结果
        """
        try:
            # 确定紧急程度（基于订单大小和市场条件）
            urgency = self._determine_urgency(estimated_value, qty)

            # 创建执行订单
            order_id = f"AUTO_{symbol}_{int(time.time())}"
            exec_order = ExecutionOrder(
                order_id=order_id,
                symbol=symbol,
                side=action.lower(),  # "buy" or "sell"
                total_quantity=qty,
                target_price=price,
                urgency=urgency,
                max_participation_rate=0.25,  # 25% max participation
                time_horizon=timedelta(minutes=30),  # 30分钟执行窗口
                current_participation_rate=0.15  # 初始参与率
            )

            # 提交订单到自适应执行引擎（异步）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            submitted_order_id = loop.run_until_complete(
                self.adaptive_execution.submit_order(exec_order)
            )

            # 监控执行（等待一段时间并检查状态）
            time.sleep(5)  # 等待初始执行
            order_status = self.adaptive_execution.get_order_status(submitted_order_id)

            if order_status:
                slippage_bps = order_status.get('implementation_shortfall', 0)
                fill_rate = order_status.get('fill_rate', 0)
                # Get Tiger API order ID for accurate tracking
                tiger_order_id = order_status.get('tiger_order_id', submitted_order_id)

                append_log(f"[ADAPTIVE_EXEC] {action} {qty} {symbol} @ ${price:.2f}")
                append_log(f"[ADAPTIVE_EXEC] Slippage: {slippage_bps:.2f} bps, Fill rate: {fill_rate:.1%}")
                if tiger_order_id:
                    append_log(f"[ADAPTIVE_EXEC] Tiger API order ID: {tiger_order_id}")

                return {
                    'success': True,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'filled_qty': order_status.get('filled_quantity', 0),
                    'price': price,
                    'avg_fill_price': order_status.get('average_fill_price', price),
                    'order_id': tiger_order_id if tiger_order_id else submitted_order_id,  # Use Tiger ID for tracking
                    'slippage_bps': slippage_bps,
                    'fill_rate': fill_rate,
                    'market_impact_bps': order_status.get('market_impact', 0),
                    'message': f"Adaptive execution - Slippage: {slippage_bps:.2f}bps",
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise Exception("Order status not available")

        except Exception as e:
            append_log(f"[ADAPTIVE_EXEC] Error: {e}")
            raise

    def _determine_urgency(self, estimated_value: float, qty: int) -> ExecutionUrgency:
        """
        根据订单大小和市场条件确定执行紧急程度

        Args:
            estimated_value: 预估价值
            qty: 数量

        Returns:
            执行紧急程度
        """
        # 基于订单价值确定紧急程度
        if estimated_value < 5000:
            return ExecutionUrgency.HIGH  # 小订单快速执行
        elif estimated_value < 10000:
            return ExecutionUrgency.MEDIUM
        elif estimated_value < 20000:
            return ExecutionUrgency.LOW
        else:
            return ExecutionUrgency.LOW  # 大订单慢速执行以减少市场冲击

    def _generate_idempotent_order_id(self, symbol: str, action: str, qty: int) -> str:
        """
        Generate an idempotent order ID for deduplication.

        Uses current date + symbol + action + qty to create a unique ID
        that will be the same for retries of the same order.
        """
        import hashlib
        date_str = datetime.now().strftime("%Y%m%d")
        order_key = f"{date_str}_{symbol}_{action}_{qty}"
        hash_suffix = hashlib.md5(order_key.encode()).hexdigest()[:8]
        return f"QS_{date_str}_{symbol}_{action}_{hash_suffix}"

    def _check_existing_order(self, symbol: str, action: str, qty: int) -> Optional[str]:
        """
        Check if an order with similar parameters was recently placed.

        Returns order_id if a matching order exists, None otherwise.
        """
        try:
            if not self.trade_client:
                return None

            # CRITICAL FIX: Add time limit to get_orders() to improve performance
            # Only query orders from the last 10 minutes instead of all historical orders
            start_time = datetime.now() - timedelta(minutes=10)
            start_time_ms = int(start_time.timestamp() * 1000)

            # Get recent orders from Tiger with time filter
            orders = self.trade_client.get_orders(start_time=start_time_ms)
            if not orders:
                return None

            now = datetime.now()
            for order in orders:
                try:
                    order_symbol = getattr(getattr(order, 'contract', None), 'symbol', None)
                    order_action = getattr(order, 'action', None)
                    order_qty = getattr(order, 'quantity', 0)
                    order_status = getattr(order, 'status', None)
                    order_time = getattr(order, 'order_time', None)

                    # Check if this is a matching order from today
                    if (order_symbol == symbol and
                        order_action == action and
                        abs(order_qty) == abs(qty) and
                        order_status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.PENDING_NEW]):

                        # Check if order was placed within the last 5 minutes
                        if order_time:
                            time_diff = (now - order_time).total_seconds() if hasattr(order_time, 'timestamp') else 300
                            if time_diff < 300:  # 5 minutes
                                order_id = getattr(order, 'id', None)
                                append_log(f"[DUPLICATE_CHECK] Found existing order for {symbol} {action} {qty}: ID={order_id}")
                                return str(order_id) if order_id else None
                except Exception as parse_e:
                    continue

            return None

        except Exception as e:
            append_log(f"[DUPLICATE_CHECK] Error checking existing orders: {e}")
            return None

    def _verify_order_status(self, order_id: int, symbol: str, max_retries: int = 3) -> bool:
        """
        Verify order was actually placed by checking order status.

        CRITICAL FIX: This prevents false success reports when order placement fails silently.

        Args:
            order_id: Order ID from place_order
            symbol: Symbol for logging
            max_retries: Maximum verification attempts

        Returns:
            True if order is verified, False otherwise
        """
        if not self.trade_client:
            return False

        verified_statuses = [
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.PENDING_NEW
        ]

        for attempt in range(max_retries):
            try:
                time.sleep(0.5)  # Brief delay before checking

                order = self.trade_client.get_order(id=order_id)

                if order and hasattr(order, 'status'):
                    order_status = order.status
                    if order_status in verified_statuses:
                        append_log(f"[ORDER_VERIFY] Order {order_id} for {symbol} VERIFIED: {order_status}")
                        return True

                    # Check for rejection
                    if order_status in [OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
                        append_log(f"[ORDER_VERIFY] Order {order_id} for {symbol} REJECTED: {order_status}")
                        return False

            except Exception as e:
                append_log(f"[ORDER_VERIFY] Verification attempt {attempt+1} failed: {e}")

        append_log(f"[ORDER_VERIFY] Could not verify order {order_id} after {max_retries} attempts")
        return False

    def _place_real_order(self, symbol: str, action: str, qty: int, price: float) -> Dict:
        """
        下真实订单 - with idempotency protection and safety guards

        Args:
            symbol: 股票代码
            action: 交易动作 (BUY/SELL)
            qty: 数量
            price: 价格

        Returns:
            执行结果
        """

        if not self.trade_client:
            return {
                'success': False,
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'price': price,
                'error': "Tiger trade client not available",
                'timestamp': datetime.now().isoformat()
            }

        # CRITICAL SAFETY #1: Market hours enforcement
        # Block orders outside regular trading hours to prevent unintended executions
        if MARKET_TIME_AVAILABLE and US_MARKET:
            market_phase = US_MARKET.get_market_phase()
            if market_phase != MarketPhase.REGULAR:
                append_log(f"[MARKET_HOURS] BLOCKED: Order for {symbol} rejected - market is {market_phase.value}")
                return {
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'error': f"Market closed: {market_phase.value}",
                    'blocked_by': 'market_hours',
                    'timestamp': datetime.now().isoformat()
                }

        # CRITICAL FIX #7: Check daily loss limit before placing any BUY orders
        if action.upper() == 'BUY':
            can_trade, halt_reason = self.check_daily_loss_limit()
            if not can_trade:
                append_log(f"[DAILY_LOSS] Blocking BUY order for {symbol}: {halt_reason}")
                return {
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'error': f"Daily loss limit exceeded: {halt_reason}",
                    'blocked_by': 'daily_loss_limit',
                    'timestamp': datetime.now().isoformat()
                }

        # CRITICAL SAFETY #2: Order rate limiting
        # Prevent excessive order placement that could lead to trading loops
        if self.rate_limiter:
            can_place, rate_reason = self.rate_limiter.can_place_order(symbol)
            if not can_place:
                append_log(f"[RATE_LIMIT] BLOCKED: Order for {symbol} rejected - {rate_reason}")
                return {
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'error': f"Rate limit: {rate_reason}",
                    'blocked_by': 'rate_limiter',
                    'timestamp': datetime.now().isoformat()
                }

        # CRITICAL: Check for pending limit order before placing new order
        if self._has_pending_limit_order(symbol):
            append_log(f"[AUTO_TRADING] Skipped {symbol} - pending limit order exists")
            return {
                'success': False,
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'price': price,
                'error': 'Pending limit order exists',
                'timestamp': datetime.now().isoformat()
            }

        # CRITICAL SAFETY #3: Acquire order lock to prevent concurrent order placement
        with self._order_lock:
            try:
                # STEP 1: Check if an identical order was already placed (idempotency check)
                existing_order_id = self._check_existing_order(symbol, action, qty)
                if existing_order_id:
                    append_log(f"[AUTO_TRADING] Order already exists on Tiger: {symbol} {action} {qty} (ID: {existing_order_id})")
                    return {
                        'success': True,
                        'symbol': symbol,
                        'action': action,
                        'qty': qty,
                        'price': price,
                        'order_id': existing_order_id,
                        'message': "Order already exists (idempotency protection)",
                        'already_existed': True,
                        'timestamp': datetime.now().isoformat()
                    }

                # STEP 2: Generate client order ID for tracking
                client_order_id = self._generate_idempotent_order_id(symbol, action, qty)
                append_log(f"[AUTO_TRADING] Placing order with client ID: {client_order_id}")

                # Create stock contract
                contract = stock_contract(symbol, Market.US)

                # SMART ORDER TYPE SELECTION: Choose between market and limit order
                order_type = self._choose_order_type(symbol, action)

                if order_type == "MARKET":
                    order = market_order(
                        account=self.account,
                        contract=contract,
                        action=action,
                        quantity=qty
                    )
                else:
                    # Limit order: BUY at +0.3%, SELL at -0.3%
                    limit_price = round(price * 1.003 if action == "BUY" else price * 0.997, 2)
                    order = limit_order(
                        account=self.account,
                        contract=contract,
                        action=action,
                        quantity=qty,
                        limit_price=limit_price
                    )
                    append_log(f"[LIMIT_ORDER] {symbol}: {action} {qty} @ ${limit_price:.2f}")

                # Place order
                order_result = self.trade_client.place_order(order)

                # CRITICAL FIX: Robust order ID extraction from different return types
                order_id = None
                if isinstance(order_result, int):
                    order_id = order_result
                elif order_result and hasattr(order_result, 'id') and order_result.id:
                    order_id = order_result.id
                elif hasattr(order, 'id') and order.id:
                    order_id = order.id

                # CRITICAL SAFETY #4: Record order to rate limiter after successful placement
                if self.rate_limiter and order_id:
                    self.rate_limiter.record_order(symbol, action, qty, str(order_id))

                if order_id:
                    # CRITICAL FIX: Verify order was actually placed on Tiger's side
                    verified = self._verify_order_status(order_id, symbol)
                    if verified:
                        append_log(f"[AUTO_TRADING] SUCCESS: {action} {order_type} order placed and VERIFIED - Order ID: {order_id}")
                        return {
                            'success': True,
                            'symbol': symbol,
                            'action': action,
                            'qty': qty,
                            'price': price,
                            'order_id': order_id,
                            'client_order_id': client_order_id,
                            'order_type': order_type,
                            'message': f"{order_type} order placed and verified successfully",
                            'verified': True,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        # Order submitted but verification failed - log warning but still return success
                        # because the order might exist on Tiger side
                        append_log(f"[AUTO_TRADING] WARNING: Order {order_id} submitted but verification uncertain")
                        return {
                            'success': True,
                            'symbol': symbol,
                            'action': action,
                            'qty': qty,
                            'price': price,
                            'order_id': order_id,
                            'client_order_id': client_order_id,
                            'order_type': order_type,
                            'message': f"{order_type} order placed (verification pending)",
                            'verified': False,
                            'timestamp': datetime.now().isoformat()
                        }
                else:
                    append_log(f"[AUTO_TRADING] ERROR: Failed to place {action} {order_type} order for {symbol}")
                    return {
                        'success': False,
                        'symbol': symbol,
                        'action': action,
                        'qty': qty,
                        'price': price,
                        'order_type': order_type,
                        'error': "Order placement failed",
                        'timestamp': datetime.now().isoformat()
                    }

            except Exception as e:
                # CRITICAL: On exception, check if the order was actually placed
                append_log(f"[AUTO_TRADING] EXCEPTION placing order for {symbol}: {e}")
                append_log(f"[AUTO_TRADING] Checking if order was placed despite exception...")

                # Wait briefly and check for existing order
                time.sleep(1)
                existing_order_id = self._check_existing_order(symbol, action, qty)
                if existing_order_id:
                    append_log(f"[AUTO_TRADING] Order WAS placed despite exception: ID={existing_order_id}")
                    # Record to rate limiter since order was placed
                    if self.rate_limiter:
                        self.rate_limiter.record_order(symbol, action, qty, existing_order_id)
                    return {
                        'success': True,
                        'symbol': symbol,
                        'action': action,
                        'qty': qty,
                        'price': price,
                        'order_id': existing_order_id,
                        'message': "Order placed (recovered after exception)",
                        'recovered': True,
                        'timestamp': datetime.now().isoformat()
                    }

                return {
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

    def _record_execution_to_supabase(self, result: Dict, signal: Dict, execution_start_time: float):
        """
        Record execution quality to Supabase for analysis.

        Args:
            result: Execution result from _execute_order
            signal: Original order signal
            execution_start_time: Time when execution started (time.time())
        """
        supabase = _get_supabase()
        if not supabase or not supabase.is_enabled():
            return

        try:
            execution_time_ms = int((time.time() - execution_start_time) * 1000)
            signal_price = float(signal.get("price", 0) or 0)
            fill_price = float(result.get("avg_fill_price") or result.get("price", 0) or 0)

            # Calculate slippage in basis points
            slippage_bps = 0.0
            if signal_price > 0 and fill_price > 0:
                slippage_bps = ((fill_price - signal_price) / signal_price) * 10000
                # For SELL orders, negative slippage is bad (sold lower than signal)
                if result.get("action", "").upper() == "SELL":
                    slippage_bps = -slippage_bps

            # Calculate estimated slippage cost
            filled_qty = int(result.get("filled_qty") or result.get("qty", 0) or 0)
            slippage_cost = abs(fill_price - signal_price) * filled_qty if signal_price > 0 else 0

            # Estimate total cost including commission
            commission = self._estimate_transaction_cost(filled_qty, fill_price) if filled_qty > 0 else 0

            execution_record = {
                "order_id": result.get("order_id"),
                "symbol": result.get("symbol"),
                "action": result.get("action"),
                "signal_price": signal_price,
                "limit_price": signal.get("limit_price"),
                "fill_price": fill_price,
                "slippage_bps": round(slippage_bps, 2),
                "market_impact_bps": result.get("market_impact_bps", 0),
                "execution_time_ms": execution_time_ms,
                "order_quantity": int(signal.get("qty", 0) or 0),
                "filled_quantity": filled_qty,
                "commission": commission,
                "slippage_cost": round(slippage_cost, 4),
                "total_cost": round(commission + slippage_cost, 4),
            }

            supabase.insert_intraday_execution(execution_record)
            append_log(f"[EXECUTION_RECORD] Recorded execution for {result.get('symbol')}: "
                      f"slippage={slippage_bps:.2f}bps, time={execution_time_ms}ms")
        except Exception as e:
            append_log(f"[EXECUTION_RECORD] Failed to record execution: {e}")

    def _update_failure_count(self, symbol: str, success: bool):
        """
        更新股票失败计数

        Args:
            symbol: 股票代码
            success: 是否成功
        """
        if success:
            # 成功则重置失败计数
            if symbol in self.failed_symbols:
                del self.failed_symbols[symbol]
                append_log(f"[AUTO_TRADING] Reset failure count for {symbol}")
        else:
            # 失败则增加计数
            self.failed_symbols[symbol] = self.failed_symbols.get(symbol, 0) + 1
            append_log(f"[AUTO_TRADING] {symbol} failed {self.failed_symbols[symbol]} times")

            # 如果达到最大失败次数，加入黑名单
            if self.failed_symbols[symbol] >= self.max_failures:
                append_log(f"[AUTO_TRADING] {symbol} added to blacklist after {self.failed_symbols[symbol]} failures")

    def reset_blacklist(self):
        """重置黑名单，清除失败的股票记录"""
        self.failed_symbols.clear()
        append_log(f"[AUTO_TRADING] Blacklist reset - all failed symbols cleared")

    def get_trading_summary(self) -> Dict:
        """获取交易摘要"""
        return {
            'daily_trade_count': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'daily_cost_total': self.daily_cost_total,
            'daily_cost_limit': self._get_daily_cost_limit(),
            'daily_cost_pct': (self.daily_cost_total / self.last_equity) if self.last_equity else None,
            'total_trades': len(self.trade_history),
            'dry_run': self.dry_run,
            'last_trades': self.trade_history[-5:] if self.trade_history else [],
            'failed_symbols': self.failed_symbols,
            'blacklisted_symbols': [symbol for symbol, count in self.failed_symbols.items() if count >= self.max_failures],
            'dynamic_config': {
                'trading_safety_factor': self.trading_safety_factor,
                'min_cash_reserve': self.min_cash_reserve,
                'max_position_percent': self.max_position_percent,
                'fallback_buying_power': self.fallback_buying_power,
                'max_daily_outflow_percent': self.max_daily_outflow_percent
            }
        }

    def get_cost_summary(self) -> Dict:
        """Get transaction cost summary."""
        return {
            'daily_cost_total': self.daily_cost_total,
            'daily_cost_limit': self._get_daily_cost_limit(),
            'daily_cost_pct': (self.daily_cost_total / self.last_equity) if self.last_equity else None,
            'commission_per_share': self.commission_per_share,
            'min_commission': self.min_commission,
            'fee_per_order': self.fee_per_order,
            'slippage_bps': self.slippage_bps
        }

    def get_fund_management_status(self) -> Dict:
        """获取资金管理状态"""
        raw_buying_power = self.get_raw_buying_power()
        safe_buying_power = self.get_buying_power()
        max_position_value = self.get_max_position_value(safe_buying_power)

        return {
            'raw_buying_power': raw_buying_power,
            'safe_buying_power': safe_buying_power,
            'cash_reserve': self.min_cash_reserve,
            'safety_factor': self.trading_safety_factor,
            'max_position_value': max_position_value,
            'max_position_percent': self.max_position_percent,
            'max_daily_outflow': safe_buying_power * self.max_daily_outflow_percent,
            'utilization_rate': (raw_buying_power - safe_buying_power) / raw_buying_power if raw_buying_power > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }

    def get_execution_quality_report(self) -> Dict:
        """
        获取执行质量报告（来自Adaptive Execution Engine）

        Returns:
            执行质量指标字典
        """
        if self.use_adaptive_execution and self.adaptive_execution:
            try:
                performance_summary = self.adaptive_execution.get_performance_summary()

                metrics = performance_summary.get('performance_metrics', {})

                return {
                    'adaptive_execution_enabled': True,
                    'total_orders_executed': metrics.get('total_orders_executed', 0),
                    'average_implementation_shortfall_bps': metrics.get('average_implementation_shortfall', 0),
                    'average_market_impact_bps': metrics.get('average_market_impact', 0),
                    'fill_rate': metrics.get('fill_rate', 0),
                    'cost_savings_vs_naive_bps': metrics.get('cost_savings_vs_naive', 0),
                    'active_orders': performance_summary.get('active_orders_count', 0),
                    'completed_orders': performance_summary.get('completed_orders_count', 0),
                    'total_slices': performance_summary.get('total_slices_executed', 0),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                append_log(f"[AUTO_TRADING] Error getting execution quality report: {e}")
                return {
                    'adaptive_execution_enabled': True,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        else:
            return {
                'adaptive_execution_enabled': False,
                'message': 'Adaptive execution not available or disabled',
                'timestamp': datetime.now().isoformat()
            }
