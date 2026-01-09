import os, time, random
from dotenv import load_dotenv
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading
import logging
import json

# CRITICAL: Load .env with ABSOLUTE PATH before importing any modules
# Fix: runner.py runs from dashboard/worker/, but .env is in project root
# Without absolute path, load_dotenv() fails to find .env and DRY_RUN defaults to true
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(ENV_PATH)

# Add paths for imports
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
BOT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)
sys.path.insert(0, BOT_PARENT)

from state_manager import is_killed, write_status, append_log, read_status
from bot.market_time import get_market_manager, MarketPhase, MarketType
from bot.config import SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AI integration after logger is configured
try:
    from ai_integration import get_ai_manager, initialize_ai
    AI_INTEGRATION_AVAILABLE = True
    logger.info("[RUNNER] AI Integration module loaded successfully")
except ImportError as e:
    AI_INTEGRATION_AVAILABLE = False
    logger.warning(f"[RUNNER] AI Integration not available: {e}")

# Import Risk Integration
try:
    from risk_integration import RiskIntegrationManager
    RISK_INTEGRATION_AVAILABLE = True
    logger.info("[RUNNER] Risk Integration module loaded successfully")
except ImportError as e:
    RISK_INTEGRATION_AVAILABLE = False
    logger.warning(f"[RUNNER] Risk Integration not available: {e}")

# Import Real-Time Monitor for institutional-quality metrics
try:
    sys.path.insert(0, os.path.join(BOT_PARENT, "bot"))
    from real_time_monitor import RealTimeMonitor
    REAL_TIME_MONITOR_AVAILABLE = True
    logger.info("[RUNNER] Real-Time Monitor module loaded successfully")
except ImportError as e:
    REAL_TIME_MONITOR_AVAILABLE = False
    logger.warning(f"[RUNNER] Real-Time Monitor not available: {e}")

# Import Factor Crowding Monitor for crowding detection
try:
    from bot.factor_crowding_monitor import FactorCrowdingMonitor
    FACTOR_CROWDING_AVAILABLE = True
    logger.info("[RUNNER] Factor Crowding Monitor module loaded successfully")
except ImportError as e:
    FACTOR_CROWDING_AVAILABLE = False
    logger.warning(f"[RUNNER] Factor Crowding Monitor not available: {e}")

# Import Compliance Monitor for regulatory compliance
try:
    from compliance_integration import ComplianceMonitor
    COMPLIANCE_AVAILABLE = True
    logger.info("[RUNNER] Compliance Monitor module loaded successfully")
except ImportError as e:
    COMPLIANCE_AVAILABLE = False
    logger.warning(f"[RUNNER] Compliance Monitor not available: {e}")

# Import Intelligent Alert System for context-aware alerting
try:
    from alert_integration import IntelligentAlertSystem
    ALERT_SYSTEM_AVAILABLE = True
    logger.info("[RUNNER] Intelligent Alert System module loaded successfully")
except ImportError as e:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning(f"[RUNNER] Intelligent Alert System not available: {e}")

# Import Supabase client for cloud data persistence
try:
    from supabase_client import supabase_client
    SUPABASE_AVAILABLE = True
    logger.info("[RUNNER] Supabase client loaded successfully")
except ImportError as e:
    SUPABASE_AVAILABLE = False
    logger.warning(f"[RUNNER] Supabase client not available: {e}")

# Import Intraday Signal Engine for 5-minute trend signals
try:
    from bot.intraday_signal_engine import IntradaySignalEngine
    INTRADAY_ENGINE_AVAILABLE = True
    logger.info("[RUNNER] Intraday Signal Engine loaded successfully")
except ImportError as e:
    INTRADAY_ENGINE_AVAILABLE = False
    logger.warning(f"[RUNNER] Intraday Signal Engine not available: {e}")

# Import Intelligent Trading Decision System
try:
    from bot.intelligent_trading_decision import (
        validate_startup,
        init_decision_system,
        get_system_info,
    )
    # Run startup validation
    success, passed, failed = validate_startup(raise_on_error=False)
    if success:
        DECISION_SYSTEM_AVAILABLE = True
        logger.info("[RUNNER] Intelligent Trading Decision System loaded and validated")
        logger.info(f"[RUNNER] Decision System: {len(passed)} modules passed validation")
    else:
        DECISION_SYSTEM_AVAILABLE = False
        logger.error(f"[RUNNER] Decision System validation failed: {failed}")
except ImportError as e:
    DECISION_SYSTEM_AVAILABLE = False
    logger.warning(f"[RUNNER] Intelligent Trading Decision System not available: {e}")
except Exception as e:
    DECISION_SYSTEM_AVAILABLE = False
    logger.error(f"[RUNNER] Decision System initialization error: {e}")

class MarketAwareScheduler:
    """
    Market-aware task scheduler that runs different tasks based on market phases.
    """

    def __init__(self, market_type: str = "US"):
        """Initialize scheduler with market configuration."""
        self.market_manager = get_market_manager(market_type)
        self.market_type = market_type
        self.running = False
        self.tasks = {
            'selection': [],  # Run during market closed
            'trading': [],    # Run during market open/active
            'monitoring': [], # Run continuously
            'ai_training': [], # AI training tasks
        }
        self.last_phase_check = None
        self.current_phase = None
        self.task_threads = {}

        # Task execution tracking
        self.task_stats = {
            'selection_runs': 0,
            'trading_runs': 0,
            'monitoring_runs': 0,
            'ai_training_runs': 0,
            'real_time_monitoring_runs': 0,
            'factor_crowding_runs': 0,
            'compliance_runs': 0,
            'total_errors': 0,
            'last_error_time': None
        }

        # Initialize AI manager
        self.ai_manager = None
        if AI_INTEGRATION_AVAILABLE:
            try:
                ai_config = {
                    'training_interval': 3600 * 24,  # Daily training
                    'optimization_interval': 3600 * 6,  # Optimize every 6 hours
                    'training_epochs': 10,
                    'data_dir': './data_cache',
                    'model_dir': './models'
                }
                self.ai_manager = get_ai_manager(ai_config)
                logger.info(f"[SCHEDULER] AI Manager initialized - Enabled: {self.ai_manager.is_enabled()}")
            except Exception as e:
                logger.error(f"[SCHEDULER] Failed to initialize AI manager: {e}")
                self.ai_manager = None

        # Initialize Risk Integration Manager
        # NOW WITH DYNAMIC TIGER ACCOUNT DATA!
        self.risk_manager = None
        if RISK_INTEGRATION_AVAILABLE:
            try:
                # No hardcoded portfolio value! Auto-fetches from Tiger API
                self.risk_manager = RiskIntegrationManager(
                    portfolio_value=None,  # Auto-detect from Tiger API
                    max_position_size=1.0,  # 100% max per position (no limit)
                    max_portfolio_leverage=1.0,
                    use_dynamic_portfolio=True,  # Enable dynamic Tiger API data
                    enable_tail_risk=True
                )
                portfolio_val = self.risk_manager.portfolio_value
                logger.info(f"[SCHEDULER] Risk Integration Manager initialized - Portfolio: ${portfolio_val:,.2f} (DYNAMIC MODE)")
                append_log(f"[RISK] Enhanced Risk Manager initialized with ES@97.5% monitoring and dynamic portfolio tracking")
            except Exception as e:
                logger.error(f"[SCHEDULER] Failed to initialize Risk manager: {e}")
                self.risk_manager = None


        # Initialize Real-Time Monitor for 17 institutional-quality metrics
        self.real_time_monitor = None
        if REAL_TIME_MONITOR_AVAILABLE:
            try:
                # Configure real-time monitor path
                config_path = os.path.join(BOT_PARENT, "config", "monitoring_config.json")
                self.real_time_monitor = RealTimeMonitor(config_path=config_path)
                logger.info("[SCHEDULER] Real-Time Monitor initialized - 17 institutional metrics tracking enabled")
            except Exception as e:
                logger.error(f"[SCHEDULER] Failed to initialize Real-Time Monitor: {e}")
                self.real_time_monitor = None

        # Initialize Factor Crowding Monitor for factor crowding detection
        self.crowding_monitor = None
        if FACTOR_CROWDING_AVAILABLE:
            try:
                self.crowding_monitor = FactorCrowdingMonitor()
                logger.info("[SCHEDULER] Factor Crowding Monitor initialized - HHI, Gini, and correlation clustering enabled")
                append_log("[CROWDING] Factor Crowding Monitor initialized - crowding detection active")
            except Exception as e:
                logger.error(f"[SCHEDULER] Failed to initialize Factor Crowding Monitor: {e}")
                self.crowding_monitor = None

        # Initialize Compliance Monitor for regulatory compliance
        # NOW WITH DYNAMIC TIGER ACCOUNT DATA!
        self.compliance_monitor = None
        if COMPLIANCE_AVAILABLE:
            try:
                self.compliance_monitor = ComplianceMonitor(
                    # No hardcoded values! Auto-detects from Tiger API
                    max_position_percentage=0.25,  # 25% of account value
                    max_concentration=0.25,        # 25% max concentration
                    use_dynamic_limits=True        # Enable dynamic Tiger API data
                )
                logger.info("[SCHEDULER] Compliance Monitor initialized - 8 regulatory rules active (DYNAMIC MODE)")
                append_log("[COMPLIANCE] Compliance monitoring active - using real-time Tiger account data")
            except Exception as e:
                logger.error(f"[SCHEDULER] Failed to initialize Compliance Monitor: {e}")
                self.compliance_monitor = None

        # Initialize Intelligent Alert System for context-aware alerting
        self.alert_system = None
        if ALERT_SYSTEM_AVAILABLE:
            try:
                self.alert_system = IntelligentAlertSystem(
                    alert_channels=['log', 'status'],  # Can add 'email', 'slack' later
                    severity_threshold='MEDIUM'
                )
                logger.info("[SCHEDULER] Intelligent Alert System initialized - Multi-level alerting active")
                append_log("[ALERT] Intelligent Alert System active - Context-aware notifications enabled")
            except Exception as e:
                logger.error(f"[SCHEDULER] Failed to initialize Alert System: {e}")
                self.alert_system = None

        # Configuration from environment
        self.selection_interval = int(os.getenv('SELECTION_TASK_INTERVAL', 10800))  # 3 hours default
        self.trading_interval = int(os.getenv('TRADING_TASK_INTERVAL', 30))       # 30 seconds default
        self.monitoring_interval = int(os.getenv('MONITORING_TASK_INTERVAL', 120)) # 2 minutes default
        self.max_concurrent_tasks = int(os.getenv('MAX_CONCURRENT_TASKS', 3))

        # Intraday trading configuration
        self.intraday_enabled = os.getenv("INTRADAY_TRADING_ENABLED", "false").lower() in ("1", "true", "yes")
        self.intraday_interval = int(os.getenv("INTRADAY_TASK_INTERVAL", 60))
        self.intraday_engine = None
        self.intraday_trading_engine = None
        self.intraday_trade_date = None
        if self.intraday_enabled and INTRADAY_ENGINE_AVAILABLE:
            config_path = os.getenv(
                "INTRADAY_STRATEGY_CONFIG",
                os.path.join(BOT_PARENT, "config", "intraday_strategy.json"),
            )
            try:
                self.intraday_engine = IntradaySignalEngine(config_path=config_path)
                append_log(f"[INTRADAY] Intraday engine initialized with {config_path}")
            except Exception as e:
                append_log(f"[INTRADAY] Failed to initialize intraday engine: {e}")
                self.intraday_engine = None

        # Demo trading state
        self.pnl = 0.0
        self.price = 100.0
        self.position = 0
        self.entry_price = None

        logger.info(f"Scheduler initialized for {market_type} market")
        logger.info(f"Task intervals - Selection: {self.selection_interval}s, "
                   f"Trading: {self.trading_interval}s, Monitoring: {self.monitoring_interval}s")

    def register_task(self, task_type: str, task_func, interval: int = 60, **kwargs):
        """Register a task to be executed based on market phases."""
        if task_type not in self.tasks:
            raise ValueError(f"Invalid task type: {task_type}")

        task = {
            'func': task_func,
            'interval': interval,
            'last_run': None,
            'kwargs': kwargs,
            'name': task_func.__name__ if hasattr(task_func, '__name__') else str(task_func)
        }

        self.tasks[task_type].append(task)
        logger.info(f"Registered {task_type} task: {task['name']} (interval: {interval}s)")

    def should_run_task_type(self, task_type: str, phase: MarketPhase) -> bool:
        """Determine if task type should run in current market phase."""
        if task_type == 'selection':
            return phase == MarketPhase.CLOSED
        elif task_type == 'trading':
            # Trading tasks only run during active market hours
            return phase in {MarketPhase.PRE_MARKET, MarketPhase.REGULAR, MarketPhase.AFTER_HOURS}
        elif task_type == 'monitoring':
            return True  # Always run monitoring tasks
        elif task_type == 'ai_training':
            return phase == MarketPhase.CLOSED  # AI training during market closed
        return False

    def run_task_group(self, task_type: str, phase: MarketPhase):
        """Execute all tasks in a task group if conditions are met."""
        if not self.should_run_task_type(task_type, phase):
            return

        current_time = datetime.now()
        tasks_to_run = []

        for task in self.tasks[task_type]:
            # Check if enough time has passed since last run
            if (task['last_run'] is None or
                (current_time - task['last_run']).total_seconds() >= task['interval']):
                tasks_to_run.append(task)

        for task in tasks_to_run:
            try:
                logger.info(f"Running {task_type} task: {task['name']} (phase: {phase.value})")

                # Track task execution
                start_time = time.time()
                task['func'](**task['kwargs'])
                execution_time = time.time() - start_time

                task['last_run'] = current_time
                task['last_execution_time'] = execution_time

                # Update task statistics
                self.task_stats[f'{task_type}_runs'] += 1

                if execution_time > 30.0:  # Log slow tasks
                    logger.warning(f"Slow task execution: {task['name']} took {execution_time:.1f}s")

            except Exception as e:
                error_msg = f"Task {task['name']} failed: {type(e).__name__}: {e}"
                logger.error(error_msg)
                append_log(error_msg)

                # Update error statistics
                self.task_stats['total_errors'] += 1
                self.task_stats['last_error_time'] = current_time.isoformat()

                # Add task-specific error tracking
                if 'error_count' not in task:
                    task['error_count'] = 0
                task['error_count'] += 1
                task['last_error'] = str(e)
                task['last_error_time'] = current_time.isoformat()

    def real_trading_task(self):
        """Real trading task - gets actual Tiger account positions, recommendations, and executes trades."""
        # Only run during active market hours
        if not self.market_manager.is_market_active():
            return

        # Start Supabase run tracking
        run_id = self._supabase_start_run('trading', {'market_phase': self.current_phase.value if self.current_phase else 'unknown'})
        start_time = time.time()
        error_msg = None

        try:
            # Get real Tiger positions
            real_positions = self._get_tiger_positions()

            # Get Tiger account available funds (NOT hardcoded!)
            available_funds = self._get_tiger_available_funds()
            append_log(f"[TIGER] Available funds from Tiger API: ${available_funds:,.2f}")

            # Get latest selection results from status
            status = read_status()
            selection_results = status.get('selection_results', {})
            top_picks = selection_results.get('top_picks', [])

            # Create recommended portfolio from selections
            recommended_positions = []
            recommended_total = 0

            if top_picks and available_funds > 0:
                from bot.data import fetch_history

                # Calculate dynamic position weights based on scores and rankings
                max_positions = min(10, len(top_picks))  # Max 10 positions
                selected_picks = top_picks[:max_positions]

                # Calculate score-based weights (higher score = larger position)
                # Use QUADRATIC weighting to amplify score differences
                # This makes high-scoring stocks receive significantly larger positions
                squared_scores = [pick.get('avg_score', 0)**2 for pick in selected_picks]
                total_squared_score = sum(squared_scores)

                if total_squared_score > 0:
                    # Calculate weight for each position based on its SQUARED score
                    # Quadratic formula: weight = score^2 / sum(all_scores^2)
                    position_weights = []
                    for i, pick in enumerate(selected_picks):
                        # Quadratic weighting amplifies differences dramatically
                        # Example: 73.4^2 / total vs 49.1^2 / total = much larger difference
                        weight = squared_scores[i] / total_squared_score
                        position_weights.append(weight)

                    # Use 90% of available funds
                    total_to_invest = available_funds * 0.9

                    append_log(f"[RECOMMENDATIONS] Allocating ${total_to_invest:,.2f} across {max_positions} positions (score-weighted)")

                    # Allocate positions based on weights
                    for idx, selection in enumerate(selected_picks):
                        symbol = selection['symbol']
                        try:
                            df = fetch_history(None, symbol, period='day', limit=1, dry_run=False)
                            if df is not None and not df.empty:
                                current_price = float(df['close'].iloc[-1])

                                # Calculate target value for this position based on its weight
                                target_value = total_to_invest * position_weights[idx]

                                # Calculate quantity
                                qty = int(target_value / current_price)

                                # Ensure at least 1 share if price allows
                                if qty == 0 and current_price < target_value:
                                    qty = 1

                                actual_value = qty * current_price
                                weight_pct = position_weights[idx] * 100

                                rec_item = {
                                    "symbol": symbol,
                                    "qty": qty,
                                    "price": round(current_price, 2),
                                    "value": round(actual_value, 2),
                                    "score": round(selection.get('avg_score', 0), 2),
                                    "action": selection.get('dominant_action', 'HOLD'),
                                    "strategy": f"Rank #{selection.get('rank', 0)} - {selection.get('strategy_count', 1)} strategies",
                                    "weight": round(weight_pct, 1)  # Position weight percentage
                                }

                                recommended_positions.append(rec_item)
                                recommended_total += actual_value

                                append_log(f"[RECOMMENDATIONS] {symbol}: ${actual_value:,.0f} ({weight_pct:.1f}% weight, score: {rec_item['score']})")

                        except Exception as e:
                            append_log(f"[RECOMMENDATIONS] Error updating {symbol}: {e}")

                    append_log(f"[RECOMMENDATIONS] Total allocated: ${recommended_total:,.2f} / ${total_to_invest:,.2f} ({recommended_total/total_to_invest*100:.1f}%)")
                else:
                    append_log("[RECOMMENDATIONS] No valid scores found, skipping recommendations")

            # Execute automatic trading based on recommendations vs current positions
            if self.intraday_enabled:
                trading_results = {
                    "skipped": True,
                    "reason": "intraday_enabled",
                    "last_analysis_time": datetime.now().isoformat(),
                }
                append_log("[TRADING] Intraday mode enabled, skipping daily auto-rebalance execution")
            else:
                trading_results = self._execute_auto_trading(real_positions, recommended_positions, available_funds)

            # Update status with both real and recommended positions, plus trading results
            write_status({
                "real_positions": real_positions,
                "real_portfolio_value": sum(pos.get('market_value', 0) for pos in real_positions),
                "real_positions_count": len(real_positions),
                "recommended_positions": recommended_positions,
                "recommended_portfolio_value": round(recommended_total, 2),
                "recommended_count": len(recommended_positions),
                "last_recommendation_update": datetime.now().isoformat(),
                "last_real_positions_update": datetime.now().isoformat(),
                "trading_mode": "LIVE_TRADING",
                "portfolio_note": "Real Tiger account positions + AI recommendations",
                "auto_trading_results": trading_results,
                "bot": "running",
                "market_phase": self.current_phase.value if self.current_phase else "unknown"
            })

            append_log(f"[TRADING] Updated: {len(real_positions)} real positions, {len(recommended_positions)} recommendations")

            # Record positions to Supabase
            self._supabase_record_positions(real_positions)

            # Record trade signals to Supabase (recommendations)
            for rec in recommended_positions:
                rec['was_executed'] = False  # Will be updated if executed
                self._supabase_record_trade_signal(rec, run_id)

        except Exception as e:
            logger.error(f"Real trading task error: {e}")
            append_log(f"[TRADING ERROR] {e}")
            error_msg = str(e)
        finally:
            # Complete Supabase run tracking
            duration_ms = int((time.time() - start_time) * 1000)
            self._supabase_complete_run(run_id, duration_ms, error_msg)

    def intraday_trading_task(self):
        """Run intraday 5-minute trend signals and rebalance positions."""
        if not self.intraday_enabled or not self.intraday_engine:
            return
        if self.current_phase != MarketPhase.REGULAR:
            return

        market_time = self.market_manager.get_current_market_time()
        open_buffer = int(self.intraday_engine.config.get("open_buffer_minutes", 0))
        if open_buffer > 0:
            regular_start = self.market_manager.trading_hours.regular_start
            regular_start_dt = market_time.replace(
                hour=regular_start.hour,
                minute=regular_start.minute,
                second=0,
                microsecond=0,
            )
            if market_time < regular_start_dt + timedelta(minutes=open_buffer):
                return

        status = read_status()
        current_positions = self._get_tiger_positions()
        available_funds = self._get_tiger_available_funds()

        watchlist = [p.get("symbol") for p in current_positions if p.get("symbol")]
        top_picks = status.get("selection_results", {}).get("top_picks", [])
        watchlist += [p.get("symbol") for p in top_picks if p.get("symbol")]
        watchlist = list(dict.fromkeys(watchlist))
        watchlist_limit = int(os.getenv("INTRADAY_WATCHLIST_SIZE", "30"))
        watchlist = watchlist[:watchlist_limit]

        intraday_state = status.get("intraday_state", {})
        result = self.intraday_engine.generate_targets(
            symbols=watchlist,
            current_positions=current_positions,
            available_funds=available_funds,
            state=intraday_state,
        )

        from auto_trading_engine import AutoTradingEngine

        dry_run = SETTINGS.dry_run
        max_daily_trades = int(os.getenv("INTRADAY_MAX_DAILY_TRADES", "200"))
        if not self.intraday_trading_engine:
            self.intraday_trading_engine = AutoTradingEngine(
                dry_run=dry_run,
                max_position_value=None,
                max_daily_trades=max_daily_trades,
            )

        market_date = self.market_manager.get_current_market_time().date().isoformat()
        if self.intraday_trade_date != market_date:
            self.intraday_trade_date = market_date
            self.intraday_trading_engine.daily_trade_count = 0
            self.intraday_trading_engine.reset_daily_costs()
            self.intraday_trading_engine.submitted_orders.clear()
            append_log("[INTRADAY] Reset intraday trade counters for new market day")

        order_statuses = self.intraday_trading_engine.refresh_order_statuses()

        self.intraday_trading_engine.update_cost_config(
            {
                "commission_per_share": float(self.intraday_engine.config.get("commission_per_share", 0.0)),
                "min_commission": float(self.intraday_engine.config.get("min_commission", 0.0)),
                "fee_per_order": float(self.intraday_engine.config.get("fee_per_order", 0.0)),
                "slippage_bps": float(self.intraday_engine.config.get("slippage_bps", 0.0)),
                "max_daily_cost_pct": float(self.intraday_engine.config.get("max_daily_cost_pct", 0.0)),
            }
        )
        self.intraday_trading_engine.update_equity(result.get("equity"))
        cost_summary = self.intraday_trading_engine.get_cost_summary()

        if result.get("risk_paused"):
            write_status(
                {
                    "intraday_state": result.get("state", intraday_state),
                    "intraday_summary": {
                        "risk_paused": True,
                        "risk_reason": result.get("risk_reason"),
                        "order_statuses": order_statuses,
                    },
                    "intraday_costs": cost_summary,
                    "intraday_last_run": datetime.now().isoformat(),
                }
            )
            append_log(f"[INTRADAY] Paused: {result.get('risk_reason')}")
            return

        data_coverage = float(result.get("data_coverage", 0.0) or 0.0)
        min_data_coverage = float(self.intraday_engine.config.get("min_data_coverage", 0.0))
        if data_coverage < min_data_coverage:
            write_status(
                {
                    "intraday_state": result.get("state", intraday_state),
                    "intraday_summary": {
                        "risk_paused": False,
                        "data_coverage": data_coverage,
                        "min_data_coverage": min_data_coverage,
                        "skipped": True,
                        "skip_reason": "data_coverage",
                        "order_statuses": order_statuses,
                    },
                    "intraday_costs": cost_summary,
                    "intraday_last_run": datetime.now().isoformat(),
                }
            )
            append_log(
                f"[INTRADAY] Skipping rebalance, data coverage {data_coverage:.0%} < {min_data_coverage:.0%}"
            )
            return

        trading_signals = self.intraday_trading_engine.build_rebalance_signals(
            current_positions=current_positions,
            target_positions=result.get("target_positions", []),
            buying_power=available_funds,
            min_trade_value=float(self.intraday_engine.config.get("min_trade_value", 0.0)),
            buy_price_buffer_pct=float(self.intraday_engine.config.get("buy_price_buffer_pct", 0.0)),
            cooldown_blocked=result.get("cooldown_blocked", {}),
        )

        # CRITICAL: Check stop loss triggers BEFORE any other processing
        # Stop loss sells should be executed with highest priority
        stop_loss_signals = self._check_stop_loss_triggers(current_positions)
        if stop_loss_signals:
            append_log(f"[INTRADAY] Adding {len(stop_loss_signals)} STOP LOSS sells to execution queue")
            existing_sells = trading_signals.get("sell", [])
            # Prepend stop loss signals (highest priority)
            trading_signals["sell"] = stop_loss_signals + existing_sells

        # CLEANUP: Sell positions not in selection list (top_picks)
        # This ensures we exit positions that AI no longer recommends
        cleanup_signals = self._generate_cleanup_sells(current_positions, top_picks, trading_signals)
        if cleanup_signals:
            append_log(f"[INTRADAY] Adding {len(cleanup_signals)} CLEANUP sells for stocks not in selection list")
            existing_sells = trading_signals.get("sell", [])
            trading_signals["sell"] = existing_sells + cleanup_signals

        risk_status = None
        if self.risk_manager:
            buy_signals = trading_signals.get("buy", [])
            if buy_signals:
                append_log(f"[INTRADAY] Risk validating {len(buy_signals)} buy signals")
                validation_results = self.risk_manager.validate_trading_signals_batch(buy_signals)
                approved_buys = [s for s in buy_signals if s.get("risk_validated", False)]
                trading_signals["buy"] = approved_buys
                append_log(
                    f"[INTRADAY] Approved {validation_results['approved_count']}/{validation_results['total_signals']} buys"
                )
            try:
                risk_status = self.risk_manager.get_risk_status_summary()
            except Exception as risk_e:
                append_log(f"[INTRADAY] Risk status error: {risk_e}")

        execution_results = self.intraday_trading_engine.execute_trading_signals(trading_signals)

        # CRITICAL: Position reconciliation after execution
        if execution_results:
            successful_trades = [r for r in execution_results if r.get('success')]
            if successful_trades:
                append_log(f"[INTRADAY] Running position reconciliation for {len(successful_trades)} trades")
                reconciliation_result = self._reconcile_positions(successful_trades, current_positions)

                if not reconciliation_result.get('success'):
                    append_log(f"[INTRADAY] RECONCILIATION FAILED - discrepancies detected")
                    # Store discrepancies in status for review
                    write_status({"reconciliation_discrepancies": reconciliation_result.get('discrepancies', [])})

        state = result.get("state", intraday_state)
        last_trade_ts = state.get("last_trade_ts", {})
        now_ts = int(time.time())
        for res in execution_results or []:
            if res.get("success") and res.get("symbol"):
                last_trade_ts[res["symbol"]] = now_ts
        state["last_trade_ts"] = last_trade_ts

        summary = {
            "buy_signals": len(trading_signals.get("buy", [])),
            "sell_signals": len(trading_signals.get("sell", [])),
            "hold_signals": len(trading_signals.get("hold", [])),
            "executed_trades": len(execution_results or []),
            "data_coverage": data_coverage,
            "daily_cost_total": cost_summary.get("daily_cost_total"),
            "daily_cost_limit": cost_summary.get("daily_cost_limit"),
        }

        # Record intraday risk snapshot to Supabase for analysis
        total_position_value = sum(p.get("market_value", 0) for p in current_positions)
        max_position_weight = 0.0
        equity = result.get("equity", 0)
        if equity > 0 and current_positions:
            position_weights = [p.get("market_value", 0) / equity for p in current_positions]
            max_position_weight = max(position_weights) if position_weights else 0.0

        risk_snapshot = {
            "equity": equity,
            "day_start_equity": state.get("day_start_equity"),
            "daily_pnl": equity - state.get("day_start_equity", equity) if state.get("day_start_equity") else 0,
            "loss_pct": result.get("loss_pct", 0),
            "positions_count": len(current_positions),
            "total_position_value": total_position_value,
            "available_funds": available_funds,
            "buying_power": available_funds,
            "max_position_weight": max_position_weight,
            "es_97_5": risk_status.get("es_97_5") if risk_status else None,
            "portfolio_beta": risk_status.get("portfolio_beta") if risk_status else None,
            "factor_hhi": risk_status.get("factor_hhi") if risk_status else None,
            "daily_costs_total": cost_summary.get("daily_cost_total"),
            "daily_cost_pct": cost_summary.get("daily_cost_pct"),
            "risk_paused": result.get("risk_paused", False),
            "risk_reason": result.get("risk_reason"),
        }
        self._supabase_record_intraday_risk_snapshot(risk_snapshot)

        # Build recent trades list with reasons for frontend display
        recent_trades = []
        for res in execution_results or []:
            trade_record = {
                "symbol": res.get("symbol", ""),
                "action": res.get("action", ""),
                "qty": res.get("qty", 0),
                "price": res.get("price", 0),
                "reason": res.get("reason", "No reason provided"),
                "success": res.get("success", False),
                "order_id": res.get("order_id", ""),
                "timestamp": res.get("timestamp", datetime.now().isoformat()),
            }
            recent_trades.append(trade_record)

        write_status(
            {
                "intraday_state": state,
                "intraday_summary": summary,
                "intraday_costs": cost_summary,
                "intraday_order_statuses": order_statuses,
                "intraday_targets": result.get("target_positions", []),
                "intraday_last_run": datetime.now().isoformat(),
                "intraday_equity": result.get("equity"),
                "intraday_loss_pct": result.get("loss_pct"),
                "intraday_risk_status": risk_status,
                "intraday_recent_trades": recent_trades,
            }
        )
        append_log(
            f"[INTRADAY] Signals: buy={summary['buy_signals']} sell={summary['sell_signals']} hold={summary['hold_signals']}"
        )

    def _execute_auto_trading(self, current_positions, recommended_positions, available_funds):
        """Execute automatic trading based on position analysis."""
        try:
            # Import auto trading engine
            from auto_trading_engine import AutoTradingEngine

            # Get dry_run setting from config (unified configuration)
            dry_run = SETTINGS.dry_run

            # Update risk manager portfolio value from current positions
            if self.risk_manager and current_positions:
                total_value = sum(pos.get('market_value', 0) for pos in current_positions)
                if total_value > 0:
                    self.risk_manager.update_portfolio_value(total_value)

            # Initialize trading engine
            trading_engine = AutoTradingEngine(
                dry_run=dry_run,
                max_position_value=None,  # 动态计算基于购买力
                max_daily_trades=100  # 移除每日交易限制
            )

            # Analyze trading opportunities
            # Pass available_funds to ensure buying power consistency between recommendation and signal generation
            trading_signals = trading_engine.analyze_trading_opportunities(
                current_positions, recommended_positions, available_funds
            )

            # Risk validation: Validate all buy signals before execution
            if self.risk_manager:
                buy_signals = trading_signals.get('buy', [])
                if buy_signals:
                    append_log(f"[RISK_VALIDATION] Validating {len(buy_signals)} buy signals...")

                    # Validate batch of buy signals
                    validation_results = self.risk_manager.validate_trading_signals_batch(buy_signals)

                    # Log validation summary
                    append_log(f"[RISK_VALIDATION] Approved: {validation_results['approved_count']}/{validation_results['total_signals']}")
                    append_log(f"[RISK_VALIDATION] Blocked: {validation_results['blocked_count']}")

                    if validation_results['blocked_signals']:
                        for blocked in validation_results['blocked_signals']:
                            append_log(f"[RISK_BLOCK] {blocked['symbol']}: {blocked['reason']}")

                    # Update trading signals with only approved buy signals
                    approved_buys = [s for s in buy_signals if s.get('risk_validated', False)]
                    trading_signals['buy'] = approved_buys
                    append_log(f"[RISK_VALIDATION] Proceeding with {len(approved_buys)} approved buy signals")

                # Assess portfolio risk if we have positions
                if current_positions:
                    try:
                        risk_assessment = self.risk_manager.assess_portfolio_risk(current_positions)
                        append_log(f"[RISK_ASSESSMENT] ES@97.5%: {risk_assessment['tail_risk_metrics']['es_97_5']:.3f}")
                        append_log(f"[RISK_ASSESSMENT] Market Regime: {risk_assessment['market_regime']}")

                        # Store risk metrics for status updates
                        risk_status = self.risk_manager.get_risk_status_summary()
                    except Exception as risk_e:
                        append_log(f"[RISK_ASSESSMENT] Error assessing portfolio risk: {risk_e}")
                        risk_status = None
                else:
                    risk_status = None
            else:
                risk_status = None

            # Execute trading signals
            execution_results = trading_engine.execute_trading_signals(trading_signals)

            # Get trading summary
            trading_summary = trading_engine.get_trading_summary()

            append_log(f"[AUTO_TRADING] Analysis complete - Buy: {len(trading_signals.get('buy', []))}, "
                      f"Sell: {len(trading_signals.get('sell', []))}, Hold: {len(trading_signals.get('hold', []))}")

            if execution_results:
                append_log(f"[AUTO_TRADING] Executed {len(execution_results)} trades")
                for result in execution_results:
                    if result.get('success'):
                        append_log(f"  OK {result.get('action', 'TRADE')} {result.get('qty', 0)} {result.get('symbol', '')} @ ${result.get('price', 0):.2f}")
                    else:
                        append_log(f"  FAIL {result.get('action', 'TRADE')} {result.get('symbol', '')}: {result.get('error', 'Unknown error')}")

            trading_results = {
                "trading_signals": trading_signals,
                "execution_results": execution_results,
                "trading_summary": trading_summary,
                "risk_status": risk_status,  # Include risk status
                "last_analysis_time": datetime.now().isoformat()
            }

            # Update AI models from trading results (reinforcement learning)
            if self.ai_manager and self.ai_manager.is_enabled():
                try:
                    self.ai_manager.update_from_trading_results(trading_results)
                    append_log("[AI_LEARNING] Updated AI models from trading results")
                except Exception as ai_e:
                    append_log(f"[AI_LEARNING] Failed to update AI: {ai_e}")

            return trading_results

        except Exception as e:
            append_log(f"[AUTO_TRADING ERROR] {e}")
            return {
                "error": str(e),
                "last_analysis_time": datetime.now().isoformat()
            }

    def _get_tiger_available_funds_internal(self):
        """Internal implementation - called by retry wrapper."""
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
        from pathlib import Path

        # Use props configuration
        props_dir = str((Path(__file__).parent.parent.parent / "props").resolve())
        client_config = TigerOpenClientConfig(props_path=props_dir)
        trade_client = TradeClient(client_config)

        # Get account assets to retrieve available funds
        assets = trade_client.get_assets()

        if assets and len(assets) > 0:
            asset = assets[0]
            available_funds = None

            # Priority 1: segments['S'].available_funds
            segments = getattr(asset, 'segments', {})
            if segments and 'S' in segments:
                sec_segment = segments['S']
                available_funds = getattr(sec_segment, 'available_funds', None)
                if available_funds and available_funds > 0:
                    append_log(f"[TIGER] Using available_funds from segments: ${float(available_funds):,.2f}")

            # Priority 2: summary.cash
            summary = getattr(asset, 'summary', None)
            if (not available_funds or available_funds <= 0) and summary:
                available_funds = getattr(summary, 'cash', None)
                if available_funds and available_funds > 0:
                    append_log(f"[TIGER] Using cash from summary: ${float(available_funds):,.2f}")

            # Priority 3: summary.buying_power
            if (not available_funds or available_funds <= 0) and summary:
                available_funds = getattr(summary, 'buying_power', None)
                if available_funds and available_funds > 0:
                    append_log(f"[TIGER] WARNING: Using buying_power (includes margin): ${float(available_funds):,.2f}")

            if available_funds is not None and available_funds > 0:
                available_funds = float(available_funds)
                append_log(f"[TIGER] Real available funds: ${available_funds:,.2f}")
                self._last_known_funds = available_funds
                if summary:
                    net_liq = getattr(summary, 'net_liquidation', 0)
                    if net_liq and net_liq > 0:
                        append_log(f"[TIGER] Total account value: ${float(net_liq):,.2f}")
                return available_funds
            else:
                append_log("[TIGER] WARNING: Available funds is 0 or None")
                return 0.0
        else:
            append_log("[TIGER] ERROR: No assets found in Tiger account")
            return 0.0

    def _get_tiger_available_funds(self):
        """
        Get real available funds from Tiger account API with retry logic.

        CRITICAL FIX #10: Added exponential backoff retry to prevent
        single network failures from halting all trading.
        """
        import time
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                return self._get_tiger_available_funds_internal()
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    append_log(f"[TIGER] API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                    append_log(f"[TIGER] Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    append_log(f"[TIGER] ERROR after {max_retries} attempts: {type(e).__name__}: {e}")
                    logger.error(f"Tiger available funds error: {e}", exc_info=True)
                    if hasattr(self, '_last_known_funds') and self._last_known_funds > 0:
                        append_log(f"[TIGER] Using cached funds: ${self._last_known_funds:,.2f}")
                        return self._last_known_funds
                    return 0.0
        return 0.0

    def _get_tiger_positions(self):
        """Get real positions from Tiger account."""
        try:
            append_log("[TIGER] Starting Tiger positions retrieval...")

            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.trade.trade_client import TradeClient
            from pathlib import Path

            # Use props configuration (go up to quant_system_full directory)
            props_dir = str((Path(__file__).parent.parent.parent / "props").resolve())
            append_log(f"[TIGER] Using props directory: {props_dir}")

            # Verify props file exists
            props_file = Path(props_dir) / "tiger_openapi_config.properties"
            append_log(f"[TIGER] Props file exists: {props_file.exists()}, path: {props_file}")

            client_config = TigerOpenClientConfig(props_path=props_dir)
            append_log("[TIGER] TigerOpenClientConfig created successfully")

            trade_client = TradeClient(client_config)
            append_log("[TIGER] TradeClient created successfully")

            positions = trade_client.get_positions()
            append_log(f"[TIGER] Raw positions response: {type(positions)}, length: {len(positions) if positions else 0}")

            real_positions = []

            if positions:
                for i, pos in enumerate(positions):
                    try:
                        contract = getattr(pos, 'contract', None)
                        symbol = getattr(contract, 'symbol', f'Position_{i+1}') if contract else f'Position_{i+1}'
                        quantity = getattr(pos, 'quantity', 0)
                        avg_cost = getattr(pos, 'average_cost', 0)
                        market_price = getattr(pos, 'market_price', 0)
                        market_value = getattr(pos, 'market_value', 0)
                        unrealized_pnl = getattr(pos, 'unrealized_pnl', 0)

                        append_log(f"[TIGER] Processing position {i+1}: {symbol}, qty={quantity}")

                        if quantity != 0:
                            calculated_value = market_value if market_value != 0 else abs(quantity) * market_price

                            position_item = {
                                "symbol": symbol,
                                "quantity": quantity,
                                "average_cost": round(avg_cost, 2),
                                "market_price": round(market_price, 2),
                                "market_value": round(calculated_value, 2),
                                "unrealized_pnl": round(unrealized_pnl, 2),
                                "position_type": "REAL"
                            }

                            real_positions.append(position_item)
                            append_log(f"[TIGER] Added position: {symbol} = ${calculated_value:.2f}")

                    except Exception as e:
                        append_log(f"[TIGER] Error processing position {i+1}: {e}")

            append_log(f"[TIGER] Successfully retrieved {len(real_positions)} real positions")
            return real_positions

        except Exception as e:
            append_log(f"[TIGER] Error getting positions: {type(e).__name__}: {e}")
            logger.error(f"Tiger positions error: {e}")
            return []

    def _reconcile_positions(self, expected_trades: List[Dict], pre_trade_positions: List[Dict]) -> Dict:
        """
        Reconcile positions after trade execution.

        Verifies that executed trades resulted in expected position changes.

        Args:
            expected_trades: List of trades that were expected to execute
            pre_trade_positions: Positions before trading

        Returns:
            Reconciliation result with any discrepancies
        """
        try:
            # Wait for settlement
            time.sleep(2)

            # Fetch current positions
            post_trade_positions = self._get_tiger_positions()

            # Build position maps
            pre_map = {p['symbol']: p.get('quantity', 0) for p in pre_trade_positions}
            post_map = {p['symbol']: p.get('quantity', 0) for p in post_trade_positions}

            discrepancies = []
            reconciled_trades = 0

            for trade in expected_trades:
                if not trade.get('success'):
                    continue

                symbol = trade.get('symbol', '')
                action = trade.get('action', '')
                qty = trade.get('qty', 0)

                pre_qty = pre_map.get(symbol, 0)
                post_qty = post_map.get(symbol, 0)

                if action == 'BUY':
                    expected_qty = pre_qty + qty
                elif action == 'SELL':
                    expected_qty = max(0, pre_qty - qty)
                else:
                    expected_qty = pre_qty

                if abs(post_qty - expected_qty) > 0.1:  # Allow small rounding
                    discrepancy = {
                        'symbol': symbol,
                        'action': action,
                        'expected_qty': expected_qty,
                        'actual_qty': post_qty,
                        'difference': post_qty - expected_qty,
                        'timestamp': datetime.now().isoformat()
                    }
                    discrepancies.append(discrepancy)
                    append_log(f"[RECONCILE] DISCREPANCY: {symbol} expected {expected_qty} got {post_qty}")
                else:
                    reconciled_trades += 1

            result = {
                'success': len(discrepancies) == 0,
                'reconciled_trades': reconciled_trades,
                'discrepancies': discrepancies,
                'total_trades': len([t for t in expected_trades if t.get('success')]),
                'timestamp': datetime.now().isoformat()
            }

            if discrepancies:
                append_log(f"[RECONCILE] WARNING: {len(discrepancies)} position discrepancies detected")
            else:
                append_log(f"[RECONCILE] All {reconciled_trades} trades reconciled successfully")

            return result

        except Exception as e:
            append_log(f"[RECONCILE] Error during reconciliation: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _check_stop_loss_triggers(self, current_positions: List[Dict]) -> List[Dict]:
        """
        Check positions against stop loss levels and generate sell signals.

        This method checks if any current positions have fallen below their
        stop loss prices and returns sell signals for immediate execution.

        Args:
            current_positions: List of current Tiger positions

        Returns:
            List of sell signals for positions that triggered stop loss
        """
        stop_loss_signals = []

        try:
            status = read_status()
            stop_loss_config = status.get("stop_loss_config", {})

            # Default stop loss percentage if not configured per-symbol
            default_stop_loss_pct = float(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.04"))  # 4% default

            for pos in current_positions:
                symbol = pos.get("symbol", "")
                if not symbol:
                    continue

                quantity = pos.get("quantity", 0)
                if quantity <= 0:  # Only check long positions
                    continue

                avg_cost = pos.get("average_cost", 0)
                market_price = pos.get("market_price", 0)

                if avg_cost <= 0 or market_price <= 0:
                    continue

                # Get stop loss price for this symbol (or calculate from default %)
                stop_loss_price = stop_loss_config.get(symbol, {}).get("stop_loss_price")

                if stop_loss_price is None:
                    # Calculate based on default percentage
                    stop_loss_price = avg_cost * (1 - default_stop_loss_pct)

                # Check if stop loss triggered
                if market_price <= stop_loss_price:
                    loss_pct = (market_price - avg_cost) / avg_cost * 100
                    append_log(f"[STOP_LOSS] TRIGGERED for {symbol}: ${market_price:.2f} <= ${stop_loss_price:.2f} (loss: {loss_pct:.2f}%)")

                    sell_signal = {
                        "symbol": symbol,
                        "action": "SELL",
                        "qty": quantity,
                        "price": market_price,
                        "reason": f"STOP_LOSS_TRIGGERED: price ${market_price:.2f} below stop ${stop_loss_price:.2f}",
                        "trigger_type": "stop_loss",
                        "stop_loss_price": stop_loss_price,
                        "avg_cost": avg_cost,
                        "loss_pct": loss_pct,
                        "priority": "URGENT",
                        "estimated_value": market_price * quantity
                    }
                    stop_loss_signals.append(sell_signal)

            if stop_loss_signals:
                append_log(f"[STOP_LOSS] {len(stop_loss_signals)} positions triggered stop loss")

        except Exception as e:
            append_log(f"[STOP_LOSS] Error checking stop loss triggers: {e}")
            logger.error(f"Stop loss check error: {e}")

        return stop_loss_signals

    def _generate_cleanup_sells(self, current_positions: List[Dict], top_picks: List[Dict], existing_signals: Dict) -> List[Dict]:
        """
        Generate sell signals for positions not in the selection list (top_picks).

        This ensures we exit positions that AI no longer recommends, keeping the
        portfolio aligned with the current stock selection strategy.

        Args:
            current_positions: List of current Tiger positions
            top_picks: List of AI-selected stocks from selection_results
            existing_signals: Current trading signals (to avoid duplicate sells)

        Returns:
            List of sell signals for positions not in selection list
        """
        cleanup_signals = []

        try:
            # Get symbols from selection list
            selection_symbols = set()
            for pick in top_picks:
                symbol = pick.get("symbol", "")
                if symbol:
                    selection_symbols.add(symbol.upper())

            # Get symbols already being sold
            existing_sell_symbols = set()
            for sell_signal in existing_signals.get("sell", []):
                symbol = sell_signal.get("symbol", "")
                if symbol:
                    existing_sell_symbols.add(symbol.upper())

            # Check each position
            for pos in current_positions:
                symbol = pos.get("symbol", "")
                if not symbol:
                    continue

                symbol_upper = symbol.upper()
                quantity = pos.get("quantity", 0)

                if quantity <= 0:  # Only check long positions
                    continue

                # Skip if already in sell signals
                if symbol_upper in existing_sell_symbols:
                    continue

                # Check if NOT in selection list
                if symbol_upper not in selection_symbols:
                    market_price = pos.get("market_price", 0)
                    avg_cost = pos.get("average_cost", 0)
                    market_value = pos.get("market_value", 0) or (market_price * quantity)

                    pnl_pct = 0
                    if avg_cost > 0:
                        pnl_pct = (market_price - avg_cost) / avg_cost * 100

                    append_log(f"[CLEANUP] {symbol} not in selection list - generating sell signal (qty={quantity}, pnl={pnl_pct:.2f}%)")

                    cleanup_signal = {
                        "symbol": symbol,
                        "action": "SELL",
                        "qty": quantity,
                        "price": market_price,
                        "reason": f"NOT_IN_SELECTION: {symbol} removed from AI selection list",
                        "trigger_type": "cleanup",
                        "avg_cost": avg_cost,
                        "pnl_pct": pnl_pct,
                        "priority": "NORMAL",
                        "estimated_value": market_value
                    }
                    cleanup_signals.append(cleanup_signal)

            if cleanup_signals:
                total_value = sum(s.get("estimated_value", 0) for s in cleanup_signals)
                append_log(f"[CLEANUP] Generated {len(cleanup_signals)} cleanup sells (total value: ${total_value:,.2f})")
            elif current_positions and selection_symbols:
                append_log(f"[CLEANUP] All {len(current_positions)} positions are in selection list ({len(selection_symbols)} stocks)")

        except Exception as e:
            append_log(f"[CLEANUP] Error generating cleanup sells: {e}")
            logger.error(f"Cleanup sell generation error: {e}")

        return cleanup_signals

    def demo_trading_task(self):
        """Demo trading task - simulates trading during market hours."""
        if not self.market_manager.is_market_active():
            return

        # Simulate price movement
        self.price *= (1.0 + random.uniform(-0.003, 0.003))
        signal = random.choice(["BUY", "SELL", "HOLD"])

        if signal == "BUY" and self.position == 0:
            self.position = 100
            self.entry_price = self.price
            append_log(f"[TRADING] BUY 100 @ ${self.entry_price:.2f}")

        elif signal == "SELL" and self.position > 0:
            realized_pnl = (self.price - self.entry_price) * self.position
            self.pnl += realized_pnl
            append_log(f"[TRADING] SELL 100 @ ${self.price:.2f} | Realized PnL: ${realized_pnl:.2f}")
            self.position = 0
            self.entry_price = None

        else:
            append_log(f"[TRADING] HOLD @ ${self.price:.2f}")

        # Update status
        write_status({
            "pnl": round(self.pnl, 2),
            "positions": [] if self.position == 0 else [
                {"symbol": "DEMO", "qty": self.position, "price": round(self.price, 2)}
            ],
            "last_signal": signal,
            "bot": "running",
            "market_phase": self.current_phase.value if self.current_phase else "unknown"
        })

    def stock_selection_task(self):
        """Stock selection task - runs during market closed periods. Supports improved strategies V2 (optional)."""
        if not self.market_manager.should_run_selection_tasks():
            return

        # Start Supabase run tracking
        run_id = self._supabase_start_run('selection')
        start_time = time.time()
        error_msg = None

        try:
            # === IMPROVED STRATEGIES V2 (OPTIONAL) ===
            # Check if improved strategies are enabled via environment variable
            use_improved = os.getenv('USE_IMPROVED_STRATEGIES', 'false').lower() == 'true'

            if use_improved:
                try:
                    # Check if weighted scoring mode is enabled
                    use_weighted_scoring = os.getenv('USE_WEIGHTED_SCORING', 'true').lower() == 'true'

                    if use_weighted_scoring:
                        from bot.selection_strategies.weighted_scoring_orchestrator import WeightedScoringOrchestrator
                        from bot.selection_strategies.base_strategy import SelectionCriteria

                        append_log("[SELECTION] Using WEIGHTED SCORING with 4 strategies (Momentum 40% / Value 30% / Technical 15% / Earnings 15%)")

                        # Initialize weighted scoring orchestrator
                        config_path = os.getenv('IMPROVED_STRATEGIES_CONFIG', 'bot/config/selection_config_v2.json')
                        orchestrator = WeightedScoringOrchestrator(config_path=config_path)
                    else:
                        from bot.selection_strategies.strategy_orchestrator_v2 import StrategyOrchestratorV2
                        from bot.selection_strategies.base_strategy import SelectionCriteria

                        append_log("[SELECTION] Using IMPROVED strategies V2 with risk management")

                        # Initialize improved orchestrator
                        config_path = os.getenv('IMPROVED_STRATEGIES_CONFIG', 'bot/config/selection_config_v2.json')
                        orchestrator = StrategyOrchestratorV2(enable_improved=True, config_path=config_path)

                    # Get stock universe
                    universe = self._get_stock_universe()

                    # Configure selection criteria
                    criteria = SelectionCriteria(
                        max_stocks=int(os.getenv('SELECTION_RESULT_SIZE', 20)),
                        min_market_cap=float(os.getenv('SELECTION_MIN_MARKET_CAP', 1e8)),
                        max_market_cap=float(os.getenv('SELECTION_MAX_MARKET_CAP', 5e12)),
                        min_volume=int(os.getenv('SELECTION_MIN_VOLUME', 50000)),
                        min_price=float(os.getenv('SELECTION_MIN_PRICE', 1.0)),
                        max_price=float(os.getenv('SELECTION_MAX_PRICE', 2000.0)),
                        min_score_threshold=float(os.getenv('SELECTION_MIN_SCORE', '80.0'))
                    )

                    # Run orchestrator (different method names for different orchestrators)
                    if use_weighted_scoring:
                        combined_selections = orchestrator.select_stocks(universe, criteria)
                        append_log(f"[SELECTION] Weighted scoring completed: {len(combined_selections)} stocks selected")
                    else:
                        combined_selections = orchestrator.select_stocks_with_risk_management(universe, criteria)
                        append_log(f"[SELECTION] Improved strategies V2 completed: {len(combined_selections)} stocks selected")

                    # === LLM Enhancement (Optional, Independent Module) ===
                    # Apply LLM enhancement to improved strategies results
                    try:
                        from bot.llm_enhancement import get_llm_pipeline
                        from bot.llm_enhancement.config import LLMEnhancementConfig

                        # Re-initialize LLM_CONFIG after load_dotenv()
                        LLM_CONFIG = LLMEnhancementConfig()

                        if LLM_CONFIG.is_available():
                            append_log("[LLM] LLM Enhancement is ENABLED")
                            append_log(f"[LLM] Configuration: mode={LLM_CONFIG.mode}, "
                                      f"model_triage={LLM_CONFIG.model_triage}, "
                                      f"model_deep={LLM_CONFIG.model_deep}")
                            append_log(f"[LLM] Funnel: {len(combined_selections)} stocks -> "
                                      f"{LLM_CONFIG.m_triage} triage -> "
                                      f"{LLM_CONFIG.m_final} deep analysis")

                            # Run LLM enhancement
                            llm_result = get_llm_pipeline().enhance(combined_selections)

                            if llm_result.get("enabled"):
                                metrics = llm_result.get("metrics", {})
                                append_log(f"[LLM] Enhancement execution completed in {metrics.get('execution_time', 0):.2f}s")
                                append_log(f"[LLM] API Calls: {metrics.get('llm_calls', 0)}, Cache Hits: {metrics.get('cache_hits', 0)}")
                                append_log(f"[LLM] Cost this run: ${metrics.get('cost_usd', 0):.4f}")

                                if not llm_result.get("errors"):
                                    combined_selections = llm_result["enhanced_results"]
                                    append_log("[LLM] Enhancement applied successfully - using LLM-enhanced results")
                        else:
                            append_log("[LLM] LLM Enhancement is DISABLED or not configured")
                    except Exception as llm_e:
                        append_log(f"[LLM] Enhancement failed: {type(llm_e).__name__}: {llm_e}")
                        logger.error(f"LLM Enhancement error: {llm_e}")

                    # Apply score filter and stock limit (consistent with original strategies)
                    min_score_threshold = float(os.getenv('SELECTION_MIN_SCORE', '80.0'))
                    max_stocks = int(os.getenv('SELECTION_RESULT_SIZE', '10'))

                    # Ensure avg_score exists in improved strategy results
                    for selection in combined_selections:
                        if 'avg_score' not in selection:
                            selection['avg_score'] = selection.get('score', 0.0)

                    # Sort by score (in case not already sorted)
                    combined_selections.sort(key=lambda x: x.get('avg_score', 0), reverse=True)

                    # Save ALL stock scores to a detailed log file (before filtering)
                    try:
                        # Calculate state_dir path and ensure it exists
                        state_dir = os.path.join(os.path.dirname(__file__), '..', 'state')
                        os.makedirs(state_dir, exist_ok=True)
                        all_scores_file = os.path.join(state_dir, 'all_stock_scores.json')

                        # Handle empty results case
                        if len(combined_selections) == 0:
                            all_scores_data = {
                                'timestamp': datetime.now().isoformat(),
                                'total_stocks_evaluated': 0,
                                'min_score_threshold': min_score_threshold,
                                'strategy_weights': {
                                    'momentum': 0.40,
                                    'value': 0.30,
                                    'technical': 0.15,
                                    'earnings': 0.15
                                },
                                'all_stocks': [],
                                'debug_info': {
                                    'error': 'No stocks were scored by any strategy',
                                    'universe_size': len(universe),
                                    'possible_causes': [
                                        'All stocks filtered out by basic criteria (market cap, price, volume)',
                                        'All stocks rejected by individual strategy scoring',
                                        'Data fetch failures for all stocks'
                                    ]
                                }
                            }
                            append_log(f"[SELECTION] WARNING: No stocks were evaluated - saving diagnostic info to all_stock_scores.json")
                        else:
                            all_scores_data = {
                                'timestamp': datetime.now().isoformat(),
                                'total_stocks_evaluated': len(combined_selections),
                                'min_score_threshold': min_score_threshold,
                                'strategy_weights': {
                                    'momentum': 0.40,
                                    'value': 0.30,
                                    'technical': 0.15,
                                    'earnings': 0.15
                                },
                                'all_stocks': [
                                    {
                                        'rank': idx + 1,
                                        'symbol': s.get('symbol', 'UNKNOWN'),
                                        'avg_score': round(s.get('avg_score', 0), 2),
                                        'strategy_count': s.get('strategy_count', 0),
                                        'action': s.get('action', 'unknown'),
                                        'component_scores': s.get('component_scores', {}),
                                        'reasoning': s.get('reasoning', ''),
                                        'passes_threshold': s.get('avg_score', 0) >= min_score_threshold
                                    }
                                    for idx, s in enumerate(combined_selections)
                                ]
                            }

                        with open(all_scores_file, 'w', encoding='utf-8') as f:
                            json.dump(all_scores_data, f, indent=2, ensure_ascii=False)

                        if len(combined_selections) > 0:
                            append_log(f"[SELECTION] Saved detailed scores for {len(combined_selections)} stocks to all_stock_scores.json")

                            # Log score distribution
                            score_ranges = {
                                '90-100': len([s for s in combined_selections if 90 <= s.get('avg_score', 0) <= 100]),
                                '80-89': len([s for s in combined_selections if 80 <= s.get('avg_score', 0) < 90]),
                                '70-79': len([s for s in combined_selections if 70 <= s.get('avg_score', 0) < 80]),
                                '60-69': len([s for s in combined_selections if 60 <= s.get('avg_score', 0) < 70]),
                                '<60': len([s for s in combined_selections if s.get('avg_score', 0) < 60])
                            }
                            append_log(f"[SELECTION] Score distribution: {score_ranges}")

                    except Exception as score_log_error:
                        append_log(f"[SELECTION] Warning: Failed to save detailed scores: {score_log_error}")

                    # Apply filters
                    filtered_selections = [s for s in combined_selections if s.get('avg_score', 0) >= min_score_threshold]

                    if len(filtered_selections) < len(combined_selections):
                        append_log(f"[SELECTION] Score filter: {len(combined_selections)} -> {len(filtered_selections)} stocks (threshold: {min_score_threshold})")

                    if len(filtered_selections) > max_stocks:
                        append_log(f"[SELECTION] Limiting to top {max_stocks} stocks")
                        filtered_selections = filtered_selections[:max_stocks]
                    elif len(filtered_selections) < max_stocks:
                        append_log(f"[SELECTION] Selected {len(filtered_selections)} stocks (less than max {max_stocks}, but all qualified)")

                    combined_selections = filtered_selections

                    # Update status with selection results (after LLM enhancement and filtering)
                    self._update_selection_status(combined_selections)

                    # Record selection results to Supabase (improved strategies)
                    if combined_selections:
                        self._supabase_record_selection(
                            run_id=run_id,
                            strategy='weighted_scoring_v2' if use_weighted_scoring else 'improved_v2',
                            picks=combined_selections[:20],
                            total=len(combined_selections)
                        )
                        # Record top signals for analysis
                        for selection in combined_selections[:50]:
                            self._supabase_record_trade_signal(selection, run_id)

                    # Complete run tracking before early return
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._supabase_complete_run(run_id, duration_ms, None)

                    return  # Exit early on success

                except Exception as improved_error:
                    # Log error but continue to original strategies (automatic fallback)
                    append_log(f"[SELECTION] Improved strategies V2 failed: {improved_error}")
                    append_log("[SELECTION] Falling back to ORIGINAL strategies")
                    logger.warning(f"Improved strategies error, falling back to original: {improved_error}")
                    # Fall through to original code below

            # === ORIGINAL STRATEGIES (DEFAULT) ===
            # Import selection strategies
            from bot.selection_strategies.value_momentum import ValueMomentumStrategy
            from bot.selection_strategies.technical_breakout import TechnicalBreakoutStrategy
            from bot.selection_strategies.earnings_momentum import EarningsMomentumStrategy
            from bot.selection_strategies.base_strategy import SelectionCriteria

            append_log("[SELECTION] Starting comprehensive stock selection process")

            # Define stock universe (top 500 stocks by market cap)
            universe = self._get_stock_universe()

            # Configure selection criteria
            criteria = SelectionCriteria(
                max_stocks=10,
                min_market_cap=1e9,  # $1B minimum
                max_market_cap=1e12,  # $1T maximum
                min_volume=100000,
                min_price=5.0,
                max_price=500.0,
                min_score_threshold=50.0
            )

            # Run multiple selection strategies
            all_results = {}
            strategies = [
                ValueMomentumStrategy(),
                TechnicalBreakoutStrategy(),
                EarningsMomentumStrategy()
            ]

            for strategy in strategies:
                try:
                    append_log(f"[SELECTION] Running {strategy.name} strategy")
                    results = strategy.select_stocks(universe, criteria)
                    all_results[strategy.name] = results

                    # Log strategy results
                    summary = results.to_summary()
                    append_log(f"[SELECTION] {strategy.name}: {summary['total_selected']} stocks selected, "
                             f"avg_score: {summary.get('avg_score', 0):.1f}, "
                             f"time: {summary['execution_time']:.1f}s")

                except Exception as e:
                    append_log(f"[SELECTION] Error running {strategy.name}: {e}")
                    continue

            # Combine and rank results
            combined_selections = self._combine_strategy_results(all_results)

            # === LLM Enhancement (Optional, Independent Module) ===
            try:
                from bot.llm_enhancement import get_llm_pipeline
                from bot.llm_enhancement.config import LLMEnhancementConfig

                # IMPORTANT: Re-initialize LLM_CONFIG after load_dotenv()
                # The module-level LLM_CONFIG was initialized before .env was loaded
                LLM_CONFIG = LLMEnhancementConfig()

                # Log LLM configuration status
                if LLM_CONFIG.is_available():
                    append_log("[LLM] LLM Enhancement is ENABLED")
                    append_log(f"[LLM] Configuration: mode={LLM_CONFIG.mode}, "
                              f"model_triage={LLM_CONFIG.model_triage}, "
                              f"model_deep={LLM_CONFIG.model_deep}")
                    append_log(f"[LLM] Funnel: {len(combined_selections)} stocks -> "
                              f"{LLM_CONFIG.m_triage} triage -> "
                              f"{LLM_CONFIG.m_final} deep analysis")
                else:
                    append_log("[LLM] LLM Enhancement is DISABLED or not configured")
                    if not LLM_CONFIG.enabled:
                        append_log("[LLM] Reason: ENABLE_LLM_ENHANCEMENT=false")
                    elif not LLM_CONFIG.openai_api_key:
                        append_log("[LLM] Reason: OPENAI_API_KEY not set")

                # Run LLM enhancement
                llm_result = get_llm_pipeline().enhance(combined_selections)

                # Log detailed results
                if llm_result.get("enabled"):
                    metrics = llm_result.get("metrics", {})
                    errors = llm_result.get("errors", [])

                    # Log execution metrics
                    append_log(f"[LLM] Enhancement execution completed in {metrics.get('execution_time', 0):.2f}s")
                    append_log(f"[LLM] API Calls: {metrics.get('llm_calls', 0)}, "
                              f"Cache Hits: {metrics.get('cache_hits', 0)}")
                    append_log(f"[LLM] Cost this run: ${metrics.get('cost_usd', 0):.4f}")

                    # Log enhancement effects
                    base_count = metrics.get('base_count', 0)
                    enhanced_count = metrics.get('enhanced_count', 0)
                    append_log(f"[LLM] Processed: {base_count} base -> {enhanced_count} enhanced stocks")

                    if not errors:
                        combined_selections = llm_result["enhanced_results"]
                        append_log("[LLM] Enhancement applied successfully - using LLM-enhanced results")

                        # Log top stock changes if available
                        base_results = llm_result.get("base_results", [])
                        enhanced_results = llm_result.get("enhanced_results", [])
                        if base_results and enhanced_results:
                            base_top = base_results[0]['symbol'] if base_results else "N/A"
                            enhanced_top = enhanced_results[0]['symbol'] if enhanced_results else "N/A"
                            if base_top != enhanced_top:
                                append_log(f"[LLM] Top pick changed: {base_top} -> {enhanced_top}")
                    else:
                        append_log(f"[LLM] Enhancement completed with {len(errors)} error(s)")
                        for error in errors[:3]:  # Log first 3 errors
                            append_log(f"[LLM] Error: {error}")
                else:
                    append_log("[LLM] Enhancement was not enabled for this run")

            except Exception as llm_e:
                append_log(f"[LLM] Enhancement failed: {type(llm_e).__name__}: {llm_e}")
                logger.error(f"LLM Enhancement error: {llm_e}")

            # Update status with selection results
            self._update_selection_status(combined_selections)

            append_log(f"[SELECTION] Selection process completed. "
                      f"Final selection: {len(combined_selections)} stocks")

            # Record selection results to Supabase
            if combined_selections:
                self._supabase_record_selection(
                    run_id=run_id,
                    strategy='multi_strategy_combined',
                    picks=combined_selections[:20],  # Top 20 picks
                    total=len(combined_selections)
                )

                # Record top signals to Supabase for analysis
                for selection in combined_selections[:50]:  # Top 50 for analysis
                    self._supabase_record_trade_signal(selection, run_id)

        except Exception as e:
            append_log(f"[SELECTION] Stock selection task failed: {e}")
            logger.error(f"Stock selection error: {e}")
            error_msg = str(e)
        finally:
            # Complete Supabase run tracking
            duration_ms = int((time.time() - start_time) * 1000)
            self._supabase_complete_run(run_id, duration_ms, error_msg)

    def factor_crowding_monitoring_task(self):
        """Factor crowding monitoring task - detects factor crowding every 5 minutes."""
        if not self.market_manager.is_market_active():
            return

        if not self.crowding_monitor:
            return

        # Start Supabase run tracking
        run_id = self._supabase_start_run('factor_crowding')
        start_time = time.time()
        error_msg = None

        try:
            # Get current positions
            positions = self._get_tiger_positions()

            if not positions:
                return

            # Build factor exposures from positions
            # For now, use standardized factor exposures
            # In production, calculate actual factor loadings per position
            factor_exposures = {
                'momentum': 0.25,
                'value': 0.15,
                'quality': 0.20,
                'volatility': -0.10,
                'size': 0.05
            }

            # Get portfolio value for regime assessment
            status = read_status()
            portfolio_value = status.get('real_portfolio_value', 500000.0)

            # Assess market regime (simplified)
            import numpy as np
            vix_estimate = 20.0  # Default VIX value
            market_data = {
                'vix': vix_estimate,
                'market_stress_index': 0.5,
                'correlation_spike': False
            }

            # Create factor exposure arrays for crowding analysis
            n_positions = len(positions)
            if n_positions > 0:
                # Generate exposure arrays (simplified - in production, use actual factor loadings)
                factor_exposure_arrays = {
                    'momentum': np.random.normal(0.25, 0.1, n_positions),
                    'value': np.random.normal(0.15, 0.08, n_positions),
                    'quality': np.random.normal(0.20, 0.12, n_positions),
                    'volatility': np.random.normal(-0.10, 0.05, n_positions),
                    'size': np.random.normal(0.05, 0.03, n_positions)
                }

                # Monitor factor crowding
                crowding_results = self.crowding_monitor.monitor_factor_crowding(
                    factor_exposures=factor_exposure_arrays,
                    portfolio_weights=None,
                    benchmark_weights=None,
                    market_data=market_data
                )

                # Extract key metrics
                if crowding_results:
                    # Find most crowded factor
                    most_crowded = max(crowding_results, key=lambda x: x.crowding_score)

                    # Calculate average metrics
                    avg_hhi = np.mean([r.herfindahl_index for r in crowding_results])
                    avg_gini = np.mean([r.gini_coefficient for r in crowding_results])
                    max_correlation = max([r.max_correlation for r in crowding_results])

                    # Determine overall crowding level
                    high_crowding_count = sum(1 for r in crowding_results if r.crowding_level.value in ['HIGH', 'EXTREME'])

                    if high_crowding_count >= 2:
                        crowding_level = 'HIGH'
                    elif high_crowding_count == 1:
                        crowding_level = 'MODERATE'
                    else:
                        crowding_level = 'LOW'

                    # Log crowding metrics
                    append_log(f"[CROWDING] HHI: {avg_hhi:.3f}, Gini: {avg_gini:.3f}, Max Corr: {max_correlation:.3f}, Level: {crowding_level}")

                    # Alert on high crowding
                    if crowding_level == 'HIGH':
                        append_log(f"[CROWDING ALERT] High factor crowding detected! Most crowded factor: {most_crowded.factor_name} (score: {most_crowded.crowding_score:.1f})")

                    # Build crowding analysis summary
                    crowding_analysis = {
                        'timestamp': datetime.now().isoformat(),
                        'crowding_level': crowding_level,
                        'factor_hhi': round(avg_hhi, 4),
                        'gini_coefficient': round(avg_gini, 4),
                        'max_correlation': round(max_correlation, 4),
                        'most_crowded_factor': most_crowded.factor_name,
                        'most_crowded_score': round(most_crowded.crowding_score, 2),
                        'high_crowding_factors': high_crowding_count,
                        'total_factors_analyzed': len(crowding_results),
                        'market_regime': self.crowding_monitor.current_regime.value,
                        'factor_details': [
                            {
                                'factor': r.factor_name,
                                'hhi': round(r.herfindahl_index, 4),
                                'gini': round(r.gini_coefficient, 4),
                                'crowding_score': round(r.crowding_score, 2),
                                'crowding_level': r.crowding_level.value
                            }
                            for r in crowding_results
                        ]
                    }

                    # Update status
                    write_status({
                        'factor_crowding_analysis': crowding_analysis,
                        'crowding_check_time': datetime.now().isoformat()
                    })

                    # Record each factor's crowding data to Supabase
                    for factor_detail in crowding_analysis.get('factor_details', []):
                        self._supabase_record_factor_crowding({
                            'factor_name': factor_detail['factor'],
                            'hhi': factor_detail['hhi'],
                            'gini_coefficient': factor_detail['gini'],
                            'crowding_score': factor_detail['crowding_score'],
                            'crowding_level': factor_detail['crowding_level'],
                            'portfolio_exposure': None,  # Can be added if available
                            'market_exposure': None
                        })

                    # Update task statistics
                    self.task_stats['factor_crowding_runs'] += 1

        except Exception as e:
            logger.error(f"Factor crowding monitoring failed: {e}")
            append_log(f"[CROWDING ERROR] {e}")
            error_msg = str(e)
        finally:
            # Complete Supabase run tracking
            duration_ms = int((time.time() - start_time) * 1000)
            self._supabase_complete_run(run_id, duration_ms, error_msg)

    def compliance_monitoring_task(self):
        """Compliance monitoring task - runs every 1 minute during trading."""
        if not self.market_manager.is_market_active():
            return

        if not self.compliance_monitor:
            return

        # Start Supabase run tracking
        run_id = self._supabase_start_run('compliance')
        start_time = time.time()
        error_msg = None

        try:
            # Get current positions for compliance checks
            positions = self._get_tiger_positions()

            # Check all compliance rules
            violations = self.compliance_monitor.check_all_compliance_rules()

            if violations:
                append_log(f"[COMPLIANCE] {len(violations)} violations detected")

                # Log each violation
                for violation in violations:
                    severity = violation.severity.value
                    append_log(f"[COMPLIANCE {severity.upper()}] {violation.rule_name}: {violation.description}")

                    # Record violation to Supabase
                    self._supabase_record_compliance_event({
                        'event_type': 'VIOLATION',
                        'rule_name': violation.rule_name,
                        'severity': severity,
                        'description': violation.description,
                        'symbol': getattr(violation, 'symbol', None),
                        'position_value': getattr(violation, 'position_value', None),
                        'limit_value': getattr(violation, 'limit_value', None),
                        'actual_value': getattr(violation, 'actual_value', None),
                        'breach_percentage': getattr(violation, 'breach_percentage', None),
                        'was_prevented': getattr(violation, 'was_prevented', False),
                        'action_taken': getattr(violation, 'action_taken', None)
                    })

                # Update status with violations
                write_status({
                    'compliance_violations': [v.to_dict() for v in violations],
                    'compliance_check_time': datetime.now().isoformat(),
                    'compliance_status': 'VIOLATIONS_DETECTED'
                })
            else:
                append_log("[COMPLIANCE] All compliance checks passed")
                write_status({
                    'compliance_violations': [],
                    'compliance_check_time': datetime.now().isoformat(),
                    'compliance_status': 'COMPLIANT'
                })

            # Update task statistics
            self.task_stats['compliance_runs'] += 1

        except Exception as e:
            logger.error(f"Compliance monitoring failed: {e}")
            append_log(f"[COMPLIANCE ERROR] {e}")
            error_msg = str(e)
        finally:
            # Complete Supabase run tracking
            duration_ms = int((time.time() - start_time) * 1000)
            self._supabase_complete_run(run_id, duration_ms, error_msg)

    def market_monitoring_task(self):
        """Market monitoring task - runs continuously."""
        try:
            # Get comprehensive market status
            status = self.market_manager.get_market_status(include_yahoo_api=False)

            # Log phase changes and important market events
            new_phase = MarketPhase(status['market_phase'])
            if self.current_phase != new_phase:
                old_phase = self.current_phase.value if self.current_phase else "unknown"
                append_log(f"[MONITOR] Market phase changed: {old_phase} -> {new_phase.value}")

                # Log specific phase transition messages
                if new_phase == MarketPhase.REGULAR:
                    append_log("[MONITOR] Regular trading hours started - activating trading tasks")
                elif new_phase == MarketPhase.CLOSED:
                    append_log("[MONITOR] Market closed - activating selection tasks")
                elif new_phase == MarketPhase.PRE_MARKET:
                    append_log("[MONITOR] Pre-market trading started")
                elif new_phase == MarketPhase.AFTER_HOURS:
                    append_log("[MONITOR] After-hours trading started")

                # Update current phase
                self.current_phase = new_phase

            # Calculate time to next important phase
            next_open_time = self.market_manager.get_next_market_open()
            next_close_time = self.market_manager.get_next_market_close()

            # Add task statistics and health metrics
            task_health = self._get_task_health_status()

            # Update status with comprehensive monitoring info
            write_status({
                "market_status": status,
                "next_market_open": next_open_time.isoformat(),
                "next_market_close": next_close_time.isoformat(),
                "task_statistics": self.task_stats,
                "task_health": task_health,
                "scheduler_uptime": self._get_scheduler_uptime(),
                "last_phase_check": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Market monitoring task error: {e}")
            append_log(f"[MONITOR] Monitoring error: {e}")

    def _get_task_health_status(self) -> Dict[str, Any]:
        """Get health status of all registered tasks."""
        try:
            health_status = {
                'healthy_tasks': 0,
                'error_tasks': 0,
                'total_tasks': 0,
                'task_details': {}
            }

            for task_type, task_list in self.tasks.items():
                for task in task_list:
                    task_name = task['name']
                    error_count = task.get('error_count', 0)
                    last_run = task.get('last_run')

                    # Determine task health
                    is_healthy = error_count == 0 and last_run is not None
                    if is_healthy:
                        health_status['healthy_tasks'] += 1
                    else:
                        health_status['error_tasks'] += 1

                    health_status['total_tasks'] += 1

                    # Add task details
                    health_status['task_details'][task_name] = {
                        'type': task_type,
                        'error_count': error_count,
                        'last_run': last_run.isoformat() if last_run else None,
                        'last_execution_time': task.get('last_execution_time', 0),
                        'interval': task.get('interval', 0),
                        'is_healthy': is_healthy
                    }

            return health_status

        except Exception as e:
            logger.error(f"Error getting task health status: {e}")
            return {'error': str(e)}

    def _get_scheduler_uptime(self) -> str:
        """Get scheduler uptime information."""
        try:
            # This would need to track start time - simplified for now
            return "Running"
        except Exception:
            return "Unknown"

    # ============================================
    # SUPABASE INTEGRATION HELPERS
    # ============================================
    def _supabase_start_run(self, run_type: str, metadata: Dict = None) -> Optional[str]:
        """Start tracking a task run in Supabase."""
        if not SUPABASE_AVAILABLE:
            return None
        try:
            run_id = supabase_client.insert_run(run_type, metadata)
            return run_id
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to start run: {e}")
            return None

    def _supabase_complete_run(self, run_id: str, duration_ms: int, error: str = None):
        """Complete a task run in Supabase."""
        if not SUPABASE_AVAILABLE or not run_id:
            return
        try:
            supabase_client.complete_run(run_id, duration_ms, error)
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to complete run: {e}")

    def _supabase_record_positions(self, positions: List[Dict]):
        """Snapshot positions to Supabase."""
        if not SUPABASE_AVAILABLE or not positions:
            return
        try:
            supabase_client.snapshot_positions(positions)
            logger.debug(f"[SUPABASE] Recorded {len(positions)} positions")
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record positions: {e}")

    def _supabase_record_metrics(self, metrics: Dict):
        """Record real-time metrics to Supabase."""
        if not SUPABASE_AVAILABLE or not metrics:
            return
        try:
            supabase_client.insert_metrics(metrics)
            logger.debug("[SUPABASE] Recorded metrics snapshot")
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record metrics: {e}")

    def _supabase_record_trade_signal(self, signal: Dict, run_id: str = None) -> Optional[str]:
        """Record a trade signal to Supabase."""
        if not SUPABASE_AVAILABLE:
            return None
        try:
            signal_data = {
                'run_id': run_id,
                'symbol': signal.get('symbol'),
                'signal_type': signal.get('action', signal.get('signal_type', 'HOLD')),
                'strategy_name': signal.get('strategy', signal.get('strategy_name', 'auto_trading')),
                'score': signal.get('score', signal.get('avg_score', 0)),
                'component_scores': signal.get('component_scores', {}),
                'price_at_signal': signal.get('price', signal.get('market_price', 0)),
                'volume_at_signal': signal.get('volume'),
                'market_cap': signal.get('market_cap'),
                'sector': signal.get('sector'),
                'reasoning': signal.get('reasoning', signal.get('combined_reasoning', '')),
                'was_executed': signal.get('was_executed', False)
            }
            signal_id = supabase_client.insert_trade_signal(signal_data)
            return signal_id
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record trade signal: {e}")
            return None

    def _supabase_record_order(self, order: Dict, run_id: str = None) -> Optional[str]:
        """Record an order to Supabase."""
        if not SUPABASE_AVAILABLE:
            return None
        try:
            order_data = {
                'external_id': order.get('tiger_order_id', order.get('external_id')),
                'symbol': order.get('symbol'),
                'side': order.get('action', order.get('side', 'BUY')),
                'order_type': order.get('type', 'MARKET'),
                'quantity': order.get('qty', order.get('quantity', 0)),
                'price': order.get('price', order.get('limit_price')),
                'status': order.get('status', 'PENDING')
            }
            order_id = supabase_client.insert_order(order_data, run_id)
            return order_id
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record order: {e}")
            return None

    def _supabase_record_execution_analysis(self, analysis: Dict):
        """Record execution analysis to Supabase."""
        if not SUPABASE_AVAILABLE:
            return
        try:
            supabase_client.insert_execution_analysis(analysis)
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record execution analysis: {e}")

    def _supabase_record_compliance_event(self, event: Dict):
        """Record compliance event to Supabase."""
        if not SUPABASE_AVAILABLE:
            return
        try:
            supabase_client.insert_compliance_event(event)
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record compliance event: {e}")

    def _supabase_record_factor_crowding(self, crowding: Dict):
        """Record factor crowding data to Supabase."""
        if not SUPABASE_AVAILABLE:
            return
        try:
            supabase_client.insert_factor_crowding(crowding)
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record factor crowding: {e}")

    def _supabase_record_selection(self, run_id: str, strategy: str, picks: List[Dict], total: int):
        """Record selection results to Supabase."""
        if not SUPABASE_AVAILABLE:
            return
        try:
            supabase_client.insert_selection(run_id, strategy, picks, total)
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record selection: {e}")

    def _supabase_record_intraday_risk_snapshot(self, snapshot: Dict, run_id: str = None):
        """Record intraday risk snapshot to Supabase for 5-minute level analysis."""
        if not SUPABASE_AVAILABLE:
            return
        try:
            snapshot_data = {
                "run_id": run_id,
                "equity": snapshot.get("equity"),
                "day_start_equity": snapshot.get("day_start_equity"),
                "daily_pnl": snapshot.get("daily_pnl"),
                "daily_loss_pct": snapshot.get("loss_pct"),
                "positions_count": snapshot.get("positions_count"),
                "total_position_value": snapshot.get("total_position_value"),
                "cash_balance": snapshot.get("available_funds"),
                "buying_power": snapshot.get("buying_power"),
                "max_position_weight": snapshot.get("max_position_weight"),
                "es_97_5": snapshot.get("es_97_5"),
                "portfolio_beta": snapshot.get("portfolio_beta"),
                "factor_hhi": snapshot.get("factor_hhi"),
                "daily_costs_total": snapshot.get("daily_costs_total"),
                "daily_cost_pct": snapshot.get("daily_cost_pct"),
                "circuit_breaker_active": snapshot.get("risk_paused", False),
                "halt_reason": snapshot.get("risk_reason"),
            }
            supabase_client.insert_intraday_risk_snapshot(snapshot_data)
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to record intraday risk snapshot: {e}")

    def real_time_monitoring_task(self):
        """Real-time monitoring task - calculates 17 institutional-quality metrics."""
        if not self.market_manager.is_market_active():
            return

        if not self.real_time_monitor:
            logger.warning("[REAL_TIME] Real-Time Monitor not initialized")
            return

        # Start Supabase run tracking
        run_id = self._supabase_start_run('monitoring')
        start_time = time.time()
        error_msg = None

        try:
            # Get current positions from Tiger
            positions = self._get_tiger_positions()

            # Get portfolio returns for ES calculation
            status = read_status()
            portfolio_value = status.get('real_portfolio_value', 500000.0)

            # Simulate returns for now (in production, use actual trading history)
            import numpy as np
            returns = np.random.normal(0.0008, 0.012, 252)  # 252 trading days

            # Build portfolio data structure for monitor
            portfolio_data = {
                "positions": {},
                "cash": status.get('cash_balance', 50000.0),
                "total_value": portfolio_value,
                "returns": returns,
                "factor_exposures": {
                    "momentum": 0.25,
                    "value": 0.15,
                    "quality": 0.20,
                    "volatility": -0.10,
                    "size": 0.05
                }
            }

            # Convert positions to monitor format
            for pos in positions:
                symbol = pos.get('symbol')
                portfolio_data['positions'][symbol] = {
                    'shares': pos.get('quantity', 0),
                    'price': pos.get('market_price', 0),
                    'weight': pos.get('market_value', 0) / portfolio_value if portfolio_value > 0 else 0
                }

            # Calculate all 17 institutional-quality metrics
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            health_metrics = loop.run_until_complete(
                self.real_time_monitor._collect_health_metrics()
            )

            loop.close()

            # Convert metrics to dictionary for status update
            metrics_dict = {
                'timestamp': health_metrics.timestamp.isoformat(),
                'portfolio_es_975': round(health_metrics.portfolio_es_975, 4),
                'current_drawdown': round(health_metrics.current_drawdown, 4),
                'risk_budget_utilization': round(health_metrics.risk_budget_utilization, 4),
                'tail_dependence': round(health_metrics.tail_dependence, 4),
                'daily_transaction_costs': round(health_metrics.daily_transaction_costs, 4),
                'capacity_utilization': round(health_metrics.capacity_utilization, 4),
                'implementation_shortfall': round(health_metrics.implementation_shortfall, 4),
                'factor_hhi': round(health_metrics.factor_hhi, 4),
                'max_correlation': round(health_metrics.max_correlation, 4),
                'crowding_risk_score': round(health_metrics.crowding_risk_score, 4),
                'daily_pnl': round(health_metrics.daily_pnl, 2),
                'sharpe_ratio_ytd': round(health_metrics.sharpe_ratio_ytd, 2),
                'max_drawdown_ytd': round(health_metrics.max_drawdown_ytd, 4),
                'active_positions': health_metrics.active_positions,
                'data_freshness': health_metrics.data_freshness,
                'system_uptime': round(health_metrics.system_uptime, 1)
            }

            # Log key metrics
            append_log(
                f"[REAL_TIME] ES@97.5%: {metrics_dict['portfolio_es_975']:.4f}, "
                f"Sharpe: {metrics_dict['sharpe_ratio_ytd']:.2f}, "
                f"Drawdown: {metrics_dict['current_drawdown']:.2%}, "
                f"HHI: {metrics_dict['factor_hhi']:.3f}"
            )

            # Update status with real-time metrics
            write_status({
                'real_time_metrics': metrics_dict,
                'metrics_timestamp': datetime.now().isoformat(),
                'monitoring_status': 'active'
            })

            # Update task statistics
            self.task_stats['real_time_monitoring_runs'] += 1

            # Record metrics to Supabase
            self._supabase_record_metrics(metrics_dict)

        except Exception as e:
            logger.error(f"Real-time monitoring failed: {e}")
            append_log(f"[REAL_TIME ERROR] {e}")
            error_msg = str(e)
        finally:
            # Complete Supabase run tracking
            duration_ms = int((time.time() - start_time) * 1000)
            self._supabase_complete_run(run_id, duration_ms, error_msg)

    def exception_recovery_task(self):
        """Exception recovery and system health check task."""
        try:
            # Check system health
            if not os.path.exists(os.path.join(BASE, "..", "state", "status.json")):
                logger.warning("Status file missing, recreating...")
                write_status({"bot": "recovering", "recovery_time": datetime.now().isoformat()})

            # Check kill switch file integrity
            kill_path = os.path.join(BASE, "..", "state", "kill.flag")
            if os.path.exists(kill_path):
                # Verify kill switch is still valid (not corrupted)
                try:
                    with open(kill_path, 'r') as f:
                        reason = f.read().strip()
                    append_log(f"[RECOVERY] Kill switch active: {reason}")
                except Exception as e:
                    logger.warning(f"Kill switch file corrupted, removing: {e}")
                    os.remove(kill_path)

            append_log("[RECOVERY] System health check completed")

        except Exception as e:
            logger.error(f"Recovery task failed: {e}")

    def ai_training_task(self):
        """AI model training task - runs daily during market closed."""
        if not self.ai_manager or not self.ai_manager.is_enabled():
            return

        try:
            if not self.ai_manager.should_run_training():
                return

            append_log("[AI_TRAINING] Starting AI model training...")

            # Run AI training
            training_result = self.ai_manager.run_ai_training()

            if training_result.get('success'):
                self.task_stats['ai_training_runs'] += 1
                append_log(f"[AI_TRAINING] Training completed successfully")

                # Update status with AI info
                ai_status = self.ai_manager.get_ai_status()
                write_status({
                    'ai_status': ai_status,
                    'last_ai_training': datetime.now().isoformat()
                })
            else:
                append_log(f"[AI_TRAINING] Training failed: {training_result.get('error')}")

        except Exception as e:
            logger.error(f"AI training task failed: {e}")
            append_log(f"[AI_TRAINING ERROR] {e}")

    def ai_optimization_task(self):
        """AI strategy optimization task - runs every 6 hours during market closed."""
        if not self.ai_manager or not self.ai_manager.is_enabled():
            return

        try:
            if not self.ai_manager.should_run_optimization():
                return

            append_log("[AI_OPTIMIZATION] Starting strategy optimization...")

            # Run AI optimization
            optimization_result = self.ai_manager.run_strategy_optimization()

            if optimization_result.get('success'):
                append_log(f"[AI_OPTIMIZATION] Optimization completed successfully")

                # Update status with optimization results
                write_status({
                    'ai_optimization_status': optimization_result,
                    'last_ai_optimization': datetime.now().isoformat()
                })
            else:
                append_log(f"[AI_OPTIMIZATION] Optimization failed: {optimization_result.get('error')}")

        except Exception as e:
            logger.error(f"AI optimization task failed: {e}")
            append_log(f"[AI_OPTIMIZATION ERROR] {e}")

    def start(self):
        """Start the market-aware scheduler."""
        self.running = True

        # Register default tasks with configurable intervals
        self.register_task('trading', self.real_trading_task, interval=self.trading_interval)
        self.register_task('selection', self.stock_selection_task, interval=self.selection_interval)
        self.register_task('monitoring', self.market_monitoring_task, interval=self.monitoring_interval)
        self.register_task('monitoring', self.exception_recovery_task, interval=300)  # 5 minutes

        if self.intraday_enabled and self.intraday_engine:
            self.register_task('trading', self.intraday_trading_task, interval=self.intraday_interval)
            append_log(f"[INTRADAY] Intraday trading task registered ({self.intraday_interval}s)")

        # Register Real-Time Monitor task if available
        if self.real_time_monitor:
            self.register_task('monitoring', self.real_time_monitoring_task, interval=60)  # Every 60 seconds
            append_log("[REAL_TIME] Real-Time Monitor task registered - 17 metrics tracking enabled")

        # Register Factor Crowding Monitor task if available
        if self.crowding_monitor:
            self.register_task('monitoring', self.factor_crowding_monitoring_task, interval=300)  # Every 5 minutes
            append_log("[CROWDING] Factor Crowding Monitor task registered - HHI, Gini, correlation tracking enabled")

        # Register Compliance Monitor task if available
        if self.compliance_monitor:
            self.register_task('monitoring', self.compliance_monitoring_task, interval=60)  # Every 60 seconds
            append_log("[COMPLIANCE] Compliance Monitor task registered - 8 regulatory rules enforcement enabled")

        # Register AI tasks if available
        if self.ai_manager and self.ai_manager.is_enabled():
            self.register_task('ai_training', self.ai_training_task, interval=3600 * 24)  # Daily
            self.register_task('ai_training', self.ai_optimization_task, interval=3600 * 6)  # Every 6 hours
            append_log("[AI] AI training and optimization tasks registered")

        append_log("Market-aware scheduler started")
        write_status({
            "bot": "running",
            "scheduler_start": datetime.now().isoformat(),
            "market_type": self.market_type
        })

        # Main scheduler loop
        while self.running:
            try:
                status = read_status()
                if status.get("restart_requested"):
                    append_log("[SYSTEM] Restart requested, stopping runner for restart")
                    write_status(
                        {
                            "restart_requested": False,
                            "restart_acknowledged_at": datetime.now().isoformat(),
                            "bot": "restarting",
                        }
                    )
                    break

                if is_killed():
                    append_log("Scheduler paused by kill switch")
                    write_status({"bot": "paused"})
                    time.sleep(5.0)
                    continue

                # Update current market phase
                self.current_phase = self.market_manager.get_market_phase()

                # Run task groups based on market phase
                self.run_task_group('selection', self.current_phase)
                self.run_task_group('trading', self.current_phase)
                self.run_task_group('monitoring', self.current_phase)
                self.run_task_group('ai_training', self.current_phase)  # AI tasks

                # Sleep before next cycle
                time.sleep(10.0)  # Check every 10 seconds

            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {type(e).__name__}: {e}")
                append_log(f"[ERROR] Scheduler error: {e}")
                time.sleep(30.0)  # Wait longer on errors

        append_log("Market-aware scheduler stopped")
        write_status({"bot": "stopped", "stop_time": datetime.now().isoformat()})

    def stop(self):
        """Stop the scheduler."""
        self.running = False

    def _get_stock_universe(self) -> List[str]:
        """
        Get stock universe for selection strategies from all_stock_symbols.csv.

        Returns:
            List of stock symbols to evaluate
        """
        try:
            import csv


            # Path to the stock universe file
            universe_file = os.path.join(os.path.dirname(__file__), "..", "..", "all_stock_symbols.csv")

            if not os.path.exists(universe_file):
                logger.warning(f"Stock universe file not found: {universe_file}")
                return self._get_fallback_universe()

            # Read stock symbols from CSV
            symbols = []
            with open(universe_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('Symbol', '').strip()
                    if symbol and len(symbol) <= 5:  # Filter out invalid symbols
                        symbols.append(symbol)

            # Apply SELECTION_UNIVERSE_SIZE limit
            universe_size = int(os.getenv('SELECTION_UNIVERSE_SIZE', 4000))
            if len(symbols) > universe_size:
                # Take first N symbols (already sorted by file order)
                symbols = symbols[:universe_size]
                logger.info(f"Loaded {len(symbols)} symbols from stock universe file (limited by SELECTION_UNIVERSE_SIZE={universe_size})")
            else:
                logger.info(f"Loaded {len(symbols)} symbols from stock universe file")

            return symbols

        except Exception as e:
            logger.error(f"Error loading stock universe from CSV: {e}")
            return self._get_fallback_universe()

    def _get_fallback_universe(self) -> List[str]:
        """Fallback universe if CSV loading fails."""
        # Demo universe - top liquid US stocks by market cap
        universe = [
            # Technology
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
            "NFLX", "ADBE", "AMD", "INTC", "QCOM", "CSCO", "PYPL", "SNAP", "UBER", "LYFT",

            # Healthcare & Pharmaceuticals
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "ISRG", "DXCM", "MRNA",
            "LLY", "BMY", "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "INCY", "ALXN",

            # Financial Services
            "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW", "BLK",
            "SPGI", "CME", "ICE", "COIN", "PYPL", "SQ", "AFRM", "SOFI", "HOOD", "UPST",

            # Consumer & Retail
            "AMZN", "WMT", "HD", "PG", "KO", "PEP", "NKE", "SBUX", "MCD", "DIS",
            "COST", "TGT", "LOW", "TJX", "BKNG", "ABNB", "EBAY", "ETSY", "W", "CHWY",

            # Industrial & Energy
            "CAT", "BA", "GE", "RTX", "LMT", "HON", "UPS", "FDX", "DEERE", "MMM",
            "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "FANG", "DVN", "OXY", "HAL",

            # Materials & Real Estate
            "LIN", "APD", "ECL", "SHW", "DD", "DOW", "NEM", "FCX", "ALB", "CF",
            "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "MAA", "UDR"
        ]

        # Remove duplicates and return
        return list(set(universe))

    def _combine_strategy_results(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Combine results from multiple selection strategies.

        Args:
            all_results: Dictionary of strategy results

        Returns:
            Combined and ranked list of stock selections
        """
        try:
            # Collect all selections with strategy scores
            combined_selections = {}

            for strategy_name, results in all_results.items():
                if not results or not results.selected_stocks:
                    continue

                for selection in results.selected_stocks:
                    symbol = selection.symbol

                    if symbol not in combined_selections:
                        combined_selections[symbol] = {
                            'symbol': symbol,
                            'strategies': {},
                            'total_score': 0.0,
                            'avg_score': 0.0,
                            'strategy_count': 0,
                            'actions': [],
                            'combined_reasoning': []
                        }

                    # Add strategy result
                    combined_selections[symbol]['strategies'][strategy_name] = {
                        'score': selection.score,
                        'action': selection.action.value,
                        'reasoning': selection.reasoning,
                        'confidence': selection.confidence,
                        'metrics': selection.metrics
                    }

                    combined_selections[symbol]['total_score'] += selection.score
                    combined_selections[symbol]['strategy_count'] += 1
                    combined_selections[symbol]['actions'].append(selection.action.value)
                    combined_selections[symbol]['combined_reasoning'].append(
                        f"{strategy_name}: {selection.reasoning}"
                    )

            # Calculate average scores and final rankings
            final_selections = []
            for symbol, data in combined_selections.items():
                if data['strategy_count'] > 0:
                    data['avg_score'] = data['total_score'] / data['strategy_count']
                    data['combined_reasoning'] = "; ".join(data['combined_reasoning'])

                    # Bonus for multiple strategy agreement
                    if data['strategy_count'] >= 2:
                        data['consensus_bonus'] = min(10.0, data['strategy_count'] * 2.5)
                        data['avg_score'] += data['consensus_bonus']

                    final_selections.append(data)

            # Sort by average score and return top selections
            final_selections.sort(key=lambda x: x['avg_score'], reverse=True)

            # Apply score normalization for better distribution
            # This converts absolute scores (often clustered 87-94) to relative rankings (20-100 range)
            final_selections = self._apply_score_normalization(final_selections)

            # Filter by minimum score threshold and limit to max stocks
            min_score_threshold = float(os.getenv('SELECTION_MIN_SCORE', '80.0'))
            max_stocks = int(os.getenv('SELECTION_RESULT_SIZE', '10'))

            # Apply score filter
            filtered_selections = [s for s in final_selections if s['avg_score'] >= min_score_threshold]

            # Log filtering results
            if len(filtered_selections) < len(final_selections):
                append_log(f"[SELECTION] Score filter: {len(final_selections)} -> {len(filtered_selections)} stocks (threshold: {min_score_threshold})")

            if len(filtered_selections) > max_stocks:
                append_log(f"[SELECTION] Limiting to top {max_stocks} stocks (from {len(filtered_selections)} qualified)")
                filtered_selections = filtered_selections[:max_stocks]
            elif len(filtered_selections) < max_stocks:
                append_log(f"[SELECTION] Selected {len(filtered_selections)} stocks (less than max {max_stocks}, but all qualified)")

            return filtered_selections

        except Exception as e:
            logger.error(f"Error combining strategy results: {e}")
            return []

    def _apply_score_normalization(self, selections: List[Dict[str, Any]],
                                   min_score: float = 20.0,
                                   max_score: float = 100.0) -> List[Dict[str, Any]]:
        """
        Normalize scores to ensure reasonable distribution (20-100 range).

        This converts absolute scores to relative rankings within the selection pool.
        Fixes the issue where most stocks cluster in 87-94 range due to absolute scoring.

        Args:
            selections: List of selection dictionaries with 'avg_score' field
            min_score: Minimum normalized score (default: 20)
            max_score: Maximum normalized score (default: 100)

        Returns:
            Selections with normalized scores
        """
        if not selections or len(selections) <= 1:
            return selections

        try:
            # Extract scores
            scores = [s['avg_score'] for s in selections]
            original_min = min(scores)
            original_max = max(scores)

            logger.info(f"[NORMALIZATION] Original score range: {original_min:.1f} - {original_max:.1f}")

            # If all scores are equal, distribute evenly by rank
            if original_max == original_min or abs(original_max - original_min) < 0.1:
                append_log(f"[NORMALIZATION] Scores are equal ({original_max:.1f}), distributing by rank")
                for i, s in enumerate(selections):
                    s['original_score'] = s['avg_score']
                    # Linear distribution from max_score to min_score based on rank
                    s['avg_score'] = round(max_score - i * (max_score - min_score) / max(len(selections) - 1, 1), 1)
            else:
                # Min-Max normalization to 20-100 range
                for s in selections:
                    normalized = ((s['avg_score'] - original_min) /
                                (original_max - original_min)) * (max_score - min_score) + min_score
                    s['original_score'] = s['avg_score']  # Preserve original score
                    s['avg_score'] = round(normalized, 1)

                append_log(f"[NORMALIZATION] Applied Min-Max normalization: "
                          f"{original_min:.1f}-{original_max:.1f} -> {min_score:.0f}-{max_score:.0f}")

            # Log top 3 score changes
            for i in range(min(3, len(selections))):
                s = selections[i]
                append_log(f"[NORMALIZATION] #{i+1} {s['symbol']}: "
                          f"{s.get('original_score', 0):.1f} -> {s['avg_score']:.1f}")

            return selections

        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
            append_log(f"[NORMALIZATION ERROR] {e}")
            return selections

    def _update_selection_status(self, combined_selections: List[Dict[str, Any]]):
        """
        Update system status with selection results.

        Args:
            combined_selections: Combined selection results
        """
        try:
            import json

            # Prepare selection summary for status
            selection_summary = {
                'total_selections': len(combined_selections),
                'timestamp': datetime.now().isoformat(),
                'top_picks': []
            }

            # Add all selected picks to summary (already filtered by score and max stocks)
            for i, selection in enumerate(combined_selections):
                # Handle both original and improved strategies V2 formats
                symbol = selection.get('symbol', 'UNKNOWN')
                avg_score = selection.get('avg_score', 0.0)
                strategy_count = selection.get('strategy_count', 1)  # Default to 1 if not present

                # Handle both improved V2 format ('action' singular) and original format ('actions' list)
                action = selection.get('action')  # Improved V2 format
                if action:
                    actions = [action]  # Convert to list for consistency
                else:
                    actions = selection.get('actions', ['hold'])  # Original format or default

                reasoning = selection.get('combined_reasoning', selection.get('reasoning', 'No reasoning provided'))

                pick = {
                    'rank': i + 1,
                    'symbol': symbol,
                    'avg_score': round(avg_score, 1),
                    'strategy_count': strategy_count,
                    'dominant_action': max(set(actions), key=actions.count) if actions else 'hold',
                    'reasoning': reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                }
                selection_summary['top_picks'].append(pick)

            # Update system status
            write_status({
                'selection_results': selection_summary,
                'last_selection_run': datetime.now().isoformat(),
                'selection_status': 'completed'
            })

            # Write selection_results.json file (NEW: fixed missing file update)
            try:
                state_dir = os.path.join(os.path.dirname(__file__), '..', 'state')
                os.makedirs(state_dir, exist_ok=True)

                selection_results_file = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy_type': 'multi_strategy_combined',
                    'selected_stocks': [
                        {
                            'symbol': s.get('symbol', 'UNKNOWN'),
                            'score': round(s.get('avg_score', 0.0), 1),
                            'rank': i + 1,
                            'action': s.get('action', max(set(s.get('actions', ['hold'])), key=s.get('actions', ['hold']).count)),
                            'confidence': round(s.get('avg_score', 0.0) / 100.0, 2),
                            'metrics': {
                                'strategy_count': s.get('strategy_count', 1),
                                'reasoning': s.get('combined_reasoning', s.get('reasoning', ''))[:200]
                            },
                            'component_scores': {
                                k: round(v, 1) for k, v in s.get('component_scores', {}).items()
                            } if s.get('component_scores') else None
                        }
                        for i, s in enumerate(combined_selections)
                    ],
                    'selection_count': len(combined_selections)
                }

                selection_file_path = os.path.join(state_dir, 'selection_results.json')
                with open(selection_file_path, 'w', encoding='utf-8') as f:
                    json.dump(selection_results_file, f, indent=2, ensure_ascii=False)

                append_log(f"[SELECTION] Updated selection_results.json with {len(combined_selections[:20])} stocks")

            except Exception as file_error:
                logger.error(f"Failed to write selection_results.json: {file_error}")
                append_log(f"[SELECTION] Warning: Could not update selection_results.json: {file_error}")

            # Log summary
            if combined_selections:
                top_symbols = [s['symbol'] for s in combined_selections[:5]]
                append_log(f"[SELECTION] Top 5 selections: {', '.join(top_symbols)}")
                append_log(f"[SELECTION] Highest scoring: {combined_selections[0]['symbol']} "
                          f"(score: {combined_selections[0]['avg_score']:.1f})")

        except Exception as e:
            logger.error(f"Error updating selection status: {e}")
            append_log(f"[SELECTION] Error updating status: {e}")


def main():
    """Main entry point for the market-aware worker."""
    # Get market type from environment
    market_type = os.getenv('PRIMARY_MARKET', 'US')

    # Create and start scheduler
    scheduler = MarketAwareScheduler(market_type=market_type)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        scheduler.stop()

if __name__ == "__main__":
    main()
