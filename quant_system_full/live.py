
# Configure Unicode encoding for Windows console
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import time, argparse, sys, json
from datetime import datetime, time as time_obj, timedelta
import pandas as pd
import yfinance as yf
import pytz # Import the timezone library

# --- Project-specific imports ---
from bot.config import SETTINGS
from bot.tradeup_client import build_clients
from bot.execution import build_order, send_order
from bot.data import fetch_history
from bot.alpha_router import get_alpha_signals
try:
    from bot.portfolio import calculate_atr, get_position_size
except ImportError:
    from bot.portfolio_native import calculate_atr, get_position_size
from scripts.daily_summary import summarize_with_gemini # Import summary function

# --- Cost optimization imports ---
try:
    improvement_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "improvement"))
    if improvement_path not in sys.path:
        sys.path.append(improvement_path)
    from integration.cost_aware_trading_adapter import create_cost_aware_trading_engine
    COST_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Cost optimization not available: {e}")
    COST_OPTIMIZATION_AVAILABLE = False

# --- Market state machine imports ---
try:
    from bot.market_state_machine import create_market_state_machine, MarketState
    MARKET_STATE_AVAILABLE = True
except ImportError as e:
    print(f"Market state machine not available: {e}")
    MARKET_STATE_AVAILABLE = False

# --- Intelligent alert system imports ---
try:
    from bot.intelligent_alert_system import get_alert_system, AlertSeverity, AlertType
    ALERT_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Alert system not available: {e}")
    ALERT_SYSTEM_AVAILABLE = False

# --- AI Training Manager imports ---
try:
    from dashboard.backend.real_ai_training_manager import real_ai_manager
    AI_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"AI training manager not available: {e}")
    AI_TRAINING_AVAILABLE = False

# --- AI Enhanced Selection Strategy imports ---
try:
    from bot.selection_strategies.ai_enhanced_strategy import create_ai_enhanced_strategy
    AI_ENHANCED_SELECTION_AVAILABLE = True
except ImportError as e:
    print(f"AI enhanced selection strategy not available: {e}")
    AI_ENHANCED_SELECTION_AVAILABLE = False

# --- Configuration Manager imports ---
try:
    from bot.config_manager import trading_config, path_config
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Configuration manager not available: {e}")
    CONFIG_MANAGER_AVAILABLE = False

# --- Local State Manager for Dashboard ---
STATE_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "state"))
if STATE_BACKEND not in sys.path:
    sys.path.append(STATE_BACKEND)
try:
    from state_manager import write_status, append_log, is_killed, read_status
except ImportError:
    def write_status(_): pass
    def append_log(_): pass
    def is_killed(): return False
    def read_status(): return {"positions": []}

# --- Dynamic Universe Definition with Caching ---
SECTOR_ETF_MAP = {
    'IT': 'XLK',
    'HEALTHCARE': 'XLV',
    'ENERGY': 'XLE'
}
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.cache'))
os.makedirs(CACHE_DIR, exist_ok=True)

def get_etf_holdings(ticker_symbol: str) -> list[str]:
    """Fetches ETF holdings, using a local cache to avoid rate limiting."""
    cache_file = os.path.join(CACHE_DIR, f"holdings_{ticker_symbol}.json")
    cache_is_valid = False

    if os.path.exists(cache_file):
        if (time.time() - os.path.getmtime(cache_file)) < timedelta(days=1).total_seconds():
            cache_is_valid = True

    if cache_is_valid:
        print(f"Reading holdings for {ticker_symbol} from cache.")
        with open(cache_file, 'r') as f:
            return json.load(f)

    print(f"Fetching holdings for {ticker_symbol} from network...")
    try:
        from curl_cffi import requests
        session = requests.Session(verify=False)
        etf = yf.Ticker(ticker_symbol, session=session)
        holdings_data = etf.info.get('holdings')
        
        if not holdings_data:
            print(f"Warning: Could not fetch holdings for {ticker_symbol}.")
            return []
        
        symbols = [item.get('symbol') for item in holdings_data if item.get('symbol')]
        
        with open(cache_file, 'w') as f:
            json.dump(symbols, f)
            
        return symbols
    except Exception as e:
        print(f"An error occurred while fetching holdings for {ticker_symbol}: {e}")
        return []

# --- Position State Management ---
positions_state = {}

def initialize_positions():
    """Initialize position states from the dashboard's state file."""
    global positions_state
    try:
        status = read_status()
        persisted_positions = status.get('positions', []) if isinstance(status, dict) else []
        for p in persisted_positions:
            symbol = p.get('symbol')
            if symbol:
                positions_state[symbol] = {
                    'qty': int(p.get('qty', 0)),
                    'entry_price': float(p.get('entry_price', 0.0)),
                    'highest_price': float(p.get('highest_price', 0.0)),
                    'trailing_stop': float(p.get('trailing_stop', 0.0))
                }
        append_log(f"Initialized positions: {positions_state}")
    except Exception as e:
        append_log(f"Error initializing positions: {e}")
        positions_state = {}

def update_dashboard_status():
    """Writes the current state to the dashboard."""
    positions_out = []
    for symbol, state in positions_state.items():
        if state['qty'] > 0:
            pos_data = {'symbol': symbol, **state}
            positions_out.append(pos_data)
    
    write_status({
        "bot": "running",
        "paused": is_killed(),
        "positions": positions_out,
        "last_update": datetime.now().isoformat()
    })

def run_intraday_trading_loop(args, universe, quote, trade, cost_engine=None, market_state_machine=None, alert_system=None):
    """The high-frequency loop for trading during market hours."""
    global positions_state
    append_log("Entering intraday trading loop...")

    if cost_engine:
        append_log("[COST_OPT] Cost optimization enabled for this session")

    if alert_system:
        append_log("[ALERT_SYSTEM] Alert monitoring enabled for this session")

    # Market state tracking
    current_market_state = None
    market_params = None

    while True:
        if is_killed():
            print("Bot is paused.")
            time.sleep(args.interval)
            continue

        # Use timezone-aware time for EOD check
        us_eastern_time = datetime.now(pytz.timezone('US/Eastern')).time()
        if us_eastern_time >= time_obj(15, 50):
            append_log("EOD EXIT triggered. Cleaning up positions...")
            for symbol, state in list(positions_state.items()):
                if state['qty'] > 0:
                    if not SETTINGS.dry_run: send_order(trade, build_order(symbol, 'SELL', state['qty']))
                    append_log(f"  - EOD SELL {state['qty']} of {symbol}")
                    del positions_state[symbol]
            update_dashboard_status()
            print("End-of-day exit complete. Exiting intraday loop.")
            return # Exit the intraday loop

        # Fetch data and generate signals
        all_data = {sym: fetch_history(quote, sym, period='1min', limit=200) for sym in universe}
        alpha_signals = get_alpha_signals(all_data)
        signal_map = dict(zip(alpha_signals['symbol'], alpha_signals['signal']))

        # --- AI ENHANCED SIGNALS: Integrate AI recommendations with traditional signals ---
        if AI_TRAINING_AVAILABLE and real_ai_manager and AI_ENHANCED_SELECTION_AVAILABLE:
            try:
                # Get AI recommendations for current universe
                ai_strategy = create_ai_enhanced_strategy()
                # Get AI weight from configuration manager
                if CONFIG_MANAGER_AVAILABLE:
                    ai_trading_weight = trading_config.ai_trading_weight
                else:
                    ai_trading_weight = float(os.getenv("AI_TRADING_WEIGHT", "0.6"))

                ai_selections = ai_strategy.select_stocks(
                    universe=universe,
                    market_data=all_data,
                    limit=min(50, len(universe)),  # Get top AI recommendations
                    ai_weight=ai_trading_weight  # Use configured AI weight for trading
                )

                # Enhance signal map with AI recommendations
                for ai_selection in ai_selections:
                    symbol = ai_selection.get('symbol')
                    ai_action = ai_selection.get('ai_action', 'HOLD')
                    ai_score = ai_selection.get('composite_score', 0.5)

                    if symbol in signal_map:
                        original_signal = signal_map[symbol]

                        # Enhance existing signals with AI recommendations
                        if ai_action == 'BUY' and ai_score > 0.7:
                            signal_map[symbol] = max(original_signal, 1)  # Boost buy signal
                        elif ai_action == 'SELL' and ai_score < 0.3:
                            signal_map[symbol] = min(original_signal, -1)  # Boost sell signal
                        # For HOLD or moderate scores, keep original signal

                append_log(f"[AI_ENHANCED] Enhanced signals for {len(ai_selections)} stocks with AI recommendations")

            except Exception as e:
                append_log(f"[AI_ENHANCED] Error integrating AI signals: {e}")
                # Continue with original signals if AI enhancement fails

        # Update market state machine if available
        position_multiplier = 1.0
        risk_multiplier = 1.0
        if market_state_machine and all_data:
            try:
                current_market_state, market_params = market_state_machine.update_state(all_data)
                position_multiplier = market_params.position_size_multiplier
                risk_multiplier = market_params.risk_threshold_multiplier

                append_log(f"[MARKET_STATE] Current state: {current_market_state.value}, "
                          f"Position multiplier: {position_multiplier:.2f}, "
                          f"Risk multiplier: {risk_multiplier:.2f}")
            except Exception as e:
                append_log(f"[MARKET_STATE] Error updating state: {e}")

        # Monitor system health and evaluate alerts
        if alert_system:
            try:
                # Calculate portfolio metrics for alert evaluation
                total_positions = len(positions_state)
                total_value = sum(state.get('qty', 0) * state.get('entry_price', 0)
                                for state in positions_state.values())

                # Basic risk metrics (simplified)
                portfolio_risk = total_value / 100000.0  # Normalize against 100k portfolio
                max_position_risk = max((state.get('qty', 0) * state.get('entry_price', 0)) / total_value
                                      for state in positions_state.values()) if positions_state else 0

                alert_context = {
                    'total_positions': total_positions,
                    'portfolio_value': total_value,
                    'portfolio_risk': portfolio_risk,
                    'max_position_risk': max_position_risk,
                    'market_state': current_market_state.value if current_market_state else 'unknown',
                    'timestamp': datetime.now().isoformat()
                }

                # Evaluate alert rules
                alert_system.evaluate_rules(alert_context)

                # Check for critical conditions
                if total_positions > 30:
                    alert_system.create_custom_alert(
                        AlertSeverity.WARNING,
                        AlertType.RISK_ALERT,
                        "High Position Count",
                        f"Portfolio has {total_positions} positions, consider reducing exposure"
                    )

                if current_market_state == MarketState.CRISIS_MODE:
                    alert_system.create_custom_alert(
                        AlertSeverity.ERROR,
                        AlertType.MARKET_ALERT,
                        "Crisis Mode Detected",
                        "Market state machine detected crisis conditions - reducing position sizes"
                    )

            except Exception as e:
                append_log(f"[ALERT_SYSTEM] Error evaluating alerts: {e}")

        for symbol in universe:
            df = all_data.get(symbol)
            if df is None or df.empty: continue

            current_price = df['close'].iloc[-1]
            signal = signal_map.get(symbol, 0)
            is_in_position = symbol in positions_state and positions_state[symbol]['qty'] > 0

            if is_in_position:
                state = positions_state[symbol]
                if signal == -1: # Technical Exit
                    append_log(f"TECH EXIT for {symbol} at {current_price}")
                    if not SETTINGS.dry_run: send_order(trade, build_order(symbol, 'SELL', state['qty']))
                    del positions_state[symbol]
                    continue

                if current_price <= state['trailing_stop']: # Trailing Stop Exit
                    append_log(f"STOP EXIT for {symbol} at {current_price} (stop was {state['trailing_stop']})")
                    if not SETTINGS.dry_run: send_order(trade, build_order(symbol, 'SELL', state['qty']))
                    del positions_state[symbol]
                    continue
                
                new_highest_price = max(state['highest_price'], current_price)
                atr = calculate_atr(df)
                new_trailing_stop = new_highest_price - (atr * args.atr_stop_multiplier)
                state['highest_price'] = new_highest_price
                state['trailing_stop'] = max(state['trailing_stop'], new_trailing_stop)

            elif signal == 1: # Buy Signal
                equity = float(trade.get_assets()[0].summary.net_liquidation)
                atr = calculate_atr(df)

                # Apply market state adjustments to risk parameters
                adjusted_risk = args.risk_per_trade * risk_multiplier

                qty_to_trade = get_position_size(equity, current_price, atr,
                                                 risk_per_trade=adjusted_risk,
                                                 atr_multiplier=args.atr_stop_multiplier)

                # Apply position size multiplier from market state
                qty_to_trade = int(qty_to_trade * position_multiplier)

                if qty_to_trade > 0:
                    # Apply cost optimization if available
                    final_qty = qty_to_trade
                    cost_info = ""

                    if cost_engine:
                        try:
                            # Calculate trading cost for this potential trade
                            cost_analysis = cost_engine.calculate_execution_cost(symbol, qty_to_trade, current_price)
                            cost_bps = cost_analysis.get("cost_basis_points", 0.0)

                            # Apply cost optimization logic
                            optimization = cost_engine.optimize_trading_decision(
                                symbol, "buy", qty_to_trade, current_price, 75.0  # Assume medium confidence
                            )

                            if optimization["action"] == "skip":
                                append_log(f"[COST_OPT] SKIPPING {symbol} due to high cost ({cost_bps:.1f} bps)")
                                continue
                            elif optimization["quantity"] != qty_to_trade:
                                final_qty = optimization["quantity"]
                                append_log(f"[COST_OPT] REDUCING {symbol} quantity from {qty_to_trade} to {final_qty} (cost: {cost_bps:.1f} bps)")

                            cost_info = f" [Cost: {cost_bps:.1f} bps]"

                        except Exception as e:
                            append_log(f"[COST_OPT] Error optimizing {symbol}: {e}")

                    # Add market state info to log
                    state_info = f" [State: {current_market_state.value}]" if current_market_state else ""
                    append_log(f"ENTRY for {symbol} at {current_price}, qty {final_qty}{cost_info}{state_info}")

                    if not SETTINGS.dry_run: send_order(trade, build_order(symbol, 'BUY', final_qty))
                    positions_state[symbol] = {
                        'qty': final_qty, 'entry_price': current_price,
                        'highest_price': current_price,
                        'trailing_stop': current_price - (atr * args.atr_stop_multiplier)
                    }

        update_dashboard_status()
        print(f"Intraday loop finished at {datetime.now()}. Waiting for {args.interval} seconds...")
        time.sleep(args.interval)

def main():
    """The main perpetual loop to manage the bot's daily lifecycle."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--sectors', type=str, required=True, help='Comma-separated sectors, e.g., IT,HEALTHCARE')
    ap.add_argument('--interval', type=int, default=60, help='Intraday loop interval in seconds')
    ap.add_argument('--risk-per-trade', type=float, default=0.01)
    ap.add_argument('--atr-stop-multiplier', type=float, default=2.0)
    ap.add_argument('--enable-cost-optimization', action='store_true', help='Enable trading cost optimization')
    args = ap.parse_args()

    # Initialize cost optimization
    cost_engine = None
    if COST_OPTIMIZATION_AVAILABLE and (args.enable_cost_optimization or os.getenv('ENABLE_COST_OPTIMIZATION', 'false').lower() == 'true'):
        try:
            cost_engine = create_cost_aware_trading_engine(
                dry_run=SETTINGS.dry_run,
                enable_cost_optimization=True
            )
            append_log("[COST_OPT] Cost optimization engine initialized")
            print("Cost optimization enabled")
        except Exception as e:
            append_log(f"[COST_OPT] Failed to initialize cost engine: {e}")
            print(f"Cost optimization failed to initialize: {e}")

    # Initialize market state machine
    market_state_machine = None
    if MARKET_STATE_AVAILABLE and os.getenv('ENABLE_REGIME_DETECTION', 'true').lower() == 'true':
        try:
            market_state_machine = create_market_state_machine()
            append_log("[MARKET_STATE] Market state machine initialized")
            print("Market state machine enabled")
        except Exception as e:
            append_log(f"[MARKET_STATE] Failed to initialize state machine: {e}")
            print(f"Market state machine failed to initialize: {e}")

    # Initialize intelligent alert system
    alert_system = None
    if ALERT_SYSTEM_AVAILABLE:
        try:
            alert_system = get_alert_system()
            append_log("[ALERT_SYSTEM] Intelligent alert system initialized")
            print("Intelligent alert system enabled")
        except Exception as e:
            append_log(f"[ALERT_SYSTEM] Failed to initialize alert system: {e}")
            print(f"Alert system failed to initialize: {e}")

    # Initialize system self-healing monitoring
    try:
        from bot.system_self_healing import start_self_healing
        start_self_healing()
        append_log("[SELF_HEALING] System self-healing monitoring started")
        print("System self-healing monitoring enabled")
    except Exception as e:
        append_log(f"[SELF_HEALING] Failed to start self-healing: {e}")
        print(f"Self-healing system failed to start: {e}")

    # Initialize performance optimization
    try:
        from bot.performance_optimizer import get_optimizer, start_performance_monitoring
        performance_optimizer = get_optimizer()
        # Don't start monitoring here as it's already started in the optimizer initialization
        append_log("[PERFORMANCE] Performance optimization system initialized")
        print("Performance optimization system enabled")
    except Exception as e:
        append_log(f"[PERFORMANCE] Failed to initialize performance optimizer: {e}")
        print(f"Performance optimizer failed to initialize: {e}")
        performance_optimizer = None

    day_initialized = False
    universe = []
    quote, trade = build_clients()
    initialize_positions()

    while True:
        # Always use timezone-aware time for decisions
        us_eastern_zone = pytz.timezone('US/Eastern')
        now_eastern = datetime.now(us_eastern_zone)
        
        is_trading_day = now_eastern.weekday() < 5 # Monday to Friday
        market_open = time_obj(9, 30)
        market_close = time_obj(16, 0)

        if not is_trading_day:
            print(f"Not a trading day. Waiting until next trading day. Current time: {now_eastern}")
            time.sleep(3600) # Sleep for an hour on weekends
            continue

        # --- PRE-MARKET: Initialize for the day ---
        if now_eastern.time() < market_open and not day_initialized:
            print(f"Market is closed. Initializing for today at {now_eastern}...")
            current_universe = set()
            selected_sectors = [s.strip().upper() for s in args.sectors.split(',')]
            for sector in selected_sectors:
                ticker = SECTOR_ETF_MAP.get(sector)
                if ticker:
                    holdings = get_etf_holdings(ticker)
                    if holdings:
                        current_universe.update(holdings)
                    time.sleep(2)
            
            if current_universe:
                base_universe = sorted(list(current_universe))

                # --- AI ENHANCED UNIVERSE: Apply AI filtering to base universe ---
                if AI_ENHANCED_SELECTION_AVAILABLE and len(base_universe) > 20:
                    try:
                        print("Applying AI-enhanced stock selection to base universe...")
                        ai_strategy = create_ai_enhanced_strategy()

                        # Get basic market data for initial filtering
                        # Use a smaller sample for pre-market analysis
                        sample_data = {}
                        for sym in base_universe[:100]:  # Limit to first 100 for performance
                            try:
                                data = fetch_history(quote, sym, period='1d', limit=10)
                                if data is not None and not data.empty:
                                    sample_data[sym] = data
                            except:
                                continue

                        # Get AI weight from configuration manager
                        if CONFIG_MANAGER_AVAILABLE:
                            ai_selection_weight = trading_config.ai_selection_weight
                        else:
                            ai_selection_weight = float(os.getenv("AI_SELECTION_WEIGHT", "0.4"))

                        # Get AI-enhanced selections
                        ai_selections = ai_strategy.select_stocks(
                            universe=list(sample_data.keys()),
                            market_data=sample_data,
                            limit=min(50, len(sample_data)),  # Top 50 AI picks
                            ai_weight=ai_selection_weight  # Use configured AI weight for selection
                        )

                        # Create enhanced universe from AI selections + original high-quality stocks
                        ai_symbols = [selection['symbol'] for selection in ai_selections]
                        enhanced_universe = list(set(ai_symbols + base_universe[:20]))  # Best AI picks + top 20 from ETF

                        universe = sorted(enhanced_universe)
                        append_log(f"AI-enhanced universe: {len(universe)} symbols from {len(base_universe)} base universe")
                        print(f"AI-enhanced universe ready: {len(universe)} symbols (from {len(base_universe)} ETF holdings)")

                    except Exception as e:
                        append_log(f"AI universe enhancement failed: {e}. Using base universe.")
                        universe = base_universe
                        print(f"Fallback to base universe: {len(universe)} symbols")
                else:
                    universe = base_universe
                    print(f"Base universe ready: {len(universe)} symbols")

                day_initialized = True
                append_log(f"Universe initialized for {now_eastern.date()}: {len(universe)} symbols.")
            else:
                print("Could not build universe. Will retry in 15 minutes.")
        
        # --- INTRADAY: Run the trading loop ---
        if now_eastern.time() >= market_open and now_eastern.time() < market_close and day_initialized:
            run_intraday_trading_loop(args, universe, quote, trade, cost_engine, market_state_machine, alert_system)
            # After the loop finishes (EOD), mark day as done and proceed to summary
            day_initialized = False 

        # --- POST-MARKET: Generate summary ---
        if now_eastern.time() >= market_close and day_initialized is False:
            # This block should only run once after the trading day is marked as done.
            # We check day_initialized is False to prevent running this repeatedly overnight.
            print(f"Market is closed. Generating daily summary at {now_eastern}...")
            
            # Import dependencies here as this block runs infrequently
            from dotenv import load_dotenv
            from scripts.daily_summary import get_daily_log_content

            dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '.env'))
            load_dotenv(dotenv_path=dotenv_path)
            api_key = os.getenv("GEMINI_API_KEY")

            # Use the function we already built to safely get log content
            log_content = get_daily_log_content()

            if log_content:
                summary = summarize_with_gemini(api_key, log_content)
                print(summary)
            else:
                print("Today's log file not found or is empty. Nothing to summarize.")

            # --- AI TRAINING TRIGGER: Daily model training after market close ---
            if AI_TRAINING_AVAILABLE and real_ai_manager:
                try:
                    # Load configuration for AI training
                    if CONFIG_MANAGER_AVAILABLE:
                        enable_daily_training = trading_config.enable_daily_ai_training
                        training_frequency = trading_config.ai_training_frequency
                    else:
                        enable_daily_training = os.getenv("ENABLE_DAILY_AI_TRAINING", "true").lower() == "true"
                        training_frequency = os.getenv("AI_TRAINING_FREQUENCY", "daily")

                    if enable_daily_training and training_frequency == "daily":
                        print("Starting daily AI model training after market close...")

                        # Check if training is needed (every day or based on schedule)
                        training_progress = real_ai_manager.get_training_progress()
                        current_status = training_progress.get('status', 'idle')

                        if current_status in ['idle', 'stopped', 'completed']:
                            # Start training with configuration
                            if CONFIG_MANAGER_AVAILABLE:
                                training_params = {
                                    'data_source': trading_config.ai_data_source,
                                    'model_type': trading_config.ai_model_type,
                                    'target_metric': trading_config.ai_target_metric,
                                    'trigger': 'daily_post_market'
                                }
                            else:
                                training_params = {
                                    'data_source': os.getenv("AI_DATA_SOURCE", "yahoo_api"),
                                    'model_type': os.getenv("AI_MODEL_TYPE", "lightgbm"),
                                    'target_metric': os.getenv("AI_TARGET_METRIC", "sharpe_ratio"),
                                    'trigger': 'daily_post_market'
                                }

                            success = real_ai_manager.start_training(**training_params)
                            if success:
                                print(f"Daily AI training started successfully with {training_params['model_type']} model.")
                            else:
                                print("Daily AI training failed to start.")
                        else:
                            print(f"AI training already in progress (status: {current_status}). Skipping daily trigger.")
                    else:
                        print(f"Daily AI training disabled or frequency set to {training_frequency}. Skipping trigger.")

                except Exception as e:
                    print(f"Error during daily AI training trigger: {e}")
            else:
                print("AI training manager not available. Skipping daily training.")

            # Mark the day as truly finished, including summary and AI training
            day_initialized = True
            print("Summary and AI training routines finished. Waiting for next trading day.")

        print(f"Outer loop check at {datetime.now()} (Local Time). Waiting for next state...")
        time.sleep(60 * 15) # Check every 15 minutes outside of trading hours

if __name__ == '__main__':
    main()
