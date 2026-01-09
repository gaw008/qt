#!/usr/bin/env python3
"""
Enhanced FastAPI Backend with Tiger API Integration
Provides comprehensive REST API and WebSocket support for React frontend with real Tiger data
"""

import os
import sys
import time
import json
import asyncio
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from decimal import Decimal

from fastapi import FastAPI, Depends, HTTPException, Header, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from state_manager import read_status, write_status, set_kill, is_killed, read_log_tail, write_daily_report
    from tiger_data_provider_real import real_tiger_provider as tiger_provider
    STATE_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"State manager not available: {e}")
    STATE_MANAGER_AVAILABLE = False

# Import additional API endpoints
try:
    from additional_api_endpoints import router as additional_router
    ADDITIONAL_ENDPOINTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Additional API endpoints not available: {e}")
    ADDITIONAL_ENDPOINTS_AVAILABLE = False

# Import investment-grade monitoring integration
try:
    import sys
    sys.path.append("../../bot")
    from monitoring_dashboard_integration import MonitoringDashboardIntegration
    MONITORING_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Monitoring integration not available: {e}")
    MONITORING_INTEGRATION_AVAILABLE = False

load_dotenv()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if not ADMIN_TOKEN or ADMIN_TOKEN == "wgyjd0508" or len(ADMIN_TOKEN) < 16:
    raise ValueError("CRITICAL SECURITY ERROR: ADMIN_TOKEN must be set to a strong value (16+ characters). Use generate_secure_config.py to create secure configuration.")

# Access password for external authentication
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "WGYJD0508")

# File-based session persistence to survive API restarts
SESSIONS_FILE = os.path.join(os.path.dirname(__file__), '..', 'state', 'sessions.json')

def load_sessions() -> Set[str]:
    """Load sessions from file to survive API restarts"""
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'r') as f:
                data = json.load(f)
                sessions = set(data.get('sessions', []))
                logging.info(f"Loaded {len(sessions)} sessions from file")
                return sessions
    except Exception as e:
        logging.warning(f"Failed to load sessions: {e}")
    return set()

def save_sessions(sessions: Set[str]):
    """Save sessions to file for persistence"""
    try:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, 'w') as f:
            json.dump({'sessions': list(sessions), 'updated': time.time()}, f)
    except Exception as e:
        logging.warning(f"Failed to save sessions: {e}")

# Load existing sessions on startup
valid_sessions: Set[str] = load_sessions()

# Initialize FastAPI app
app = FastAPI(
    title="Quantitative Trading System API",
    description="Complete API for quantitative trading system with real-time capabilities and Tiger integration",
    version="2.1.0"
)

# Import centralized configuration
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "bot"))
    from config_manager import server_config, path_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Configuration manager not available: {e}")
    CONFIG_AVAILABLE = False

# CORS middleware for React frontend
if CONFIG_AVAILABLE:
    cors_origins = server_config.cors_origins
else:
    cors_origins = ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004", "http://localhost:3005", "http://localhost:5173", "http://localhost:8080"]

# Always include production domains for Vultr deployment
production_origins = [
    "https://trade.wgyjdaiassistant.cc",
    "https://api.wgyjdaiassistant.cc",
    "https://dash.wgyjdaiassistant.cc"
]
cors_origins = list(set(cors_origins + production_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle OPTIONS preflight requests before auth check
@app.middleware("http")
async def cors_preflight_handler(request: Request, call_next):
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "*")
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    return await call_next(request)

# Include additional API endpoints
if ADDITIONAL_ENDPOINTS_AVAILABLE:
    app.include_router(additional_router, tags=["AI & System Extensions"])
    logging.info("Additional AI and system endpoints included")

# Include backtesting report endpoints
try:
    from backtesting_api_endpoints import router as backtesting_router
    app.include_router(backtesting_router, tags=["Backtesting Reports"])
    logging.info("Backtesting report endpoints included")
except ImportError as e:
    logging.warning(f"Backtesting report endpoints not available: {e}")

# Initialize investment-grade monitoring integration
monitoring_integration = None
if MONITORING_INTEGRATION_AVAILABLE:
    try:
        monitoring_integration = MonitoringDashboardIntegration()
        app.include_router(monitoring_integration.get_router())
        logging.info("Investment-grade monitoring integration included")
    except Exception as e:
        logging.error(f"Failed to initialize monitoring integration: {e}")
        MONITORING_INTEGRATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

INTRADAY_CONFIG_DEFAULTS = {
    "signal_period": "5min",
    "lookback_bars": 120,
    "fast_ema": 9,
    "slow_ema": 21,
    "atr_period": 14,
    "trail_atr": 2.5,
    "hard_stop_atr": 3.0,
    "momentum_lookback": 6,
    "min_volume_ratio": 1.0,
    "entry_score_threshold": 0.6,
    "weight_power": 1.4,
    "max_positions": 10,
    "max_position_percent": 0.12,
    "min_trade_value": 200,
    "min_data_coverage": 0.6,
    "cooldown_seconds": 600,
    "buy_price_buffer_pct": 0.005,
    "commission_per_share": 0.005,
    "min_commission": 1.0,
    "fee_per_order": 0.0,
    "slippage_bps": 5.0,
    "max_daily_cost_pct": 0.003,
    "max_daily_loss_pct": 0.02,
    "open_buffer_minutes": 5,
}
INTRADAY_ALLOWED_PERIODS = {"1min", "5min", "15min", "30min", "60min"}
INTRADAY_VALID_RANGES = {
    "lookback_bars": (30, 1000),
    "fast_ema": (3, 50),
    "slow_ema": (5, 200),
    "atr_period": (5, 50),
    "trail_atr": (0.5, 10.0),
    "hard_stop_atr": (0.5, 10.0),
    "momentum_lookback": (2, 50),
    "min_volume_ratio": (0.0, 5.0),
    "entry_score_threshold": (0.0, 1.0),
    "weight_power": (0.5, 3.0),
    "max_positions": (1, 50),
    "max_position_percent": (0.01, 0.5),
    "min_trade_value": (0.0, 10000.0),
    "min_data_coverage": (0.0, 1.0),
    "cooldown_seconds": (0, 7200),
    "buy_price_buffer_pct": (0.0, 0.05),
    "commission_per_share": (0.0, 0.1),
    "min_commission": (0.0, 20.0),
    "fee_per_order": (0.0, 10.0),
    "slippage_bps": (0.0, 50.0),
    "max_daily_cost_pct": (0.0, 0.02),
    "max_daily_loss_pct": (0.005, 0.1),
    "open_buffer_minutes": (0, 60),
}

def _intraday_config_path() -> str:
    if CONFIG_AVAILABLE:
        return os.path.join(path_config.base_dir, "config", "intraday_strategy.json")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "intraday_strategy.json"))

def _load_intraday_config() -> Dict[str, Any]:
    path = _intraday_config_path()
    config = INTRADAY_CONFIG_DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config.update(data)
        except Exception as exc:
            logger.warning(f"Failed to read intraday config: {exc}")
    return config

def _validate_intraday_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    unknown = [k for k in payload.keys() if k not in INTRADAY_CONFIG_DEFAULTS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown keys: {', '.join(unknown)}")

    updates: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        default_value = INTRADAY_CONFIG_DEFAULTS.get(key)
        if key == "signal_period":
            if str(value) not in INTRADAY_ALLOWED_PERIODS:
                raise HTTPException(status_code=400, detail="signal_period must be one of 1min/5min/15min/30min/60min")
            updates[key] = str(value)
            continue

        if isinstance(default_value, int) and not isinstance(default_value, bool):
            cast_value = int(value)
        else:
            cast_value = float(value)

        min_val, max_val = INTRADAY_VALID_RANGES.get(key, (None, None))
        if min_val is not None and cast_value < min_val:
            raise HTTPException(status_code=400, detail=f"{key} must be >= {min_val}")
        if max_val is not None and cast_value > max_val:
            raise HTTPException(status_code=400, detail=f"{key} must be <= {max_val}")
        updates[key] = cast_value

    candidate = INTRADAY_CONFIG_DEFAULTS.copy()
    candidate.update(_load_intraday_config())
    candidate.update(updates)
    if candidate["fast_ema"] >= candidate["slow_ema"]:
        raise HTTPException(status_code=400, detail="fast_ema must be < slow_ema")
    if candidate["hard_stop_atr"] < candidate["trail_atr"]:
        raise HTTPException(status_code=400, detail="hard_stop_atr must be >= trail_atr")

    return candidate

# Pydantic models for API
class ApiResponse(BaseModel):
    success: bool
    data: Any = None
    message: Optional[str] = None
    error: Optional[str] = None

class IntradayConfigUpdate(BaseModel):
    signal_period: Optional[str] = None
    lookback_bars: Optional[int] = None
    fast_ema: Optional[int] = None
    slow_ema: Optional[int] = None
    atr_period: Optional[int] = None
    trail_atr: Optional[float] = None
    hard_stop_atr: Optional[float] = None
    momentum_lookback: Optional[int] = None
    min_volume_ratio: Optional[float] = None
    entry_score_threshold: Optional[float] = None
    weight_power: Optional[float] = None
    max_positions: Optional[int] = None
    max_position_percent: Optional[float] = None
    min_trade_value: Optional[float] = None
    min_data_coverage: Optional[float] = None
    cooldown_seconds: Optional[int] = None
    buy_price_buffer_pct: Optional[float] = None
    commission_per_share: Optional[float] = None
    min_commission: Optional[float] = None
    fee_per_order: Optional[float] = None
    slippage_bps: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    open_buffer_minutes: Optional[int] = None
    max_daily_cost_pct: Optional[float] = None

class StrategyProfileSwitch(BaseModel):
    profile_id: str

class RestartRunnerRequest(BaseModel):
    reason: Optional[str] = None

def _strategy_profiles_path() -> str:
    if CONFIG_AVAILABLE:
        return os.path.join(path_config.base_dir, "config", "strategy_profiles.json")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "strategy_profiles.json"))

def _active_strategy_path() -> str:
    if CONFIG_AVAILABLE:
        return os.path.join(path_config.base_dir, "config", "active_strategy.json")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "active_strategy.json"))

def _load_strategy_profiles() -> Dict[str, Any]:
    path = _strategy_profiles_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"Failed to read strategy profiles: {exc}")
    return {"profiles": {}}

def _load_active_strategy() -> str:
    path = _active_strategy_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("active_profile"):
                return str(data["active_profile"])
        except Exception as exc:
            logger.warning(f"Failed to read active strategy: {exc}")
    return ""

def _write_active_strategy(profile_id: str):
    path = _active_strategy_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"active_profile": profile_id, "updated_at": datetime.now().isoformat()}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def _env_path() -> str:
    if CONFIG_AVAILABLE:
        return os.path.join(path_config.base_dir, ".env")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def _update_env_file(path: str, updates: Dict[str, str]):
    if not updates:
        return
    lines = []
    found = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in line:
                    lines.append(line.rstrip("\n"))
                    continue
                key, _ = line.split("=", 1)
                key = key.strip()
                if key in updates:
                    lines.append(f"{key}={updates[key]}")
                    found.add(key)
                else:
                    lines.append(line.rstrip("\n"))
    for key, value in updates.items():
        if key not in found:
            lines.append(f"{key}={value}")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp_path, path)

class Asset(BaseModel):
    symbol: str
    name: str
    type: str = "stock"
    sector: Optional[str] = None
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    last_update: str

class Position(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    realized_pnl: Optional[float] = None
    entry_time: str
    last_update: str

class Order(BaseModel):
    id: str
    symbol: str
    side: str
    type: str
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str
    filled_quantity: Optional[int] = None
    avg_fill_price: Optional[float] = None
    created_at: str
    updated_at: str

class OrderRequest(BaseModel):
    symbol: str
    side: str
    type: str
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None

class LoginRequest(BaseModel):
    password: str

class PortfolioSummary(BaseModel):
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    positions_count: int
    cash_balance: float
    available_funds: float = 0.0  # Actual available funds from Tiger API segments (cash - accrued_cash)
    buying_power: float
    margin_used: float
    risk_metrics: Dict[str, float]

class Alert(BaseModel):
    id: str
    timestamp: str
    severity: str
    type: str
    status: str
    title: str
    message: str
    symbol: Optional[str] = None
    price_change: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

class MarketState(BaseModel):
    status: str
    market_trend: Optional[float] = None
    volatility: Optional[float] = None
    volume_ratio: Optional[float] = None
    fear_greed_index: Optional[int] = None
    next_open: Optional[str] = None
    regime: Optional[str] = None
    risk_level: Optional[str] = None

class HeatmapData(BaseModel):
    symbol: str
    name: str
    sector: str
    change_percent_1d: float
    change_percent_1w: float
    change_percent_1m: float
    volume_change: float
    market_cap: float
    price: float

# Authentication dependency with enhanced security
# Supports both Session token (for external access) and Bearer token (for admin)
def auth(authorization: str = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check Session token (for external access via login)
    if authorization.startswith("Session "):
        session_token = authorization.split(" ", 1)[1]
        if session_token in valid_sessions:
            return session_token
        raise HTTPException(status_code=403, detail="Invalid or expired session")

    # Check Bearer token (for admin access)
    if authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        if token == ADMIN_TOKEN:
            return token
        # Log authentication failure for security monitoring
        logger.warning(f"Authentication failed from {authorization}")
        raise HTTPException(status_code=403, detail="Invalid authentication token")

    raise HTTPException(status_code=401, detail="Invalid authorization format")

# Health and system endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}

# Authentication endpoints for external access
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Authenticate with password and get session token"""
    if request.password == ACCESS_PASSWORD:
        # Generate secure session token
        session_token = secrets.token_urlsafe(32)
        valid_sessions.add(session_token)
        save_sessions(valid_sessions)  # Persist sessions to survive restarts
        logger.info(f"New session created, total active sessions: {len(valid_sessions)}")
        return {"success": True, "session_token": session_token}
    else:
        logger.warning("Failed login attempt with invalid password")
        raise HTTPException(status_code=401, detail="Invalid password")

@app.post("/api/auth/logout")
async def logout(authorization: str = Header(default=None)):
    """Logout and invalidate session"""
    if authorization and authorization.startswith("Session "):
        token = authorization.split(" ", 1)[1]
        valid_sessions.discard(token)
        save_sessions(valid_sessions)  # Persist sessions to survive restarts
        logger.info(f"Session invalidated, remaining sessions: {len(valid_sessions)}")
    return {"success": True}

@app.get("/api/auth/verify")
async def verify_session(authorization: str = Header(default=None)):
    """Verify if session is valid"""
    if authorization and authorization.startswith("Session "):
        token = authorization.split(" ", 1)[1]
        if token in valid_sessions:
            return {"valid": True}
    return {"valid": False}

@app.get("/api/system/status")
async def system_status(_=Depends(auth)):
    if STATE_MANAGER_AVAILABLE:
        status = read_status()
        status["tiger_available"] = tiger_provider.is_available()
        return ApiResponse(success=True, data=status)
    return ApiResponse(success=False, error="State manager not available")

# Intraday strategy config endpoints
@app.get("/api/strategy/intraday")
async def get_intraday_strategy_config(_=Depends(auth)):
    config = _load_intraday_config()
    return ApiResponse(
        success=True,
        data={
            "config": config,
            "restart_required": True,
            "config_path": _intraday_config_path(),
        },
    )

@app.put("/api/strategy/intraday")
async def update_intraday_strategy_config(payload: IntradayConfigUpdate, _=Depends(auth)):
    updates = payload.dict(exclude_unset=True)
    merged = _validate_intraday_update(updates)
    config_path = _intraday_config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    tmp_path = f"{config_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, config_path)

    return ApiResponse(
        success=True,
        data={
            "config": merged,
            "restart_required": True,
            "config_path": config_path,
        },
        message="Config saved. Restart runner to apply changes.",
    )

# Strategy profile endpoints
@app.get("/api/strategy/profiles")
async def get_strategy_profiles(_=Depends(auth)):
    profiles_data = _load_strategy_profiles()
    profiles = profiles_data.get("profiles", {})
    active_profile = _load_active_strategy()
    return ApiResponse(
        success=True,
        data={
            "profiles": profiles,
            "active_profile": active_profile,
        },
    )

@app.put("/api/strategy/active")
async def set_active_strategy(payload: StrategyProfileSwitch, _=Depends(auth)):
    profiles_data = _load_strategy_profiles()
    profiles = profiles_data.get("profiles", {})
    profile_id = payload.profile_id
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Unknown strategy profile")

    env_overrides = profiles[profile_id].get("env_overrides", {})
    env_updates = {k: str(v) for k, v in env_overrides.items()}
    _update_env_file(_env_path(), env_updates)
    _write_active_strategy(profile_id)
    if STATE_MANAGER_AVAILABLE:
        write_status(
            {
                "active_strategy": profile_id,
                "strategy_switch_at": datetime.now().isoformat(),
            }
        )

    return ApiResponse(
        success=True,
        data={
            "active_profile": profile_id,
            "restart_required": True,
        },
        message="Strategy profile applied. Restart runner to apply changes.",
    )

@app.post("/api/runner/restart")
async def request_runner_restart(payload: RestartRunnerRequest, _=Depends(auth)):
    if not STATE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="State manager not available")
    reason = payload.reason or "manual"
    write_status(
        {
            "restart_requested": True,
            "restart_reason": reason,
            "restart_requested_at": datetime.now().isoformat(),
        }
    )
    return ApiResponse(success=True, data={"restart_requested": True, "reason": reason})

# Market data endpoints
@app.get("/api/markets/assets")
async def get_assets(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    asset_type: Optional[str] = Query(None),
    _=Depends(auth)
):
    try:
        assets_data = await tiger_provider.get_assets(limit, offset, asset_type)

        # Convert to Pydantic models
        assets = []
        for asset_dict in assets_data:
            assets.append(Asset(**asset_dict))

        return ApiResponse(success=True, data=assets)
    except Exception as e:
        logger.error(f"Failed to fetch assets: {e}")
        return ApiResponse(success=False, error=str(e))

@app.get("/api/markets/heatmap")
async def get_heatmap_data(sector: Optional[str] = Query(None), _=Depends(auth)):
    try:
        # For heatmap, we'll use a simplified version for now
        heatmap_data = []
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

        for i, symbol in enumerate(symbols):
            heatmap_data.append(HeatmapData(
                symbol=symbol,
                name=f"{symbol} Inc.",
                sector="Technology",
                change_percent_1d=5.0 - i * 1.5,
                change_percent_1w=10.0 - i * 2.0,
                change_percent_1m=15.0 - i * 3.0,
                volume_change=1.2 + i * 0.1,
                market_cap=1000000000000.0 - i * 100000000000,
                price=150.0 + i * 20
            ))

        return ApiResponse(success=True, data=heatmap_data)
    except Exception as e:
        logger.error(f"Failed to fetch heatmap data: {e}")
        return ApiResponse(success=False, error=str(e))

@app.get("/api/market-state")
async def get_market_state(_=Depends(auth)):
    try:
        market_state_data = await tiger_provider.get_market_state()
        market_state = MarketState(**market_state_data)
        return ApiResponse(success=True, data=market_state)
    except Exception as e:
        logger.error(f"Failed to fetch market state: {e}")
        return ApiResponse(success=False, error=str(e))

# Portfolio endpoints
@app.get("/api/positions")
async def get_positions(_=Depends(auth)):
    try:
        positions_data = await tiger_provider.get_positions()

        # Convert to Pydantic models
        positions = []
        for position_dict in positions_data:
            positions.append(Position(**position_dict))

        return ApiResponse(success=True, data=positions)
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return ApiResponse(success=False, error=str(e))

@app.get("/api/portfolio/summary")
async def get_portfolio_summary(_=Depends(auth)):
    try:
        portfolio_data = await tiger_provider.get_portfolio_summary()
        logger.info(f"[API] Raw portfolio_data available_funds: {portfolio_data.get('available_funds', 'MISSING')}")
        portfolio = PortfolioSummary(**portfolio_data)
        logger.info(f"[API] Pydantic model available_funds: {portfolio.available_funds}")
        return ApiResponse(success=True, data=portfolio)
    except Exception as e:
        logger.error(f"Failed to fetch portfolio summary: {e}")
        return ApiResponse(success=False, error=str(e))

# Trading endpoints
@app.get("/api/orders")
async def get_orders(
    status: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    _=Depends(auth)
):
    try:
        orders_data = await tiger_provider.get_orders(status, symbol, limit)

        # Convert to Pydantic models
        orders = []
        for order_dict in orders_data:
            orders.append(Order(**order_dict))

        return ApiResponse(success=True, data=orders)
    except Exception as e:
        logger.error(f"Failed to fetch orders: {e}")
        return ApiResponse(success=False, error=str(e))

@app.post("/api/orders")
async def create_order(order_request: OrderRequest, _=Depends(auth)):
    try:
        order_data = await tiger_provider.create_order(order_request.dict())
        order = Order(**order_data)

        # Broadcast order update via WebSocket
        await manager.broadcast(json.dumps({
            "type": "order_update",
            "data": order.dict(),
            "timestamp": datetime.now().isoformat()
        }))

        return ApiResponse(success=True, data=order)
    except Exception as e:
        logger.error(f"Failed to create order: {e}")
        return ApiResponse(success=False, error=str(e))

# Risk endpoints
@app.get("/api/risk/metrics")
async def get_risk_metrics(_=Depends(auth)):
    try:
        # Risk metrics calculation - for now using mock data
        risk_metrics = {
            "var_1d": -2500.0,
            "var_5d": -8000.0,
            "expected_shortfall": -4000.0,
            "portfolio_beta": 1.2,
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.1,
            "max_drawdown": -0.08,
            "volatility": 0.16,
            "concentration_risk": 0.25,
            "sector_allocation": {
                "Technology": 0.6,
                "Healthcare": 0.2,
                "Finance": 0.15,
                "Energy": 0.05
            }
        }

        return ApiResponse(success=True, data=risk_metrics)
    except Exception as e:
        logger.error(f"Failed to fetch risk metrics: {e}")
        return ApiResponse(success=False, error=str(e))

# Alert endpoints

@app.get("/api/metrics/realtime")
async def get_realtime_metrics(_=Depends(auth)):
    """Get real-time institutional-quality metrics (17 metrics)"""
    if STATE_MANAGER_AVAILABLE:
        status = read_status()
        real_time_metrics = status.get('real_time_metrics', {})
        
        # Add descriptive information
        metrics_info = {
            'metrics': real_time_metrics,
            'last_update': real_time_metrics.get('timestamp'),
            'monitoring_status': status.get('monitoring_status', 'inactive'),
            'metrics_description': {
                'portfolio_es_975': 'Expected Shortfall at 97.5% confidence (tail risk)',
                'current_drawdown': 'Current portfolio drawdown from peak',
                'risk_budget_utilization': 'Percentage of risk budget utilized',
                'tail_dependence': 'Tail dependence coefficient',
                'daily_transaction_costs': 'Daily transaction costs (bps)',
                'capacity_utilization': 'AUM capacity utilization ratio',
                'implementation_shortfall': 'Implementation shortfall vs benchmark',
                'factor_hhi': 'Factor concentration (Herfindahl-Hirschman Index)',
                'max_correlation': 'Maximum pairwise factor correlation',
                'crowding_risk_score': 'Overall factor crowding risk score',
                'daily_pnl': 'Daily profit and loss',
                'sharpe_ratio_ytd': 'Year-to-date Sharpe ratio',
                'max_drawdown_ytd': 'Maximum drawdown year-to-date',
                'active_positions': 'Number of active positions',
                'data_freshness': 'Data freshness (seconds)',
                'system_uptime': 'System uptime (hours)'
            }
        }
        
        return ApiResponse(success=True, data=metrics_info)
    return ApiResponse(success=False, error="State manager not available")

@app.get("/api/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    _=Depends(auth)
):
    try:
        alerts_data = await tiger_provider.get_alerts(severity, alert_type, limit)

        # Convert to Pydantic models
        alerts = []
        for alert_dict in alerts_data:
            alerts.append(Alert(**alert_dict))

        return ApiResponse(success=True, data=alerts)
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
        return ApiResponse(success=False, error=str(e))

# Stock selection endpoints
@app.get("/api/stocks/screener")
async def get_stock_selection(
    strategy: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    _=Depends(auth)
):
    """Get current stock selection results from the screener system"""
    try:
        import json
        from pathlib import Path

        # Read selection results from state files
        state_dir = Path("../state")
        results_file = state_dir / "selection_results.json"
        streaming_file = state_dir / "streaming_selection_final.json"

        selection_data = {"stocks": [], "timestamp": None, "strategy": "unknown"}

        # Try to read from both possible files
        for file_path in [streaming_file, results_file]:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle different file formats
                    stocks = data.get("selected_stocks", []) or data.get("stocks", [])
                    if stocks:
                        selection_data = {
                            "stocks": stocks[:limit],
                            "timestamp": data.get("timestamp"),
                            "strategy": data.get("strategy_name", data.get("strategy", "unknown")),
                            "total_analyzed": data.get("total_processed", data.get("total_analyzed", 0)),
                            "execution_time": data.get("execution_time", data.get("execution_time_seconds", 0))
                        }
                        break
                except Exception as e:
                    logger.warning(f"Failed to read selection file {file_path}: {e}")
                    continue

        return ApiResponse(success=True, data=selection_data)

    except Exception as e:
        logger.error(f"Failed to fetch stock selection: {e}")
        return ApiResponse(success=False, error=str(e))

@app.get("/api/stocks/recommendations")
async def get_stock_recommendations(_=Depends(auth)):
    """Get trading recommendations based on current stock selection"""
    try:
        # Get current selection
        selection_response = await get_stock_selection()
        if not selection_response.success:
            return selection_response

        stocks = selection_response.data.get("stocks", [])

        # Convert to trading recommendations
        recommendations = []
        for i, stock in enumerate(stocks[:10]):  # Top 10 recommendations
            recommendations.append({
                "symbol": stock.get("symbol"),
                "action": "buy",
                "confidence": stock.get("score", 0.5),
                "target_weight": round(1.0 / min(len(stocks), 10), 4),  # Equal weight
                "reasons": stock.get("reasons", ["quantitative_selection"]),
                "rank": i + 1,
                "score": stock.get("score", 0.5)
            })

        return ApiResponse(success=True, data={
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "strategy": selection_response.data.get("strategy", "unknown"),
            "last_updated": selection_response.data.get("timestamp")
        })

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        return ApiResponse(success=False, error=str(e))

# Legacy endpoints for backward compatibility
@app.get("/status")
async def legacy_status(_=Depends(auth)):
    if STATE_MANAGER_AVAILABLE:
        return read_status()
    return {"error": "State manager not available"}

@app.post("/kill")
async def legacy_kill(body: dict, _=Depends(auth)):
    if STATE_MANAGER_AVAILABLE:
        reason = body.get("reason", "manual")
        set_kill(True, reason)
        return {"ok": True, "killed": True, "reason": reason}
    return {"error": "State manager not available"}

@app.post("/resume")
async def legacy_resume(body: dict, _=Depends(auth)):
    if STATE_MANAGER_AVAILABLE:
        note = body.get("note", None)
        set_kill(False, note)
        return {"ok": True, "killed": False}
    return {"error": "State manager not available"}

@app.get("/logs")
async def legacy_logs(n: int = Query(200, ge=1, le=2000), _=Depends(auth)):
    if STATE_MANAGER_AVAILABLE:
        return {"lines": read_log_tail(n)}
    return {"error": "State manager not available"}

@app.get("/api/runner/logs")
async def get_runner_logs(lines: int = Query(100, ge=1, le=500), _=Depends(auth)):
    """Get runner console logs for frontend display"""
    if STATE_MANAGER_AVAILABLE:
        log_lines = read_log_tail(lines)
        return {"success": True, "data": {"lines": log_lines, "total": len(log_lines)}}
    return {"success": False, "error": "State manager not available"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(5)

            # Send heartbeat
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            }))

            # Send market updates
            if tiger_provider.is_available():
                # Get real data updates
                try:
                    positions_data = await tiger_provider.get_positions()
                    await websocket.send_text(json.dumps({
                        "type": "position_update",
                        "data": positions_data,
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Failed to send position update: {e}")

            # Send mock price updates for demo
            market_update = {
                "type": "price_update",
                "data": {
                    "AAPL": {"price": 155.0 + (time.time() % 10), "change": 2.5},
                    "GOOGL": {"price": 2550.0 + (time.time() % 100), "change": 1.8}
                },
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(market_update))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================
# SUPABASE HISTORY API ENDPOINTS
# ============================================

# Import Supabase client for history queries
try:
    from supabase_client import supabase_client
    SUPABASE_AVAILABLE = True
    logger.info("Supabase client loaded for history API")
except ImportError as e:
    SUPABASE_AVAILABLE = False
    logger.warning(f"Supabase client not available: {e}")

@app.get("/api/history/orders")
async def get_order_history(
    days: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=100, ge=1, le=1000),
    symbol: Optional[str] = None,
    _=Depends(auth)
):
    """Get order history from Supabase (hot data)."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        orders = supabase_client.get_recent_orders(days, limit)

        # Filter by symbol if specified
        if symbol and orders:
            orders = [o for o in orders if o.get('symbol') == symbol.upper()]

        return ApiResponse(success=True, data=orders)
    except Exception as e:
        logger.error(f"Error fetching order history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/positions")
async def get_position_history(
    limit: int = Query(default=100, ge=1, le=1000),
    _=Depends(auth)
):
    """Get position snapshots from Supabase (hot data)."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        positions = supabase_client.get_latest_positions(limit)
        return ApiResponse(success=True, data=positions)
    except Exception as e:
        logger.error(f"Error fetching position history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/performance")
async def get_performance_history(
    days: int = Query(default=30, ge=1, le=365),
    _=Depends(auth)
):
    """Get daily performance history from Supabase."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        performance = supabase_client.get_daily_performance(days)
        return ApiResponse(success=True, data=performance)
    except Exception as e:
        logger.error(f"Error fetching performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/runs")
async def get_run_history(
    days: int = Query(default=7, ge=1, le=30),
    run_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    _=Depends(auth)
):
    """Get task run history from Supabase."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        runs = supabase_client.get_recent_runs(days, run_type, limit)
        return ApiResponse(success=True, data=runs)
    except Exception as e:
        logger.error(f"Error fetching run history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/signals")
async def get_signal_history(
    days: int = Query(default=7, ge=1, le=90),
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    _=Depends(auth)
):
    """Get trade signal history from Supabase."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        signals = supabase_client.get_trade_signals(days, strategy, symbol, limit)
        return ApiResponse(success=True, data=signals)
    except Exception as e:
        logger.error(f"Error fetching signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/compliance")
async def get_compliance_history(
    days: int = Query(default=30, ge=1, le=365),
    severity: Optional[str] = None,
    _=Depends(auth)
):
    """Get compliance event history from Supabase."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        events = supabase_client.get_compliance_events(days, severity)
        return ApiResponse(success=True, data=events)
    except Exception as e:
        logger.error(f"Error fetching compliance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/metrics")
async def get_metrics_history(
    days: int = Query(default=7, ge=1, le=30),
    limit: int = Query(default=100, ge=1, le=500),
    _=Depends(auth)
):
    """Get metrics snapshots from Supabase."""
    if not SUPABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Supabase not available")

    try:
        metrics = supabase_client.get_metrics_history(days, limit)
        return ApiResponse(success=True, data=metrics)
    except Exception as e:
        logger.error(f"Error fetching metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supabase/status")
async def get_supabase_status(_=Depends(auth)):
    """Check Supabase connection status."""
    if not SUPABASE_AVAILABLE:
        return ApiResponse(success=False, message="Supabase client not loaded")

    try:
        # Test connection with simple query
        client = supabase_client.get_client()
        if client:
            return ApiResponse(
                success=True,
                data={
                    "connected": True,
                    "url": supabase_client.url[:30] + "..." if supabase_client.url else None
                }
            )
        return ApiResponse(success=False, message="Supabase client not initialized")
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Quantitative Trading System API...")

    # Initialize Tiger data provider
    success = await tiger_provider.initialize()
    if success:
        logger.info("Tiger API integration initialized successfully")
    else:
        logger.warning("Tiger API not available - using mock data")

    # Start investment-grade monitoring system
    if monitoring_integration:
        try:
            await monitoring_integration.start_background_monitoring()
            logger.info("Investment-grade monitoring system started")
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Quantitative Trading System API...")

    # Stop investment-grade monitoring system
    if monitoring_integration:
        try:
            await monitoring_integration.stop_background_monitoring()
            logger.info("Investment-grade monitoring system stopped")
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {e}")

if __name__ == "__main__":
    import uvicorn

    # Use configuration manager for host and port
    if CONFIG_AVAILABLE:
        host = server_config.api_host
        port = server_config.api_port
    else:
        host = "0.0.0.0"
        port = 8000

    print(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
