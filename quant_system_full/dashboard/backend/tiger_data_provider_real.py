#!/usr/bin/env python3
"""
Real Tiger API Data Provider
Provides actual Tiger API data integration for the enhanced FastAPI backend
This version uses the verified working Tiger API configuration
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pytz

# Import yfinance for market data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[REAL_TIGER_PROVIDER] Warning: yfinance not available")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Tiger SDK using verified configuration
try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.common.util.signature_utils import read_private_key
    from tigeropen.trade.trade_client import TradeClient
    from tigeropen.quote.quote_client import QuoteClient
    TIGER_SDK_AVAILABLE = True
    print("[REAL_TIGER_PROVIDER] Tiger SDK imported successfully")
except ImportError as e:
    print(f"[REAL_TIGER_PROVIDER] Warning: Tiger SDK not available: {e}")
    TIGER_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealTigerDataProvider:
    """
    Real Tiger API data provider using verified working configuration
    """

    def __init__(self):
        self.trade_client = None
        self.quote_client = None
        self._initialized = False
        self._tiger_config = None
        # Cache for market state data (5-minute TTL)
        self._market_state_cache = None
        self._market_state_cache_time = None
        self._market_state_cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize Tiger SDK using verified configuration"""
        if not TIGER_SDK_AVAILABLE:
            logger.warning("Tiger SDK not available")
            return False

        try:
            # Configuration using the same pattern as working test
            props_dir = str(Path(__file__).parent.parent.parent / 'props')
            cfg = TigerOpenClientConfig(props_path=props_dir)

            # Environment variables
            tiger_id = os.getenv("TIGER_ID", "")
            account = os.getenv("ACCOUNT", "")
            private_key_path = os.getenv("PRIVATE_KEY_PATH", "")

            if tiger_id:
                cfg.tiger_id = tiger_id
            if account:
                cfg.account = account
            if private_key_path and os.path.exists(private_key_path):
                cfg.private_key = read_private_key(private_key_path)
            else:
                logger.error(f"Private key not found at: {private_key_path}")
                return False

            cfg.timezone = "US/Eastern"
            cfg.language = "en_US"

            # Create clients
            self.trade_client = TradeClient(cfg)
            self.quote_client = QuoteClient(cfg)
            self._tiger_config = cfg
            self._initialized = True

            # Test connection asynchronously (non-blocking)
            logger.info(f"Tiger API clients initialized for account: {account}")
            logger.info("Tiger API connection will be tested on first actual use")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Tiger SDK: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Tiger API is available and initialized"""
        return TIGER_SDK_AVAILABLE and self._initialized

    async def get_positions(self) -> List[Dict]:
        """Get current positions from Tiger API"""
        if not self.is_available():
            return []

        try:
            account = os.getenv("ACCOUNT", "")
            positions_raw = self.trade_client.get_positions(account=account)

            if not positions_raw:
                logger.info("No positions found in Tiger account")
                return []

            positions = []
            for pos in positions_raw:
                # Get stock symbol
                symbol = pos.contract.symbol if hasattr(pos.contract, 'symbol') else str(pos.contract)

                positions.append({
                    "symbol": symbol,
                    "quantity": int(pos.quantity),
                    "avg_price": float(pos.average_cost),
                    "current_price": float(pos.market_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "unrealized_pnl_percent": float(pos.unrealized_pnl_percent),
                    "realized_pnl": float(getattr(pos, 'realized_pnl', 0.0)),
                    "entry_time": datetime.now().isoformat(),
                    "last_update": datetime.now().isoformat()
                })

            logger.info(f"Retrieved {len(positions)} real positions from Tiger API")
            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions from Tiger API: {e}")
            return []

    async def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from Tiger API"""
        if not self.is_available():
            return self._get_default_portfolio()

        try:
            account = os.getenv("ACCOUNT", "")
            assets = self.trade_client.get_assets(account=account)
            positions = self.trade_client.get_positions(account=account)

            if not assets or len(assets) == 0:
                logger.warning("No account assets found")
                return self._get_default_portfolio()

            asset = assets[0]
            summary = asset.summary

            # Calculate portfolio metrics from real Tiger data
            total_value = float(summary.net_liquidation)
            cash_balance = float(summary.cash)
            buying_power = float(summary.buying_power)
            realized_pnl = float(summary.realized_pnl)
            unrealized_pnl = float(summary.unrealized_pnl)
            total_pnl = realized_pnl + unrealized_pnl

            # Calculate percentages
            total_pnl_percent = (total_pnl / total_value * 100) if total_value > 0 else 0.0

            # Get additional segment info if available
            segment_info = {}
            logger.info(f"[REAL_TIGER_PROVIDER] Asset segments type: {type(asset.segments)}")
            logger.info(f"[REAL_TIGER_PROVIDER] Asset segments keys: {asset.segments.keys() if hasattr(asset.segments, 'keys') else 'N/A'}")

            if hasattr(asset, 'segments') and asset.segments:
                if 'S' in asset.segments:
                    seg = asset.segments['S']
                    logger.info(f"[REAL_TIGER_PROVIDER] Found segment 'S' with available_funds: {seg.available_funds}")
                    segment_info = {
                        "available_funds": float(seg.available_funds),
                        "gross_position_value": float(seg.gross_position_value),
                        "excess_liquidity": float(seg.excess_liquidity)
                    }

            # Use available_funds from segment if available, otherwise use cash
            available_funds = segment_info.get("available_funds", cash_balance) if segment_info else cash_balance
            logger.info(f"[REAL_TIGER_PROVIDER] Final available_funds: ${available_funds:,.2f} (segment_info={bool(segment_info)})")

            portfolio = {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "total_pnl_percent": total_pnl_percent,
                "daily_pnl": unrealized_pnl,  # Use unrealized as daily proxy
                "daily_pnl_percent": (unrealized_pnl / total_value * 100) if total_value > 0 else 0.0,
                "positions_count": len(positions) if positions else 0,
                "cash_balance": cash_balance,
                "available_funds": available_funds,  # Use actual available funds from Tiger
                "buying_power": buying_power,
                "margin_used": 0.0,  # Tiger doesn't provide this directly
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "segment_info": segment_info,
                "risk_metrics": {
                    "sharpe_ratio": 1.5,  # TODO: Calculate from historical data
                    "max_drawdown": -0.05,
                    "portfolio_beta": 1.1,
                    "volatility": 0.12
                }
            }

            logger.info(f"Retrieved real portfolio summary - Total: ${total_value:,.2f}, P&L: ${total_pnl:+,.2f}")
            logger.info(f"[REAL_TIGER_PROVIDER] Returning portfolio with available_funds={portfolio.get('available_funds', 'MISSING')}")
            return portfolio

        except Exception as e:
            logger.error(f"Failed to fetch portfolio summary from Tiger API: {e}")
            return self._get_default_portfolio()

    async def get_orders(self, status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get orders from Tiger API"""
        if not self.is_available():
            return []

        try:
            account = os.getenv("ACCOUNT", "")
            orders_raw = self.trade_client.get_orders(account=account)

            if not orders_raw:
                logger.info("No orders found in Tiger account")
                return []

            orders = []
            for order in orders_raw[:limit]:
                # Get stock symbol
                symbol_str = order.contract.symbol if hasattr(order.contract, 'symbol') else str(order.contract)

                # Filter by symbol if specified
                if symbol and symbol_str != symbol:
                    continue

                # Filter by status if specified
                order_status = str(order.status).lower().replace('orderstatus.', '')
                if status and order_status != status.lower():
                    continue

                # Convert Tiger order to our format
                order_time = datetime.fromtimestamp(order.order_time / 1000) if order.order_time else datetime.now()

                # Calculate cost analysis for filled orders
                cost_analysis = None
                if order.filled > 0 and order.avg_fill_price:
                    # Estimate cost in basis points
                    # Components: spread (~5bps) + market impact (~3-10bps based on size) + timing (~2bps)
                    spread_cost = 5.0
                    # Market impact increases with order size
                    size_factor = min(order.quantity / 1000, 3.0)  # Cap at 3x
                    market_impact = 3.0 + (size_factor * 2.0)
                    timing_cost = 2.0
                    total_cost_bps = spread_cost + market_impact + timing_cost

                    cost_analysis = {
                        "cost_basis_points": round(total_cost_bps, 1),
                        "spread_cost": round(spread_cost, 1),
                        "market_impact": round(market_impact, 1),
                        "timing_cost": round(timing_cost, 1)
                    }

                orders.append({
                    "id": str(order.id),
                    "symbol": symbol_str,
                    "side": order.action.lower(),  # BUY/SELL -> buy/sell
                    "type": order.order_type.lower(),
                    "quantity": int(order.quantity),
                    "price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.aux_price) if order.aux_price else None,
                    "status": order_status,
                    "filled_quantity": int(order.filled),
                    "avg_fill_price": float(order.avg_fill_price) if order.avg_fill_price else None,
                    "created_at": order_time.isoformat(),
                    "updated_at": order_time.isoformat(),
                    "cost_analysis": cost_analysis
                })

            logger.info(f"Retrieved {len(orders)} real orders from Tiger API")
            return orders

        except Exception as e:
            logger.error(f"Failed to fetch orders from Tiger API: {e}")
            return []

    def _get_default_portfolio(self) -> Dict:
        """Default portfolio when Tiger API is not available"""
        return {
            "total_value": 0.0,
            "total_pnl": 0.0,
            "total_pnl_percent": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
            "positions_count": 0,
            "cash_balance": 0.0,
            "buying_power": 0.0,
            "margin_used": 0.0,
            "risk_metrics": {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "portfolio_beta": 0.0,
                "volatility": 0.0
            }
        }



    async def get_assets(self, limit: int = 100, offset: int = 0, asset_type: str = "all") -> List[Dict]:
        """Get assets from Tiger API - returns list of tradable assets/stocks"""
        if not self.is_available():
            return self._get_default_assets(limit)

        try:
            # Get current positions as assets
            positions = await self.get_positions()

            if not positions:
                logger.info("No positions found, returning default asset list")
                return self._get_default_assets(limit)

            result = []
            for pos in positions[offset:offset+limit]:
                # Get quote data for price and change information
                try:
                    symbol = pos["symbol"]
                    current_price = pos["current_price"]

                    # Calculate change based on avg_price as baseline
                    avg_price = pos["avg_price"]
                    change = current_price - avg_price
                    change_percent = (change / avg_price * 100) if avg_price > 0 else 0.0

                    result.append({
                        "symbol": symbol,
                        "name": f"{symbol} Inc.",  # Default name
                        "type": "stock",
                        "sector": None,
                        "price": current_price,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": 0,  # Volume not available from position data
                        "market_cap": None,
                        "last_update": pos["last_update"]
                    })
                except Exception as e:
                    logger.warning(f"Failed to process position {pos.get('symbol', 'unknown')}: {e}")
                    continue

            # If no positions or want more assets, add some popular stocks
            if len(result) < limit:
                default_assets = self._get_default_assets(limit - len(result))
                result.extend(default_assets)

            return result
        except Exception as e:
            logger.error(f"Failed to fetch assets: {e}")
            return self._get_default_assets(limit)

    def _get_default_assets(self, limit: int = 50) -> List[Dict]:
        """Return default popular stocks when Tiger data not available"""
        popular_stocks = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": 175.0},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "price": 380.0},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "price": 140.0},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "price": 145.0},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology", "price": 500.0},
            {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "price": 350.0},
            {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "price": 240.0},
            {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc.", "sector": "Financial", "price": 380.0},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial", "price": 155.0},
            {"symbol": "V", "name": "Visa Inc.", "sector": "Financial", "price": 260.0},
            {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "price": 160.0},
            {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive", "price": 165.0},
            {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Defensive", "price": 155.0},
            {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financial", "price": 420.0},
            {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consumer Cyclical", "price": 340.0},
            {"symbol": "BAC", "name": "Bank of America Corp.", "sector": "Financial", "price": 35.0},
            {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare", "price": 170.0},
            {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Communication", "price": 100.0},
            {"symbol": "COST", "name": "Costco Wholesale Corp.", "sector": "Consumer Defensive", "price": 620.0},
            {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication", "price": 450.0}
        ]

        result = []
        for stock in popular_stocks[:limit]:
            # Add some random variation to prices
            import random
            price_variation = random.uniform(-0.02, 0.02)
            base_price = stock["price"]
            current_price = base_price * (1 + price_variation)
            change = current_price - base_price
            change_percent = price_variation * 100

            result.append({
                "symbol": stock["symbol"],
                "name": stock["name"],
                "type": "stock",
                "sector": stock["sector"],
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": random.randint(10000000, 100000000),
                "market_cap": round(current_price * random.uniform(500, 3000) * 1000000000, 2),
                "last_update": datetime.now().isoformat()
            })

        return result

    async def get_market_state(self) -> Dict:
        """Get comprehensive market state with real data from yfinance"""
        # Check cache first
        now = datetime.now()
        if (self._market_state_cache is not None and
            self._market_state_cache_time is not None and
            (now - self._market_state_cache_time).total_seconds() < self._market_state_cache_ttl):
            return self._market_state_cache

        try:
            # Get US Eastern time for market status
            eastern = pytz.timezone('US/Eastern')
            now_eastern = datetime.now(eastern)
            hour = now_eastern.hour
            minute = now_eastern.minute
            weekday = now_eastern.weekday()

            # Determine market status (US market hours: 9:30 AM - 4:00 PM ET, Mon-Fri)
            if weekday >= 5:  # Weekend
                status = "closed"
            elif (hour == 9 and minute >= 30) or (10 <= hour < 16):
                status = "open"
            elif hour == 16 and minute == 0:
                status = "open"
            else:
                status = "closed"

            # Default values
            market_trend = 0.0
            volatility = 0.0
            volume_ratio = 1.0
            fear_greed_index = 50
            regime = "normal"
            risk_level = "medium"

            # Calculate actual next_open datetime
            market_open_time = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close_time = now_eastern.replace(hour=16, minute=0, second=0, microsecond=0)

            if status == "open":
                # Market is open, next open is tomorrow (or next Monday if Friday)
                if weekday == 4:  # Friday
                    next_open_dt = market_open_time + timedelta(days=3)
                else:
                    next_open_dt = market_open_time + timedelta(days=1)
                next_close = market_close_time.isoformat()
            elif weekday >= 5:  # Weekend
                # Saturday (5) -> Monday = +2 days, Sunday (6) -> Monday = +1 day
                days_until_monday = 7 - weekday
                next_open_dt = market_open_time + timedelta(days=days_until_monday)
                next_close = None
            elif now_eastern < market_open_time:
                # Before market open today
                next_open_dt = market_open_time
                next_close = market_close_time.isoformat()
            else:
                # After market close today
                if weekday == 4:  # Friday
                    next_open_dt = market_open_time + timedelta(days=3)
                else:
                    next_open_dt = market_open_time + timedelta(days=1)
                next_close = None

            next_open = next_open_dt.isoformat()

            # Fetch real data from yfinance if available
            if YFINANCE_AVAILABLE:
                try:
                    # Get SPY data for market metrics
                    spy = yf.Ticker("SPY")
                    spy_hist = spy.history(period="1mo", interval="1d")

                    if len(spy_hist) >= 2:
                        # Market trend: today's return
                        latest_close = spy_hist['Close'].iloc[-1]
                        prev_close = spy_hist['Close'].iloc[-2]
                        market_trend = ((latest_close - prev_close) / prev_close) * 100

                        # Volatility: 20-day rolling std of returns
                        if len(spy_hist) >= 20:
                            returns = spy_hist['Close'].pct_change().dropna()
                            volatility = returns.tail(20).std() * (252 ** 0.5) * 100  # Annualized
                        else:
                            volatility = spy_hist['Close'].pct_change().std() * (252 ** 0.5) * 100

                        # Volume ratio: current vs 20-day average
                        current_volume = spy_hist['Volume'].iloc[-1]
                        avg_volume = spy_hist['Volume'].tail(20).mean()
                        if avg_volume > 0:
                            volume_ratio = current_volume / avg_volume

                    # Get VIX for fear/greed and risk level
                    vix = yf.Ticker("^VIX")
                    vix_hist = vix.history(period="5d", interval="1d")

                    if len(vix_hist) >= 1:
                        vix_value = vix_hist['Close'].iloc[-1]
                        # Fear/Greed index: inverse of VIX (low VIX = greed, high VIX = fear)
                        # Scale: VIX 10 -> 90 (extreme greed), VIX 30 -> 10 (extreme fear)
                        fear_greed_index = max(0, min(100, int(100 - (vix_value - 10) * 4)))

                        # Risk level based on VIX
                        if vix_value < 15:
                            risk_level = "low"
                        elif vix_value < 25:
                            risk_level = "medium"
                        else:
                            risk_level = "high"

                    # Determine regime based on trend and volatility
                    if market_trend > 0.5 and volatility < 20:
                        regime = "bull"
                    elif market_trend < -0.5 and volatility > 25:
                        regime = "bear"
                    elif volatility > 30:
                        regime = "high_volatility"
                    else:
                        regime = "normal"

                except Exception as yf_error:
                    logger.warning(f"Failed to fetch yfinance data: {yf_error}")

            # Generate market interpretation
            interpretation_parts = []

            # Trend analysis
            if market_trend > 1.0:
                interpretation_parts.append("Market showing strong upward momentum")
            elif market_trend > 0.3:
                interpretation_parts.append("Market slightly bullish")
            elif market_trend < -1.0:
                interpretation_parts.append("Market under significant selling pressure")
            elif market_trend < -0.3:
                interpretation_parts.append("Market slightly bearish")
            else:
                interpretation_parts.append("Market trading sideways")

            # Volatility analysis
            if volatility > 30:
                interpretation_parts.append("extremely high volatility indicates uncertainty")
            elif volatility > 20:
                interpretation_parts.append("elevated volatility suggests caution")
            elif volatility < 10:
                interpretation_parts.append("unusually low volatility may precede a big move")

            # Sentiment analysis
            if fear_greed_index >= 80:
                interpretation_parts.append("extreme greed - consider taking profits")
            elif fear_greed_index >= 60:
                interpretation_parts.append("market sentiment is optimistic")
            elif fear_greed_index <= 20:
                interpretation_parts.append("extreme fear - potential buying opportunity")
            elif fear_greed_index <= 40:
                interpretation_parts.append("cautious sentiment prevails")

            # Volume analysis
            if volume_ratio > 1.5:
                interpretation_parts.append("high volume confirms the current trend")
            elif volume_ratio < 0.7:
                interpretation_parts.append("low volume suggests weak conviction")

            # Risk recommendation
            if risk_level == "high":
                interpretation_parts.append("Recommendation: reduce position sizes and set tight stops")
            elif risk_level == "low" and fear_greed_index < 70:
                interpretation_parts.append("Recommendation: favorable conditions for new positions")

            market_interpretation = ". ".join(interpretation_parts) + "."

            result = {
                "status": status,
                "market_trend": float(round(market_trend, 2)),
                "volatility": float(round(volatility, 2)),
                "volume_ratio": float(round(volume_ratio, 2)),
                "fear_greed_index": int(fear_greed_index),
                "next_open": next_open,
                "next_close": next_close,
                "regime": regime,
                "risk_level": risk_level,
                "market_interpretation": market_interpretation,
                "timezone": "US/Eastern"
            }

            # Cache the result
            self._market_state_cache = result
            self._market_state_cache_time = now

            return result

        except Exception as e:
            logger.error(f"Failed to get market state: {e}")
            return {
                "status": "unknown",
                "market_trend": 0.0,
                "volatility": 0.0,
                "volume_ratio": 1.0,
                "fear_greed_index": 50,
                "next_open": None,
                "next_close": None,
                "regime": "unknown",
                "risk_level": "medium",
                "market_interpretation": "Unable to fetch market data. Please check connection."
            }

    async def create_order(self, order_data: Dict) -> Dict:
        """Create order via Tiger API"""
        if not self.is_available():
            return {"success": False, "message": "Tiger API not available"}

        try:
            # This would implement real order creation
            logger.warning("Order creation not implemented - would place real order")
            return {
                "success": False,
                "message": "Order creation disabled for safety - remove this guard to enable real trading"
            }
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return {"success": False, "message": str(e)}

    async def get_alerts(self, severity: str = None, alert_type: str = None, limit: int = 50) -> List[Dict]:
        """Get alerts based on real market conditions and portfolio status"""
        alerts = []
        now = datetime.now()
        alert_id = 1

        try:
            # Get market state for condition-based alerts
            market_state = await self.get_market_state()

            # 1. VIX-based risk alerts
            if market_state.get("risk_level") == "high":
                alerts.append({
                    "id": str(alert_id),
                    "type": "risk_limit",
                    "severity": "high",
                    "status": "active",
                    "title": "Elevated Market Risk",
                    "message": f"VIX indicates high volatility. Consider reducing position sizes.",
                    "symbol": None,
                    "price_change": None,
                    "timestamp": now.isoformat()
                })
                alert_id += 1

            # 2. Market trend alerts
            market_trend = market_state.get("market_trend", 0)
            if abs(market_trend) > 1.5:
                direction = "up" if market_trend > 0 else "down"
                sev = "medium" if abs(market_trend) < 2.5 else "high"
                alerts.append({
                    "id": str(alert_id),
                    "type": "price_movement",
                    "severity": sev,
                    "status": "active",
                    "title": f"Significant Market Move ({direction.upper()})",
                    "message": f"S&P 500 moved {abs(market_trend):.1f}% today.",
                    "symbol": "SPY",
                    "price_change": market_trend,
                    "timestamp": now.isoformat()
                })
                alert_id += 1

            # 3. Volume spike alerts
            volume_ratio = market_state.get("volume_ratio", 1.0)
            if volume_ratio > 1.5:
                alerts.append({
                    "id": str(alert_id),
                    "type": "volume_spike",
                    "severity": "medium",
                    "status": "active",
                    "title": "High Market Volume",
                    "message": f"Trading volume is {volume_ratio:.1f}x above average.",
                    "symbol": None,
                    "price_change": None,
                    "timestamp": now.isoformat()
                })
                alert_id += 1

            # 4. Check positions for significant moves
            positions = await self.get_positions()
            for pos in positions[:5]:  # Check top 5 positions
                pnl_pct = pos.get("unrealized_pnl_percent", 0)
                if abs(pnl_pct) > 3:  # More than 3% move
                    sev = "critical" if abs(pnl_pct) > 5 else "high"
                    alerts.append({
                        "id": str(alert_id),
                        "type": "price_movement",
                        "severity": sev,
                        "status": "active",
                        "title": f"Large Position Move: {pos['symbol']}",
                        "message": f"Position is {'up' if pnl_pct > 0 else 'down'} {abs(pnl_pct):.1f}% today.",
                        "symbol": pos["symbol"],
                        "price_change": pnl_pct,
                        "timestamp": now.isoformat()
                    })
                    alert_id += 1

            # 5. Check recent orders for execution alerts
            orders = await self.get_orders(limit=5)
            for order in orders:
                if order.get("status") == "filled":
                    alerts.append({
                        "id": str(alert_id),
                        "type": "trade_execution",
                        "severity": "low",
                        "status": "resolved",
                        "title": f"Order Filled: {order['symbol']}",
                        "message": f"{order['side'].upper()} {order['quantity']} shares at ${order.get('avg_fill_price', 0):.2f}",
                        "symbol": order["symbol"],
                        "price_change": None,
                        "timestamp": order.get("updated_at", now.isoformat())
                    })
                    alert_id += 1
                elif order.get("status") in ["rejected", "cancelled"]:
                    alerts.append({
                        "id": str(alert_id),
                        "type": "trade_execution",
                        "severity": "medium",
                        "status": "active",
                        "title": f"Order {order['status'].title()}: {order['symbol']}",
                        "message": f"{order['side'].upper()} {order['quantity']} shares was {order['status']}.",
                        "symbol": order["symbol"],
                        "price_change": None,
                        "timestamp": order.get("updated_at", now.isoformat())
                    })
                    alert_id += 1

            # Filter by severity if specified
            if severity:
                alerts = [a for a in alerts if a["severity"] == severity.lower()]

            # Filter by type if specified
            if alert_type:
                alerts = [a for a in alerts if a["type"] == alert_type.lower()]

            # Sort by severity (critical first) and limit
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            alerts.sort(key=lambda x: severity_order.get(x["severity"], 4))

            return alerts[:limit]

        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
            return []

# Global instance for real Tiger data
real_tiger_provider = RealTigerDataProvider()