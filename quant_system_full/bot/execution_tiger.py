"""
Tiger API Standard Order Execution Engine

This module provides Tiger API compliant order execution using the official SDK,
replacing the previous custom order management system with standardized Tiger API calls.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tigeropen.common.util.contract_utils import stock_contract, future_contract, option_contract
    from tigeropen.common.util.order_utils import market_order, limit_order
    from tigeropen.common.consts import Market, SecurityType, OrderStatus, Currency
    from tigeropen.trade.trade_client import TradeClient
    from tigeropen.quote.quote_client import QuoteClient
    TIGER_SDK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Tiger SDK not available: {e}")
    TIGER_SDK_AVAILABLE = False

# Import multi-asset managers for asset type detection
try:
    from .multi_asset_data_manager import AssetType, MultiAssetDataManager
    from .futures_manager import futures_manager
    from .etf_manager import etf_manager
    MULTI_ASSET_AVAILABLE = True
except ImportError:
    try:
        # Fallback to absolute imports
        from multi_asset_data_manager import AssetType, MultiAssetDataManager
        from futures_manager import futures_manager
        from etf_manager import etf_manager
        MULTI_ASSET_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Multi-asset modules not available: {e}")
        AssetType = None
        MultiAssetDataManager = None
        MULTI_ASSET_AVAILABLE = False


class TigerOrderSide(Enum):
    """Tiger API order sides"""
    BUY = "BUY"
    SELL = "SELL"


class TigerOrderType(Enum):
    """Tiger API order types"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"


@dataclass
class TigerOrderResult:
    """Tiger order execution result"""
    success: bool
    order_id: Optional[int] = None
    order_object: Optional[Any] = None
    message: str = ""
    error: Optional[str] = None
    asset_type: Optional[str] = None
    contract_info: Optional[Dict] = None


class TigerExecutionEngine:
    """
    Tiger API Standard Order Execution Engine
    
    Uses official Tiger API methods for order placement and management,
    providing standardized order execution for the quantitative trading system.
    """
    
    def __init__(self, quote_client: QuoteClient, trade_client: TradeClient):
        """Initialize Tiger execution engine with API clients"""
        if not TIGER_SDK_AVAILABLE:
            raise RuntimeError("Tiger SDK is required but not available")
            
        self.quote_client = quote_client
        self.trade_client = trade_client
        # Get account from trade client configuration
        try:
            # Try different ways to get account number
            self.account = (
                getattr(trade_client, '_account', None) or  # Private attribute
                getattr(trade_client, 'account', None) or   # Public attribute
                getattr(getattr(trade_client, 'client_config', None), 'account', None) or  # Config attribute
                'Unknown'
            )
        except (AttributeError, TypeError):
            self.account = 'Unknown'
        
        # Multi-asset data manager for asset type detection
        if MULTI_ASSET_AVAILABLE:
            self.data_manager = MultiAssetDataManager()
        else:
            self.data_manager = None
        
        logger.info(f"[TigerExecution] Engine initialized for account: {self.account}")
        logger.info(f"[TigerExecution] Multi-asset support: {MULTI_ASSET_AVAILABLE}")
    
    def _detect_asset_type(self, symbol: str) -> str:
        """
        Detect asset type for the given symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset type string ('stock', 'etf', 'future', 'option', etc.)
        """
        if not MULTI_ASSET_AVAILABLE or not self.data_manager:
            return 'stock'  # Default fallback
        
        try:
            asset_type = self.data_manager.detect_asset_type(symbol)
            return asset_type.value if asset_type else 'stock'
        except Exception as e:
            logger.warning(f"[TigerExecution] Asset type detection failed for {symbol}: {e}")
            return 'stock'
    
    def _create_contract(self, symbol: str, asset_type: str = None, currency: str = "USD"):
        """
        Create appropriate contract based on asset type.
        
        Args:
            symbol: Asset symbol
            asset_type: Asset type override (optional)
            currency: Currency code
            
        Returns:
            Tiger API contract object
        """
        if not asset_type:
            asset_type = self._detect_asset_type(symbol)
        
        logger.info(f"[TigerExecution] Creating {asset_type} contract for {symbol}")
        
        try:
            if asset_type == 'stock' or asset_type == 'etf' or asset_type == 'reit' or asset_type == 'adr':
                # All equity-like instruments use stock_contract
                return stock_contract(symbol=symbol, currency=currency)
            
            elif asset_type == 'futures':
                # Handle futures contracts
                if MULTI_ASSET_AVAILABLE:
                    contract_spec = futures_manager.get_contract_spec(symbol)
                    if contract_spec:
                        # Use futures contract with proper exchange and expiry
                        return future_contract(
                            symbol=symbol,
                            exchange=contract_spec.exchange.value,
                            currency=contract_spec.currency,
                            expiry=contract_spec.expiry_month
                        )
                
                # Fallback to basic futures contract
                return future_contract(symbol=symbol, currency=currency)
            
            else:
                # Default to stock contract for unknown types
                logger.warning(f"[TigerExecution] Unknown asset type {asset_type}, using stock contract")
                return stock_contract(symbol=symbol, currency=currency)
                
        except Exception as e:
            logger.error(f"[TigerExecution] Error creating contract for {symbol}: {e}")
            # CRITICAL FIX: Only fallback to stock for stock-like assets, raise error for others
            if asset_type in ['stock', 'etf', 'reit', 'adr', None]:
                logger.warning(f"[TigerExecution] Falling back to stock contract for {symbol}")
                return stock_contract(symbol=symbol, currency=currency)
            else:
                # For non-stock assets (futures, options), do NOT silently convert
                logger.error(f"[TigerExecution] CRITICAL: Cannot create {asset_type} contract for {symbol}, refusing to fallback to stock")
                raise ValueError(f"Failed to create {asset_type} contract for {symbol}: {e}")
    
    def _get_security_type(self, asset_type: str) -> SecurityType:
        """
        Get Tiger API SecurityType for asset type.
        
        Args:
            asset_type: Asset type string
            
        Returns:
            Tiger SecurityType enum
        """
        type_mapping = {
            'stock': SecurityType.STK,
            'etf': SecurityType.STK,
            'reit': SecurityType.STK,
            'adr': SecurityType.STK,
            'futures': SecurityType.FUT,
            'option': SecurityType.OPT
        }
        
        return type_mapping.get(asset_type, SecurityType.STK)
    
    def _get_market(self, asset_type: str, symbol: str) -> Market:
        """
        Get appropriate market for asset type and symbol.
        
        Args:
            asset_type: Asset type
            symbol: Asset symbol
            
        Returns:
            Tiger Market enum
        """
        # Default to US market for most assets
        if asset_type == 'futures':
            # Futures may trade on different markets
            if symbol in ['ES', 'NQ', 'YM', 'RTY']:  # US index futures
                return Market.US
            elif symbol in ['CL', 'NG', 'GC', 'SI']:  # Commodities
                return Market.US  # NYMEX/COMEX
            else:
                return Market.US
        
        # Stocks, ETFs, REITs, ADRs typically on US market
        return Market.US
    
    def _validate_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None, asset_type: str = None) -> Tuple[bool, str]:
        """
        Validate order parameters before submission
        
        Args:
            symbol: Asset symbol
            side: BUY or SELL
            quantity: Number of shares/contracts
            price: Order price (optional for market orders)
            asset_type: Asset type override (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate symbol
        if not symbol or not symbol.strip():
            return False, "Symbol cannot be empty"
        
        # Validate side
        if side.upper() not in ['BUY', 'SELL']:
            return False, f"Invalid order side: {side}. Must be BUY or SELL"
        
        # Validate quantity
        if not isinstance(quantity, int) or quantity <= 0:
            return False, f"Invalid quantity: {quantity}. Must be positive integer"
        
        if quantity > 10000:  # Safety limit
            return False, f"Quantity too large: {quantity}. Maximum allowed: 10000 shares"
        
        # Validate price if provided
        if price is not None:
            if not isinstance(price, (int, float)) or price <= 0:
                return False, f"Invalid price: {price}. Must be positive number"
            
            if price > 10000:  # Sanity check for price
                return False, f"Price too high: {price}. Maximum allowed: $10000"
            
            # Check price precision (0.01 tick size)
            if round(price, 2) != price:
                return False, f"Price precision error: {price}. Must be rounded to 0.01"
        
        # All validations passed
        return True, ""
    
    def place_market_order(self, symbol: str, side: str, quantity: int, 
                          currency: str = "USD", asset_type: str = None) -> TigerOrderResult:
        """
        Place market order using Tiger API standard methods with multi-asset support
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'SPY', 'ES')
            side: 'BUY' or 'SELL'
            quantity: Number of shares/contracts
            currency: Currency code (default: 'USD')
            asset_type: Asset type override (optional)
            
        Returns:
            TigerOrderResult with execution details
        """
        try:
            # Detect asset type if not provided
            if not asset_type:
                asset_type = self._detect_asset_type(symbol)
            
            # Validate order parameters
            is_valid, error_msg = self._validate_order(symbol, side, quantity, None, asset_type)
            if not is_valid:
                logger.error(f"[TigerExecution] Order validation failed: {error_msg}")
                return TigerOrderResult(
                    success=False,
                    message="Order validation failed",
                    error=error_msg,
                    asset_type=asset_type
                )
            
            # Check market hours for equity-like assets (futures may have different hours)
            from datetime import datetime
            import pytz
            
            ny_tz = pytz.timezone('America/New_York')
            current_time = datetime.now(ny_tz)
            hour = current_time.hour
            minute = current_time.minute
            
            # Market orders for stocks/ETFs only work during regular hours
            if asset_type in ['stock', 'etf', 'reit', 'adr']:
                is_regular_hours = (hour == 9 and minute >= 30) or (10 <= hour < 16)
                
                if not is_regular_hours:
                    logger.warning(f"[TigerExecution] Market closed - converting to limit order")
                    try:
                        current_price = self._get_current_market_price(symbol)
                        if current_price:
                            # Add premium/discount for buy/sell orders
                            if side.upper() == 'BUY':
                                limit_price = round(current_price * 1.01, 2)
                            else:
                                limit_price = round(current_price * 0.99, 2)
                            return self.place_limit_order(symbol, side, quantity, limit_price, currency, asset_type)
                    except Exception as e:
                        logger.warning(f"[TigerExecution] Could not get current price: {e}")
            
            logger.info(f"[TigerExecution] Placing market order: {side} {quantity} {symbol} ({asset_type})")
            
            # Create appropriate contract based on asset type
            contract = self._create_contract(symbol, asset_type, currency)
            logger.info(f"[TigerExecution] Created {asset_type} contract: {contract}")
            
            # Create market order using Tiger utilities
            order = market_order(
                account=self.account,
                contract=contract,
                action=side.upper(),
                quantity=quantity
            )
            
            logger.info(f"[TigerExecution] Created order object: {order}")
            
            # Submit order
            result = self.trade_client.place_order(order)

            if result:
                # CRITICAL FIX: Robust order ID extraction from different return types
                order_id = None
                if isinstance(result, int):
                    order_id = result
                elif hasattr(result, 'id') and result.id:
                    order_id = result.id
                elif hasattr(order, 'id') and order.id:
                    order_id = order.id

                if order_id is None:
                    logger.error(f"[TigerExecution] Order submitted but could not extract order ID")
                    return TigerOrderResult(
                        success=False,
                        message="Order submitted but order ID extraction failed",
                        error="Could not extract order ID from place_order result",
                        asset_type=asset_type
                    )

                logger.info(f"[TigerExecution] Order submitted: ID={order_id}, verifying...")

                # CRITICAL: Verify order was actually placed
                verified, status, verify_error = self._verify_order_placement(order_id, symbol)

                if verified:
                    logger.info(f"[TigerExecution] SUCCESS: Order {order_id} VERIFIED with status: {status}")
                    return TigerOrderResult(
                        success=True,
                        order_id=order_id,
                        order_object=order,
                        message=f"Market order placed and verified: {side} {quantity} {symbol} ({asset_type}) - {status}",
                        asset_type=asset_type,
                        contract_info={'symbol': symbol, 'asset_type': asset_type, 'currency': currency, 'verified_status': status}
                    )
                else:
                    # Order submitted but verification failed - CRITICAL situation
                    logger.error(f"[TigerExecution] CRITICAL: Order {order_id} submitted but verification failed: {verify_error}")
                    return TigerOrderResult(
                        success=False,
                        order_id=order_id,
                        order_object=order,
                        message=f"Order submitted but verification failed - DO NOT RETRY",
                        error=f"Verification failed: {verify_error}. Order ID {order_id} may exist on Tiger side.",
                        asset_type=asset_type,
                        contract_info={'symbol': symbol, 'order_id': order_id, 'verification_failed': True}
                    )
            else:
                logger.error(f"[TigerExecution] FAILED: Order placement failed")
                return TigerOrderResult(
                    success=False,
                    message="Order placement returned False",
                    error="Tiger API place_order returned False",
                    asset_type=asset_type
                )

        except Exception as e:
            logger.error(f"[TigerExecution] ERROR: Error placing market order: {e}")
            return TigerOrderResult(
                success=False,
                message="Market order failed",
                error=str(e),
                asset_type=asset_type
            )
    
    def place_limit_order(self, symbol: str, side: str, quantity: int,
                         limit_price: float, currency: str = "USD", asset_type: str = None) -> TigerOrderResult:
        """
        Place limit order using Tiger API standard methods with multi-asset support
        
        Args:
            symbol: Asset symbol
            side: 'BUY' or 'SELL'
            quantity: Number of shares/contracts
            limit_price: Limit price
            currency: Currency code
            asset_type: Asset type override (optional)
            
        Returns:
            TigerOrderResult with execution details
        """
        try:
            # Detect asset type if not provided
            if not asset_type:
                asset_type = self._detect_asset_type(symbol)
            
            # Validate order parameters
            is_valid, error_msg = self._validate_order(symbol, side, quantity, limit_price, asset_type)
            if not is_valid:
                logger.error(f"[TigerExecution] Order validation failed: {error_msg}")
                return TigerOrderResult(
                    success=False,
                    message="Order validation failed", 
                    error=error_msg,
                    asset_type=asset_type
                )
                
            # Additional price validation - check if limit price is reasonable vs market
            try:
                market_price = self._get_current_market_price(symbol)
                if market_price:
                    price_deviation = abs(limit_price - market_price) / market_price
                    if price_deviation > 0.15:  # More than 15% deviation
                        logger.error(f"[TigerExecution] Limit price ${limit_price:.2f} deviates {price_deviation:.1%} from market ${market_price:.2f}")
                        return TigerOrderResult(
                            success=False,
                            message="Limit price too far from market",
                            error=f"Price deviation {price_deviation:.1%} exceeds 15% limit",
                            asset_type=asset_type
                        )
            except Exception as e:
                logger.warning(f"[TigerExecution] Could not validate market price: {e}")
            
            # Round price to appropriate tick size
            if asset_type == 'futures':
                # Futures may have different tick sizes
                rounded_price = round(limit_price, 2)  # Default to 0.01, could be asset-specific
            else:
                # Stocks/ETFs use 0.01 tick size
                rounded_price = round(limit_price, 2)
            
            logger.info(f"[TigerExecution] Placing limit order: {side} {quantity} {symbol} ({asset_type}) @ ${rounded_price}")
            
            # Create appropriate contract based on asset type
            contract = self._create_contract(symbol, asset_type, currency)
            
            # Create limit order
            order = limit_order(
                account=self.account,
                contract=contract,
                action=side.upper(),
                limit_price=rounded_price,
                quantity=quantity
            )
            
            # Submit order
            result = self.trade_client.place_order(order)

            if result:
                # CRITICAL FIX: Robust order ID extraction from different return types
                order_id = None
                if isinstance(result, int):
                    order_id = result
                elif hasattr(result, 'id') and result.id:
                    order_id = result.id
                elif hasattr(order, 'id') and order.id:
                    order_id = order.id

                if order_id is None:
                    logger.error(f"[TigerExecution] Limit order submitted but could not extract order ID")
                    return TigerOrderResult(
                        success=False,
                        message="Limit order submitted but order ID extraction failed",
                        error="Could not extract order ID from place_order result",
                        asset_type=asset_type
                    )

                logger.info(f"[TigerExecution] Limit order submitted: ID={order_id}, verifying...")

                # CRITICAL: Verify order was actually placed
                verified, status, verify_error = self._verify_order_placement(order_id, symbol)

                if verified:
                    logger.info(f"[TigerExecution] SUCCESS: Limit order {order_id} VERIFIED with status: {status}")
                    return TigerOrderResult(
                        success=True,
                        order_id=order_id,
                        order_object=order,
                        message=f"Limit order placed and verified: {side} {quantity} {symbol} ({asset_type}) @ ${limit_price} - {status}",
                        asset_type=asset_type,
                        contract_info={'symbol': symbol, 'asset_type': asset_type, 'currency': currency, 'limit_price': rounded_price, 'verified_status': status}
                    )
                else:
                    # Order submitted but verification failed - CRITICAL situation
                    logger.error(f"[TigerExecution] CRITICAL: Limit order {order_id} submitted but verification failed: {verify_error}")
                    return TigerOrderResult(
                        success=False,
                        order_id=order_id,
                        order_object=order,
                        message=f"Limit order submitted but verification failed - DO NOT RETRY",
                        error=f"Verification failed: {verify_error}. Order ID {order_id} may exist on Tiger side.",
                        asset_type=asset_type,
                        contract_info={'symbol': symbol, 'order_id': order_id, 'limit_price': rounded_price, 'verification_failed': True}
                    )
            else:
                return TigerOrderResult(
                    success=False,
                    message="Limit order placement failed",
                    error="Tiger API returned False",
                    asset_type=asset_type
                )

        except Exception as e:
            logger.error(f"[TigerExecution] Error placing limit order: {e}")
            return TigerOrderResult(
                success=False,
                message="Limit order failed",
                error=str(e),
                asset_type=asset_type
            )
    
    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """
        Get order status using Tiger API
        
        Args:
            order_id: Tiger order ID
            
        Returns:
            Order status dictionary or None
        """
        try:
            logger.info(f"[TigerExecution] Checking order status: {order_id}")
            
            order = self.trade_client.get_order(id=order_id)
            
            if order:
                status_info = {
                    'order_id': order.id,
                    'symbol': order.contract.symbol if order.contract else None,
                    'action': order.action,
                    'quantity': order.quantity,
                    'filled': order.filled,
                    'status': order.status,
                    'avg_fill_price': order.avg_fill_price,
                    'order_time': order.order_time,
                    'remaining': getattr(order, 'remaining', order.quantity - order.filled)
                }
                
                logger.info(f"[TigerExecution] Order status: {order.status}")
                return status_info
            else:
                logger.warning(f"[TigerExecution] Order not found: {order_id}")
                return None
                
        except Exception as e:
            logger.error(f"[TigerExecution] Error getting order status: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel order using Tiger API
        
        Args:
            order_id: Tiger order ID
            
        Returns:
            Success status
        """
        try:
            logger.info(f"[TigerExecution] Canceling order: {order_id}")
            
            result = self.trade_client.cancel_order(id=order_id)
            
            if result:
                logger.info(f"[TigerExecution] SUCCESS: Order canceled: {order_id}")
                return True
            else:
                logger.warning(f"[TigerExecution] Order cancellation failed: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"[TigerExecution] Error canceling order: {e}")
            return False
    
    def get_account_positions(self) -> List[Dict]:
        """
        Get current account positions
        
        Returns:
            List of position dictionaries
        """
        try:
            logger.info("[TigerExecution] Getting account positions")
            
            positions = self.trade_client.get_positions(
                account=self.account,  # CRITICAL FIX: Add account parameter
                sec_type=SecurityType.STK,
                market=Market.US
            )
            
            position_list = []
            if positions:
                for pos in positions:
                    position_info = {
                        'symbol': pos.contract.symbol if pos.contract else None,
                        'quantity': pos.quantity,
                        'average_cost': pos.average_cost,
                        'market_price': pos.market_price,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl
                    }
                    position_list.append(position_info)
                    
            logger.info(f"[TigerExecution] Found {len(position_list)} positions")
            return position_list
            
        except Exception as e:
            logger.error(f"[TigerExecution] Error getting positions: {e}")
            return []
    
    def _verify_order_placement(self, order_id: int, symbol: str, max_retries: int = 5,
                                   retry_delay: float = 0.5) -> Tuple[bool, str, Optional[str]]:
        """
        Verify order was actually placed by polling order status.

        CRITICAL: This prevents duplicate orders from network timeout scenarios.
        After place_order() returns, we verify the order exists on Tiger's side.

        Args:
            order_id: Order ID from place_order
            symbol: Symbol for logging
            max_retries: Maximum verification attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Tuple of (verified, status_string, error_message)
        """
        # CRITICAL FIX: Use OrderStatus enum for comparison instead of string matching
        verified_enum_statuses = [
            OrderStatus.INITIAL,
            OrderStatus.PENDING_NEW,
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED
        ]
        rejected_enum_statuses = [
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED
        ]

        for attempt in range(max_retries):
            try:
                time.sleep(retry_delay)

                order = self.trade_client.get_order(id=order_id)

                if order:
                    order_status = order.status
                    status_str = str(order_status) if order_status else 'Unknown'
                    logger.info(f"[TigerExecution] Order {order_id} verification attempt {attempt+1}: status={status_str}")

                    # CRITICAL FIX: Use enum comparison instead of string matching
                    if order_status in verified_enum_statuses:
                        logger.info(f"[TigerExecution] Order {order_id} VERIFIED: {status_str}")
                        return True, status_str, None

                    # Check for rejection using enum comparison
                    if order_status in rejected_enum_statuses:
                        error_msg = f"Order rejected/cancelled: {status_str}"
                        logger.error(f"[TigerExecution] Order {order_id} {error_msg}")
                        return False, status_str, error_msg

                else:
                    logger.warning(f"[TigerExecution] Order {order_id} not found on attempt {attempt+1}")

            except Exception as e:
                logger.warning(f"[TigerExecution] Order verification attempt {attempt+1} failed: {e}")

        # Max retries exceeded - order status uncertain
        error_msg = f"Order verification timeout after {max_retries} attempts"
        logger.error(f"[TigerExecution] {error_msg} for order {order_id}")
        return False, 'Unknown', error_msg

    def _get_current_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol via Tiger API
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current market price or None if failed
        """
        try:
            # Use symbol strings directly - Tiger API doesn't need Contract objects for briefs
            briefs = self.quote_client.get_stock_briefs([symbol])
            
            if briefs and len(briefs) > 0:
                brief = briefs[0]
                # Try different price fields in order of preference
                if hasattr(brief, 'latest_price') and brief.latest_price:
                    return float(brief.latest_price)
                elif hasattr(brief, 'pre_close') and brief.pre_close:
                    return float(brief.pre_close)
                elif hasattr(brief, 'close') and brief.close:
                    return float(brief.close)
            
            return None
            
        except Exception as e:
            logger.warning(f"[TigerExecution] Could not get market price for {symbol}: {e}")
            return None
    
    def get_account_assets(self) -> Optional[Dict]:
        """
        Get account asset information
        
        Returns:
            Asset information dictionary or None
        """
        try:
            logger.info("[TigerExecution] Getting account assets")
            
            # Try comprehensive account method first
            assets = self.trade_client.get_prime_assets()
            
            if assets:
                # Handle different types of asset responses
                if isinstance(assets, list) and len(assets) > 0:
                    asset = assets[0]
                else:
                    asset = assets
                
                # Debug: print asset structure
                logger.info(f"[TigerExecution] Asset type: {type(asset)}")
                logger.info(f"[TigerExecution] Asset attributes: {dir(asset)}")
                
                # Try different ways to access cash balance
                cash_available = None
                net_liquidation = None
                
                # Method 1: Check for segments structure
                if hasattr(asset, 'segments') and hasattr(asset.segments, 'get'):
                    try:
                        segment = asset.segments.get('S')
                        if segment:
                            cash_available = getattr(segment, 'cash_available_for_trade', None)
                            net_liquidation = getattr(segment, 'net_liquidation', None)
                    except Exception as e:
                        logger.warning(f"[TigerExecution] Error accessing segments: {e}")
                
                # Method 2: Direct access to cash attributes
                if cash_available is None:
                    cash_available = getattr(asset, 'cash_available_for_trade', None)
                    cash_available = cash_available or getattr(asset, 'cash_balance', None)
                    cash_available = cash_available or getattr(asset, 'available_cash', None)
                
                if net_liquidation is None:
                    net_liquidation = getattr(asset, 'net_liquidation', None)
                    net_liquidation = net_liquidation or getattr(asset, 'total_value', None)
                
                # Create asset info with available data
                asset_info = {
                    'account': getattr(asset, 'account', 'unknown'),
                    'cash_available': float(cash_available) if cash_available else 0.0,
                    'net_liquidation': float(net_liquidation) if net_liquidation else 0.0,
                    'buying_power': float(getattr(asset, 'buying_power', 0.0)),
                    'unrealized_pnl': float(getattr(asset, 'unrealized_pl', 0.0)),
                    'realized_pnl': float(getattr(asset, 'realized_pl', 0.0)),
                    'gross_position_value': float(getattr(asset, 'gross_position_value', 0.0))
                }
                
                logger.info(f"[TigerExecution] Assets retrieved: ${asset_info['cash_available']:.2f} available")
                return asset_info
                    
            logger.warning("[TigerExecution] No asset information available")
            return None
            
        except Exception as e:
            logger.error(f"[TigerExecution] Error getting assets: {e}")
            return None
    
    def get_recent_orders(self, hours: int = 24) -> List[Dict]:
        """
        Get recent orders within specified hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent orders
        """
        try:
            logger.info(f"[TigerExecution] Getting orders from last {hours} hours")
            
            from datetime import datetime, timedelta
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            orders = self.trade_client.get_orders(
                sec_type=SecurityType.STK,
                market=Market.US,
                start_time=int(start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000)
            )
            
            order_list = []
            if orders:
                for order in orders:
                    order_info = {
                        'order_id': order.id,
                        'symbol': order.contract.symbol if order.contract else None,
                        'action': order.action,
                        'quantity': order.quantity,
                        'filled': order.filled,
                        'status': order.status,
                        'avg_fill_price': order.avg_fill_price,
                        'order_time': order.order_time,
                        'order_type': order.order_type
                    }
                    order_list.append(order_info)
                    
            logger.info(f"[TigerExecution] Found {len(order_list)} recent orders")
            return order_list
            
        except Exception as e:
            logger.error(f"[TigerExecution] Error getting recent orders: {e}")
            return []


def create_tiger_execution_engine(quote_client, trade_client) -> Optional[TigerExecutionEngine]:
    """
    Factory function to create Tiger execution engine
    
    Args:
        quote_client: Tiger QuoteClient instance
        trade_client: Tiger TradeClient instance
        
    Returns:
        TigerExecutionEngine instance or None if SDK not available
    """
    if not TIGER_SDK_AVAILABLE:
        logger.error("Tiger SDK not available - cannot create execution engine")
        return None
        
    if not quote_client or not trade_client:
        logger.error("Invalid Tiger clients provided")
        return None
        
    try:
        engine = TigerExecutionEngine(quote_client, trade_client)
        logger.info("Tiger execution engine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create Tiger execution engine: {e}")
        return None


# Compatibility function for existing code
def place_market_order_tiger(quote_client, trade_client, symbol: str, side: str, quantity: int) -> Dict:
    """
    Compatibility function for placing market orders
    
    Returns:
        Dictionary with success status and details
    """
    engine = create_tiger_execution_engine(quote_client, trade_client)
    if not engine:
        return {'success': False, 'error': 'Tiger execution engine unavailable'}
    
    result = engine.place_market_order(symbol, side, quantity)
    
    return {
        'success': result.success,
        'order_id': result.order_id,
        'message': result.message,
        'error': result.error,
        'asset_type': result.asset_type
    }


def place_multi_asset_order(quote_client, trade_client, symbol: str, side: str, 
                           quantity: int, order_type: str = "market", 
                           limit_price: float = None, asset_type: str = None) -> Dict:
    """
    Enhanced multi-asset order placement function.
    
    Args:
        quote_client: Tiger QuoteClient
        trade_client: Tiger TradeClient  
        symbol: Asset symbol (stocks, ETFs, futures, etc.)
        side: 'BUY' or 'SELL'
        quantity: Number of shares/contracts
        order_type: 'market' or 'limit'
        limit_price: Limit price (required for limit orders)
        asset_type: Asset type override (optional)
        
    Returns:
        Dictionary with order execution results
    """
    engine = create_tiger_execution_engine(quote_client, trade_client)
    if not engine:
        return {'success': False, 'error': 'Tiger execution engine unavailable'}
    
    try:
        if order_type.lower() == "market":
            result = engine.place_market_order(symbol, side, quantity, asset_type=asset_type)
        elif order_type.lower() == "limit":
            if limit_price is None:
                return {'success': False, 'error': 'Limit price required for limit orders'}
            result = engine.place_limit_order(symbol, side, quantity, limit_price, asset_type=asset_type)
        else:
            return {'success': False, 'error': f'Unsupported order type: {order_type}'}
        
        return {
            'success': result.success,
            'order_id': result.order_id,
            'message': result.message,
            'error': result.error,
            'asset_type': result.asset_type,
            'contract_info': result.contract_info
        }
        
    except Exception as e:
        logger.error(f"Multi-asset order failed: {e}")
        return {'success': False, 'error': str(e)}


# Compatibility alias for legacy code that imports TigerExecutor class
TigerExecutor = TigerExecutionEngine