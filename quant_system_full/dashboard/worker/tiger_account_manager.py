"""
Tiger Account Manager - Dynamic account data retrieval from Tiger API
Provides real-time account information without hardcoded values
"""
import os
import sys
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient

logger = logging.getLogger(__name__)


class TigerAccountManager:
    """
    Manages Tiger Brokers account data with real-time API access.
    Eliminates hardcoded account values by fetching data directly from Tiger API.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Tiger Account Manager

        Args:
            config_path: Path to Tiger API config directory (defaults to props/)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.client_config = None
        self.trade_client = None
        self.account_id = None
        self._cached_assets = None
        self._cache_timestamp = 0
        self._cache_ttl = 60  # Cache for 60 seconds

        # CRITICAL: Position caching to prevent empty returns on API failure
        self._cached_positions = None
        self._position_cache_timestamp = 0

        self._initialize_client()

    def _get_default_config_path(self) -> str:
        """Get default config path relative to current file"""
        base_dir = Path(__file__).parent.parent.parent
        props_dir = base_dir / "props"
        return str(props_dir.resolve())

    def _initialize_client(self):
        """Initialize Tiger API client"""
        try:
            self.client_config = TigerOpenClientConfig(props_path=self.config_path)
            self.client_config.language = 'en_US'
            self.trade_client = TradeClient(self.client_config)

            # Get account ID from environment or config
            self.account_id = os.getenv('ACCOUNT', '41169270')

            logger.info(f"[TIGER_ACCOUNT] Initialized for account {self.account_id}")
        except Exception as e:
            logger.error(f"[TIGER_ACCOUNT] Failed to initialize: {e}")
            raise

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed"""
        import time
        return (time.time() - self._cache_timestamp) > self._cache_ttl

    def get_account_assets(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get account assets from Tiger API

        Args:
            force_refresh: Force refresh cache

        Returns:
            Dictionary with account asset information or None on error
        """
        if not force_refresh and self._cached_assets and not self._should_refresh_cache():
            return self._cached_assets

        try:
            assets_list = self.trade_client.get_assets()

            if not assets_list:
                logger.warning("[TIGER_ACCOUNT] No assets returned from API")
                return None

            asset = assets_list[0]  # Get first account

            result = {
                'account_id': asset.account,
                'net_liquidation': float(asset.summary.net_liquidation or 0),
                'cash': float(asset.summary.cash or 0),
                'buying_power': float(asset.summary.buying_power or 0),
                'gross_position_value': float(asset.summary.gross_position_value or 0)
                    if asset.summary.gross_position_value != float('inf') else 0,
                'equity_with_loan': float(asset.summary.equity_with_loan or 0)
                    if asset.summary.equity_with_loan != float('inf') else 0,
            }

            # Cache the result
            self._cached_assets = result
            import time
            self._cache_timestamp = time.time()

            logger.info(f"[TIGER_ACCOUNT] Assets updated: Net=${result['net_liquidation']:,.2f}, "
                       f"Cash=${result['cash']:,.2f}, Buying Power=${result['buying_power']:,.2f}")

            return result

        except Exception as e:
            logger.error(f"[TIGER_ACCOUNT] Failed to get assets: {e}")
            return self._cached_assets  # Return cached data on error

    def get_positions(self) -> list:
        """
        Get current positions from Tiger API with caching fallback

        Returns:
            List of position objects (cached on API failure)
        """
        import time

        try:
            positions = self.trade_client.get_positions()
            position_count = len(positions) if positions else 0
            logger.info(f"[TIGER_ACCOUNT] Retrieved {position_count} positions")

            # CRITICAL: Cache successful results
            if positions:
                self._cached_positions = positions
                self._position_cache_timestamp = time.time()
                logger.debug(f"[TIGER_ACCOUNT] Cached {position_count} positions")

            return positions or []

        except Exception as e:
            logger.error(f"[TIGER_ACCOUNT] Failed to get positions: {e}")

            # CRITICAL: Return cached positions instead of empty list
            if self._cached_positions is not None:
                cache_age = time.time() - self._position_cache_timestamp
                logger.warning(f"[TIGER_ACCOUNT] Using cached positions (age: {cache_age:.1f}s, count: {len(self._cached_positions)})")
                return self._cached_positions

            # Only return empty if we have never successfully fetched positions
            logger.error("[TIGER_ACCOUNT] No cached positions available, returning empty list")
            return []

    def get_position_count(self) -> int:
        """Get number of open positions"""
        positions = self.get_positions()
        return len(positions)

    def get_total_position_value(self) -> float:
        """Get total market value of all positions"""
        positions = self.get_positions()
        if not positions:
            return 0.0

        total = sum(float(pos.market_value or 0) for pos in positions)
        return total

    def get_net_liquidation(self) -> float:
        """Get net liquidation value (total account value)"""
        assets = self.get_account_assets()
        return assets['net_liquidation'] if assets else 0.0

    def get_cash_balance(self) -> float:
        """Get available cash balance"""
        assets = self.get_account_assets()
        return assets['cash'] if assets else 0.0

    def get_buying_power(self) -> float:
        """Get available buying power"""
        assets = self.get_account_assets()
        return assets['buying_power'] if assets else 0.0

    def calculate_max_position_size(self, percentage: float = 0.25) -> float:
        """
        Calculate maximum position size based on account value

        Args:
            percentage: Maximum percentage of account for single position (default 25%)

        Returns:
            Maximum position size in dollars
        """
        net_liq = self.get_net_liquidation()
        max_size = net_liq * percentage

        logger.info(f"[TIGER_ACCOUNT] Max position size: ${max_size:,.2f} "
                   f"({percentage*100}% of ${net_liq:,.2f})")

        return max_size

    def calculate_position_concentration(self, position_value: float) -> float:
        """
        Calculate position concentration as percentage of account

        Args:
            position_value: Value of position in dollars

        Returns:
            Concentration as decimal (e.g., 0.25 for 25%)
        """
        net_liq = self.get_net_liquidation()
        if net_liq == 0:
            return 0.0

        concentration = position_value / net_liq
        return concentration

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive account summary

        Returns:
            Dictionary with all account information
        """
        assets = self.get_account_assets()
        positions = self.get_positions()

        if not assets:
            return {
                'error': 'Failed to retrieve account data',
                'account_id': self.account_id
            }

        position_details = []
        for pos in positions[:10]:  # Limit to first 10 for summary
            position_details.append({
                'symbol': pos.contract.symbol,
                'quantity': pos.quantity,
                'average_cost': float(pos.average_cost or 0),
                'market_value': float(pos.market_value or 0),
                'unrealized_pnl': float(getattr(pos, 'unrealized_pnl', 0) or 0)
            })

        summary = {
            'account_id': assets['account_id'],
            'net_liquidation': assets['net_liquidation'],
            'cash': assets['cash'],
            'buying_power': assets['buying_power'],
            'gross_position_value': assets['gross_position_value'],
            'position_count': len(positions),
            'positions': position_details,
            'max_position_size_25pct': self.calculate_max_position_size(0.25),
            'max_position_size_20pct': self.calculate_max_position_size(0.20),
            'max_position_size_15pct': self.calculate_max_position_size(0.15),
        }

        return summary

    def validate_trade_size(self, trade_value: float, max_concentration: float = 0.25) -> Dict[str, Any]:
        """
        Validate if trade size is within account limits

        Args:
            trade_value: Proposed trade value in dollars
            max_concentration: Maximum allowed concentration (default 25%)

        Returns:
            Dictionary with validation result and details
        """
        net_liq = self.get_net_liquidation()
        max_size = net_liq * max_concentration
        concentration = self.calculate_position_concentration(trade_value)

        is_valid = trade_value <= max_size

        return {
            'valid': is_valid,
            'trade_value': trade_value,
            'max_allowed': max_size,
            'concentration': concentration,
            'max_concentration': max_concentration,
            'account_value': net_liq,
            'message': 'Trade size OK' if is_valid else f'Trade size ${trade_value:,.2f} exceeds max ${max_size:,.2f}'
        }


# Singleton instance
_account_manager_instance = None


def get_account_manager() -> TigerAccountManager:
    """
    Get singleton instance of TigerAccountManager

    Returns:
        TigerAccountManager instance
    """
    global _account_manager_instance
    if _account_manager_instance is None:
        _account_manager_instance = TigerAccountManager()
    return _account_manager_instance


if __name__ == '__main__':
    # Test the account manager
    logging.basicConfig(level=logging.INFO)

    print("Testing Tiger Account Manager")
    print("=" * 60)

    manager = get_account_manager()
    summary = manager.get_account_summary()

    print(f"\nAccount ID: {summary['account_id']}")
    print(f"Net Liquidation: ${summary['net_liquidation']:,.2f}")
    print(f"Cash: ${summary['cash']:,.2f}")
    print(f"Buying Power: ${summary['buying_power']:,.2f}")
    print(f"Position Count: {summary['position_count']}")
    print(f"\nRecommended Max Position Sizes:")
    print(f"  25%: ${summary['max_position_size_25pct']:,.2f}")
    print(f"  20%: ${summary['max_position_size_20pct']:,.2f}")
    print(f"  15%: ${summary['max_position_size_15pct']:,.2f}")

    if summary['positions']:
        print(f"\nCurrent Positions:")
        for pos in summary['positions']:
            print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['average_cost']:.2f} = ${pos['market_value']:,.2f}")

    # Test trade validation
    print(f"\nTrade Validation Examples:")
    for test_value in [1000, 3000, 5000, 10000]:
        result = manager.validate_trade_size(test_value)
        status = "OK" if result['valid'] else "REJECTED"
        print(f"  ${test_value:,}: {status} (concentration: {result['concentration']*100:.1f}%)")