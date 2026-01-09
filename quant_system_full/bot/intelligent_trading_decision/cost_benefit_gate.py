"""
CostBenefitGate - Only trade when Expected Edge > Transaction Cost.

This module implements FIX 5-7 from the plan:
- FIX 5: All costs in consistent units ($/share)
- FIX 6: Use half-spread (more realistic execution)
- FIX 7: Distinguish per-share vs per-order fees

Purpose: Kill all small meaningless trades where costs exceed expected gains.
This is the KEY to preventing "trading cost hell".
"""

import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class CostBenefitGate:
    """
    Expected edge must exceed transaction cost by a multiple.
    This kills all the small meaningless trades.

    CRITICAL: All values in $/share units for consistency.
    """

    def __init__(self, data_provider=None):
        """
        Initialize CostBenefitGate.

        Args:
            data_provider: Object that provides:
                - get_intraday_atr(symbol, periods)
                - get_bid_ask(symbol) -> (bid, ask)
                - get_average_daily_volume(symbol)
                - get_current_price(symbol)
        """
        # Conservative first week (was 2.0)
        self.edge_multiple = 2.5

        # Expected edge as fraction of ATR
        self.edge_atr_factor = 0.7  # Conservative: 70% of ATR

        # Tiger Brokers fee structure
        self.fee_per_share = 0.005  # $0.005/share
        self.min_order_fee = 1.00   # $1.00 minimum per order

        self._data_provider = data_provider

    def check(
        self,
        symbol: str,
        price: float,
        signal_score: float,
        shares: int = 100
    ) -> Tuple[bool, str]:
        """
        Check if expected edge justifies transaction cost.

        Args:
            symbol: Stock symbol
            price: Current price
            signal_score: Signal score (0-100)
            shares: Expected trade size (for per-order cost conversion)

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        # Calculate expected edge ($/share)
        atr_intraday = self._get_intraday_atr(symbol, periods=20)

        if atr_intraday is None or atr_intraday <= 0:
            # No ATR data - use default estimate
            atr_intraday = price * 0.01  # 1% default
            logger.warning(f"{symbol}: No ATR data, using 1% estimate")

        expected_edge_per_share = self.edge_atr_factor * atr_intraday

        # Calculate transaction cost (ALL in $/share)
        total_cost_per_share = self._calculate_total_cost(symbol, atr_intraday, shares, price)

        # Check if edge justifies cost
        if total_cost_per_share <= 0:
            return True, "Zero cost"

        edge_ratio = expected_edge_per_share / total_cost_per_share

        if edge_ratio < self.edge_multiple:
            return False, (
                f"Edge/Cost={edge_ratio:.2f} < {self.edge_multiple} | "
                f"Edge=${expected_edge_per_share:.3f}/sh, Cost=${total_cost_per_share:.3f}/sh"
            )

        return True, f"Edge/Cost={edge_ratio:.2f} OK"

    def _calculate_total_cost(
        self,
        symbol: str,
        atr: float,
        shares: int,
        price: float
    ) -> float:
        """
        Calculate total cost per share with consistent units.

        FIX 5: All returns in $/share.
        FIX 6: Use half-spread (more realistic than full spread).
        FIX 7: Distinguish per-share vs per-order fees.

        Args:
            symbol: Stock symbol
            atr: Intraday ATR
            shares: Number of shares (for per-order cost conversion)
            price: Current price

        Returns:
            Total cost in $/share
        """
        # 1. FIX 6: Half-spread (more realistic than full spread)
        bid, ask = self._get_bid_ask(symbol)
        if bid and ask and ask > bid:
            half_spread = (ask - bid) / 2  # $/share
        else:
            # Default: 0.02% of price
            half_spread = price * 0.0002

        # 2. Slippage estimate
        slippage = self._estimate_slippage(symbol, atr, price)  # $/share

        # 3. FIX 7: Commission/Fee handling
        # Tiger Brokers: $0.005/share, min $1.00/order
        # If min fee dominates, convert to per-share equivalent
        fee = max(self.fee_per_share, self.min_order_fee / shares)  # $/share

        total = half_spread + slippage + fee

        logger.debug(
            f"{symbol} costs: half_spread=${half_spread:.4f}, "
            f"slippage=${slippage:.4f}, fee=${fee:.4f}, total=${total:.4f}/share"
        )

        return total

    def _estimate_slippage(
        self,
        symbol: str,
        atr: float,
        price: float
    ) -> float:
        """
        Estimate slippage in $/share based on volatility and liquidity.

        Args:
            symbol: Stock symbol
            atr: Intraday ATR
            price: Current price

        Returns:
            Estimated slippage in $/share
        """
        adv = self._get_average_daily_volume(symbol)

        if adv is None or adv <= 0:
            adv = 1_000_000  # Default 1M shares

        adv_dollars = adv * price

        # Base slippage: 1bp of price
        base_slippage = price * 0.0001  # 1bp

        # Volatility factor: higher ATR = more slippage
        # Normalize ATR to 1% as baseline
        vol_factor = min(2.0, atr / (price * 0.01))  # Relative to 1%

        # Liquidity factor: lower ADV$ = more slippage
        # $50M ADV as baseline (neutral = 1.0)
        liq_factor = max(0.5, min(2.0, 50_000_000 / adv_dollars))

        return base_slippage * vol_factor * liq_factor

    def _get_intraday_atr(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Get intraday ATR from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_intraday_atr'):
                return self._data_provider.get_intraday_atr(symbol, periods)
            return None
        except Exception as e:
            logger.error(f"Error getting ATR for {symbol}: {e}")
            return None

    def _get_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get bid/ask from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_bid_ask'):
                return self._data_provider.get_bid_ask(symbol)
            return None, None
        except Exception as e:
            logger.error(f"Error getting bid/ask for {symbol}: {e}")
            return None, None

    def _get_average_daily_volume(self, symbol: str) -> Optional[float]:
        """Get average daily volume from data provider."""
        try:
            if self._data_provider and hasattr(self._data_provider, 'get_average_daily_volume'):
                return self._data_provider.get_average_daily_volume(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting ADV for {symbol}: {e}")
            return None

    def set_data_provider(self, data_provider) -> None:
        """Set the data provider."""
        self._data_provider = data_provider

    def set_edge_multiple(self, multiple: float) -> None:
        """
        Set the required edge/cost multiple.

        Args:
            multiple: Required ratio of expected edge to transaction cost
        """
        self.edge_multiple = multiple
        logger.info(f"CostBenefitGate: Edge multiple set to {multiple}")

    def get_cost_breakdown(
        self,
        symbol: str,
        price: float,
        shares: int = 100
    ) -> Dict[str, Any]:
        """
        Get detailed cost breakdown for a trade (for debugging/monitoring).

        Args:
            symbol: Stock symbol
            price: Current price
            shares: Number of shares

        Returns:
            Dict with cost components
        """
        atr = self._get_intraday_atr(symbol, periods=20) or (price * 0.01)
        bid, ask = self._get_bid_ask(symbol)

        if bid and ask and ask > bid:
            half_spread = (ask - bid) / 2
        else:
            half_spread = price * 0.0002

        slippage = self._estimate_slippage(symbol, atr, price)
        fee = max(self.fee_per_share, self.min_order_fee / shares)

        expected_edge = self.edge_atr_factor * atr
        total_cost = half_spread + slippage + fee
        edge_ratio = expected_edge / total_cost if total_cost > 0 else float('inf')

        return {
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'atr_intraday': atr,
            'expected_edge_per_share': expected_edge,
            'half_spread': half_spread,
            'slippage': slippage,
            'fee': fee,
            'total_cost_per_share': total_cost,
            'edge_cost_ratio': edge_ratio,
            'threshold': self.edge_multiple,
            'would_pass': edge_ratio >= self.edge_multiple,
        }


# Global singleton instance
_cost_benefit_gate: Optional[CostBenefitGate] = None


def get_cost_benefit_gate() -> CostBenefitGate:
    """Get the global CostBenefitGate singleton."""
    global _cost_benefit_gate
    if _cost_benefit_gate is None:
        _cost_benefit_gate = CostBenefitGate()
    return _cost_benefit_gate


def set_cost_benefit_gate(gate: CostBenefitGate) -> None:
    """Set the global CostBenefitGate singleton."""
    global _cost_benefit_gate
    _cost_benefit_gate = gate
