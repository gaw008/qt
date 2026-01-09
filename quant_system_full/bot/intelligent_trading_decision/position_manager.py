"""
PositionManager - Manages active trading positions.

This module implements FIX 16 from the plan:
- Key by position_id instead of symbol
- This enables: adding to positions, partial exits, same symbol long/short

Position Structure:
- position_id: Unique identifier for the position
- symbol: Stock symbol
- direction: 'LONG' or 'SHORT'
- entry_price: Average entry price
- quantity: Number of shares
- entry_time: When position was opened
- status: 'OPEN' or 'CLOSED'
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)


class Position:
    """Represents a single trading position."""

    def __init__(
        self,
        position_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        entry_time: datetime
    ):
        """
        Initialize a Position.

        Args:
            position_id: Unique identifier
            symbol: Stock symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            quantity: Number of shares
            entry_time: Entry timestamp
        """
        self.position_id = position_id
        self.symbol = symbol
        self.direction = direction  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.status = 'OPEN'

        # Exit fields (populated on close)
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: Optional[str] = None
        self.pnl: Optional[float] = None
        self.pnl_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'status': self.status,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
        }

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in dollars
        """
        if self.direction == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity

    def calculate_unrealized_pnl_percent(self, current_price: float) -> float:
        """
        Calculate unrealized P&L percentage.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L as percentage
        """
        if self.entry_price == 0:
            return 0.0
        pnl = self.calculate_unrealized_pnl(current_price)
        return (pnl / (self.entry_price * self.quantity)) * 100


class PositionManager:
    """
    Manages active trading positions.

    FIX 16: Key by position_id, not symbol.
    This enables:
    - Adding to positions (second ENTRY doesn't overwrite first)
    - Partial exits
    - Same symbol long/short simultaneously

    MVP Constraint: single_position_per_symbol = True
    This can be relaxed later for more complex strategies.
    """

    def __init__(self, single_position_per_symbol: bool = True):
        """
        Initialize PositionManager.

        Args:
            single_position_per_symbol: If True, enforce only one position per symbol (MVP mode)
        """
        # FIX 16: Key by position_id, not symbol
        self.active_positions: Dict[str, Position] = {}  # position_id -> Position

        # Optional: Quick lookup if enforcing 1 position per symbol
        self.symbol_to_position_id: Dict[str, str] = {}  # symbol -> position_id

        # MVP: explicit constraint
        self.single_position_per_symbol = single_position_per_symbol

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        entry_time: datetime,
        position_id: Optional[str] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            quantity: Number of shares
            entry_time: Entry timestamp
            position_id: Optional position ID (auto-generated if not provided)

        Returns:
            The opened Position object

        Raises:
            ValueError: If single_position_per_symbol is True and symbol already has a position
        """
        # Generate position_id if not provided
        if position_id is None:
            position_id = str(uuid.uuid4())

        # FIX 16: Enforce single position per symbol if configured
        if self.single_position_per_symbol:
            if symbol in self.symbol_to_position_id:
                existing_id = self.symbol_to_position_id[symbol]
                raise ValueError(
                    f"Symbol {symbol} already has position {existing_id}. "
                    f"Close existing position first or set single_position_per_symbol=False."
                )

        position = Position(
            position_id=position_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time
        )

        self.active_positions[position_id] = position
        self.symbol_to_position_id[symbol] = position_id

        logger.info(
            f"POSITION OPENED: {position_id} | {symbol} {direction} | "
            f"{quantity} shares @ ${entry_price:.2f}"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        exit_time: datetime
    ) -> Position:
        """
        Close an existing position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_reason: Reason for exit (e.g., "Stop loss hit", "Take profit")
            exit_time: Exit timestamp

        Returns:
            The closed Position object with P&L calculated

        Raises:
            ValueError: If position_id not found
        """
        if position_id not in self.active_positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.active_positions[position_id]
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.exit_time = exit_time
        position.status = 'CLOSED'

        # Calculate P&L (direction-aware)
        if position.direction == 'LONG':
            position.pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            position.pnl = (position.entry_price - exit_price) * position.quantity

        # Calculate P&L percentage
        if position.entry_price > 0:
            position.pnl_percent = (position.pnl / (position.entry_price * position.quantity)) * 100
        else:
            position.pnl_percent = 0.0

        # Clean up lookups
        symbol = position.symbol
        if self.symbol_to_position_id.get(symbol) == position_id:
            del self.symbol_to_position_id[symbol]

        del self.active_positions[position_id]

        logger.info(
            f"POSITION CLOSED: {position_id} | {symbol} {position.direction} | "
            f"P&L: ${position.pnl:.2f} ({position.pnl_percent:.2f}%) | "
            f"Reason: {exit_reason}"
        )

        return position

    def partial_close(
        self,
        position_id: str,
        exit_price: float,
        close_quantity: int,
        exit_reason: str,
        exit_time: datetime
    ) -> tuple:
        """
        Partially close a position.

        Args:
            position_id: Position ID
            exit_price: Exit price
            close_quantity: Number of shares to close
            exit_reason: Reason for partial close
            exit_time: Exit timestamp

        Returns:
            Tuple of (closed_position: Position, remaining_position: Position or None)
        """
        if position_id not in self.active_positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.active_positions[position_id]

        if close_quantity >= position.quantity:
            # Full close
            return self.close_position(position_id, exit_price, exit_reason, exit_time), None

        # Create closed portion as a new position record
        closed_portion = Position(
            position_id=str(uuid.uuid4()),
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            quantity=close_quantity,
            entry_time=position.entry_time
        )
        closed_portion.exit_price = exit_price
        closed_portion.exit_time = exit_time
        closed_portion.exit_reason = exit_reason
        closed_portion.status = 'CLOSED'

        # Calculate P&L for closed portion
        if position.direction == 'LONG':
            closed_portion.pnl = (exit_price - position.entry_price) * close_quantity
        else:
            closed_portion.pnl = (position.entry_price - exit_price) * close_quantity

        if position.entry_price > 0:
            closed_portion.pnl_percent = (closed_portion.pnl / (position.entry_price * close_quantity)) * 100

        # Reduce remaining position
        position.quantity -= close_quantity

        logger.info(
            f"PARTIAL CLOSE: {position_id} | {position.symbol} | "
            f"Closed {close_quantity} shares @ ${exit_price:.2f} | "
            f"Remaining: {position.quantity} shares"
        )

        return closed_portion, position

    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get a position by position_id.

        Args:
            position_id: Position ID

        Returns:
            Position object or None if not found
        """
        return self.active_positions.get(position_id)

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol (single position mode).

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None if no position exists
        """
        position_id = self.symbol_to_position_id.get(symbol)
        if position_id:
            return self.active_positions.get(position_id)
        return None

    def has_open_position(self, symbol: str) -> bool:
        """
        Check if symbol has an open position.

        Args:
            symbol: Stock symbol

        Returns:
            True if symbol has an open position
        """
        return symbol in self.symbol_to_position_id

    def get_all_positions(self) -> List[Position]:
        """
        Get all active positions.

        Returns:
            List of all active Position objects
        """
        return list(self.active_positions.values())

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of Position objects for the symbol
        """
        return [p for p in self.active_positions.values() if p.symbol == symbol]

    def get_total_exposure(self, symbol: str) -> int:
        """
        Get total share exposure for a symbol (considering direction).

        Args:
            symbol: Stock symbol

        Returns:
            Net share count (positive = net long, negative = net short)
        """
        total = 0
        for position in self.active_positions.values():
            if position.symbol == symbol:
                if position.direction == 'LONG':
                    total += position.quantity
                else:  # SHORT
                    total -= position.quantity
        return total

    def get_total_value(self, price_getter) -> float:
        """
        Get total value of all positions.

        Args:
            price_getter: Function that returns current price for a symbol

        Returns:
            Total position value in dollars
        """
        total = 0.0
        for position in self.active_positions.values():
            current_price = price_getter(position.symbol)
            if current_price:
                total += current_price * position.quantity
        return total

    def get_unrealized_pnl(self, price_getter) -> float:
        """
        Get total unrealized P&L.

        Args:
            price_getter: Function that returns current price for a symbol

        Returns:
            Total unrealized P&L in dollars
        """
        total = 0.0
        for position in self.active_positions.values():
            current_price = price_getter(position.symbol)
            if current_price:
                total += position.calculate_unrealized_pnl(current_price)
        return total

    def reset_daily(self) -> None:
        """
        Reset for new trading day.

        Note: This does NOT close positions, only resets internal tracking.
        Positions should be reconciled with broker at market open.
        """
        logger.info(f"PositionManager: Daily reset with {len(self.active_positions)} open positions")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all positions to dictionary format.

        Returns:
            Dict with position data
        """
        return {
            'positions': [p.to_dict() for p in self.active_positions.values()],
            'symbol_count': len(self.symbol_to_position_id),
            'position_count': len(self.active_positions),
            'single_position_per_symbol': self.single_position_per_symbol,
        }


# Global singleton instance
_position_manager: Optional[PositionManager] = None


def get_position_manager() -> PositionManager:
    """Get the global PositionManager singleton."""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager


def set_position_manager(manager: PositionManager) -> None:
    """Set the global PositionManager singleton."""
    global _position_manager
    _position_manager = manager
