"""
Market Time Detection and Trading Hours Management

This module provides comprehensive market timing functionality for the quantitative trading system.
It supports multiple markets (US, CN) and different trading phases (pre-market, regular, after-hours).
"""

import os
import pytz
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
from bot.config import SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market trading phases."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class MarketType(Enum):
    """Supported market types."""
    US = "US"
    CN = "CN"


@dataclass
class TradingHours:
    """Trading hours configuration for a market."""
    pre_market_start: time
    pre_market_end: time
    regular_start: time
    regular_end: time
    after_hours_start: time
    after_hours_end: time
    timezone: str


class MarketTimeManager:
    """
    Manages market timing, trading hours, and market phase detection.
    """
    
    # Trading hours configuration
    MARKET_HOURS = {
        MarketType.US: TradingHours(
            pre_market_start=time(4, 0),    # 4:00 AM ET
            pre_market_end=time(9, 30),     # 9:30 AM ET
            regular_start=time(9, 30),      # 9:30 AM ET
            regular_end=time(16, 0),        # 4:00 PM ET
            after_hours_start=time(16, 0),  # 4:00 PM ET
            after_hours_end=time(20, 0),    # 8:00 PM ET
            timezone="America/New_York"
        ),
        MarketType.CN: TradingHours(
            pre_market_start=time(8, 0),    # 8:00 AM CST
            pre_market_end=time(9, 30),     # 9:30 AM CST
            regular_start=time(9, 30),      # 9:30 AM CST
            regular_end=time(15, 0),        # 3:00 PM CST (with 11:30-13:00 break)
            after_hours_start=time(15, 0),  # 3:00 PM CST
            after_hours_end=time(17, 0),    # 5:00 PM CST
            timezone="Asia/Shanghai"
        )
    }
    
    # Weekend days
    WEEKEND_DAYS = {5, 6}  # Saturday, Sunday
    
    def __init__(self, market_type: MarketType = MarketType.US):
        """Initialize market time manager for specified market type."""
        self.market_type = market_type
        self.trading_hours = self.MARKET_HOURS[market_type]
        self.timezone = pytz.timezone(self.trading_hours.timezone)
        
    def get_current_market_time(self) -> datetime:
        """Get current time in market timezone."""
        return datetime.now(self.timezone)
    
    def get_market_phase(self, dt: Optional[datetime] = None) -> MarketPhase:
        """
        Determine current market phase.
        
        Args:
            dt: Optional datetime to check. If None, uses current time.
            
        Returns:
            Current market phase
        """
        if dt is None:
            dt = self.get_current_market_time()
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        elif dt.tzinfo != self.timezone:
            dt = dt.astimezone(self.timezone)
            
        # Check if weekend
        if dt.weekday() in self.WEEKEND_DAYS:
            return MarketPhase.CLOSED
            
        current_time = dt.time()
        
        # Check market phases
        if (self.trading_hours.pre_market_start <= current_time < self.trading_hours.pre_market_end):
            return MarketPhase.PRE_MARKET
        elif (self.trading_hours.regular_start <= current_time < self.trading_hours.regular_end):
            # For CN market, handle lunch break (11:30-13:00)
            if (self.market_type == MarketType.CN and 
                time(11, 30) <= current_time < time(13, 0)):
                return MarketPhase.CLOSED
            return MarketPhase.REGULAR
        elif (self.trading_hours.after_hours_start <= current_time < self.trading_hours.after_hours_end):
            return MarketPhase.AFTER_HOURS
        else:
            return MarketPhase.CLOSED
    
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if market is open (regular trading hours)."""
        phase = self.get_market_phase(dt)
        return phase == MarketPhase.REGULAR
    
    def is_market_active(self, dt: Optional[datetime] = None) -> bool:
        """Check if market is active (any trading phase)."""
        phase = self.get_market_phase(dt)
        return phase in {MarketPhase.PRE_MARKET, MarketPhase.REGULAR, MarketPhase.AFTER_HOURS}
    
    def get_next_market_open(self, dt: Optional[datetime] = None) -> datetime:
        """Get next regular market opening time."""
        if dt is None:
            dt = self.get_current_market_time()
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
            
        # Start from next day if after market close today
        current_time = dt.time()
        if current_time >= self.trading_hours.regular_end:
            dt = dt + timedelta(days=1)
            
        # Find next weekday
        while dt.weekday() in self.WEEKEND_DAYS:
            dt = dt + timedelta(days=1)
            
        # Set to market open time
        return dt.replace(
            hour=self.trading_hours.regular_start.hour,
            minute=self.trading_hours.regular_start.minute,
            second=0,
            microsecond=0
        )
    
    def get_next_market_close(self, dt: Optional[datetime] = None) -> datetime:
        """Get next regular market closing time."""
        if dt is None:
            dt = self.get_current_market_time()
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
            
        # If before market open today, return today's close
        current_time = dt.time()
        if current_time < self.trading_hours.regular_start:
            return dt.replace(
                hour=self.trading_hours.regular_end.hour,
                minute=self.trading_hours.regular_end.minute,
                second=0,
                microsecond=0
            )
            
        # Otherwise, find next market day
        dt = dt + timedelta(days=1)
        while dt.weekday() in self.WEEKEND_DAYS:
            dt = dt + timedelta(days=1)
            
        return dt.replace(
            hour=self.trading_hours.regular_end.hour,
            minute=self.trading_hours.regular_end.minute,
            second=0,
            microsecond=0
        )
    
    def get_time_to_next_phase(self, target_phase: MarketPhase, dt: Optional[datetime] = None) -> timedelta:
        """Get time until next occurrence of target market phase."""
        if dt is None:
            dt = self.get_current_market_time()
            
        current_phase = self.get_market_phase(dt)
        
        if target_phase == MarketPhase.REGULAR:
            next_open = self.get_next_market_open(dt)
            return next_open - dt
        elif target_phase == MarketPhase.CLOSED:
            next_close = self.get_next_market_close(dt)
            return next_close - dt
        else:
            # For pre-market and after-hours, calculate based on current phase
            if current_phase == target_phase:
                return timedelta(0)
            
            # Simple approximation - could be enhanced
            if target_phase == MarketPhase.PRE_MARKET:
                next_pre = self.get_next_market_open(dt) - timedelta(
                    hours=self.trading_hours.regular_start.hour - self.trading_hours.pre_market_start.hour,
                    minutes=self.trading_hours.regular_start.minute - self.trading_hours.pre_market_start.minute
                )
                return next_pre - dt
            
            return timedelta(hours=1)  # Default fallback
    
    def get_market_status(self, include_yahoo_api: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive market status information.
        
        Args:
            include_yahoo_api: Whether to include Yahoo Finance API status
            
        Returns:
            Dictionary with market status details
        """
        current_time = self.get_current_market_time()
        phase = self.get_market_phase(current_time)
        
        status = {
            'market_type': self.market_type.value,
            'current_time': current_time.isoformat(),
            'market_phase': phase.value,
            'is_market_open': self.is_market_open(current_time),
            'is_market_active': self.is_market_active(current_time),
            'next_market_open': self.get_next_market_open(current_time).isoformat(),
            'next_market_close': self.get_next_market_close(current_time).isoformat(),
            'timezone': self.trading_hours.timezone,
            'trading_hours': {
                'pre_market': f"{self.trading_hours.pre_market_start} - {self.trading_hours.pre_market_end}",
                'regular': f"{self.trading_hours.regular_start} - {self.trading_hours.regular_end}",
                'after_hours': f"{self.trading_hours.after_hours_start} - {self.trading_hours.after_hours_end}"
            }
        }
        
        # Include Yahoo Finance market status if requested
        if include_yahoo_api:
            try:
                from .yahoo_data import get_market_status
                yahoo_status = get_market_status()
                status['yahoo_finance'] = yahoo_status
            except Exception as e:
                logger.warning(f"Failed to get Yahoo Finance market status: {e}")
                status['yahoo_finance'] = {'error': str(e)}
        
        return status
    
    def should_run_selection_tasks(self, dt: Optional[datetime] = None) -> bool:
        """
        Determine if stock selection tasks should run.
        Selection tasks run during market closed periods.
        """
        phase = self.get_market_phase(dt)
        return phase == MarketPhase.CLOSED
    
    def should_run_trading_tasks(self, dt: Optional[datetime] = None) -> bool:
        """
        Determine if trading monitoring tasks should run.
        Trading tasks run during market active periods.
        """
        phase = self.get_market_phase(dt)
        return phase in {MarketPhase.PRE_MARKET, MarketPhase.REGULAR, MarketPhase.AFTER_HOURS}


# Global instances for different markets
US_MARKET = MarketTimeManager(MarketType.US)
CN_MARKET = MarketTimeManager(MarketType.CN)


def get_market_manager(market_type: str = "US") -> MarketTimeManager:
    """Get market manager for specified market type."""
    if market_type.upper() == "CN":
        return CN_MARKET
    return US_MARKET


def get_current_market_phase(market_type: str = "US") -> MarketPhase:
    """Get current market phase for specified market."""
    manager = get_market_manager(market_type)
    return manager.get_market_phase()


def get_comprehensive_market_status() -> Dict[str, Any]:
    """Get market status for all supported markets."""
    return {
        'US': US_MARKET.get_market_status(),
        'CN': CN_MARKET.get_market_status(),
        'timestamp': datetime.utcnow().isoformat(),
        'primary_market': os.getenv('PRIMARY_MARKET', 'US')
    }


if __name__ == "__main__":
    # Test market time functionality
    print("=== Market Time Manager Test ===")
    
    for market_type in [MarketType.US, MarketType.CN]:
        manager = MarketTimeManager(market_type)
        status = manager.get_market_status(include_yahoo_api=False)
        
        print(f"\n{market_type.value} Market Status:")
        print(f"Current Time: {status['current_time']}")
        print(f"Market Phase: {status['market_phase']}")
        print(f"Is Market Open: {status['is_market_open']}")
        print(f"Next Market Open: {status['next_market_open']}")
        print(f"Should Run Selection: {manager.should_run_selection_tasks()}")
        print(f"Should Run Trading: {manager.should_run_trading_tasks()}")