"""
Trading Calendar Management System

This module provides comprehensive trading calendar management for multiple markets,
including holiday detection, market hours, and trading session management.

Key Features:
- Multi-market calendar support (US, CN, EU, etc.)
- Holiday detection and custom holiday rules
- Market hours and timezone handling
- Trading session validation
- Suspension and resumption tracking
- Custom market rules configuration
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pytz
from pathlib import Path

try:
    import pandas_market_calendars as mcal
    MCAL_AVAILABLE = True
except ImportError:
    MCAL_AVAILABLE = False
    logger.warning("[trading_calendar] pandas_market_calendars not available, using basic calendar")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Market status types."""
    OPEN = "open"
    CLOSED = "closed"
    SUSPENDED = "suspended"
    EARLY_CLOSE = "early_close"
    LATE_OPEN = "late_open"
    UNKNOWN = "unknown"


class SessionType(Enum):
    """Trading session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    EXTENDED = "extended"


@dataclass
class TradingSession:
    """Trading session information."""
    market: str
    date: str
    session_type: SessionType
    start_time: str  # ISO format with timezone
    end_time: str    # ISO format with timezone
    status: MarketStatus = MarketStatus.OPEN

    # Optional session metadata
    volume_multiplier: float = 1.0  # Expected volume relative to normal
    liquidity_factor: float = 1.0   # Expected liquidity factor
    notes: str = ""


@dataclass
class MarketHoliday:
    """Market holiday definition."""
    name: str
    date: str
    market: str
    holiday_type: str = "full_day"  # full_day, early_close, late_open
    early_close_time: Optional[str] = None
    late_open_time: Optional[str] = None
    notes: str = ""


@dataclass
class TradingCalendarConfig:
    """Configuration for trading calendar."""
    default_market: str = "XNYS"  # NYSE
    timezone_mapping: Dict[str, str] = None
    custom_holidays: List[MarketHoliday] = None
    market_hours: Dict[str, Dict] = None
    suspension_rules: Dict[str, Any] = None

    def __post_init__(self):
        if self.timezone_mapping is None:
            self.timezone_mapping = {
                'XNYS': 'America/New_York',  # NYSE
                'XNAS': 'America/New_York',  # NASDAQ
                'XSHG': 'Asia/Shanghai',     # Shanghai
                'XSHE': 'Asia/Shanghai',     # Shenzhen
                'XHKG': 'Asia/Hong_Kong',   # Hong Kong
                'XLON': 'Europe/London',    # London
                'XPAR': 'Europe/Paris',     # Paris
                'XFRA': 'Europe/Berlin',    # Frankfurt
                'XTKS': 'Asia/Tokyo'        # Tokyo
            }

        if self.custom_holidays is None:
            self.custom_holidays = []

        if self.market_hours is None:
            self.market_hours = {
                'XNYS': {
                    'pre_market': {'start': '04:00', 'end': '09:30'},
                    'regular': {'start': '09:30', 'end': '16:00'},
                    'post_market': {'start': '16:00', 'end': '20:00'}
                },
                'XNAS': {
                    'pre_market': {'start': '04:00', 'end': '09:30'},
                    'regular': {'start': '09:30', 'end': '16:00'},
                    'post_market': {'start': '16:00', 'end': '20:00'}
                },
                'XSHG': {
                    'pre_market': {'start': '09:00', 'end': '09:30'},
                    'regular_morning': {'start': '09:30', 'end': '11:30'},
                    'lunch_break': {'start': '11:30', 'end': '13:00'},
                    'regular_afternoon': {'start': '13:00', 'end': '15:00'}
                }
            }

        if self.suspension_rules is None:
            self.suspension_rules = {
                'circuit_breaker_levels': [7, 13, 20],  # Percentage drops
                'volatility_threshold': 10,  # Percentage for single stock
                'volume_threshold': 0.1      # Minimum volume ratio
            }


class TradingCalendarManager:
    """
    Manages trading calendars for multiple markets.
    """

    def __init__(self, config: Optional[TradingCalendarConfig] = None):
        """
        Initialize trading calendar manager.

        Args:
            config: Calendar configuration
        """
        self.config = config or TradingCalendarConfig()

        # Market calendars cache
        self.market_calendars: Dict[str, Any] = {}
        self.holiday_cache: Dict[str, List[date]] = {}
        self.session_cache: Dict[str, List[TradingSession]] = {}

        # Initialize market calendars
        self._initialize_market_calendars()

        logger.info(f"[trading_calendar] Manager initialized for {len(self.market_calendars)} markets")

    def _initialize_market_calendars(self):
        """Initialize pandas market calendars if available."""
        try:
            if MCAL_AVAILABLE:
                # Initialize common market calendars
                common_markets = ['XNYS', 'XNAS', 'XSHG', 'XHKG', 'XLON']

                for market in common_markets:
                    try:
                        calendar = mcal.get_calendar(market)
                        self.market_calendars[market] = calendar
                        logger.debug(f"[trading_calendar] Initialized calendar for {market}")
                    except Exception as e:
                        logger.warning(f"[trading_calendar] Failed to initialize {market} calendar: {e}")

                logger.info(f"[trading_calendar] Initialized {len(self.market_calendars)} market calendars")
            else:
                logger.warning("[trading_calendar] Using basic calendar implementation")

        except Exception as e:
            logger.error(f"[trading_calendar] Calendar initialization failed: {e}")

    def is_trading_day(self, date_input: Union[str, date, datetime],
                      market: str = None) -> bool:
        """
        Check if a date is a trading day for the specified market.

        Args:
            date_input: Date to check
            market: Market code (default: config default)

        Returns:
            True if trading day
        """
        try:
            market = market or self.config.default_market
            check_date = self._normalize_date(date_input)

            # Check weekends first
            if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
                return False

            # Check holidays
            holidays = self._get_market_holidays(market, check_date.year)
            if check_date in holidays:
                return False

            # Use pandas market calendar if available
            if market in self.market_calendars and MCAL_AVAILABLE:
                calendar = self.market_calendars[market]
                trading_days = calendar.valid_days(start_date=check_date, end_date=check_date)
                return len(trading_days) > 0

            return True

        except Exception as e:
            logger.error(f"[trading_calendar] Trading day check failed: {e}")
            return False

    def get_trading_days(self, start_date: Union[str, date, datetime],
                        end_date: Union[str, date, datetime],
                        market: str = None) -> List[date]:
        """
        Get list of trading days between two dates.

        Args:
            start_date: Start date
            end_date: End date
            market: Market code

        Returns:
            List of trading days
        """
        try:
            market = market or self.config.default_market
            start = self._normalize_date(start_date)
            end = self._normalize_date(end_date)

            if market in self.market_calendars and MCAL_AVAILABLE:
                calendar = self.market_calendars[market]
                trading_days = calendar.valid_days(start_date=start, end_date=end)
                return [day.date() for day in trading_days]

            # Fallback: manual calculation
            trading_days = []
            current_date = start

            while current_date <= end:
                if self.is_trading_day(current_date, market):
                    trading_days.append(current_date)
                current_date += timedelta(days=1)

            return trading_days

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to get trading days: {e}")
            return []

    def get_next_trading_day(self, date_input: Union[str, date, datetime],
                           market: str = None) -> Optional[date]:
        """
        Get next trading day after the given date.

        Args:
            date_input: Reference date
            market: Market code

        Returns:
            Next trading day
        """
        try:
            market = market or self.config.default_market
            current_date = self._normalize_date(date_input) + timedelta(days=1)

            for _ in range(10):  # Look ahead up to 10 days
                if self.is_trading_day(current_date, market):
                    return current_date
                current_date += timedelta(days=1)

            return None

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to get next trading day: {e}")
            return None

    def get_previous_trading_day(self, date_input: Union[str, date, datetime],
                               market: str = None) -> Optional[date]:
        """
        Get previous trading day before the given date.

        Args:
            date_input: Reference date
            market: Market code

        Returns:
            Previous trading day
        """
        try:
            market = market or self.config.default_market
            current_date = self._normalize_date(date_input) - timedelta(days=1)

            for _ in range(10):  # Look back up to 10 days
                if self.is_trading_day(current_date, market):
                    return current_date
                current_date -= timedelta(days=1)

            return None

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to get previous trading day: {e}")
            return None

    def get_trading_sessions(self, date_input: Union[str, date, datetime],
                           market: str = None) -> List[TradingSession]:
        """
        Get trading sessions for a specific date and market.

        Args:
            date_input: Date to get sessions for
            market: Market code

        Returns:
            List of trading sessions
        """
        try:
            market = market or self.config.default_market
            session_date = self._normalize_date(date_input)

            # Check cache first
            cache_key = f"{market}_{session_date.isoformat()}"
            if cache_key in self.session_cache:
                return self.session_cache[cache_key]

            sessions = []

            if not self.is_trading_day(session_date, market):
                return sessions

            # Get market timezone
            timezone = pytz.timezone(self.config.timezone_mapping.get(market, 'UTC'))

            # Get market hours configuration
            market_hours = self.config.market_hours.get(market, {})

            for session_name, hours in market_hours.items():
                if 'start' in hours and 'end' in hours:
                    # Create session datetime objects
                    start_time = timezone.localize(
                        datetime.combine(session_date, time.fromisoformat(hours['start']))
                    )
                    end_time = timezone.localize(
                        datetime.combine(session_date, time.fromisoformat(hours['end']))
                    )

                    # Determine session type
                    if 'pre' in session_name.lower():
                        session_type = SessionType.PRE_MARKET
                    elif 'post' in session_name.lower():
                        session_type = SessionType.POST_MARKET
                    else:
                        session_type = SessionType.REGULAR

                    session = TradingSession(
                        market=market,
                        date=session_date.isoformat(),
                        session_type=session_type,
                        start_time=start_time.isoformat(),
                        end_time=end_time.isoformat(),
                        status=MarketStatus.OPEN
                    )

                    sessions.append(session)

            # Handle special cases (early close, late open)
            sessions = self._apply_special_session_rules(sessions, session_date, market)

            # Cache the result
            self.session_cache[cache_key] = sessions

            return sessions

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to get trading sessions: {e}")
            return []

    def get_market_status(self, datetime_input: Union[str, datetime],
                         market: str = None) -> Tuple[MarketStatus, Optional[TradingSession]]:
        """
        Get market status at a specific datetime.

        Args:
            datetime_input: Datetime to check
            market: Market code

        Returns:
            Market status and current session (if any)
        """
        try:
            market = market or self.config.default_market
            check_datetime = self._normalize_datetime(datetime_input)

            # Get sessions for the date
            sessions = self.get_trading_sessions(check_datetime.date(), market)

            if not sessions:
                return MarketStatus.CLOSED, None

            # Check which session we're in
            for session in sessions:
                session_start = datetime.fromisoformat(session.start_time)
                session_end = datetime.fromisoformat(session.end_time)

                if session_start <= check_datetime <= session_end:
                    return session.status, session

            # Not in any session
            return MarketStatus.CLOSED, None

        except Exception as e:
            logger.error(f"[trading_calendar] Market status check failed: {e}")
            return MarketStatus.UNKNOWN, None

    def add_custom_holiday(self, holiday: MarketHoliday):
        """Add a custom holiday to the calendar."""
        try:
            self.config.custom_holidays.append(holiday)

            # Clear relevant caches
            self._clear_holiday_cache(holiday.market)

            logger.info(f"[trading_calendar] Added custom holiday: {holiday.name} on {holiday.date}")

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to add custom holiday: {e}")

    def add_market_suspension(self, market: str, start_datetime: datetime,
                            end_datetime: datetime, reason: str = ""):
        """
        Add a market suspension period.

        Args:
            market: Market code
            start_datetime: Suspension start
            end_datetime: Suspension end
            reason: Suspension reason
        """
        try:
            # Implementation would track suspensions in a database or file
            # For now, we'll just log the suspension
            logger.info(f"[trading_calendar] Market suspension: {market} from {start_datetime} to {end_datetime} - {reason}")

            # Clear session cache for affected dates
            current_date = start_datetime.date()
            end_date = end_datetime.date()

            while current_date <= end_date:
                cache_key = f"{market}_{current_date.isoformat()}"
                if cache_key in self.session_cache:
                    del self.session_cache[cache_key]
                current_date += timedelta(days=1)

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to add market suspension: {e}")

    def _get_market_holidays(self, market: str, year: int) -> List[date]:
        """Get holidays for a specific market and year."""
        try:
            cache_key = f"{market}_{year}"
            if cache_key in self.holiday_cache:
                return self.holiday_cache[cache_key]

            holidays = []

            # Use pandas market calendar if available
            if market in self.market_calendars and MCAL_AVAILABLE:
                calendar = self.market_calendars[market]
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"

                try:
                    market_holidays = calendar.holidays().holidays
                    year_holidays = [h.date() for h in market_holidays if h.year == year]
                    holidays.extend(year_holidays)
                except Exception as e:
                    logger.debug(f"[trading_calendar] Failed to get holidays from market calendar: {e}")

            # Add custom holidays
            for holiday in self.config.custom_holidays:
                if holiday.market == market:
                    holiday_date = datetime.fromisoformat(holiday.date).date()
                    if holiday_date.year == year:
                        holidays.append(holiday_date)

            # Remove duplicates and sort
            holidays = sorted(list(set(holidays)))

            # Cache the result
            self.holiday_cache[cache_key] = holidays

            return holidays

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to get market holidays: {e}")
            return []

    def _apply_special_session_rules(self, sessions: List[TradingSession],
                                   session_date: date, market: str) -> List[TradingSession]:
        """Apply special rules for early close, late open, etc."""
        try:
            # Check for custom holidays with special hours
            for holiday in self.config.custom_holidays:
                if (holiday.market == market and
                    datetime.fromisoformat(holiday.date).date() == session_date):

                    if holiday.holiday_type == "early_close" and holiday.early_close_time:
                        # Modify regular session end time
                        for session in sessions:
                            if session.session_type == SessionType.REGULAR:
                                # Update end time
                                timezone = pytz.timezone(self.config.timezone_mapping.get(market, 'UTC'))
                                new_end = timezone.localize(
                                    datetime.combine(session_date, time.fromisoformat(holiday.early_close_time))
                                )
                                session.end_time = new_end.isoformat()
                                session.status = MarketStatus.EARLY_CLOSE

                    elif holiday.holiday_type == "late_open" and holiday.late_open_time:
                        # Modify regular session start time
                        for session in sessions:
                            if session.session_type == SessionType.REGULAR:
                                timezone = pytz.timezone(self.config.timezone_mapping.get(market, 'UTC'))
                                new_start = timezone.localize(
                                    datetime.combine(session_date, time.fromisoformat(holiday.late_open_time))
                                )
                                session.start_time = new_start.isoformat()
                                session.status = MarketStatus.LATE_OPEN

            return sessions

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to apply special session rules: {e}")
            return sessions

    def _normalize_date(self, date_input: Union[str, date, datetime]) -> date:
        """Normalize various date inputs to date object."""
        if isinstance(date_input, str):
            return datetime.fromisoformat(date_input).date()
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, date):
            return date_input
        else:
            raise ValueError(f"Invalid date input: {date_input}")

    def _normalize_datetime(self, datetime_input: Union[str, datetime]) -> datetime:
        """Normalize various datetime inputs to datetime object."""
        if isinstance(datetime_input, str):
            return datetime.fromisoformat(datetime_input)
        elif isinstance(datetime_input, datetime):
            return datetime_input
        else:
            raise ValueError(f"Invalid datetime input: {datetime_input}")

    def _clear_holiday_cache(self, market: str):
        """Clear holiday cache for a specific market."""
        keys_to_remove = [key for key in self.holiday_cache.keys() if key.startswith(f"{market}_")]
        for key in keys_to_remove:
            del self.holiday_cache[key]

    def get_calendar_summary(self, market: str = None) -> Dict[str, Any]:
        """Get summary of calendar information."""
        try:
            market = market or self.config.default_market

            # Get current year holidays
            current_year = datetime.now().year
            holidays = self._get_market_holidays(market, current_year)

            # Next few trading days
            today = date.today()
            next_trading_days = []
            current_date = today
            for _ in range(5):
                current_date = self.get_next_trading_day(current_date, market)
                if current_date:
                    next_trading_days.append(current_date.isoformat())
                else:
                    break

            # Market status today
            now = datetime.now()
            if market in self.config.timezone_mapping:
                market_tz = pytz.timezone(self.config.timezone_mapping[market])
                now = now.astimezone(market_tz)

            market_status, current_session = self.get_market_status(now, market)

            return {
                'market': market,
                'timezone': self.config.timezone_mapping.get(market, 'UTC'),
                'current_status': market_status.value,
                'current_session': asdict(current_session) if current_session else None,
                'is_trading_day_today': self.is_trading_day(today, market),
                'holidays_this_year': len(holidays),
                'next_trading_days': next_trading_days,
                'market_hours': self.config.market_hours.get(market, {}),
                'cache_stats': {
                    'holiday_cache_size': len(self.holiday_cache),
                    'session_cache_size': len(self.session_cache)
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[trading_calendar] Failed to get calendar summary: {e}")
            return {'error': str(e)}


def create_trading_calendar_manager(custom_config: Optional[Dict] = None) -> TradingCalendarManager:
    """
    Create and configure a trading calendar manager.

    Args:
        custom_config: Custom configuration parameters

    Returns:
        Configured TradingCalendarManager instance
    """
    config = TradingCalendarConfig()

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return TradingCalendarManager(config)


if __name__ == "__main__":
    # Test trading calendar functionality
    print("=== Trading Calendar Manager Test ===")

    # Create manager
    manager = create_trading_calendar_manager()

    # Test basic functionality
    today = date.today()
    print(f"Is today ({today}) a trading day? {manager.is_trading_day(today)}")

    # Next trading day
    next_day = manager.get_next_trading_day(today)
    print(f"Next trading day: {next_day}")

    # Trading days this week
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    trading_days = manager.get_trading_days(week_start, week_end)
    print(f"Trading days this week: {len(trading_days)}")

    # Market status
    now = datetime.now()
    status, session = manager.get_market_status(now)
    print(f"Current market status: {status.value}")
    if session:
        print(f"Current session: {session.session_type.value}")

    # Add custom holiday
    custom_holiday = MarketHoliday(
        name="Test Holiday",
        date=(today + timedelta(days=30)).isoformat(),
        market="XNYS",
        holiday_type="full_day"
    )
    manager.add_custom_holiday(custom_holiday)

    # Calendar summary
    summary = manager.get_calendar_summary()
    print(f"Calendar summary: {summary['market']} - {summary['timezone']}")
    print(f"Holidays this year: {summary['holidays_this_year']}")