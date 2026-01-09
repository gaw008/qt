"""
Yahoo Finance MCP Data Provider

This module provides integration with Yahoo Finance MCP server for market data.
It uses MCP tools when available through Claude Code environment.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

from bot.config import SETTINGS


def fetch_yahoo_mcp_price_history(symbol: str, period: str = 'day', limit: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data using Yahoo Finance MCP server.
    
    This function would be called by Claude Code when MCP tools are available.
    In a standalone environment, it returns None to indicate MCP is not available.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        period: Time period ('day', '1min', '5min', etc.)
        limit: Number of data points to fetch
        
    Returns:
        DataFrame with columns [time, open, high, low, close, volume] or None if MCP unavailable
    """
    if not SETTINGS.use_mcp_tools:
        return None
    
    try:
        # Map internal period format to Yahoo Finance API format
        yahoo_period = _map_period_to_yahoo(period, limit)
        yahoo_interval = _map_interval_to_yahoo(period)
        
        print(f"[mcp_data] Attempting Yahoo Finance MCP: symbol={symbol}, period={yahoo_period}, interval={yahoo_interval}")
        
        # This is where the actual MCP tool call would happen in Claude Code environment
        # The tool call would look something like:
        # result = mcp_tool_call("get-price-history", {
        #     "symbol": symbol,
        #     "period": yahoo_period,
        #     "interval": yahoo_interval
        # })
        
        # For now, return None to indicate MCP is not available in this context
        # When running in Claude Code with MCP enabled, this would be replaced with actual tool calls
        return None
        
    except Exception as e:
        print(f"[mcp_data] Yahoo Finance MCP error: {type(e).__name__}: {e}")
        return None


def fetch_yahoo_mcp_ticker_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive ticker information using Yahoo Finance MCP.
    
    Returns company details, financials, and trading metrics.
    """
    if not SETTINGS.use_mcp_tools:
        return None
        
    try:
        print(f"[mcp_data] Fetching ticker info for {symbol}")
        # MCP tool call placeholder
        # result = mcp_tool_call("get-ticker-info", {"symbol": symbol})
        return None
    except Exception as e:
        print(f"[mcp_data] Ticker info MCP error: {type(e).__name__}: {e}")
        return None


def fetch_yahoo_mcp_news(symbol: str, count: int = 10) -> Optional[list]:
    """
    Fetch recent news articles for a stock symbol using Yahoo Finance MCP.
    """
    if not SETTINGS.use_mcp_tools:
        return None
        
    try:
        print(f"[mcp_data] Fetching news for {symbol}, count={count}")
        # MCP tool call placeholder
        # result = mcp_tool_call("get-ticker-news", {"symbol": symbol, "count": count})
        return None
    except Exception as e:
        print(f"[mcp_data] News MCP error: {type(e).__name__}: {e}")
        return None


def search_yahoo_mcp_symbols(query: str, count: int = 10) -> Optional[list]:
    """
    Search for stocks, ETFs, and other financial instruments using Yahoo Finance MCP.
    """
    if not SETTINGS.use_mcp_tools:
        return None
        
    try:
        print(f"[mcp_data] Searching symbols for query: {query}")
        # MCP tool call placeholder
        # result = mcp_tool_call("search", {"query": query, "count": count})
        return None
    except Exception as e:
        print(f"[mcp_data] Search MCP error: {type(e).__name__}: {e}")
        return None


def _map_period_to_yahoo(period: str, limit: int = 300) -> str:
    """Map internal period format to Yahoo Finance period format."""
    p = (period or '').lower()
    
    # For intraday data, calculate appropriate period based on limit
    if 'min' in p or 'm' in p:
        if limit <= 60:
            return '1d'  # Last day
        elif limit <= 300:
            return '5d'  # Last 5 days
        else:
            return '1mo'  # Last month
    elif 'day' in p:
        if limit <= 30:
            return '1mo'
        elif limit <= 90:
            return '3mo'
        elif limit <= 180:
            return '6mo'
        elif limit <= 365:
            return '1y'
        else:
            return '2y'
    else:
        return '1y'  # Default


def _map_interval_to_yahoo(period: str) -> str:
    """Map internal period format to Yahoo Finance interval format."""
    p = (period or '').lower()
    
    if p in ('1min', '1m', 'min1', 'minute1'):
        return '1m'
    elif p in ('5min', '5m', 'min5', 'minute5'):
        return '5m'
    elif p in ('15min', '15m', 'min15', 'minute15'):
        return '15m'
    elif p in ('30min', '30m', 'min30', 'minute30'):
        return '30m'
    elif p in ('60min', '60m', '1h', 'hour1', 'hourly'):
        return '1h'
    elif 'day' in p:
        return '1d'
    elif 'week' in p:
        return '1wk'
    elif 'month' in p:
        return '1mo'
    else:
        return '1d'  # Default