"""
Simple Market Data Provider
Emergency fallback for getting basic stock prices when Tiger API has limited permissions.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import time

def get_current_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
    """
    Get current market prices for a list of symbols using Yahoo Finance.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dictionary mapping symbols to their current prices
    """
    prices = {}
    
    for symbol in symbols:
        try:
            # Use yfinance to get current price
            ticker = yf.Ticker(symbol)
            
            # Get recent data (last 2 days to ensure we have current price)
            data = ticker.history(period='2d', interval='1m')
            
            if data is not None and len(data) > 0:
                # Get the most recent closing price
                current_price = float(data['Close'].iloc[-1])
                prices[symbol] = current_price
                print(f"[MARKET_DATA] {symbol}: ${current_price:.2f}")
            else:
                print(f"[MARKET_DATA] {symbol}: No data available")
                prices[symbol] = None
                
        except Exception as e:
            print(f"[MARKET_DATA] {symbol}: Error - {e}")
            prices[symbol] = None
            
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return prices

def estimate_position_sizes(symbols: List[str], available_capital: float) -> Dict[str, dict]:
    """
    Calculate position sizes based on current prices and available capital.
    
    Args:
        symbols: List of stock symbols to trade
        available_capital: Total available capital
        
    Returns:
        Dictionary with symbol -> {price, quantity, value} mapping
    """
    prices = get_current_prices(symbols)
    
    # Filter out symbols with no price data
    valid_symbols = [s for s in symbols if prices.get(s) is not None]
    
    if not valid_symbols:
        return {}
    
    # Equal allocation across all valid symbols
    capital_per_stock = available_capital / len(valid_symbols)
    
    positions = {}
    for symbol in valid_symbols:
        price = prices[symbol]
        if price and price > 0:
            # Calculate quantity (round down to whole shares)
            quantity = int(capital_per_stock / price)
            value = quantity * price
            
            positions[symbol] = {
                'price': price,
                'quantity': quantity,
                'value': value,
                'allocation_pct': (value / available_capital) * 100
            }
    
    return positions

def create_simple_ohlcv_data(symbol: str, price: float) -> pd.DataFrame:
    """
    Create a simple OHLCV DataFrame for a symbol with current price.
    This is used when we only have current price but need OHLCV format.
    """
    from datetime import datetime
    
    # Create a minimal OHLCV row with current price
    data = {
        'open': [price],
        'high': [price * 1.01],  # Slight variation for realism
        'low': [price * 0.99],
        'close': [price],
        'volume': [100000],  # Default volume
        'time': [datetime.now()]
    }
    
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    
    return df