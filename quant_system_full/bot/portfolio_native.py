import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    """
    Calculates the latest ATR value from a given OHLCV DataFrame using native implementation.

    Args:
        df (pd.DataFrame): DataFrame with columns ['high', 'low', 'close'].
        length (int): The lookback period for ATR. PDF suggests 14.

    Returns:
        float: The most recent ATR value.
    """
    if df is None or len(df) < length:
        return 0.0
    
    df = df.copy()
    
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR using exponential moving average
    alpha = 2.0 / (length + 1)
    df['atr'] = df['true_range'].ewm(alpha=alpha, adjust=False).mean()
    
    # Return the most recent ATR value
    if not df['atr'].empty:
        return df['atr'].iloc[-1]
    return 0.0

def get_position_size(equity: float, price: float, atr: float, 
                        risk_per_trade: float = 0.01, atr_multiplier: float = 2.0) -> int:
    """
    Calculates position size based on the inverse volatility method using ATR.

    Args:
        equity (float): Total current equity of the portfolio.
        price (float): The current price of the asset.
        atr (float): The Average True Range of the asset.
        risk_per_trade (float): The fraction of equity to risk on a single trade (e.g., 0.01 for 1%).
        atr_multiplier (float): The multiplier for ATR to determine the stop loss distance. 
                                (e.g., 2.0 for a 2x ATR stop).

    Returns:
        int: The number of shares to trade. Returns 0 if ATR is zero or price is zero.
    """
    if atr <= 0 or price <= 0:
        return 0

    # 1. How much money are we risking on this trade?
    dollar_risk = equity * risk_per_trade
    
    # 2. How much risk are we taking per share, based on volatility?
    stop_loss_per_share = atr * atr_multiplier
    
    # 3. How many shares can we buy with that risk?
    shares = int(dollar_risk / stop_loss_per_share)
    
    return shares