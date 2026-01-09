"""
Momentum factors for quantitative trading system.

This module implements various momentum-based factors including:
- Price momentum and volume momentum calculations  
- Relative Strength Index (RSI), Momentum Oscillator (MOM)
- Rate of Change (ROC) and other momentum indicators
- Support for different lookback periods

All factor calculations are vectorized using pandas/numpy for performance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


def _zscore(s: pd.Series) -> pd.Series:
    """
    Compute z-score normalization for a series.
    
    Args:
        s: Input series
        
    Returns:
        Z-score normalized series
    """
    s = pd.to_numeric(s, errors='coerce')
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        close: Close price series
        period: RSI calculation period (default: 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # Use Wilder's smoothing (exponential moving average with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral value


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Compute Momentum Oscillator (MOM).
    
    Args:
        close: Close price series
        period: Lookback period (default: 10)
        
    Returns:
        Momentum values
    """
    return ((close - close.shift(period)) / close.shift(period) * 100).fillna(0)


def compute_roc(close: pd.Series, period: int = 12) -> pd.Series:
    """
    Compute Rate of Change (ROC).
    
    Args:
        close: Close price series  
        period: Lookback period (default: 12)
        
    Returns:
        ROC values as percentage
    """
    roc = ((close - close.shift(period)) / close.shift(period) * 100)
    return roc.fillna(0)


def compute_price_momentum(close: pd.Series, short_period: int = 5, 
                          long_period: int = 20) -> pd.Series:
    """
    Compute price momentum as ratio of short vs long term moving averages.
    
    Args:
        close: Close price series
        short_period: Short-term moving average period
        long_period: Long-term moving average period
        
    Returns:
        Price momentum ratio
    """
    short_ma = close.rolling(window=short_period).mean()
    long_ma = close.rolling(window=long_period).mean()
    
    momentum = (short_ma / long_ma - 1) * 100
    return momentum.fillna(0)


def compute_volume_momentum(volume: pd.Series, period: int = 10) -> pd.Series:
    """
    Compute volume momentum as current vs historical volume.
    
    Args:
        volume: Volume series
        period: Lookback period for average volume
        
    Returns:
        Volume momentum ratio
    """
    avg_volume = volume.rolling(window=period).mean()
    vol_momentum = (volume / avg_volume - 1) * 100
    return vol_momentum.fillna(0)


def compute_stochastic_momentum(high: pd.Series, low: pd.Series, 
                               close: pd.Series, k_period: int = 14,
                               d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Stochastic Momentum Oscillator (%K and %D).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Period for %K calculation
        d_period: Period for %D smoothing
        
    Returns:
        Tuple of (%K, %D) series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
    k_percent = k_percent.fillna(50)
    
    d_percent = k_percent.rolling(window=d_period).mean().fillna(50)
    
    return k_percent, d_percent


def compute_williams_r(high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Williams %R momentum indicator.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        period: Lookback period
        
    Returns:
        Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-9))
    return williams_r.fillna(-50)


def compute_commodity_channel_index(high: pd.Series, low: pd.Series,
                                   close: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Commodity Channel Index (CCI).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series  
        period: Lookback period
        
    Returns:
        CCI values
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    
    cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-9)
    return cci.fillna(0)


def momentum_features(df: pd.DataFrame, 
                     rsi_period: int = 14,
                     mom_period: int = 10,
                     roc_period: int = 12,
                     price_mom_short: int = 5,
                     price_mom_long: int = 20,
                     vol_mom_period: int = 10,
                     stoch_k_period: int = 14,
                     stoch_d_period: int = 3,
                     williams_period: int = 14,
                     cci_period: int = 20) -> pd.DataFrame:
    """
    Calculate comprehensive momentum features for a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI calculation period
        mom_period: Momentum oscillator period
        roc_period: Rate of change period
        price_mom_short: Short-term MA period for price momentum
        price_mom_long: Long-term MA period for price momentum
        vol_mom_period: Volume momentum period
        stoch_k_period: Stochastic %K period
        stoch_d_period: Stochastic %D period
        williams_period: Williams %R period
        cci_period: CCI period
        
    Returns:
        DataFrame with momentum features added
    """
    result = df.copy()
    
    # Basic momentum indicators
    result['rsi'] = compute_rsi(df['close'], rsi_period)
    result['momentum'] = compute_momentum(df['close'], mom_period)
    result['roc'] = compute_roc(df['close'], roc_period)
    
    # Price and volume momentum
    result['price_momentum'] = compute_price_momentum(
        df['close'], price_mom_short, price_mom_long
    )
    result['volume_momentum'] = compute_volume_momentum(df['volume'], vol_mom_period)
    
    # Advanced momentum indicators
    stoch_k, stoch_d = compute_stochastic_momentum(
        df['high'], df['low'], df['close'], stoch_k_period, stoch_d_period
    )
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    
    result['williams_r'] = compute_williams_r(
        df['high'], df['low'], df['close'], williams_period
    )
    result['cci'] = compute_commodity_channel_index(
        df['high'], df['low'], df['close'], cci_period
    )
    
    # Composite momentum score
    momentum_components = [
        'rsi', 'momentum', 'roc', 'price_momentum', 
        'volume_momentum', 'stoch_k', 'williams_r', 'cci'
    ]
    
    # Normalize components before combining
    normalized_components = []
    for component in momentum_components:
        if component in result.columns:
            # Transform RSI and Stochastic to centered around 0
            if component in ['rsi', 'stoch_k']:
                normalized = _zscore(result[component] - 50)
            elif component == 'williams_r':
                normalized = _zscore(result[component] + 50)
            else:
                normalized = _zscore(result[component])
            normalized_components.append(normalized)
    
    if normalized_components:
        component_df = pd.concat(normalized_components, axis=1)
        valid_count = component_df.notna().sum(axis=1)
        component_sum = component_df.fillna(0).sum(axis=1)
        result['momentum_score'] = (component_sum / valid_count.replace(0, np.nan)).fillna(0)
    else:
        result['momentum_score'] = 0
    
    return result


def cross_section_momentum_score(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional momentum scores for multiple symbols.
    
    Args:
        df_latest: DataFrame with latest momentum scores for multiple symbols
        
    Returns:
        DataFrame with normalized MomentumScore
    """
    if 'momentum_score' not in df_latest.columns:
        raise ValueError("DataFrame must contain 'momentum_score' column")
        
    result = df_latest[['symbol', 'momentum_score']].copy()
    result['MomentumScore'] = _zscore(result['momentum_score'])
    
    return result[['symbol', 'MomentumScore']]


def get_momentum_signals(df: pd.DataFrame, 
                        rsi_oversold: float = 30,
                        rsi_overbought: float = 70,
                        momentum_threshold: float = 0) -> pd.DataFrame:
    """
    Generate trading signals based on momentum factors.
    
    Args:
        df: DataFrame with momentum features
        rsi_oversold: RSI oversold threshold
        rsi_overbought: RSI overbought threshold  
        momentum_threshold: Momentum score threshold
        
    Returns:
        DataFrame with momentum-based signals
    """
    result = df.copy()
    
    # Initialize signals
    result['momentum_signal'] = 0
    
    # RSI-based signals
    rsi_buy = result['rsi'] < rsi_oversold
    rsi_sell = result['rsi'] > rsi_overbought
    
    # Momentum score signals
    momentum_buy = result['momentum_score'] > momentum_threshold
    momentum_sell = result['momentum_score'] < -momentum_threshold
    
    # Stochastic signals
    stoch_buy = (result['stoch_k'] < 20) & (result['stoch_d'] < 20)
    stoch_sell = (result['stoch_k'] > 80) & (result['stoch_d'] > 80)
    
    # Combine signals (majority vote)
    buy_signals = rsi_buy.astype(int) + momentum_buy.astype(int) + stoch_buy.astype(int)
    sell_signals = rsi_sell.astype(int) + momentum_sell.astype(int) + stoch_sell.astype(int)
    
    result.loc[buy_signals >= 2, 'momentum_signal'] = 1
    result.loc[sell_signals >= 2, 'momentum_signal'] = -1
    
    return result[['symbol', 'momentum_signal']] if 'symbol' in result.columns else result[['momentum_signal']]


# Backwards compatibility functions
def momentum_score(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Legacy function for backwards compatibility."""
    return momentum_features(df, **kwargs)