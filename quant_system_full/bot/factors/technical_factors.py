"""
Technical indicator factors for quantitative trading system.

This module implements various technical analysis indicators including:
- MACD, Bollinger Bands, KDJ and other technical indicators
- Breakout signals, support/resistance level identification  
- Moving averages convergence/divergence signals
- Technical pattern recognition

All factor calculations are vectorized using pandas/numpy for performance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


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


def compute_macd(close: pd.Series, fast_period: int = 12, 
                slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Close price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period  
        signal_period: Signal line EMA period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = close.ewm(span=fast_period).mean()
    slow_ema = close.ewm(span=slow_period).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close: pd.Series, period: int = 20, 
                           std_multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Args:
        close: Close price series
        period: Moving average period
        std_multiplier: Standard deviation multiplier
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle_band = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    
    upper_band = middle_band + (std_dev * std_multiplier)
    lower_band = middle_band - (std_dev * std_multiplier)
    
    return upper_band, middle_band, lower_band


def compute_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute KDJ indicator (Enhanced Stochastic).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: K line period
        d_period: D line smoothing period
        j_period: J line smoothing period
        
    Returns:
        Tuple of (K, D, J) series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    rsv = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
    rsv = rsv.fillna(50)
    
    # K and D are smoothed versions
    k = rsv.ewm(alpha=1/d_period).mean()
    d = k.ewm(alpha=1/j_period).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period
        
    Returns:
        ATR series
    """
    high_low = high - low
    high_close_prev = (high - close.shift(1)).abs()
    low_close_prev = (low - close.shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr.fillna(0)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute ADX (Average Directional Index) and directional indicators.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period
        
    Returns:
        Tuple of (ADX, +DI, -DI) series
    """
    # True Range
    atr = compute_atr(high, low, close, period)
    
    # Directional Movement
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / (atr + 1e-9))
    minus_di = 100 * (minus_dm_smooth / (atr + 1e-9))
    
    # ADX calculation
    dx = 100 * (((plus_di - minus_di).abs()) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def compute_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series,
                              window: int = 20, threshold: float = 0.02) -> Dict[str, pd.Series]:
    """
    Identify support and resistance levels.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Window for identifying levels
        threshold: Threshold for level significance
        
    Returns:
        Dictionary with support/resistance levels and distances
    """
    # Find local maxima and minima
    high_rolling_max = high.rolling(window=window, center=True).max()
    low_rolling_min = low.rolling(window=window, center=True).min()
    
    resistance_levels = high.where(high == high_rolling_max).dropna()
    support_levels = low.where(low == low_rolling_min).dropna()
    
    # Calculate distances to nearest support/resistance
    current_price = close.iloc[-1] if len(close) > 0 else 0
    
    # Distance to resistance (negative if above resistance)
    resistance_distance = pd.Series(index=close.index, dtype=float)
    if len(resistance_levels) > 0:
        nearest_resistance = resistance_levels[resistance_levels >= current_price * (1 - threshold)]
        if len(nearest_resistance) > 0:
            resistance_distance.iloc[-1] = (nearest_resistance.min() - current_price) / current_price
        else:
            resistance_distance.iloc[-1] = threshold  # Far from resistance
    
    # Distance to support (positive if above support)  
    support_distance = pd.Series(index=close.index, dtype=float)
    if len(support_levels) > 0:
        nearest_support = support_levels[support_levels <= current_price * (1 + threshold)]
        if len(nearest_support) > 0:
            support_distance.iloc[-1] = (current_price - nearest_support.max()) / current_price
        else:
            support_distance.iloc[-1] = threshold  # Far from support
    
    return {
        'resistance_distance': resistance_distance.fillna(0),
        'support_distance': support_distance.fillna(0),
        'resistance_levels': resistance_levels,
        'support_levels': support_levels
    }


def compute_breakout_signals(high: pd.Series, low: pd.Series, close: pd.Series,
                           volume: pd.Series, period: int = 20,
                           volume_threshold: float = 1.5) -> Dict[str, pd.Series]:
    """
    Detect breakout signals based on price and volume.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Lookback period for breakout calculation
        volume_threshold: Volume threshold for confirming breakouts
        
    Returns:
        Dictionary with breakout signals
    """
    # Calculate rolling statistics
    high_max = high.rolling(window=period).max()
    low_min = low.rolling(window=period).min()
    volume_avg = volume.rolling(window=period).mean()
    
    # Price breakouts
    upward_breakout = (close > high_max.shift(1)) & (volume > volume_avg * volume_threshold)
    downward_breakout = (close < low_min.shift(1)) & (volume > volume_avg * volume_threshold)
    
    # Breakout strength (how far beyond the breakout level)
    upward_strength = ((close - high_max.shift(1)) / high_max.shift(1)).fillna(0)
    downward_strength = ((low_min.shift(1) - close) / low_min.shift(1)).fillna(0)
    
    return {
        'upward_breakout': upward_breakout.astype(int),
        'downward_breakout': downward_breakout.astype(int),
        'upward_strength': upward_strength.clip(0, 1),
        'downward_strength': downward_strength.clip(0, 1)
    }


def compute_moving_average_signals(close: pd.Series, 
                                 periods: List[int] = [5, 10, 20, 50, 200]) -> Dict[str, pd.Series]:
    """
    Compute moving average convergence/divergence signals.
    
    Args:
        close: Close price series
        periods: List of MA periods to compute
        
    Returns:
        Dictionary with MA signals and relationships
    """
    mas = {}
    for period in periods:
        mas[f'ma_{period}'] = close.rolling(window=period).mean()
    
    signals = {}
    
    # Price relative to MAs
    for period in periods:
        ma_key = f'ma_{period}'
        signals[f'price_above_{ma_key}'] = (close > mas[ma_key]).astype(int)
        signals[f'price_distance_{ma_key}'] = ((close - mas[ma_key]) / mas[ma_key]).fillna(0)
    
    # MA crossover signals (golden cross, death cross)
    if len(periods) >= 2:
        short_ma = mas[f'ma_{min(periods)}']
        long_ma = mas[f'ma_{max(periods)}']
        
        signals['golden_cross'] = ((short_ma > long_ma) & 
                                 (short_ma.shift(1) <= long_ma.shift(1))).astype(int)
        signals['death_cross'] = ((short_ma < long_ma) & 
                                (short_ma.shift(1) >= long_ma.shift(1))).astype(int)
        
        signals['ma_trend'] = ((short_ma - long_ma) / long_ma).fillna(0)
    
    # MA alignment (all MAs in ascending/descending order)
    if len(periods) >= 3:
        ma_values = [mas[f'ma_{p}'] for p in sorted(periods)]
        
        bullish_alignment = pd.Series(True, index=close.index)
        bearish_alignment = pd.Series(True, index=close.index)
        
        for i in range(len(ma_values) - 1):
            bullish_alignment &= (ma_values[i] > ma_values[i + 1])
            bearish_alignment &= (ma_values[i] < ma_values[i + 1])
        
        signals['bullish_ma_alignment'] = bullish_alignment.astype(int)
        signals['bearish_ma_alignment'] = bearish_alignment.astype(int)
    
    return signals


def detect_chart_patterns(high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 10) -> Dict[str, pd.Series]:
    """
    Simple chart pattern detection using basic heuristics.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        window: Pattern detection window
        
    Returns:
        Dictionary with pattern signals
    """
    patterns = {}
    
    # Double top pattern (simplified)
    high_peaks = high.rolling(window=window, center=True).max() == high
    recent_peaks = high.where(high_peaks).dropna().tail(3)
    
    if len(recent_peaks) >= 2:
        double_top = abs(recent_peaks.iloc[-1] - recent_peaks.iloc[-2]) / recent_peaks.iloc[-1] < 0.02
        patterns['double_top'] = pd.Series(0, index=close.index)
        if double_top:
            patterns['double_top'].iloc[-1] = 1
    else:
        patterns['double_top'] = pd.Series(0, index=close.index)
    
    # Double bottom pattern (simplified)
    low_troughs = low.rolling(window=window, center=True).min() == low
    recent_troughs = low.where(low_troughs).dropna().tail(3)
    
    if len(recent_troughs) >= 2:
        double_bottom = abs(recent_troughs.iloc[-1] - recent_troughs.iloc[-2]) / recent_troughs.iloc[-1] < 0.02
        patterns['double_bottom'] = pd.Series(0, index=close.index)
        if double_bottom:
            patterns['double_bottom'].iloc[-1] = 1
    else:
        patterns['double_bottom'] = pd.Series(0, index=close.index)
    
    # Head and shoulders (very simplified)
    if len(recent_peaks) >= 3:
        left_shoulder = recent_peaks.iloc[-3]
        head = recent_peaks.iloc[-2]  
        right_shoulder = recent_peaks.iloc[-1]
        
        head_shoulders = (head > left_shoulder and head > right_shoulder and
                         abs(left_shoulder - right_shoulder) / left_shoulder < 0.05)
        patterns['head_shoulders'] = pd.Series(0, index=close.index)
        if head_shoulders:
            patterns['head_shoulders'].iloc[-1] = 1
    else:
        patterns['head_shoulders'] = pd.Series(0, index=close.index)
    
    return patterns


def technical_features(df: pd.DataFrame,
                      macd_fast: int = 12,
                      macd_slow: int = 26, 
                      macd_signal: int = 9,
                      bb_period: int = 20,
                      bb_std: float = 2,
                      kdj_period: int = 9,
                      atr_period: int = 14,
                      adx_period: int = 14,
                      sr_window: int = 20,
                      breakout_period: int = 20,
                      ma_periods: List[int] = None) -> pd.DataFrame:
    """
    Calculate comprehensive technical analysis features for a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        bb_period: Bollinger Bands period
        bb_std: Bollinger Bands standard deviation
        kdj_period: KDJ period
        atr_period: ATR period
        adx_period: ADX period
        sr_window: Support/resistance window
        breakout_period: Breakout detection period
        ma_periods: Moving average periods
        
    Returns:
        DataFrame with technical features added
    """
    if ma_periods is None:
        ma_periods = [5, 10, 20, 50]
        
    result = df.copy()
    
    # MACD
    macd_line, signal_line, histogram = compute_macd(
        df['close'], macd_fast, macd_slow, macd_signal
    )
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(
        df['close'], bb_period, bb_std
    )
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    result['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # KDJ
    k, d, j = compute_kdj(df['high'], df['low'], df['close'], kdj_period)
    result['kdj_k'] = k
    result['kdj_d'] = d  
    result['kdj_j'] = j
    
    # ATR and volatility measures
    result['atr'] = compute_atr(df['high'], df['low'], df['close'], atr_period)
    result['volatility'] = result['atr'] / df['close']
    
    # ADX and trend strength
    adx, plus_di, minus_di = compute_adx(df['high'], df['low'], df['close'], adx_period)
    result['adx'] = adx
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    result['trend_strength'] = adx / 100  # Normalized
    
    # Support/Resistance
    sr_data = compute_support_resistance(df['high'], df['low'], df['close'], sr_window)
    result['resistance_distance'] = sr_data['resistance_distance']
    result['support_distance'] = sr_data['support_distance']
    
    # Breakout signals
    breakout_data = compute_breakout_signals(
        df['high'], df['low'], df['close'], df['volume'], breakout_period
    )
    result['upward_breakout'] = breakout_data['upward_breakout']
    result['downward_breakout'] = breakout_data['downward_breakout']
    result['breakout_strength'] = (breakout_data['upward_strength'] - 
                                  breakout_data['downward_strength'])
    
    # Moving average signals
    ma_signals = compute_moving_average_signals(df['close'], ma_periods)
    for key, value in ma_signals.items():
        result[f'ma_{key}'] = value
    
    # Chart patterns
    pattern_data = detect_chart_patterns(df['high'], df['low'], df['close'])
    for key, value in pattern_data.items():
        result[f'pattern_{key}'] = value
    
    # Composite technical score
    technical_components = [
        'macd_histogram', 'bb_position', 'kdj_j', 'trend_strength',
        'breakout_strength', 'resistance_distance', 'support_distance'
    ]
    
    # Normalize components
    normalized_components = []
    for component in technical_components:
        if component in result.columns:
            normalized = _zscore(result[component])
            normalized_components.append(normalized)
    
    if normalized_components:
        component_df = pd.concat(normalized_components, axis=1)
        valid_count = component_df.notna().sum(axis=1)
        component_sum = component_df.fillna(0).sum(axis=1)
        result['technical_score'] = (component_sum / valid_count.replace(0, np.nan)).fillna(0)
    else:
        result['technical_score'] = 0
    
    return result


def cross_section_technical_score(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional technical scores for multiple symbols.
    
    Args:
        df_latest: DataFrame with latest technical scores for multiple symbols
        
    Returns:
        DataFrame with normalized TechnicalScore
    """
    if 'technical_score' not in df_latest.columns:
        raise ValueError("DataFrame must contain 'technical_score' column")
        
    result = df_latest[['symbol', 'technical_score']].copy()
    result['TechnicalScore'] = _zscore(result['technical_score'])
    
    return result[['symbol', 'TechnicalScore']]


def get_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive trading signals based on technical indicators.
    
    Args:
        df: DataFrame with technical features
        
    Returns:
        DataFrame with technical-based signals
    """
    result = df.copy()
    
    # Initialize signal
    result['technical_signal'] = 0
    
    # MACD signals
    macd_bullish = (result['macd'] > result['macd_signal']) & (result['macd_histogram'] > 0)
    macd_bearish = (result['macd'] < result['macd_signal']) & (result['macd_histogram'] < 0)
    
    # Bollinger Band signals
    bb_oversold = result['bb_position'] < 0.2
    bb_overbought = result['bb_position'] > 0.8
    
    # KDJ signals
    kdj_bullish = (result['kdj_k'] > result['kdj_d']) & (result['kdj_k'] < 80)
    kdj_bearish = (result['kdj_k'] < result['kdj_d']) & (result['kdj_k'] > 20)
    
    # ADX trend signals
    strong_trend = result['adx'] > 25
    bullish_trend = strong_trend & (result['plus_di'] > result['minus_di'])
    bearish_trend = strong_trend & (result['plus_di'] < result['minus_di'])
    
    # Breakout signals  
    breakout_bullish = result['upward_breakout'] == 1
    breakout_bearish = result['downward_breakout'] == 1
    
    # Combine signals with weights
    bullish_score = (
        macd_bullish.astype(int) * 0.2 +
        bb_oversold.astype(int) * 0.15 +
        kdj_bullish.astype(int) * 0.2 +
        bullish_trend.astype(int) * 0.25 +
        breakout_bullish.astype(int) * 0.2
    )
    
    bearish_score = (
        macd_bearish.astype(int) * 0.2 +
        bb_overbought.astype(int) * 0.15 +
        kdj_bearish.astype(int) * 0.2 +
        bearish_trend.astype(int) * 0.25 +
        breakout_bearish.astype(int) * 0.2
    )
    
    # Generate final signal
    result.loc[bullish_score > 0.5, 'technical_signal'] = 1
    result.loc[bearish_score > 0.5, 'technical_signal'] = -1
    
    return result[['symbol', 'technical_signal']] if 'symbol' in result.columns else result[['technical_signal']]


# Backwards compatibility
def technical_score(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Legacy function for backwards compatibility."""
    return technical_features(df, **kwargs)