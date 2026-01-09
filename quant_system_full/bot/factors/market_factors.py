"""
Market sentiment factors for quantitative trading system.

This module implements various market sentiment and macro factors including:
- Market heat index, sector rotation factors
- VIX fear index analysis, fund flow direction analysis
- Market breadth indicators
- Relative performance vs market/sector benchmarks

All factor calculations are vectorized using pandas/numpy for performance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import warnings


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


def compute_market_heat_index(price_data: Dict[str, pd.DataFrame], 
                             volume_data: Dict[str, pd.DataFrame] = None,
                             period: int = 20) -> pd.Series:
    """
    Compute market heat index based on price momentum and volume.
    
    Args:
        price_data: Dictionary of symbol -> price DataFrame
        volume_data: Dictionary of symbol -> volume data (optional)
        period: Calculation period
        
    Returns:
        Market heat index time series
    """
    if not price_data:
        return pd.Series(dtype=float)
    
    # Calculate individual stock momentum
    momentum_scores = []
    volume_scores = []
    
    for symbol, df in price_data.items():
        if df is not None and not df.empty and 'close' in df.columns:
            # Price momentum
            returns = df['close'].pct_change(period).fillna(0)
            momentum_scores.append(returns)
            
            # Volume momentum (if available)
            if volume_data and symbol in volume_data:
                vol_df = volume_data[symbol]
                if vol_df is not None and 'volume' in vol_df.columns:
                    vol_ratio = (vol_df['volume'] / 
                               vol_df['volume'].rolling(period).mean()).fillna(1)
                    volume_scores.append(vol_ratio)
    
    if not momentum_scores:
        return pd.Series(dtype=float)
    
    # Combine momentum scores
    momentum_df = pd.concat(momentum_scores, axis=1)
    avg_momentum = momentum_df.mean(axis=1, skipna=True)
    
    # Combine with volume if available
    if volume_scores:
        volume_df = pd.concat(volume_scores, axis=1)
        avg_volume = volume_df.mean(axis=1, skipna=True)
        
        # Market heat = weighted combination of momentum and volume
        heat_index = 0.7 * avg_momentum + 0.3 * (avg_volume - 1)
    else:
        heat_index = avg_momentum
    
    return heat_index.fillna(0)


def compute_sector_rotation_factor(sector_data: Dict[str, Dict[str, pd.DataFrame]],
                                  benchmark_data: pd.DataFrame = None,
                                  period: int = 20) -> Dict[str, pd.Series]:
    """
    Compute sector rotation factors showing sector strength relative to market.
    
    Args:
        sector_data: Dictionary of sector -> {symbol: DataFrame}
        benchmark_data: Market benchmark data
        period: Calculation period
        
    Returns:
        Dictionary of sector -> rotation factor series
    """
    sector_factors = {}
    
    if benchmark_data is not None and 'close' in benchmark_data.columns:
        benchmark_returns = benchmark_data['close'].pct_change(period).fillna(0)
    else:
        benchmark_returns = None
    
    for sector, stocks in sector_data.items():
        if not stocks:
            continue
            
        # Calculate sector average performance
        sector_returns = []
        for symbol, df in stocks.items():
            if df is not None and not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change(period).fillna(0)
                sector_returns.append(returns)
        
        if sector_returns:
            sector_avg_returns = pd.concat(sector_returns, axis=1).mean(axis=1, skipna=True)
            
            # Relative performance vs benchmark
            if benchmark_returns is not None:
                relative_performance = sector_avg_returns - benchmark_returns
            else:
                relative_performance = sector_avg_returns
            
            sector_factors[sector] = relative_performance.fillna(0)
    
    return sector_factors


def compute_vix_fear_factor(vix_data: pd.DataFrame, 
                           fear_threshold: float = 25,
                           greed_threshold: float = 15) -> pd.Series:
    """
    Compute fear/greed factor based on VIX levels.
    
    Args:
        vix_data: VIX data DataFrame with 'close' column
        fear_threshold: VIX level indicating fear
        greed_threshold: VIX level indicating greed
        
    Returns:
        Fear factor series (-1 to 1, negative = fear, positive = greed)
    """
    if vix_data is None or vix_data.empty or 'close' not in vix_data.columns:
        return pd.Series(dtype=float)
    
    vix_level = vix_data['close']
    
    # Normalize VIX to fear/greed scale
    fear_factor = pd.Series(index=vix_level.index, dtype=float)
    
    # High VIX = Fear (negative values)
    fear_mask = vix_level > fear_threshold
    fear_factor.loc[fear_mask] = -np.clip((vix_level[fear_mask] - fear_threshold) / 25, 0, 1)
    
    # Low VIX = Greed (positive values)
    greed_mask = vix_level < greed_threshold
    fear_factor.loc[greed_mask] = np.clip((greed_threshold - vix_level[greed_mask]) / 10, 0, 1)
    
    # Neutral zone
    neutral_mask = (vix_level >= greed_threshold) & (vix_level <= fear_threshold)
    fear_factor.loc[neutral_mask] = 0
    
    return fear_factor.fillna(0)


def compute_fund_flow_factor(price_data: Dict[str, pd.DataFrame],
                            volume_data: Dict[str, pd.DataFrame],
                            period: int = 10) -> pd.Series:
    """
    Compute fund flow direction based on price-volume relationship.
    
    Args:
        price_data: Dictionary of symbol -> price DataFrame
        volume_data: Dictionary of symbol -> volume DataFrame  
        period: Calculation period
        
    Returns:
        Fund flow factor series
    """
    if not price_data or not volume_data:
        return pd.Series(dtype=float)
    
    flow_scores = []
    
    for symbol in price_data.keys():
        if (symbol in volume_data and 
            price_data[symbol] is not None and 
            volume_data[symbol] is not None):
            
            price_df = price_data[symbol]
            volume_df = volume_data[symbol]
            
            if ('close' in price_df.columns and 
                'volume' in volume_df.columns and
                len(price_df) == len(volume_df)):
                
                # Price change and volume correlation
                price_change = price_df['close'].pct_change().fillna(0)
                volume_change = volume_df['volume'].pct_change().fillna(0)
                
                # Rolling correlation
                correlation = price_change.rolling(period).corr(volume_change).fillna(0)
                
                # On-Balance Volume trend
                obv = np.where(price_change > 0, volume_df['volume'], 
                              np.where(price_change < 0, -volume_df['volume'], 0))
                obv_trend = pd.Series(obv, index=price_df.index).rolling(period).mean()
                obv_normalized = obv_trend / volume_df['volume'].rolling(period).mean()
                
                # Combine correlation and OBV
                flow_score = 0.6 * correlation + 0.4 * obv_normalized.fillna(0)
                flow_scores.append(flow_score)
    
    if flow_scores:
        flow_df = pd.concat(flow_scores, axis=1)
        return flow_df.mean(axis=1, skipna=True).fillna(0)
    else:
        return pd.Series(dtype=float)


def compute_market_breadth(price_data: Dict[str, pd.DataFrame], 
                          period: int = 20) -> Dict[str, pd.Series]:
    """
    Compute market breadth indicators.
    
    Args:
        price_data: Dictionary of symbol -> price DataFrame
        period: Calculation period
        
    Returns:
        Dictionary with breadth indicators
    """
    if not price_data:
        return {}
    
    # Collect daily advances/declines
    advances = []
    declines = []
    new_highs = []
    new_lows = []
    
    for symbol, df in price_data.items():
        if df is not None and not df.empty and 'close' in df.columns:
            # Daily price change
            price_change = df['close'].pct_change().fillna(0)
            advances.append((price_change > 0).astype(int))
            declines.append((price_change < 0).astype(int))
            
            # New highs/lows
            if 'high' in df.columns and 'low' in df.columns:
                rolling_high = df['high'].rolling(period).max()
                rolling_low = df['low'].rolling(period).min()
                
                new_highs.append((df['high'] == rolling_high).astype(int))
                new_lows.append((df['low'] == rolling_low).astype(int))
    
    breadth_indicators = {}
    
    if advances and declines:
        # Advance-Decline Line
        total_advances = pd.concat(advances, axis=1).sum(axis=1, skipna=True)
        total_declines = pd.concat(declines, axis=1).sum(axis=1, skipna=True)
        
        ad_line = (total_advances - total_declines).cumsum()
        breadth_indicators['advance_decline_line'] = ad_line
        
        # Advance-Decline Ratio
        ad_ratio = total_advances / (total_declines + 1e-9)
        breadth_indicators['advance_decline_ratio'] = ad_ratio.fillna(1)
        
        # McClellan Oscillator (simplified)
        ad_diff = total_advances - total_declines
        mcclella_fast = ad_diff.ewm(span=19).mean()
        mcclella_slow = ad_diff.ewm(span=39).mean()
        breadth_indicators['mcclellan_oscillator'] = mcclella_fast - mcclella_slow
    
    if new_highs and new_lows:
        # New Highs - New Lows
        total_new_highs = pd.concat(new_highs, axis=1).sum(axis=1, skipna=True)
        total_new_lows = pd.concat(new_lows, axis=1).sum(axis=1, skipna=True)
        
        breadth_indicators['new_highs_lows'] = total_new_highs - total_new_lows
        
        # High-Low Index
        hl_index = total_new_highs / (total_new_highs + total_new_lows + 1e-9)
        breadth_indicators['high_low_index'] = hl_index.fillna(0.5)
    
    return breadth_indicators


def compute_relative_performance(stock_data: pd.DataFrame,
                                benchmark_data: pd.DataFrame,
                                period: int = 20) -> Dict[str, pd.Series]:
    """
    Compute relative performance metrics vs benchmark.
    
    Args:
        stock_data: Stock price DataFrame
        benchmark_data: Benchmark price DataFrame
        period: Calculation period
        
    Returns:
        Dictionary with relative performance metrics
    """
    if (stock_data is None or benchmark_data is None or
        'close' not in stock_data.columns or 'close' not in benchmark_data.columns):
        return {}
    
    stock_price = stock_data['close']
    benchmark_price = benchmark_data['close']
    
    # Align data
    aligned_data = pd.concat([stock_price, benchmark_price], axis=1, join='inner')
    if aligned_data.empty:
        return {}
    
    stock_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]
    
    metrics = {}
    
    # Relative strength
    relative_strength = stock_aligned / benchmark_aligned
    metrics['relative_strength'] = relative_strength.fillna(1)
    
    # Relative performance (returns)
    stock_returns = stock_aligned.pct_change().fillna(0)
    benchmark_returns = benchmark_aligned.pct_change().fillna(0)
    relative_returns = stock_returns - benchmark_returns
    metrics['relative_returns'] = relative_returns
    
    # Rolling relative performance
    rolling_stock_perf = (stock_aligned / stock_aligned.shift(period) - 1).fillna(0)
    rolling_benchmark_perf = (benchmark_aligned / benchmark_aligned.shift(period) - 1).fillna(0)
    rolling_relative_perf = rolling_stock_perf - rolling_benchmark_perf
    metrics['rolling_relative_performance'] = rolling_relative_perf
    
    # Beta calculation
    if len(stock_returns) >= period:
        rolling_cov = stock_returns.rolling(period).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(period).var()
        beta = rolling_cov / (rolling_var + 1e-9)
        metrics['beta'] = beta.fillna(1)
    
    # Alpha calculation (simplified)
    if 'beta' in metrics:
        alpha = relative_returns - metrics['beta'] * benchmark_returns
        metrics['alpha'] = alpha.fillna(0)
    
    return metrics


def market_sentiment_features(price_data: Dict[str, pd.DataFrame],
                             volume_data: Dict[str, pd.DataFrame] = None,
                             vix_data: pd.DataFrame = None,
                             benchmark_data: pd.DataFrame = None,
                             sector_data: Dict[str, Dict[str, pd.DataFrame]] = None,
                             symbol: str = None,
                             heat_period: int = 20,
                             rotation_period: int = 20,
                             flow_period: int = 10,
                             breadth_period: int = 20,
                             relative_period: int = 20) -> pd.DataFrame:
    """
    Calculate comprehensive market sentiment features.
    
    Args:
        price_data: Dictionary of symbol -> price DataFrame
        volume_data: Dictionary of symbol -> volume DataFrame
        vix_data: VIX data DataFrame
        benchmark_data: Benchmark data DataFrame
        sector_data: Sector-wise stock data
        symbol: Target symbol for relative metrics
        heat_period: Market heat calculation period
        rotation_period: Sector rotation calculation period
        flow_period: Fund flow calculation period
        breadth_period: Market breadth calculation period
        relative_period: Relative performance calculation period
        
    Returns:
        DataFrame with market sentiment features
    """
    # Get base time index
    if symbol and symbol in price_data and price_data[symbol] is not None:
        base_index = price_data[symbol].index
    elif price_data:
        base_index = list(price_data.values())[0].index
    else:
        base_index = pd.Index([])
    
    result = pd.DataFrame(index=base_index)
    
    # Market heat index
    try:
        market_heat = compute_market_heat_index(price_data, volume_data, heat_period)
        if not market_heat.empty:
            result['market_heat'] = market_heat.reindex(base_index, fill_value=0)
        else:
            result['market_heat'] = 0
    except Exception:
        result['market_heat'] = 0
    
    # VIX fear factor
    try:
        if vix_data is not None:
            fear_factor = compute_vix_fear_factor(vix_data)
            if not fear_factor.empty:
                result['fear_factor'] = fear_factor.reindex(base_index, fill_value=0)
            else:
                result['fear_factor'] = 0
        else:
            result['fear_factor'] = 0
    except Exception:
        result['fear_factor'] = 0
    
    # Fund flow factor
    try:
        if volume_data:
            fund_flow = compute_fund_flow_factor(price_data, volume_data, flow_period)
            if not fund_flow.empty:
                result['fund_flow'] = fund_flow.reindex(base_index, fill_value=0)
            else:
                result['fund_flow'] = 0
        else:
            result['fund_flow'] = 0
    except Exception:
        result['fund_flow'] = 0
    
    # Market breadth indicators
    try:
        breadth_indicators = compute_market_breadth(price_data, breadth_period)
        for key, series in breadth_indicators.items():
            if not series.empty:
                result[f'breadth_{key}'] = series.reindex(base_index, fill_value=0)
            else:
                result[f'breadth_{key}'] = 0
    except Exception:
        result['breadth_advance_decline_line'] = 0
        result['breadth_advance_decline_ratio'] = 1
    
    # Sector rotation factors
    try:
        if sector_data:
            sector_factors = compute_sector_rotation_factor(
                sector_data, benchmark_data, rotation_period
            )
            for sector, factor in sector_factors.items():
                if not factor.empty:
                    result[f'sector_{sector}_rotation'] = factor.reindex(base_index, fill_value=0)
    except Exception:
        pass
    
    # Relative performance (for specific symbol)
    try:
        if symbol and symbol in price_data and benchmark_data is not None:
            rel_perf = compute_relative_performance(
                price_data[symbol], benchmark_data, relative_period
            )
            for key, series in rel_perf.items():
                if not series.empty:
                    result[f'relative_{key}'] = series.reindex(base_index, fill_value=0)
    except Exception:
        pass
    
    # Composite market sentiment score
    sentiment_components = [
        'market_heat', 'fear_factor', 'fund_flow'
    ]
    
    # Add breadth components if available
    breadth_cols = [col for col in result.columns if col.startswith('breadth_')]
    if breadth_cols:
        # Use first available breadth indicator
        sentiment_components.append(breadth_cols[0])
    
    # Normalize components
    normalized_components = []
    for component in sentiment_components:
        if component in result.columns:
            normalized = _zscore(result[component])
            normalized_components.append(normalized)
    
    if normalized_components:
        component_df = pd.concat(normalized_components, axis=1)
        valid_count = component_df.notna().sum(axis=1)
        component_sum = component_df.fillna(0).sum(axis=1)
        result['market_sentiment_score'] = (component_sum / valid_count.replace(0, np.nan)).fillna(0)
    else:
        result['market_sentiment_score'] = 0
    
    return result


def cross_section_market_score(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional market sentiment scores for multiple symbols.
    
    Args:
        df_latest: DataFrame with latest market sentiment scores for multiple symbols
        
    Returns:
        DataFrame with normalized MarketScore
    """
    if 'market_sentiment_score' not in df_latest.columns:
        raise ValueError("DataFrame must contain 'market_sentiment_score' column")
        
    result = df_latest[['symbol', 'market_sentiment_score']].copy()
    result['MarketScore'] = _zscore(result['market_sentiment_score'])
    
    return result[['symbol', 'MarketScore']]


def get_market_sentiment_signals(df: pd.DataFrame,
                                market_heat_threshold: float = 0.5,
                                fear_threshold: float = -0.3,
                                flow_threshold: float = 0.2) -> pd.DataFrame:
    """
    Generate trading signals based on market sentiment factors.
    
    Args:
        df: DataFrame with market sentiment features
        market_heat_threshold: Market heat threshold for bullish signal
        fear_threshold: Fear factor threshold for contrarian signal
        flow_threshold: Fund flow threshold for bullish signal
        
    Returns:
        DataFrame with market sentiment signals
    """
    result = df.copy()
    result['market_sentiment_signal'] = 0
    
    # Market conditions
    hot_market = result.get('market_heat', 0) > market_heat_threshold
    fearful_market = result.get('fear_factor', 0) < fear_threshold  # Contrarian
    positive_flow = result.get('fund_flow', 0) > flow_threshold
    
    # Breadth confirmation
    strong_breadth = result.get('breadth_advance_decline_ratio', 1) > 1.2
    weak_breadth = result.get('breadth_advance_decline_ratio', 1) < 0.8
    
    # Bullish market sentiment
    bullish_sentiment = (
        hot_market.astype(int) + 
        fearful_market.astype(int) +  # Contrarian
        positive_flow.astype(int) + 
        strong_breadth.astype(int)
    )
    
    # Bearish market sentiment  
    bearish_sentiment = (
        (~hot_market).astype(int) + 
        (~fearful_market).astype(int) +
        (~positive_flow).astype(int) + 
        weak_breadth.astype(int)
    )
    
    # Generate signals based on majority
    result.loc[bullish_sentiment >= 3, 'market_sentiment_signal'] = 1
    result.loc[bearish_sentiment >= 3, 'market_sentiment_signal'] = -1
    
    return result[['symbol', 'market_sentiment_signal']] if 'symbol' in result.columns else result[['market_sentiment_signal']]


# Backwards compatibility
def market_score(price_data: Dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Legacy function for backwards compatibility."""
    return market_sentiment_features(price_data, **kwargs)