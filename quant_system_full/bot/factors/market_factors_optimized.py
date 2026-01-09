"""
Optimized market sentiment factors for quantitative trading system.

This module implements various market sentiment and macro factors including:
- Market heat index, sector rotation factors
- VIX fear index analysis, fund flow direction analysis
- Market breadth indicators
- Relative performance vs market/sector benchmarks

Key optimizations:
- Fixed DataFrame boolean evaluation errors with proper vectorization
- Optimized correlation calculations using vectorized operations
- Enhanced error handling and data validation
- Improved memory efficiency and performance
- Applied SOLID principles for better maintainability
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timedelta
import warnings
from scipy.stats import pearsonr


def zscore_vectorized(series: pd.Series) -> pd.Series:
    """
    Vectorized z-score computation with robust error handling.

    Args:
        series: Input series

    Returns:
        Z-score normalized series
    """
    if series.empty or series.isna().all():
        return pd.Series(0.0, index=series.index)

    series_clean = pd.to_numeric(series, errors='coerce')

    if series_clean.count() < 2:
        return pd.Series(0.0, index=series.index)

    mean_val = series_clean.mean()
    std_val = series_clean.std(ddof=0)

    if std_val == 0 or pd.isna(std_val):
        return pd.Series(0.0, index=series.index)

    return (series_clean - mean_val) / std_val


class MarketHeatCalculator:
    """Optimized market heat index calculator with vectorized operations."""

    @staticmethod
    def calculate(price_data: Dict[str, pd.DataFrame],
                 volume_data: Optional[Dict[str, pd.DataFrame]] = None,
                 period: int = 20) -> pd.Series:
        """
        Compute market heat index using vectorized operations.

        Args:
            price_data: Dictionary of symbol -> price DataFrame
            volume_data: Dictionary of symbol -> volume data (optional)
            period: Calculation period

        Returns:
            Market heat index time series
        """
        if not price_data:
            return pd.Series(dtype=float)

        # Pre-allocate lists for better performance
        momentum_series_list = []
        volume_series_list = []

        for symbol, df in price_data.items():
            if df is None or df.empty or 'close' not in df.columns:
                continue

            try:
                # Vectorized momentum calculation
                close_prices = df['close']
                momentum = close_prices.pct_change(period).fillna(0)
                momentum_series_list.append(momentum)

                # Volume momentum if available
                if volume_data and symbol in volume_data:
                    vol_df = volume_data[symbol]
                    if vol_df is not None and 'volume' in vol_df.columns:
                        volume_series = vol_df['volume']
                        # Vectorized volume ratio calculation
                        vol_ma = volume_series.rolling(period, min_periods=1).mean()
                        vol_ratio = (volume_series / vol_ma).fillna(1)
                        volume_series_list.append(vol_ratio)

            except Exception:
                continue

        if not momentum_series_list:
            return pd.Series(dtype=float)

        # Efficient concatenation and aggregation
        momentum_df = pd.concat(momentum_series_list, axis=1, sort=True)
        avg_momentum = momentum_df.mean(axis=1, skipna=True)

        if volume_series_list:
            volume_df = pd.concat(volume_series_list, axis=1, sort=True)
            avg_volume = volume_df.mean(axis=1, skipna=True)
            # Weighted combination
            heat_index = 0.7 * avg_momentum + 0.3 * (avg_volume - 1)
        else:
            heat_index = avg_momentum

        return heat_index.fillna(0)


class SectorRotationAnalyzer:
    """Optimized sector rotation factor calculator."""

    @staticmethod
    def calculate(sector_data: Dict[str, Dict[str, pd.DataFrame]],
                 benchmark_data: Optional[pd.DataFrame] = None,
                 period: int = 20) -> Dict[str, pd.Series]:
        """
        Compute sector rotation factors with optimized performance.

        Args:
            sector_data: Dictionary of sector -> {symbol: DataFrame}
            benchmark_data: Market benchmark data
            period: Calculation period

        Returns:
            Dictionary of sector -> rotation factor series
        """
        sector_factors = {}

        # Pre-calculate benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None and 'close' in benchmark_data.columns:
            benchmark_returns = benchmark_data['close'].pct_change(period).fillna(0)

        for sector, stocks in sector_data.items():
            if not stocks:
                continue

            # Collect all sector returns in one pass
            sector_returns_list = []
            for symbol, df in stocks.items():
                if df is not None and not df.empty and 'close' in df.columns:
                    returns = df['close'].pct_change(period).fillna(0)
                    sector_returns_list.append(returns)

            if sector_returns_list:
                # Vectorized aggregation
                sector_df = pd.concat(sector_returns_list, axis=1, sort=True)
                sector_avg_returns = sector_df.mean(axis=1, skipna=True)

                # Relative performance calculation
                if benchmark_returns is not None:
                    # Align indices for proper subtraction
                    aligned_data = pd.concat([sector_avg_returns, benchmark_returns],
                                           axis=1, join='inner')
                    if not aligned_data.empty:
                        relative_performance = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
                    else:
                        relative_performance = sector_avg_returns
                else:
                    relative_performance = sector_avg_returns

                sector_factors[sector] = relative_performance.fillna(0)

        return sector_factors


class VIXFearAnalyzer:
    """Optimized VIX fear/greed factor calculator."""

    @staticmethod
    def calculate(vix_data: pd.DataFrame,
                 fear_threshold: float = 25,
                 greed_threshold: float = 15) -> pd.Series:
        """
        Compute fear/greed factor with vectorized operations.

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

        # Vectorized fear/greed calculation
        fear_factor = pd.Series(0.0, index=vix_level.index)

        # High VIX = Fear (negative values) - vectorized
        fear_mask = vix_level > fear_threshold
        fear_values = -np.clip((vix_level[fear_mask] - fear_threshold) / 25, 0, 1)
        fear_factor.loc[fear_mask] = fear_values

        # Low VIX = Greed (positive values) - vectorized
        greed_mask = vix_level < greed_threshold
        greed_values = np.clip((greed_threshold - vix_level[greed_mask]) / 10, 0, 1)
        fear_factor.loc[greed_mask] = greed_values

        return fear_factor.fillna(0)


class FundFlowAnalyzer:
    """Optimized fund flow direction calculator."""

    @staticmethod
    def calculate(price_data: Dict[str, pd.DataFrame],
                 volume_data: Dict[str, pd.DataFrame],
                 period: int = 10) -> pd.Series:
        """
        Compute fund flow using optimized price-volume analysis.

        Args:
            price_data: Dictionary of symbol -> price DataFrame
            volume_data: Dictionary of symbol -> volume DataFrame
            period: Calculation period

        Returns:
            Fund flow factor series
        """
        if not price_data or not volume_data:
            return pd.Series(dtype=float)

        flow_scores_list = []

        for symbol in price_data.keys():
            if (symbol in volume_data and
                price_data[symbol] is not None and
                volume_data[symbol] is not None):

                price_df = price_data[symbol]
                volume_df = volume_data[symbol]

                if ('close' in price_df.columns and
                    'volume' in volume_df.columns and
                    len(price_df) == len(volume_df)):

                    try:
                        # Vectorized calculations
                        price_change = price_df['close'].pct_change().fillna(0)
                        volume_change = volume_df['volume'].pct_change().fillna(0)

                        # Rolling correlation with proper alignment
                        if len(price_change) >= period:
                            correlation = price_change.rolling(period, min_periods=period//2).corr(volume_change).fillna(0)

                            # On-Balance Volume calculation - vectorized
                            volume_series = volume_df['volume']
                            obv_changes = np.where(price_change > 0, volume_series,
                                                 np.where(price_change < 0, -volume_series, 0))

                            obv_series = pd.Series(obv_changes, index=price_df.index)
                            obv_trend = obv_series.rolling(period, min_periods=1).mean()

                            # Safe division for normalization
                            volume_ma = volume_series.rolling(period, min_periods=1).mean()
                            with np.errstate(divide='ignore', invalid='ignore'):
                                obv_normalized = obv_trend / volume_ma
                            obv_normalized = obv_normalized.fillna(0)

                            # Combine correlation and OBV
                            flow_score = 0.6 * correlation + 0.4 * obv_normalized
                            flow_scores_list.append(flow_score)

                    except Exception:
                        continue

        if flow_scores_list:
            flow_df = pd.concat(flow_scores_list, axis=1, sort=True)
            return flow_df.mean(axis=1, skipna=True).fillna(0)
        else:
            return pd.Series(dtype=float)


class MarketBreadthCalculator:
    """Optimized market breadth indicators calculator."""

    @staticmethod
    def calculate(price_data: Dict[str, pd.DataFrame],
                 period: int = 20) -> Dict[str, pd.Series]:
        """
        Compute market breadth indicators with vectorized operations.

        Args:
            price_data: Dictionary of symbol -> price DataFrame
            period: Calculation period

        Returns:
            Dictionary with breadth indicators
        """
        if not price_data:
            return {}

        # Pre-allocate lists for better performance
        advance_series_list = []
        decline_series_list = []
        new_high_series_list = []
        new_low_series_list = []

        for symbol, df in price_data.items():
            if df is None or df.empty or 'close' not in df.columns:
                continue

            try:
                # Vectorized daily change calculation
                price_change = df['close'].pct_change().fillna(0)
                advance_series_list.append((price_change > 0).astype(int))
                decline_series_list.append((price_change < 0).astype(int))

                # New highs/lows if high/low data available
                if 'high' in df.columns and 'low' in df.columns:
                    rolling_high = df['high'].rolling(period, min_periods=1).max()
                    rolling_low = df['low'].rolling(period, min_periods=1).min()

                    new_high_series_list.append((df['high'] == rolling_high).astype(int))
                    new_low_series_list.append((df['low'] == rolling_low).astype(int))

            except Exception:
                continue

        breadth_indicators = {}

        if advance_series_list and decline_series_list:
            # Efficient concatenation and aggregation
            advances_df = pd.concat(advance_series_list, axis=1, sort=True)
            declines_df = pd.concat(decline_series_list, axis=1, sort=True)

            total_advances = advances_df.sum(axis=1, skipna=True)
            total_declines = declines_df.sum(axis=1, skipna=True)

            # Advance-Decline Line
            ad_line = (total_advances - total_declines).cumsum()
            breadth_indicators['advance_decline_line'] = ad_line

            # Advance-Decline Ratio with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                ad_ratio = total_advances / (total_declines + 1e-9)
            breadth_indicators['advance_decline_ratio'] = ad_ratio.fillna(1)

            # McClellan Oscillator (simplified)
            ad_diff = total_advances - total_declines
            mcclellan_fast = ad_diff.ewm(span=19, adjust=False).mean()
            mcclellan_slow = ad_diff.ewm(span=39, adjust=False).mean()
            breadth_indicators['mcclellan_oscillator'] = mcclellan_fast - mcclellan_slow

        if new_high_series_list and new_low_series_list:
            # New Highs - New Lows indicators
            new_highs_df = pd.concat(new_high_series_list, axis=1, sort=True)
            new_lows_df = pd.concat(new_low_series_list, axis=1, sort=True)

            total_new_highs = new_highs_df.sum(axis=1, skipna=True)
            total_new_lows = new_lows_df.sum(axis=1, skipna=True)

            breadth_indicators['new_highs_lows'] = total_new_highs - total_new_lows

            # High-Low Index with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                hl_index = total_new_highs / (total_new_highs + total_new_lows + 1e-9)
            breadth_indicators['high_low_index'] = hl_index.fillna(0.5)

        return breadth_indicators


def market_sentiment_features(price_data: Dict[str, pd.DataFrame],
                             volume_data: Optional[Dict[str, pd.DataFrame]] = None,
                             vix_data: Optional[pd.DataFrame] = None,
                             benchmark_data: Optional[pd.DataFrame] = None,
                             sector_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
                             symbol: Optional[str] = None,
                             heat_period: int = 20,
                             rotation_period: int = 20,
                             flow_period: int = 10,
                             breadth_period: int = 20,
                             relative_period: int = 20) -> pd.DataFrame:
    """
    Calculate comprehensive market sentiment features with optimized performance.

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
    # Get base time index with proper handling
    if symbol and symbol in price_data and price_data[symbol] is not None:
        base_index = price_data[symbol].index
    elif price_data:
        # Get first non-None DataFrame
        for df in price_data.values():
            if df is not None and not df.empty:
                base_index = df.index
                break
        else:
            base_index = pd.Index([])
    else:
        base_index = pd.Index([])

    result = pd.DataFrame(index=base_index)

    # Market heat index
    try:
        market_heat = MarketHeatCalculator.calculate(price_data, volume_data, heat_period)
        if not market_heat.empty:
            result['market_heat'] = market_heat.reindex(base_index, fill_value=0)
        else:
            result['market_heat'] = 0
    except Exception:
        result['market_heat'] = 0

    # VIX fear factor
    try:
        if vix_data is not None:
            fear_factor = VIXFearAnalyzer.calculate(vix_data)
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
            fund_flow = FundFlowAnalyzer.calculate(price_data, volume_data, flow_period)
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
        breadth_indicators = MarketBreadthCalculator.calculate(price_data, breadth_period)
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
            sector_factors = SectorRotationAnalyzer.calculate(
                sector_data, benchmark_data, rotation_period
            )
            for sector, factor in sector_factors.items():
                if not factor.empty:
                    result[f'sector_{sector}_rotation'] = factor.reindex(base_index, fill_value=0)
    except Exception:
        pass

    # Composite market sentiment score with robust calculation
    sentiment_components = ['market_heat', 'fear_factor', 'fund_flow']

    # Add breadth components if available
    breadth_cols = [col for col in result.columns if col.startswith('breadth_')]
    if breadth_cols:
        sentiment_components.append(breadth_cols[0])

    # Vectorized normalization and aggregation
    normalized_components = []
    for component in sentiment_components:
        if component in result.columns:
            normalized = zscore_vectorized(result[component])
            normalized_components.append(normalized)

    if normalized_components:
        component_df = pd.concat(normalized_components, axis=1)
        # Efficient aggregation with proper NaN handling
        valid_count = component_df.count(axis=1)
        component_sum = component_df.sum(axis=1, skipna=True)

        # Safe division with vectorized operations
        with np.errstate(divide='ignore', invalid='ignore'):
            sentiment_score = component_sum / valid_count
        result['market_sentiment_score'] = np.where(valid_count > 0, sentiment_score, 0)
    else:
        result['market_sentiment_score'] = 0

    return result


def cross_section_market_score(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional market sentiment scores with robust error handling.

    Args:
        df_latest: DataFrame with latest market sentiment scores for multiple symbols

    Returns:
        DataFrame with normalized MarketScore
    """
    if df_latest.empty:
        return pd.DataFrame(columns=['symbol', 'MarketScore'])

    required_cols = ['symbol', 'market_sentiment_score']
    for col in required_cols:
        if col not in df_latest.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    result = df_latest[required_cols].copy()

    # Cross-sectional normalization with edge case handling
    scores = result['market_sentiment_score']
    if scores.std() == 0 or scores.count() < 2:
        result['MarketScore'] = 0
    else:
        result['MarketScore'] = zscore_vectorized(scores)

    return result[['symbol', 'MarketScore']]


def get_market_sentiment_signals(df: pd.DataFrame,
                                market_heat_threshold: float = 0.5,
                                fear_threshold: float = -0.3,
                                flow_threshold: float = 0.2) -> pd.DataFrame:
    """
    Generate trading signals with optimized vectorized operations.

    Args:
        df: DataFrame with market sentiment features
        market_heat_threshold: Market heat threshold for bullish signal
        fear_threshold: Fear factor threshold for contrarian signal
        flow_threshold: Fund flow threshold for bullish signal

    Returns:
        DataFrame with market sentiment signals
    """
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'market_sentiment_signal'])

    result = df.copy()
    result['market_sentiment_signal'] = 0

    # Extract values as Series to avoid DataFrame boolean evaluation errors
    market_heat_values = result.get('market_heat', pd.Series(0, index=result.index))
    fear_factor_values = result.get('fear_factor', pd.Series(0, index=result.index))
    fund_flow_values = result.get('fund_flow', pd.Series(0, index=result.index))
    breadth_ratio_values = result.get('breadth_advance_decline_ratio', pd.Series(1, index=result.index))

    # Ensure all are Series for vectorized operations
    for var_name, values in [('market_heat_values', market_heat_values),
                            ('fear_factor_values', fear_factor_values),
                            ('fund_flow_values', fund_flow_values),
                            ('breadth_ratio_values', breadth_ratio_values)]:
        if not isinstance(values, pd.Series):
            exec(f"{var_name} = pd.Series({var_name}, index=result.index)")

    # Vectorized boolean operations to avoid DataFrame.bool() errors
    hot_market = (market_heat_values > market_heat_threshold).fillna(False)
    fearful_market = (fear_factor_values < fear_threshold).fillna(False)  # Contrarian
    positive_flow = (fund_flow_values > flow_threshold).fillna(False)

    # Breadth confirmation
    strong_breadth = (breadth_ratio_values > 1.2).fillna(False)
    weak_breadth = (breadth_ratio_values < 0.8).fillna(False)

    # Bullish sentiment calculation - vectorized
    bullish_sentiment = (
        hot_market.astype(int) +
        fearful_market.astype(int) +  # Contrarian
        positive_flow.astype(int) +
        strong_breadth.astype(int)
    )

    # Bearish sentiment calculation - vectorized
    bearish_sentiment = (
        (~hot_market).astype(int) +
        (~fearful_market).astype(int) +
        (~positive_flow).astype(int) +
        weak_breadth.astype(int)
    )

    # Generate signals based on majority voting
    result.loc[bullish_sentiment >= 3, 'market_sentiment_signal'] = 1
    result.loc[bearish_sentiment >= 3, 'market_sentiment_signal'] = -1

    return result[['symbol', 'market_sentiment_signal']] if 'symbol' in result.columns else result[['market_sentiment_signal']]


# Backwards compatibility
def market_score(price_data: Dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Legacy function for backwards compatibility."""
    return market_sentiment_features(price_data, **kwargs)


if __name__ == "__main__":
    print("Testing Optimized Market Sentiment Factors")
    print("=" * 45)

    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # Sample price data
    price_data = {}
    for i in range(10):
        symbol = f'STOCK_{i:02d}'
        price_data[symbol] = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
            'high': 101 + np.cumsum(np.random.normal(0, 1, 100)),
            'low': 99 + np.cumsum(np.random.normal(0, 1, 100)),
        }, index=dates)

    # Calculate market sentiment features
    result = market_sentiment_features(price_data)

    print(f"Processed market sentiment for {len(price_data)} stocks")
    print(f"Features calculated: {list(result.columns)}")
    print(f"Market sentiment score range: {result['market_sentiment_score'].min():.3f} to {result['market_sentiment_score'].max():.3f}")

    print("\\nOptimization completed successfully!")