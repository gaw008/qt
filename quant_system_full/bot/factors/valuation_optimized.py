"""
Optimized valuation factors module for quantitative trading system.

This module provides:
- Robust valuation metric calculations with proper error handling
- Vectorized operations for performance optimization
- Consistent function signatures across all factor modules
- Enhanced data quality validation and outlier treatment

Key improvements:
- Fixed function signature to match other factor modules
- Added comprehensive type hints and docstrings
- Implemented robust error handling for missing data
- Optimized performance using vectorized operations
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings


def winsorize_robust(series: pd.Series, percentile: float = 0.01) -> pd.Series:
    """
    Robust winsorization with proper handling of edge cases.

    Args:
        series: Input series to winsorize
        percentile: Percentile threshold for winsorization

    Returns:
        Winsorized series
    """
    if series.empty or series.isna().all():
        return series

    # Convert to numeric and handle infinities
    series_clean = pd.to_numeric(series, errors='coerce')
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)

    if series_clean.count() < 2:  # Need at least 2 values for quantiles
        return series_clean

    try:
        lower_bound = series_clean.quantile(percentile)
        upper_bound = series_clean.quantile(1 - percentile)
        return series_clean.clip(lower_bound, upper_bound)
    except Exception:
        return series_clean


def zscore_robust(series: pd.Series) -> pd.Series:
    """
    Robust z-score normalization with proper error handling.

    Args:
        series: Input series to normalize

    Returns:
        Z-score normalized series
    """
    if series.empty or series.isna().all():
        return pd.Series(dtype=float, index=series.index)

    series_clean = pd.to_numeric(series, errors='coerce')

    if series_clean.count() < 2:
        return pd.Series(0.0, index=series.index)

    mean_val = series_clean.mean()
    std_val = series_clean.std(ddof=0)

    # Avoid division by zero
    if std_val == 0 or pd.isna(std_val):
        return pd.Series(0.0, index=series.index)

    return (series_clean - mean_val) / std_val


def compute_enterprise_value(df: pd.DataFrame) -> pd.Series:
    """
    Calculate enterprise value with robust handling of missing components.

    Args:
        df: DataFrame with financial metrics

    Returns:
        Enterprise value series
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Required components
    market_cap = df.get('market_cap', 0)
    total_debt = df.get('total_debt', 0)
    cash_equiv = df.get('cash_equiv', 0)

    # Optional components
    minority_interest = df.get('minority_interest', 0)
    preferred_equity = df.get('preferred_equity', 0)

    enterprise_value = (
        market_cap +
        total_debt +
        minority_interest +
        preferred_equity -
        cash_equiv
    )

    return enterprise_value


def valuation_factors(data: pd.DataFrame, period: int = 252) -> pd.DataFrame:
    """
    Calculate comprehensive valuation factors with optimized performance.

    This function maintains compatibility with other factor modules by accepting
    a period parameter and returning a standardized DataFrame format.

    Args:
        data: DataFrame with financial and market data
        period: Not used for valuation (for signature compatibility)

    Returns:
        DataFrame with valuation scores and metrics
    """
    if data.empty:
        return pd.DataFrame(columns=['symbol', 'ValuationScore'])

    # Ensure we have required columns
    required_cols = ['market_cap']
    if not all(col in data.columns for col in required_cols):
        warnings.warn("Missing required valuation data columns")
        return pd.DataFrame(columns=['symbol', 'ValuationScore'])

    df = data.copy()

    # Calculate enterprise value
    df['ev'] = compute_enterprise_value(df)

    # Valuation ratios with robust error handling
    valuation_metrics = {}

    # EV/EBITDA
    if 'ebitda_ttm' in df.columns:
        ebitda = df['ebitda_ttm'].replace(0, np.nan)
        valuation_metrics['ev_ebitda'] = df['ev'] / ebitda

    # EV/Sales
    if 'revenue_ttm' in df.columns:
        revenue = df['revenue_ttm'].replace(0, np.nan)
        valuation_metrics['ev_sales'] = df['ev'] / revenue

    # Price-to-Book
    if 'book_equity' in df.columns:
        book_equity = df['book_equity'].replace(0, np.nan)
        valuation_metrics['pb'] = df['market_cap'] / book_equity

    # EV/FCF
    if 'fcf_ttm' in df.columns:
        fcf = df['fcf_ttm'].replace(0, np.nan)
        valuation_metrics['ev_fcf'] = df['ev'] / fcf

    # Price/Earnings
    if 'earnings_ttm' in df.columns:
        earnings = df['earnings_ttm'].replace(0, np.nan)
        valuation_metrics['pe'] = df['market_cap'] / earnings

    # Add calculated metrics to dataframe
    for metric_name, metric_values in valuation_metrics.items():
        # Winsorize outliers
        df[metric_name] = winsorize_robust(metric_values)

    # Industry-adjusted normalization if industry data available
    z_score_metrics = {}

    if 'industry' in df.columns and not df['industry'].isna().all():
        # Industry-relative scoring
        for metric_name in valuation_metrics.keys():
            if metric_name in df.columns:
                z_col_name = f'z_{metric_name.split("_")[-1]}'  # e.g., 'z_ebitda'
                try:
                    df[z_col_name] = df.groupby('industry')[metric_name].transform(zscore_robust)
                    z_score_metrics[z_col_name] = metric_name
                except Exception:
                    # Fallback to overall normalization
                    df[z_col_name] = zscore_robust(df[metric_name])
                    z_score_metrics[z_col_name] = metric_name
    else:
        # Overall market normalization
        for metric_name in valuation_metrics.keys():
            z_col_name = f'z_{metric_name.split("_")[-1]}'
            df[z_col_name] = zscore_robust(df[metric_name])
            z_score_metrics[z_col_name] = metric_name

    # Composite valuation score with robust weights
    score_components = []
    weights = {}

    # Define weights for available metrics
    if 'z_ebitda' in df.columns:
        score_components.append(-0.30 * df['z_ebitda'])  # Lower is better
        weights['ev_ebitda'] = 0.30

    if 'z_fcf' in df.columns:
        score_components.append(-0.25 * df['z_fcf'])
        weights['ev_fcf'] = 0.25

    if 'z_pb' in df.columns:
        score_components.append(-0.20 * df['z_pb'])
        weights['pb'] = 0.20

    if 'z_sales' in df.columns:
        score_components.append(-0.15 * df['z_sales'])
        weights['ev_sales'] = 0.15

    if 'z_pe' in df.columns:
        score_components.append(-0.10 * df['z_pe'])
        weights['pe'] = 0.10

    # Calculate composite score
    if score_components:
        df['ValuationScore'] = sum(score_components)
    else:
        df['ValuationScore'] = 0.0

    # Prepare result DataFrame
    result_columns = ['symbol', 'ValuationScore']

    # Include individual metrics for analysis
    metric_columns = list(valuation_metrics.keys()) + list(z_score_metrics.keys())
    available_columns = [col for col in result_columns + metric_columns if col in df.columns]

    result = df[available_columns].copy()

    # Ensure symbol column exists
    if 'symbol' not in result.columns:
        if hasattr(df, 'index') and df.index.name == 'symbol':
            result['symbol'] = df.index
        else:
            result['symbol'] = range(len(result))

    return result


def cross_section_valuation_score(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-sectional valuation scores for multiple symbols.

    Args:
        df_latest: DataFrame with latest valuation scores for multiple symbols

    Returns:
        DataFrame with normalized valuation scores
    """
    if df_latest.empty:
        return pd.DataFrame(columns=['symbol', 'ValuationScore'])

    if 'ValuationScore' not in df_latest.columns:
        raise ValueError("DataFrame must contain 'ValuationScore' column")

    if 'symbol' not in df_latest.columns:
        raise ValueError("DataFrame must contain 'symbol' column")

    result = df_latest[['symbol', 'ValuationScore']].copy()

    # Cross-sectional normalization
    scores = result['ValuationScore']
    if scores.std() == 0:
        result['ValuationScore'] = 0
    else:
        result['ValuationScore'] = zscore_robust(scores)

    return result


# Backwards compatibility
def valuation_score(fund: pd.DataFrame) -> pd.DataFrame:
    """Legacy function for backwards compatibility."""
    return valuation_factors(fund)


if __name__ == "__main__":
    # Test the optimized valuation factors
    print("Testing Optimized Valuation Factors")
    print("=" * 40)

    # Create sample data
    np.random.seed(42)
    n_stocks = 100

    sample_data = pd.DataFrame({
        'symbol': [f'STOCK_{i:03d}' for i in range(n_stocks)],
        'market_cap': np.random.lognormal(10, 1, n_stocks) * 1e6,
        'total_debt': np.random.lognormal(8, 1.5, n_stocks) * 1e6,
        'cash_equiv': np.random.lognormal(7, 1, n_stocks) * 1e6,
        'ebitda_ttm': np.random.lognormal(8, 1, n_stocks) * 1e6,
        'revenue_ttm': np.random.lognormal(9, 1, n_stocks) * 1e6,
        'book_equity': np.random.lognormal(8.5, 1, n_stocks) * 1e6,
        'fcf_ttm': np.random.lognormal(7.5, 1.5, n_stocks) * 1e6,
        'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy'], n_stocks)
    })

    # Calculate valuation factors
    result = valuation_factors(sample_data)

    print(f"Processed {len(result)} stocks")
    print(f"Valuation Score range: {result['ValuationScore'].min():.3f} to {result['ValuationScore'].max():.3f}")
    print(f"Valuation Score mean: {result['ValuationScore'].mean():.3f}")
    print(f"Valuation Score std: {result['ValuationScore'].std():.3f}")

    # Show top performers
    top_stocks = result.nlargest(5, 'ValuationScore')
    print("\\nTop 5 Value Stocks:")
    print(top_stocks[['symbol', 'ValuationScore']].to_string(index=False))