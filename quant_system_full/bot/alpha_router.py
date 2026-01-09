import pandas as pd
try:
    from bot.strategies import mean_reversion # 首先尝试pandas_ta版本
except ImportError:
    from bot.strategies import mean_reversion_native as mean_reversion # 回退到原生版本

def get_alpha_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Processes historical data for multiple symbols to generate trading signals
    based on the PDF's core technical strategy.

    Args:
        data (dict): A dictionary where keys are symbols and values are pandas
                     DataFrames containing the historical OHLCV data for that symbol.

    Returns:
        pd.DataFrame: A DataFrame with columns ['symbol', 'signal'] 
                      containing the latest signal for each symbol.
    """
    signals = []
    for symbol, df in data.items():
        if df is not None and not df.empty:
            # Get the most recent signal from our PDF-based strategy
            latest_signal = mean_reversion.get_signal(df)
            signals.append({'symbol': symbol, 'signal': latest_signal})
    
    if not signals:
        return pd.DataFrame(columns=['symbol', 'signal'])

    return pd.DataFrame(signals)