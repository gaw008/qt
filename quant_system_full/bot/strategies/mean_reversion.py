import pandas as pd
import pandas_ta as ta

def generate_pdf_signals(df: pd.DataFrame, rsi_len=2, rsi_entry=20, rsi_exit=70, obv_len=10):
    """
    Generates trading signals based on the multi-factor strategy described in the PDF.
    Combines RSI for entry, filtered by MACD for trend and OBV for volume confirmation.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        rsi_len (int): The lookback period for RSI. PDF suggests a very short one.
        rsi_entry (int): The RSI level for a buy signal (oversold).
        rsi_exit (int): The RSI level for a sell signal (overbought).
        obv_len (int): The moving average lookback for OBV trend confirmation.

    Returns:
        pd.DataFrame: The original DataFrame with added indicators and a 'signal' column.
    """
    df = df.copy()
    
    # Calculate indicators using pandas_ta
    df.ta.rsi(length=rsi_len, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.obv(append=True)
    
    # Define indicator names for clarity
    rsi_col = f'RSI_{rsi_len}'
    macd_col = 'MACD_12_26_9'
    obv_col = 'OBV'
    
    # Calculate OBV moving average for trend confirmation
    df['OBV_MA'] = df[obv_col].rolling(window=obv_len).mean()

    # --- Define Signal Conditions based on PDF ---
    
    # 1. Buy Signal: RSI is oversold, in a confirmed uptrend (MACD > 0), with volume confirmation (OBV > MA).
    buy_conditions = (
        (df[rsi_col] < rsi_entry) &
        (df[macd_col] > 0) &
        (df[obv_col] > df['OBV_MA'])
    )
    
    # 2. Sell Signal: RSI is overbought (profit taking).
    sell_conditions = (
        df[rsi_col] > rsi_exit
    )
    
    # --- Generate Signals ---
    df['signal'] = 0
    df.loc[buy_conditions, 'signal'] = 1
    df.loc[sell_conditions, 'signal'] = -1
    
    # For backtesting, a position is taken on the next bar after the signal
    df['position'] = df['signal'].shift(1).fillna(0)
    
    return df

def get_signal(df: pd.DataFrame, rsi_len=2, rsi_entry=20, rsi_exit=70, obv_len=10):
    """
    Gets the latest signal for a single symbol.
    
    Args:
        df (pd.DataFrame): DataFrame with historical data for one symbol.
        
    Returns:
        int: The latest signal (1, -1, or 0).
    """
    if len(df) < max(26, obv_len): # Need enough data for MACD and OBV MA
        return 0
        
    # Calculate indicators for the entire series to ensure accuracy
    df_with_signals = generate_pdf_signals(df, rsi_len, rsi_entry, rsi_exit, obv_len)
    
    # Return the most recent signal
    return df_with_signals['signal'].iloc[-1]