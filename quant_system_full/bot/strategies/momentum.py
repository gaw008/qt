import pandas as pd
def generate_signals(df: pd.DataFrame, lookback=20, upper=0.05, lower=-0.05):
    df = df.copy()
    df['ret_lb'] = df['close'].pct_change(lookback)
    df['signal'] = 0
    df.loc[df['ret_lb'] >= upper, 'signal'] = 1
    df.loc[df['ret_lb'] <= lower, 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    return df
