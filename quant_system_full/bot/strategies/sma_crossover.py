import pandas as pd
def generate_signals(df: pd.DataFrame, short=5, long=20):
    df = df.copy()
    df['sma_s'] = df['close'].rolling(short).mean()
    df['sma_l'] = df['close'].rolling(long).mean()
    df['signal'] = 0
    df.loc[df['sma_s'] > df['sma_l'], 'signal'] = 1
    df.loc[df['sma_s'] < df['sma_l'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)
    return df
