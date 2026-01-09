import pandas as pd
def simple_pnl(df: pd.DataFrame):
    df = df.copy()
    df['ret'] = df['close'].pct_change().fillna(0)
    df['pnl'] = df['position'] * df['ret']
    df['cum_pnl'] = (1 + df['pnl']).cumprod() - 1
    return df
