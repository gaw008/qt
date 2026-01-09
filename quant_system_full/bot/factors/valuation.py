import pandas as pd, numpy as np

def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def zscore(s):
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def compute_ev(df):
    return (df['market_cap'] + df['total_debt'] + df.get('minority_interest',0) +
            df.get('preferred_equity',0) - df['cash_equiv'])

def valuation_score(fund: pd.DataFrame) -> pd.DataFrame:
    df = fund.copy()
    df['ev'] = compute_ev(df)
    df['ev_ebitda'] = df['ev'] / df['ebitda_ttm'].replace(0,np.nan)
    df['ev_sales']  = df['ev'] / df['revenue_ttm'].replace(0,np.nan)
    df['pb']        = df['market_cap'] / df['book_equity'].replace(0,np.nan)
    df['ev_fcf']    = df['ev'] / df['fcf_ttm'].replace(0,np.nan)
    for col in ['ev_ebitda','ev_sales','pb','ev_fcf']:
        df[col] = winsorize(df[col].replace([np.inf,-np.inf], np.nan).fillna(df[col].median()))
    if 'industry' in df:
        for src, dst in [('ev_ebitda','z_ebitda'),('ev_sales','z_sales'),('pb','z_pb'),('ev_fcf','z_fcf')]:
            df[dst] = df.groupby('industry')[src].transform(zscore)
    else:
        df['z_ebitda'] = zscore(df['ev_ebitda']); df['z_sales']=zscore(df['ev_sales'])
        df['z_pb']=zscore(df['pb']); df['z_fcf']=zscore(df['ev_fcf'])
    df['ValuationScore'] = (-0.35*df['z_ebitda'] + -0.25*df['z_fcf'] + -0.20*df['z_pb'] + -0.20*df['z_sales'])
    return df[['symbol','ValuationScore']]
