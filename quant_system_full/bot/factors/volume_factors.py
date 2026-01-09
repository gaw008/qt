import pandas as pd
import numpy as np

def _zscore(s: pd.Series):
    s = pd.to_numeric(s, errors='coerce')
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    obv = (sign * volume).fillna(0).cumsum()
    return obv

def compute_anchored_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    cum_pv = (tp * df['volume']).cumsum()
    cum_v  = (df['volume']).cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def compute_mfi(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    money_flow = tp * df['volume']
    delta_tp = tp.diff()
    pos_flow = money_flow.where(delta_tp > 0, 0.0)
    neg_flow = money_flow.where(delta_tp < 0, 0.0)
    pos_sum = pos_flow.rolling(lookback, min_periods=lookback).sum()
    neg_sum = neg_flow.rolling(lookback, min_periods=lookback).sum().abs()
    ratio = (pos_sum / (neg_sum + 1e-9)).replace([np.inf, -np.inf], np.nan)
    mfi = 100 - (100 / (1 + ratio))
    return mfi

def compute_volume_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    ma = volume.rolling(lookback, min_periods=lookback).mean()
    return volume / (ma + 1e-9)

def volume_features(df: pd.DataFrame, lookback_obv: int = 20, lookback_mfi: int = 14, lookback_vr: int = 20):
    x = df.copy()
    x['obv'] = compute_obv(x['close'], x['volume'])
    x['obv_slope'] = (x['obv'] - x['obv'].shift(lookback_obv)) / (lookback_obv + 1e-9)
    x['vwap'] = compute_anchored_vwap(x)
    x['vwap_diff'] = (x['close'] - x['vwap']) / (x['vwap'].replace(0, np.nan))
    x['mfi'] = compute_mfi(x, lookback=lookback_mfi)
    x['vol_ratio'] = compute_volume_ratio(x['volume'], lookback=lookback_vr)
    # 逐列标准化后按行求均值；skipna 避免空切片告警，并在全 NaN 时置 0
    comp_df = pd.concat([
        _zscore(x['obv_slope']),
        _zscore(x['vwap_diff']),
        _zscore(x['mfi']),
        _zscore(x['vol_ratio'])
    ], axis=1)
    # 手动按行求“有值列”的平均，避免 numpy.nanmean 的空切片告警
    valid_cnt = comp_df.notna().sum(axis=1)
    row_sum = comp_df.fillna(0).sum(axis=1)
    x['vol_score'] = (row_sum / valid_cnt.replace(0, np.nan)).fillna(0)
    return x

def cross_section_volume_score(df_latest: pd.DataFrame) -> pd.DataFrame:
    out = df_latest[['symbol','vol_score']].copy()
    out['VolumeScore'] = _zscore(out['vol_score'])
    return out[['symbol','VolumeScore']]
