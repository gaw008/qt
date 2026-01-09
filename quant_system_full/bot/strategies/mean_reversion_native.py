"""
均值回归策略 - 原生实现版本

不依赖pandas_ta，使用原生pandas和numpy计算技术指标
避免版本兼容性问题
"""

import pandas as pd
import numpy as np


def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """计算MACD指标"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """计算OBV指标（能量潮）"""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def generate_pdf_signals(df: pd.DataFrame, rsi_len=2, rsi_entry=20, rsi_exit=70, obv_len=10):
    """
    基于PDF多因子策略生成交易信号
    结合RSI入场、MACD趋势过滤和OBV量能确认
    
    Args:
        df (pd.DataFrame): 包含 ['open', 'high', 'low', 'close', 'volume'] 的数据框
        rsi_len (int): RSI周期，PDF建议使用短周期
        rsi_entry (int): RSI买入阈值（超卖）
        rsi_exit (int): RSI卖出阈值（超买）
        obv_len (int): OBV移动平均周期
        
    Returns:
        pd.DataFrame: 添加了指标和信号列的原始数据框
    """
    df = df.copy()
    
    # 计算技术指标
    df['RSI'] = calculate_rsi(df['close'], rsi_len)
    macd, macd_signal, macd_histogram = calculate_macd(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_histogram
    
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    df['OBV_MA'] = df['OBV'].rolling(window=obv_len).mean()
    
    # 定义信号条件（基于PDF策略）
    
    # 买入信号：RSI超卖 + MACD上行趋势确认 + 量能确认
    buy_conditions = (
        (df['RSI'] < rsi_entry) &           # RSI超卖 (<20)
        (df['MACD'] > 0) &                  # MACD线在零轴上方（上行趋势）
        (df['OBV'] > df['OBV_MA'])          # OBV高于移动平均线（量能确认）
    )
    
    # 卖出信号：RSI超买（获利了结）
    sell_conditions = (
        df['RSI'] > rsi_exit                # RSI超买 (>70)
    )
    
    # 生成信号
    df['signal'] = 0
    df.loc[buy_conditions, 'signal'] = 1
    df.loc[sell_conditions, 'signal'] = -1
    
    # 用于回测的头寸（信号延迟一期执行）
    df['position'] = df['signal'].shift(1).fillna(0)
    
    return df


def get_signal(df: pd.DataFrame, rsi_len=2, rsi_entry=20, rsi_exit=70, obv_len=10):
    """
    获取单个标的的最新交易信号
    
    Args:
        df (pd.DataFrame): 单个标的的历史数据
        
    Returns:
        int: 最新信号 (1=买入, -1=卖出, 0=持有)
    """
    if len(df) < max(26, obv_len):  # 确保有足够数据计算MACD和OBV移动平均
        return 0
    
    # 计算整个序列的指标以确保准确性
    df_with_signals = generate_pdf_signals(df, rsi_len, rsi_entry, rsi_exit, obv_len)
    
    # 返回最新信号
    latest_signal = df_with_signals['signal'].iloc[-1]
    
    # 确保返回值为int类型
    if pd.isna(latest_signal):
        return 0
    
    return int(latest_signal)


def calculate_additional_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算额外的技术因子（扩展功能）
    """
    df = df.copy()
    
    # 布林带
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # 成交量移动平均
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    
    # 价格动量
    df['Price_Momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['Price_Momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # ATR（真实波动范围）
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df


def get_strategy_summary() -> dict:
    """
    返回策略概要信息
    """
    return {
        "strategy_name": "均值回归多因子策略",
        "version": "2.0_native",
        "description": "基于RSI、MACD、OBV的多因子技术分析策略，使用原生计算避免依赖问题",
        "indicators": ["RSI", "MACD", "OBV"],
        "signal_logic": {
            "buy": "RSI < 20 AND MACD > 0 AND OBV > OBV_MA",
            "sell": "RSI > 70"
        },
        "parameters": {
            "rsi_length": 2,
            "rsi_entry": 20,
            "rsi_exit": 70,
            "obv_ma_length": 10
        }
    }


if __name__ == "__main__":
    # 简单测试
    import random
    from datetime import datetime, timedelta
    
    # 生成测试数据
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    test_data = pd.DataFrame({
        'time': dates,
        'open': [100 + random.uniform(-2, 2) for _ in range(100)],
        'high': [102 + random.uniform(-1, 3) for _ in range(100)],
        'low': [98 + random.uniform(-3, 1) for _ in range(100)],
        'close': [100 + random.uniform(-2, 2) for _ in range(100)],
        'volume': [100000 + random.randint(-20000, 20000) for _ in range(100)]
    })
    
    # 测试策略
    signal = get_signal(test_data)
    print(f"测试信号: {signal}")
    
    # 显示策略概要
    summary = get_strategy_summary()
    print(f"策略: {summary['strategy_name']}")
    print(f"版本: {summary['version']}")