# Bot — Multi-Factor Scaffold

- `strategies/`：SMA、Momentum、Mean Reversion
- `factors/`：估值（ValuationScore）、量价（OBV/VWAP/MFI/量比）
- `alpha_router.py`：多因子融合（估值×技术×量价）
- `execution.py`：统一下单参数封装（按 SDK 实际签名替换）
- `data.py`：行情读取对接位（TradeUP SDK）

**运行**
```bash
python ../backtest.py --symbol AAPL --short 5 --long 20 --csv your_bars.csv
python ../live.py --symbol AAPL --short 5 --long 20 --interval 300
```
