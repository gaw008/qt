# Quant System (Full) — Multi-Factor Bot + Dashboard + Kill Switch + Daily Summary

本项目集成：
- **bot/** 多因子量化机器人脚手架（估值 + 技术 + 量价），可与 TradeUP OpenAPI 对接
- **backtest.py / live.py** 回测/线上运行示例
- **dashboard/** 远程管理看板（FastAPI + Streamlit），含 **紧急中止**（Kill Switch）与 **每日总结**
- **scripts/** 提供 CLI 方式生成每日总结（cron 定时）
- **state/** 共享状态、日志与日报输出目录（由后台/worker自动维护）

> 免责声明：本仓库仅用于教育/测试示例。非投资建议；实盘交易须自行承担风险并遵循当地法律法规。

---

## 0) 环境准备
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r bot/requirements.txt
```

可选安装 TradeUP SDK（官方 GitHub）：
```bash
pip install git+https://github.com/tigerbrokers/openapi-python-sdk.git
```

复制并修改配置：
```bash
cp config.example.env .env
# 按需填写 TIGER_ID / ACCOUNT / PRIVATE_KEY_PATH / DRY_RUN 等
```

---

## 1) 回测（SMA 交叉示例）
```bash
python backtest.py --symbol AAPL --short 5 --long 20 --csv your_bars.csv
```
> CSV 需包含列：`time,open,high,low,close,volume`

---

## 2) 运行（Live 占位示例）
```bash
python live.py --symbol AAPL --short 5 --long 20 --interval 300
```
> 初期建议 `DRY_RUN=true`（在 `.env` 中设置）。`bot/data.py` 留有 TradeUP SDK 对接位；完成后可替换为真实行情/下单。

---

## 3) 启动远程看板 + 紧急中止 + 每日总结

### 3.1 后端 API（FastAPI）
```bash
cd dashboard/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 可选：echo ADMIN_TOKEN=your_token > .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3.2 Worker（策略循环示例）
```bash
cd dashboard/worker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python runner.py
```
> Worker 每轮检查 `state/kill.flag`，存在则暂停；状态与日志写入 `state/status.json` 与 `state/bot.log`。

### 3.3 前端（Streamlit）
```bash
cd dashboard/frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
> 侧栏填 `API Base = http://localhost:8000`，`Bearer Token = 你的 ADMIN_TOKEN`。  
> 按钮：**Emergency STOP**、**Resume**、**Generate Daily Summary**。

---

## 4) 每日总结（API/CLI/定时）

- 面板按钮：📝 Generate Daily Summary（调用 `GET /summary`）
- CLI：
```bash
cd scripts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export API_BASE=http://localhost:8000
export ADMIN_TOKEN=your_token
python daily_summary.py
```
- cron 示例（每日 21:10）：
```
10 21 * * * /path/to/.venv/bin/python /path/to/quant_system_full/scripts/daily_summary.py >> /var/log/quant_daily.log 2>&1
```

---

## 5) 生产建议
- **安全**：后端置于 HTTPS 反向代理后（Nginx/Caddy），启用 `ADMIN_TOKEN` 并限制来源 IP；敏感配置用 `.env`/密钥服务。
- **持久化**：`dashboard/state/` 放到持久盘；日志轮转。
- **可观测性**：结构化日志（信号/下单/成交），拒单与异常告警（可接企业 IM）。
- **灰度上线**：先 paper，再小仓位；阈值/权重来自回测与走查。

祝顺！
