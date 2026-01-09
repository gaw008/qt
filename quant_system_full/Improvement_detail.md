# 量化系统升级路线图与操作手册 v1.0

> 目标：在不推翻现有架构的前提下，用 2–4 周完成一轮“高性价比”升级，缩小回测-实盘偏差、降低尾部回撤、提升执行韧性与可观测性。

---

## 总览：三阶段推进

**Phase 0（当天可做，\~2小时）**

* 建立项目结构与基线指标：开一个 `improvement/` 目录，接入实验追踪（MLflow）与统一配置（.env）。
* 拉通“体检看板”脚本，输出当前：胜率/盈亏比、换手率、%ADV 占用、行业/风格暴露、滑点分布、最大回撤归因。

**Phase 1（第 1 周）：交易成本 & 组合层风险**

* 成本与滑点模型 + %ADV 参与度限制落地到回测与选股/下单流程。
* 组合优化加入行业/风格暴露约束 + 协方差收缩（Ledoit-Wolf）。

**Phase 2（第 2 周）：稳健性与执行韧性**

* Purged/Combinatorial KFold、样本外滚动；Deflated Sharpe；参数敏感性热图。
* 执行层：幂等下单、部分成交回补、断线重连、时钟校准、重试/超时、价格保护。

**Phase 3（第 3–4 周）：自适应 & 数据治理 & 监控告警**

* 三态市场“状态机”（牛/熊/震荡）联动止损/仓位/持仓数/因子权重。
* 公司行为与日历治理：复权、分红/拆并、交易日历、停复牌、数据质量校验。
* 监控：滑点/成本偏差、P\&L 归因、延迟/失败率 SLO、心跳/Kill Switch 演练。

---

## 工具与依赖清单（一次性）

```bash
# Python 库
pip install pandas numpy scipy scikit-learn statsmodels cvxpy \
            pandas_market_calendars yfinance ta \
            mlflow optuna prometheus-client pydantic python-dotenv

# 可选：数据可视化 & API
pip install matplotlib plotly fastapi uvicorn[standard]
```

* **实验追踪**：MLflow（本地 `mlruns/` 或 SQLite）
* **优化**：cvxpy（OSQP/SCS 求解器）
* **监控**：prometheus-client（导出指标）+ 现有前端/Streamlit 仪表
* **时钟**：NTP/Chrony（Linux），Windows 时间服务（Windows）

> 配置 `.env`：`BROKER_API_KEY=...`、`COST_MODEL=basic_v1`、`ADV_CAP=0.08`、`INDUSTRY_LIMIT=0.25`、`MAX_TURNOVER_DAILY=0.25` 等。

---

## 模块 A｜交易成本 & 执行（Phase 1）

### 目标

在**回测、选股、下单**三个环节引入一致的成本与流动性约束：

1. 随机滑点 + 冲击成本（基于 %ADV）
2. 参与度上限（单票单日成交额 ≤ 5–10% ADV）
3. 基础执行策略（TWAP/VWAP、价格偏离保护、撤单/回补）

### 操作步骤

1. **计算 ADV / 参与度**

   * 用 20–60 日滚动成交额均值作为 `ADV_i`；计划下单额为 `Q_i`，参与度 `p_i = Q_i / ADV_i`。
2. **成本函数**

   * `impact_i = k * p_i**α`（建议 `k ∈ [5, 20] bp`，`α ≈ 0.5–1.0`）
   * `slippage_i ~ Normal(μ=spread_bp, σ=σ_slip)`（`spread_bp` 用买卖价差基线，`σ_slip` 用历史成交误差估计）
   * **总成本（bp）**：`cost_i = impact_i + slippage_i + fees_bp`
3. **选股前置**

   * 在选股评分→仓位计算前，先估算 `cost_i`，将**净 α = 粗收益 – 成本**作为排名依据之一（例如扣 1/2 成本做保守）
4. **执行策略**

   * 大单分片：`TWAP(n=4~8)`；若 `p_i` 超 5% 切换 `VWAP` 或跨日执行
   * **价格偏离保护**：若 `|fill_price - ref_price| > gate_bp`，则撤单/重报（`gate_bp` 取 `max(2*spread, 10bp)`）
5. **参数落地**

   * `.env`：`ADV_LOOKBACK=40`、`ADV_CAP=0.08`、`IMPACT_K=0.0015`、`IMPACT_ALPHA=0.7`、`SLIP_SIGMA_BP=8`

### 验收标准（DoD）

* 回测报告新增 **成本前/后收益**、**滑点分布**、**%ADV 占用**三张图
* 实盘/纸交后 1 周内，**实测滑点分布**与模型偏差 < 30%

---

## 模块 B｜组合层风险与优化（Phase 1）

### 目标

在 8% 单票上限基础上，加入**行业/风格暴露约束**与**协方差收缩**，避免“拥挤一跌齐跌”。

### 操作步骤

1. **协方差矩阵**：

   * 以 60–120 日收益计算样本协方差，使用 **Ledoit–Wolf 收缩**（`sklearn.covariance.LedoitWolf`）
2. **行业与风格暴露**：

   * 行业：按 GICS/自有行业映射矩阵 `B_industry`，约束 `|B_industry^T w| ≤ 25%`
   * 风格：取常用 4–6 类（Size/Value/Quality/Momentum/Vol）标准化因子，约束 `|β_style| ≤ 0.5`
3. **优化问题（cvxpy）**：

   * 目标：最大化 `w^T μ_net – λ * w^T Σ w`（`μ_net` 为扣成本后的预期超额收益）
   * 约束：`sum(w)=exposure`，`0 ≤ w_i ≤ 0.08`，行业/风格上限，日换手 ≤ 配置上限
4. **去拥挤**：同一行业内仅保留 1–2 只代表，或在目标函数加入多样性惩罚 `–γ * ||B_industry^T w||_2`

### 验收标准

* 持仓“行业热力图”最大权重 ≤ 25%；风格 β 在阈值内
* 组合层年化波动/回撤相对基线下降，夏普稳定性提升

---

## 模块 C｜稳健性回测与防过拟合（Phase 2）

### 操作步骤

1. **Purged K-Fold**：按时间划分 K 折，清除标签泄露（禁用相邻样本）
2. **Combinatorial CV**：对训练/验证区间做组合抽样，评估稳定性分布
3. **样本外滚动**：例：3 年训练 → 6 个月验证 → 前滚 6 个月复用
4. **Deflated Sharpe**：按因子/参数自由度对夏普折减（DSR>0.1 作为通过线）
5. **参数敏感性**：对关键参数 ±20% 网格，生成热图与弹性指标

### 产出

* `reports/stability/` 下保存：每折绩效、DSR、敏感性热图、最坏 10% 分位表现

---

## 模块 D｜执行韧性工程（Phase 2）

### 操作步骤

1. **幂等下单**：

   * `client_order_id = hash(symbol, side, qty, target_price, intent_ts)`；所有写操作先查此 ID 防重复
2. **订单生命周期**：

   * 状态机：`NEW → PARTIALLY_FILLED ↔ REPLACED → FILLED/CANCELED/REJECTED`
   * **部分成交回补**：按剩余量继续 TWAP 分片
3. **断线/重试/超时**：

   * 指数退避重试（上限 3 次）；超时 3–5s；网络恢复后做**订单对账（reconcile）**
4. **时钟与延迟**：

   * NTP 校时；在日志中记录 `signal_ts → order_sent → ack_ts → fill_ts` 延迟
5. **价格保护**：

   * 若市价偏离参考价超 gate（bp）则撤单

### 产出

* `ops/playbooks.md`：失败重试、断线重连、幽灵单排查的“手册”
* 每日“执行审计”报表：提交/成交/撤单/失败/延迟分布

---

## 模块 E｜市场状态机（Phase 3）

### 逻辑

* **输入**：VIX 水平与斜率、涨跌家数差、52 周新高-新低、指数 MA 上下关系
* **输出**：牛/熊/震荡三态；映射到：止损（−7/−3/−5% 基准×ATR）、目标仓位（单票 6/3/4%）、持仓数（18/12/15）、因子权重（动量+/估值+/标准）

### 操作步骤

1. 指标标准化并打分，阈值投票（≥3 票为该状态）
2. 状态切换“去抖动”：需要连续 3 天信号一致才切换
3. 策略层读取状态，动态设置参数（止损/持仓/权重）

### 产出

* `state/regime.json`（含当前状态与依据）+ 仪表盘可视化

---

## 模块 F｜数据治理：公司行为 & 日历（Phase 3）

### 操作步骤

1. **复权/公司行为**：对价格序列应用拆并/分红调整；保存 `raw/`、`adj/` 双轨数据
2. **交易日历**：用 `pandas_market_calendars` 管理美股交易日与时区
3. **数据质量规则**：

   * 缺失/异常（跳变>20%）自动标注；
   * 停复牌/退市检测；
   * 生存者偏差防护：回测时按历史成分/上市日期过滤

### 产出

* `dq/reports/`：数据质量日报；异常告警

---

## 模块 G｜监控、归因与告警（Phase 3）

### 指标分层

* **执行/TCA**：滑点（bp）、成交率、%ADV 占用、提交→成交延迟
* **风险/敞口**：行业权重、风格 β、净/毛敞口、换手率
* **收益/归因**：因子/行业/个股/成本归因，PnL 滚动分解

### 操作步骤

1. 在交易环节埋点，导出 Prometheus 指标端口（:9100）
2. 在现有 Streamlit 仪表添加三板块：TCA、暴露、归因
3. 告警：

   * **红线**：单日换手>30%、任一行业>25%、实测滑点>模型×2、延迟>3s、净值回撤>5%
   * 触发 Lark/Webhook 推送与自动“降风控档”（收紧仓位/止损）

---

## 验证与放量“金字塔”

1. **回测（含成本）通过** → 2) **纸交/模拟**（≥ 2 周，TCA 偏差<30%）→ 3) **小额实盘**（资金 10–20%）→ 4) **分阶段放量**（每 +20% 资金需再次通过 TCA/风控验收）

---

## 快速上手：脚手架与示例（可复制到你的代码库）

### 1. 成本模型（伪代码）

```python
def estimate_cost_bp(qty, price, adv, spread_bp=8, k=0.0015, alpha=0.7, slip_sigma_bp=8):
    notional = qty * price
    p = notional / max(adv, 1e-9)
    impact_bp = 1e4 * k * (p ** alpha)
    slip_bp = np.random.normal(loc=spread_bp, scale=slip_sigma_bp)
    return max(0.0, impact_bp + slip_bp)
```

### 2. 组合优化（cvxpy 轮廓）

```python
w = cp.Variable(n)
objective = cp.Maximize(mu_net @ w - lam * cp.quad_form(w, Sigma))
constraints = [cp.sum(w) == exposure, w >= 0, w <= 0.08]
constraints += [cp.norm(B_industry.T @ w, 1) <= 0.25]
constraints += [cp.norm(B_style.T @ w, 2) <= 0.5]
prob = cp.Problem(objective, constraints)
prob.solve(solver="OSQP")
```

### 3. Purged K-Fold（时间序列交叉验证轮廓）

```python
def purged_kfold_idx(n, k=5, purge=5):
    fold = n // k
    for i in range(k):
        test = range(i*fold, (i+1)*fold)
        train = list(range(0, max(0, i*fold - purge))) + list(range(min(n, (i+1)*fold + purge), n))
        yield train, test
```

### 4. 幂等下单（订单指纹）

```python
from hashlib import blake2b

def make_client_order_id(symbol, side, qty, price, intent_ts):
    s = f"{symbol}|{side}|{qty}|{price:.4f}|{intent_ts}"
    return blake2b(s.encode(), digest_size=12).hexdigest()
```

###
