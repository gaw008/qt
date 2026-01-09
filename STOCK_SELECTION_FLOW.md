# 选股系统完整流程说明

## 流程概览

```
5000+ 股票
    ↓
基础筛选 (SelectionCriteria)
    ↓
~300-1000 股票
    ↓
策略分析 (3个策略并行)
    ↓
策略合并 + 共识加成
    ↓
~20 股票
    ↓
LLM增强 (可选)
    ↓
最终选股 (Top 10-20)
```

---

## 第一步：股票池获取（Universe）

**代码位置**: `runner.py:722-755` (`_get_stock_universe`)

### 股票来源
```python
# 从CSV文件读取股票列表
universe_file = "all_stock_symbols.csv"
```

**实际股票数量**: ~5000只美股
- 包括: NYSE, NASDAQ, AMEX上市股票
- 排除: 无效symbol（长度>5的）

**Fallback机制**:
- 如果CSV不存在，使用hardcoded的120只大盘股
- 包括: AAPL, MSFT, GOOGL, AMZN, NVDA等

---

## 第二步：基础筛选（Pre-filtering）

**代码位置**: `base_strategy.py:214-268` (`filter_universe`)

### 筛选条件（SelectionCriteria）

#### 默认配置（原始策略）
```python
SelectionCriteria(
    max_stocks=10,           # 最终选出10只
    min_market_cap=1e9,      # 市值 >= $1B (10亿美元)
    max_market_cap=1e12,     # 市值 <= $1T (1万亿美元)
    min_volume=100000,       # 日成交量 >= 10万股
    min_price=5.0,           # 价格 >= $5
    max_price=500.0,         # 价格 <= $500
    min_score_threshold=50.0 # 策略分数 >= 50
)
```

#### 改进策略V2配置
```python
SelectionCriteria(
    max_stocks=20,                # 选出20只
    min_market_cap=1e8,           # 市值 >= $100M (更宽松)
    max_market_cap=5e12,          # 市值 <= $5T (更宽松)
    min_volume=50000,             # 日成交量 >= 5万股 (更宽松)
    min_price=1.0,                # 价格 >= $1 (更宽松)
    max_price=2000.0,             # 价格 <= $2000 (包括高价股)
    min_score_threshold=0.0       # 不设分数门槛
)
```

### 筛选逻辑
```
对于5000只股票中的每一只:
1. 获取基本数据（价格、成交量、市值）
2. 应用价格过滤: min_price <= price <= max_price
3. 应用成交量过滤: volume >= min_volume
4. 应用市值过滤: min_market_cap <= market_cap <= max_market_cap
5. 应用行业过滤（如果配置）: 排除/包含特定行业
6. 应用黑名单过滤: 排除exclude_symbols

通过的股票 → 进入候选池
```

**筛选后数量**: ~300-1000只股票（取决于市场条件）

---

## 第三步：策略分析（Strategy Analysis）

**代码位置**:
- `runner.py:630-674` (原始策略)
- `strategy_orchestrator_v2.py:155-223` (改进策略V2)

### 路径A：原始策略（默认）

#### 三个策略并行运行

**1. ValueMomentumStrategy (价值动量策略)**
```python
from bot.selection_strategies.value_momentum import ValueMomentumStrategy
```
**评分因子**:
- 估值指标: P/E, P/B, EV/EBITDA (权重35%)
- 动量指标: 价格动量, ROC (权重30%)
- 基本面: 盈利增长, ROE (权重20%)
- 技术面: RSI, MACD (权重15%)

**输出**: 每只股票的value_momentum分数 (0-100)

**2. TechnicalBreakoutStrategy (技术突破策略)**
```python
from bot.selection_strategies.technical_breakout import TechnicalBreakoutStrategy
```
**评分因子**:
- 突破信号: 突破阻力位 (权重30%)
- 成交量确认: OBV, 量价配合 (权重25%)
- 趋势强度: ADX, 趋势线 (权重20%)
- 形态识别: 旗形、三角形等 (权重15%)
- 短期动量: 短期均线 (权重10%)

**输出**: 每只股票的technical_breakout分数 (0-100)

**3. EarningsMomentumStrategy (盈利动量策略)**
```python
from bot.selection_strategies.earnings_momentum import EarningsMomentumStrategy
```
**评分因子**:
- 盈利增长: EPS增长率 (权重30%)
- 盈利惊喜: 实际vs预期 (权重25%)
- 营收增长: Revenue增长率 (权重20%)
- 利润率: Gross/Operating Margin (权重15%)
- 估值合理性: PEG ratio (权重10%)

**输出**: 每只股票的earnings_momentum分数 (0-100)

### 路径B：改进策略V2（USE_IMPROVED_STRATEGIES=true）

**代码位置**: `runner.py:588-620`

使用`StrategyOrchestratorV2`，包含3个改进策略：

**1. ImprovedValueMomentumV2**
- 增强的价值动量因子
- 行业归一化处理
- 风险调整后收益

**2. DefensiveValue (防御性价值)**
- 低波动率价值股
- 高股息收益
- 财务稳健性

**3. BalancedMomentum (平衡动量)**
- 多时间框架动量
- 相对强度指标
- 动量质量评估

**风险管理集成**:
```python
# 市场状态过滤
MarketRegimeFilter().should_reduce_exposure()
  ↓ 如果市场不好，减少选股数量

# 组合风险控制
PortfolioRiskControl().validate_portfolio()
  ↓ 检查单一持仓、行业集中度
```

---

## 第四步：策略合并（Strategy Combination）

**代码位置**:
- `runner.py:788-856` (`_combine_strategy_results`) - 原始
- `strategy_orchestrator_v2.py:252-339` (`_combine_with_diversification`) - V2

### 原始合并方法（Simple Consensus）

```python
# 对于每只股票
for symbol in all_stocks:
    # 收集所有策略的分数
    total_score = sum(各策略分数)
    strategy_count = 参与策略数量

    # 计算平均分
    avg_score = total_score / strategy_count

    # 共识加成
    if strategy_count >= 2:
        consensus_bonus = min(10.0, strategy_count * 2.5)
        final_score = avg_score + consensus_bonus

    # 例子:
    # AAPL: ValueMomentum=85, TechnicalBreakout=78, EarningsMomentum=82
    # avg_score = (85+78+82)/3 = 81.67
    # consensus_bonus = 3 * 2.5 = 7.5
    # final_score = 81.67 + 7.5 = 89.17
```

**排序**: 按final_score降序
**输出**: Top 20只股票

### 改进合并方法V2（Style Diversification）

**代码位置**: `strategy_orchestrator_v2.py:252-339`

**风格配置**:
```python
value_allocation = 40%      # 价值风格
momentum_allocation = 30%   # 动量风格
balanced_allocation = 30%   # 平衡风格
```

**合并算法**:
```python
max_stocks = 20

# 1. 价值槽位 (8只)
value_slots = 20 * 0.4 = 8
从 ImprovedValueMomentumV2 + DefensiveValue 中
选取分数最高的8只

# 2. 动量槽位 (6只)
momentum_slots = 20 * 0.3 = 6
从 BalancedMomentum 中
选取分数最高的6只

# 3. 平衡槽位 (6只)
balanced_slots = 20 - 8 - 6 = 6
从所有策略中（排除已选）
选取分数最高的6只

最终组合 = 8只价值 + 6只动量 + 6只平衡 = 20只
```

**优势**:
- 强制风格多样化
- 避免单一策略dominate
- 更稳定的组合表现

---

## 第五步：LLM增强（可选）

**代码位置**: `runner.py:679-739`

**前置条件**:
```python
ENABLE_LLM_ENHANCEMENT=true
OPENAI_API_KEY=sk-...
```

### LLM管道流程

**配置的漏斗参数**:
```python
LLM_M_TRIAGE = 60        # 新闻分析数量（配置值，实际取min(输入数量, 60)）
LLM_M_FINAL = 15         # 深度分析数量（配置值，实际取min(输入数量, 15)）
```

**实际流程**:
```
20 stocks (from策略合并)
    ↓
Step 1: Event Queue Builder
    识别需要特别关注的股票
    (技术突破、盈利发布、新闻事件)
    ↓
Step 2: Triage (新闻分析) - 分析全部20只
    配置: LLM_M_TRIAGE=60
    实际: min(20, 60) = 20只全部分析
    模型: gpt-5-nano
    数据: 最近3天的新闻

    For each stock:
        ├─ 获取Yahoo Finance新闻
        ├─ GPT分析新闻情绪和风险
        ├─ 生成 news_quality (0-100)
        ├─ 生成 risk_flags (0-100)
        └─ 应用规则（仅对Technical Breakout策略生效）:
            ├─ news_quality < 40 → GATED (TB分数归零)
            ├─ risk_flags > 70 → PENALIZED (TB分数减半)
            └─ 其他 → PASSED (无变化)

    注: Triage规则只影响包含Technical Breakout策略的股票
    ↓
Step 3: Re-sort
    根据调整后的分数重新排序
    ↓
Step 4: Deep Analysis (深度分析) - 分析前15只
    配置: LLM_M_FINAL=15
    实际: 从20只中选top 15进行分析
    模型: gpt-5
    数据: SEC EDGAR财报

    For each stock:
        ├─ 获取SEC EDGAR财报
        ├─ GPT分析盈利质量
        ├─ 生成 earnings_score (-100 to +100)
        ├─ 生成 quality_score (-100 to +100)
        └─ 应用boost:
            ├─ earnings_score → Earnings Momentum (+/- 20分)
            └─ quality_score → Value Momentum (+/- 15分)

    ↓
Step 5: Final Sort
    根据LLM增强后的分数重新排序
    ↓
LLM增强后的20只股票（最终输出）
```

### LLM效果举例

**Example 1: Triage封禁**
```
TSLA原始分数:
  - ValueMomentum: 75
  - TechnicalBreakout: 88  ← 技术突破分数高
  - EarningsMomentum: 70
  - 合并分数: 77.67 + 7.5(共识) = 85.17

LLM Triage分析:
  - 新闻: "特斯拉CEO不当言论引发争议"
  - news_quality = 35 (< 40门槛)
  - 动作: GATED

调整后分数:
  - ValueMomentum: 75
  - TechnicalBreakout: 0  ← 归零
  - EarningsMomentum: 70
  - 合并分数: 48.33 + 5(共识) = 53.33

排名: 第2名 → 第18名 (大幅下降)
```

**Example 2: Deep增强**
```
AAPL原始分数:
  - ValueMomentum: 82
  - TechnicalBreakout: 75
  - EarningsMomentum: 85
  - 合并分数: 80.67 + 7.5 = 88.17

LLM Deep分析:
  - 财报分析: 盈利质量优秀，现金流强劲
  - earnings_score = +75
  - quality_score = +60

调整:
  - ValueMomentum: 82 + (60/100)*15 = 82 + 9 = 91
  - TechnicalBreakout: 75 (无变化)
  - EarningsMomentum: 85 + (75/100)*20 = 85 + 15 = 100
  - 合并分数: 88.67 + 7.5 = 96.17

排名: 第5名 → 第1名 (提升)
```

### LLM成本控制

**每次运行成本估算**:
```python
# Triage (gpt-5-nano)
# ~800 input + 80 output tokens per stock
# 20 stocks * $0.0001 = $0.002

# Deep (gpt-5)
# ~3000 input + 200 output tokens per stock
# 15 stocks * $0.001 = $0.015

Total per run: ~$0.017
Daily budget: 60 runs = ~$1.02/day
```

**缓存机制**:
```python
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL_DAYS=7

# 相同股票的新闻分析可以缓存7天
# 大幅减少重复API调用
```

---

## 第六步：最终输出

**代码位置**: `runner.py:858-901` (`_update_selection_status`)

### 输出格式

```python
selection_results = {
    'total_selections': 20,
    'timestamp': '2025-10-12T10:30:00',
    'top_picks': [
        {
            'rank': 1,
            'symbol': 'AAPL',
            'avg_score': 96.2,
            'strategy_count': 3,
            'dominant_action': 'BUY',
            'reasoning': 'ImprovedValueMomentum: Strong fundamentals...;
                         BalancedMomentum: Upward trend confirmed...;
                         DefensiveValue: High dividend yield...'
        },
        {
            'rank': 2,
            'symbol': 'MSFT',
            'avg_score': 94.5,
            'strategy_count': 3,
            'dominant_action': 'STRONG_BUY',
            'reasoning': '...'
        },
        # ... 更多股票
    ]
}
```

### 写入位置

1. **状态文件**: `dashboard/state/status.json`
```json
{
  "selection_results": { ... },
  "last_selection_run": "2025-10-12T10:30:00",
  "selection_status": "completed"
}
```

2. **LLM增强状态** (如果启用): `dashboard/state/llm_enhanced_status.json`
```json
{
  "enabled": true,
  "mode": "full",
  "base_results": [...],
  "enhanced_results": [...],
  "metrics": {
    "execution_time": 15.67,
    "llm_calls": 35,
    "cache_hits": 15,
    "cost_usd": 0.0187
  }
}
```

---

## 完整流程示例

### 输入
```
5000只美股 (从all_stock_symbols.csv)
```

### 第1步：基础筛选
```
应用SelectionCriteria:
- 价格: $1 - $2000
- 市值: $100M - $5T
- 成交量: >= 50,000股
- 行业: 无限制

通过筛选: ~800只
```

### 第2步：策略分析
```
并行运行3个策略 (改进V2):
- ImprovedValueMomentumV2: 评估800只 → 产生分数
- DefensiveValue: 评估800只 → 产生分数
- BalancedMomentum: 评估800只 → 产生分数

每个策略选出Top 50
```

### 第3步：策略合并
```
风格多样化合并:
- 8只价值股 (40%)
- 6只动量股 (30%)
- 6只平衡股 (30%)

共20只股票
```

### 第4步：LLM增强
```
Step 1: 识别20只中需要关注的
Step 2: Triage - 分析全部20只的新闻
  → 2只被GATED (技术突破股)
  → 3只被PENALIZED
  → 15只PASSED

Step 3: 重新排序

Step 4: Deep - 分析前15只的财报
  → 10只获得正向boost
  → 3只获得负向boost
  → 2只无变化

Step 5: 最终排序
```

### 最终输出
```
Top 10股票写入status.json:
1. AAPL (score: 96.2) - LLM增强后排名第1
2. MSFT (score: 94.5) - 稳定的高分
3. NVDA (score: 93.1) - 动量强劲
4. GOOGL (score: 91.8) - 价值+增长
5. META (score: 90.2) - 财报优秀
6. AMZN (score: 88.7) - 平衡选择
7. BRK-B (score: 87.3) - 防御性价值
8. JPM (score: 86.1) - 金融龙头
9. UNH (score: 85.4) - 医疗保健
10. LLY (score: 84.9) - 制药明星

这10只股票将被用于:
- 展示在Dashboard
- 自动交易引擎的买入候选
- 组合优化的输入
```

---

## 配置对比

### 原始策略配置
```python
universe_size = 5000
after_filtering = ~300
strategies = 3 (原始)
final_output = 10 stocks
LLM_enhancement = disabled
```

### 改进策略V2配置
```python
universe_size = 5000
after_filtering = ~800 (更宽松的筛选)
strategies = 3 (改进版 + 风险管理)
style_diversification = 40% value + 30% momentum + 30% balanced
final_output = 20 stocks
LLM_enhancement = enabled (full mode)
```

---

## 性能指标

### 执行时间
```
第1步 基础筛选: ~30-60秒 (5000只 → 800只)
第2步 策略分析: ~60-120秒 (800只并行分析)
第3步 策略合并: ~1秒 (计算和排序)
第4步 LLM增强: ~15-30秒 (API调用)
第5步 输出写入: <1秒

总计: ~2-4分钟
```

### API调用统计
```
数据API调用:
- fetch_history: ~800次 (筛选阶段)
- fetch_ticker_info: ~800次 (基本面数据)

LLM API调用:
- Triage: 20次 (gpt-5-nano)
- Deep: 15次 (gpt-5)
- 缓存命中: ~30-50%

总成本: $0.01-0.02 per run
```

---

## 日志追踪

完整的日志标记：
```
[SELECTION] Starting comprehensive stock selection process
[SELECTION] Loaded 5000 symbols from stock universe file
[SELECTION] Filtered universe: 5000 -> 823 stocks
[SELECTION] Running ImprovedValueMomentumV2 strategy
[SELECTION] ImprovedValueMomentumV2: 50 stocks selected, avg_score: 78.5
[SELECTION] Running DefensiveValue strategy
[SELECTION] DefensiveValue: 50 stocks selected, avg_score: 76.2
[SELECTION] Running BalancedMomentum strategy
[SELECTION] BalancedMomentum: 50 stocks selected, avg_score: 81.3
[SELECTION] Combining strategies with diversification
[LLM] LLM Enhancement is ENABLED
[LLM_PIPELINE] Starting LLM Enhancement Pipeline
[LLM_TRIAGE] Starting news analysis for 20 stocks
[LLM_TRIAGE] Complete: 20 analyzed, 15 passed, 2 gated, 3 penalized
[LLM_DEEP] Starting earnings/quality analysis for 15 stocks
[LLM_DEEP] Complete: 15 analyzed, 10 earnings enhanced, 8 quality enhanced
[LLM] Enhancement applied successfully - using LLM-enhanced results
[LLM] Top pick changed: MSFT -> AAPL
[SELECTION] Selection process completed. Final selection: 20 stocks
[SELECTION] Top 5 selections: AAPL, MSFT, NVDA, GOOGL, META
```

---

## 总结

完整的选股流程从**5000只股票**开始，经过：
1. **基础筛选** → 800只 (市值、价格、流动性)
2. **策略分析** → 各50只 (3个策略并行)
3. **策略合并** → 20只 (风格多样化)
4. **LLM增强** → 20只 (新闻+财报分析)
5. **最终输出** → Top 10-20只 (用于交易)

整个流程大约需要**2-4分钟**，结合了传统量化因子、多策略共识、以及先进的LLM文本分析，确保选出的股票既有数据支持，又有新闻和财报的confirmation。
