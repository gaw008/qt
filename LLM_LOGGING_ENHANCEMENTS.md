# LLM日志增强说明

## 概述
为OpenAI LLM选股增强模块添加了详细的日志输出，以便在系统日志中清楚地看到LLM的运行状态。

## 修改的文件

### 1. `dashboard/worker/runner.py` (lines 679-739)
**改进内容：**
- 添加了LLM启用状态检测和日志
- 显示LLM配置信息（模式、使用的模型）
- 显示选股漏斗流程（总数 -> triage数 -> deep分析数）
- 记录API调用次数和缓存命中数
- 显示每次运行的成本
- 记录处理的股票数量
- 显示增强前后的top pick变化
- 详细的错误日志

**日志示例：**
```
[LLM] LLM Enhancement is ENABLED
[LLM] Configuration: mode=full, model_triage=gpt-5-nano, model_deep=gpt-5
[LLM] Funnel: 20 stocks -> 60 triage -> 15 deep analysis
[LLM] Enhancement execution completed in 12.34s
[LLM] API Calls: 25, Cache Hits: 10
[LLM] Cost this run: $0.0152
[LLM] Processed: 20 base -> 20 enhanced stocks
[LLM] Enhancement applied successfully - using LLM-enhanced results
[LLM] Top pick changed: AAPL -> MSFT
```

### 2. `bot/llm_enhancement/clients/openai_client.py` (lines 184-196)
**改进内容：**
- 优化API调用成功日志，更清晰地显示token使用情况
- 添加会话统计日志（总调用次数、总token数、累计成本）

**日志示例：**
```
[LLM_API] Call successful: model=gpt-5-nano, tokens=850in+120out, cost=$0.000088, latency=1.23s
[LLM_API] Session stats: total_calls=15, total_tokens=14520, cumulative_cost=$0.0125
```

### 3. `bot/llm_enhancement/enhancer.py`
**改进内容：**

#### Triage阶段 (lines 108-198)
- 显示新闻分析的目标股票列表
- 统计通过、封禁、惩罚的股票数量
- 详细的阶段完成总结

**日志示例：**
```
[LLM_TRIAGE] Starting news analysis for 60 stocks
[LLM_TRIAGE] Target stocks: AAPL, MSFT, GOOGL, AMZN, NVDA...
[LLM] TSLA: Gated (quality=35), TB: 85.0 -> 0
[LLM] NFLX: Penalized (risk=75), TB: 78.5 -> 39.3
[LLM_TRIAGE] Complete: 60 analyzed, 52 passed, 4 gated, 4 penalized
```

#### Deep阶段 (lines 230-324)
- 显示财报/质量分析的目标股票列表
- 分别统计财报增强和质量增强的股票数量
- 详细的boost信息

**日志示例：**
```
[LLM_DEEP] Starting earnings/quality analysis for 15 stocks
[LLM_DEEP] Target stocks: AAPL, MSFT, GOOGL, AMZN, NVDA...
[LLM_DEEP] AAPL: Earnings boost=+12.5, EM: 82.0 -> 94.5
[LLM_DEEP] MSFT: Quality boost=+8.7, VM: 75.0 -> 83.7
[LLM_DEEP] Complete: 15 analyzed, 12 earnings enhanced, 10 quality enhanced
```

### 4. `bot/llm_enhancement/pipeline.py`
**改进内容：**

#### 开始日志 (lines 140-145)
- 添加清晰的分隔线
- 显示管道启动信息和配置

**日志示例：**
```
============================================================
[LLM_PIPELINE] Starting LLM Enhancement Pipeline
[LLM_PIPELINE] Mode: full
[LLM_PIPELINE] Input: 20 base selections
[LLM_PIPELINE] Models: triage=gpt-5-nano, deep=gpt-5
============================================================
```

#### 阶段日志 (lines 182-221)
- 清楚标注每个阶段（Step 2/3, Step 3/3）
- 说明每个阶段的目的
- 警告信息如果跳过阶段

**日志示例：**
```
[LLM_PIPELINE] Step 2/3: Triage (News Analysis)
[LLM_PIPELINE] Analyzing 60 stocks for news sentiment and risk flags
[LLM_PIPELINE] Triage stage completed
[LLM_PIPELINE] Step 3/3: Deep Analysis (Earnings & Quality)
[LLM_PIPELINE] Analyzing top 15 stocks for earnings and financial quality
[LLM_PIPELINE] Deep analysis stage completed
```

#### 结束日志 (lines 266-277)
- 添加清晰的分隔线
- 总结执行时间、API调用、成本
- 列出前5个错误（如果有）

**日志示例：**
```
============================================================
[LLM_PIPELINE] Successfully enhanced 20 selections in 15.67s
[LLM_PIPELINE] API calls: 35, cache hits: 15, total cost: $0.0187
============================================================
```

## 配置要求

确保`.env`文件中配置了以下参数：

```bash
# 启用LLM增强
ENABLE_LLM_ENHANCEMENT=true

# OpenAI API密钥
OPENAI_API_KEY=sk-...

# LLM模型配置
LLM_MODEL_TRIAGE=gpt-5-nano
LLM_MODEL_DEEP=gpt-5

# 运行模式 (full, triage_only, deep_only)
LLM_ENHANCEMENT_MODE=full

# 漏斗参数
LLM_M_TRIAGE=60  # 新闻分析数量
LLM_M_FINAL=15   # 深度分析数量
```

## 日志级别

- **INFO**: 主要流程和结果
- **DEBUG**: 详细的处理信息（需要设置 `LLM_LOG_LEVEL=DEBUG`）
- **WARNING**: 跳过或部分失败
- **ERROR**: 错误和异常

## 查看日志

### 1. 系统主日志
```bash
tail -f dashboard/state/bot.log | grep "\[LLM"
```

### 2. LLM专用日志
```bash
tail -f logs/llm_enhancement.log
```

### 3. LLM状态文件
```bash
cat dashboard/state/llm_enhanced_status.json
```

## 日志标签说明

- `[LLM]`: 一般LLM相关日志
- `[LLM_API]`: OpenAI API调用日志
- `[LLM_PIPELINE]`: 管道流程日志
- `[LLM_TRIAGE]`: 新闻分析阶段日志
- `[LLM_DEEP]`: 深度分析阶段日志

## 故障排查

### LLM未启用
如果看到：
```
[LLM] LLM Enhancement is DISABLED or not configured
[LLM] Reason: ENABLE_LLM_ENHANCEMENT=false
```

解决方法：在`.env`中设置 `ENABLE_LLM_ENHANCEMENT=true`

### API密钥未设置
如果看到：
```
[LLM] Reason: OPENAI_API_KEY not set
```

解决方法：在`.env`中设置 `OPENAI_API_KEY=sk-...`

### API调用失败
查看详细错误信息：
```bash
grep "LLM_API.*failed" logs/llm_enhancement.log
```

## 性能监控

通过日志可以监控：
1. **执行时间**: 每次运行的总耗时
2. **API调用次数**: 实际调用OpenAI API的次数
3. **缓存命中率**: 缓存有效性
4. **成本**: 每次运行和累计成本
5. **增强效果**: 多少股票被增强、封禁、惩罚

## 下一步

运行系统后，日志会自动显示详细的LLM运行状态。无需额外配置，只需确保：
1. `ENABLE_LLM_ENHANCEMENT=true`
2. `OPENAI_API_KEY` 已设置
3. 系统正常运行

日志将实时显示在：
- 控制台输出
- `dashboard/state/bot.log`
- `logs/llm_enhancement.log`
