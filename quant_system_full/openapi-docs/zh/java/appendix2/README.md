---
title: 枚举参数对照表
---

### 合约类型

标识|合约类型
---|---
STK|股票
OPT|美股期权
WAR|港股窝轮 
IOPT|港股牛熊证
CASH|外汇
FUT|期货
FOP|期货期权

### 货币类型

标识|货币类型
---|---
USD|美元 
HKD|港币
CNH|人民币

### 订单状态

状态 | 状态码 |说明
--- | --- | ---
Invalid|-2|非法状态
Initial|-1|订单初始状态
PendingCancel|3 | 待取消(综合账号和模拟账号没有) 
Cancelled |4| 已取消
Submitted|5|订单已经提交(综合账号和模拟账号没有)
Filled |6| 完全成交
Inactive |7| 已失效
PendingSubmit |8| 待提交

### 账户状态

状态|说明
--- | ---
New| 新账户
Funded| 已入金
Open|已开通
Pending|待确认
Abandoned|已废弃
Rejected|已拒绝
Closed|已关闭
Unknown|未知

### 订单类型

类型|说明
--- | ---
MKT|市价单
LMT|限价单
STP|止损单
STP_LMT|止损限价单
TRAIL|跟踪止损单

### 账户类型

类型|说明
--- | ---
CASH | 现金账户
MGRN | Reg T 保证金账户
PMGRN | 投资组合保证金

### 订阅主题

Subject|说明
---|---
OrderStatus|订单变化
Asset|资产
Position|持仓
Quote|股票行情
Option|期权行情
Future|期货行情
QuoteDepth|股票深度行情

### K线类型

类型|说明
---|---
day|日K
week|周K
month|月K
year|年K
min1|1分钟
min5|5分钟
min15|15分钟
min30|30分钟
min60|60分钟

### 订单变动 

* OrderChangeKey

字段|说明
---|---
id|全局唯一订单号
account|用户账户
type|类型
timestamp|服务器时间
orderId|账户自增订单号
name|股票名称
latestPrice|最新价
symbol|股票代码
action|买入/卖出
orderType|订单类型
secType|合约类型
currency|货币类型
localSymbol|股票代码 (港股用于识别窝轮和牛熊证)
originSymbol|原始股票代码
strike|底层价格 期权、窝轮、牛熊证专属
expiry|过期日 期权、窝轮、牛熊证专属
right|期权方向 PUT/CALL 期权、窝轮、牛熊证专属
limitPrice|限价单价格
auxPrice|跟踪额
trailingPercent|跟踪百分比
timeInForce|有效时间
goodTillDate|GTD时间，格式 20060505 08:00:00 EST
outsideRth|true 允许盘前盘后交易(美股专属)
totalQuantity|订单数量
filledQuantity|执行数目
avgFillPrice|平均成本
lastFillPrice|最后执行价
openTime|订单创建时间
latestTime|订单最近修改时间
remaining|未获执行的股数
status|订单状态
source|订单来源
liquidation|清算值
errorCode|错误码
errorMsg|错误描述
errorMsgCn|错误描述
errorMsgTw|错误描述

### 持仓变动

* PositionChangeKey

字段|说明
---|---
account|用户账户
symbol|股票代码
type|类型
timestamp|服务器时间
secType|合约类型
market|交易市场
currency|货币类型
latestPrice|最新价
marketValue|市值
position|持仓
averageCost|平均成本
unrealizedPnl|浮动盈亏
expiry|过期日 期权、窝轮、牛熊证专属
strike|底层价格 期权、窝轮、牛熊证专属
right|期权方向 PUT/CALL 期权、窝轮、牛熊证专属
multiplier|1手单位 期权、窝轮、牛熊证专属


### 资产变动

* AssetChangeKey

字段|说明
---|---
account|用户账户
type|类型
timestamp|服务器时间
currency |货币类型
deposit|出入金
buyingPower |购买力
cashBalance |账户现金余额
grossPositionValue |持仓市值
netLiquidation |净清算值
equityWithLoan |含借贷值股权(含贷款价值资产)
initMarginReq |当前初始保证金
maintMarginReq |当前维持保证金
availableFunds |可用资金(含借贷股权-初始保证金)
excessLiquidity |剩余流动性(借贷值股权-维持保证金)

### 行情变动

* QuoteChangeKey

字段|说明
---|---
symbol|股票代码
type|类型
timestamp|服务器时间
latestPrice|最新价格
latestTime|最新时间
preClose|昨日收盘价
volume|成交量
open|开盘价
close|收盘价
high|最高价格
low|最低价格
marketStatus|市场状态
askPrice|卖盘价格
askSize|卖盘数量
bidPrice|买盘价格
hourTradingTag|盘前盘后标识，取值：盘前 or 盘后
hourTradingLatestPrice|盘前/盘后最新价
hourTradingLatestTime|盘前/盘后最新价成交时间
hourTradingVolume|盘前成交量
bidSize|买盘数量
p|分钟价格
a|分钟平均价格
t|分钟时间
v|分钟成交量

### 行情权限
字段|说明
---|---
hkStockQuoteLv2|港股Lv2
usQuoteBasic|美股nasdaq Lv1
usStockQuoteLv2Totalview|美股TotalView Lv2
usOptionQuote|美股期权

### 订单描述

| 描述信息   | 说明         |
| ---------- | ------------ |
| Exercise   | 期权行权     |
| Expiry     | 期权过期     |
| Assignment | 期权被动行权 |

