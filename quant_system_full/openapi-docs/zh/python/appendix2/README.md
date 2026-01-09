---
title: 枚举参数对照表
---
关于枚举参数及常用字段参数含义，请参考本节

### 语言
`tigeropen.common.consts.Language` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/consts/__init__.py)
| 标识 | 语言  |
|------|------------|
|zh_CN |简体中文|
|zh_TW |繁体中文|
|en_US |英文|

### 市场
`tigeropen.common.consts.Market` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/consts/__init__.py)
| 标识 | 市场  |
|------|------------|
|    ALL |全部|
|    US | 美股|
|    HK | 港股|
|    CN | A股|
|   SG | 新加坡|

### 合约类型
`tigeropen.common.consts.SecurityType` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/consts/__init__.py)
| 标识 | 合约类型   |
|------|------------|
| STK  | 股票       |
| OPT  | 美股期权   |
| WAR  | 港股窝轮   |
| IOPT | 港股牛熊证 |
| CASH | 外汇       |
| FUT  | 期货       |
| FOP  | 期货期权   |

### 货币类型
`tigeropen.common.consts.Currency` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/consts/__init__.py)
| 标识 | 货币类型 |
|------|----------|
| ALL  | 全部     |
| USD  | 美元     |
| HKD  | 港币     |
| CNH  | 人民币   |
| SGD  | 新加坡币  |

### 订单状态
`tigeropen.common.consts.OrderStatus` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/consts/__init__.py)  
SDK通过 `tigeropen.common.util.order_utils.get_order_status` 将状态值处理成枚举标识。[source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/util/order_utils.py#L134)

枚举标识 | 状态值          | 状态码 | 说明         |
|------|---------------|--------|--------------|
|EXPIRED| Invalid       | -2     | 非法状态     |
|NEW| Initial       | -1     | 订单初始状态 |
|PENDING_CANCEL| PendingCancel | 3      | 待取消(综合账号和模拟账号没有)       |
|CANCELLED| Cancelled     | 4      | 已取消       |
|HELD| Submitted     | 5      | 订单已经提交(综合账号和模拟账号没有) |
|FILLED| Filled        | 6      | 完全成交     |
|REJECTED| Inactive      | 7      | 已失效       |
|HELD| PendingSubmit | 8      | 待提交       |

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
RegTMargin | Reg T 保证金账户
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
`tigeropen.common.consts.BarPeriod` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/consts/__init__.py)

枚举类型|枚举值|说明
---|---|---
DAY|day| 日K
WEEK|week| 周K
MONTH|month| 月K
YEAR|year| 年K
ONE_MINUTE|1min| 1分钟
THREE_MINUTES|3min| 3分钟
FIVE_MINUTES|5min| 5分钟
TEN_MINUTES|10min| 10分钟
FIFTEEN_MINUTES|15min| 15分钟
HALF_HOUR|30min| 30分钟
FORTY_FIVE_MINUTES|45min| 45分钟
ONE_HOUR|60min| 60分钟
TWO_HOURS|2hour| 2小时
THREE_HOURS|3hour| 3小时
FOUR_HOURS|4hour| 4小时
SIX_HOURS|6hour| 6小时

### 订单变动

* OrderChangeKey

字段|说明
---|---
id|全局唯一订单号
account|用户账户
type|类型
timestamp|老虎服务器时间
order_id|账户自增订单号
name|股票名称
latest_price|最新价
symbol|股票代码
action|买入/卖出
order_type|订单类型
sec_type|合约类型
currency|货币类型
local_symbol|股票代码 (港股用于识别窝轮和牛熊证)
origin_symbol|原始股票代码
strike|底层价格_期权、窝轮、牛熊证专属
expiry|过期日_期权、窝轮、牛熊证专属
right|期权方向_put/call_期权、窝轮、牛熊证专属
limit_price|限价单价格
aux_price|跟踪额
trailing_percent|跟踪百分比
time_in_force|有效时间
good_till_date|gtd时间，格式 2006050508:00:00est
outside_rth|true_允许盘前盘后交易(美股专属)
total_quantity|订单数量
filled_quantity|执行数目
avg_fill_price|平均成本
last_fill_price|最后执行价
open_time|订单创建时间
latest_time|订单最近修改时间
remaining|未获执行的股数
status|订单状态
source|订单来源
liquidation|清算值
error_code|错误码
error_msg|错误描述
error_msg_cn|错误描述
error_msg_tw|错误描述

### 持仓变动

* PositionChangeKey

字段|说明
---|---
account|用户账户
symbol|股票代码
type|类型
timestamp|老虎服务器时间
sec_type|合约类型
market|交易市场
currency|货币类型
latest_price|最新价
market_value|市值
position|持仓
average_cost|平均成本
unrealized_pnl|浮动盈亏
expiry|过期日_期权、窝轮、牛熊证专属
strike|底层价格_期权、窝轮、牛熊证专属
right|期权方向_put/call_期权、窝轮、牛熊证专属
multiplier|1_手单位_期权、窝轮、牛熊证专属


### 资产变动

* AssetChangeKey

字段|说明
---|---
account|用户账户
type|类型
timestamp|老虎服务器时间
currency |货币类型
deposit|出入金
buying_power |购买力
cash_balance |账户现金余额
gross_position_value |持仓市值
net_liquidation |净清算值
equity_with_loan |含借贷值股权(含贷款价值资产)
init_margin_req |当前初始保证金
maint_margin_req |当前维持保证金
available_funds |可用资金(含借贷股权_初始保证金)
excess_liquidity |剩余流动性(借贷值股权_维持保证金)

### 行情变动

* QuoteChangeKey

字段|说明
---|---
symbol|股票代码
type|类型
timestamp|老虎服务器时间
latest_price|最新价格
latest_time|行情最新时间
pre_close|昨日收盘价
volume|成交量
open|开盘价
close|收盘价
high|最高价格
low|最低价格
market_status|市场状态
ask_price|卖盘价格
ask_size|卖盘数量
bid_price|买盘价格
hour_trading_tag|盘前盘后标识，取值：盘前_or_盘后
hour_trading_latest_price|盘前/盘后最新价
hour_trading_latest_time|盘前/盘后最新价成交时间
hour_trading_volume|盘前成交量
bid_size|买盘数量
p|分钟价格
a|分钟平均价格
t|分钟时间
v|分钟成交量

### 行情权限
字段|说明
---|---
hkStockQuoteLv2|港股Lv2
usQuoteBasic|美股nasdaq Lv1
usStockQuote|美股Lv1
usStockQuoteLv2Totalview|美股TotalView Lv2
usStockQuoteLv2Arca|美股Arca Lv2
usOptionQuote|美股期权
aStockQuoteLv1|A股Lv1